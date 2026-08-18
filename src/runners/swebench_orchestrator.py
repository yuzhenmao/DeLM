# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from base.engine.async_llm import set_events_path
from base.engine.logs import logger
from benchmark.common.runner import LevelResult
from benchmark.benchmark import Benchmark, LevelSpec
from benchmark.bench_swebench import SWEBenchConfig, SWEBenchEnvironment
from benchmark.swebench.data_loader import SWEBenchDataLoader, SWEBenchInstance
from src.config import SWEBenchOrchestraConfig
from src.prompts.swebench_global import SWEBenchPlannerPrompt
from src.results import save_csv_row
from src.runners.solver_thread import SWEBenchRunner
from src.shared_lessons import SharedLessons

class SWEBenchOrchestra(Benchmark):
    """SWE-bench benchmark orchestrator (PlannerAgent + ImplementerAgent)."""

    def __init__(self, config: SWEBenchOrchestraConfig):
        self.config = config
        self.planner_model = config.planner_model
        self.implementer_models = config.implementer_models
        self.max_attempts = config.max_attempts

        self._data_loader = SWEBenchDataLoader(
            dataset_name=config.dataset_name,
            split=config.split,
            cache_dir=config.cache_dir,
            subset_seed=config.subset_seed,
            subset_sizes=config.subset_sizes,
            subset_role=config.subset_role,
        )

        self._instances: Dict[str, SWEBenchInstance] = {}

        trajectory_dir = config.trajectory_dir or (config.result_folder / "trajectories")
        csv_path = config.csv_summary_path or (config.result_folder / "results.csv")

        # Per-run events.jsonl: AsyncLLM appends one line per LLM call
        # (role / model / tokens / latency / task_id).
        self._events_path = Path(config.result_folder) / "events.jsonl"
        set_events_path(self._events_path)

        # Number of parallel solvers per task (pass@N). N=1 falls back to
        # the single-solver path.
        self.n_solvers = int(getattr(config, "n_solvers", 2))

        self._trajectory_dir = trajectory_dir
        self._csv_path = csv_path
        self._csv_lock = asyncio.Lock()
        self._lessons_root = config.result_folder / "lessons"

    def _sum_events_cost(
        self,
        level_id: str,
        thread_ids: Optional[Set[int]] = None,
    ) -> float:
        """Sum per-LLM-call cost from events.jsonl for the given task.

        Cap-hit recovery has two modes:
          - thread_ids=None: sum every event tagged for this task
            (`level_id` or `{level_id}#t*`).
          - thread_ids={...}: sum only events tagged `{level_id}#t{tid}`
            for those tids; useful when the returned threads' costs are
            already in their LevelResult and we want to recover only the
            cancelled siblings' costs.

        Missing file / malformed lines contribute 0. events.jsonl is
        authoritative for calls that returned usage to this process but is
        a lower bound for calls cancelled mid-flight.
        """
        if not self._events_path.exists():
            return 0.0
        wanted_tags: Optional[Set[str]] = None
        if thread_ids is not None:
            wanted_tags = {f"{level_id}#t{int(t)}" for t in thread_ids}
            if not wanted_tags:
                return 0.0
        thread_prefix = f"{level_id}#t"
        total = 0.0
        try:
            with self._events_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    tid = e.get("task_id")
                    if wanted_tags is not None:
                        if not isinstance(tid, str) or tid not in wanted_tags:
                            continue
                    else:
                        if not (tid == level_id or (isinstance(tid, str) and tid.startswith(thread_prefix))):
                            continue
                    c = e.get("cost", 0.0) or 0.0
                    try:
                        total += float(c)
                    except (TypeError, ValueError):
                        continue
        except Exception as e:
            logger.warning(f"[SWEBenchOrchestra] _sum_events_cost({level_id}) failed: {e}")
            return 0.0
        return total

    def _load_selected_ids(self) -> Optional[Set[str]]:
        """Load selected instance IDs from file if configured."""
        if not self.config.selected_ids_file:
            return None

        ids_file = self.config.selected_ids_file
        if not ids_file.exists():
            logger.warning(f"Selected IDs file not found: {ids_file}")
            return None

        try:
            with ids_file.open("r", encoding="utf-8") as f:
                ids = json.load(f)
            if isinstance(ids, list):
                selected = set(ids)
                logger.info(f"Loaded {len(selected)} selected instance IDs from {ids_file}")
                return selected
            else:
                logger.warning(f"Selected IDs file should contain a JSON array: {ids_file}")
                return None
        except Exception as e:
            logger.error(f"Failed to load selected IDs from {ids_file}: {e}")
            return None

    def _recover_partial_trajectory_stats(
        self,
        level_id: str,
        started_at_epoch: float,
    ) -> Dict[str, float]:
        """Recover partial thread progress from trajectories written on cancellation.

        A task-level wall-clock cap can cancel all solver tasks before any of
        them returns a LevelResult to the orchestrator. The solver finally block
        still writes per-thread trajectories, so use them to avoid recording the
        aggregate row as a misleading zero-attempt infra failure.
        """
        stats: Dict[str, float] = {
            "thread_count": 0.0,
            "max_attempts": 0.0,
            "sum_attempts": 0.0,
            "total_cost": 0.0,
        }
        try:
            if not self._trajectory_dir.exists():
                return stats

            safe_id = level_id.replace("/", "_")
            for path in self._trajectory_dir.glob(f"{safe_id}_t*_*.json"):
                try:
                    if path.stat().st_mtime + 2.0 < started_at_epoch:
                        continue
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                attempts = data.get("total_attempts")
                if attempts is None:
                    attempts = len(data.get("attempts") or [])
                try:
                    attempts_f = float(attempts or 0.0)
                except (TypeError, ValueError):
                    attempts_f = 0.0

                try:
                    cost_f = float(data.get("total_cost") or 0.0)
                except (TypeError, ValueError):
                    cost_f = 0.0

                stats["thread_count"] += 1.0
                stats["max_attempts"] = max(stats["max_attempts"], attempts_f)
                stats["sum_attempts"] += attempts_f
                stats["total_cost"] += cost_f
        except Exception as e:
            logger.warning(
                f"[SWEBenchOrchestra] [{level_id}] partial trajectory recovery failed: {e}"
            )
        return stats

    def list_levels(self) -> List[LevelSpec]:
        """List all available SWE-bench instances."""
        instances = self._data_loader.load_instances()

        selected_ids = self._load_selected_ids()

        levels = []
        for inst in instances:

            if selected_ids is not None and inst.instance_id not in selected_ids:
                continue

            self._instances[inst.instance_id] = inst
            levels.append({
                "id": inst.instance_id,
                "_instance": inst,
            })

        logger.info(f"Loaded {len(levels)} SWE-bench instances")
        if selected_ids is not None:
            logger.info(f"  (filtered from {len(selected_ids)} selected IDs)")
        return levels

    def make_env(self, level: LevelSpec, thread_id: int = 0) -> SWEBenchEnvironment:
        """Create environment for a specific instance.

        thread_id is used to derive a per-thread name_suffix so the
        docker container name + workspace tempdir + per-task logs folder are
        distinct across parallel solver threads. thread_id=0 with single
        solver (n_solvers=1) produces no suffix — backward compatible.
        """
        instance = level.get("_instance") or self._instances.get(level["id"])
        if not instance:
            raise ValueError(f"Instance not found: {level['id']}")

        swe_config = SWEBenchConfig(
            dataset_name=self.config.dataset_name,
            split=self.config.split,
            subset_seed=self.config.subset_seed,
            subset_sizes=self.config.subset_sizes,
            subset_role=self.config.subset_role,
            max_steps=self.config.max_steps,
            max_tasks=self.config.max_tasks,
            docker_timeout=self.config.docker_timeout,
            grading_timeout=getattr(self.config, "grading_timeout", 600),
            model=self.config.planner_model,
            result_folder=self.config.result_folder,
            trajectory_dir=self.config.trajectory_dir,
            csv_summary_path=self.config.csv_summary_path,
            timestamp=self.config.timestamp,
            env_init=self.config.env_init,
            cache_dir=self.config.cache_dir,
            window_size=self.config.window_size,
        )

        # No suffix for single-solver runs (n_solvers=1) — preserves old
        # behavior + lets existing recovery tooling find containers by id.
        name_suffix = "" if self.n_solvers <= 1 else f"_t{int(thread_id)}"
        return SWEBenchEnvironment(
            level,
            swe_config,
            instance,
            name_suffix=name_suffix,
            # Multi-agent flow: only the PlannerAgent's explicit submit grades;
            # an ImplementerAgent exhausting max_steps must not auto-run eval.
            auto_grade_on_max_steps=False,
        )

    async def run(self, levels: List[LevelSpec], max_concurrency: int = 1):
        """Run benchmark — @N: each task spawns N parallel solvers.

        Each solver gets its own SWEBenchEnvironment (own container) and its
        own SWEBenchRunner instance. All N share the per-task SharedLessons
        blackboard. asyncio.gather across the N solvers; the per-task wall-
        clock cap wraps the gather (so the cap is shared across threads, not
        per-thread).

        Pass@N aggregation: final_reward = max(per_thread rewards). One CSV
        row per task, attributed to the winning thread; new columns capture
        the ensemble structure (n_solvers_total/passed, winning_thread_id).
        Per-thread costs / tokens / wall / active sum across the N threads.
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results = {}
        N = self.n_solvers
        # task_token_budget is per-thread, not split N-ways.
        budget_per_thread = int(getattr(self.config, "task_token_budget", 1_000_000))

        async def _run_one_thread(level: LevelSpec, thread_id: int, shared_lessons,
                                  wall_clock_deadline_perf: Optional[float] = None) -> Optional[LevelResult]:
            """Spawn a single solver for `level` on thread `thread_id`. Returns LevelResult or None."""
            env = None
            level_id = level.get("id", str(level))
            try:
                env = self.make_env(level, thread_id=thread_id)
                logger.info(f"[SWEBenchOrchestra] [t{thread_id}] resetting env for {level_id}")
                await env.reset()
                runner = SWEBenchRunner(
                    planner_model=self.config.planner_model,
                    implementer_models=self.config.implementer_models,
                    max_attempts=self.config.max_attempts,
                    prompt_builder=SWEBenchPlannerPrompt,
                    trajectory_dir=self._trajectory_dir,
                    csv_summary_path=None,
                    task_token_budget=budget_per_thread,
                    implementer_model=getattr(self.config, "implementer_model", None),
                    implementer_max_memory=getattr(self.config, "implementer_max_memory", 8),
                    implementer_max_obs_chars=getattr(self.config, "implementer_max_obs_chars", 2000),
                    implementer_max_thinking_chars=getattr(self.config, "implementer_max_thinking_chars", 500),
                    summarizer_model=getattr(self.config, "summarizer_model", None),
                    memory_compaction_mode=getattr(self.config, "memory_compaction_mode", "llm"),
                    shared_lessons_implementer_window_tokens=getattr(self.config, "shared_lessons_implementer_window_tokens", 1000),
                    shared_lessons_planner_window_tokens=getattr(self.config, "shared_lessons_planner_window_tokens", 2000),
                    shared_lessons_render_mode=getattr(self.config, "shared_lessons_render_mode", "full"),
                    planner_max_tokens=getattr(self.config, "planner_max_tokens", 1200),
                    patch_ready_checkpoint_enabled=getattr(self.config, "patch_ready_checkpoint_enabled", False),
                    peer_patch_checkpoint_enabled=getattr(self.config, "peer_patch_checkpoint_enabled", False),
                    auto_submit_on_done_handoff=getattr(self.config, "auto_submit_on_done_handoff", False),
                    implementer_fast_finish_prompt_enabled=getattr(self.config, "implementer_fast_finish_prompt_enabled", False),
                    peer_patch_protocol_enabled=getattr(self.config, "peer_patch_protocol_enabled", False),
                    patch_summary_timing_rule_enabled=getattr(self.config, "patch_summary_timing_rule_enabled", False),
                    patch_summary_finish_handoff_enabled=getattr(self.config, "patch_summary_finish_handoff_enabled", False),
                    implementer_strategy_prompt_enabled=getattr(self.config, "implementer_strategy_prompt_enabled", False),
                    prompt_cache_enabled=getattr(self.config, "prompt_cache_enabled", False),
                    prompt_cache_min_prefix_chars=getattr(self.config, "prompt_cache_min_prefix_chars", 16000),
                    thread_id=thread_id,
                    n_threads=N,
                    shared_lessons=shared_lessons,
                    write_csv=False,
                    step_timeout=getattr(self.config, "step_timeout", 120),
                    exec_timeout=getattr(self.config, "docker_timeout", 300) + 60,
                    wall_clock_deadline_perf=wall_clock_deadline_perf,
                    grading_reserve_seconds=getattr(self.config, "grading_reserve_seconds", 120),
                    note_verifier_mode=getattr(self.config, "note_verifier_mode", "deterministic"),
                    refresh_mode=getattr(self.config, "refresh", "legacy_full"),
                )
                return await runner.run(None, env)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                import traceback
                logger.error(f"[SWEBenchOrchestra] [t{thread_id}] {level_id} failed: {e}")
                logger.error(f"[SWEBenchOrchestra] Traceback:\n{traceback.format_exc()}")
                return None
            finally:
                if env and hasattr(env, "close"):
                    try:
                        await env.close()
                    except Exception as cleanup_err:
                        logger.warning(f"[SWEBenchOrchestra] [t{thread_id}] cleanup error for {level_id}: {cleanup_err}")

        async def run_level(level):
            level_id = level.get("id", str(level))
            async with semaphore:
                level_started_at_epoch = time.time()
                start_stagger = max(
                    0.0,
                    float(getattr(self.config, "solver_start_stagger_seconds", 0.0) or 0.0),
                )
                stagger_msg = f", start_stagger={start_stagger:.1f}s" if start_stagger else ""
                logger.info(
                    f"[SWEBenchOrchestra] [{level_id}] spawning {N} parallel solvers "
                    f"(pass@{N}{stagger_msg})"
                )
                lessons_enabled = getattr(self.config, "shared_lessons_enabled", True)
                if not lessons_enabled:
                    shared_lessons = None
                    logger.info(f"[SWEBenchOrchestra] [{level_id}] shared_lessons disabled")
                else:
                    lessons_path = self._lessons_root / f"{level_id.replace('/', '_')}.jsonl"
                    feature_flags = {
                        "tried_admission_filter": getattr(self.config, "tried_admission_filter", True),
                        "patch_summary_enabled": getattr(self.config, "patch_summary_enabled", False),
                        "claims_enabled": getattr(self.config, "claims_enabled", False),
                        "patch_summary_latest_wins_enabled": getattr(
                            self.config,
                            "patch_summary_latest_wins_enabled",
                            False,
                        ),
                        "patch_summary_lifecycle_enabled": getattr(
                            self.config,
                            "patch_summary_lifecycle_enabled",
                            False,
                        ),
                    }
                    shared_lessons = SharedLessons(
                        task_id=level_id,
                        file_path=lessons_path,
                        feature_flags=feature_flags,
                    )
                cap = getattr(self.config, "task_max_wall_clock_seconds", 360)

                if start_stagger > 0 and N > 1:
                    max_stagger = max(0.0, cap - 120.0) / (N - 1)
                    if start_stagger > max_stagger:
                        logger.warning(
                            f"[SWEBenchOrchestra] [{level_id}] solver_start_stagger_seconds="
                            f"{start_stagger:.1f}s too large for cap={cap}s/N={N}; clamping to "
                            f"{max_stagger:.1f}s so t{N-1} keeps >=120s of runtime"
                        )
                        start_stagger = max_stagger

                async def _log_shared_lessons_stats() -> None:
                    """Log render-window stats once per task for shared-lessons tuning."""
                    if shared_lessons is None:
                        return
                    try:
                        implementer_w = getattr(self.config, "shared_lessons_implementer_window_tokens", 1000)
                        planner_w = getattr(self.config, "shared_lessons_planner_window_tokens", 2000)
                        implementer_stats = await shared_lessons.get_stats(window_tokens=implementer_w)
                        planner_stats = await shared_lessons.get_stats(window_tokens=planner_w)
                        logger.info(
                            f"[SharedLessons] [{level_id}] "
                            f"implementer_window={implementer_stats} planner_window={planner_stats}"
                        )
                    except Exception as stats_err:
                        logger.warning(
                            f"[SharedLessons] [{level_id}] stats unavailable: {stats_err}"
                        )
                    lifecycle_path = self._lessons_root / (
                        f"{level_id.replace('/', '_')}_lifecycle.jsonl"
                    )
                    await shared_lessons.write_lifecycle_summary(lifecycle_path)

                async def _run_one_thread_after_stagger(
                    level: LevelSpec,
                    thread_id: int,
                    shared_lessons,
                    wall_clock_deadline_perf: Optional[float] = None,
                ) -> Optional[LevelResult]:
                    delay = start_stagger * int(thread_id)
                    if delay > 0:
                        logger.info(
                            f"[SWEBenchOrchestra] [{level_id}] [t{thread_id}] "
                            f"delaying start by {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                    return await _run_one_thread(
                        level, thread_id, shared_lessons,
                        wall_clock_deadline_perf=wall_clock_deadline_perf,
                    )

                deadline_perf = time.perf_counter() + cap
                tasks = [
                    asyncio.ensure_future(
                        _run_one_thread_after_stagger(level, tid, shared_lessons, deadline_perf)
                    )
                    for tid in range(N)
                ]
                done, pending = await asyncio.wait(tasks, timeout=cap)
                if pending:
                    logger.warning(
                        f"[SWEBenchOrchestra] [{level_id}] hit wall-clock cap ({cap}s); "
                        f"cancelling {len(pending)} unfinished solver(s), keeping "
                        f"{len(done)} completed"
                    )
                    for t in pending:
                        t.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                raw_results = []
                for t in tasks:
                    if t.cancelled():
                        raw_results.append(None)
                    else:
                        try:
                            raw_results.append(t.result())
                        except Exception as e:
                            raw_results.append(e)

                per_thread: Dict[int, LevelResult] = {}
                for tid, r in enumerate(raw_results):
                    if isinstance(r, LevelResult):
                        per_thread[tid] = r
                    elif isinstance(r, Exception):
                        logger.warning(f"[SWEBenchOrchestra] [t{tid}] {level_id} exception: {r}")

                if not per_thread:
                    recovered_cost = self._sum_events_cost(level_id)
                    partial_stats = self._recover_partial_trajectory_stats(
                        level_id,
                        started_at_epoch=level_started_at_epoch,
                    )
                    recovered_steps = int(partial_stats.get("max_attempts", 0.0) or 0.0)
                    if recovered_steps > 0:
                        logger.warning(
                            f"[SWEBenchOrchestra] [{level_id}] recovered partial progress "
                            f"from {int(partial_stats.get('thread_count', 0.0))} trajectory "
                            f"file(s): max_attempts={recovered_steps}, "
                            f"sum_attempts={int(partial_stats.get('sum_attempts', 0.0))}"
                        )
                    logger.warning(
                        f"[SWEBenchOrchestra] [{level_id}] all {N} solvers failed/cancelled "
                        f"(cap={cap}s, pending={len(pending)}); recording reward-0 fail row "
                        f"(cost=${recovered_cost:.4f} recovered from events.jsonl)"
                    )
                    fail_result = LevelResult(
                        model=self.planner_model,
                        total_reward=0.0,
                        steps=recovered_steps,
                        done=False,
                        trace=[],
                        cost=recovered_cost,
                        forced_cap=bool(pending),
                        wall_clock_seconds=float(cap) if pending else 0.0,
                        n_solvers_total=N,
                        n_solvers_passed=0,
                        winning_thread_id="-",
                    )
                    results[level_id] = fail_result
                    try:
                        await save_csv_row(self._csv_path, self._csv_lock, level_id, fail_result)
                    except Exception as e:
                        logger.error(f"[SWEBenchOrchestra] [{level_id}] fail-row CSV write failed: {e}")
                    await _log_shared_lessons_stats()
                    return

                rewards = {tid: lr.total_reward for tid, lr in per_thread.items()}
                max_reward = max(rewards.values())
                winning_tid = min(
                    (tid for tid, r in rewards.items() if r == max_reward),
                    default=None,
                )
                n_passed = sum(1 for r in rewards.values() if r > 0)

                cancelled_tids = {tid for tid in range(N) if tid not in per_thread}
                recovered_cost = 0.0
                if cancelled_tids:
                    recovered_cost = self._sum_events_cost(level_id, thread_ids=cancelled_tids)
                    if recovered_cost > 0:
                        logger.warning(
                            f"[SWEBenchOrchestra] [{level_id}] partial cap: recovered "
                            f"${recovered_cost:.4f} from {len(cancelled_tids)} cancelled "
                            f"thread(s) {sorted(cancelled_tids)}"
                        )

                primary = per_thread[winning_tid]
                primary.total_reward = max_reward
                primary.cost = sum(lr.cost for lr in per_thread.values()) + recovered_cost
                primary.input_tokens = sum(lr.input_tokens for lr in per_thread.values())
                primary.output_tokens = sum(lr.output_tokens for lr in per_thread.values())
                primary.wall_clock_seconds = max(lr.wall_clock_seconds for lr in per_thread.values())
                primary.active_llm_time_seconds = sum(lr.active_llm_time_seconds for lr in per_thread.values())
                primary.delegate_count = sum(lr.delegate_count for lr in per_thread.values())
                primary.submit_count = sum(lr.submit_count for lr in per_thread.values())
                primary.error_count = sum(lr.error_count for lr in per_thread.values())
                primary.implementer_steps_total = sum(lr.implementer_steps_total for lr in per_thread.values())
                primary.forced_cap = any(lr.forced_cap for lr in per_thread.values()) or bool(cancelled_tids)
                primary.forced_submit = any(lr.forced_submit for lr in per_thread.values())
                primary.n_solvers_total = N
                primary.n_solvers_passed = n_passed
                primary.winning_thread_id = str(winning_tid) if max_reward > 0 else "-"

                logger.info(
                    f"[SWEBenchOrchestra] [{level_id}] pass@{N} reward={max_reward} "
                    f"(passed={n_passed}/{N} winner=t{winning_tid if max_reward > 0 else '-'} "
                    f"cost=${primary.cost:.4f})"
                )
                results[level_id] = primary

                # Aggregate CSV write (single row for this task)
                try:
                    await save_csv_row(self._csv_path, self._csv_lock, level_id, primary)
                except Exception as e:
                    logger.error(f"[SWEBenchOrchestra] [{level_id}] CSV write failed: {e}")
                await _log_shared_lessons_stats()

                if len(per_thread) >= 2:
                    try:
                        from src.metrics import write_divergence_summary
                        div_path = self._trajectory_dir / f"{level_id.replace('/', '_')}_divergence.json"
                        write_divergence_summary(div_path, level_id, per_thread)
                    except Exception as e:
                        logger.warning(
                            f"[SWEBenchOrchestra] [{level_id}] divergence-metrics write failed: {e}"
                        )

        # Run tasks with proper exception handling
        tasks = [asyncio.create_task(run_level(level)) for level in levels]
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for any unhandled exceptions from gather
        for i, res in enumerate(gather_results):
            if isinstance(res, Exception):
                level_id = levels[i].get("id", str(levels[i]))
                logger.error(f"[SWEBenchOrchestra] Unhandled exception for {level_id}: {res}")
                if level_id not in results:
                    results[level_id] = None

        total = len(results)
        passN_count = sum(1 for r in results.values() if r and r.total_reward > 0)
        passed_threads = sum((getattr(r, "n_solvers_passed", 0) or 0) for r in results.values() if r)
        total_threads = sum((getattr(r, "n_solvers_total", 0) or 0) if r else self.n_solvers
                            for r in results.values())
        avg1 = (passed_threads / total_threads) if total_threads else 0.0

        logger.info(f"\n{'='*60}")
        logger.info(f"SWE-bench Summary:")
        logger.info(f"  Total tasks: {total}")
        logger.info(f"  avg@1 (per-thread, leaderboard-comparable): "
                    f"{passed_threads}/{total_threads} = {avg1*100:.1f}%")
        logger.info(f"  pass@{self.n_solvers} (ORACLE best-of-{self.n_solvers}, upper bound only): "
                    f"{passN_count}/{total}")
        logger.info(f"{'='*60}")

        return results
