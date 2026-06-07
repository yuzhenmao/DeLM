"""
Single solver thread for SWE-bench.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List, Optional

from base.engine.async_llm import (
    LLMsConfig,
    create_llm_instance,
    current_task_id,
    current_task_latencies_ms,
)
from base.engine.logs import logger
from benchmark.common.env import BasicInfo, Environment
from benchmark.common.runner import Runner, StepRecord, LevelResult
from src.agents.planner_agent import PlannerAgent
from src.prompts.swebench_global import SWEBenchPlannerPrompt
from src.results import save_csv_row, save_trajectory
from src.runners.implementer_runner import SWEBenchImplementerRunner
from src.shared_lessons import SharedLessons
from src.swebench_modes import SWEBENCH_GLOBAL_MODE
from src.tools.delegate import DelegateTaskTool
from src.tools.submit import SubmitTool

class SWEBenchRunner(Runner):
    """Runner for PlannerAgent with ImplementerAgent delegation for SWE-bench."""

    def __init__(
        self,
        planner_model: str,
        implementer_models: List[str],
        max_attempts: int = 6,
        prompt_builder=None,
        trajectory_dir: Path | None = None,
        csv_summary_path: Path | None = None,
        task_token_budget: int = 1_000_000,
        implementer_model: str = None,
        thread_id: int = 0,
        n_threads: int = 1,
        shared_lessons: Any = None,
        write_csv: bool = True,
        step_timeout: Optional[float] = 120,
        exec_timeout: Optional[float] = None,
        wall_clock_deadline_perf: Optional[float] = None,
        grading_reserve_seconds: float = 120.0,
        note_verifier_mode: str = "deterministic",
        implementer_max_memory: int = 8,
        implementer_max_obs_chars: int = 2000,
        implementer_max_thinking_chars: int = 500,
        summarizer_model: str = None,
        shared_lessons_implementer_window_tokens: int = 1000,
        shared_lessons_planner_window_tokens: int = 2000,
        shared_lessons_render_mode: str = "full",
        planner_max_tokens: Optional[int] = 1200,
        memory_compaction_mode: str = "llm",
        patch_ready_checkpoint_enabled: bool = False,
        peer_patch_checkpoint_enabled: bool = False,
        auto_submit_on_done_handoff: bool = False,
        implementer_fast_finish_prompt_enabled: bool = False,
        peer_patch_protocol_enabled: bool = False,
        patch_summary_timing_rule_enabled: bool = False,
        patch_summary_finish_handoff_enabled: bool = False,
        implementer_strategy_prompt_enabled: bool = False,
        prompt_cache_enabled: bool = False,
        prompt_cache_min_prefix_chars: int = 16000,
    ):
        self.planner_model = planner_model
        self.implementer_models = implementer_models
        self.step_timeout = step_timeout
        self.exec_timeout = exec_timeout
        self.wall_clock_deadline_perf = wall_clock_deadline_perf
        self.grading_reserve_seconds = float(grading_reserve_seconds)
        self.note_verifier_mode = str(note_verifier_mode or "deterministic").lower()
        self.implementer_max_memory = int(implementer_max_memory)
        self.implementer_max_obs_chars = int(implementer_max_obs_chars)
        self.implementer_max_thinking_chars = int(implementer_max_thinking_chars)
        self.summarizer_model = summarizer_model
        self.memory_compaction_mode = memory_compaction_mode
        self.patch_ready_checkpoint_enabled = bool(patch_ready_checkpoint_enabled)
        self.peer_patch_checkpoint_enabled = bool(peer_patch_checkpoint_enabled)
        self.auto_submit_on_done_handoff = bool(auto_submit_on_done_handoff)
        self.implementer_fast_finish_prompt_enabled = bool(implementer_fast_finish_prompt_enabled)
        self.peer_patch_protocol_enabled = bool(peer_patch_protocol_enabled)
        self.patch_summary_timing_rule_enabled = bool(patch_summary_timing_rule_enabled)
        self.patch_summary_finish_handoff_enabled = bool(patch_summary_finish_handoff_enabled)
        self.implementer_strategy_prompt_enabled = bool(implementer_strategy_prompt_enabled)
        self.prompt_cache_enabled = bool(prompt_cache_enabled)
        self.prompt_cache_min_prefix_chars = int(prompt_cache_min_prefix_chars)
        self.shared_lessons_implementer_window_tokens = int(shared_lessons_implementer_window_tokens)
        self.shared_lessons_planner_window_tokens = int(shared_lessons_planner_window_tokens)
        self.shared_lessons_render_mode = str(shared_lessons_render_mode or "full").lower()
        self.planner_max_tokens = planner_max_tokens
        self.implementer_model = implementer_model or (implementer_models[0] if implementer_models else "")
        self.max_attempts = max_attempts
        self.task_token_budget = task_token_budget
        self.prompt_builder = prompt_builder or SWEBenchPlannerPrompt
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir else None
        self.csv_summary_path = Path(csv_summary_path) if csv_summary_path else None
        self._csv_lock = asyncio.Lock()
        self.thread_id = int(thread_id)
        self.n_threads = int(n_threads)
        self.shared_lessons = shared_lessons
        self.write_csv = bool(write_csv)

    async def run(self, agent, env: Environment) -> LevelResult:
        """Run PlannerAgent orchestration for SWE-bench."""
        env_info = env.get_basic_info()
        logger.info(f"[SWEBenchOrchestra] Starting task: {env_info.env_id}")

        _task_event_id = env_info.env_id if self.n_threads <= 1 else f"{env_info.env_id}#t{self.thread_id}"
        current_task_id.set(_task_event_id)

        # initialize per-task LLM latency accumulator + wall-clock anchor.
        import time as _time_mod
        task_latencies_ms: List[float] = []
        current_task_latencies_ms.set(task_latencies_ms)
        task_start_perf = _time_mod.perf_counter()

        agent_stop_perf: Optional[float] = None
        if self.wall_clock_deadline_perf is not None:
            agent_stop_perf = self.wall_clock_deadline_perf - self.grading_reserve_seconds
            _executor0 = getattr(env, "_executor", None)
            if _executor0 is not None:
                _executor0.grading_deadline_perf = self.wall_clock_deadline_perf - 15.0

        # Create PlannerAgent info
        planner_info = BasicInfo(
            env_id=env_info.env_id,
            instruction=env_info.instruction,
            action_space="",
            max_steps=self.max_attempts,
            meta_data=env_info.meta_data,
        )

        # Create PlannerAgent with tools
        logger.info(f"[SWEBenchOrchestra] Creating PlannerAgent with model={self.planner_model}")
        planner_llm = create_llm_instance(LLMsConfig.default().get(self.planner_model), role="PlannerAgent")
        planner_llm.prompt_cache_enabled = self.prompt_cache_enabled
        planner_llm.prompt_cache_min_prefix_chars = self.prompt_cache_min_prefix_chars
        implementer_runner = SWEBenchImplementerRunner()
        if self.step_timeout is not None:
            implementer_runner.step_timeout = self.step_timeout
        if self.exec_timeout is not None:
            implementer_runner.exec_timeout = self.exec_timeout
        implementer_runner.patch_ready_checkpoint_enabled = self.patch_ready_checkpoint_enabled
        implementer_runner.patch_summary_finish_handoff_enabled = self.patch_summary_finish_handoff_enabled
        implementer_runner.agent_stop_perf = agent_stop_perf
        implementer_runner.note_verifier_mode = self.note_verifier_mode

        delegate = DelegateTaskTool(
            env=env,
            runner=implementer_runner,
            models=self.implementer_models,
            implementer_model=self.implementer_model,
            thread_id=self.thread_id,
            shared_lessons=self.shared_lessons,
            implementer_max_memory=self.implementer_max_memory,
            implementer_max_obs_chars=self.implementer_max_obs_chars,
            implementer_max_thinking_chars=self.implementer_max_thinking_chars,
            summarizer_model=self.summarizer_model,
            shared_lessons_window_tokens=self.shared_lessons_implementer_window_tokens,  # 
            shared_lessons_render_mode=self.shared_lessons_render_mode,
            memory_compaction_mode=self.memory_compaction_mode,  # 
            peer_patch_checkpoint_enabled=self.peer_patch_checkpoint_enabled,
            implementer_fast_finish_prompt_enabled=self.implementer_fast_finish_prompt_enabled,
            peer_patch_protocol_enabled=self.peer_patch_protocol_enabled,
            patch_summary_timing_rule_enabled=self.patch_summary_timing_rule_enabled,
            implementer_strategy_prompt_enabled=self.implementer_strategy_prompt_enabled,
            prompt_cache_enabled=self.prompt_cache_enabled,
            prompt_cache_min_prefix_chars=self.prompt_cache_min_prefix_chars,
        )
        submit = SubmitTool(env=env)

        planner_agent = PlannerAgent(
            llm=planner_llm,
            implementer_models=self.implementer_models,
            tools=[delegate, submit],
            prompt_builder=self.prompt_builder,
            max_attempts=self.max_attempts,
            thread_id=self.thread_id,
            n_threads=self.n_threads,
            shared_lessons=self.shared_lessons,
            shared_lessons_window_tokens=self.shared_lessons_planner_window_tokens,  # 
            shared_lessons_render_mode=self.shared_lessons_render_mode,
            planner_max_tokens=self.planner_max_tokens,
            mode_spec=SWEBENCH_GLOBAL_MODE,
        )
        planner_agent.reset(planner_info)

        # Orchestration loop
        history = []
        total_reward = 0.0
        total_implementer_cost = 0.0
        total_implementer_input_tokens = 0
        total_implementer_output_tokens = 0
        total_implementer_steps = 0
        delegate_count = 0
        submit_count = 0
        error_count = 0
        done = False
        level_result = None
        exception_occurred = None
        forced_cap = False
        forced_submit = False

        try:
            attempt_idx = 0
            while attempt_idx < self.max_attempts:
                if agent_stop_perf is not None and _time_mod.perf_counter() >= agent_stop_perf:
                    logger.warning(
                        f"[SWEBenchOrchestra] wall-clock reserve reached "
                        f"(~{self.grading_reserve_seconds:.0f}s before cap); stopping agent work "
                        f"and force-submitting so the official grade fits in budget"
                    )
                    break
                # Compute live token total across PlannerAgent + all ImplementerAgents so far.
                _usage = planner_agent.llm.get_usage_summary()
                _live_tokens = (
                    _usage.get("total_input_tokens", 0)
                    + _usage.get("total_output_tokens", 0)
                    + total_implementer_input_tokens
                    + total_implementer_output_tokens
                )

                if self.task_token_budget and _live_tokens >= self.task_token_budget:
                    logger.warning(
                        f"[SWEBenchOrchestra] Token budget exhausted "
                        f"({_live_tokens:,}/{self.task_token_budget:,}); force-submitting partial state"
                    )
                    forced_submit = True
                    executor = getattr(env, "_executor", None)
                    container_ready = hasattr(env, '_container_started') and env._container_started
                    if container_ready and executor and hasattr(executor, "run_tests"):
                        try:
                            reward, _ = await executor.run_tests()
                            total_reward = float(reward)
                            done = True
                        except Exception as fs_err:
                            logger.error(f"[SWEBenchOrchestra] Budget force-submit failed: {fs_err}")
                            done = True
                            total_reward = 0.0
                    else:
                        done = True
                        total_reward = 0.0
                    break

                attempt_idx += 1
                logger.info(
                    f"[SWEBenchOrchestra] PlannerAgent attempt {attempt_idx}/{self.max_attempts} "
                    f"(tokens={_live_tokens:,}/{self.task_token_budget:,})"
                )

                try:
                    action, resp = await planner_agent.step(None, history)
                except Exception as step_error:
                    logger.error(
                        f"[SWEBenchOrchestra] PlannerAgent.step() FAILED: {step_error}; "
                        f"breaking to force-submit current container state",
                        exc_info=True,
                    )
                    error_count += 1
                    break

                action_name = action.get("action")
                result = action.get("result", {})
                reward = result.get("reward", 0.0)
                step_done = result.get("done", False)
                is_submit = action_name == "submit"

                if action_name == "delegate_task":
                    delegate_count += 1
                    total_implementer_cost += result.get("cost", 0.0)
                    total_implementer_input_tokens += result.get("input_tokens", 0)
                    total_implementer_output_tokens += result.get("output_tokens", 0)
                    total_implementer_steps += result.get("steps_taken", 0)
                elif is_submit:
                    submit_count += 1
                else:
                    error_count += 1

                history.append(StepRecord(
                    observation={},
                    action=action,
                    reward=reward,
                    raw_response=resp,
                    done=step_done,
                    info=result,
                ))

                if is_submit:
                    total_reward = reward

                if step_done and is_submit:
                    done = True
                    break

                if (
                    self.auto_submit_on_done_handoff
                    and action_name == "delegate_task"
                    and isinstance(result, dict)
                ):
                    finish_result = result.get("finish_result") or {}
                    finish_status = str(finish_result.get("status", "")).lower()
                    finish_issues = finish_result.get("issues") or []
                    if finish_status == "done" and not finish_issues:
                        reason = (
                            "auto_submit_on_done_handoff: ImplementerAgent returned "
                            "`finish done` with no reported issues"
                        )
                        logger.info(
                            "[SWEBenchOrchestra] auto-submitting after ImplementerAgent "
                            "finish-done handoff"
                        )
                        try:
                            submit_result = await submit(reason=reason)
                        except Exception as auto_submit_err:
                            logger.error(
                                f"[SWEBenchOrchestra] auto-submit after done handoff failed: "
                                f"{auto_submit_err}",
                                exc_info=True,
                            )
                            error_count += 1
                            break
                        submit_count += 1
                        submit_reward = float(submit_result.get("reward", 0.0) or 0.0)
                        total_reward = submit_reward
                        history.append(StepRecord(
                            observation={},
                            action={
                                "action": "submit",
                                "params": {"reason": reason},
                                "result": submit_result,
                                "auto_submit_on_done_handoff": True,
                            },
                            reward=submit_reward,
                            raw_response="[auto_submit_on_done_handoff]",
                            done=bool(submit_result.get("done", True)),
                            info=submit_result,
                        ))
                        done = True
                        break

            if not done:
                logger.warning(
                    f"[SWEBenchOrchestra] max_attempts={self.max_attempts} reached without submit; "
                    f"force-submitting partial state"
                )
                forced_submit = True
                executor = getattr(env, "_executor", None)
                container_ready = hasattr(env, '_container_started') and env._container_started
                if container_ready and executor and hasattr(executor, "run_tests"):
                    try:
                        reward, _ = await executor.run_tests()
                        total_reward = float(reward)
                        done = True
                    except Exception as e:
                        logger.error(f"[SWEBenchOrchestra] Forced submit failed: {e}")
                        done = True
                        total_reward = 0.0
                else:
                    done = True
                    total_reward = 0.0

        except asyncio.CancelledError:
            logger.warning("[SWEBenchOrchestra] CancelledError received (likely wall-clock cap); forced_cap=True")
            forced_cap = True
            raise

        except Exception as e:
            logger.error(f"[SWEBenchOrchestra] Exception: {e}", exc_info=True)
            exception_occurred = e

        finally:
            # Cleanup
            if hasattr(env, 'close'):
                try:
                    await env.close()
                except Exception as e:
                    logger.error(f"[SWEBenchOrchestra] Cleanup error: {e}")

            try:
                usage = planner_agent.llm.get_usage_summary() if planner_agent else {}
                planner_cost = usage.get("total_cost", 0.0)
                total_cost = planner_cost + total_implementer_cost
                planner_input_tokens = usage.get("total_input_tokens", 0)
                planner_output_tokens = usage.get("total_output_tokens", 0)

                wall_clock_seconds = _time_mod.perf_counter() - task_start_perf
                active_llm_time_seconds = sum(task_latencies_ms) / 1000.0

                level_result = LevelResult(
                    model=usage.get("model", self.planner_model),
                    total_reward=total_reward,
                    steps=len(history),
                    done=done,
                    trace=history,
                    cost=total_cost,
                    input_tokens=planner_input_tokens + total_implementer_input_tokens,
                    output_tokens=planner_output_tokens + total_implementer_output_tokens,
                    delegate_count=delegate_count,
                    submit_count=submit_count,
                    error_count=error_count,
                    implementer_steps_total=total_implementer_steps,
                    forced_cap=forced_cap,
                    forced_submit=forced_submit,
                    wall_clock_seconds=wall_clock_seconds,
                    active_llm_time_seconds=active_llm_time_seconds,
                )

                if self.trajectory_dir:
                    self._save_trajectory(env_info, level_result, history)

                if self.csv_summary_path and self.write_csv:
                    await self._save_csv(env_info.env_id, level_result)

            except Exception as save_error:
                logger.error(f"[SWEBenchOrchestra] Save error: {save_error}")

        if exception_occurred:
            raise exception_occurred

        return level_result

    def _save_trajectory(self, info: BasicInfo, result: LevelResult,
                         history: List[StepRecord]) -> None:
        """Save the per-thread trajectory JSON.

        Pass `thread_id` so parallel solver threads on the same task don't
        overwrite each other's file. n_threads==1 keeps the legacy filename
        (thread_id=None).
        """
        save_trajectory(
            trajectory_dir=self.trajectory_dir,
            info=info,
            result=result,
            history=history,
            planner_model=self.planner_model,
            implementer_models=self.implementer_models,
            thread_id=self.thread_id if self.n_threads > 1 else None,
        )

    async def _save_csv(self, task_id: str, result: LevelResult) -> None:
        """Delegate to src.results.save_csv_row ()."""
        await save_csv_row(
            csv_path=self.csv_summary_path,
            lock=self._csv_lock,
            task_id=task_id,
            result=result,
        )
