"""Configuration for the SWE-bench orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

@dataclass
class SWEBenchOrchestraConfig:
    """SWE-bench Orchestra configuration."""

    # Models
    planner_model: str
    implementer_models: List[str]

    # Dataset
    dataset_name: str = "princeton-nlp/SWE-bench_Verified"
    split: str = "test"
    subset_seed: Optional[int] = None
    subset_sizes: Optional[Dict[str, int]] = None
    subset_role: Optional[str] = None
    selected_ids_file: Optional[Path] = None

    # Task settings
    max_tasks: Optional[int] = None
    max_steps: int = 50                          # ImplementerAgent max steps per delegation
    max_attempts: int = 6                        # PlannerAgent ceiling; token/wall caps usually bind first

    # Execution
    max_concurrency: int = 1
    docker_timeout: int = 300                    # per-shell-command timeout (docker exec)
    step_timeout: int = 120                      # per-step LLM-call cap
    grading_timeout: int = 300                   # cap on the eval.sh verification run
    task_max_wall_clock_seconds: int = 480       # per-task overall cap
    grading_reserve_seconds: int = 120           # wall-clock kept back for the final grade: the
                                                 # solver self-stops agent work this many seconds
                                                 # before the cap so a force-submit can grade inside it.
    task_token_budget: int = 1_000_000           # per-thread token cap; force-submit on exhaust. 0 disables.

    implementer_model: Optional[str] = None      # pinned ImplementerAgent model; falls back to implementer_models[0]
    n_solvers: int = 2                           # parallel solver threads per task (pass@N)
    solver_start_stagger_seconds: float = 0.0    # optional delay between thread launches

    # ImplementerAgent memory clamps
    implementer_max_memory: int = 8              # raw records retained before summarization
    implementer_max_obs_chars: int = 2000        # per-observation/action char cap
    implementer_max_thinking_chars: int = 500    # per-`thinking` field char cap

    summarizer_model: Optional[str] = "google/gemini-3-flash-preview"
                                                 # LLM for memory compression; null in YAML → fall back to implementer_model
    memory_compaction_mode: str = "llm"          # "llm" (LLM summarizer) or "deterministic" (zero-cost structured)
    note_verifier_mode: str = "deterministic"    # shared-lessons note hygiene: "deterministic" (rule-based, free)
                                                 # or "llm" (1 extra LLM call/step, agent model). Default deterministic.

    # Shared-lessons feature flags. The four core channels default ON; all
    # opt-in nudges and prompt blocks default OFF.
    shared_lessons_enabled: bool = True
    tried_admission_filter: bool = True
    patch_summary_enabled: bool = True
    claims_enabled: bool = True

    # Opt-in PATCH_SUMMARY mechanics
    patch_ready_checkpoint_enabled: bool = False
    peer_patch_checkpoint_enabled: bool = False
    auto_submit_on_done_handoff: bool = False
    implementer_fast_finish_prompt_enabled: bool = False
    peer_patch_protocol_enabled: bool = False
    patch_summary_timing_rule_enabled: bool = False
    patch_summary_finish_handoff_enabled: bool = False
    patch_summary_latest_wins_enabled: bool = False
    patch_summary_lifecycle_enabled: bool = False
    implementer_strategy_prompt_enabled: bool = False

    # Shared-lessons render windows + mode
    shared_lessons_implementer_window_tokens: int = 1000
    shared_lessons_planner_window_tokens: int = 2000
    shared_lessons_render_mode: str = "full"
                                                 # "full"                  = legacy selected notes
                                                 # "peer_digest"           = high-signal peer notes only
                                                 # "patch_digest"          = finalized peer PATCH_SUMMARY only
                                                 # "patch_fail_digest"     = patch_digest + peer FAIL
                                                 # "patch_evidence_digest" = PlannerAgent sees patch_digest; ImplementerAgent also sees peer FACT/OBSERVED
                                                 # "patch_selective_unfold" = PATCH_SUMMARY overview + recent FAIL details

    planner_max_tokens: Optional[int] = 1200     # cap on PlannerAgent completion; null in YAML disables

    # Opt-in prompt caching for OpenRouter Claude routes
    prompt_cache_enabled: bool = False
    prompt_cache_min_prefix_chars: int = 16000

    # How the implementer sees the shared board (prompt assembly only):
    # "legacy_full"   windowed re-render every step (default, unchanged)
    # "dispatch_only" board rendered once at delegation start, then frozen
    # "every_step"    frozen dispatch snapshot plus an append-only
    #                 [BOARD UPDATE] section of entries admitted since
    refresh: str = "legacy_full"

    # Output
    result_folder: Path = field(default_factory=lambda: Path("workspace/logs"))
    trajectory_dir: Optional[Path] = None
    csv_summary_path: Optional[Path] = None
    timestamp: Optional[str] = None

    env_init: Optional[Dict[str, str]] = None
    cache_dir: Optional[str] = None              # HuggingFace cache directory
    window_size: int = 100                       # ACI file-view window size

    def __post_init__(self) -> None:
        """Warn on time-budget settings that can't physically fit.

        The per-task wall-clock cap must accommodate the final official grade
        (`grading_timeout`) plus some agent working time; otherwise a submit's
        eval gets guillotined by the cap and a possibly-correct fix is recorded
        as a failure. `grading_reserve_seconds` is the slice the solver keeps
        back for that grade, so it must be < the cap too.
        """
        cap = self.task_max_wall_clock_seconds
        if cap and self.grading_timeout >= cap:
            self._warn(
                f"grading_timeout={self.grading_timeout}s >= "
                f"task_max_wall_clock_seconds={cap}s: a submit's grading can be "
                f"cancelled by the task cap. Lower grading_timeout or raise the cap."
            )
        if cap and self.grading_reserve_seconds >= cap:
            self._warn(
                f"grading_reserve_seconds={self.grading_reserve_seconds}s >= "
                f"task_max_wall_clock_seconds={cap}s: the solver would self-stop "
                f"immediately, leaving no time for agent work."
            )
        if self.note_verifier_mode not in ("deterministic", "llm"):
            self._warn(
                f"note_verifier_mode={self.note_verifier_mode!r} is not "
                "'deterministic' or 'llm'; the runner falls back to deterministic."
            )
        if self.refresh not in ("legacy_full", "dispatch_only", "every_step"):
            self._warn(
                f"refresh={self.refresh!r} unknown; falling back to legacy_full."
            )
            self.refresh = "legacy_full"
        if self.refresh == "every_step" and self.patch_summary_latest_wins_enabled:
            self._warn(
                "refresh=every_step requires an append-only board; "
                "patch_summary_latest_wins_enabled=True replaces entries in "
                "place and can make the delta ledger skip or repeat entries. "
                "Disabling patch_summary_latest_wins_enabled."
            )
            self.patch_summary_latest_wins_enabled = False

    @staticmethod
    def _warn(msg: str) -> None:
        try:
            from base.engine.logs import logger
            logger.warning(f"[SWEBenchOrchestraConfig] {msg}")
        except Exception:
            pass

    @classmethod
    def load(cls, config_path: Path | str) -> "SWEBenchOrchestraConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        planner_model = raw.get("planner_model")
        if not planner_model:
            raise ValueError("planner_model is required")

        implementer_models = raw.get("implementer_models")
        if not implementer_models or not isinstance(implementer_models, list):
            raise ValueError("implementer_models must be a non-empty list")

        dataset_name = raw.get("dataset_name", "princeton-nlp/SWE-bench_Verified")
        split = raw.get("split", "test")

        subset_seed = raw.get("subset_seed")
        if subset_seed is not None:
            subset_seed = int(subset_seed)

        subset_sizes = raw.get("subset_sizes")
        if subset_sizes:
            subset_sizes = {
                str(k): int(v)
                for k, v in subset_sizes.items()
                if v is not None
            }

        subset_role = raw.get("subset_role")

        result_folder = cls._resolve_path(
            raw.get("result_folder", "workspace/logs"),
            config_path,
        )

        trajectory_dir = raw.get("trajectory_dir")
        if trajectory_dir:
            trajectory_dir = cls._resolve_path(trajectory_dir, config_path)

        csv_summary_path = raw.get("csv_summary_path")
        if csv_summary_path:
            csv_summary_path = cls._resolve_path(csv_summary_path, config_path)

        selected_ids_file = raw.get("selected_ids_file")
        if selected_ids_file:
            selected_ids_file = cls._resolve_path(selected_ids_file, config_path)

        return cls(
            planner_model=str(planner_model),
            implementer_models=[str(m) for m in implementer_models],
            dataset_name=str(dataset_name),
            split=str(split),
            subset_seed=subset_seed,
            subset_sizes=subset_sizes,
            subset_role=subset_role,
            selected_ids_file=selected_ids_file,
            max_tasks=raw.get("max_tasks"),
            max_steps=int(raw.get("max_steps", 50)),
            max_attempts=int(raw.get("max_attempts", 6)),
            max_concurrency=int(raw.get("max_concurrency", 1)),
            docker_timeout=int(raw.get("docker_timeout", 300)),
            step_timeout=int(raw.get("step_timeout", 120)),
            grading_timeout=int(raw.get("grading_timeout", 300)),
            task_max_wall_clock_seconds=int(raw.get("task_max_wall_clock_seconds", 480)),
            grading_reserve_seconds=int(raw.get("grading_reserve_seconds", 120)),
            task_token_budget=int(raw.get("task_token_budget", 1_000_000)),
            implementer_model=raw.get("implementer_model"),
            n_solvers=int(raw.get("n_solvers", 2)),
            solver_start_stagger_seconds=float(raw.get("solver_start_stagger_seconds", 0.0)),
            implementer_max_memory=int(raw.get("implementer_max_memory", 8)),
            implementer_max_obs_chars=int(raw.get("implementer_max_obs_chars", 2000)),
            implementer_max_thinking_chars=int(raw.get("implementer_max_thinking_chars", 500)),
            # `summarizer_model: null` in YAML → memory falls back to implementer_model.
            # Omitting the key entirely → gemini-flash default.
            summarizer_model=raw.get("summarizer_model", "google/gemini-3-flash-preview"),
            memory_compaction_mode=str(raw.get("memory_compaction_mode", "llm")).lower(),
            note_verifier_mode=str(raw.get("note_verifier_mode", "deterministic")).lower(),
            shared_lessons_enabled=bool(raw.get("shared_lessons_enabled", True)),
            tried_admission_filter=bool(raw.get("tried_admission_filter", True)),
            patch_summary_enabled=bool(raw.get("patch_summary_enabled", True)),
            claims_enabled=bool(raw.get("claims_enabled", True)),
            patch_ready_checkpoint_enabled=bool(raw.get("patch_ready_checkpoint_enabled", False)),
            peer_patch_checkpoint_enabled=bool(raw.get("peer_patch_checkpoint_enabled", False)),
            auto_submit_on_done_handoff=bool(raw.get("auto_submit_on_done_handoff", False)),
            implementer_fast_finish_prompt_enabled=bool(raw.get("implementer_fast_finish_prompt_enabled", False)),
            peer_patch_protocol_enabled=bool(raw.get("peer_patch_protocol_enabled", False)),
            patch_summary_timing_rule_enabled=bool(raw.get("patch_summary_timing_rule_enabled", False)),
            patch_summary_finish_handoff_enabled=bool(raw.get("patch_summary_finish_handoff_enabled", False)),
            patch_summary_latest_wins_enabled=bool(raw.get("patch_summary_latest_wins_enabled", False)),
            patch_summary_lifecycle_enabled=bool(raw.get("patch_summary_lifecycle_enabled", False)),
            implementer_strategy_prompt_enabled=bool(raw.get("implementer_strategy_prompt_enabled", False)),
            shared_lessons_implementer_window_tokens=int(raw.get("shared_lessons_implementer_window_tokens", 1000)),
            shared_lessons_planner_window_tokens=int(raw.get("shared_lessons_planner_window_tokens", 2000)),
            shared_lessons_render_mode=str(raw.get("shared_lessons_render_mode", "full")).lower(),
            planner_max_tokens=(
                int(raw["planner_max_tokens"])
                if "planner_max_tokens" in raw and raw["planner_max_tokens"] is not None
                else (None if "planner_max_tokens" in raw else 1200)
            ),
            prompt_cache_enabled=bool(raw.get("prompt_cache_enabled", False)),
            prompt_cache_min_prefix_chars=int(raw.get("prompt_cache_min_prefix_chars", 16000)),
            refresh=str(raw.get("refresh", "legacy_full")).lower(),
            result_folder=result_folder,
            trajectory_dir=trajectory_dir,
            csv_summary_path=csv_summary_path,
            env_init=raw.get("env_init"),
            cache_dir=raw.get("cache_dir"),
            window_size=int(raw.get("window_size", 100)),
        )

    @staticmethod
    def _resolve_path(path_str: str, config_path: Path) -> Path:
        """Resolve path relative to config file or project root."""
        path = Path(path_str)
        if path.is_absolute():
            return path
        rel_to_config = config_path.parent / path
        if rel_to_config.exists():
            return rel_to_config.resolve()
        PROJECT_ROOT = Path(__file__).parent.parent
        return (PROJECT_ROOT / path).resolve()
