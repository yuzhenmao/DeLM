"""
SWE-bench concrete ModeSpec instances.
"""
from __future__ import annotations

from src.common.utils import parse_json_response
from src.modes import ModeSpec
from src.prompts.swebench_global import SWEBenchPlannerPrompt

SWEBENCH_GLOBAL_MODE = ModeSpec(
    name="global",
    prompt_builder=SWEBenchPlannerPrompt,
    parser=parse_json_response,
    allowed_actions=("delegate_task", "submit"),
    default_max_tokens=None,
)
