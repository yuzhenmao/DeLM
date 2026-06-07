"""
ModeSpec: declarative description of an agent's action surface for one mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

@dataclass(frozen=True)
class ModeSpec:
    """Declarative description of an agent mode.

    Fields:
        name: short identifier for logs/diagnostics (e.g. "global", "local").
        prompt_builder: the class/callable used to construct the system prompt.
            Typed loosely (Any) to avoid coupling the generic mode module to
            any one prompt class; concrete instances are validated by their
            host benchmark module.
        parser: callable mapping a raw LLM response string to a decision dict
            with at least an `"action"` key. `decision["action"]`
            and checks it against `allowed_actions`.
        allowed_actions: closed allowlist of action names this mode may emit.
            An action_name not in this tuple is rejected at step time without
            tool lookup. Use a tuple (not a list) so the spec is immutable.
        default_max_tokens: optional completion-length cap for this mode.
            None means uncapped (host's existing default applies). Useful for
            weak models that otherwise run away into fake tool-call transcripts
            in a mode they shouldn't have access to.
    """

    name: str
    prompt_builder: Any
    parser: Callable[[str], dict]
    allowed_actions: Tuple[str, ...]
    default_max_tokens: Optional[int] = None
