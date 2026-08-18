"""Board refresh modes for the implementer prompt.

Prompt assembly only. No admission, verification, or queue logic here.

Modes (config `refresh`):
  legacy_full   windowed board re-render every step (default, unchanged)
  dispatch_only board rendered once at delegation start, then frozen for the
                delegation's lifetime
  every_step    frozen dispatch snapshot plus an append-only [BOARD UPDATE]
                section listing entries admitted after the dispatch
                high-water mark, excluding the viewer thread's own entries

The every_step layout keeps everything before the Runtime State marker
byte-stable within a delegation, so provider-side prompt caching still sees a
constant prefix while updates accumulate at the tail. Each update is shown
exactly once and stays in the ledger for the rest of the delegation, tagged
with the step it arrived at.
"""
from __future__ import annotations

from typing import Any, Dict, List

REFRESH_MODES = ("legacy_full", "dispatch_only", "every_step")


class DeltaLedger:
    """Per-delegation accumulator of board entries admitted after dispatch.

    Relies on the board being append-only (see SharedLessons.entries_since):
    `hwm` is an index into the entry list, and entries below it are assumed
    never to move.
    """

    def __init__(self, task_id: str, thread_id: int, delegation_id: int,
                 dispatch_hwm: int, board_start_time: float):
        self.task_id = task_id
        self.thread_id = int(thread_id)
        self.delegation_id = int(delegation_id)
        self.hwm = int(dispatch_hwm)
        self._board_start = float(board_start_time)
        self.lines: List[str] = []
        self.n_entries = 0

    def absorb(self, board_len: int, new_entries: List[Dict[str, Any]],
               arrival_step: int) -> int:
        """Add peer entries [hwm..board_len) to the ledger. Returns count added."""
        added = 0
        for e in new_entries:
            if int(e.get("thread_id", -1)) == self.thread_id:
                continue  # own entries are already in this agent's memory/NOTE history
            rel_min = (float(e.get("ts", 0.0)) - self._board_start) / 60.0
            self.lines.append(
                f"[arrived step {arrival_step}] "
                f"[t{e.get('thread_id')}/{e.get('type')} +{rel_min:.1f}m] "
                f"{e.get('content', '')}"
            )
            added += 1
        self.hwm = int(board_len)
        self.n_entries += added
        return added

    def render_section(self) -> str:
        if not self.lines:
            return ""
        return (
            "\n==== BOARD UPDATE (peer entries admitted since your dispatch snapshot) ====\n"
            + "\n".join(self.lines)
            + "\n"
        )


BOARD_UPDATE_SYSTEM_PARAGRAPH = (
    "BOARD UPDATES: While you work, new entries admitted to the shared board by "
    "peer threads are appended at the end of your prompt in a [BOARD UPDATE] "
    "section. Treat these deltas as authoritative shared state: drop any "
    "hypothesis of yours that a delta FAIL entry contradicts, and if a delta "
    "already completes your claimed task (e.g. a verified peer fix covering the "
    "same defect), finish early citing that entry instead of re-deriving it."
)
