"""Prefix byte-identity tests for the board refresh modes.

Asserts, for build_implementer_prompt_step_refresh:
  1. The static prefix (everything before "==== Runtime State ====") is
     byte-identical across steps within a delegation, in both modes, while
     memory, observation, step counter, and delta ledger all change.
  2. The every_step prompt is the dispatch_only prompt plus exactly two
     additive blocks: the board-update system paragraph (static prefix) and
     the [BOARD UPDATE] section (tail). Removing both restores dispatch_only
     byte-for-byte.
  3. DeltaLedger absorbs each peer entry exactly once and never absorbs the
     viewer's own entries.

Runnable via pytest or `python tests/test_prefix_identity.py`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts.swebench_local import build_implementer_prompt_step_refresh
from src.step_refresh import BOARD_UPDATE_SYSTEM_PARAGRAPH, DeltaLedger

RUNTIME_MARKER = "\n==== Runtime State ====\n"

BASE_KWARGS = dict(
    task_instruction="Fix the widget frobnicator (issue #123).",
    context="Attempt 2; prior attempt touched foo.py.",
    command_docs="open <file> | str_replace ... | pytest ...",
    dispatch_board_block=(
        "[t1/FACT +0.5m] frobnicate() lives in src/widget.py:88\n"
        "[t2/FAIL +0.9m] patching printer layer does not change output"
    ),
    thread_id=0,
)


def _build(step, memory, observation, board_update_section, every_step):
    return build_implementer_prompt_step_refresh(
        state_info=f"(Open file: foo.py) (Current directory: /testbed) step={step}",
        memory=memory,
        observation=observation,
        current_step=step,
        max_steps=50,
        board_update_section=board_update_section,
        board_updates_enabled=every_step,
        board_update_paragraph=BOARD_UPDATE_SYSTEM_PARAGRAPH,
        **BASE_KWARGS,
    )


def _prefix(prompt: str) -> str:
    idx = prompt.find(RUNTIME_MARKER)
    assert idx > 0, "Runtime State marker missing from prompt"
    return prompt[:idx]


def _make_ledger():
    return DeltaLedger(
        task_id="test__task-1", thread_id=0, delegation_id=1,
        dispatch_hwm=2, board_start_time=0.0,
    )


def test_prefix_stable_across_steps_both_modes():
    for every_step in (False, True):
        ledger = _make_ledger()
        prefixes = set()
        for step in range(1, 8):
            if every_step and step == 3:
                ledger.absorb(4, [
                    {"thread_id": 1, "type": "FACT", "content": "new finding A", "ts": 60.0},
                    {"thread_id": 2, "type": "FAIL", "content": "dead end B", "ts": 65.0},
                ], arrival_step=step)
            if every_step and step == 5:
                ledger.absorb(5, [
                    {"thread_id": 1, "type": "PATCH_SUMMARY", "content": "files=x | idea=y | evidence=z", "ts": 120.0},
                ], arrival_step=step)
            prompt = _build(
                step=step,
                memory=f"memory blob changes at step {step} " * step,
                observation=f"observation for step {step}",
                board_update_section=(ledger.render_section() if every_step else ""),
                every_step=every_step,
            )
            prefixes.add(_prefix(prompt))
        assert len(prefixes) == 1, (
            f"static prefix changed across steps (every_step={every_step}): "
            f"{len(prefixes)} distinct prefixes"
        )


def test_every_step_is_strictly_additive():
    ledger = _make_ledger()
    ledger.absorb(3, [
        {"thread_id": 2, "type": "FACT", "content": "peer found the bug at bar.py:12", "ts": 30.0},
    ], arrival_step=2)
    section = ledger.render_section()
    assert section.startswith("\n==== BOARD UPDATE")

    common = dict(step=4, memory="mem", observation="obs")
    base = _build(board_update_section="", every_step=False, **common)
    full = _build(board_update_section=section, every_step=True, **common)

    stripped = full.replace(section, "", 1)
    stripped = stripped.replace(f"\n{BOARD_UPDATE_SYSTEM_PARAGRAPH}\n", "", 1)
    assert stripped == base, "every_step prompt is not dispatch_only + additive blocks"

    # the delta section must sit after the observation (tail position)
    assert full.rfind("==== BOARD UPDATE") > full.rfind("==== Current Observation ====")
    # and before the final action request
    assert full.rfind("==== BOARD UPDATE") < full.rfind("Remember: output exactly")


def test_ledger_once_only_and_self_excluded():
    ledger = _make_ledger()
    n = ledger.absorb(5, [
        {"thread_id": 0, "type": "FACT", "content": "own note", "ts": 10.0},   # self
        {"thread_id": 1, "type": "FACT", "content": "peer note 1", "ts": 11.0},
        {"thread_id": 2, "type": "TRIED", "content": "peer note 2", "ts": 12.0},
    ], arrival_step=2)
    assert n == 2, "self entry must be excluded"
    assert ledger.hwm == 5
    # no new entries: nothing added, section unchanged
    before = ledger.render_section()
    assert ledger.absorb(5, [], arrival_step=3) == 0
    assert ledger.render_section() == before
    joined = ledger.render_section()
    assert joined.count("peer note 1") == 1
    assert "own note" not in joined


if __name__ == "__main__":
    test_prefix_stable_across_steps_both_modes()
    test_every_step_is_strictly_additive()
    test_ledger_once_only_and_self_excluded()
    print("OK: all prefix-identity tests passed")
