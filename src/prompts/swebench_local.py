# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

"""
SWE-bench LOCAL (ImplementerAgent) prompt + builder.
"""
from src.prompts.note_rules import NOTE_RULES_BLOCK, build_note_rules_block

_EMPTY_SHARED_LESSONS_BLOCKS = {
    "(no shared lessons yet)",
    "(no peer shared lessons yet)",
}

def _has_real_shared_lessons(block: str) -> bool:
    text = (block or "").strip()
    return bool(text and text not in _EMPTY_SHARED_LESSONS_BLOCKS)

def _has_peer_patch_summary(block: str) -> bool:
    return "/PATCH_SUMMARY" in (block or "")

# =============================================================================
# SWEBENCH IMPLEMENTER PROMPT
# =============================================================================
FAST_FINISH_RULE = "After you edit SOURCE files, stop broad exploration. The next useful step is a focused verification command; if it passes, emit PATCH_SUMMARY and `finish done` immediately. Do not keep re-reading code after a verified source edit unless the last observation showed a concrete failure."

SWEBENCH_IMPLEMENTER_PROMPT_TEMPLATE = """
You are an autonomous software engineering agent (THREAD t{thread_id}) tasked with solving GitHub issues.
You have access to a specialized command interface (ACI) for navigating, viewing, editing, and testing code.
You will work in a Docker container with the repository already cloned and checked out to the correct commit.

This task is being attempted by multiple parallel solver threads (you are one of them). Threads do NOT see
each other's code or container — only the SHARED LESSONS blackboard below. Cooperate via lessons.

==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}
If you run out of steps without "finish", an automatic partial handoff is recorded and your repo edits are preserved — but an explicit `finish partial ...` is strongly preferred, because you can summarize what you did, what's left, and any pitfalls far more usefully than the automatic handoff can.
{strategy_section}
==== Your Task (from PlannerAgent) ====
{task_instruction}

==== Context (from previous attempts) ====
{context}

{shared_lessons_section}

==== Current State ====
{state_info}

==== Command Reference ====
{command_docs}

When your fix adds a branch that serializes/parses/converts a value, guard it so an unexpected input falls back to the prior behavior instead of raising.

EXISTING TESTS ARE READ-ONLY. Make the minimal fix in SOURCE files only. Do NOT edit files under a `tests/` or `testing/` directory to make old tests match your change — the grader supplies the official tests and discards any test edits you make, so it never helps (and wastes your budget). You may READ and RUN existing tests freely. If an existing test fails only because it still asserts the OLD behavior, that is expected verification evidence — `finish done` once your source change is verified (write a temporary repro script OUTSIDE the test tree if you need to confirm behavior).

=== SUBMIT DISCIPLINE ===
Before `finish done`: run a reproduction script OUTSIDE tests OR a focused `pytest path/to/test_x.py::test_y`. Put the exact command+result in PATCH_SUMMARY.evidence (no "TBD"/"pending"/"should work"; the verifier drops those). For broad framework edits (base class, __init__, widely-imported module), run one adjacent NEGATIVE CHECK and name it in evidence. Prefer focused tests over broad suites. If a peer FAIL is related, explain in evidence/risk why your fix avoids that failure mode.
{fast_finish_rule}

=== FINISH (Report to PlannerAgent) ===
finish <status> <message>
    Report your progress back to PlannerAgent. Status MUST be one of:
    - done: fix verified by RUNNING it — on the issue's case AND one awkward input (empty/None/odd type) — and it returns, not raises
    - partial: Made progress but not finished (e.g., found bug but fix not working)

==== Memory ====
Recent memory:
{memory}

==== Current Observation ====
{observation}

==== OUTPUT FORMAT (STRICT) ====
You MUST output EXACTLY three sections in this order. No other text allowed.

DISCUSSION
<your reasoning here>

NOTE
<TYPE> <content>
[optional additional lines: <TYPE> <content>]

COMMAND
<single command here>

RULES:
- All three sections (DISCUSSION, NOTE, COMMAND) are REQUIRED. A response missing
  the NOTE header is a parse error and will be replayed.
- DISCUSSION must contain your step-by-step reasoning.
- __NOTE_RULES_SENTINEL__
- COMMAND must contain exactly ONE command on a single line. After the COMMAND
  line, do NOT add any explanation, examples, or comments. Do NOT output
  anything after the command.
"""

SWEBENCH_IMPLEMENTER_PROMPT_CACHE_STATIC_TEMPLATE = """
You are an autonomous software engineering agent (THREAD t{thread_id}) tasked with solving GitHub issues.
You have access to a specialized command interface (ACI) for navigating, viewing, editing, and testing code.
You will work in a Docker container with the repository already cloned and checked out to the correct commit.

This task is being attempted by multiple parallel solver threads (you are one of them). Threads do NOT see
each other's code or container — only the SHARED LESSONS blackboard below. Cooperate via lessons.
{strategy_section}
==== Your Task (from PlannerAgent) ====
{task_instruction}

==== Context (from previous attempts) ====
{context}

==== Command Reference ====
{command_docs}

When your fix adds a branch that serializes/parses/converts a value, guard it so an unexpected input falls back to the prior behavior instead of raising.

EXISTING TESTS ARE READ-ONLY. Make the minimal fix in SOURCE files only. Do NOT edit files under a `tests/` or `testing/` directory to make old tests match your change — the grader supplies the official tests and discards any test edits you make, so it never helps (and wastes your budget). You may READ and RUN existing tests freely. If an existing test fails only because it still asserts the OLD behavior, that is expected verification evidence — `finish done` once your source change is verified (write a temporary repro script OUTSIDE the test tree if you need to confirm behavior).

=== SUBMIT DISCIPLINE ===
Before `finish done`: run a reproduction script OUTSIDE tests OR a focused `pytest path/to/test_x.py::test_y`. Put the exact command+result in PATCH_SUMMARY.evidence (no "TBD"/"pending"/"should work"; the verifier drops those). For broad framework edits (base class, __init__, widely-imported module), run one adjacent NEGATIVE CHECK and name it in evidence. Prefer focused tests over broad suites. If a peer FAIL is related, explain in evidence/risk why your fix avoids that failure mode.
{fast_finish_rule}

=== FINISH (Report to PlannerAgent) ===
finish <status> <message>
    Report your progress back to PlannerAgent. Status MUST be one of:
    - done: fix verified by RUNNING it — on the issue's case AND one awkward input (empty/None/odd type) — and it returns, not raises
    - partial: Made progress but not finished (e.g., found bug but fix not working)

==== OUTPUT FORMAT (STRICT) ====
You MUST output EXACTLY three sections in this order. No other text allowed.

DISCUSSION
<your reasoning here>

NOTE
<TYPE> <content>
[optional additional lines: <TYPE> <content>]

COMMAND
<single command here>

RULES:
- All three sections (DISCUSSION, NOTE, COMMAND) are REQUIRED. A response missing
  the NOTE header is a parse error and will be replayed.
- DISCUSSION must contain your step-by-step reasoning.
- __NOTE_RULES_SENTINEL__
- COMMAND must contain exactly ONE command on a single line. After the COMMAND
  line, do NOT add any explanation, examples, or comments. Do NOT output
  anything after the command.

==== Runtime State ====
==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}
If you run out of steps without "finish", an automatic partial handoff is recorded and your repo edits are preserved — but an explicit `finish partial ...` is strongly preferred, because you can summarize what you did, what's left, and any pitfalls far more usefully than the automatic handoff can.

{shared_lessons_section}

==== Current State ====
{state_info}

==== Memory ====
Recent memory:
{memory}

==== Current Observation ====
{observation}

Remember: output exactly DISCUSSION, NOTE, COMMAND.
"""

SWEBENCH_IMPLEMENTER_PROMPT = SWEBENCH_IMPLEMENTER_PROMPT_TEMPLATE.replace(
    "__NOTE_RULES_SENTINEL__",
    NOTE_RULES_BLOCK,
).replace("{fast_finish_rule}", "").replace("{strategy_section}", "")
SWEBENCH_IMPLEMENTER_PROMPT_CACHE_STATIC = SWEBENCH_IMPLEMENTER_PROMPT_CACHE_STATIC_TEMPLATE.replace(
    "__NOTE_RULES_SENTINEL__",
    NOTE_RULES_BLOCK,
).replace("{fast_finish_rule}", "").replace("{strategy_section}", "")

SHARED_LESSONS_GUIDANCE = """Read these before each command:
- [Patch summaries] (if present) are PEER-PROPOSED CANDIDATE FIXES. Read them BEFORE committing to your own approach: if a peer's summary covers your idea and the evidence holds up, you can corroborate it (run their reproduction, check their target) rather than re-deriving it; if the evidence is thin or you see a different fix, propose your own with your own PATCH_SUMMARY. Do not blindly copy — verify before adopting.
- Avoid repeating the SAME mistake/reason named in peer FAIL entries, but do NOT abandon a correct file/target just because a peer's execution failed there (a tooling/execution error is not a logic dead end).
- Build on FACT entries (objective findings — file/line/error text).
- OBSERVED entries are empirical test/run results; TRIED entries are actions taken.
- [Active claims] shows peer threads' CURRENT TARGETS. If a peer is already working your planned target, pick a different angle (different file, different test, different hypothesis) so it doesn't duplicate work — but only if you can pursue a genuinely different lead. Don't abandon a strong target just to be different."""

PEER_PATCH_PROTOCOL = """Peer PATCH_SUMMARY protocol:
- If a peer patch summary covers your likely fix, first choose one concrete response: verify it, refine it minimally, or reject it with a specific failed command/result.
- Verification means run the peer's evidence command or an equally focused repro/check in your own container; do not trust the summary without executing a check.
- Prefer summaries marked final/verified with fewer local_failures; treat high local_failures or high risk as a warning to inspect the failure mode before adopting it.
- If you reject or refine it, publish the reason as FAIL or PATCH_SUMMARY.evidence/risk so the next peer does not repeat the same partial patch."""

def build_implementer_prompt(
    task_instruction: str,
    context: str,
    command_docs: str,
    state_info: str,
    memory: str,
    observation: str,
    current_step: int,
    max_steps: int,
    shared_lessons_block: str = "",
    thread_id: int = 0,
    cache_static_prefix_layout: bool = False,
    cache_static_prefix_min_chars: int = 16000,
    implementer_fast_finish_prompt_enabled: bool = False,
    peer_patch_protocol_enabled: bool = False,
    patch_summary_timing_rule_enabled: bool = False,
    strategy_prefix: str = "",
) -> str:
    """Build the complete prompt for ImplementerAgent."""
    remaining_steps = max_steps - current_step

    # Budget warning
    if remaining_steps <= 3:
        budget_warning = "🚨 CRITICAL: Only {} steps left! Use 'finish' NOW to report your progress!".format(remaining_steps)
    elif remaining_steps <= 5:
        budget_warning = "⚠️ Warning: {} steps remaining. Plan to finish soon.".format(remaining_steps)
    else:
        budget_warning = ""

    has_lessons = _has_real_shared_lessons(shared_lessons_block)
    if has_lessons:
        peer_patch_protocol = (
            f"\n{PEER_PATCH_PROTOCOL}\n"
            if peer_patch_protocol_enabled and _has_peer_patch_summary(shared_lessons_block)
            else ""
        )
        shared_lessons_section = (
            "\n==== SHARED LESSONS (peer threads' notes on this task) ====\n"
            f"{shared_lessons_block}\n\n"
            f"{SHARED_LESSONS_GUIDANCE}\n"
            f"{peer_patch_protocol}"
        )
    else:
        shared_lessons_section = ""

    if strategy_prefix:
        strategy_section = (
            "\n==== STRATEGY (this thread's framing) ====\n"
            f"{strategy_prefix}\n"
        )
    else:
        strategy_section = ""

    fmt = dict(
        task_instruction=task_instruction,
        context=context if context else "No additional context provided.",
        command_docs=command_docs,
        state_info=state_info,
        memory=memory,
        observation=observation,
        shared_lessons_section=shared_lessons_section,
        thread_id=thread_id,
        current_step=current_step,
        max_steps=max_steps,
        remaining_steps=remaining_steps,
        budget_warning=budget_warning,
        fast_finish_rule=(
            FAST_FINISH_RULE if implementer_fast_finish_prompt_enabled else ""
        ),
        strategy_section=strategy_section,
    )

    note_rules_block = build_note_rules_block(patch_summary_timing_rule_enabled)

    if cache_static_prefix_layout:
        candidate = SWEBENCH_IMPLEMENTER_PROMPT_CACHE_STATIC_TEMPLATE.replace(
            "__NOTE_RULES_SENTINEL__",
            note_rules_block,
        ).format(**fmt)
        idx = candidate.find("\n==== Runtime State ====\n")
        if idx >= int(cache_static_prefix_min_chars or 0):
            return candidate

    return SWEBENCH_IMPLEMENTER_PROMPT_TEMPLATE.replace(
        "__NOTE_RULES_SENTINEL__",
        note_rules_block,
    ).format(**fmt)
