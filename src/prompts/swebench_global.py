# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

"""SWE-bench global (PlannerAgent / orchestrator) prompt + helpers."""
from typing import Any, Dict, List

_EMPTY_SHARED_LESSONS_BLOCKS = {
    "(no shared lessons yet)",
    "(no peer shared lessons yet)",
}

def _has_real_shared_lessons(block: str) -> bool:
    text = (block or "").strip()
    return bool(text and text not in _EMPTY_SHARED_LESSONS_BLOCKS)

_DIVERSIFY_PREFIXES = {
    0: (
        "STRATEGY (thread t0 — minimal-repro-first): Begin by constructing the "
        "smallest possible input that triggers the bug. Strip away framework "
        "scaffolding and indirection to isolate the failing primitive, then trace "
        "upward from that minimal case to identify where the source code diverges "
        "from the expected behavior."
    ),
    1: (
        "STRATEGY (thread t1 — code-first): Begin by examining the files, symbols, "
        "or modules mentioned in the issue and their immediate callers/callees. "
        "Map the affected code surface before proposing any edit."
    ),
    2: (
        "STRATEGY (thread t2 — spec-first): Begin by carefully reading the issue "
        "description to extract the expected behavior, observed behavior, relevant "
        "API contracts, and any hints or proposed fixes. Before editing, state the "
        "minimal behavioral change the patch must implement."
    ),
    3: (
        "STRATEGY (thread t3 — test-first): Begin by reproducing the reported "
        "failure when possible. Run the failing test or the most relevant test "
        "from the issue text, inspect the full error output, and use it to localize "
        "the root cause before editing."
    ),
}

def diversify_prompt(thread_id: int) -> str:
    """Return the diversification prefix for this thread, or '' if not configured."""
    return _DIVERSIFY_PREFIXES.get(int(thread_id), "")

# SWE-bench ImplementerAgent uses ACI commands, not explicit tool objects
SWEBENCH_TOOLS_DESCRIPTION = """
ImplementerAgent uses ACI (Agentic Code Interface) commands to navigate and edit code.

=== FILE VIEWING ===
- open <file> [line]: Open file at specified line
- scroll_down/scroll_up: Navigate within open file
- goto <line>: Jump to specific line

=== FILE EDITING ===
- str_replace <file>: Replace exact text match (preferred for edits)
- edit <start>:<end>: Edit line range with syntax check
- insert <file> <line>: Insert content after line
- create <file>: Create new file

=== SEARCHING ===
- search_file <pattern> [file]: Search content in file
- search_dir <pattern> [dir]: Search in directory
- find_file <name> [dir]: Find files by name

=== BASH ===
- Any bash command (ls, cat, grep, python, pytest, etc.)

=== SUBMIT ===
- submit: Submit fix and run official tests
""".strip()

class SWEBenchPlannerPrompt:
    """Build prompts for SWE-bench tasks (GitHub issue fixing)."""

    @staticmethod
    def build_prompt(
        instruction: str,
        meta: Dict[str, Any],
        attempt_index: int,
        max_attempts: int,
        implementer_models: List[str],
        subtask_history: str = "",
        model_to_alias: Dict[str, str] = None,
        thread_id: int = 0,
        n_threads: int = 1,
        shared_lessons_block: str = "",
        diversify_prefix: str = "",
    ) -> str:
        remaining_attempts = max_attempts - attempt_index + 1

        # Extract repository info from metadata
        repo = meta.get("repo", "unknown")
        instance_id = meta.get("instance_id", "unknown")

        # Budget warning
        if remaining_attempts <= 2:
            budget_warning = f"🚨 CRITICAL: Only {remaining_attempts} attempt(s) left! Submit now if fix is verified, or make final attempt."
        elif remaining_attempts <= 4:
            budget_warning = f"⚠️ Warning: {remaining_attempts} attempts remaining. Plan carefully."
        else:
            budget_warning = ""

        # SHARED LESSONS section. Empty string when sharing is
        # disabled (n_threads=1) or no entries yet.
        lessons_section = ""
        if _has_real_shared_lessons(shared_lessons_block):
            lessons_guidance = (
                "Consult these lessons as compact background context. The optional "
                "[Patch summaries] block at the top lists CANDIDATE FIXES proposed "
                "by completed ImplementerAgent delegations — when deciding whether to submit "
                "or to launch another delegation, weigh each summary's evidence and "
                "risk fields. If a peer summary covers a credible fix with passing "
                "evidence, prefer to verify and submit rather than re-derive. "
                "Avoid repeating the same mistake/reason from peer FAIL entries, but do "
                "not abandon a named target solely because a peer's execution failed "
                "there. FACT entries are objective findings; OBSERVED entries are "
                "empirical results; TRIED entries are actions taken. The optional "
                "[Active claims] block shows peer threads' current targets — when "
                "delegating, consider framing the task instruction to pursue a "
                "different angle if a peer claim covers your default plan.\n"
                "peer-FAIL directive: when delegating, if a peer FAIL entry "
                "describes a dead end on the file/test/symbol you're about to ask "
                "the ImplementerAgent to work on, the task_instruction MUST explicitly "
                "reference that failure (e.g. \"thread t1 hit X — try Y instead\" or "
                "\"avoid the approach that failed with TypeError\"). Stops the next "
                "delegation from rediscovering the same dead end.\n"
            )
            lessons_section = (
                "\n==== SHARED LESSONS (across all parallel threads on this task) ====\n"
                f"{shared_lessons_block}\n\n"
                f"{lessons_guidance}"
            )

        # diversification prefix on the task framing
        task_header = instruction
        if diversify_prefix:
            task_header = f"{diversify_prefix}\n\n{instruction}"

        # Identity header — make thread membership explicit when N>1
        if n_threads > 1:
            identity = (
                f"You are the PlannerAgent for THREAD t{thread_id} of {n_threads} "
                f"parallel solvers on this SWE-bench task. Each thread has its own "
                f"container and ImplementerAgent runner; threads share information ONLY via "
                f"the SHARED LESSONS blackboard below.\n\n"
                "Your goal: fix this GitHub issue by delegating work to ImplementerAgents."
            )
        else:
            identity = "You are the PlannerAgent (Orchestrator) for a SWE-bench task. Your goal is to fix a GitHub issue by delegating work to ImplementerAgents."

        return f"""{identity}

==== TASK ====
{task_header}

REPOSITORY: {repo}
INSTANCE: {instance_id}
{lessons_section}
==== DECISION PROCESS ====
1. READ the TASK carefully - understand the GitHub issue and what needs to be fixed
2. REVIEW SUBTASK HISTORY - check ImplementerAgent's progress, completed steps, and test results
3. VERIFY against TASK requirements:
   - Did ImplementerAgent locate the buggy code?
   - Did ImplementerAgent make appropriate code changes?
   - Did ImplementerAgent run tests and confirm the fix works?
4. DECIDE:
   - ✅ status="done" AND the source fix is verified (by a targeted repro or relevant command) → Use 'submit'
   - ✅ status="done" but some EXISTING local tests still fail ONLY because they assert the OLD behavior (stale expected-output) → still Use 'submit' — the official grader supplies the correct tests; do NOT keep delegating to "fix" existing tests
   - ⚠️ status="done" BUT the source fix is unverified or genuinely broken → Use 'delegate_task' to fix remaining issues
   - ⚠️ status="partial" → Use 'delegate_task' with guidance on next steps

CRITICAL: SWE-BENCH CONTAINER BEHAVIOR
- When the source fix is verified, use 'submit' to trigger final evaluation — do NOT chase passing every existing local test by editing them; the grader checks out test files and applies the official test_patch.
- 'submit' runs the official test suite (FAIL_TO_PASS + PASS_TO_PASS tests) to determine success

==== Progress ====
[Attempt {attempt_index}/{max_attempts}] Remaining {remaining_attempts} attempts
{budget_warning}

==== SUBTASK HISTORY ====
{subtask_history if subtask_history else "No subtasks completed yet."}

==== YOUR ACTIONS (you are an ORCHESTRATOR — you do NOT run code yourself) ====
You have EXACTLY TWO actions: "delegate_task" and "submit".
- delegate_task: hand a coding subtask to a ImplementerAgent — it navigates/edits/runs code and reports back.
- submit: submit the current repository state for official evaluation.

STRICT OUTPUT CONTRACT (follow exactly):
- Do NOT explore or edit the repo yourself, and do NOT run any commands.
- Do NOT emit `<function_calls>`, tool-call XML, markdown, bash, or ACI commands. Commands like open/str_replace/search_dir/edit/bash are ImplementerAgent-only — you cannot use them.
- Return EXACTLY ONE JSON object and NOTHING else; the very first character of your response must be an opening curly brace.
- "action" MUST be exactly "delegate_task" or "submit".

==== OUTPUT ====
Return JSON:

If ImplementerAgent status="done" AND the SOURCE fix is verified (by a targeted repro or relevant command) — even if some existing local tests still FAIL only because they assert the OLD behavior:
{{
  "action": "submit",
  "reasoning": "Source fix verified by [repro/command]. Remaining local test failures (if any) are stale assertions of the old behavior; the grader supplies the official tests. Submitting.",
  "params": {{ "reason": "Source fix verified: [specific fix description]" }}
}}

If ImplementerAgent status="done" BUT the SOURCE fix is unverified or genuinely broken (syntax/runtime regression, wrong logic, or an UNRELATED real failure) — NOT merely existing tests asserting old behavior, and NOT to edit existing tests:
{{
  "action": "delegate_task",
  "reasoning": "ImplementerAgent reported done but the source fix is not verified: [specific real problem, e.g. repro still fails / regression]",
  "params": {{
    "task_instruction": "CRITICAL: source fix incomplete/unverified. [specific next steps]. Do NOT edit existing tests; fix the source and verify with a repro.",
    "context": "⚠️ ISSUE: [what real failure]\\n- ✅ DONE: [completed work]\\n- ❌ TODO: [remaining source work]"
  }}
}}

If ImplementerAgent status="partial":
{{
  "action": "delegate_task",
  "reasoning": "ImplementerAgent made partial progress: [summary]. Need to [next steps]",
  "params": {{
    "task_instruction": "Continue: [specific next steps based on SUBTASK HISTORY]",
    "context": "From previous attempt:\\n- ✅ WORKED: [keep these]\\n- ❌ FAILED: [avoid these]"
  }}
}}
""".strip()
