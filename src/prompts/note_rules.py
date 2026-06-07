_PATCH_SUMMARY_TIMING_RULES = """      Do NOT emit PATCH_SUMMARY on the same turn as an edit command. Only emit
      PATCH_SUMMARY after the Current Observation already contains the result of
      a verification command run after the source edit. If you are about to edit
      or about to run verification next, use FACT/TRIED/OBSERVED/CLAIM instead.
"""

def build_note_rules_block(patch_summary_timing_rule_enabled: bool = True) -> str:
    timing_rules = _PATCH_SUMMARY_TIMING_RULES if patch_summary_timing_rule_enabled else ""
    return f"""NOTE writes short typed entries to the SHARED LESSONS blackboard so other
  parallel threads can see your findings. Each entry is one line: <TYPE> <content>
  TYPE must be EXACTLY one of: FACT | TRIED | OBSERVED | FAIL | CLAIM | PATCH_SUMMARY
    - FACT: objective discovery (file path, function signature, error text)
    - TRIED: action you just took (one-line summary)
    - OBSERVED: empirical result (test output, exit code)
    - FAIL: a failed attempt with brief reason (so other threads don't repeat it)
    - CLAIM: your CURRENT TARGET / hypothesis before expensive work (file:line,
      test name, fix idea). SPARSE by design: emit at most ONE CLAIM when starting a distinct hypothesis, 
      and a SECOND only after a real pivot (not the same target re-stated). DO NOT
      emit CLAIM for routine execution, follow-up commands, test reruns, or
      restating a target you already claimed. Prefer `(none)` over a CLAIM that
      restates ongoing work. Per-delegation hard cap: 2 CLAIMs total.
    - PATCH_SUMMARY: a concise summary of a CANDIDATE FIX you have actually
      applied (not a plan). Use a structured 4-field schema separated by ` | `:
        files=<comma-separated paths> | idea=<one-sentence patch idea> |
        evidence=<what verified it works> | risk=<known regression risk>
      Emit ONCE per delegation, after the patch is in place and you've checked
      it (reproduction passes, no obvious regressions). Up to ~300 chars total.
{timing_rules.rstrip()}
      PlannerAgent and peer ImplementerAgents use these to decide which candidate to
      submit — be specific and honest about evidence + risk.
      REQUIRED finish trigger: before issuing `finish done`, emit one
      PATCH_SUMMARY in the same NOTE block summarizing the applied fix,
      evidence, and risk. This is the salient handoff signal.
  Publish only evidence — let other threads draw their own conclusions.
  Content size: FACT/TRIED/OBSERVED/FAIL/CLAIM ≤100 chars; PATCH_SUMMARY
  ≤300 chars (needed for the 4-field schema). Write 0-3 entries per turn.
  If you genuinely have nothing worth sharing this turn, write the literal
  `(none)` as the body.
  Examples:
      NOTE
      FACT app/parsers/options.py:88 drops blank label fallback
      TRIED added blank-label fallback near line 91

      NOTE
      OBSERVED demo_checks.py::test_blank_label FAILED with ValueError

      NOTE
      CLAIM target: app/formatters/labels.py:42 normalize_label fallback

      NOTE
      PATCH_SUMMARY files=app/formatters/labels.py | idea=preserve blank labels before formatting | evidence=ran demo_checks.py::test_blank_label and it PASSED | risk=may preserve whitespace-only labels

      NOTE
      (none)"""

NOTE_RULES_BLOCK = build_note_rules_block(False)
