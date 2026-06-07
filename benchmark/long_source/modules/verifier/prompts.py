"""
Prompt builders for the Verifier module.
"""
from typing import List


def build_verify_prompt(shared_context: str,
                                      candidates: List[str], label: str,
                                      task: str = "",
                                      cited_sources_block: str = "") -> str:
    """
    The `cited_sources_block` contains, for each [W<i> S<j>] label the
    CHAIN cites, BOTH the L1 summary AND the raw source slice (full
    chunk for chunk-level cites; ~2,500-char sub-chunk slice for
    sub-chunk cites of the form W<i> S<j>.<n>). This lets the verifier
    ground value-level claims against verbatim raw text (the L1 is
    structural; raw is authoritative for exact numbers, qualifiers,
    and row labels) while still using the L1 for context / topical
    framing.

    Output contract — one of APPROVED / WRONG, each preceded
    by a structured VERIFICATION block:

      VERIFICATION:
        U1 citation-support: PASS | FAIL: <step N: ...>
        U2 output-coherence: PASS | FAIL: <...>
        C1 arithmetic:       PASS | N/A | FAIL: <...>
        C2 qualifier-drift:  PASS | N/A | FAIL: <step N: ...>
        C3 editorial:        PASS | N/A | FAIL: <step N: ...>

      [one of:]
      APPROVED:
      <polished prose summary derived STRICTLY from the CHAIN>

      WRONG: <one-line specific error, for any U1/U2/C1/C2/C3 failure>
    """
    chain_block = candidates[0].strip() if candidates else ""
    task_block = f"Sub-task:\n{task.strip()}\n\n---\n\n" if task else ""
    cited_block = (
        f"Cited source material (for each `[W<i> S<j>]` or `[W<i> S<j>.<n>]`\n"
        f"the CHAIN cites: L1 summary + raw source text). The raw text is\n"
        f"authoritative for exact values, row labels, and qualifiers; the L1\n"
        f"is structural context. When the CHAIN claims a specific number or\n"
        f"row qualifier, ground the check against the RAW block below.\n\n"
        f"{cited_sources_block}\n\n---\n\n"
    )
    return f"""{shared_context}

---
TASK:
{task_block}

---
STRUCTURED REASONING TO VERIFY
Candidate summary for [{label}]:

{chain_block}

---
CITED SOURCES:
{cited_block}

---
JUDGE INSTRUCTIONS

You are a verification judge assessing a worker's reasoning chain for factual accuracy, citation support, output coherence, arithmetic correctness, qualifier drift, and logical soundness.

Run each VERIFICATION AXIS below, then emit the required output format. After the VERIFICATION section, emit exactly ONE verdict section: APPROVED or WRONG.

VERIFICATION AXES

U1 citation-support (semantic):
  For each [W<i> S<j>] or [W<i> S<j>.<n>] label the CHAIN cites, the cited source above must concretely support the CHAIN step's claim.

  U1 failure means:
  - the cited source is about a different entity, document, topic, row, period, or qualifier than the CHAIN claim;
  - the claim is not present in the cited source at all;
  - the CHAIN attributes a value to the wrong row label, aggregation level, period, entity, or qualifier.

  For value claims involving specific numbers, dates, amounts, percentages, ratios, periods, or counts:
  - Ground against the RAW block, not the L1 summary.
  - The L1 may omit or approximate a value that the worker correctly extracted from raw.
  - If the raw block contains the claimed value under the claimed row label / period / qualifier, U1 PASSES even if the L1 does not mention it.
  - If the cited label has NO raw source text present in CITED SOURCES (only an L1 summary, or nothing), the value claim is UNVERIFIABLE: FAIL U1 and state "unverifiable: no raw for [W<i> S<j>.<n>]". Do NOT pass a specific-value claim merely because nothing visibly contradicts it — absence of raw grounding is a failure, not a pass.

U2 output-coherence:
  The OUTPUT line or candidate summary must follow from the CHAIN steps.

  U2 failure means:
  - the OUTPUT introduces an entity, value, date, qualifier, comparison, conclusion, or claim not present in the CHAIN;
  - the OUTPUT strengthens, reverses, or changes the CHAIN's conclusion;
  - the OUTPUT omits a necessary qualifier from the CHAIN in a way that changes the meaning.

C1 arithmetic:
  Apply whenever the CHAIN states a DERIVED number — a difference, sum, ratio, percentage, growth rate, any "= <number>" — OR a numeric comparison ("X is larger than Y", "increased by ...").

  - If the CHAIN shows its computation, recompute it independently. If your computation disagrees with the CHAIN, FAIL C1 and state the step number, the CHAIN's computed value, and the correct value.
  - If the CHAIN states a derived value or comparison WITHOUT showing the arithmetic, reconstruct it yourself from the cited raw values and check it. You may NOT mark a derived or compared value as N/A just because the steps were not shown — a hidden computation is exactly what this axis exists to catch.
  - If you cannot reproduce the derived value from the cited sources, FAIL C1 and state the step, the claimed value, and why it cannot be reproduced.

  Mark C1 = N/A ONLY when every number in the CHAIN is quoted directly from a cited source with no derivation and no comparison (those are graded by U1 instead).

C2 qualifier-drift:
  Apply when a CHAIN step attributes emphasis, intent, priority, significance, causality, or interpretive framing to a source.

  Trigger phrases include, but are not limited to:
  "emphasises," "focuses on," "treats as a priority," "signals," "represents a shift," "highlights," "is driven by," "is mainly due to," "shows that," "demonstrates that," or similar language.

  For each triggered step, check whether the cited source actually uses equivalent wording or directly supports that level of strength.

  C2 failure means:
  - the source merely mentions, lists, or describes something, but the CHAIN says the source emphasises, prioritizes, signals, or highlights it;
  - the source states an association, but the CHAIN turns it into causation;
  - the source gives a limited qualifier, but the CHAIN broadens it.

  If C2 fails, state the step number and the source's weaker actual phrasing.

C3 editorial / unsupported synthesis:
  Apply when a CHAIN step makes an overall synthesis, interpretation, or document-level claim.

  C3 failure means:
  - a synthesis step asserts what a document, act, regulation, table, policy, or source means/does without a direct citation; and
  - the synthesis is not strictly entailed by earlier cited CHAIN steps.

  C3 does NOT fail a synthesis step that is a purely mechanical combination of earlier cited steps, as long as it introduces no new document-level claim, qualifier, causal explanation, or interpretation.

DECISION RULE

- Any U1 FAIL, U2 FAIL, C1 FAIL, C2 FAIL, or C3 FAIL → WRONG.
- All applicable checks PASS → APPROVED.

For APPROVED:
  Emit a clean prose summary of the CHAIN. Use concrete facts only. Avoid editorial framing such as "emphasises," "reflects," "signals," "represents," or "highlights" unless that wording is directly supported by the cited source or explicitly derived in the CHAIN.

For WRONG:
  Emit a one-line specific error. Cite the failing step number and the conflicting value, missing support, wrong row label, wrong qualifier, incoherent output claim, or arithmetic error.

Citation preservation:
  In the APPROVED summary, preserve the source labels needed to ground concrete claims, using the same [W<i> S<j>] or [W<i> S<j>.<n>] labels from the CHAIN. Do not add new citations that were not cited in the CHAIN.

OUTPUT FORMAT

Emit EXACTLY these sections in this order:

VERIFICATION:
  U1 citation-support: <PASS | FAIL: ...>
  U2 output-coherence: <PASS | FAIL: ...>
  C1 arithmetic:       <PASS | N/A | FAIL: ...>
  C2 qualifier-drift:  <PASS | N/A | FAIL: ...>
  C3 editorial:        <PASS | N/A | FAIL: ...>

Then emit exactly one of:

APPROVED:
<one-paragraph clean prose summary with necessary source labels preserved>

WRONG: <one-line specific error citing the step number and the conflicting values / missing support / wrong qualifier / arithmetic error>

[{label}] VERIFY:"""


def build_gate_and_gist_prompt(task: str, reasoning: str, label: str) -> str:
    """First the model gates on whether the block has a recoverable,
    citation-grounded finding. If it has none, it outputs a
    `NO_RECOVERABLE_FINDING: ...` line and stops. Otherwise it produces the
    `**[label] GIST**` block.

    Caller distinguishes outcomes by checking whether the response starts
    with `NO_RECOVERABLE_FINDING`.
    """
    return f"""You process a worker's citation-verified reasoning block in two steps.

STEP 1 — GATE: IS THERE A RECOVERABLE FINDING?

This gate checks only whether the reasoning block contains a recoverable, citation-grounded finding that STEP 2 can turn into a self-contained gist.

This gate is NOT a self-containment check. STEP 2 will restate the sub-task and make the gist self-contained using the assigned sub-task below.

Fail the gate ONLY when the block has nothing usable for STEP 2 to gist, such as when it:
- gives only a bare option letter, answer choice, or unsupported verdict, with no sourced factual finding;
- is empty, circular, or a stub with no recoverable finding;
- lacks the `[W<i> S<j>]` or `[W<i> S<j>.<n>]` citation labels needed to ground its concrete findings.

Do NOT fail the gate merely because the block refers to "the task", "the question", "the prior summary", "Option A", or similar context-dependent wording. 
STEP 2 can restate that context from the assigned sub-task.

Do NOT re-verify citation correctness or arithmetic. Those checks have already been performed.

If the block fails the gate, output exactly:
NO_RECOVERABLE_FINDING: <one-line reason naming what STEP 2 would have nothing to work from>

Then stop. Otherwise, proceed to STEP 2.

STEP 2 — GIST GENERATION

Only if the block passes the gate, write a compact gist that another reader can use WITHOUT seeing the task description, the reasoning block, or any other context.

This gist becomes SETTLED EVIDENCE: the final answerer relies on it directly and never sees the reasoning block behind it. So the gist must state the finding at EXACTLY the strength and scope the reasoning block established — never stronger, never broader. An over-claimed gist (a scoped or tentative finding flattened into a flat universal assertion) directly mis-leads the final answer.

Use the assigned sub-task only to restate what was being asked. The RESULT must be derived from the verified reasoning block, not invented from the task wording.

Write EXACTLY this structure and nothing else:

**[{label}] GIST**
SUB-TASK: <the assigned sub-task, restated plainly and completely in neutral wording; never "the task above" or "the question">
RESULT:   <the concrete finding from the reasoning block — a fact, value, computed comparison, absence finding, or extracted passage — with necessary source label(s) preserved inline>

Rules for the gist:
- SUB-TASK must be understandable with zero outside context.
- RESULT must be self-contained and informative.
- RESULT must never be a bare option letter.
- RESULT must never say "see above" or depend on hidden context.
- FIDELITY — do NOT strengthen the reasoning block's conclusion. Preserve every hedge and scope it used ("based on the available summaries", "appears to", "the documents reviewed do not state X"). If the block hedged or scoped a claim, the gist keeps that hedge; never turn a tentative or scoped finding into a flat assertion. Add no conclusion, comparison, or claim the block did not make.
- ABSENCE findings must be SCOPED to what was actually searched — e.g. "searched [W0 S5-S8]; no mention of X there" — never a bare universal negative like "X does not exist" or "X is not used". Absence in the searched sources is not proof of absence.
- Preserve `[W<i> S<j>]` and `[W<i> S<j>.<n>]` citation labels exactly as they appear in the reasoning block.
- Keep each line condensed: one line of substance, no preamble.

---

Assigned sub-task ({label}):
{task}

---

Verified reasoning block:
{reasoning}
"""



def build_gist_verifier_prompt(gist: str) -> str:
    """gist verifier. Checks the gist is SELF-CONTAINED: another
    agent seeing ONLY this gist can tell what was investigated and what
    was found. The self-containment criteria are operationalised as
    concrete, scannable conditions — the old prompt asked the same thing
    ("self-contained and informative?") but vaguely, which forced a
    subjective judgement into one word and made the vote unreliable."""
    return f"""You are running a FORMAT CHECK on a gist — verifying concrete,
scannable conditions, not making an open-ended judgement. Do NOT fact-check
the finding or judge whether it is correct or complete.

A gist PASSES when ALL of these hold:
1. It has a `SUB-TASK:` line that names the actual question or task in plain
   words — not a placeholder like "the task above", "the question", or "as requested".
2. It has a `RESULT:` line.
3. The text after `RESULT:` is an actual finding — a fact, value, comparison,
   extracted passage, or an explicit "not found / absent" statement — and is
   NOT a bare option letter (e.g. just "C"), NOT empty, and NOT a one-word stub.
4. The gist does not lean on text the reader cannot see — no "see above",
   "the prior summary", or "as noted" references.

Gist:
{gist}

Answer with exactly one word: yes (passes) or no (fails).
"""


def build_verify_redundancy_prompt(shared_context: str, summary: str) -> str:
    return f"""{shared_context}

Proposed summary to add to the shared context:
{summary}

You are a verifier. Check whether this summary adds NEW, useful
information to the shared context that brings us closer to the final
answer, or whether the shared context already contains the same
information.

Answer with exactly ONE of:

ACCEPT — the summary adds new information (facts, intermediate results,
  or conclusions) not already present in the shared context.
REDUNDANT — the shared context already contains equivalent information.
  Adding this summary would not help.

Your verdict (one word, then optional one-line explanation):"""


__all__ = [
    "build_verify_prompt",
    "build_gate_and_gist_prompt",
    "build_gist_verifier_prompt",
    "build_verify_redundancy_prompt",
]
