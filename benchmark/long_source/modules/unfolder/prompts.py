"""Prompt builder for the Unfolder module (selective unfolding, step 2a).

Self-contained copy of `build_selective_unfold_prompt` (inlined from the
root `prompts.py`). Custom unfolders can write their own prompt function
and pass it to `DefaultUnfolder(prompt_fn=...)`, or subclass and override
`build_prompt`.
"""

# step2a_selective_unfold cap: how many initial chunks the worker picks
# from DOC GISTs before reasoning.
K_COARSE = 5


def build_selective_unfold_prompt(shared_context: str, task: str) -> str:
    return f"""{shared_context}

**Task:** {task}

Before reasoning, select which context entries should be expanded to full content.

Your goal is to request the smallest set of labels needed to obtain direct evidence for the task.

Entry types:
- **DOC GIST entries** (`W0 S<i>`): brief descriptions of document chunks. Request these when the task needs a specific fact, date, entity, quantity, table value, exact wording, or qualifier that the gist does not fully provide.
- **Worker SUMMARY entries** (`W<n> S<j>`, n >= 1): verified worker summaries. Request these only when the task depends on that worker's intermediate reasoning, intermediate values, or trace details, not merely its committed summary.

Rules:
- Request only labels that appear in the context above.
- Do not invent labels.
- Prefer DOC GIST entries when the task needs direct documentary evidence.
- Prefer Worker SUMMARY entries only when prior reasoning details are necessary.
- Select at most {K_COARSE} labels total.
- Select the fewest labels that are likely to resolve the task — but never at the cost of skipping evidence the task actually needs.
- If the task turns on a specific fact, number, date, amount, quantity, table value, exact wording, or qualifier, you MUST expand at least the DOC GIST entry/entries that would carry it. Do NOT output `NONE` for such tasks: a brief gist is not direct evidence for a precise value.
- Output `NONE` ONLY when the task is answerable from content that is ALREADY expanded in the context above AND it does not hinge on any precise value or exact wording.
- If expansion is needed, output only a comma-separated label list.
- Do not explain your choices.

Output format:
<label1>, <label2>, <label3>

or:

NONE
"""


__all__ = ["build_selective_unfold_prompt"]
