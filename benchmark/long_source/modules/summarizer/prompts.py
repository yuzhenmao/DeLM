"""Prompt builders for the phi_1/2 Summarizer module.

  - build_doc_summary_prompt
      phi_1, query-independent L1 summary.
  - build_doc_summary_gapfill_prompt
      Conditional retry that fills coverage gaps left by the initial L1
      pass; shares the same prefix so the provider's prompt cache can
      serve the bulk of the input tokens.
  - build_doc_gist_prompt
      phi_2, query-aware L2 gist.
"""

# L1 generation cap (tokens). Densifies the sub-chunk index, not noise.
DOC_SUMMARY_MAX_TOKENS = 2500
# phi_2 query-aware L2 gist cap (tokens).
DOC_GIST_MAX_TOKENS = 100


def build_doc_summary_prompt(doc_title: str, doc_text: str) -> str:
    """Phase 0 — QUERY-INDEPENDENT hybrid sub-chunk-index L1 summary.

    the L1 is a pure STRUCTURAL INDEX:
      - Chunk topic (1–2 sentences)
      - Sub-chunks: N≈⌈chunk_chars/2500⌉ contiguous region pointers,
        each anchored by `[anchor: <head> ... <tail>]` (head/tail copied
        verbatim from raw, validated post-hoc via _match_head_tail)
    """
    import math
    n_chars = len(doc_text)
    n_sub = max(3, min(12, math.ceil(n_chars / 2500)))

    return f"""You are reading ONE chunk from a multi-document archive.

Your job is to build a STRUCTURAL INDEX of sub-chunk pointers. Future workers will use this index to decide which raw regions to fetch and inspect in detail. The L1 summary itself should NOT capture specific values, dates, or claims; those remain in the raw source and are accessed through the anchor pointers.

The summary must be QUERY-INDEPENDENT. Describe what the chunk contains, not what any downstream question may ask.

This chunk is {n_chars:,} characters long. Target {n_sub} sub-chunks in the index, give or take 1–2. Each sub-chunk should cover roughly 2,500 source characters on average, unless the chunk structure makes a different split more natural.

INPUT

Document title: [{doc_title}]

Chunk text:
{doc_text}

OUTPUT REQUIREMENTS

Produce exactly TWO sections, in this order:

Chunk topic:
<1–2 sentences describing the chunk's main subject matter.>

Sub-chunks:
- [Sub-chunk 1] <1–2 sentence description>; row labels: <comma-list>; columns/periods: <header labels> [anchor: <head 5–8 words> ... <tail 5–8 words>]
- [Sub-chunk 2] <1–2 sentence description> [anchor: <head 5–8 words> ... <tail 5–8 words>]
- ...

SECTION 1 — Chunk topic

Write 1–2 sentences describing the chunk's overall subject matter.

The topic should be descriptive and query-independent. It should tell a future worker what kind of content appears in the chunk, such as narrative discussion, risk factors, policies, financial tables, operating results, reconciliation data, legal terms, methodology, or other document-specific material.

SECTION 2 — Sub-chunks

Create an index of {n_sub} sub-chunks, give or take 1–2. **Treat the target as an upper guideline, not a quota.** If the chunk's content does not support that many natural divisions — for example, it's one continuous narrative section (a single letter, essay, or commentary), one large table without internal sub-divisions, or otherwise a single semantic unit — produce FEWER sub-chunks. Even 1–3 sub-chunks for a {n_chars:,}-char chunk is correct when the content is one cohesive piece. Fewer well-anchored sub-chunks beats more sub-chunks with generic or duplicated anchors.

Each sub-chunk is a contiguous region of the source text. Together, the sub-chunks must tile the full chunk in source order.

Sub-chunk coverage rules:
1. Sub-chunk 1 must begin at the chunk's first meaningful source content.
2. The final sub-chunk must end at the chunk's last meaningful source content.
3. The sub-chunks must cover the entire chunk with no gaps.
4. The sub-chunks must not overlap.
5. Preserve source order.
6. Split at natural section, paragraph, table, list, or topic boundaries when possible.
7. Group related content together. For example, a financial statement should usually be one sub-chunk, not many; its title, header rows, line items, subtotals, reconciliation rows, and totals belong together.
8. A long narrative section may be split into multiple sub-chunks at natural paragraph or subsection breaks — but only if those internal breaks have distinctive anchor text available. If the narrative is one continuous flow without distinctive internal markers, keep it as a single sub-chunk rather than splitting it into pieces that share the same generic anchor.

Sub-chunk description rules:
- Each sub-chunk entry must be exactly one bullet.
- Start each bullet with:
  [Sub-chunk N]
- Use sequential 1-based numbering in source order.
- Describe what the region contains in 1–2 sentences.
- Describe only the type and scope of content.
- Do NOT include specific values, numbers, dates, percentages, monetary amounts, or quoted source text in the description. (Exception: dates, years, or period labels may appear in `columns/periods` when they are table headers. Do not include table cell values.)
- Mention the material type when useful: narrative explanation, table, reconciliation, list, policy text, risk discussion, definitions, methodology, operating results, legal terms, etc.
- For tabular regions, include:
  row labels: <comma-list of EVERY row label>
  columns/periods: <header labels>
- For non-tabular regions, omit the `row labels:` and `columns/periods:` segments.

Anchor rules — CRITICAL:
- Every sub-chunk bullet must end with an anchor.
- Anchor format:
  [anchor: <head 5–8 words> ... <tail 5–8 words>]
- The anchor head must be copied VERBATIM from a short span at or near the start of that sub-chunk's source region.
- The anchor tail must be copied VERBATIM from a short span at or near the end of that sub-chunk's source region.
- The head and tail should identify the intended source span reliably.
- Preserve punctuation, capitalization, Unicode characters, hyphens, dashes, and quotation marks exactly.
- The head and tail must literally appear in the source as substrings.
- Never invent, normalize, or paraphrase anchor text.
- Do not use answer-choice wording, outside knowledge, or your own summary wording as anchor text.

Anchor distinctiveness:
- Each anchor head and tail should contain at least one distinctive token, such as a number, named entity, technical term, capitalized proper noun, dated phrase, row label, section header, defined term, or rare phrase.
- Avoid anchors made only of generic phrases that could appear elsewhere in the same chunk, such as "as a result of the," "In addition to this," or "during the year."
- If the literal first or last words of a region are too generic, choose a nearby distinctive substring still close to the boundary.
- Prefer anchor text that is visually easy to locate in the source, such as a section header, row label, fiscal period, person name, organization name, or defined term.
- Two different sub-chunks must NOT share the same head text. If the model is tempted to use the same section header / running title for multiple sub-chunks, that is a signal those sub-chunks should be merged into ONE sub-chunk (see the upper-guideline rule above), not split with identical anchors.

FINAL OUTPUT FORMAT

Chunk topic:
<text>

Sub-chunks:
- [Sub-chunk 1] <description>; row labels: <...>; columns/periods: <...> [anchor: <head> ... <tail>]
- [Sub-chunk 2] <description> [anchor: <head> ... <tail>]
- ...
"""



def build_doc_summary_gapfill_prompt(doc_title: str, doc_text: str,
                                               previous_l1: str,
                                               gap_regions: list,
                                               next_subchunk_idx: int) -> str:
    """phi_1 — conditional retry that fills coverage gaps left by the
    initial L1 pass. Designed to reuse the same prompt prefix as the
    initial call (build_doc_summary_prompt) so the provider's
    prompt cache can serve the bulk of the input tokens; only the
    appended "PREVIOUS ATTEMPT + GAP-FILL TASK" block is fresh.

    `gap_regions` = list of dicts {"start": int, "end": int, "head_excerpt": str,
                                    "tail_excerpt": str, "length": int} in
    whitespace-normalised raw space. We pass these to the model as
    excerpt anchors (first/last ~60 chars of each gap) so the model can
    locate the region in the chunk it already has in context.

    `next_subchunk_idx` = the integer N to use for the FIRST new sub-chunk
    (so numbering continues from where the previous attempt left off).
    """
    base = build_doc_summary_prompt(doc_title, doc_text)
    base = base.rstrip()

    gaps_block = []
    for i, g in enumerate(gap_regions, start=1):
        gaps_block.append(
            f"Gap {i}: ~{g['length']:,} chars\n"
            f"  starts with: \"{g['head_excerpt']}\"\n"
            f"  ends with:   \"{g['tail_excerpt']}\""
        )
    gaps_text = "\n\n".join(gaps_block)

    suffix = f"""

────────────────────────────────────────────────────────
PREVIOUS ATTEMPT
The previous L1 output is shown below for reference only. Do NOT repeat, revise, or restate any earlier sub-chunks.

{previous_l1}

────────────────────────────────────────────────────────
GAP-FILL TASK

The previous sub-chunk anchors failed to cover the following source regions. Produce ADDITIONAL sub-chunk entries that cover these missing regions.

Use the same sub-chunk format and rules as the original structural-index task.

Missing regions:
{gaps_text}

Rules for this retry:
- Output ONLY new sub-chunk bullets.
- Do NOT output Chunk topic.
- Do NOT repeat any earlier sub-chunks.
- Do NOT include prose, headers, explanations, or diagnostics.
- Number the new sub-chunks starting from [Sub-chunk {next_subchunk_idx}] and continue sequentially.
- Produce at least one new sub-chunk for each missing region.
- If one missing region contains multiple unrelated content blocks, you may split it into multiple new sub-chunks.
- Each new sub-chunk must cover a contiguous span within the provided gap text.
- Together, the new sub-chunks for a gap must cover the full missing region in source order.
- Anchor text must be copied VERBATIM from the gap text only.
- Do NOT use wording from the previous attempt as anchor text unless that wording also appears verbatim in the gap text.
- Choose anchor heads and tails that are as close as possible to the beginning and end of the missing region.
- If the exact boundary words are generic, choose the nearest distinctive substring inside the gap.
- Prefer distinctive anchor text containing numbers, proper nouns, technical terms, row labels, section headers, defined terms, or rare phrases.
- For tabular gaps, include row labels and columns/periods.
- For non-tabular gaps, omit row labels and columns/periods.
- Do NOT include table cell values in the description.
- Do NOT include specific values, numbers, dates, percentages, monetary amounts, or quoted source text in the description, except when they are row labels or column/period headers.
- Every bullet must end with an anchor in this format:
  [anchor: <head 5–8 words> ... <tail 5–8 words>]

Output format:
- [Sub-chunk {next_subchunk_idx}] <description>; row labels: <...>; columns/periods: <...> [anchor: <head> ... <tail>]
- [Sub-chunk {next_subchunk_idx + 1}] <description> [anchor: <head> ... <tail>]
- ...
"""
    return base + suffix



def build_doc_gist_prompt(question: str, chunk_summary: str,
                                     choices: dict | None = None) -> str:
    """phi_2 — question-aware gist for one document chunk.

    Produces a 1–2 sentence (≤ ~DOC_GIST_MAX_TOKENS word) neutral description of what the
    chunk contains, with question-relevant material surfaced first. The
    gist is appended to `shared_ctx` and seen by every Phase-2 worker
    on every reasoning call — so it must be dense and short, NOT a
    relevance verdict.
    """
    choices_block = ""
    if choices:
        choices_block = (
            "\n\nAnswer choices:\n"
            + "\n".join(f"{L}) {choices.get(L, '')}"
                        for L in "ABCD" if choices.get(L))
        )
    return f"""You are reviewing ONE document chunk for a downstream reasoning system.

You are given:
1. A question with answer choices.
2. A question-independent evidence map for one document chunk (chunk topic + sub-chunk descriptions).

Your task is to write a 1–2 sentence gist (target ≤ {DOC_GIST_MAX_TOKENS} words) describing what this chunk contains.

The gist is appended to a shared context that every downstream worker sees on every reasoning call. It must be DENSE and NEUTRAL — it helps workers decide whether to fetch raw text from this chunk; it is NOT a relevance verdict and NOT a stand-in for the chunk itself.

Question:
{question}{choices_block}

Chunk evidence map:
{chunk_summary}

Rules:

ORDERING — the only role of the question/choices
- Use the question and answer choices ONLY to decide what to mention first.
- Surface entities, dates, values, qualifiers, mechanisms, or claims that may bear on the question before other content.
- Do not omit content. If the chunk also contains material unrelated to the question, briefly describe that too.

GROUNDING
- Every statement must be supported by content in the evidence map.
- Do not use outside knowledge.
- Do not infer beyond what the evidence map states.

NEUTRALITY — do not leak verdicts
- Do not state which answer choice is correct, plausible, or implausible.
- Do not use the words "relevant," "irrelevant," "supports," "contradicts," "proves," or "rules out."
- Describe what the chunk CONTAINS or REPORTS, not what should be CONCLUDED from it.

Output a single line in this exact format:

Gist: <1–2 sentences>"""



__all__ = [
    "build_doc_summary_prompt",
    "build_doc_summary_gapfill_prompt",
    "build_doc_gist_prompt",
]
