"""long_source_split.py — boundary detection + LLM sub-chunking.

Usage:

    from client import OpenRouterClient
    from long_source_split import split_documents

    client = OpenRouterClient("deepseek/deepseek-v4-flash",
                              disable_reasoning=True)
    chunks = split_documents(context, client=client)

If `client` is None, large chunks are returned un-split.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

# ----------------------------------------------------------------------
# Step 1: word-headed boundary patterns.
# ----------------------------------------------------------------------

_WORD_HEADED_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?:^|\n)\s*(?P<title>Section\s+\d+[^\n]{0,120})\n", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?P<title>Article\s+\d+[^\n]{0,120})\n", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?P<title>Chapter\s+\d+[^\n]{0,120})\n", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?P<title>Source\s*:[^\n]{1,200})\n", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?P<title>Part\s+\d+[^\n]{0,120})\n", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?P<title>Title\s+\d+[^\n]{0,120})\n", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?P<title>Item\s+\d+[A-Z]?[.\s][^\n]{0,120})\n", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?P<title>(?:Schedule|Annex|Appendix)\s+[A-Z\d]+[^\n]{0,120})\n", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?P<title>Document\s+\[?\d+\]?[^\n]{0,80})\n", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?P<title>Passage\s+\[?\d+\]?[^\n]{0,80})\n", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?P<title>Doc\s+\[?\d+\]?[^\n]{0,80})\n", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?P<title>Title:[^\n]{0,200})\n", re.IGNORECASE),
]

# Min matches to accept a pattern. Priority order — first pattern with
# enough matches wins.
_BOUNDARY_MIN_MATCHES = 3

# ----------------------------------------------------------------------
# Sub-chunking thresholds.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Tuned for downstream per-doc summarization with deepseek-v4-flash:
#   • MIN ≈ 1.5K input tokens — keeps summary fixed-overhead cost amortized
#     (a doc-summary call has ~300 tokens of instructions + ~200 of output;
#      below ~1.5K input the overhead dominates).
#   • MAX ≈ 5K input tokens — comfortably below the lost-in-middle threshold
#     for Flash-class models (~8-12K tokens) so dense facts are recalled
#     reliably.
#   • TRIGGER just above MAX so the LLM sub-split kicks in for chunks that
#     would otherwise sit awkwardly above the comfortable summarization
#     band; smooths the discontinuity (a 24K chunk passes; 26K splits to
#     2 × 13K).
# ----------------------------------------------------------------------
LLM_SPLIT_TRIGGER = 30_000
MIN_SUBCHUNK_CHARS = 6_000
MAX_SUBCHUNK_CHARS = 25_000
MAX_CANDIDATES_TO_LLM = 80
SUBSPLIT_MAX_TOKENS = 600


# ----------------------------------------------------------------------
# Step 1 helpers
# ----------------------------------------------------------------------

def _is_real_boundary(context: str, match: re.Match) -> bool:
    """Check whether `match` is a real paragraph/section boundary, not a
    soft line-wrap inside running prose.

    Walks back from the start of the captured title, counting newlines
    encountered through whitespace. Accepts the match iff:
      • start-of-string (nothing before), OR
      • ≥2 newlines walked through (blank line precedes the marker), OR
      • exactly 1 newline + the previous non-whitespace/non-closing-punct
        character is a sentence terminator (`.`, `!`, `?`, `:`).

    Rejects inline references like "...as defined in\\nSection 5(a) of
    the Act..." (1 newline, no terminator before).
    """
    title_start = match.start("title")
    if title_start == 0:
        return True
    i = title_start - 1
    n_newlines = 0
    while i >= 0 and context[i] in " \t\n":
        if context[i] == "\n":
            n_newlines += 1
        i -= 1
    if n_newlines >= 2:
        return True  # blank line above the marker
    if i < 0:
        return True  # only whitespace between BOF and the marker
    # Skip closing punctuation that often follows a sentence terminator
    # (e.g. .", .', .)).
    while i >= 0 and context[i] in '"\')]}':
        i -= 1
    return i >= 0 and context[i] in ".!?:"


def _score_pattern_matches(matches: List[re.Match]) -> float:
    """Score = n_matches × spacing_regularity. Higher is better.
    Regularity = 1 / (1 + cv_of_gaps), so a pattern with many evenly-spaced
    matches beats one with the same count clustered together."""
    if len(matches) < _BOUNDARY_MIN_MATCHES:
        return 0.0
    positions = [m.start() for m in matches]
    if len(positions) < 2:
        return float(len(matches))
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap == 0:
        return 0.0
    var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    std = var ** 0.5
    regularity = 1.0 / (1.0 + std / mean_gap)
    return len(matches) * regularity


def _select_boundary_pattern(context: str) -> Optional[List[re.Match]]:
    """Score-based picker: every word-headed pattern is evaluated by
    `_score_pattern_matches` and the highest-scoring one wins. This handles
    cases where the priority order would pick a weaker match (e.g. 3
    `Section` matches winning over 20 well-spaced `Source:` matches).
    Each candidate match is filtered by `_is_real_boundary` so inline
    references inside sentences don't trigger false positives."""
    best_score = 0.0
    best_matches: Optional[List[re.Match]] = None
    for pat in _WORD_HEADED_PATTERNS:
        matches = [m for m in pat.finditer(context)
                   if _is_real_boundary(context, m)]
        score = _score_pattern_matches(matches)
        if score > best_score:
            best_score = score
            best_matches = matches
    return best_matches


def _split_at_boundaries(context: str,
                         matches: List[re.Match]) -> List[Tuple[str, str]]:
    """Slice context at each match. Anything before the first match becomes
    a Preamble chunk if it's substantial (>200 chars)."""
    chunks: List[Tuple[str, str]] = []
    preamble = context[:matches[0].start()].strip()
    if len(preamble) > 200:
        chunks.append(("Preamble", preamble))
    for i, m in enumerate(matches):
        title = m.group("title").strip().rstrip(":").strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
        body = context[body_start:body_end].strip()
        if body:
            chunks.append((title, body))
    return chunks


def _force_split_deterministic(body: str, max_chars: int) -> List[str]:
    """Recursively split `body` at the paragraph boundary closest to the
    midpoint until every piece is ≤ max_chars. No LLM, no heuristics —
    pure safety net for chunks the LLM couldn't tame.

    Falls back gracefully when no paragraph break exists (returns the body
    un-split rather than slicing mid-sentence)."""
    if len(body) <= max_chars:
        return [body]
    breaks = list(re.finditer(r"\n\s*\n", body))
    if not breaks:
        # Try single newline as second choice.
        breaks = list(re.finditer(r"\n", body))
    if not breaks:
        return [body]  # truly no break point — give up
    mid = len(body) // 2
    best = min(breaks, key=lambda m: abs(m.start() - mid))
    cut = best.end()
    left = body[:cut].strip()
    right = body[cut:].strip()
    if not left or not right:
        return [body]  # split didn't produce two non-empty halves
    return (_force_split_deterministic(left, max_chars) +
            _force_split_deterministic(right, max_chars))


def _join_titles(a: str, b: str) -> str:
    """Concatenate two titles with ` + `, collapsing duplicates and capping
    length so a chunk that absorbs many tiny neighbours doesn't grow a huge
    title."""
    if a == b or not b:
        return a
    if not a:
        return b
    # Avoid duplicating sub-strings already present (e.g. "Item 5 + Item 5
    # (part 2/3)" → "Item 5 (part 2/3)").
    if a in b:
        return b
    if b in a:
        return a
    joined = f"{a} + {b}"
    if len(joined) > 200:
        return f"{a} + …"
    return joined


def _merge_tiny_chunks(chunks: List[Tuple[str, str]],
                       min_chars: int) -> List[Tuple[str, str]]:
    """Merge any chunk with body < min_chars into a neighbour to avoid
    wasteful tiny chunks (TOC fragments, trailing recursion pieces,
    naturally short clauses).

    Direction:
      • If the FIRST chunk is tiny, merge it into the NEXT chunk.
      • Otherwise merge a tiny chunk into the PREVIOUS chunk.
    """
    result = list(chunks)
    # Merge tiny first chunk into next (repeat in case multiple lead with tiny).
    while len(result) >= 2 and len(result[0][1]) < min_chars:
        t0, b0 = result[0]
        t1, b1 = result[1]
        result = [(_join_titles(t0, t1), b0 + "\n\n" + b1)] + result[2:]
    # Merge each subsequent tiny chunk into its previous.
    i = 1
    while i < len(result):
        if len(result[i][1]) < min_chars:
            prev_t, prev_b = result[i - 1]
            cur_t, cur_b = result[i]
            result[i - 1] = (_join_titles(prev_t, cur_t),
                             prev_b + "\n\n" + cur_b)
            del result[i]
        else:
            i += 1
    return result


# ----------------------------------------------------------------------
# Step 3: paragraph-leading sentence candidates + LLM sub-split
# ----------------------------------------------------------------------

# Soft line-wrap: a single \n not preceded by a sentence terminator
# (.,!,?,:) — common in PDF-extracted text where lines wrap mid-sentence.
_SOFT_WRAP_RE = re.compile(r"(?<![.!?:])\n(?!\n)")


def _first_sentence(text: str, max_chars: int = 200) -> str:
    """First sentence of `text`, capped at `max_chars`.

    Soft single-newline line wraps are collapsed to spaces so PDF-style
    line-wrapped paragraphs read as continuous sentences. The sentence
    ends at the first hard terminator (./!/?), a paragraph break (\\n\\n),
    or the cap.
    """
    text = text.lstrip()
    if not text:
        return ""
    cleaned = _SOFT_WRAP_RE.sub(" ", text)
    pat = re.compile(r"(.{1," + str(max_chars) + r"}?(?:[.!?](?:\s|$)|\n\n))",
                     re.DOTALL)
    m = pat.match(cleaned)
    if m:
        return m.group(1).strip()
    return cleaned[:max_chars].strip()




# Characters that count as "closing punctuation" when sitting between a
# sentence terminator and the newline (e.g. .") or .')).
_TRAILING_PUNCT = set(' \t"\')]}')
# Characters that mark a real sentence end.
_SENT_TERMINATORS = set(".!?:")


def _find_paragraph_candidates(body: str) -> List[Tuple[int, str]]:
    """Find character offsets where a real new sentence starts.

    Returns offset 0 plus every offset after a `\\n+` run that is a true
    sentence boundary, as opposed to a soft line-wrap.

    Boundary kept iff:
      • It's a paragraph break (the run is `\\n\\n` or longer), OR
      • Single `\\n` whose preceding non-whitespace/non-closing-punct
        character is a sentence terminator (`.`, `!`, `?`, `:`).

    PDF-extracted text wraps lines mid-sentence with single `\\n`s; this
    filter drops those positions so the LLM doesn't see mid-paragraph
    candidates that look like topic starts but aren't.
    """
    candidates: List[Tuple[int, str]] = []
    seen = set()
    candidates.append((0, _first_sentence(body)))
    seen.add(0)
    for m in re.finditer(r"\n+", body):
        pos = m.end()
        if pos in seen or pos >= len(body):
            continue
        is_paragraph_break = (m.end() - m.start()) >= 2
        if not is_paragraph_break:
            # Walk back over whitespace and closing punctuation.
            i = m.start() - 1
            while i >= 0 and body[i] in _TRAILING_PUNCT:
                i -= 1
            prev_char = body[i] if i >= 0 else ""
            if prev_char not in _SENT_TERMINATORS:
                continue  # mid-paragraph soft wrap, not a real boundary
        seen.add(pos)
        snippet = _first_sentence(body[pos:pos + 300])
        if snippet:
            candidates.append((pos, snippet))
    return candidates


_HEADING_RE_NUMBERED = re.compile(
    r"^\s*(?:"
    r"\d+\.\d+(?:\.\d+)*\.?\s+[A-Z]"      # decimal: "3.1 Data", "3.2.4 Foo"
    r"|\d+\.\s+[A-Z]"                      # period: "3. Methods"
    r")"
)
_HEADING_RE_NAMED = re.compile(
    r"^\s*(?:"
    r"(?:Section|Article|Chapter|Part|Item)\s+\d+[A-Z]?\b"  # "Section 5", "Item 7A"
    r"|(?:Appendix|Schedule|Annex)\s+[A-Z\d]+\b"             # "Appendix A", "Schedule 5"
    r"|(?:Document|Passage|Doc)\s+\[?\d+\]?\b"               # "Document 5", "Doc [3]"
    r"|(?:Source|Title)\s*:"                                 # "Source:", "Title:"
    r"|#{1,6}\s+\S"                                          # markdown #, ##, ...
    r")",
    re.IGNORECASE,
)
_HEADING_RE_ALLCAPS = re.compile(
    r"^\s*[A-Z][A-Z0-9 \-.,:'\"&/()]{3,60}\b"
)


def _looks_like_heading(snippet: str) -> bool:
    s = snippet or ""
    return bool(
        _HEADING_RE_NUMBERED.match(s)
        or _HEADING_RE_NAMED.match(s)
        or _HEADING_RE_ALLCAPS.match(s)
    )


def _downsample_candidates(candidates: List[Tuple[int, str]],
                           max_count: int) -> List[Tuple[int, str]]:
    """Reduce candidates to at most max_count while keeping spatial coverage
    AND prioritising heading-like previews.

    Strategy: divide the candidate list into `max_count - 1` spatial
    buckets (excluding offset 0, which is always kept). From each bucket
    pick the FIRST heading-like candidate; if no heading lives in a bucket,
    pick its first candidate. This guarantees:
      • Every region of the document gets at least one boundary candidate
        (no unsegmented tails).
      • Where headings exist, the LLM sees them.
    """
    if len(candidates) <= max_count:
        return candidates

    keep: set = {0}
    n_buckets = max_count - 1
    bucket_size = len(candidates) / n_buckets

    for b in range(n_buckets):
        lo = max(1, int(b * bucket_size))           # skip offset-0 candidate
        hi = max(lo + 1, int((b + 1) * bucket_size))
        if lo >= len(candidates):
            break
        # Within this bucket, prefer the first heading-like candidate.
        chosen = lo
        for j in range(lo, min(hi, len(candidates))):
            if _looks_like_heading(candidates[j][1]):
                chosen = j
                break
        keep.add(chosen)

    return [candidates[i] for i in sorted(keep) if i < len(candidates)]


def _max_boundaries_for(body_len: int) -> int:
    """At most floor(body_len / MAX_SUBCHUNK_CHARS) + 2 sub-chunks. The +2
    gives the LLM a small amount of flexibility above the natural minimum,
    so it can include one or two extra topic-shift boundaries when warranted
    without over-splitting."""
    return body_len // MAX_SUBCHUNK_CHARS + 2


def _build_subsplit_prompt(body: str,
                           candidates: List[Tuple[int, str]],
                           title: str = "") -> str:
    """Prompt asking the LLM which candidate offsets to use as sub-chunk
    boundaries. Output format is the SAME tuple list as the input but with
    only the chosen entries."""
    candidates_str = "\n".join(
        f"  ({pos}, {snippet[:140]!r})"
        for pos, snippet in candidates
    )
    max_b = _max_boundaries_for(len(body))
    return (
        "You are splitting a long text chunk into smaller, semantically "
        "coherent sub-chunks at paragraph boundaries.\n\n"
        "INPUT\n"
        "  • CHUNK: identified by title and total length in characters.\n"
        "  • CANDIDATES: a Python-style list of (char_offset, "
        "first_sentence_preview) tuples — every position in the chunk "
        "where a paragraph or line begins.\n\n"
        "TASK\n"
        "Pick a SUBSET of the candidates to use as sub-chunk START "
        "boundaries. The text from one chosen offset to the next chosen "
        "offset (or to the chunk end, for the last one) becomes one sub-"
        "chunk.\n\n"
        "RULES\n"
        f"  1. ALWAYS include offset 0 (the very first candidate).\n"
        f"  2. Output AT MOST {max_b} tuples in total (including offset 0). "
        f"This cap is computed from the chunk size: "
        f"floor({len(body):,} / {MAX_SUBCHUNK_CHARS:,}) + 2 = {max_b}. "
        f"Picking fewer is fine; picking more is NOT allowed.\n"
        f"  3. Each resulting sub-chunk should ideally be between "
        f"{MIN_SUBCHUNK_CHARS:,} and {MAX_SUBCHUNK_CHARS:,} characters. "
        f"Estimate sizes by subtracting consecutive offsets (and the "
        f"chunk's total length for the last sub-chunk).\n"
        f"  4. AVOID any sub-chunk LARGER than {MAX_SUBCHUNK_CHARS:,} "
        f"characters unless the candidate list makes this impossible "
        f"(e.g. no candidates exist in a long stretch).\n"
        f"  5. AVOID any sub-chunk SMALLER than {MIN_SUBCHUNK_CHARS:,} "
        f"characters unless needed to prevent an oversized neighbour or "
        f"to preserve a clear semantic boundary.\n"
        "  6. Prefer offsets whose preview signals a clear semantic "
        "transition (new topic, section, entity, event, time period, "
        "document, or argument) — NOT mere continuation of the previous "
        "paragraph.\n"
        "  7. AVOID putting a boundary inside a logical unit: a list of "
        "bullets, a table, a multi-line quote, a Q/A pair, a dialogue, "
        "an example, or a tightly connected explanation.\n"
        "  8. Select tuples VERBATIM from CANDIDATES. Do NOT invent, "
        "edit, summarize, or renumber offsets or previews.\n"
        "  9. The chunk total length is given; do NOT include an offset "
        "equal to the total length.\n\n"
        "OUTPUT FORMAT\n"
        "A Python list literal of the chosen tuples in ASCENDING offset "
        "order, copying the offsets and snippets EXACTLY from the input. "
        "Output nothing else — no explanation, no markdown, no code "
        "block, no headers.\n\n"
        "Example:\n"
        "[(0, 'First sentence of the chunk.'), (5234, 'Topic shift "
        "begins here.'), (12345, 'Next major section starts.')]\n\n"
        f"CHUNK TITLE: {title or '(untitled)'}\n"
        f"CHUNK TOTAL LENGTH: {len(body)} characters\n"
        f"MAX BOUNDARIES TO OUTPUT: {max_b}\n\n"
        f"CANDIDATES:\n{candidates_str}\n\n"
        "Your answer (Python list of tuples only):"
    )


# Match each (offset, "snippet") tuple in the LLM response. Permissive about
# quoting style and snippet content.
_TUPLE_RE = re.compile(
    r"\(\s*(\d+)\s*,\s*[\"'](.*?)[\"']\s*\)",
    re.DOTALL,
)


def _parse_llm_positions(text: str) -> List[int]:
    """Extract offset integers from the LLM's tuple-list output. Accepts
    single or double quotes around the snippet."""
    positions = []
    for m in _TUPLE_RE.finditer(text or ""):
        try:
            positions.append(int(m.group(1)))
        except ValueError:
            continue
    return positions


def _slice_at_positions(body: str, positions: List[int]) -> List[str]:
    """Slice body at the given (sorted, in-bounds) start positions."""
    out: List[str] = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(body)
        piece = body[pos:end].strip()
        if piece:
            out.append(piece)
    return out


# Recursive sub-split: any sub-chunk still bigger than this threshold gets
# another LLM pass. Set as a multiple of MAX_SUBCHUNK_CHARS so we tolerate
# small overshoots but recurse on moderate-or-worse ones (Flash-class models
# tend to cluster boundaries near the start of long prompts, leaving big
# tails that benefit from a second pass).
_RECURSE_THRESHOLD_FACTOR = 1.2
_RECURSE_MAX_DEPTH = 2


def _llm_subsplit(body: str, client, title: str = "",
                  verbose: bool = False, _depth: int = 0) -> List[str]:
    """Use the LLM to pick sub-chunk boundaries inside `body`. Returns the
    sliced sub-chunks. Falls back to [body] on any failure or empty result.

    Recursive: if any resulting sub-chunk is still > _RECURSE_THRESHOLD_FACTOR ×
    MAX_SUBCHUNK_CHARS, we run another LLM pass on just that piece. Bounded
    by `_RECURSE_MAX_DEPTH` to prevent runaway recursion."""
    candidates = _find_paragraph_candidates(body)
    if len(candidates) < 2:
        return [body]
    candidates = _downsample_candidates(candidates, MAX_CANDIDATES_TO_LLM)
    prompt = _build_subsplit_prompt(body, candidates, title=title)
    if verbose:
        depth_tag = f" depth={_depth}" if _depth else ""
        print(f"  [source-split{depth_tag}] LLM call for chunk '{title[:40]}': "
              f"{len(body):,} chars, {len(candidates)} candidates")
    resp = client.generate(prompt, max_new_tokens=SUBSPLIT_MAX_TOKENS,
                           temperature=0.0)
    text = (resp.get("text") or "").strip()
    positions = _parse_llm_positions(text)
    candidate_offsets = sorted({pos for pos, _ in candidates})
    snapped = []
    for p in positions:
        if not (0 <= p < len(body)):
            continue
        nearest = min(candidate_offsets, key=lambda c: abs(c - p))
        if abs(nearest - p) <= 200:  # tolerance ≈ one short paragraph
            snapped.append(nearest)
    valid = sorted(set(snapped))
    if not valid or valid[0] != 0:
        valid = [0] + [p for p in valid if p > 0]
    # Hard-cap: if the LLM returned more boundaries than allowed, keep an
    # evenly-spaced subset (always preserving offset 0).
    max_b = _max_boundaries_for(len(body))
    if len(valid) > max_b:
        if verbose:
            print(f"  [source-split] LLM returned {len(valid)} positions, "
                  f"capping to {max_b}")
        step = len(valid) / max_b
        keep_indices = sorted({0} | {int(i * step) for i in range(max_b)})
        valid = [valid[i] for i in keep_indices if i < len(valid)]
    if len(valid) < 2:
        if verbose:
            print(f"  [source-split] LLM returned <2 positions — "
                  f"keeping chunk un-split")
        return [body]
    pieces = _slice_at_positions(body, valid)
    if verbose:
        sizes = [len(p) for p in pieces]
        print(f"  [source-split] split into {len(pieces)} sub-chunks: "
              f"sizes {sizes}")

    # Recursive pass on still-oversized pieces.
    if _depth < _RECURSE_MAX_DEPTH:
        threshold = int(_RECURSE_THRESHOLD_FACTOR * MAX_SUBCHUNK_CHARS)
        recursed = []
        for piece in pieces:
            if len(piece) > threshold:
                if verbose:
                    print(f"  [source-split] piece still {len(piece):,}c "
                          f"> {threshold:,}, recursing")
                sub = _llm_subsplit(piece, client, title=title,
                                    verbose=verbose, _depth=_depth + 1)
                recursed.extend(sub)
            else:
                recursed.append(piece)
        pieces = recursed

    return pieces or [body]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def split_documents(context: str,
                    client=None,
                    llm_split_trigger: int = LLM_SPLIT_TRIGGER,
                    n_workers: int = 8,
                    verbose: bool = False) -> List[Tuple[str, str]]:
    """Split `context` into [(title, body), ...].

    Pipeline:
      1. Word-headed pattern detection (first hit wins). If no pattern fires,
         the whole context is returned as a single chunk titled
         'Whole document'.
      2. For each chunk whose body exceeds `llm_split_trigger` chars, call
         `client` with paragraph-leading-sentence candidates and ask it to
         pick sub-chunk boundaries targeting [MIN_SUBCHUNK_CHARS,
         MAX_SUBCHUNK_CHARS]. Sub-chunks inherit the parent title with a
         '(part i/N)' suffix when split into >1 piece. The LLM calls run
         concurrently across oversized chunks via ThreadPoolExecutor with
         up to `n_workers` workers.
      3. Returns the assembled list, with original chunk ordering preserved.

    `client` should expose `.generate(prompt, max_new_tokens, temperature) ->
    {"text": str, ...}` (e.g. an OpenRouterClient with reasoning disabled).
    If None, oversized chunks are kept as-is (no LLM call).
    """
    matches = _select_boundary_pattern(context)
    if matches is None:
        # Pattern didn't fire — treat whole context as a single level-1
        # chunk. The LLM sub-split step below will still kick in if the
        # body exceeds llm_split_trigger.
        level1 = [("Whole document", context.strip())]
    else:
        level1 = _split_at_boundaries(context, matches)

    # Identify which chunks need LLM sub-splitting; run them concurrently.
    needs_split: List[int] = []
    for idx, (_, body) in enumerate(level1):
        if len(body) > llm_split_trigger and client is not None:
            needs_split.append(idx)

    sub_results: dict = {}
    if needs_split:
        max_workers = min(n_workers, len(needs_split))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {
                ex.submit(_llm_subsplit, level1[idx][1], client,
                          level1[idx][0], verbose): idx
                for idx in needs_split
            }
            for fut in future_to_idx:
                idx = future_to_idx[fut]
                try:
                    sub_results[idx] = fut.result()
                except Exception as e:
                    # API timeout / rate limit / parse error — fall back to
                    # the un-split chunk so one bad call doesn't tank the
                    # whole sample. Caller still gets a usable result.
                    if verbose:
                        print(f"  [source-split] LLM call failed for chunk "
                              f"'{level1[idx][0][:40]}': {type(e).__name__}: {e}")
                    sub_results[idx] = [level1[idx][1]]

    final: List[Tuple[str, str]] = []
    for idx, (title, body) in enumerate(level1):
        if idx in sub_results:
            sub_pieces = sub_results[idx]
            if len(sub_pieces) == 1:
                final.append((title, sub_pieces[0]))
            else:
                for i, piece in enumerate(sub_pieces):
                    final.append(
                        (f"{title} (part {i + 1}/{len(sub_pieces)})", piece))
        else:
            final.append((title, body))

    safe: List[Tuple[str, str]] = []
    hard_ceiling = 5 * MAX_SUBCHUNK_CHARS
    for title, body in final:
        if len(body) <= hard_ceiling:
            safe.append((title, body))
            continue
        pieces = _force_split_deterministic(body, MAX_SUBCHUNK_CHARS)
        if len(pieces) == 1:
            safe.append((title, pieces[0]))
        else:
            for i, p in enumerate(pieces):
                safe.append(
                    (f"{title} (force {i + 1}/{len(pieces)})", p))
    final = safe

    # Final cleanup: merge any chunk below MIN_SUBCHUNK_CHARS into a
    # neighbour so wasteful tiny chunks (TOC fragments, recursion tails,
    # naturally short clauses) get consolidated rather than emitted as
    # standalone summary calls. Title is concatenated to preserve source
    # traceability.
    final = _merge_tiny_chunks(final, MIN_SUBCHUNK_CHARS)
    return final


def make_default_client(api_key: Optional[str] = None,
                        model: str = "deepseek/deepseek-v4-flash"):
    """Convenience constructor for a fast no-thinking client suitable for
    sub-split decisions."""
    from client import OpenRouterClient
    return OpenRouterClient(model, api_key=api_key, disable_reasoning=True)
