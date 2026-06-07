from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---- canonicalization -----------------------------------------------------

def _step_action_signature(action: Any) -> str:
    """Canonical signature for ONE ImplementerAgent step. Mirrors
    `local_agent_runner._action_signature` so divergence metrics
    line up with the runner's own loop-detection heuristic.

    `action` is a dict (the StepRecord.action serialized to JSON).
    Returns "" on malformed input — the caller filters these out.
    """
    if not isinstance(action, dict):
        return ""
    a = action.get("action", "")
    p = action.get("params", {}) or {}
    if isinstance(p, dict) and "command" in p:
        return f"{a}:{(p.get('command') or '').strip()}"
    if isinstance(p, dict):
        scalars = {
            k: v for k, v in p.items()
            if k not in ("reasoning", "notes") and isinstance(v, (str, int, float, bool))
        }
        return f"{a}:{sorted(scalars.items())}"
    return f"{a}:{str(p)[:160]}"

# ---- extraction -----------------------------------------------------------

def _iter_implementer_steps(level_result: Any):
    """Yield each serialized ImplementerAgent step from a thread's LevelResult.

    Walks the PlannerAgent history: every `delegate_task` step's
    `info["trace"]` is the ImplementerAgent's inner step list. Steps from
    `submit`/`error` PlannerAgent actions are skipped.
    """
    trace = getattr(level_result, "trace", None)
    if not trace:
        return
    for planner_step in trace:
        info = getattr(planner_step, "info", None)
        if not isinstance(info, dict):
            continue
        implementer_trace = info.get("trace")
        if not isinstance(implementer_trace, list):
            continue
        for implementer_step in implementer_trace:
            if isinstance(implementer_step, dict):
                yield implementer_step

def extract_command_sequence(level_result: Any) -> List[str]:
    """Per-thread ImplementerAgent command signatures, in execution order."""
    sigs: List[str] = []
    for s in _iter_implementer_steps(level_result):
        action = s.get("action")
        sig = _step_action_signature(action)
        if sig:
            sigs.append(sig)
    return sigs

# Path-like substring inside an action's command. Restricted to .py so we
# don't flag generic `/testbed` directory tokens. Captures most pytest /
# repo navigation work without over-firing.
_PY_FILE_RE = re.compile(r"([A-Za-z0-9_./\-]+\.py)")
# `def name(` or `class name(` inside an action's command/params text.
_DEF_RE = re.compile(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
# Pytest-style path-or-target token: `file.py` or `file.py::Class::test_x`.
# Used as the candidate set for `_looks_like_test_path`. We deliberately
# do NOT regex `.*test.*\.py` directly: SWE-bench commands typically run
# under `/testbed`, so a substring-only check classifies ordinary source
# paths like `/testbed/django/http/response.py` as test targets [GPT
# ].
_PY_TARGET_RE = re.compile(r"([A-Za-z0-9_./\-]+\.py(?:::[A-Za-z0-9_:]+)?)")

def _looks_like_test_path(path: str) -> bool:
    """is `path` plausibly a TEST target?

    A path qualifies if its basename starts with `test_` / ends with
    `_test.py`, OR if any non-basename directory in the path is exactly
    `tests` or `test`. Plain substring containment ("testbed") does NOT
    qualify — that's what produced the false positive in the original
    `_TEST_PATH_RE`.
    """
    bare = path.split("::", 1)[0]
    parts = bare.split("/")
    if not parts:
        return False
    basename = parts[-1]
    if basename.startswith("test_") and basename.endswith(".py"):
        return True
    if basename.endswith("_test.py") and basename != "_test.py":
        return True
    # Path-component check: only directory segments count, not basename.
    return any(p in ("tests", "test") for p in parts[:-1])

def _params_text(step: Dict[str, Any]) -> str:
    """Concatenate the per-step strings the extractors regex over.

    We deliberately include `command`, `path`, `file_text` and other
    common params — different ImplementerAgent tools use different field names
    for the target. Joining them lets one regex pass cover all of them.
    """
    a = step.get("action") or {}
    if not isinstance(a, dict):
        return ""
    p = a.get("params") or {}
    if not isinstance(p, dict):
        return str(p)
    parts: List[str] = []
    for k in ("command", "path", "file_path", "file_text", "old_str", "new_str", "view_path"):
        v = p.get(k)
        if isinstance(v, str):
            parts.append(v)
    return "\n".join(parts)

def extract_file_sequence(level_result: Any) -> List[str]:
    """First .py path seen in each ImplementerAgent step's params text, in order.

    Steps that don't mention a .py path contribute the empty string —
    keeping positional alignment with the command sequence so callers
    can index step N's file alongside step N's command if needed.
    """
    files: List[str] = []
    for s in _iter_implementer_steps(level_result):
        text = _params_text(s)
        m = _PY_FILE_RE.search(text) if text else None
        files.append(m.group(1) if m else "")
    return files

def extract_function_sequence(level_result: Any) -> List[str]:
    """First def/class name seen in each step's params text, in order."""
    out: List[str] = []
    for s in _iter_implementer_steps(level_result):
        text = _params_text(s)
        m = _DEF_RE.search(text) if text else None
        out.append(m.group(1) if m else "")
    return out

def extract_test_sequence(level_result: Any) -> List[str]:
    """First test-file token (per `_looks_like_test_path`) seen in each
    step's params text, in order. Scans all .py path candidates and
    returns the first that passes the test-path heuristic."""
    out: List[str] = []
    for s in _iter_implementer_steps(level_result):
        text = _params_text(s)
        tok = ""
        if text:
            for m in _PY_TARGET_RE.finditer(text):
                cand = m.group(1)
                if _looks_like_test_path(cand):
                    tok = cand
                    break
        out.append(tok)
    return out

# ---- pairwise metrics -----------------------------------------------------

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(s for s in a if s), set(s for s in b if s)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def _common_over_min(a: List[str], b: List[str]) -> float:
    sa, sb = set(s for s in a if s), set(s for s in b if s)
    m = min(len(sa), len(sb))
    if m == 0:
        return 0.0
    return len(sa & sb) / m

def _shared_prefix_len(a: List[str], b: List[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x == y and x != "":
            n += 1
        else:
            break
    return n

def _first_distinct(a: List[str], b: List[str]) -> Optional[Dict[str, Any]]:
    """First index where the two sequences disagree on a non-empty token.

    Returns {index, a, b} or None if they never disagree within the
    overlap. Empty tokens (steps that didn't mention a file/function)
    are skipped — they're noise, not divergence signals.
    """
    for i, (x, y) in enumerate(zip(a, b)):
        if x == "" or y == "":
            continue
        if x != y:
            return {"index": i, "a": x, "b": y}
    return None

def compute_pairwise_metrics(per_thread: Dict[int, Any]) -> Dict[str, Any]:
    """Compute per-pair divergence metrics from a {thread_id: LevelResult}.

    Returns a dict with:
      - per_thread_lengths: {tid: command_seq_len}
      - pairs: [{a, b, jaccard, common_over_min, shared_prefix_len,
                 first_distinct_file, first_distinct_function,
                 first_distinct_test}, ...]
    Empty / single-thread inputs return a stub with empty pairs (no
    crash). All extractors guard against malformed steps.
    """
    tids = sorted(per_thread.keys())
    seqs: Dict[int, Dict[str, List[str]]] = {}
    for tid in tids:
        lr = per_thread[tid]
        seqs[tid] = {
            "cmd": extract_command_sequence(lr),
            "file": extract_file_sequence(lr),
            "func": extract_function_sequence(lr),
            "test": extract_test_sequence(lr),
        }

    per_thread_lengths = {tid: len(seqs[tid]["cmd"]) for tid in tids}
    pairs: List[Dict[str, Any]] = []
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            a, b = tids[i], tids[j]
            sa, sb = seqs[a], seqs[b]
            pairs.append({
                "a": a,
                "b": b,
                "jaccard_cmd": round(_jaccard(sa["cmd"], sb["cmd"]), 4),
                "common_over_min_cmd": round(_common_over_min(sa["cmd"], sb["cmd"]), 4),
                "shared_prefix_len_cmd": _shared_prefix_len(sa["cmd"], sb["cmd"]),
                "first_distinct_file": _first_distinct(sa["file"], sb["file"]),
                "first_distinct_function": _first_distinct(sa["func"], sb["func"]),
                "first_distinct_test": _first_distinct(sa["test"], sb["test"]),
            })
    return {
        "per_thread_command_count": per_thread_lengths,
        "pairs": pairs,
    }

# ---- writer ---------------------------------------------------------------

def write_divergence_summary(out_path: Path, task_id: str,
                             per_thread: Dict[int, Any]) -> None:
    try:
        metrics = compute_pairwise_metrics(per_thread)
        metrics["task_id"] = task_id
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception:
        return
