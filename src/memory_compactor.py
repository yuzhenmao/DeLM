"""
Deterministic memory compactor for SWE-bench ImplementerAgents.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

# Regex set — narrow on purpose so we don't false-positive on prose.
_PY_FILE_RE = re.compile(r"\b([A-Za-z0-9_./\-]+\.py)\b")
_PY_LINE_RE = re.compile(r"\b([A-Za-z0-9_./\-]+\.py):(\d+)\b")
_PYTEST_TARGET_RE = re.compile(
    r"\b((?:[A-Za-z0-9_./\-]+/)?(?:test_[A-Za-z0-9_]+|[A-Za-z0-9_]+_test)\.py(?:::[A-Za-z0-9_:]+)?)\b"
)
_ERROR_CLASS_RE = re.compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception|Warning))\b")
_FAILED_MARKER_RE = re.compile(r"\b(?:FAILED|FAIL|raised|raises|threw|throws)\b", re.IGNORECASE)
_PASSED_MARKER_RE = re.compile(r"\b(?:PASSED|PASS|succeeded|succeeds|ok)\b", re.IGNORECASE)

_EDIT_PARAM_KEYS = ("file_path", "path", "target")
_EDIT_ACTION_NAMES = ("str_replace", "create", "edit", "insert", "delete_lines",
                     "write", "edit_file", "str_replace_editor")
_EDIT_COMMAND_VERBS = _EDIT_ACTION_NAMES
_FIRST_TOKEN_RE = re.compile(r"^([A-Za-z_]+)(?:\s+([^\s\n]+))?")

def _action_command(action: Any) -> str:
    """Extract the bash/command string from an action dict (best effort)."""
    if not isinstance(action, dict):
        return ""
    p = action.get("params") or {}
    if isinstance(p, dict):
        cmd = p.get("command") or ""
        if isinstance(cmd, str):
            return cmd.strip()
    return ""

def _record_text_blob(record: Dict[str, Any]) -> str:
    """Concatenate the searchable text of one record (action params,
    observation, raw_response). Used by the regex extractors so we
    don't have to special-case per-field."""
    parts: List[str] = []
    action = record.get("action") or {}
    if isinstance(action, dict):
        p = action.get("params") or {}
        if isinstance(p, dict):
            for v in p.values():
                if isinstance(v, str):
                    parts.append(v)
        # Notes embedded in action["notes"] (parsed PATCH_SUMMARY etc.)
        for note in (action.get("notes") or []):
            if isinstance(note, dict):
                t = note.get("type") or ""
                c = note.get("content") or ""
                if isinstance(t, str) and isinstance(c, str):
                    parts.append(f"{t} {c}")
    obs = record.get("observation")
    if isinstance(obs, str):
        parts.append(obs)
    elif isinstance(obs, dict):
        for v in obs.values():
            if isinstance(v, str):
                parts.append(v)
    raw = record.get("raw_response")
    if isinstance(raw, str):
        parts.append(raw)
    return "\n".join(parts)

def _extract_edited_files(records: List[Dict[str, Any]]) -> List[str]:
    """Files touched by an edit-style action. Returns .py paths in
    chronological first-seen order, deduped.
    """
    seen: List[str] = []
    for r in records:
        action = r.get("action") or {}
        if not isinstance(action, dict):
            continue
        name = action.get("action", "")
        p = action.get("params") or {}
        if not isinstance(p, dict):
            continue

        # Shape 1: direct edit verb in action.action.
        if name in _EDIT_ACTION_NAMES:
            for key in _EDIT_PARAM_KEYS:
                v = p.get(key)
                if isinstance(v, str) and v.endswith(".py") and v not in seen:
                    seen.append(v)
                    break
            continue

        cmd = p.get("command")
        if not isinstance(cmd, str):
            continue
        first_line = cmd.split("\n", 1)[0].strip()
        m = _FIRST_TOKEN_RE.match(first_line)
        if not m:
            continue
        verb = m.group(1)
        path = m.group(2) or ""
        if verb in _EDIT_COMMAND_VERBS and path.endswith(".py") and path not in seen:
            seen.append(path)
    return seen

def _extract_failing_tests(records: List[Dict[str, Any]]) -> List[str]:
    """Pytest targets that the observation/raw text indicates failed.
    Heuristic: a test path token within ~120 chars of a FAILED-style
    marker in the same record's blob."""
    found: List[str] = []
    for r in records:
        blob = _record_text_blob(r)
        if not blob:
            continue
        for tm in _PYTEST_TARGET_RE.finditer(blob):
            tok = tm.group(1)
            # Window check: a FAILED-like marker within 120 chars on either side.
            start, end = max(0, tm.start() - 120), min(len(blob), tm.end() + 120)
            if _FAILED_MARKER_RE.search(blob[start:end]) and tok not in found:
                found.append(tok)
    return found

def _extract_exception_names(records: List[Dict[str, Any]]) -> List[str]:
    """Distinct exception/error class names found in the records'
    observation/action text. Order = first occurrence."""
    seen: List[str] = []
    for r in records:
        blob = _record_text_blob(r)
        for m in _ERROR_CLASS_RE.finditer(blob):
            n = m.group(1)
            if n not in seen:
                seen.append(n)
    return seen

def _extract_line_refs(records: List[Dict[str, Any]], cap: int = 8) -> List[str]:
    """`file.py:NNN` references in chronological first-seen order, capped."""
    seen: List[str] = []
    for r in records:
        blob = _record_text_blob(r)
        for m in _PY_LINE_RE.finditer(blob):
            tok = f"{m.group(1)}:{m.group(2)}"
            if tok not in seen:
                seen.append(tok)
                if len(seen) >= cap:
                    return seen
    return seen

def _extract_commands_run(records: List[Dict[str, Any]], cap: int = 8) -> List[str]:
    """Trim each command to ~80 chars; chronological order; capped."""
    out: List[str] = []
    for r in records:
        cmd = _action_command(r.get("action"))
        if cmd:
            short = cmd if len(cmd) <= 80 else (cmd[:77] + "...")
            out.append(short)
            if len(out) >= cap:
                break
    return out

def _extract_latest_patch_summary(records: List[Dict[str, Any]]) -> str:
    """Most recent PATCH_SUMMARY note embedded in action['notes'] (if
    parse_implementer_response carried one through). Empty string when none."""
    for r in reversed(records):
        action = r.get("action") or {}
        if not isinstance(action, dict):
            continue
        for note in (action.get("notes") or []):
            if isinstance(note, dict) and note.get("type") == "PATCH_SUMMARY":
                content = note.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return ""

def _extract_unresolved_risks(records: List[Dict[str, Any]]) -> List[str]:
    # Track first-failure index per exception class.
    first_fail_idx: Dict[str, int] = {}
    resolved: set = set()
    for idx, r in enumerate(records):
        blob = _record_text_blob(r)
        if not blob:
            continue
        # Mark exceptions first-seen in failure context.
        for m in _ERROR_CLASS_RE.finditer(blob):
            n = m.group(1)
            if n not in first_fail_idx:
                start, end = max(0, m.start() - 120), min(len(blob), m.end() + 120)
                if _FAILED_MARKER_RE.search(blob[start:end]):
                    first_fail_idx[n] = idx
        # Mark resolved if a later record has the exception name AND a
        # PASSED marker in its window.
        for name in list(first_fail_idx.keys()):
            if name in resolved:
                continue
            if first_fail_idx[name] >= idx:
                continue
            # name appears + passed marker nearby?
            for m in re.finditer(rf"\b{name}\b", blob):
                start, end = max(0, m.start() - 120), min(len(blob), m.end() + 120)
                if _PASSED_MARKER_RE.search(blob[start:end]):
                    resolved.add(name)
                    break
    return [n for n in first_fail_idx if n not in resolved]

def extract_structured_summary(records: List[Dict[str, Any]]) -> str:
    """— Deterministic compactor entry point.

    Given a list of `Memory` records (each a dict with at least
    `action` / `observation` / `thinking` keys), produce a structured
    multi-line summary string. Pure-Python, zero LLM cost, fully
    deterministic. Returns a fixed-shape block so downstream readers
    (the ImplementerAgent's next prompt) can rely on the layout.

    Empty input → returns "" (skip the block entirely).
    """
    if not records:
        return ""

    edited_files = _extract_edited_files(records)
    failing_tests = _extract_failing_tests(records)
    exception_names = _extract_exception_names(records)
    line_refs = _extract_line_refs(records)
    commands_run = _extract_commands_run(records)
    latest_patch = _extract_latest_patch_summary(records)
    unresolved = _extract_unresolved_risks(records)

    any_signal = bool(
        edited_files or failing_tests or exception_names
        or line_refs or commands_run or latest_patch or unresolved
    )
    if not any_signal:
        return (
            f"[Memory summary — no structured signals in "
            f"{len(records)} compressed steps]"
        )

    def _list_line(label: str, items: List[str]) -> str:
        if not items:
            return f"{label}: (none)"
        return f"{label}: {', '.join(items)}"

    lines: List[str] = [
        f"[Memory summary — deterministic extract from {len(records)} compressed steps]",
        _list_line("Edited files", edited_files),
        _list_line("Failing tests", failing_tests),
        _list_line("Exception classes seen", exception_names),
        _list_line("Key line refs", line_refs),
        _list_line("Commands tried (recent)", commands_run),
        _list_line("Failure classes seen", unresolved),
    ]
    if latest_patch:
        lines.append(f"Latest PATCH_SUMMARY: {latest_patch}")
    return "\n".join(lines)
