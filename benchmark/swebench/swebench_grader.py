from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"

PASSED, FAILED, ERROR, SKIPPED, XFAIL, XPASS = (
    "PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS"
)

# ---------------------------------------------------------------------------
# Try the official package first.
# ---------------------------------------------------------------------------
USING_OFFICIAL = False
try:
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER as _OFFICIAL_PARSERS
    from swebench.harness.grading import (
        get_eval_tests_report as _official_eval_report,
        get_resolution_status as _official_resolution,
    )
    from swebench.harness.constants import (
        FAIL_TO_PASS as _K_F2P,
        PASS_TO_PASS as _K_P2P,
        ResolvedStatus as _ResolvedStatus,
        FAIL_ONLY_REPOS as _FAIL_ONLY_REPOS,
        EvalType as _EvalType,
        START_TEST_OUTPUT as _OFFICIAL_START,
        END_TEST_OUTPUT as _OFFICIAL_END,
    )
    START_TEST_OUTPUT = _OFFICIAL_START
    END_TEST_OUTPUT = _OFFICIAL_END
    USING_OFFICIAL = True
except Exception:  # pragma: no cover - exercised only without swebench installed
    _OFFICIAL_PARSERS = None


# ============================================================================
# Vendored fallback parsers (used only if the official package is unavailable).
# Faithful to swebench/harness/log_parsers/python.py.
# ============================================================================

def _v_parse_pytest(log: str) -> Dict[str, str]:
    """status-first ``STATUS <nodeid> [- msg]``; also tolerate status-last."""
    out: Dict[str, str] = {}
    statuses = (PASSED, FAILED, ERROR, SKIPPED, XFAIL, XPASS)
    for raw in log.split("\n"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw).strip()  # strip ANSI
        if any(line.startswith(s) for s in statuses):
            norm = line.replace(" - ", " ") if line.startswith(FAILED) else line
            parts = norm.split()
            if len(parts) >= 2:
                out.setdefault(parts[1], parts[0])
        elif any(line.endswith(s) for s in statuses):
            parts = line.split()
            if len(parts) >= 2:
                out.setdefault(parts[0], parts[1])
    return out


def _v_parse_seaborn(log: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith(FAILED):
            p = line.split()
            if len(p) >= 2:
                out[p[1]] = FAILED
        elif f" {PASSED} " in line:
            p = line.split()
            if len(p) >= 2 and p[1] == PASSED:
                out[p[0]] = PASSED
        elif line.startswith(PASSED):
            p = line.split()
            if len(p) >= 2:
                out[p[1]] = PASSED
    return out


def _v_parse_django(log: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    suffixes = [(" ... ok", PASSED), (" ... OK", PASSED), (" ... skipped", SKIPPED),
                (" ... FAIL", FAILED), (" ... ERROR", ERROR)]
    for raw in log.split("\n"):
        line = raw.strip()
        if "Testing against Django installed" in line:
            continue
        hit = False
        for suf, st in suffixes:
            idx = line.find(suf)
            if idx != -1:
                name = line[:idx].strip()
                if name:
                    out[name] = st
                hit = True
                break
        if hit:
            continue
        if line.startswith("FAIL:"):
            p = line.split()
            if len(p) >= 2:
                out.setdefault(p[1], FAILED)
        elif line.startswith("ERROR:"):
            p = line.split()
            if len(p) >= 2:
                out.setdefault(p[1], ERROR)
    return out


def _v_parse_sympy(log: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for match in re.findall(r"(_*) (.*)\.py:(.*) (_*)", log):
        out[f"{match[1]}.py:{match[2]}"] = FAILED
    for raw in log.split("\n"):
        line = raw.strip()
        if line.startswith("test_"):
            if line.endswith(" E"):
                out[line.split()[0]] = ERROR
            elif line.endswith(" F"):
                out[line.split()[0]] = FAILED
            elif line.endswith(" ok"):
                out[line.split()[0]] = PASSED
    return out


_VENDORED_PARSERS = {
    "django/django": _v_parse_django,
    "sympy/sympy": _v_parse_sympy,
    "mwaskom/seaborn": _v_parse_seaborn,
}
_VENDORED_PASS = {PASSED, XFAIL}
_VENDORED_FAIL = {FAILED, ERROR}


# ============================================================================
# Public API — dispatches to official package when present, vendored otherwise.
# ============================================================================

def extract_test_region(test_output: str) -> str:
    start = test_output.find(START_TEST_OUTPUT)
    end = test_output.find(END_TEST_OUTPUT)
    if start != -1 and end != -1 and end > start:
        return test_output[start:end]
    return test_output


def parse_log(repo: str, test_log: str) -> Dict[str, str]:
    if USING_OFFICIAL:
        parser = _OFFICIAL_PARSERS.get(repo)
        if parser is None:
            parser = _OFFICIAL_PARSERS.get("pytest-dev/pytest")  # pytest default
        return parser(test_log, None)  # official py parsers ignore test_spec
    return _VENDORED_PARSERS.get(repo, _v_parse_pytest)(test_log)


def build_report(
    repo: str,
    status_map: Dict[str, str],
    fail_to_pass: Optional[List[str]],
    pass_to_pass: Optional[List[str]],
) -> Dict[str, Dict[str, List[str]]]:
    f2p = list(fail_to_pass or [])
    p2p = list(pass_to_pass or [])
    if USING_OFFICIAL:
        eval_type = (
            _EvalType.FAIL_ONLY if repo in _FAIL_ONLY_REPOS else _EvalType.PASS_AND_FAIL
        )
        return _official_eval_report(
            status_map, {_K_F2P: f2p, _K_P2P: p2p}, eval_type=eval_type
        )
    # Vendored: replicate official check_pass_and_fail semantics exactly.
    report = {
        "FAIL_TO_PASS": {"success": [], "failure": []},
        "PASS_TO_PASS": {"success": [], "failure": []},
    }
    for key, tests in (("FAIL_TO_PASS", f2p), ("PASS_TO_PASS", p2p)):
        for t in tests:
            st = status_map.get(t)
            if st in _VENDORED_PASS:
                report[key]["success"].append(t)
            elif st is None or st in _VENDORED_FAIL:
                report[key]["failure"].append(t)
            # SKIPPED / other -> dropped from both, as in the official harness
    return report


def is_resolved(report: Dict[str, Dict[str, List[str]]]) -> bool:
    if USING_OFFICIAL:
        return _official_resolution(report) == _ResolvedStatus.FULL.value
    return (
        len(report["FAIL_TO_PASS"]["failure"]) == 0
        and len(report["PASS_TO_PASS"]["failure"]) == 0
    )


def grade(
    repo: str,
    test_output: str,
    fail_to_pass: Optional[List[str]],
    pass_to_pass: Optional[List[str]],
) -> Tuple[bool, Dict[str, Dict[str, List[str]]]]:
    """raw test_output -> (resolved, report). Used by the executor + CLI."""
    status_map = parse_log(repo, extract_test_region(test_output))
    # Mirror official get_logs_eval fallback: if nothing parsed between the
    # markers, retry on the whole log (output sometimes escapes the markers).
    if not status_map:
        status_map = parse_log(repo, test_output)
    report = build_report(repo, status_map, fail_to_pass, pass_to_pass)
    return is_resolved(report), report


def repo_from_instance_id(instance_id: str) -> str:
    """``scikit-learn__scikit-learn-13439`` -> ``scikit-learn/scikit-learn``."""
    if "__" not in instance_id:
        return instance_id
    owner, rest = instance_id.split("__", 1)
    return f"{owner}/{rest.rsplit('-', 1)[0]}"


def _iter_patch_file_blocks(patch: str):
    matches = list(re.finditer(r"^diff --git a/(.*?) b/(.*?)$", patch or "", re.M))
    for i, match in enumerate(matches):
        block_start = match.start()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(patch)
        yield match.group(1), match.group(2), patch[block_start:block_end]


def _get_modified_files(patch: str) -> List[str]:
    return [
        source
        for source, _target, block in _iter_patch_file_blocks(patch)
        if not re.search(r"^--- /dev/null\s*$", block, re.M)
    ]


def _get_new_files(patch: str) -> List[str]:
    return [
        target
        for _source, target, block in _iter_patch_file_blocks(patch)
        if re.search(r"^--- /dev/null\s*$", block, re.M)
    ]


def _reset_test_commands(base_commit: str, test_patch: str) -> List[str]:
    commands: List[str] = []
    modified_files = _get_modified_files(test_patch)
    new_files = _get_new_files(test_patch)
    if modified_files:
        commands.append(f"git checkout {base_commit} {' '.join(modified_files)}")
    if new_files:
        commands.append(f"rm -f {' '.join(new_files)}")
    return commands


def _patch_legacy_reset_commands(
    eval_list: List[str], base_commit: str, test_patch: str
) -> List[str]:
    """Patch older installed swebench reset commands to current upstream rules."""
    new_files = _get_new_files(test_patch)
    if new_files:
        expected_cleanup = f"rm -f {' '.join(new_files)}"
        if expected_cleanup in eval_list:
            return eval_list

    reset_commands = _reset_test_commands(base_commit, test_patch)
    if not reset_commands:
        return eval_list

    patched: List[str] = []
    for line in eval_list:
        if line.startswith(f"git checkout {base_commit}"):
            patched.extend(reset_commands)
        else:
            patched.append(line)
    return patched


def official_eval_script(
    instance: Dict, repo_directory: str = "/testbed", env_name: str = "testbed"
) -> Optional[str]:
    """Build a current-compatible official harness eval script for an instance.

    Returns the full bash script (version-specific test_cmd, repo eval_commands,
    and the eval-time install/rebuild step) the leaderboard runs, or ``None`` if
    the official package isn't installed or the repo/version isn't in the specs
    (caller should fall back to a local script).

    We call ``make_eval_script_list`` directly and wrap it as ``TestSpec`` does.
    If the installed package is older than current upstream, we patch the
    new-file reset commands locally. We deliberately do NOT use
    ``make_test_spec``, because that also builds the *environment* script, which
    fetches the repo's requirements file over the network — unnecessary here
    (the container image already has the env) and would require a live commit +
    connectivity.

    ``instance`` must be a dict with repo, version, base_commit, test_patch,
    instance_id, FAIL_TO_PASS, PASS_TO_PASS.
    """
    if not USING_OFFICIAL:
        return None
    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
        from swebench.harness.test_spec.create_scripts import make_eval_script_list
        specs = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][str(instance["version"])]
        eval_list = make_eval_script_list(
            instance, specs, env_name, repo_directory,
            instance["base_commit"], instance["test_patch"],
        )
        # The installed swebench package may predate the current upstream fix
        # for newly-created gold test files. Patch only that reset behavior.
        # `pre_install` belongs to image/repo setup, not eval-time replay.
        eval_list = _patch_legacy_reset_commands(
            eval_list, instance["base_commit"], instance["test_patch"]
        )
        return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + eval_list) + "\n"
    except Exception:
        return None


# ============================================================================
# Offline CLI re-grader for already-saved logs
# ============================================================================

_LIST_RE = {
    "f2p_pass": re.compile(r"^F2P passed:\s*(\[.*\])\s*$", re.M),
    "f2p_fail": re.compile(r"^F2P failed:\s*(\[.*\])\s*$", re.M),
    "p2p_pass": re.compile(r"^P2P passed:\s*(\[.*\])\s*$", re.M),
    "p2p_fail": re.compile(r"^P2P failed:\s*(\[.*\])\s*$", re.M),
}


def _lit(text: str, rx: "re.Pattern") -> List[str]:
    m = rx.search(text)
    if not m:
        return []
    try:
        v = ast.literal_eval(m.group(1))
        return list(v) if isinstance(v, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def _gold_from_results_log(text: str) -> Tuple[List[str], List[str]]:
    """Recover the full gold F2P/P2P sets = union(passed, failed) the old grader
    partitioned. The union is reliable even when the partition was wrong."""
    return (
        _lit(text, _LIST_RE["f2p_pass"]) + _lit(text, _LIST_RE["f2p_fail"]),
        _lit(text, _LIST_RE["p2p_pass"]) + _lit(text, _LIST_RE["p2p_fail"]),
    )


def regrade_logs(logs_dir: Path) -> int:
    results_logs = sorted(logs_dir.rglob("test_results.log"))
    if not results_logs:
        print(f"No test_results.log found under {logs_dir}", file=sys.stderr)
        return 1

    by_instance: Dict[str, List[Tuple[str, bool, bool]]] = defaultdict(list)
    thread_total = thread_old_true = thread_new_true = 0
    flips: List[str] = []
    skipped = 0

    for rlog in results_logs:
        text = rlog.read_text(errors="replace")
        m_inst = re.search(r"Instance:\s*(\S+)", text)
        out_file = rlog.parent / "test_output.txt"
        if not m_inst or not out_file.exists():
            skipped += 1
            continue
        instance_id = m_inst.group(1)
        old_resolved = bool(re.search(r"Resolved:\s*True", text))
        f2p, p2p = _gold_from_results_log(text)
        new_resolved, report = grade(
            repo_from_instance_id(instance_id),
            out_file.read_text(errors="replace"), f2p, p2p,
        )

        thread_total += 1
        thread_old_true += int(old_resolved)
        thread_new_true += int(new_resolved)
        by_instance[instance_id].append((rlog.parent.parent.name, old_resolved, new_resolved))
        if old_resolved and not new_resolved:
            f2pf = report["FAIL_TO_PASS"]["failure"]
            why = f"F2P failing={f2pf[:2]}" if f2pf else f"P2P regressions={len(report['PASS_TO_PASS']['failure'])}"
            flips.append(f"{rlog.parent.parent.name}: {why}")

    n = len(by_instance)
    passN_old = sum(1 for v in by_instance.values() if any(o for _, o, _ in v))
    passN_new = sum(1 for v in by_instance.values() if any(x for _, _, x in v))
    avg_old = sum(sum(o for _, o, _ in v) / len(v) for v in by_instance.values())
    avg_new = sum(sum(x for _, _, x in v) / len(v) for v in by_instance.values())

    backend = "OFFICIAL swebench package" if USING_OFFICIAL else "vendored fallback port"
    print(f"\n{'='*64}\nCorrected SWE-bench grading [{backend}]\n{logs_dir}\n{'='*64}")
    print(f"thread-level verdicts re-graded : {thread_total}  (skipped {skipped})")
    print(f"  Resolved=True  OLD (buggy) : {thread_old_true}")
    print(f"  Resolved=True  NEW (fixed) : {thread_new_true}")
    print(f"  flipped True->False        : {len(flips)}")
    if n:
        print(f"\ninstances : {n}")
        print(f"  pass@N OLD={passN_old}/{n} ({passN_old/n*100:.1f}%)   NEW={passN_new}/{n} ({passN_new/n*100:.1f}%)")
        print(f"  avg@1  OLD={avg_old/n*100:.1f}%   NEW={avg_new/n*100:.1f}%")
    if flips:
        print(f"\nfalse positives removed ({len(flips)}), first 25:")
        for line in flips[:25]:
            print(f"  - {line}")
    return 0


def _main() -> int:
    ap = argparse.ArgumentParser(description="Re-grade saved SWE-bench logs correctly.")
    ap.add_argument("logs_dir", type=Path, help="Dir to walk for */logs/test_results.log")
    args = ap.parse_args()
    return regrade_logs(args.logs_dir)


if __name__ == "__main__":
    raise SystemExit(_main())
