"""Read-only Markdown reports for the unified n=2/n=4 run schema."""

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

from reporting.records import ROOT, RUNS, read_run, scan_runs

REPORT = ROOT / ".local/reports/results.md"


def cell(value):
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "pass" if value else "fail"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def format_number(value, seconds=False):
    if value is None:
        return "unknown"
    return f"{value:,.3f}" if seconds else f"{value:,}"


def first_success(summary):
    if "first_success_seconds" not in summary:
        return "unknown"
    value = summary["first_success_seconds"]
    return "none" if value is None else format_number(value, seconds=True)


def errors(record):
    return [f"{key}: {value}" for key, value in record.items()
            if value and (key == "error" or key.endswith(("_error", "_errors")))]


def pull_text(kind, p):
    """The transcript text of a files event: registration, expand, apply; None for other kinds."""
    if kind == "body":
        return "attached " + ", ".join(f"{f['path']} ({f['bytes']:,} bytes, {f['change']})" for f in p["files"])
    if kind == "attached":
        return f"listed the files behind {p['id'][:8]}: {', '.join(p['files'])}"
    if kind == "expand":
        return f"expanded {p['id'][:8]} ({'in full' if p.get('shown') == 'full' else 'as diffs'}): {', '.join(p['files'])}"
    if kind == "apply":
        decided = ", ".join(f"{path} {decision}" for path, decision in p["decisions"].items())
        action = ("applied" if p.get("confirmed", True) else "apply granted") if p["applied"] else "apply refused"
        return f"{action} {p['id'][:8]}: {decided}"
    if kind == "apply_result":
        return f"apply {'completed' if p['success'] else 'failed'} {p['id'][:8]}: {len(p['written'])} files written"
    if kind == "apply_error":
        return f"apply of {p['id'][:8]} granted but writing failed: {p['error']}"
    return None


def board_lines(folder, record):
    path = folder / "context-events.json"
    if not path.exists():
        return []
    lines = ["", "## Board", ""]
    try:
        events = json.loads(path.read_text())
        for event in events:
            kind, payload = event["kind"], event["payload"]
            if kind == "read":
                continue
            text = (pull_text(kind, payload) or payload.get("text") or payload.get("summary")
                    or payload.get("subject") or payload.get("reason") or payload.get("item", ""))
            start = record.get("started_ns")
            stamp = f"{(event['time_ns'] - start) / 1e9:.1f}s" if start is not None else "unknown"
            lines.append(f"- {stamp} {cell(event['actor'])} {cell(kind)}: {cell(text)}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        lines.append(f"Board unavailable: {cell(error)}")
    return lines


def run_report(folder):
    run = read_run(folder)
    record, summary = run.record, run.summary
    lines = [f"# {cell(record.get('case', run.folder.parent.name))} "
             f"(N={cell(record.get('count'))}) — {cell(record.get('status'))}", "",
             f"Harness: {cell(record.get('harness'))}; model: {cell(record.get('model'))}; "
             f"effort: {cell(record.get('effort'))}; auth mode: {cell(record.get('auth_mode'))}.", ""]
    if run.error:
        lines.append(f"Record unavailable: {cell(run.error)}")
    if summary:
        lines += [f"Passing workers: {cell(summary.get('passing_workers'))}/{cell(record.get('count'))}; "
                  f"first success: {first_success(summary)} s.", ""]
        tokens = summary.get("tokens") or {}
        lines.append("Tokens: " + "; ".join(f"{key}: {format_number(value)}" for key, value in tokens.items()) + ".")
        if "cost_usd" in summary:
            lines.append(f"Recorded cost (USD): {summary['cost_usd']}.")
    else:
        lines.append("Summary unavailable; run totals and execution times are unknown.")
    lines += ["", "| Agent | Correct | Execution seconds | Input tokens | Output tokens | Transport failures |",
              "|---|---|---:|---:|---:|---:|"]
    workers = record.get("workers", [])
    measured = {w["actor"]: w for w in summary.get("workers", [])}
    for worker in workers:
        actor = worker.get("actor")
        result = measured.get(actor, {})
        usage = worker.get("native_usage") or {}
        transport = [worker.get(k) for k in ("model_transport_failures", "context_command_transport_failures")]
        failures = sum(transport) if all(v is not None for v in transport) else None
        lines.append("| " + " | ".join([
            cell(actor), cell(result.get("correct", worker.get("correct"))),
            format_number(result.get("execution_seconds"), seconds=True),
            format_number(usage.get("input_tokens")), format_number(usage.get("output_tokens")), format_number(failures),
        ]) + " |")
    count = record.get("count")
    if isinstance(count, int) and count != len(workers):
        lines += ["", f"Worker records: {len(workers)} of {count} declared."]
    notes = errors(record)
    for worker in workers:
        notes += [f"{worker.get('actor', 'unknown')} {note}" for note in errors(worker)]
        notes += [f"{worker.get('actor', 'unknown')} grading {note}" for note in errors(worker.get("grading") or {})]
    if notes:
        lines += ["", "## Errors", "", *[f"- {cell(note)}" for note in notes]]
    lines += board_lines(run.folder, record)
    return "\n".join(lines) + "\n"


def results_markdown(runs_root=RUNS, output=REPORT):
    lines = ["# DeLM runs", "", "One row per saved run; unknown values are not counted as zero.", "",
             "| Run | N | Harness | Model | Effort | Auth mode | Status | Passing workers | First success (s) | Input tokens | Output tokens |",
             "|---|---:|---|---|---|---|---|---:|---:|---:|---:|"]
    notes = []
    runs = scan_runs(runs_root)
    for run in runs:
        record, summary = run.record, run.summary
        label = run.folder.relative_to(runs_root).as_posix()
        target = quote(os.path.relpath(run.folder / "run.json", Path(output).parent))
        tokens = summary.get("tokens") or {}
        values = [f"[{cell(label)}]({target})", cell(record.get("count")), cell(record.get("harness")),
                  cell(record.get("model")), cell(record.get("effort")), cell(record.get("auth_mode")),
                  cell(record.get("status")), cell(summary.get("passing_workers")), first_success(summary),
                  format_number(tokens.get("input_tokens")), format_number(tokens.get("output_tokens"))]
        lines.append("| " + " | ".join(values) + " |")
        problems = ([run.error] if run.error else []) + errors(record)
        if not summary:
            problems.append("summary unavailable")
        for worker in record.get("workers", []):
            problems += [f"{worker.get('actor', 'unknown')} {e}" for e in errors(worker)]
        if problems:
            notes.append(f"- {cell(label)}: " + "; ".join(cell(p) for p in problems))
    if not runs:
        lines += ["", "No run folders found."]
    if notes:
        lines += ["", "## Unavailable data and errors", "", *notes]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(args.runs_root.resolve()):
        parser.error("report output must be outside the run records")
    text = results_markdown(args.runs_root, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(args.output)


if __name__ == "__main__":
    main()
