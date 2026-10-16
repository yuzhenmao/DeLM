"""Common execution metrics; ungraded submissions retain unknown correctness."""
from runtime.config import PROTOCOLS
from runtime.delm_timing import require_number, task_start
from runtime import runtime_support as runtime


def summarize(run):
    """Report complete records only; failures retain times and usage."""
    rows = run["workers"]
    count = run["count"]
    runtime.validate_records(run)
    roster = [f"agent-{i}" for i in range(1, count + 1)]
    if (run.get("startup_protocol") != PROTOCOLS[count] or len(rows) != count
            or sorted(r.get("actor", "") for r in rows) != roster):
        raise ValueError("Expected the declared distinct worker records")
    if any(not row.get("completed") or row.get("process_returncode") != 0 for row in rows):
        raise ValueError("Not all workers completed")
    starts = [task_start(row, require_number(run, "started_ns")) for row in rows]
    start = ((run["release_ns"] - run["started_ns"]) / 1e9
             if run.get("timing_policy") == "synchronized" else min(starts))
    agents, totals = [], dict(input_tokens=0, cached_input_tokens=0, output_tokens=0, total_tokens=0)
    for row, worker_start in zip(rows, starts):
        finish = require_number(row, "agent_result_received_seconds")
        ungraded = run.get("evaluator") == "review" and row.get("correct") is None
        if finish < worker_start or (not ungraded and type(row.get("correct")) is not bool):
            raise ValueError("Missing grade or inconsistent completion time")
        if run.get("timing_policy") == "synchronized" and finish - start > run["timeout_seconds"]:
            raise ValueError("Submission arrived after the common deadline")
        if row.get("grading", {}).get("verifier_timeout") or row.get("usage_agrees") is not True:
            raise ValueError("Incomplete grading or inconsistent usage")
        for key in totals:
            value = require_number(row["native_usage"], key)
            if type(value) is not int:
                raise ValueError("Token counts must be integers")
            totals[key] += value
        usage = row["native_usage"]
        if usage["cached_input_tokens"] > usage["input_tokens"] or usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
            raise ValueError("Invalid native token accounting")
        agents.append(dict(actor=row["actor"], execution_seconds=finish - start, correct=row["correct"]))
    successful = [r["execution_seconds"] for r in agents if r["correct"]]
    totals["uncached_input_tokens"] = totals["input_tokens"] - totals["cached_input_tokens"]
    summary = dict(task=run["case"], count=count, model=run["model"], effort=run["effort"],
                workers=agents, passing_workers=len(successful),
                first_success_seconds=min(successful) if successful else None,
                mean_worker_seconds=sum(r["execution_seconds"] for r in agents) / count,
                last_worker_seconds=max(r["execution_seconds"] for r in agents), tokens=totals)
    summary["harness"] = run["harness"]
    summary["timing_policy"] = run.get("timing_policy", "per-worker")
    if any(r["correct"] is None for r in agents):
        summary["passing_workers"] = None
        summary.pop("first_success_seconds")
    if run["harness"] == "claude-code":
        summary["cost_usd"] = sum(r["native_usage"]["cost_usd"] for r in rows)
        totals["cache_creation_input_tokens"] = sum(r["native_usage"]["cache_creation_input_tokens"] for r in rows)
    return summary
