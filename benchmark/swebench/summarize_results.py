"""Summarize SWE-bench result CSVs with zero-attempt diagnostics.

Usage:
    python benchmark/swebench/summarize_results.py workspace/logs
    python benchmark/swebench/summarize_results.py path/to/results.csv

The reporter separates true task outcomes from infrastructure rows where no
solver entered the agent loop, and from wall-clock-cap rows where partial
trajectories exist but the aggregate CSV row could not use a returned
LevelResult.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TRUE_VALUES = {"1", "true", "yes"}


@dataclass
class ResultRow:
    folder: Path
    task_id: str
    success: bool
    attempts: float
    cost: float
    wall_clock_seconds: float
    forced_cap: bool
    n_solvers_total: float
    n_solvers_passed: float


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: object) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def _find_csvs(paths: Iterable[Path]) -> list[Path]:
    csvs: list[Path] = []
    for path in paths:
        if path.is_file() and path.name == "results.csv":
            csvs.append(path)
        elif path.is_dir():
            csvs.extend(sorted(path.rglob("results.csv")))
    return sorted(dict.fromkeys(csvs))


def _load_rows(csv_paths: Iterable[Path]) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                task_id = raw.get("task_id") or raw.get("id")
                if not task_id:
                    continue
                rows.append(
                    ResultRow(
                        folder=csv_path.parent,
                        task_id=task_id,
                        success=_to_bool(raw.get("success")),
                        attempts=_to_float(raw.get("attempts")),
                        cost=_to_float(raw.get("cost")),
                        wall_clock_seconds=_to_float(raw.get("wall_clock_seconds")),
                        forced_cap=_to_bool(raw.get("forced_cap")),
                        n_solvers_total=_to_float(raw.get("n_solvers_total"), 1.0),
                        n_solvers_passed=_to_float(raw.get("n_solvers_passed")),
                    )
                )
    return rows


def _trajectory_stats(folder: Path, task_id: str) -> tuple[int, int, bool]:
    traj_dir = folder / "trajectories"
    if not traj_dir.exists():
        return (0, 0, False)

    safe_id = task_id.replace("/", "_")
    traj_count = 0
    total_attempts = 0
    has_cost = False
    for path in traj_dir.glob(f"{safe_id}_t*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        traj_count += 1
        attempts = data.get("total_attempts")
        if attempts is None:
            attempts = len(data.get("attempts") or [])
        total_attempts += int(_to_float(attempts))
        has_cost = has_cost or _to_float(data.get("total_cost")) > 0
    return (traj_count, total_attempts, has_cost)


def _classify_zero_row(row: ResultRow) -> str:
    traj_count, traj_attempts, traj_has_cost = _trajectory_stats(row.folder, row.task_id)
    if row.attempts > 0:
        return "nonzero"
    if row.cost == 0 and row.wall_clock_seconds == 0 and traj_count == 0:
        return "infra_zero"
    if row.forced_cap or row.wall_clock_seconds > 0:
        if traj_count > 0 or row.cost > 0 or traj_has_cost:
            if traj_attempts > 0:
                return "cap_partial_with_attempts"
            return "cap_partial_no_completed_attempt"
    if row.cost > 0 or traj_count > 0 or traj_has_cost:
        return "llm_zero_step"
    return "zero_other"


def _pct(num: float, den: float) -> str:
    if den <= 0:
        return "0.00%"
    return f"{(num / den) * 100:.2f}%"


def summarize(csv_paths: list[Path]) -> None:
    rows = _load_rows(csv_paths)
    by_task: dict[str, list[ResultRow]] = defaultdict(list)
    for row in rows:
        by_task[row.task_id].append(row)

    success_rows = sum(1 for row in rows if row.success)
    nonzero_rows = [row for row in rows if row.attempts > 0]
    success_nonzero_rows = sum(1 for row in nonzero_rows if row.success)

    task_success = {task: any(row.success for row in task_rows) for task, task_rows in by_task.items()}
    task_has_nonzero = {
        task: any(row.attempts > 0 for row in task_rows)
        for task, task_rows in by_task.items()
    }
    only_zero_tasks = [task for task, has_nonzero in task_has_nonzero.items() if not has_nonzero]

    threads_passed = sum(row.n_solvers_passed for row in rows)
    threads_total = sum(row.n_solvers_total for row in rows)
    threads_passed_nonzero = sum(row.n_solvers_passed for row in nonzero_rows)
    threads_total_nonzero = sum(row.n_solvers_total for row in nonzero_rows)

    zero_classes = Counter(_classify_zero_row(row) for row in rows if row.attempts == 0)

    print("SWE-bench result summary")
    print(f"results.csv files: {len(csv_paths)}")
    print(f"row-level traces: {len(rows)}")
    print(f"row-level success: {success_rows}/{len(rows)} = {_pct(success_rows, len(rows))}")
    print(
        "row-level success excluding attempts=0: "
        f"{success_nonzero_rows}/{len(nonzero_rows)} = {_pct(success_nonzero_rows, len(nonzero_rows))}"
    )
    print(
        "avg@1 from CSV rows: "
        f"{threads_passed:.0f}/{threads_total:.0f} = {_pct(threads_passed, threads_total)}"
    )
    print(
        "avg@1 excluding attempts=0 rows: "
        f"{threads_passed_nonzero:.0f}/{threads_total_nonzero:.0f} = "
        f"{_pct(threads_passed_nonzero, threads_total_nonzero)}"
    )
    print()
    print(f"task-level unique tasks: {len(by_task)}")
    task_success_count = sum(1 for ok in task_success.values() if ok)
    print(
        "task-level success: "
        f"{task_success_count}/{len(by_task)} = {_pct(task_success_count, len(by_task))}"
    )
    print(f"task-level only attempts=0: {len(only_zero_tasks)}")
    only_zero_set = set(only_zero_tasks)
    nonzero_task_count = len(by_task) - len(only_zero_tasks)
    nonzero_task_success_count = sum(
        1 for task, ok in task_success.items()
        if task not in only_zero_set and ok
    )
    print(
        "task-level success excluding only-zero tasks: "
        f"{nonzero_task_success_count}/{nonzero_task_count} = "
        f"{_pct(nonzero_task_success_count, nonzero_task_count)}"
    )
    print()
    print("attempts=0 row classes:")
    for name, count in sorted(zero_classes.items()):
        print(f"  {name}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more results.csv files or directories containing results.csv files.",
    )
    args = parser.parse_args()

    csv_paths = _find_csvs(args.paths)
    if not csv_paths:
        parser.error("no results.csv files found")
    summarize(csv_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
