"""
results.csv row writer.
"""
from __future__ import annotations

import asyncio
import csv
from pathlib import Path

from base.engine.logs import logger
from benchmark.common.runner import LevelResult


_FIELDNAMES = [
    "task_id", "model",
    "oracle_resolved",            # pass@N oracle (any thread resolved) — was "success"
    "success",                    # kept = oracle_resolved, for backward compat with old parsers
    "avg1_passed",                # # of threads that resolved (numerator of avg@1 for this task)
    "reward", "attempts", "cost", "timestamp",
    "input_tokens", "output_tokens",
    "delegate_count", "submit_count", "error_count", "implementer_steps_total",
    "forced_cap", "forced_submit",
    "wall_clock_seconds", "active_llm_time_seconds",
    "n_solvers_total", "n_solvers_passed", "winning_thread_id",
]

async def save_csv_row(
    csv_path: Path,
    lock: asyncio.Lock,
    task_id: str,
    result: LevelResult,
) -> None:
    """Append one task's summary row to results.csv. Writes a header on
    first call (when the file is missing or empty)."""
    async with lock:
        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            need_header = not csv_path.exists() or csv_path.stat().st_size == 0
            if need_header:
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
                    writer.writeheader()

            oracle_resolved = result.total_reward > 0   # pass@N oracle (any thread)
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
                writer.writerow({
                    "task_id": task_id,
                    "model": result.model,
                    "oracle_resolved": oracle_resolved,
                    "success": oracle_resolved,
                    "avg1_passed": getattr(result, "n_solvers_passed", 0),
                    "reward": f"{result.total_reward:.4f}",
                    "attempts": result.steps,
                    "cost": f"{result.cost:.6f}",
                    "timestamp": result.timestamp,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "delegate_count": result.delegate_count,
                    "submit_count": result.submit_count,
                    "error_count": result.error_count,
                    "implementer_steps_total": result.implementer_steps_total,
                    "forced_cap": result.forced_cap,
                    "forced_submit": result.forced_submit,
                    "wall_clock_seconds": f"{result.wall_clock_seconds:.2f}",
                    "active_llm_time_seconds": f"{result.active_llm_time_seconds:.2f}",
                    "n_solvers_total": result.n_solvers_total,
                    "n_solvers_passed": result.n_solvers_passed,
                    "winning_thread_id": result.winning_thread_id,
                })
        except Exception as e:
            logger.error(f"[SWEBenchOrchestra] Failed to save CSV: {e}")
