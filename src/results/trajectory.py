"""Trajectory JSON persistence — one file per (task, thread).

Filename pattern:
- with thread_id (normal pass@N path): `{env_id}_t{thread_id}_{YYYYMMDD_HHMMSS}.json`
- without thread_id (single-thread callers): `{env_id}_{YYYYMMDD_HHMMSS}.json`

`/` in env_id is sanitized to `_`.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from base.engine.logs import logger
from benchmark.common.env import BasicInfo
from benchmark.common.runner import StepRecord, LevelResult

def save_trajectory(
    trajectory_dir: Path,
    info: BasicInfo,
    result: LevelResult,
    history: List[StepRecord],
    planner_model: str,
    implementer_models: List[str],
    thread_id: Optional[int] = None,
) -> None:
    """Save detailed trajectory JSON for one (task, thread)."""
    try:
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        attempts = []
        for i, record in enumerate(history):
            action_data = record.action
            action_name = action_data.get("action")
            result_data = action_data.get("result", {})

            attempt = {
                "attempt": i + 1,
                "subtask_history": action_data.get("subtask_history", ""),
                "planner_agent": {
                    "action": action_name,
                    "params": action_data.get("params", {}),
                    "raw_input": record.raw_input,
                    "raw_response": record.raw_response,
                },
            }

            if action_name == "delegate_task":
                attempt["implementer_agent"] = {
                    "model": result_data.get("model"),
                    "tools_assigned": result_data.get("tools_assigned", []),
                    "steps": result_data.get("steps_taken", 0),
                    "cost": result_data.get("cost", 0.0),
                    "finish_result": result_data.get("finish_result"),
                    "trace": result_data.get("trace", []),
                }
            elif action_name == "submit":
                attempt["submit_result"] = {
                    "success": result_data.get("success"),
                    "reward": result_data.get("reward"),
                }

            attempts.append(attempt)

        trajectory = {
            "task_id": info.env_id,
            "instruction": info.instruction,
            "metadata": info.meta_data,
            "planner_model": planner_model,
            "implementer_models": implementer_models,
            "total_reward": result.total_reward,
            "success": result.total_reward > 0,
            "done": result.done,
            "total_attempts": len(history),
            "total_cost": result.cost,
            "timestamp": result.timestamp,
            "attempts": attempts,
        }

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if thread_id is not None:
            filename = f"{info.env_id}_t{int(thread_id)}_{ts}.json"
        else:
            filename = f"{info.env_id}_{ts}.json"
        filename = filename.replace("/", "_")
        filepath = trajectory_dir / filename

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(trajectory, f, indent=2, ensure_ascii=False)

        logger.info(f"[SWEBenchOrchestra] Trajectory saved to {filepath}")
    except Exception as e:
        logger.error(f"[SWEBenchOrchestra] Failed to save trajectory: {e}")
