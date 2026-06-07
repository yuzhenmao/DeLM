# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.


"""Incremental Runner: base class with step-by-step trajectory saving."""
from __future__ import annotations

import csv
import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from benchmark.common.env import BasicInfo, Environment
from benchmark.common.runner import LevelResult, Runner, StepRecord
from base.agent.base_agent import BaseAgent
from base.engine.logs import LogLevel, logger


class IncrementalRunner(Runner):
    """
    Runner with trajectory (JSON) and CSV saving.
    
    Features:
    - Saves complete trajectory as JSON after task completion
    - Appends summary row to CSV after each task
    - Structured and readable trajectory format
    
    Subclasses should override run() to add resource cleanup.
    """

    def __init__(self, trajectory_dir: Optional[Path] = None, csv_summary_path: Optional[Path] = None):
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir else None
        self.csv_summary_path = Path(csv_summary_path) if csv_summary_path else None
        self._csv_initialized = False

    async def run(self, agent: BaseAgent, env: Environment) -> LevelResult:
        """Run with trajectory saving."""
        # Record task start time
        start_time = datetime.now().isoformat()
        
        info = env.get_basic_info()
        agent.reset(info)
        
        reset_result = env.reset()
        obs = await reset_result if inspect.isawaitable(reset_result) else reset_result
        
        history: list[StepRecord] = []
        total_reward = 0.0
        max_steps = info.max_steps

        # Run interaction loop
        for t in range(max_steps):
            logger.log_to_file(LogLevel.INFO, f"Step {t+1}/{max_steps}")
            
            step_result = await agent.step(observation=obs, history=history)
            if isinstance(step_result, (list, tuple)):
                if len(step_result) == 3:
                    action, raw_response, raw_input = step_result
                elif len(step_result) == 2:
                    action, raw_response = step_result
                    raw_input = None
                else:
                    raise ValueError(f"agent.step returned {len(step_result)} values, expected 2 or 3")
            else:
                raise TypeError(f"agent.step returned unsupported type: {type(step_result)}")
            obs_next, reward, done, step_info = await env.step(action)

            step_record = StepRecord(
                observation=obs,
                action=action,
                reward=reward,
                raw_response=raw_response,
                done=done,
                info=step_info,
                raw_input=raw_input,
            )
            
            history.append(step_record)
            total_reward += reward
            
            obs = obs_next
            if done:
                break

        # Record task end time
        end_time = datetime.now().isoformat()

        # Build result
        result = self._build_result(agent, history, total_reward, start_time, end_time)
        
        # Save complete trajectory as JSON
        if self.trajectory_dir:
            self._save_trajectory(info, result, agent)
        
        # Append to CSV summary
        if self.csv_summary_path:
            self._append_csv_row(info.env_id, result)

        return result
    
    def _build_result(self, agent: BaseAgent, history: list[StepRecord], total_reward: float, start_time: str = None, end_time: str = None) -> LevelResult:
        """Build LevelResult from history."""
        usage_summary = agent.llm.get_usage_summary()
        return LevelResult(
            model=usage_summary.get("model", ""),
            total_reward=total_reward,
            steps=len(history),
            done=history[-1].done if history else False,
            trace=history,
            cost=usage_summary.get("total_cost", 0.0),
            input_tokens=usage_summary.get("total_input_tokens", 0),
            output_tokens=usage_summary.get("total_output_tokens", 0),
            start_time=start_time,
            end_time=end_time,
        )

    def _save_trajectory(self, info: BasicInfo, result: LevelResult, agent: BaseAgent) -> None:
        """Save complete trajectory as JSON."""
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        task_id = info.meta_data.get("task_id") or info.env_id
        trajectory_file = self.trajectory_dir / f"{task_id}.json"
        
        trajectory = {
            "id": task_id,
            "model": result.model,
            "total_reward": result.total_reward,
            "steps": result.steps,
            "max_steps": info.max_steps,
            "done": result.done,
            "cost": result.cost,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "timestamp": result.timestamp,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "instruction": info.instruction,
            "action_space": info.action_space,
            "meta_data": info.meta_data,
            "trace": [
                {
                    "step": i,
                    "observation": step.observation,
                    "action": step.action,
                    "reward": step.reward,
                    "done": step.done,
                    "info": step.info,
                    "raw_input": step.raw_input,
                    "raw_response": step.raw_response,
                }
                for i, step in enumerate(result.trace)
            ],
        }
        
        with trajectory_file.open("w", encoding="utf-8") as f:
            json.dump(trajectory, f, indent=2, ensure_ascii=False)
    
    def _init_csv(self) -> None:
        """Initialize CSV file with header."""
        if not self.csv_summary_path or self._csv_initialized:
            return
        
        self.csv_summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and has content
        if self.csv_summary_path.exists() and self.csv_summary_path.stat().st_size > 0:
            self._csv_initialized = True
            return
        
        # Write header
        with self.csv_summary_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["id", "model", "steps", "total_reward", "timestamp", "start_time", "end_time", "cost", "input_tokens", "output_tokens", "done"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        
        self._csv_initialized = True
    
    def _append_csv_row(self, task_id: str, result: LevelResult) -> None:
        """Append summary row to CSV after task completion."""
        self._init_csv()
        
        with self.csv_summary_path.open("a", newline="", encoding="utf-8") as f:
            fieldnames = ["id", "model", "steps", "total_reward", "timestamp", "start_time", "end_time", "cost", "input_tokens", "output_tokens", "done"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                "id": task_id,
                "model": result.model,
                "steps": result.steps,
                "total_reward": result.total_reward,
                "timestamp": result.timestamp,
                "start_time": result.start_time,
                "end_time": result.end_time,
                "cost": f"{result.cost:.6f}",
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "done": result.done,
            })


def load_trajectory(file_path: Path) -> dict:
    """Load trajectory JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {file_path}")
    
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)
