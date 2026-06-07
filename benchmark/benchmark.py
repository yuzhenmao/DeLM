# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

import asyncio
import csv
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Type
from datetime import datetime

from benchmark.common.env import Environment
import base.agent.base_agent as Agent
from benchmark.common.runner import Runner, LevelResult

LevelSpec = Dict[str, Any]

class Benchmark:
    """
    Base class for benchmarks.
    
    Responsibilities:
    - Define available levels (list_levels)
    - Create environment instances (make_env)
    - Coordinate task execution (run)
    
    Note: Trajectory saving is handled by Runner (fine-grained, real-time).
    Use IncrementalRunner for incremental JSONL saving.
    """
    
    def list_levels(self) -> List[LevelSpec]:
        raise NotImplementedError

    def make_env(self, level: LevelSpec) -> "Environment":
        raise NotImplementedError

    async def run(
        self,
        agent_cls: Type[Agent],
        agent_kwargs: Optional[Dict[str, Any]] = None,
        runner: Runner | None = None,
        levels: Optional[List[LevelSpec]] = None,
        max_concurrency: int = 1,
        on_task_complete: Optional[callable] = None,
    ):
        """
        Run benchmark and return results.
        
        Args:
            agent_cls: Agent class to instantiate
            agent_kwargs: Arguments for agent constructor
            runner: Runner instance (uses default runner from benchmark if not provided)
            levels: List of levels to run (default: all levels)
            max_concurrency: Maximum concurrent tasks
            on_task_complete: Optional callback(level_id, result) called after each task
        
        Returns:
            Dict[level_id, LevelResult]: Results for all tasks
        
        Note: 
            Trajectory saving is handled by the runner.
            Each benchmark provides a default runner with incremental saving.
        """
        # Use benchmark's default runner if not provided
        if runner is None:
            runner = getattr(self, '_runner', Runner())
        
        levels = levels or self.list_levels()
        agent_kwargs = agent_kwargs or {}

        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def run_level(level: LevelSpec):
            async with semaphore:
                env = self.make_env(level)
                agent = agent_cls(**agent_kwargs)
                result = await runner.run(agent, env)
                return level, result

        tasks = [asyncio.create_task(run_level(level)) for level in levels]

        results = {}
        for task in asyncio.as_completed(tasks):
            level, result = await task
            level_id = level.get("id", str(level))
            results[level_id] = result
            
            if on_task_complete:
                try:
                    await_result = on_task_complete(level_id, result)
                    if asyncio.iscoroutine(await_result):
                        await await_result
                except Exception as e:
                    from base.engine.logs import logger
                    logger.warning(f"on_task_complete callback failed for {level_id}: {e}")

        return results
