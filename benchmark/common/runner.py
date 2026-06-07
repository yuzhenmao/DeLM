# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from benchmark.common.env import Action, BasicInfo, Environment, Observation
from base.agent.base_agent import BaseAgent
from base.engine.logs import logger, LogLevel

@dataclass
class StepRecord:
    observation: Observation
    action: Action
    reward: float
    raw_response: str
    done: bool
    info: Dict[str, Any]
    raw_input: Optional[str] = None
    act_prompt: Optional[str] = None


@dataclass
class LevelResult:
    model: str
    total_reward: float
    steps: int
    done: bool
    trace: List[StepRecord]
    cost: float
    input_tokens: int = 0
    output_tokens: int = 0
    delegate_count: int = 0
    submit_count: int = 0
    error_count: int = 0
    implementer_steps_total: int = 0
    forced_cap: bool = False
    forced_submit: bool = False
    wall_clock_seconds: float = 0.0
    active_llm_time_seconds: float = 0.0
    n_solvers_total: int = 1
    n_solvers_passed: int = 0
    winning_thread_id: str = "-"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class Runner:
    """
    Base Runner: standard agent-environment interaction loop.
    
    Subclasses should override this method to add:
    - Resource cleanup (containers, sandboxes, etc.)
    - Error handling and recovery
    - Custom logging or checkpointing
    """
    # Wall-clock timeout per agent.step call (seconds). None disables.
    step_timeout: Optional[float] = 600.0
    
    async def run(self, agent: BaseAgent, env: Environment) -> LevelResult:
        """Run agent-environment interaction loop and return result."""
        # Record task start time
        start_time = datetime.now().isoformat()
        
        info = env.get_basic_info()
        agent.reset(info)

        reset_result = env.reset()
        obs = await reset_result if inspect.isawaitable(reset_result) else reset_result

        history: List[StepRecord] = []
        total_reward = 0.0
        max_steps = info.max_steps

        for t in range(max_steps):
            logger.log_to_file(LogLevel.INFO, f"Environment Observation:{obs}")
            try:
                if self.step_timeout:
                    step_result = await asyncio.wait_for(
                        agent.step(observation=obs, history=history),
                        timeout=self.step_timeout,
                    )
                else:
                    step_result = await agent.step(observation=obs, history=history)
            except asyncio.TimeoutError:
                logger.error(f"Agent step timed out after {self.step_timeout} seconds; terminating level early.")
                step_record = StepRecord(
                    observation=obs,
                    action={"error": "step_timeout"},
                    reward=0.0,
                    raw_response="step timeout",
                    done=True,
                    info={"error": "step_timeout", "timeout_s": self.step_timeout},
                    raw_input=None,
                )
                history.append(step_record)
                break

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