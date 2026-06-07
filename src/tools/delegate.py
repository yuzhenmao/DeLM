# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

"""DelegateTaskTool - Unified task delegation tool"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from pydantic import Field, PrivateAttr

from base.agent.base_action import BaseAction
from base.agent.memory import Memory
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.logs import logger
from src.memory_compactor import extract_structured_summary
from src.tools.trace_formatter import create_swebench_formatter

def _make_serializable(obj: Any) -> Any:
    """Recursively convert an object to JSON-serializable format."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _make_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]

    return str(obj)

class DelegateTaskTool(BaseAction):
    """SWE-bench task delegation tool.

    model choice is removed from the tool surface. ImplementerAgent runs
    on a pinned `implementer_model` (set at construction time); PlannerAgent no
    longer picks a model per delegation.
    """

    name: str = "delegate_task"
    description: str = "Delegate task to ImplementerAgent that executes commands"
    parameters: Dict[str, Any] = Field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "task_instruction": {"type": "string", "description": "Task for ImplementerAgent"},
            "context": {"type": "string", "description": "Additional context/hints"},
            "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools for ImplementerAgent (optional)"},
        },
        "required": ["task_instruction"]
    })

    env: Any = Field(default=None, exclude=True)
    runner: Any = Field(default=None, exclude=True)
    models: list = Field(default_factory=list)
    implementer_model: str = Field(default="")  # pinned ImplementerAgent model

    alias_to_model: Dict[str, str] = Field(default_factory=dict)  # Model name masking (optional)

    thread_id: int = Field(default=0)
    shared_lessons: Any = Field(default=None, exclude=True)

    implementer_max_memory: int = Field(default=8)
    implementer_max_obs_chars: int = Field(default=2000)
    implementer_max_thinking_chars: int = Field(default=500)
    summarizer_model: Optional[str] = Field(default=None)
    shared_lessons_window_tokens: int = Field(default=1000)
    shared_lessons_render_mode: str = Field(default="full")
    peer_patch_checkpoint_enabled: bool = Field(default=False)
    implementer_fast_finish_prompt_enabled: bool = Field(default=False)
    peer_patch_protocol_enabled: bool = Field(default=False)
    patch_summary_timing_rule_enabled: bool = Field(default=False)
    implementer_strategy_prompt_enabled: bool = Field(default=False)
    prompt_cache_enabled: bool = Field(default=False)
    prompt_cache_min_prefix_chars: int = Field(default=16000)
    # Memory compaction mode:
    # "llm"           — LLM prose summarizer (uses `summarizer_model`).
    # "deterministic" — zero-LLM structured extractor (memory_compactor).
    memory_compaction_mode: str = Field(default="llm")

    _trace_formatter: Any = PrivateAttr(default=None)
    _delegation_seq: int = PrivateAttr(default=0)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        env,
        runner,
        models: list,
        alias_to_model: Dict[str, str] = None,
        implementer_model: str = None,
        thread_id: int = 0,
        shared_lessons: Any = None,
        implementer_max_memory: int = 8,
        implementer_max_obs_chars: int = 2000,
        implementer_max_thinking_chars: int = 500,
        summarizer_model: str = None,
        shared_lessons_window_tokens: int = 1000,
        shared_lessons_render_mode: str = "full",
        memory_compaction_mode: str = "llm",
        peer_patch_checkpoint_enabled: bool = False,
        implementer_fast_finish_prompt_enabled: bool = False,
        peer_patch_protocol_enabled: bool = False,
        patch_summary_timing_rule_enabled: bool = False,
        implementer_strategy_prompt_enabled: bool = False,
        prompt_cache_enabled: bool = False,
        prompt_cache_min_prefix_chars: int = 16000,
    ):
        super().__init__()
        self.env = env
        self.runner = runner
        self.models = models
        self.alias_to_model = alias_to_model or {}
        self.implementer_model = implementer_model or (models[0] if models else "")
        self.thread_id = int(thread_id)
        self.shared_lessons = shared_lessons
        self.implementer_max_memory = int(implementer_max_memory)
        self.implementer_max_obs_chars = int(implementer_max_obs_chars)
        self.implementer_max_thinking_chars = int(implementer_max_thinking_chars)
        self.summarizer_model = summarizer_model
        self.shared_lessons_window_tokens = int(shared_lessons_window_tokens)
        self.shared_lessons_render_mode = str(shared_lessons_render_mode or "full").lower()
        self.peer_patch_checkpoint_enabled = bool(peer_patch_checkpoint_enabled)
        self.implementer_fast_finish_prompt_enabled = bool(implementer_fast_finish_prompt_enabled)
        self.peer_patch_protocol_enabled = bool(peer_patch_protocol_enabled)
        self.patch_summary_timing_rule_enabled = bool(patch_summary_timing_rule_enabled)
        self.implementer_strategy_prompt_enabled = bool(implementer_strategy_prompt_enabled)
        self.prompt_cache_enabled = bool(prompt_cache_enabled)
        self.prompt_cache_min_prefix_chars = int(prompt_cache_min_prefix_chars)
        mode = str(memory_compaction_mode or "llm").lower()
        if mode not in ("llm", "deterministic"):
            logger.warning(
                f"[DelegateTool] unknown memory_compaction_mode={memory_compaction_mode!r}; "
                f"falling back to 'llm'"
            )
            mode = "llm"
        self.memory_compaction_mode = mode
        self._trace_formatter = create_swebench_formatter()

        self.parameters = {
            "type": "object",
            "properties": {
                "task_instruction": {"type": "string", "description": "Task for ImplementerAgent"},
                "context": {"type": "string", "description": "Additional context/hints"},
                "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools for ImplementerAgent (optional)"},
            },
            "required": ["task_instruction"]
        }

    async def __call__(
        self,
        task_instruction: str,
        context: str = "",
        tools: List[str] = None,
        **_extra,
    ) -> Dict:
        """Execute delegated task

        Args:
            task_instruction: Task description for ImplementerAgent to execute
            context: Additional context information
            tools: List of tools allowed for ImplementerAgent

        Returns:
            Dictionary containing execution results
        """
        real_model = self.implementer_model
        if real_model not in self.models:
            return {"error": f"Invalid implementer_model: {real_model}", "steps_taken": 0, "done": False}

        logger.info(f"[DelegateTool] Creating ImplementerAgent with model={real_model}, tools={tools}")
        
        original_question = getattr(self.env, 'instruction', '') or ''
        
        self._delegation_seq += 1
        delegation_id = self._delegation_seq
        llm = create_llm_instance(LLMsConfig.default().get(real_model), role="ImplementerAgent")
        llm.prompt_cache_enabled = self.prompt_cache_enabled
        llm.prompt_cache_min_prefix_chars = self.prompt_cache_min_prefix_chars
        summarizer_llm = None
        if self.memory_compaction_mode == "llm" and self.summarizer_model:
            try:
                summarizer_llm = create_llm_instance(
                    LLMsConfig.default().get(self.summarizer_model), role="Summarizer")
            except Exception as e:
                logger.warning(
                    f"[DelegateTool] summarizer_model={self.summarizer_model!r} unavailable "
                    f"({e}); using worker model for memory compression")

        from src.agents.swebench_implementer_agent import SWEBenchImplementerAgent
        implementer_agent = SWEBenchImplementerAgent(
            llm=llm,
            task_instruction=task_instruction,
            context=context,
            original_question=original_question,
            memory=Memory(llm=llm, max_memory=self.implementer_max_memory,
                          max_obs_chars=self.implementer_max_obs_chars,
                          max_thinking_chars=self.implementer_max_thinking_chars,
                          summarizer_llm=summarizer_llm,
                          record_compactor=(
                              extract_structured_summary
                              if self.memory_compaction_mode == "deterministic"
                              else None
                          )),
            thread_id=self.thread_id,
            delegation_id=delegation_id,
            shared_lessons=self.shared_lessons,
            shared_lessons_window_tokens=self.shared_lessons_window_tokens,  # 
            shared_lessons_render_mode=self.shared_lessons_render_mode,
            peer_patch_checkpoint_enabled=self.peer_patch_checkpoint_enabled,
            implementer_fast_finish_prompt_enabled=self.implementer_fast_finish_prompt_enabled,
            peer_patch_protocol_enabled=self.peer_patch_protocol_enabled,
            patch_summary_timing_rule_enabled=self.patch_summary_timing_rule_enabled,
            implementer_strategy_prompt_enabled=self.implementer_strategy_prompt_enabled,
            prompt_cache_static_layout=self.prompt_cache_enabled,
            prompt_cache_min_prefix_chars=self.prompt_cache_min_prefix_chars,
        )
        
        original_instruction = getattr(self.env, 'instruction', None)
        if hasattr(self.env, 'instruction'):
            self.env.instruction = task_instruction
        
        try:
            result = await self.runner.run(implementer_agent, self.env)
            
            finish_result = None
            max_steps_exhausted = False
            if result.trace:
                last = result.trace[-1]
                if last.info.get("finished") and last.info.get("finish_result"):
                    finish_result = last.info["finish_result"]
                max_steps_exhausted = bool(last.info.get("max_steps_exhausted", False))
            try:
                max_steps_actual = int(self.env.get_basic_info().max_steps)
            except Exception:
                max_steps_actual = result.steps
            
            trace_serializable = [_make_serializable(step) for step in result.trace] if result.trace else []
            
            return {
                "model": real_model,
                "tools_assigned": tools,
                "steps_taken": result.steps,
                "done": result.done,
                "cost": result.cost,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "finish_result": finish_result,
                "max_steps_exhausted": max_steps_exhausted,
                "trace": trace_serializable,
                "statistics": {
                    "total_steps": result.steps,
                    "max_steps": max_steps_actual,
                    "max_steps_exhausted": max_steps_exhausted,
                },
            }
            
        except Exception as e:
            logger.error(f"[DelegateTool] Error: {e}")
            return {"error": str(e), "steps_taken": 0, "done": False, "cost": 0.0}
            
        finally:
            if original_instruction is not None and hasattr(self.env, 'instruction'):
                self.env.instruction = original_instruction
    
