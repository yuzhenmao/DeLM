"""PlannerAgent: orchestrates ImplementerAgents via tool calls."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Optional

from pydantic import Field

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.async_llm import ModelPricing
from base.engine.logs import logger, LogLevel
from benchmark.common.env import BasicInfo
from src.common.utils import parse_json_response, indent_text

def build_model_pricing_table(
    implementer_models: List[str], 
    model_to_alias: Dict[str, str] = None
) -> str:
    """Generate a pricing table for available sub-models."""
    lines = ["| Model | Input $/1K | Output $/1K |"]
    lines.append("|-------|-----------|------------|")
    
    alias_to_model = {v: k for k, v in model_to_alias.items()} if model_to_alias else {}
    
    for model_display in implementer_models:
        real_model = alias_to_model.get(model_display, model_display)
        input_price = ModelPricing.get_price(real_model, "input")
        output_price = ModelPricing.get_price(real_model, "output")
        lines.append(f"| {model_display} | ${input_price:.5f} | ${output_price:.5f} |")
    
    return "\n".join(lines)

class PlannerAgent(BaseAgent):
    """Orchestrator that delegates tasks to ImplementerAgents."""
    
    name: str = Field(default="PlannerAgent")
    description: str = Field(default="Multi-agent orchestrator")
    
    implementer_models: List[str] = Field(default_factory=list)
    tools: List[Any] = Field(default_factory=list)
    implementer_tools: List[Any] = Field(default_factory=list)  # Tools for ImplementerAgent (used in prompt)
    prompt_builder: Optional[Any] = Field(default=None)
    max_attempts: int = Field(default=6)
    # Model name masking (optional)
    mask_model_names: bool = Field(default=False)
    model_to_alias: Dict[str, str] = Field(default_factory=dict)
    alias_to_model: Dict[str, str] = Field(default_factory=dict)
    masked_implementer_models: List[str] = Field(default_factory=list)
    
    memory: Memory = Field(default=None)
    instruction: str = Field(default="")
    meta: Dict[str, Any] = Field(default_factory=dict)
    attempt: int = Field(default=0)
    task_entries: List[Dict] = Field(default_factory=list)

    thread_id: int = Field(default=0)
    n_threads: int = Field(default=1)
    shared_lessons: Optional[Any] = Field(default=None)
    shared_lessons_window_tokens: int = Field(default=2000)
    shared_lessons_render_mode: str = Field(default="full")
    planner_max_tokens: Optional[int] = Field(default=1200)

    mode_spec: Optional[Any] = Field(default=None)
    mode_name: Optional[str] = Field(default=None)
    allowed_actions: Optional[Any] = Field(default=None)
    decision_parser: Optional[Any] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if self.mode_spec is not None:
            spec = self.mode_spec
            if self.prompt_builder is None:
                self.prompt_builder = spec.prompt_builder
            self.decision_parser = spec.parser
            self.allowed_actions = spec.allowed_actions
            self.mode_name = spec.name
            if self.planner_max_tokens is None and spec.default_max_tokens is not None:
                self.planner_max_tokens = spec.default_max_tokens
        if self.mask_model_names and self.implementer_models:
            self.model_to_alias = {
                model: f"model_{i+1}" for i, model in enumerate(self.implementer_models)
            }
            self.alias_to_model = {v: k for k, v in self.model_to_alias.items()}
            self.masked_implementer_models = list(self.model_to_alias.values())
        else:
            self.model_to_alias = {m: m for m in self.implementer_models}
            self.alias_to_model = {m: m for m in self.implementer_models}
            self.masked_implementer_models = self.implementer_models
    
    def reset(self, env_info: BasicInfo) -> None:
        self.memory = Memory(llm=self.llm, max_memory=20)
        self.instruction = env_info.instruction
        self.meta = env_info.meta_data or {}
        self.attempt = 0
        self.task_entries = []
    
    def get_usage_cost(self) -> float:
        return self.llm.get_usage_summary().get("total_cost", 0.0)
    
    
    def _format_subtask_history(self) -> str:
        """Generate subtask history for prompt usage"""
        if not self.task_entries:
            return "No subtasks completed yet."
        
        lines = []
        done_count = 0
        all_issues = []

        for e in self.task_entries:
            if e.get("status") == "rejected":
                entry_lines = [
                    f'[Attempt {e["attempt"]}] ❌ rejected | '
                    f'action={e.get("rejected_action", "?")!r} | mode={e.get("mode", "?")}',
                    f'└─ {e.get("corrective_message", "Action rejected (no further detail).")}',
                ]
                lines.append("\n".join(entry_lines))
                continue

            emoji = "✅" if e["status"] == "done" else "⚠️"
            steps_info = f'{e.get("steps_taken", "?")}/{e.get("max_steps", 30)}'
            model_display = e.get("model", "?")

            # Model name masking
            if self.mask_model_names and model_display in self.model_to_alias:
                model_display = self.model_to_alias[model_display]

            entry_lines = [
                f'[Attempt {e["attempt"]}] {emoji} {e["status"]} | Model: {model_display} | Steps: {steps_info}',
                f'├─ Task: {e.get("instruction", "N/A")}',
            ]

            if e.get("message"):
                entry_lines.append(f'├─ Message: {e["message"]}')
            issues = e.get("issues", [])
            if issues:
                entry_lines.append(f'├─ ❌ Issues: {issues}')
                all_issues.extend(issues)

            entry_lines[-1] = entry_lines[-1].replace('├─', '└─')

            lines.append("\n".join(entry_lines))

            if e["status"] == "done":
                done_count += 1

        # Summary
        summary_lines = [f"---", f"Summary: {done_count}/{len(self.task_entries)} subtasks done"]
        if all_issues:
            summary_lines.append(f"❌ All issues: {all_issues}")

        lines.append("\n".join(summary_lines))

        return "\n\n".join(lines)
    
    async def step(self, observation, history, **kwargs) -> tuple:
        """Execute one orchestration decision."""
        self.attempt += 1
        logger.info(f"[PlannerAgent] Step {self.attempt}/{self.max_attempts}")
        
        subtask_history = self._format_subtask_history()
        logger.info(f"[PlannerAgent] Subtask history:\n{subtask_history}")
        
        shared_lessons_block = ""
        if self.shared_lessons is not None:
            try:
                if self.shared_lessons_render_mode == "patch_selective_unfold":
                    shared_lessons_block = await self.shared_lessons.render_patch_selective_unfold()
                elif self.shared_lessons_render_mode in (
                    "peer_digest",
                    "patch_digest",
                    "patch_fail_digest",
                    "patch_evidence_digest",
                    "patch_digest_local_only",
                    "patch_clean_local_only",
                    "patch_clean_delayed_local",
                ):
                    if self.shared_lessons_render_mode in (
                        "patch_digest_local_only",
                        "patch_clean_local_only",
                        "patch_clean_delayed_local",
                    ):
                        shared_lessons_block = ""
                    else:
                        shared_lessons_block = await self.shared_lessons.render_peer_digest(
                            window_tokens=self.shared_lessons_window_tokens,
                            viewer_thread_id=self.thread_id,
                            allowed_types=(
                                ("PATCH_SUMMARY",)
                                if self.shared_lessons_render_mode in (
                                    "patch_digest",
                                    "patch_fail_digest",
                                    "patch_evidence_digest",
                                )
                                else None
                            ),
                        )
                else:
                    shared_lessons_block = await self.shared_lessons.render(
                        window_tokens=self.shared_lessons_window_tokens)
            except Exception as e:
                logger.warning(f"[PlannerAgent] shared_lessons render failed: {e}")

        diversify_prefix = ""
        if self.attempt == 1 and self.n_threads > 1:
            try:
                from src.prompts.swebench_global import diversify_prompt
                diversify_prefix = diversify_prompt(self.thread_id)
            except Exception:
                diversify_prefix = ""

        if self.prompt_builder:
            prompt = self.prompt_builder.build_prompt(
                instruction=self.instruction,
                meta=self.meta,
                attempt_index=self.attempt,
                max_attempts=self.max_attempts,
                implementer_models=self.masked_implementer_models,
                subtask_history=subtask_history,
                model_to_alias=self.model_to_alias if self.mask_model_names else None,
                thread_id=self.thread_id,
                n_threads=self.n_threads,
                shared_lessons_block=shared_lessons_block,
                diversify_prefix=diversify_prefix,
            )
        else:
            prompt = f"Task: {self.instruction}\n\nReturn JSON: " \
                     "{{\"action\": \"...\", \"reasoning\": \"...\", \"params\": {{...}}}}"

        # Log prompt
        prompt_msg = f"\n{'='*80}\n[PlannerAgent Attempt {self.attempt}] PROMPT:\n{'='*80}\n{prompt}\n{'='*80}\n"
        logger.warning(prompt_msg)
        logger.log_to_file(LogLevel.INFO, prompt_msg)
        
        # Get LLM decision
        logger.info(f"[PlannerAgent] Calling LLM...")
        resp = await self.llm(prompt, max_tokens=self.planner_max_tokens)
        
        # Log response
        response_msg = f"\n{'='*80}\n[PlannerAgent Attempt {self.attempt}] RAW RESPONSE:\n{'='*80}\n{resp}\n{'='*80}\n"
        logger.warning(response_msg)
        logger.log_to_file(LogLevel.INFO, response_msg)
        
        parser = self.decision_parser or parse_json_response
        decision = parser(resp)

        # Log parsed decision
        decision_msg = f"\n{'='*80}\n[PlannerAgent Attempt {self.attempt}] PARSED DECISION:\n{'='*80}\n{json.dumps(decision, indent=2, ensure_ascii=False)}\n{'='*80}\n"
        logger.warning(decision_msg)
        logger.log_to_file(LogLevel.INFO, decision_msg)

        action_name = decision.get("action")
        params = decision.get("params", {})

        if self.allowed_actions is not None and action_name not in self.allowed_actions:
            mode_tag = self.mode_name or "unknown"
            allowed_list = list(self.allowed_actions)
            err = (
                f"action {action_name!r} not allowed in mode={mode_tag}; "
                f"allowed actions: {allowed_list}"
            )
            logger.warning(
                f"[PlannerAgent] mode={mode_tag} action={action_name!r} REJECTED ({err})"
            )
            self.task_entries.append({
                "attempt": self.attempt,
                "status": "rejected",
                "rejected_action": action_name,
                "mode": mode_tag,
                "allowed_actions": allowed_list,
                "corrective_message": (
                    f"Action '{action_name}' is not a PlannerAgent action. "
                    f"Allowed in mode={mode_tag}: {', '.join(allowed_list)}. "
                    "Do NOT emit local ImplementerAgent commands here (bash, open, "
                    "str_replace, create, goto, scroll_*, edit, find_file, "
                    "search_*) — those are executed by the ImplementerAgent via "
                    "delegate_task, not by PlannerAgent."
                ),
            })
            return {"action": "error", "error": err, "mode": mode_tag,
                    "rejected_action": action_name}, resp

        # Execute tool
        tool = next((t for t in self.tools if t.name == action_name), None)
        if not tool:
            return {"action": "error", "error": f"Unknown action: {action_name}"}, resp
        
        result = await tool(**params)

        self._update_task_entries(action_name, params, result)

        return {
            "action": action_name,
            "params": params,
            "result": result,
            "subtask_history": subtask_history,
        }, resp

    def _update_task_entries(self, action: str, params: Dict, result: Dict) -> None:
        """Append a task_entries row after a tool call. The next attempt's
        prompt picks this up via `_format_subtask_history()`.

        renamed from `_update_context` (which used to maintain a
        `self.context` string that was passed to `build_prompt` as
        `prior_context` and never interpolated into the prompt template).
        Only the delegate-task path produces an entry; `submit` is
        terminal and has nothing to feed back to a next attempt.
        """
        if action != "delegate_task":
            return

        finish_result = result.get('finish_result') or {}
        if finish_result:
            entry_status = finish_result.get('status', 'partial')
            entry_message = finish_result.get('message', '')
            entry_issues = finish_result.get('issues', [])
        else:
            entry_status = 'partial'
            entry_message = 'ImplementerAgent did not finish (max steps reached).'
            entry_issues = ['ImplementerAgent timeout - did not call finish']

        self.task_entries.append({
            "attempt": self.attempt,
            "status": entry_status,
            "instruction": params.get('task_instruction', 'N/A'),
            "model": params.get('model', 'unknown'),
            "steps_taken": result.get('steps_taken', 0),
            "max_steps": result.get('statistics', {}).get('max_steps', 30),
            "cost": result.get('cost', 0),
            "message": entry_message,
            "issues": entry_issues,
        })
    
    async def run(self, request: Optional[str] = None) -> str:
        return "Orchestration via Runner"
