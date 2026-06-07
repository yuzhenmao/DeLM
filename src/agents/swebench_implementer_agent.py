"""
SWE-bench local agent.
"""
import re
from typing import Any, Dict, Optional

from pydantic import Field

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory, truncate_middle
from base.engine.logs import logger, LogLevel
from benchmark.common.env import BasicInfo
from src.prompts.note_rules import NOTE_RULES_BLOCK  # noqa: F401 (kept for verifier reference + re-export)
from src.prompts.swebench_local import (  # noqa: F401 (re-exported for backward compat)
    SHARED_LESSONS_GUIDANCE,
    SWEBENCH_IMPLEMENTER_PROMPT,
    build_implementer_prompt,
)
from src.shared_lessons import VALID_TYPES as LESSON_TYPES, parse_note_body

PATCH_CLEAN_DELAY_STEPS = 8
_PEER_PATCH_LINE_RE = re.compile(
    r"^\[(?P<threads>[^\]]+)/PATCH_SUMMARY[^\]]*\]\s+"
    r"(?P<content>.*?)(?:\s+\[status=(?P<status>[^\]]+)\])?\s*$"
)

def _peer_patch_checkpoint_from_block(block: str, viewer_thread_id: int) -> str:
    """
    Return a compact peer patch checkpoint for the local observation.
    """
    if "/PATCH_SUMMARY" not in (block or ""):
        return ""
    viewer = f"t{int(viewer_thread_id)}"
    candidates = []
    for raw_line in str(block).splitlines():
        line = raw_line.strip()
        if "/PATCH_SUMMARY" not in line:
            continue
        match = _PEER_PATCH_LINE_RE.match(line)
        if not match:
            continue
        writers = {
            part.strip()
            for part in match.group("threads").split(",")
            if part.strip()
        }
        if writers and writers == {viewer}:
            continue
        status = (match.group("status") or "").lower()
        if "final" not in status and "verified" not in status:
            continue
        priority = (
            0 if ("final" in status and "verified" in status)
            else 1 if "verified" in status
            else 2
        )
        content = " ".join(match.group("content").split())
        if len(content) > 420:
            content = content[:420].rstrip()
        candidates.append((priority, content, status))
    if not candidates:
        return ""
    _, content, status = min(candidates, key=lambda item: item[0])
    return (
        "A peer has a finalized/verified candidate patch. Keep your own "
        "container independent, but before broad exploration, verify it, refine "
        "it minimally, or reject it with a concrete failed command/result. "
        f"Candidate: {content} [status={status}]"
    )

def parse_implementer_response(response: str) -> Dict[str, Any]:
    """
    Parse ImplementerAgent response to extract DISCUSSION, NOTE, and COMMAND.
    """
    discussion = ""
    command = ""
    notes: list = []

    # Pattern for DISCUSSION section
    discussion_match = re.search(
        r'DISCUSSION\s*\n(.*?)(?=\nNOTE\s*\n|\nCOMMAND\s*\n|\Z)',
        response,
        re.DOTALL | re.IGNORECASE
    )
    if discussion_match:
        discussion = discussion_match.group(1).strip()

    # (NOTE required) — header MUST be present
    note_match = re.search(
        r'\bNOTE\s*\n(.*?)(?=\nCOMMAND\s*\n|\Z)',
        response,
        re.DOTALL | re.IGNORECASE,
    )
    note_present = note_match is not None
    if note_present:
        notes = parse_note_body(note_match.group(1))

    # Pattern for COMMAND section - extract all content after COMMAND
    command_match = re.search(
        r'COMMAND\s*\n(.*?)(?=\n(?:DISCUSSION|OUTPUT)|\Z)',
        response,
        re.DOTALL | re.IGNORECASE
    )
    if command_match:
        raw_command = command_match.group(1).strip()

        # Only take the first meaningful line as the command
        # This prevents LLM's extra text/thoughts from being included
        lines = raw_command.split('\n')
        for idx, line in enumerate(lines):
            line = line.strip()
            # Skip empty lines and comment-like lines
            if line and not line.startswith('#') and not line.startswith('('):
                # For multi-line commands like str_replace, create, edit, etc.
                # we need to capture the full block
                if any(line.startswith(cmd) for cmd in ['str_replace', 'create', 'edit', 'insert', 'delete_lines']):
                    clean_lines = []
                    for l in lines[idx:]:
                        # Stop if we hit a line that looks like commentary
                        if l.strip().startswith('(') or l.strip().startswith('**') or l.strip().startswith('Wait'):
                            break
                        clean_lines.append(l)
                    command = '\n'.join(clean_lines).strip()
                else:
                    # Single-line commands: just take the first line
                    command = line
                break

        # If no command found, use the first line anyway
        if not command and lines:
            command = lines[0].strip()

    if not note_present:
        return {
            "action": "error",
            "params": {
                "message": (
                    "missing required NOTE section — every response MUST include "
                    "DISCUSSION + NOTE + COMMAND in that order. If you have nothing "
                    "to share this turn, write `NOTE\\n(none)`."
                ),
            },
            "reasoning": discussion,
            "notes": [],
        }

    # Handle finish command (ImplementerAgent specific)
    # Format: finish <status> <message>
    # status: done | partial
    if command.lower().startswith('finish'):
        # Extract content after 'finish'
        finish_content = command[6:].strip() if len(command) > 6 else ""
        
        # Parse status and message
        status = "done"  # default
        message = finish_content or discussion
        
        # Check if content starts with a valid status (done or partial)
        content_lower = finish_content.lower()
        if content_lower.startswith('done'):
            status = "done"
            message = finish_content[4:].strip() or discussion
        elif content_lower.startswith('partial'):
            status = "partial"
            message = finish_content[7:].strip() or discussion
        
        return {
            "action": "finish",
            "params": {
                "status": status,
                "message": message,
            },
            "reasoning": discussion,
            "notes": notes,
        }

    # Handle submit command (should not be used by ImplementerAgent, convert to finish)
    if command.lower() == 'submit':
        return {
            "action": "finish",
            "params": {
                "status": "done",
                "message": "ImplementerAgent attempted to submit - converting to finish",
            },
            "reasoning": discussion,
            "notes": notes,
        }

    # Regular ACI or bash command - use aci_command like Baseline
    if command:
        return {
            "action": "aci_command",
            "params": {"command": command},
            "reasoning": discussion,
            "notes": notes,
        }

    # Fallback: extract any command-like content
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('DISCUSSION'):
            return {
                "action": "aci_command",
                "params": {"command": line},
                "reasoning": response,
                "notes": notes,
            }

    return {
        "action": "error",
        "params": {"message": "Could not parse response"},
        "reasoning": response,
        "notes": notes,
    }

class SWEBenchImplementerAgent(BaseAgent):
    """
    ImplementerAgent for SWE-bench.
    
    Uses DISCUSSION + COMMAND format (same as baseline SWEAgent),
    with 'finish' command for reporting back to PlannerAgent.
    """
    name: str = Field(default="SWEBenchImplementerAgent")
    description: str = Field(default="ImplementerAgent for SWE-bench with ACI tools")
    
    # Task context from PlannerAgent
    task_instruction: str = Field(default="")
    context: str = Field(default="")
    original_question: str = Field(default="")

    current_instruction: str = Field(default="")
    command_docs: str = Field(default="")
    state_info: str = Field(default="(Open file: n/a) (Current directory: /testbed)")
    memory: Optional[Memory] = Field(default=None)

    thread_id: int = Field(default=0)
    delegation_id: int = Field(default=0)
    shared_lessons: Optional[Any] = Field(default=None)
    # shared-lessons render window (tokens), from config.
    shared_lessons_window_tokens: int = Field(default=1000)
    shared_lessons_render_mode: str = Field(default="full")
    peer_patch_checkpoint_enabled: bool = Field(default=False)
    implementer_fast_finish_prompt_enabled: bool = Field(default=False)
    peer_patch_protocol_enabled: bool = Field(default=False)
    patch_summary_timing_rule_enabled: bool = Field(default=False)
    implementer_strategy_prompt_enabled: bool = Field(default=False)
    prompt_cache_static_layout: bool = Field(default=False)
    prompt_cache_min_prefix_chars: int = Field(default=16000)
    shared_lessons_block: str = Field(default="")
    
    class Config:
        arbitrary_types_allowed = True

    def reset(self, env_info: BasicInfo) -> None:
        """Reset agent state for new task."""
        if self.memory is None:
            self.memory = Memory(llm=self.llm, max_memory=8, max_obs_chars=2000)
        else:
            self.memory.clear()
        
        # Use task_instruction if set by PlannerAgent, otherwise use env instruction
        self.current_instruction = self.task_instruction or env_info.instruction
        self.command_docs = env_info.meta_data.get("command_docs", env_info.action_space)
        self.state_info = "(Open file: n/a) (Current directory: /testbed)"
        
        # Store original question for reference
        if not self.original_question:
            self.original_question = env_info.instruction
        
        logger.info(f"[SWEBenchImplementerAgent] Reset for task")

    def parse_action(self, resp: str) -> Dict[str, Any]:
        """Parse LLM response using DISCUSSION + COMMAND format."""
        return parse_implementer_response(resp)

    def _get_memory(self) -> str:
        """Get formatted memory context."""
        if self.memory:
            return self.memory.as_text()
        return "No previous observations."

    def _profile_prompt(self, prompt: str, current_step: int) -> None:
        """Measurement-only: if env SL_PROMPT_PROFILE is set to a path, append a JSONL
        record of the prompt's per-section char counts (split on '==== X ====' headers)
        so we can see the module distribution + max_len. No-op when the env is unset."""
        import os
        path = os.environ.get("SL_PROMPT_PROFILE")
        if not path:
            return
        try:
            import json, re
            parts = re.split(r'(?m)^==== (.+?) ====\s*$', prompt)
            sections = {"_preamble": len(parts[0])}
            for i in range(1, len(parts), 2):
                name = parts[i].strip()
                body = parts[i + 1] if i + 1 < len(parts) else ""
                sections[name] = sections.get(name, 0) + len(body)
            rec = {"thread": self.thread_id, "step": current_step,
                   "total_chars": len(prompt), "sections": sections}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    async def step(self, observation: Any, history: Any, current_step: int = 1, max_steps: int = 50) -> tuple:
        """Execute one step of the agent loop.

        Returns:
            tuple: (action, raw_response, raw_input_prompt)
        """
        # Format observation
        if isinstance(observation, dict):
            obs_str = observation.get("output", str(observation))
            if "state_info" in observation:
                self.state_info = observation["state_info"]
        else:
            obs_str = str(observation)

        obs_str = truncate_middle(obs_str, 16000)

        if self.shared_lessons is not None:
            try:
                if self.shared_lessons_render_mode == "patch_selective_unfold":
                    self.shared_lessons_block = await self.shared_lessons.render_patch_selective_unfold()
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
                        "patch_digest",
                        "patch_digest_local_only",
                        "patch_clean_local_only",
                        "patch_clean_delayed_local",
                    ):
                        allowed_types = ("PATCH_SUMMARY",)
                    elif self.shared_lessons_render_mode == "patch_fail_digest":
                        allowed_types = ("PATCH_SUMMARY", "FAIL")
                    elif self.shared_lessons_render_mode == "patch_evidence_digest":
                        allowed_types = ("PATCH_SUMMARY", "OBSERVED", "FACT")
                    else:
                        allowed_types = None
                    if (
                        self.shared_lessons_render_mode == "patch_clean_delayed_local"
                        and current_step < PATCH_CLEAN_DELAY_STEPS
                    ):
                        self.shared_lessons_block = "(no peer shared lessons yet)"
                    else:
                        self.shared_lessons_block = await self.shared_lessons.render_peer_digest(
                            window_tokens=self.shared_lessons_window_tokens,
                            viewer_thread_id=self.thread_id,
                            allowed_types=allowed_types,
                            fail_requires_finalized_patch=(
                                self.shared_lessons_render_mode == "patch_fail_digest"
                            ),
                            patch_requires_no_fail=(
                                self.shared_lessons_render_mode in (
                                    "patch_clean_local_only",
                                    "patch_clean_delayed_local",
                                )
                            ),
                        )
                else:
                    self.shared_lessons_block = await self.shared_lessons.render(
                        window_tokens=self.shared_lessons_window_tokens,
                        viewer_thread_id=self.thread_id,
                        viewer_delegation_id=self.delegation_id,
                        self_policy="demote_current",
                    )
            except Exception as e:
                logger.warning(f"[SWEBenchImplementerAgent] shared_lessons render failed: {e}")
                self.shared_lessons_block = ""

        if self.peer_patch_checkpoint_enabled:
            peer_patch_checkpoint = _peer_patch_checkpoint_from_block(
                self.shared_lessons_block,
                self.thread_id,
            )
            if peer_patch_checkpoint:
                obs_str = (
                    f"{obs_str}\n\nPEER_PATCH_CHECKPOINT:\n"
                    f"{peer_patch_checkpoint}"
                )

        strategy_prefix = ""
        if (
            self.implementer_strategy_prompt_enabled
            and int(getattr(self, "delegation_id", 0)) == 1
        ):
            try:
                from src.prompts.swebench_global import diversify_prompt
                strategy_prefix = diversify_prompt(self.thread_id)
            except Exception:
                strategy_prefix = ""

        prompt = build_implementer_prompt(
            task_instruction=self.current_instruction,
            context=self.context,
            command_docs=self.command_docs,
            state_info=self.state_info,
            memory=self._get_memory(),
            observation=obs_str,
            current_step=current_step,
            max_steps=max_steps,
            shared_lessons_block=self.shared_lessons_block,
            thread_id=self.thread_id,
            cache_static_prefix_layout=self.prompt_cache_static_layout,
            cache_static_prefix_min_chars=self.prompt_cache_min_prefix_chars,
            implementer_fast_finish_prompt_enabled=self.implementer_fast_finish_prompt_enabled,
            peer_patch_protocol_enabled=self.peer_patch_protocol_enabled,
            patch_summary_timing_rule_enabled=self.patch_summary_timing_rule_enabled,
            strategy_prefix=strategy_prefix,
        )
        self._profile_prompt(prompt, current_step)  # env-gated (SL_PROMPT_PROFILE); no-op otherwise

        logger.log_to_file(LogLevel.INFO, f"[SWEBenchImplementerAgent] Step {current_step} Input:\n{prompt}\n")
        
        try:
            resp = await self.llm(prompt)
        except Exception as e:
            logger.error(f"[SWEBenchImplementerAgent] LLM call failed: {e}")
            resp = "DISCUSSION\nLLM call failed, reporting back.\nCOMMAND\nfinish LLM call failed"
        
        action = self.parse_action(resp)
        logger.agent_action(f"[SWEBenchImplementerAgent] Action: {action}")
        
        reasoning = action.get("reasoning", "")
        
        if self.memory:
            await self.memory.add_memory(
                obs=obs_str,
                action=action,
                thinking=reasoning,
                raw_response=resp
            )
        
        return action, resp, prompt

    async def run(self, request: Optional[str] = None) -> str:
        """Main run method (not used in benchmark loop)."""
        return ""
