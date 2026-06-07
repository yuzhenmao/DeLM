# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Dict, List, Optional

import json

from base.engine.async_llm import AsyncLLM
from benchmark.common.env import Observation, Action


def truncate_middle(text: Any, max_chars: int) -> str:
    """
    Clamp a (possibly huge) observation/text to max_chars, keeping the head and
    tail and eliding the middle. File windows and test/SQL dumps put the useful
    bits at the ends (headers/first-error at the top, result/summary at the
    bottom); the verbose middle is what balloons context. Head-heavy split.
    """
    s = text if isinstance(text, str) else str(text)
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    head = (max_chars * 3) // 5
    tail = max_chars - head
    elided = len(s) - head - tail
    return f"{s[:head]}\n... <{elided} chars elided> ...\n{s[-tail:]}"


class Memory(ABC):

    def __init__(self, llm: AsyncLLM, max_memory: int = 10, keep_recent: int = 3,
                 max_obs_chars: Optional[int] = None, summarizer_llm: "AsyncLLM" = None,
                 max_thinking_chars: int = 500,
                 record_compactor: Optional[Callable[[List[Dict[str, Any]]], str]] = None):
        self.llm = llm
        self.summarizer_llm = summarizer_llm or llm
        self.record_compactor = record_compactor
        self.max_memory = max_memory
        self.keep_recent = keep_recent
        self.max_obs_chars = max_obs_chars
        self.max_thinking_chars = max_thinking_chars
        self._records: List[Dict[str, Any]] = []
        self._summary: Optional[str] = None

    async def add_memory(
        self,
        obs: Observation,
        action: Action,
        thinking: Optional[str] = None,
        reward: Optional[float] = None,
        raw_response: Optional[str] = None,
    ) -> None:
        """
        Add a new memory record, then trigger compression if needed.
        """
        record = {
            "observation": obs,
            "action": action,
            "thinking": thinking,
            "reward": reward,
            "raw_response": raw_response,
        }
        self._records.append(record)
        await self._compress()

    def _get_text(self, include_thinking: bool = True, include_summary: bool = True) -> str:
        """
        Return readable memory text for prompts.
        """
        if not self._summary and not self._records:
            return "None"

        parts: List[str] = []

        if self._summary and include_summary:
            parts.append("[Summary of earlier steps]")
            parts.append(self._summary)
            parts.append("")

        if self._records:
            parts.append("[Recent steps (oldest first)]")
            for idx, r in enumerate(self._records, 1):
                act = r.get("action")
                obs = r.get("observation")
                reward = r.get("reward")
                if self.max_obs_chars:
                    obs = truncate_middle(obs, self.max_obs_chars)
                    act = truncate_middle(act, self.max_obs_chars)
                if include_thinking:
                    thinking = r.get("thinking")
                    if self.max_thinking_chars and thinking is not None:
                        thinking = truncate_middle(str(thinking), self.max_thinking_chars)
                    parts.append(f"{idx}. act={act}, obs={obs}, thinking={thinking}, reward={reward}")
                else:
                    parts.append(f"{idx}. act={act}, obs={obs}, reward={reward}")

        return "\n".join(parts)

    def as_text(self, include_thinking: bool = True, include_summary: bool = True) -> str:
        return self._get_text(include_thinking=include_thinking, include_summary=include_summary)

    async def _compress(self) -> None:
        """
        When record count reaches max_memory:
        - Compress older records into the summary.
        - Keep only the most recent keep_recent raw records.
        """
        if len(self._records) < self.max_memory:
            return

        if self.keep_recent > 0:
            head = self._records[:-self.keep_recent]
            tail = self._records[-self.keep_recent:]
        else:
            head = self._records[:]
            tail = []

        if head:
            head_summary = await self._summarize_records(head)
            if self._summary:
                # Append to existing summary for continuity
                self._summary += "\n\n" + head_summary
            else:
                self._summary = head_summary
                
        self._records = tail

    async def _summarize_records(self, records: List[Dict[str, Any]]) -> str:
        """
        Compress a batch of records into a memory summary.
        """
        compactor = getattr(self, "record_compactor", None)
        if compactor is not None:
            try:
                return compactor(records)
            except Exception:
                pass

        record_lines: List[str] = []
        for idx, r in enumerate(records, 1):
            act = r.get("action")
            obs = r.get("observation")
            reward = r.get("reward")
            thinking = r.get("thinking")
            record_lines.append(
                f"{idx}. action={json.dumps(act, ensure_ascii=False)}, "
                f"observation={json.dumps(obs, ensure_ascii=False)}, "
                f"thinking={json.dumps(thinking, ensure_ascii=False)}, "
                f"reward={reward}"
            )
        records_text = "\n".join(record_lines)

        summary_prompt = f"""
You are the memory compression module of a language-model-based agent.

You are given several past interaction steps in chronological order (oldest first).
Each step includes:
- the agent's action,
- the environment observation,
- the reward signal.
- the agent's thinking.

Your task is to write a compact, **persistent memory** block that lets the agent
continue its work without seeing the full history.

Please:
- Focus on:
  1) stable facts and rules about the environment/world,
  2) useful strategies / plans / tools the agent tried,
  3) important mistakes or failure patterns to avoid later,
  4) partial progress and remaining goals / TODOs.
- Use at most 8–10 lines.
- Use a neutral, factual tone.
- Do NOT repeat low-level JSON details unless they are crucial.
- Do NOT include meta text like "here is the summary" or any explanation.
- Output ONLY the memory lines, one per line.

===== PAST STEPS =====
{records_text}
===== SUMMARY (start here) =====
""".strip()

        resp = await self.summarizer_llm(summary_prompt, role_override="Summarizer")

        return resp

    def clear(self) -> None:
        """
        Clear all memories and summary.
        """
        self._records.clear()
        self._summary = None
