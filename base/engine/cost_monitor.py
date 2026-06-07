# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

"""
Global cost monitoring for LLM calls using contextvars.

Usage:
    with CostMonitor() as monitor:
        await pipeline.run(...)
        print(f"Total cost: ${monitor.total_cost:.4f}")
        monitor.save()  # saves to workspace/costs/<timestamp>.json
"""

import contextvars
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_current_monitor: contextvars.ContextVar["CostMonitor | None"] = contextvars.ContextVar(
    "cost_monitor", default=None
)


@dataclass
class CostRecord:
    """Single LLM call cost record."""

    model: str
    input_tokens: int
    output_tokens: int
    cost: float


@dataclass
class CostMonitor:
    """Aggregates LLM costs during a context scope."""

    records: list[CostRecord] = field(default_factory=list)
    _token: contextvars.Token | None = field(default=None, repr=False)

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def call_count(self) -> int:
        return len(self.records)

    def record(self, model: str, input_tokens: int, output_tokens: int, cost: float) -> None:
        """Record a single LLM call's cost."""
        self.records.append(CostRecord(model, input_tokens, output_tokens, cost))

    def summary(self) -> dict:
        """Get aggregated summary."""
        return {
            "total_cost": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "call_count": self.call_count,
            "by_model": self._group_by_model(),
        }

    def _group_by_model(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for r in self.records:
            if r.model not in result:
                result[r.model] = {"cost": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0}
            result[r.model]["cost"] += r.cost
            result[r.model]["input_tokens"] += r.input_tokens
            result[r.model]["output_tokens"] += r.output_tokens
            result[r.model]["calls"] += 1
        return result

    def save(self, save_dir: str = "workspace/costs") -> Path:
        """Save cost summary to JSON file."""
        cost_dir = Path(save_dir)
        cost_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = cost_dir / f"{ts}.json"
        path.write_text(
            json.dumps(self.summary(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def __enter__(self) -> "CostMonitor":
        self._token = _current_monitor.set(self)
        return self

    def __exit__(self, *args) -> None:
        if self._token is not None:
            _current_monitor.reset(self._token)


def get_current_monitor() -> "CostMonitor | None":
    """Get the current cost monitor from context, if any."""
    return _current_monitor.get()


def record_cost(model: str, input_tokens: int, output_tokens: int, cost: float) -> None:
    """Record cost to the current monitor if one is active."""
    monitor = get_current_monitor()
    if monitor is not None:
        monitor.record(model, input_tokens, output_tokens, cost)
