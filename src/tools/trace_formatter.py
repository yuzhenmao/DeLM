# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

"""
Trace Formatter - Abstracted trace formatting utility

Provides extensible trace formatting interface supporting different benchmark action/observation formats.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol, Callable

class StepLike(Protocol):
    """Step protocol, compatible with StepRecord structure"""
    action: Dict[str, Any]
    observation: Any
    reward: float
    done: bool
    info: Dict[str, Any]

class ActionFormatter(ABC):
    """Action formatter base class"""
    
    @property
    @abstractmethod
    def action_type(self) -> str:
        """Return the action type this formatter handles"""
        ...
    
    @abstractmethod
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        """Format action to readable string"""
        ...

class ObservationFormatter(ABC):
    """Observation formatter base class"""
    
    @abstractmethod
    def can_format(self, obs: Dict[str, Any]) -> bool:
        """Check if this formatter can handle the observation"""
        ...
    
    @abstractmethod
    def format(self, obs: Dict[str, Any], max_len: int = 300) -> tuple[str, str]:
        """
        Format observation
        
        Returns:
            tuple[str, str]: (status line, output content)
        """
        ...

# ============== Generic Formatters ==============

class ExecuteActionFormatter(ActionFormatter):
    """execute action formatter"""
    
    @property
    def action_type(self) -> str:
        return "execute"
    
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        cmd = params.get("command", "")[:max_len]
        return f'execute(command="{cmd}")'

class FinishActionFormatter(ActionFormatter):
    """finish action formatter (generic)"""
    
    @property
    def action_type(self) -> str:
        return "finish"
    
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        status = params.get("status", "done")
        msg = params.get("message", "")[:60]
        return f'finish(status="{status}", msg="{msg}")'

class SubmitActionFormatter(ActionFormatter):
    """submit action formatter (generic)"""
    
    @property
    def action_type(self) -> str:
        return "submit"
    
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        return "submit()"

class ExitCodeObservationFormatter(ObservationFormatter):
    """observation formatter (exit_code + output)"""
    
    def can_format(self, obs: Dict[str, Any]) -> bool:
        return "exit_code" in obs
    
    def format(self, obs: Dict[str, Any], max_len: int = 300) -> tuple[str, str]:
        exit_code = obs.get("exit_code", "N/A")
        output = str(obs.get("output", ""))
        return f"exit_code={exit_code}", output

# ============== SWE-bench Formatters ==============

class ACICommandActionFormatter(ActionFormatter):
    """SWE-bench aci_command action formatter"""
    
    @property
    def action_type(self) -> str:
        return "aci_command"
    
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        cmd = params.get("command", "")
        # Truncate long commands (e.g., str_replace, edit with multi-line content)
        if "\n" in cmd:
            first_line = cmd.split("\n")[0][:60]
            return f'aci_command("{first_line}...")'
        return f'aci_command("{cmd[:max_len]}")'

class SWEBenchObservationFormatter(ObservationFormatter):
    """SWE-bench observation formatter (state_info + output)"""
    
    def can_format(self, obs: Dict[str, Any]) -> bool:
        return "state_info" in obs or "command" in obs
    
    def format(self, obs: Dict[str, Any], max_len: int = 300) -> tuple[str, str]:
        state_info = obs.get("state_info", "")
        output = str(obs.get("output", ""))
        exit_code = obs.get("exit_code", "N/A")
        return f"exit_code={exit_code}, {state_info}", output

# ============== Fallback Formatters ==============

class FallbackActionFormatter(ActionFormatter):
    """Generic fallback action formatter"""
    
    def __init__(self, action_type: str = "unknown"):
        self._action_type = action_type
    
    @property
    def action_type(self) -> str:
        return self._action_type
    
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        param_keys = list(params.keys())[:3]
        return f'{self._action_type}({param_keys})'

class FallbackObservationFormatter(ObservationFormatter):
    """Generic fallback observation formatter"""
    
    def can_format(self, obs: Dict[str, Any]) -> bool:
        return True  # Always handles
    
    def format(self, obs: Dict[str, Any], max_len: int = 300) -> tuple[str, str]:
        return "", str(obs)

# ============== TraceFormatter Main Class ==============

class TraceFormatter:
    """
    Trace formatter
    
    Implements extensible formatting logic via registered ActionFormatter and ObservationFormatter.
    """
    
    def __init__(self):
        self._action_formatters: Dict[str, ActionFormatter] = {}
        self._obs_formatters: List[ObservationFormatter] = []
        self._fallback_obs_formatter = FallbackObservationFormatter()
    
    def register_action_formatter(self, formatter: ActionFormatter) -> "TraceFormatter":
        """Register action formatter"""
        self._action_formatters[formatter.action_type] = formatter
        return self
    
    def register_obs_formatter(self, formatter: ObservationFormatter) -> "TraceFormatter":
        """Register observation formatter"""
        self._obs_formatters.append(formatter)
        return self
    
    def format_action(self, action: Dict[str, Any], max_len: int = 100) -> str:
        """Format single action"""
        action_type = action.get("action", "unknown")
        params = action.get("params", {})
        
        formatter = self._action_formatters.get(action_type)
        if formatter:
            return formatter.format(params, max_len)
        return FallbackActionFormatter(action_type).format(params, max_len)
    
    def format_observation(self, obs: Any, max_len: int = 300) -> tuple[str, str]:
        """Format single observation"""
        if not isinstance(obs, dict):
            return "", str(obs)
        
        for formatter in self._obs_formatters:
            if formatter.can_format(obs):
                return formatter.format(obs, max_len)
        
        return self._fallback_obs_formatter.format(obs, max_len)
    
    def format_trace(self, trace: List[StepLike], max_output_len: int = 300) -> str:
        """
        Format complete trace
        
        Args:
            trace: List of steps
            max_output_len: Output truncation length
        
        Returns:
            str: Formatted trace text
        """
        if not trace:
            return "No steps executed"
        
        lines = []
        for i, step in enumerate(trace, 1):
            # Format action
            action_str = self.format_action(step.action)
            lines.append(f"Step {i}: {action_str}")
            
            # Format observation
            status_line, output = self.format_observation(step.observation, max_output_len)
            if status_line:
                lines.append(f"  → {status_line}")
            
            # Truncate output
            if len(output) > max_output_len:
                output = output[:max_output_len] + f"...[+{len(output)-max_output_len} chars]"
            output = output.replace("\n", " ").strip()
            lines.append(f"  → output: {output}")
            lines.append("")
        
        return "\n".join(lines)

# ============== Pre-built Formatter Factory ==============

def create_swebench_formatter() -> TraceFormatter:
    """Create SWE-bench formatter"""
    return (
        TraceFormatter()
        .register_action_formatter(ACICommandActionFormatter())
        .register_action_formatter(ExecuteActionFormatter())
        .register_action_formatter(FinishActionFormatter())
        .register_action_formatter(SubmitActionFormatter())
        .register_obs_formatter(SWEBenchObservationFormatter())
        .register_obs_formatter(ExitCodeObservationFormatter())
    )
