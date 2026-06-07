# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

"""Common utility functions"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

def parse_json_response(resp: str) -> Dict[str, Any]:
    """Parse JSON response that may contain markdown code blocks.

    Args:
        resp: LLM response string

    Returns:
        Parsed dict (or an {"action": "error", ...} dict on unrepairable input)
    """
    s = resp.strip()

    extractions = []
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b > a:
        extractions.append(s[a:b + 1])          # outermost braces (primary)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
    if fence:
        extractions.append(fence.group(1))       # cleanly fenced object (fallback)
    extractions.append(s)                        # whole string (last resort)

    for cand in extractions:
        no_trailing = re.sub(r",(\s*[}\]])", r"\1", cand)
        escaped = re.sub(r'\\(?!["\\/bfnrtu]|u[0-9a-fA-F]{4})', r"\\\\", cand)
        escaped_no_trailing = re.sub(r",(\s*[}\]])", r"\1", escaped)
        for candidate in (cand, no_trailing, escaped, escaped_no_trailing):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    objs, depth, start = [], 0, -1
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                objs.append(s[start:i + 1])
    for cand in reversed(objs):
        for c in (cand, re.sub(r",(\s*[}\]])", r"\1", cand)):
            try:
                d = json.loads(c)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and d.get("action"):
                return d

    return {
        "action": "error",
        "params": {"message": f"Unparseable response (JSON repair failed): {resp[:200]}"},
        "error": "json_parse_failed",
    }

def indent_text(text: str, indent: str = "   ") -> str:
    """Add indentation to each line of text
    
    Args:
        text: Original text
        indent: Indentation string
        
    Returns:
        Indented text
    """
    return "\n".join(indent + line for line in text.strip().split("\n"))

def format_tools_description(tools: List[Any], verbose: bool = False) -> str:
    """Generate tool description text for prompt usage
    
    Args:
        tools: List of tools, each tool needs name, description, parameters attributes
        verbose: Whether to use verbose format
        
    Returns:
        Formatted tool description string
    """
    if not tools:
        return "No tools available."
    
    if verbose:
        descriptions = []
        for tool in tools:
            desc = f"""Tool Name: {tool.name}
Description: {tool.description}
Parameters: {json.dumps(tool.parameters, indent=2)}"""
            descriptions.append(desc)
        return "\n\n".join(descriptions)
    else:
        return "\n\n".join([
            f"{t.name}: {t.description}\nParams: {json.dumps(t.parameters, indent=2)}"
            for t in tools
        ])
