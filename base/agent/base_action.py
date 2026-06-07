# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

from abc import abstractmethod
from typing import Dict, Any
from pydantic import BaseModel


class BaseAction(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = None

    @abstractmethod
    async def __call__(self, **kwargs) -> str:
        """Execute the action with given parameters."""

    def to_param(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
