from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


@dataclass
class AgentState:
    user_goal: str
    memory: List[Tuple[str, str]] = field(default_factory=list)  # (role, content)
    observations: List[str] = field(default_factory=list)
    iters: int = 0


@dataclass
class AgentResult:
    status: AgentStatus  # "responded" | "needs_input" | "done" | "error"
    message: str
    state: AgentState
    request_kind: Optional[str] = None  # "confirm" | "clarify"

    def __post_init__(self):
        if not isinstance(self.status, AgentStatus):
            raise TypeError(f"status must be AgentStatus, got {type(self.status)}: {self.status!r}")

class AgentStatus(str, Enum):
    RESPONDED = "responded"
    NEEDS_INPUT = "needs_input" # wirklich User-Input nötig
    CONTINUE = "continue"  # kein User-Input nötig, nächster Step sofort
    DONE = "done"
    ERROR = "error"