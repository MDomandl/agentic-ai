from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass(frozen=True)
class RiskPolicy:
    auto_threshold: float
    log_threshold: float

POLICIES: dict[RiskLevel, RiskPolicy] = {
    RiskLevel.LOW:    RiskPolicy(auto_threshold=0.75, log_threshold=0.50),
    RiskLevel.MEDIUM: RiskPolicy(auto_threshold=0.85, log_threshold=0.70),
    RiskLevel.HIGH:   RiskPolicy(auto_threshold=0.95, log_threshold=0.85),
}

def decide(confidence: float, level: RiskLevel) -> str:
    """
    Returns: 'auto' | 'log' | 'manual'
    """
    p = POLICIES[level]
    if confidence >= p.auto_threshold:
        return "auto"
    if confidence >= p.log_threshold:
        return "log"
    return "manual"
