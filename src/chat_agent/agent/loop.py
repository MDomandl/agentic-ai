from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, List, TYPE_CHECKING

from pydantic import BaseModel, Field

from chat_agent.llm.structured import parse_structured
from chat_agent.policy.risk import RiskLevel, decide
from chat_agent.agent.types import AgentState, AgentResult, AgentStatus

ToolFn = Callable[[Dict[str, Any]], str]

class AgentAction(str, Enum):
    TOOL = "tool"
    RESPOND = "respond"
    DONE = "done"

class AgentDecision(BaseModel):
    action: AgentAction = Field(description="Nächste Aktion: tool|respond|done")
    tool_name: Optional[str] = Field(default=None, description="Tool-Name (nur bei action=tool)")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Tool-Args (nur bei action=tool)")
    final_answer: Optional[str] = Field(default=None, description="Antwort (nur bei action=respond)")
    confidence_score: float = Field(description="0..1: Sicherheit der Entscheidung", ge=0.0, le=1.0)

DeciderFn = Callable[[str, str], "AgentDecision"]  # (system_prompt, user_payload) -> decision

@dataclass
class AgentConfig:
    max_iters: int = 50
    max_memory_chars: int = 6000
    risk_level: RiskLevel = RiskLevel.LOW
    allow_tools: bool = True


def _trim_memory(memory: List[Tuple[str, str]], max_chars: int) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    total = 0
    for role, content in reversed(memory):
        total += len(content)
        if total > max_chars and out:
            break
        out.append((role, content))
    return list(reversed(out))


def _build_system_prompt(tools: Dict[str, ToolFn]) -> str:
    tool_list = ", ".join(sorted(tools.keys())) if tools else "(keine)"
    return (
        "Du bist ein Agenten-Controller. Du gibst AUSSCHLIESSLICH JSON gemäß Schema zurück.\n"
        "Du steuerst einen Loop (decide → act → observe).\n\n"       
        "Regeln:\n"
        "- Wenn du ein Tool brauchst: action='tool', tool_name aus Tool-Liste wählen, tool_args setzen.\n"
        "- Wenn du direkt antworten kannst: action='respond' und final_answer setzen.\n"
        "- Wenn das Ziel erreicht ist und keine Antwort nötig ist: action='done'.\n"
        "- Erfinde keine Fakten. Wenn Infos fehlen: fordere Klärung an.\n"
        "- Halte final_answer kurz und konkret.\n\n"
        "Tool-Spezifikation:\n"
        "- add: tool_args hat genau die Keys {\"a\": number, \"b\": number}\n"
        "- echo: tool_args hat {\"text\": string}\n"
        f"Tool-Liste: {tool_list}\n"
    )


def _build_user_payload(state: AgentState, cfg: AgentConfig) -> str:
    mem = _trim_memory(state.memory, cfg.max_memory_chars)

    def _last_user_message(memory):
        for role, content in reversed(memory):
            if role == "user" and content.strip():
                return content.strip()
        return ""

    payload = {
        "user_goal": state.user_goal,
      #  "current_user_input": _last_user_message(state.memory),
        "memory": [{"role": r, "content": c} for r, c in mem],
        "observations": state.observations[-10:],
        "iteration": state.iters,
    }
    return json.dumps(payload, ensure_ascii=False)


class AgentLoop:
    """
    Fortsetzbarer Agenten-Loop.
    - step(): ein Iterationsschritt (inkl. ggf. Tool-Ausführung)
    - run_cli(): einfacher CLI-Runner (siehe scripts/agent_cli.py)
    """

    def __init__(self, tools: Optional[Dict[str, ToolFn]] = None, cfg: Optional[AgentConfig] = None, decider: Optional[DeciderFn] = None):
        self.tools = tools or {}
        self.cfg = cfg or AgentConfig()
        self._system_prompt = _build_system_prompt(self.tools)
        self._decider = decider

    def start(self, user_goal: str) -> AgentState:
        st = AgentState(user_goal=user_goal)
        st.memory.append(("user", user_goal))
        return st

    def step(self, state: AgentState, user_input: Optional[str] = None) -> AgentResult:
        # 0) optional neuen User-Input übernehmen (Bestätigung/Präzisierung)
        if user_input is not None and user_input.strip():
            state.memory.append(("user", user_input.strip()))

        # Stop-Schutz
        if state.iters >= self.cfg.max_iters:
            return AgentResult(
                status=AgentStatus.ERROR,
                message="Iterationslimit erreicht. Bitte Ziel präzisieren oder max_iters erhöhen.",
                state=state,
            )

        state.iters += 1

        # 1) DECIDE
        user_payload = _build_user_payload(state, self.cfg)

        if self._decider is not None:
            decision = self._decider(self._system_prompt, user_payload)
        else:
            decision = parse_structured(
                AgentDecision,
                user_payload,
                system=self._system_prompt,
                max_retries=2,
            )

        mode = decide(decision.confidence_score, self.cfg.risk_level)

        # Risk/Confidence Logging
        if mode in ("log", "manual"):
            state.observations.append(
                f"RISK_GATE mode={mode} risk={self.cfg.risk_level.value} conf={decision.confidence_score:.2f}"
            )

        # 2) Handle RESPOND
        if decision.action == AgentAction.RESPOND:
            if mode == "manual" and self.cfg.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH):
                msg = (
                    f"Ich bin mir (confidence={decision.confidence_score:.2f}) zu unsicher für "
                    f"RiskLevel {self.cfg.risk_level.value}. Bitte präzisiere oder bestätige."
                )
                state.memory.append(("assistant", msg))
                return AgentResult(status=AgentStatus.NEEDS_INPUT, message=msg, state=state, request_kind="clarify")

            answer = (decision.final_answer or "").strip()
            if not answer:
                answer = "Dazu fehlen mir Informationen. Kannst du präzisieren, was genau du willst?"
            state.memory.append(("assistant", answer))
            return AgentResult(status=AgentStatus.RESPONDED, message=answer, state=state)

        # 3) Handle DONE
        if decision.action == AgentAction.DONE:
            msg = "Erledigt."
            state.memory.append(("assistant", msg))
            return AgentResult(status=AgentStatus.DONE, message=msg, state=state)

        # 4) Handle TOOL
        if decision.action == AgentAction.TOOL:
            if not self.cfg.allow_tools:
                msg = "Tool-Nutzung ist deaktiviert. Bitte erlaube Tools oder ändere das Ziel."
                state.memory.append(("assistant", msg))
                return AgentResult(status=AgentStatus.NEEDS_INPUT, message=msg, state=state, request_kind="confirm")

            if mode == "manual" and self.cfg.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH):
                msg = (
                    f"Ich bin mir (confidence={decision.confidence_score:.2f}) zu unsicher für eine "
                    f"Tool-Aktion im RiskLevel {self.cfg.risk_level.value}. Bitte bestätige oder präzisiere."
                )
                state.memory.append(("assistant", msg))
                return AgentResult(status=AgentStatus.NEEDS_INPUT, message=msg, state=state, request_kind="confirm")

            tool_name = (decision.tool_name or "").strip()
            if tool_name not in self.tools:
                state.observations.append(f"TOOL_ERROR: unknown tool '{tool_name}'")
                # weiterlaufen lassen: Agent soll umplanen
                return AgentResult(status=AgentStatus.NEEDS_INPUT, message=f"Unbekanntes Tool: {tool_name}", state=state)

            try:
                result = self.tools[tool_name](decision.tool_args or {})
                state.observations.append(f"TOOL_OK {tool_name}: {result}")
            except Exception as e:
                state.observations.append(f"TOOL_ERROR {tool_name}: {type(e).__name__}: {e}")

            # Nach Tool-Aktion nicht sofort raus: CLI ruft step() erneut auf (mit None input),
            # damit der Agent aus Observations eine Antwort ableiten kann.
            return AgentResult(status=AgentStatus.CONTINUE, message="(Tool ausgeführt) Weiter...", state=state)

        # Fallback
        msg = "Unbekannte Agentenaktion."
        state.memory.append(("assistant", msg))
        return AgentResult(status=AgentStatus.ERROR, message=msg, state=state)
