from chat_agent.agent.loop import AgentLoop, AgentConfig, AgentDecision, AgentAction
from chat_agent.agent.tools_builtin import tool_add
from chat_agent.policy.risk import RiskLevel
from chat_agent.agent.types import AgentStatus

def test_agentloop_tool_then_respond():
    # Fake-Decider: erst Tool, dann Respond
    decisions = [
        AgentDecision(action=AgentAction.TOOL, tool_name="add", tool_args={"a": 12.5, "b": 7.25}, final_answer=None, confidence_score=0.99),
        AgentDecision(action=AgentAction.RESPOND, tool_name=None, tool_args={}, final_answer="19.75", confidence_score=0.99),
    ]

    def fake_decider(system_prompt: str, user_payload: str):
        return decisions.pop(0)

    agent = AgentLoop(
        tools={"add": tool_add},
        cfg=AgentConfig(max_iters=5, risk_level=RiskLevel.LOW, allow_tools=True),
        decider=fake_decider,
    )

    st = agent.start("Rechne 12.5 + 7.25 aus und sag mir das Ergebnis.")
    r1 = agent.step(st)
    assert r1.status == AgentStatus.NEEDS_INPUT  # Tool wurde ausgeführt → Weiter...
    assert any("TOOL_OK add: 19.75" in o for o in r1.state.observations)

    r2 = agent.step(r1.state)
    assert r2.status == AgentStatus.RESPONDED
    assert r2.message.strip() == "19.75"
