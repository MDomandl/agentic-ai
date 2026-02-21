from chat_agent.agent.loop import AgentLoop, AgentConfig
from chat_agent.agent.tools_builtin import tool_echo, tool_add
from chat_agent.policy.risk import RiskLevel

if __name__ == "__main__":
    tools = {
        "echo": tool_echo,
        "add": tool_add,
    }

    #cfg = AgentConfig(max_iters=6, risk_level=RiskLevel.LOW, allow_tools=True)
    cfg = AgentConfig(max_iters=6, risk_level=RiskLevel.HIGH, allow_tools=True)
    agent = AgentLoop(tools=tools, cfg=cfg)

    #goal = "Mach das mal bitte."
    goal = "Rechne 12.5 + 7.25 aus und sag mir das Ergebnis."
    print(agent.start(goal))
