from __future__ import annotations

from chat_agent.agent.loop import AgentLoop, AgentConfig
from chat_agent.agent.tools_builtin import tool_echo, tool_add
from chat_agent.policy.risk import RiskLevel
from chat_agent.agent.types import AgentStatus


def main() -> int:
    tools = {"echo": tool_echo, "add": tool_add}
    cfg = AgentConfig(max_iters=50, risk_level=RiskLevel.LOW, allow_tools=True)
    agent = AgentLoop(tools=tools, cfg=cfg)

    goal = input("Ziel (goal) > ").strip()
    state = agent.start(goal)
    user_input = goal

    print("\n--- Agent Loop --- (Strg+C zum Abbruch)\n")

    # Interaktiv: wenn Agent needs_input liefert, fragen wir den User.
    while True:
        res = agent.step(state, user_input=user_input)

        print(f"res.status > {res.status} \n")
        print(f"state.iters > {res.state.iters} \n")
        print(f"res.status > {res.state.memory} \n")

        if res.status in (AgentStatus.RESPONDED, AgentStatus.DONE, AgentStatus.ERROR):
            print(f"Agent > {res.message}")
            if res.status in (AgentStatus.DONE, AgentStatus.ERROR):
                return 0 if res.status == AgentStatus.DONE else 2
            # responded: Ziel evtl. noch nicht „done“ -> weiter, falls Agent noch iterieren will
            # (hier könntest du optional breaken, wir lassen es erstmal weiterlaufen)
            state = res.state
            # Optional: direkt beenden nach erster Antwort
            # return 0

        # needs_input:
        # Wenn es nur "Weiter..." nach Tool war, geben wir keinen User-Input,
        # sondern lassen den Agent aus den Observations weiterdenken.
        if res.message.strip() == "(Tool ausgeführt) Weiter...":
            state = res.state
            continue

        #print(f"Agent > {res.message}")
        user_input = input("Du > ").strip()


if __name__ == "__main__":
    raise SystemExit(main())
