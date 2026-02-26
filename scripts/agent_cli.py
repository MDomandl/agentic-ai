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
    pending_user_input: str | None = None

    print("\n--- Agent Loop --- (Strg+C zum Abbruch)\n")

    # Interaktiv: wenn Agent needs_input liefert, fragen wir den User.
    while True:
        res = agent.step(state, user_input=pending_user_input)
        state = res.state
        pending_user_input = None

        print(f"res.status > {res.status} \n")
        print(f"state.iters > {res.state.iters} \n")
        print(f"res.state.memory > {res.state.memory} \n")

        if res.message:
            print(f"Agent > {res.message}")

        if res.status in (AgentStatus.ERROR):
            continue

        if res.status in (AgentStatus.DONE):
            break

        if res.status == AgentStatus.CONTINUE:
            # kein User-Input nötig -> nächste Iteration macht den nächsten step()
            continue

        if res.status == AgentStatus.NEEDS_INPUT:
            # genau hier ist der EINZIGE Ort für User input
            pending_user_input = input("Du > ").strip()
            continue

        if res.status == AgentStatus.RESPONDED:
            break
            # pending_user_input = input("Du > ").strip()
            # if pending_user_input in ("quit", "exit"):
            #     break
            # continue

        # Fallback
        pending_user_input = input("Du > ").strip()
        if pending_user_input in ('quit', 'exit'):
            break


if __name__ == "__main__":
    raise SystemExit(main())
