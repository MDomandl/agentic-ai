import json
from chat_agent.agent.loop import _build_user_payload, AgentConfig
from chat_agent.agent.types import AgentState

def test_current_user_input_is_latest_user_message():
    st = AgentState(user_goal="x")
    st.memory.append(("user", "blöder esel"))
    st.memory.append(("assistant", "Bitte präzisieren"))
    st.memory.append(("user", "Rechne 12.5 + 7.25 aus und sag mir das Ergebnis."))
    st.memory.append(("assistant", "19.75"))
    st.memory.append(("user", "Rechne 120.5 + 70.25 aus und sag mir das Ergebnis."))

    payload = json.loads(_build_user_payload(st, AgentConfig()))
    assert payload["current_user_input"].startswith("Rechne 120.5 + 70.25")
