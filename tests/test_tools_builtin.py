from chat_agent.agent.tools_builtin import tool_add

def test_tool_add_accepts_a_b():
    assert tool_add({"a": 12.5, "b": 7.25}) == "19.75"

def test_tool_add_accepts_num1_num2():
    assert tool_add({"num1": 12.5, "num2": 7.25}) == "19.75"

def test_tool_add_accepts_x_y():
    assert tool_add({"x": 12.5, "y": 7.25}) == "19.75"
