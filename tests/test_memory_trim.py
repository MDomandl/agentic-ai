from chat_agent.agent.loop import _trim_memory

def test_trim_memory_keeps_recent():
    mem = [("user", "a"*1000)] * 10
    out = _trim_memory(mem, max_chars=2500)
    assert len(out) >= 2
    assert sum(len(c) for _, c in out) <= 3000
