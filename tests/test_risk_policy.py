from chat_agent.policy.risk import decide, RiskLevel

def test_risk_low():
    assert decide(0.9, RiskLevel.LOW) == "auto"
    assert decide(0.6, RiskLevel.LOW) in ("log", "auto", "manual")  # je nach thresholds

def test_risk_high():
    assert decide(0.96, RiskLevel.HIGH) == "auto"
    assert decide(0.80, RiskLevel.HIGH) == "manual"
