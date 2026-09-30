from crisiseval.schema import escalation_score


def test_escalation_score_rounds_and_uses_formula():
    assert escalation_score(3, 2, 2, 3, 1) == 0.83


def test_escalation_score_handles_zero_denominator():
    assert escalation_score(2, 3, 0, 0, 0) == 5.0
