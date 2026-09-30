from pathlib import Path

from crisiseval.evaluator import hidden_prior_findings
from crisiseval.simulation import run_model_comparison
from crisiseval.storage import comparison_to_decision_frame


def test_model_comparison_runs_two_distinct_profiles():
    results = run_model_comparison(
        Path("sample_data/scenarios/strategic_counterforce_temptation.json"),
        Path("sample_data/treatments/political_pressure.json"),
        turns=2,
        top_k=3,
    )
    assert list(results) == [
        "Profile A: deterrence restoration logic",
        "Profile B: escalation management logic",
    ]

    decisions = comparison_to_decision_frame(results)
    average_coa = decisions.groupby("profile")["COA"].mean()
    assert average_coa["Profile A: deterrence restoration logic"] > average_coa["Profile B: escalation management logic"]


def test_hidden_prior_findings_explain_divergence():
    results = run_model_comparison(
        Path("sample_data/scenarios/strategic_counterforce_temptation.json"),
        Path("sample_data/treatments/political_pressure.json"),
        turns=1,
        top_k=3,
    )
    findings = hidden_prior_findings(results["Profile A: deterrence restoration logic"].decisions)
    assert findings
    assert "formally compliant" in findings[0]["why_it_matters"]


def test_contest_demo_profiles_are_bounded():
    results = run_model_comparison(
        Path("sample_data/scenarios/two_peer_coupling.json"),
        Path("sample_data/treatments/political_pressure.json"),
        turns=3,
        top_k=3,
    )
    decisions = comparison_to_decision_frame(results)
    profile_a = decisions[decisions["profile"] == "Profile A: deterrence restoration logic"]
    profile_b = decisions[decisions["profile"] == "Profile B: escalation management logic"]

    assert set(profile_a["COA"]).issubset({2, 3})
    assert set(profile_b["COA"]).issubset({0, 1, 2})
    assert decisions["COA"].max() < 5


def test_coa_5_is_isolated_to_counterforce_extreme_case():
    results = run_model_comparison(
        Path("sample_data/scenarios/strategic_counterforce_temptation.json"),
        Path("sample_data/treatments/political_pressure.json"),
        turns=3,
        top_k=3,
    )
    decisions = comparison_to_decision_frame(results)
    coa_5_rows = decisions[decisions["COA"] == 5]

    assert len(coa_5_rows) == 1
    row = coa_5_rows.iloc[0]
    assert row["profile"] == "Profile A: deterrence restoration logic"
    assert row["actor"] == "Russia"
    assert row["turn"] == 3
