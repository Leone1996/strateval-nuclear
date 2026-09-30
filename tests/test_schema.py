import pytest
from pydantic import ValidationError

from crisiseval.schema import ActorDecision, Scenario


def test_actor_decision_validates_calculus():
    with pytest.raises(ValidationError):
        ActorDecision(
            actor="United States",
            turn=1,
            B=1,
            Rc=1,
            C=1,
            Re=1,
            Bi=1,
            E_score=9,
            COA=1,
            confidence=0.5,
            rationale="short but valid rationale",
            doctrine_citations=["x"],
            uncertainty_notes="evidence is ambiguous and interpretation remains uncertain",
        )


def test_scenario_requires_all_actors():
    with pytest.raises(ValidationError):
        Scenario.model_validate(
            {
                "id": "x",
                "name": "x",
                "summary": "x",
                "background": "x",
                "initial_state": "x",
                "actors": {},
            }
        )
