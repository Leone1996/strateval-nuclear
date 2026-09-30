from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crisiseval.agents import STRATEGIC_PROFILES  # noqa: E402
from crisiseval.corpus import load_markdown_corpus  # noqa: E402
from crisiseval.model_adapters import GeminiAdapter, actor_decision_response_schema, build_live_actor_prompt  # noqa: E402
from crisiseval.retrieval import TfidfRetriever, build_query, priority_terms_for  # noqa: E402
from crisiseval.schema import Scenario, Treatment, TurnState, data_path  # noqa: E402
from crisiseval.simulation import load_json_model  # noqa: E402


def main() -> int:
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    print(f"Python executable: {sys.executable}")
    print("provider=gemini")
    print(f"model={model}")
    print(f"GEMINI_API_KEY loaded={bool(api_key)}")

    scenario = load_json_model(data_path("scenarios", "dual_use_missile_ambiguity.json"), Scenario)
    treatment = load_json_model(data_path("treatments", "ambiguous_intelligence.json"), Treatment)
    corpus_chunks = load_markdown_corpus(data_path("corpus"))
    retriever = TfidfRetriever(corpus_chunks)
    actor = "India"
    turn = 1
    profile = "deterrence_restoration"
    chunks = retriever.query(
        build_query(scenario, actor, treatment),
        top_k=3,
        priority_terms=priority_terms_for(scenario, actor),
    )
    prompt = build_live_actor_prompt(
        actor=actor,
        scenario=scenario,
        treatment=treatment,
        turn_state=TurnState(turn=turn, narrative=f"Initial crisis signals are ambiguous. {scenario.initial_state}"),
        chunks=chunks,
        profile_label=STRATEGIC_PROFILES[profile]["label"],
        profile_config=STRATEGIC_PROFILES[profile],
    )
    config = {
        "provider": "gemini",
        "model": model,
        "api_key": api_key,
        "actor": actor,
        "turn": turn,
        "chunks": chunks,
        "profile": profile,
        "log_dir": ROOT / "outputs" / "smoke_gemini_coded_primary",
        "call_id": 1,
        "fallback_enabled": False,
        "gemini_native_enabled": False,
        "gemini_max_retries": 2,
        "coded_repair_enabled": True,
        "temperature": 0.2,
    }

    try:
        result = GeminiAdapter().generate_actor_decision(prompt, actor_decision_response_schema(), config)
        decision = result.actor_decision
        print(f"validation_status={result.validation_status}")
        print(f"coding_method={result.coding_method}")
        print(f"coded_fallback_attempted={result.coded_fallback_attempted}")
        print(f"coded_fallback_success={result.coded_fallback_success}")
        print(f"coded_fallback_error={result.coded_fallback_error}")
        print(f"raw_response_path={result.raw_response_path}")
        print(f"parse_error={result.parse_error}")
        print(f"adapter_notes={result.notes}")
        if decision is not None:
            print(f"actor={decision.actor}")
            print(f"turn={decision.turn}")
            print(f"COA={decision.COA}")
            print(f"E_score={decision.E_score}")
            print(f"confidence={decision.confidence}")
        else:
            print(f"actor={result.actor}")
            print(f"turn={result.turn}")
            print("COA=")
            print("E_score=")
            print("confidence=")
            return 1
        return 0 if result.validation_status in {"coded_from_machine_block", "coded_from_machine_block_retried", "coded_from_machine_block_repaired", "coded_from_narrative"} else 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
