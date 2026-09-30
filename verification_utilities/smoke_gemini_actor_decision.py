from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crisiseval.agents import STRATEGIC_PROFILES  # noqa: E402
from crisiseval.corpus import load_markdown_corpus  # noqa: E402
from crisiseval.model_adapters import (  # noqa: E402
    GeminiAdapter,
    actor_decision_response_schema,
    build_live_actor_prompt,
    parse_json_object,
)
from crisiseval.retrieval import TfidfRetriever, build_query, priority_terms_for  # noqa: E402
from crisiseval.schema import Scenario, Treatment, TurnState, data_path  # noqa: E402
from crisiseval.simulation import load_json_model  # noqa: E402


def failure_category(result) -> str:
    error = (result.parse_error or "").lower()
    status = (result.validation_status or "").lower()
    if status in {"valid", "valid_repaired", "native_structured_valid", "native_structured_repaired"}:
        return "none"
    if "missing gemini_api_key" in error:
        return "missing key"
    if "not found" in error or "model" in error and "not" in error:
        return "invalid model"
    if "quota" in error or "billing" in error or "resource_exhausted" in error:
        return "quota/billing error"
    if status in {"refusal", "partial_refusal", "safety_reframing", "doctrine_only_answer"}:
        return "safety block/refusal"
    if status == "invalid_json":
        return "malformed JSON or parsing/extraction failure"
    if status == "schema_invalid":
        return "schema validation failure"
    if status == "api_error":
        return "SDK/API syntax error or provider error"
    return "unknown"


def main() -> int:
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    print(f"Python executable: {sys.executable}")
    print(f"GEMINI_MODEL: {model}")
    print(f"GEMINI_API_KEY loaded: {bool(api_key)}")

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
    log_root = ROOT / "outputs" / "smoke_gemini_actor_decision"
    config = {
        "provider": "gemini",
        "model": model,
        "api_key": api_key,
        "actor": actor,
        "turn": turn,
        "chunks": chunks,
        "profile": profile,
        "log_dir": log_root,
        "call_id": 1,
        "fallback_enabled": False,
        "repair_enabled": True,
        "temperature": 0.2,
    }

    try:
        result = GeminiAdapter().generate_actor_decision(prompt, actor_decision_response_schema(), config)
        raw_path = Path(result.raw_response_path) if result.raw_response_path else None
        raw_text = raw_path.read_text(encoding="utf-8") if raw_path and raw_path.exists() else ""
        parsed, parse_error = parse_json_object(raw_text)

        print(f"raw provider response path: {raw_path or ''}")
        print("raw provider response first 1500 chars:")
        print(raw_text[:1500])
        print("extracted JSON if any:")
        print(json.dumps(parsed, indent=2) if parsed is not None else f"<none: {parse_error}>")
        print(f"validation_status = {result.validation_status}")
        print(f"parse_error = {result.parse_error}")
        print(f"repair_attempted = {result.repair_attempted}")
        print(f"refusal_or_reframing_type = {result.refusal_or_reframing_type}")
        print(f"adapter_notes = {result.notes}")
        print(f"failure_category = {failure_category(result)}")
        if result.actor_decision is not None:
            print("ActorDecision:")
            print(json.dumps(result.actor_decision.model_dump(), indent=2))
        usable_statuses = {
            "valid",
            "valid_repaired",
            "native_structured_valid",
            "native_structured_repaired",
            "coded_from_machine_block",
            "coded_from_machine_block_retried",
            "coded_from_machine_block_repaired",
            "coded_from_narrative",
        }
        if result.validation_status not in usable_statuses or result.actor_decision is None:
            return 1
        decision = result.actor_decision
        if decision.provider != "gemini" or decision.live_or_mock != "live" or decision.fallback_used:
            return 1
        return 0
    except Exception:
        print("failure_category = smoke script exception")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
