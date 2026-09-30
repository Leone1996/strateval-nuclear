from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .analytics import compute_summary_metrics, evaluation_observations_for_result, failure_modes_for_result
from .agents import MockModel, STRATEGIC_PROFILES
from .actor_registry import (
    actor_sequence_for_selection,
    attach_source_metadata,
    source_card_rows_for_actor,
    source_chunks_for_actor,
    source_metadata_for_decision,
)
from .corpus import load_markdown_corpus
from .evaluator import evaluate_decisions
from .model_adapters import (
    MockModelAdapter,
    actor_decision_response_schema,
    build_live_actor_prompt,
    default_model_for,
    generate_actor_decision,
)
from .retrieval import TfidfRetriever, build_query, priority_terms_for
from .schema import AdapterResult, Scenario, SimulationResult, Treatment, TurnState, data_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROVIDER_LABELS = {
    "mock": "MockModel",
    "openai": "OpenAI live",
    "gemini": "Gemini live",
    "anthropic": "Anthropic live",
}
PROVIDER_NAMES = {
    "mock": "MockModel",
    "openai": "OpenAI",
    "gemini": "Gemini",
    "anthropic": "Anthropic",
}
NATIVE_LIVE_STATUSES = {"valid", "valid_repaired", "native_structured_valid", "native_structured_repaired"}
CODED_LIVE_STATUSES = {
    "coded_from_machine_block",
    "coded_from_machine_block_retried",
    "coded_from_narrative",
    "coded_from_machine_block_repaired",
    "coded_from_narrative_repaired",
}
LIVE_VALID_STATUSES = NATIVE_LIVE_STATUSES | CODED_LIVE_STATUSES


def load_json_model(path: Path, model_cls):
    return model_cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


def list_scenarios() -> list[Path]:
    return sorted(data_path("scenarios").glob("*.json"))


def list_treatments() -> list[Path]:
    return sorted(data_path("treatments").glob("*.json"))


def run_simulation(
    scenario_path: Path,
    treatment_path: Path,
    turns: int = 3,
    top_k: int = 5,
    model_mode: str = "MockModel",
    strategic_profile: str = "balanced",
    provider: str = "mock",
    adapter_config: dict | None = None,
    selected_actors: list[str] | None = None,
) -> SimulationResult:
    scenario = load_json_model(scenario_path, Scenario)
    treatment = load_json_model(treatment_path, Treatment)
    selected_actor_sequence = actor_sequence_for_selection(scenario, selected_actors)
    corpus_chunks = load_markdown_corpus(data_path("corpus"))
    retriever = TfidfRetriever(corpus_chunks)
    model = MockModel(profile=strategic_profile)
    mock_adapter = MockModelAdapter()
    adapter_config = adapter_config or {}
    provider = provider or "mock"

    decisions = []
    adapter_results: list[AdapterResult] = []
    retrieved_by_id = {}
    source_card_rows = []
    run_stamp = adapter_config.get("run_stamp") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_root = Path(adapter_config.get("log_root") or PROJECT_ROOT / "outputs" / "live_runs" / run_stamp)
    call_index = 0
    for turn in range(1, turns + 1):
        turn_state = TurnState(
            turn=turn,
            narrative=_turn_narrative(scenario, treatment, turn),
            prior_decisions=list(decisions),
        )
        for actor in selected_actor_sequence:
            doctrine_pack, source_cards, source_chunks = source_chunks_for_actor(actor)
            source_metadata = source_metadata_for_decision(
                selected_actor_sequence,
                actor,
                doctrine_pack,
                source_cards,
            )
            source_card_rows.extend(source_card_rows_for_actor(actor))
            query = build_query(scenario, actor, treatment)
            actor_chunks = retriever.query(query, top_k=top_k, priority_terms=priority_terms_for(scenario, actor))
            actor_chunks = _dedupe_chunks([*source_chunks, *actor_chunks])
            for chunk in actor_chunks:
                retrieved_by_id[chunk.id] = chunk
            if provider == "mock":
                decision = model.decide(actor, scenario, treatment, turn_state, actor_chunks)
                attach_source_metadata(decision, source_metadata)
                adapter_results.append(mock_adapter.result_for_decision(decision, model=model.name))
                attach_source_metadata(adapter_results[-1], source_metadata)
                decisions.append(decision)
                continue

            call_index += 1
            provider_model = adapter_config.get("models", {}).get(provider) or adapter_config.get("model") or default_model_for(provider)
            prompt = build_live_actor_prompt(
                actor=actor,
                scenario=scenario,
                treatment=treatment,
                turn_state=turn_state,
                chunks=actor_chunks,
                profile_label=STRATEGIC_PROFILES[strategic_profile]["label"],
                profile_config=STRATEGIC_PROFILES[strategic_profile],
            )
            prompt = prompt.replace(
                "Retrieved public/unclassified corpus snippets:",
                "Evidence basis and retrieved public/unclassified corpus snippets:",
            )
            result = generate_actor_decision(
                prompt,
                actor_decision_response_schema(),
                {
                    **adapter_config,
                    "provider": provider,
                    "model": provider_model,
                    "actor": actor,
                    "turn": turn,
                    "chunks": actor_chunks,
                    "profile": strategic_profile,
                    "log_dir": log_root,
                    "call_id": call_index,
                },
            )
            if result.actor_decision is not None:
                attach_source_metadata(result.actor_decision, source_metadata)
                attach_source_metadata(result, source_metadata)
                decisions.append(result.actor_decision)
            elif adapter_config.get("fallback_enabled", True):
                fallback = model.decide(actor, scenario, treatment, turn_state, actor_chunks)
                attach_source_metadata(fallback, source_metadata)
                fallback.provider = provider
                fallback.model = provider_model
                fallback.live_or_mock = "mock"
                fallback.validation_status = "fallback_used"
                fallback.fallback_used = True
                fallback.raw_response_path = result.raw_response_path
                fallback.parse_error = result.parse_error
                fallback.repair_attempted = result.repair_attempted
                fallback.refusal_or_reframing_type = result.refusal_or_reframing_type
                fallback.coding_method = result.coding_method
                fallback.coded_fallback_attempted = result.coded_fallback_attempted
                fallback.coded_fallback_success = result.coded_fallback_success
                fallback.coded_fallback_error = result.coded_fallback_error
                fallback.gemini_attempt_count = result.gemini_attempt_count
                fallback.gemini_retry_attempted = result.gemini_retry_attempted
                fallback.gemini_retry_success = result.gemini_retry_success
                fallback.gemini_retry_error = result.gemini_retry_error
                fallback.adapter_notes = f"{PROVIDER_LABELS.get(provider, provider)} returned {result.validation_status}; MockModel fallback used."
                result.actor_decision = fallback
                attach_source_metadata(result, source_metadata)
                result.live_or_mock = "mock"
                result.validation_status = "fallback_used"
                result.fallback_used = True
                result.coding_method = fallback.coding_method
                result.coded_fallback_attempted = fallback.coded_fallback_attempted
                result.coded_fallback_success = fallback.coded_fallback_success
                result.coded_fallback_error = fallback.coded_fallback_error
                result.gemini_attempt_count = fallback.gemini_attempt_count
                result.gemini_retry_attempted = fallback.gemini_retry_attempted
                result.gemini_retry_success = fallback.gemini_retry_success
                result.gemini_retry_error = fallback.gemini_retry_error
                result.notes = fallback.adapter_notes
                decisions.append(fallback)
            else:
                attach_source_metadata(result, source_metadata)
            adapter_results.append(result)

    evaluations = evaluate_decisions(decisions, treatment, scenario)
    for decision in decisions:
        matching = next(
            evaluation
            for evaluation in evaluations
            if evaluation.actor == decision.actor and evaluation.turn == decision.turn
        )
        decision.red_flags = matching.red_flags
        decision.observation_flags = matching.observation_flags

    result = SimulationResult(
        scenario=scenario,
        treatment=treatment,
        top_k=top_k,
        model_mode=model_mode if provider != "mock" else model.name,
        retrieved_chunks=list(retrieved_by_id.values()),
        decisions=decisions,
        evaluations=evaluations,
        adapter_results=adapter_results,
        selected_actor_set=selected_actor_sequence,
        source_cards_used=_dedupe_source_card_rows(source_card_rows),
    )
    result.summary_metrics = compute_summary_metrics(result)
    result.failure_modes = failure_modes_for_result(result)
    result.evaluation_observations = evaluation_observations_for_result(result)
    return result


def run_model_comparison(
    scenario_path: Path,
    treatment_path: Path,
    turns: int = 3,
    top_k: int = 5,
    profiles: list[str] | None = None,
    providers: list[str] | None = None,
    adapter_config: dict | None = None,
    selected_actors: list[str] | None = None,
) -> dict[str, SimulationResult]:
    profiles = profiles or ["deterrence_restoration", "escalation_management"]
    providers = providers or ["mock"]
    results: dict[str, SimulationResult] = {}
    for provider in providers:
        for profile in profiles:
            result = run_simulation(
                scenario_path,
                treatment_path,
                turns=turns,
                top_k=top_k,
                model_mode=PROVIDER_LABELS.get(provider, provider),
                strategic_profile=profile,
                provider=provider,
                adapter_config=adapter_config,
                selected_actors=selected_actors,
            )
            result.model_mode = actual_provider_label(provider, result)
            results[result_execution_label(provider, profile, result, providers)] = result
    comparison_available = complete_profile_comparison_available(results)
    for result in results.values():
        result.summary_metrics["complete_profile_comparison_available"] = comparison_available
    return results


def result_execution_label(
    provider: str,
    profile: str,
    result: SimulationResult,
    providers: list[str] | None = None,
) -> str:
    profile_label = STRATEGIC_PROFILES[profile]["label"]
    providers = providers or [provider]
    if provider == "mock" and providers == ["mock"]:
        return profile_label

    provider_label = actual_provider_label(provider, result)
    return f"{provider_label} · {profile_label}"


def actual_provider_label(provider: str, result: SimulationResult) -> str:
    provider_name = PROVIDER_NAMES.get(provider, provider.title())
    if provider == "mock":
        return provider_name
    if any(decision.fallback_used for decision in result.decisions):
        return f"{provider_name} requested · MockModel fallback used"
    coded_count = sum(
        1
        for decision in result.decisions
        if decision.validation_status in CODED_LIVE_STATUSES
        and decision.live_or_mock == "live"
        and not decision.fallback_used
    )
    live_valid_count = sum(
        1
        for decision in result.decisions
        if decision.validation_status in LIVE_VALID_STATUSES
        and decision.live_or_mock == "live"
        and not decision.fallback_used
    )
    failed_live_count = sum(
        1
        for adapter_result in result.adapter_results
        if adapter_result.provider != "mock"
        and adapter_result.validation_status not in LIVE_VALID_STATUSES
        and not adapter_result.fallback_used
    )
    if coded_count and failed_live_count:
        return f"{provider_name} live · coded output partial"
    if coded_count:
        return f"{provider_name} live · coded output"
    if live_valid_count and failed_live_count:
        return f"{provider_name} live · partial structured output"
    if result.decisions and all(
        decision.validation_status in LIVE_VALID_STATUSES
        and decision.live_or_mock == "live"
        and not decision.fallback_used
        for decision in result.decisions
    ):
        return f"{provider_name} live"
    return f"{provider_name} requested · no valid live output"


def complete_profile_comparison_available(results: dict[str, SimulationResult]) -> bool:
    if len(results) < 2:
        return False
    profiles = list(results)[:2]
    requested_sets = []
    usable_sets = []
    for profile in profiles:
        result = results[profile]
        requested = {
            (adapter_result.actor, adapter_result.turn)
            for adapter_result in result.adapter_results
            if adapter_result.actor is not None and adapter_result.turn is not None
        } or {
            (decision.actor, decision.turn)
            for decision in result.decisions
        }
        usable = {
            (decision.actor, decision.turn)
            for decision in result.decisions
            if decision.validation_status in LIVE_VALID_STATUSES
            or decision.fallback_used
            or decision.live_or_mock == "mock"
        }
        requested_sets.append(requested)
        usable_sets.append(usable)
    expected = set.union(*requested_sets) if requested_sets else set()
    return bool(expected) and all(expected <= usable for usable in usable_sets)


def actor_sequence_for_scenario(scenario: Scenario, selected_actors: list[str] | None = None) -> list[str]:
    return actor_sequence_for_selection(scenario, selected_actors)


def _dedupe_chunks(chunks) -> list:
    seen = set()
    deduped = []
    for chunk in chunks:
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        deduped.append(chunk)
    return deduped


def _dedupe_source_card_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for row in rows:
        key = (row.get("actor"), row.get("source_id"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _turn_narrative(scenario: Scenario, treatment: Treatment, turn: int) -> str:
    if turn == 1:
        return f"Initial crisis signals are ambiguous. {scenario.initial_state}"
    if turn == 2:
        return f"Political and military audiences react to early moves. {treatment.modifier}"
    if turn == 3:
        return "Commanders report compressed timelines, incomplete intelligence, and public pressure for resolve."
    if turn == 4:
        return "Back-channel messages arrive late and are interpreted differently by each actor."
    return "The crisis nears exhaustion; leaders must decide whether to pause, signal, or intensify."
