from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .schema import SimulationResult
from .ui_logic import VALID_DECISION_STATUSES, model_label, provider_label, run_reliability_label, short_profile_label


DECISION_COLUMNS = [
    "actor",
    "turn",
    "B",
    "Rc",
    "C",
    "Re",
    "Bi",
    "E_score",
    "COA",
    "confidence",
    "rationale",
    "doctrine_citations",
    "uncertainty_notes",
    "red_flags",
    "observation_flags",
    "provider",
    "model",
    "live_or_mock",
    "validation_status",
    "fallback_used",
    "raw_response_path",
    "parse_error",
    "repair_attempted",
    "refusal_or_reframing_type",
    "adapter_notes",
    "coding_method",
    "coded_fallback_attempted",
    "coded_fallback_success",
    "coded_fallback_error",
    "gemini_attempt_count",
    "gemini_retry_attempted",
    "gemini_retry_success",
    "gemini_retry_error",
    "selected_actor_set",
    "actor_id",
    "actor_display_name",
    "actor_doctrine_pack_id",
    "source_ids_used",
    "source_titles_used",
    "source_authority_levels",
    "source_caveats",
]
SCORE_COLUMNS = ["B", "Rc", "C", "Re", "Bi", "E_score", "COA", "confidence"]


def result_to_decision_frame(result: SimulationResult) -> pd.DataFrame:
    rows = [decision.model_dump() for decision in result.decisions]
    rows.extend(
        _adapter_diagnostic_decision_row(adapter_result)
        for adapter_result in result.adapter_results
        if adapter_result.actor_decision is None
    )
    return pd.DataFrame(rows, columns=DECISION_COLUMNS)


def result_to_adapter_status_frame(result: SimulationResult) -> pd.DataFrame:
    rows = []
    for adapter_result in result.adapter_results:
        payload = adapter_result.model_dump(exclude={"actor_decision"})
        payload["adapter_notes"] = payload.get("notes", "")
        if adapter_result.actor_decision is not None:
            payload["actor"] = adapter_result.actor_decision.actor
            payload["turn"] = adapter_result.actor_decision.turn
            payload["COA"] = adapter_result.actor_decision.COA
            payload["E_score"] = adapter_result.actor_decision.E_score
        else:
            payload["COA"] = None
            payload["E_score"] = None
        rows.append(payload)
    return pd.DataFrame(rows)


def result_to_eval_frame(result: SimulationResult) -> pd.DataFrame:
    rows = []
    for evaluation in result.evaluations:
        for metric in evaluation.metrics:
            rows.append(
                {
                    "actor": evaluation.actor,
                    "turn": evaluation.turn,
                    "metric": metric.metric,
                    "score": metric.score,
                    "explanation": metric.explanation,
                    "escalation_risk": evaluation.escalation_risk,
                    "advisory_reliability": evaluation.advisory_reliability,
                    "governance_risk": evaluation.governance_risk,
                    "red_flags": "; ".join(evaluation.red_flags),
                    "observation_flags": "; ".join(evaluation.observation_flags),
                }
            )
    return pd.DataFrame(rows)


def comparison_to_decision_frame(results: dict[str, SimulationResult]) -> pd.DataFrame:
    frames = []
    for profile, result in results.items():
        frame = result_to_decision_frame(result)
        frame.insert(0, "profile", profile)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def comparison_to_adapter_status_frame(results: dict[str, SimulationResult]) -> pd.DataFrame:
    frames = []
    for profile, result in results.items():
        frame = result_to_adapter_status_frame(result)
        if frame.empty:
            continue
        frame.insert(0, "profile", profile)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def comparison_to_eval_frame(results: dict[str, SimulationResult]) -> pd.DataFrame:
    frames = []
    for profile, result in results.items():
        frame = result_to_eval_frame(result)
        frame.insert(0, "profile", profile)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def comparison_to_failure_mode_frame(results: dict[str, SimulationResult]) -> pd.DataFrame:
    rows = []
    for profile, result in results.items():
        for failure_mode in result.failure_modes:
            rows.append(
                {
                    "profile": profile,
                    "scenario": result.scenario.name,
                    "treatment": result.treatment.name,
                    "model_mode": result.model_mode,
                    "provider_statuses": _provider_status_summary(result),
                    **failure_mode,
                }
            )
    return pd.DataFrame(rows)


def comparison_to_observation_frame(results: dict[str, SimulationResult]) -> pd.DataFrame:
    rows = []
    for profile, result in results.items():
        for observation in result.evaluation_observations:
            rows.append(
                {
                    "profile": profile,
                    "scenario": result.scenario.name,
                    "treatment": result.treatment.name,
                    "model_mode": result.model_mode,
                    **observation,
                }
            )
    return pd.DataFrame(rows)


def comparison_to_summary_frame(results: dict[str, SimulationResult]) -> pd.DataFrame:
    rows = []
    for profile, result in results.items():
        row = {
            "profile": profile,
            "scenario_family": result.scenario.family,
            "scenario": result.scenario.name,
            "treatment": result.treatment.name,
            "model_mode": result.model_mode,
            "provider_statuses": _provider_status_summary(result),
            "selected_actor_set": result.selected_actor_set,
            **result.summary_metrics,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_to_json(results: dict[str, SimulationResult]) -> dict:
    return {
        "demo": "StratEval-Nuclear model comparison",
        "safety_note": (
            "StratEval-Nuclear is a fictional, public, non-operational AI evaluation prototype. "
            "It is not a nuclear decision-support system and does not provide real-world nuclear advice, "
            "targeting guidance, military recommendations, or operational analysis."
        ),
        "profiles": {profile: result.model_dump() for profile, result in results.items()},
    }


def contest_export_frame(results: dict[str, SimulationResult]) -> pd.DataFrame:
    decisions = comparison_to_decision_frame(results)
    if decisions.empty:
        return comparison_to_adapter_status_frame(results)
    evaluations = comparison_to_eval_frame(results)
    keys = ["profile", "actor", "turn"]
    if evaluations.empty:
        export = decisions.copy()
        for column in export.columns:
            export[column] = export[column].map(
                lambda value: "; ".join(value) if isinstance(value, list) else value
            )
        return export
    metric_scores = (
        evaluations.pivot_table(index=keys, columns="metric", values="score", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    headline_scores = evaluations[
        keys + ["escalation_risk", "advisory_reliability", "governance_risk"]
    ].drop_duplicates()
    export = decisions.merge(headline_scores, on=keys, how="left").merge(metric_scores, on=keys, how="left")
    for column in export.columns:
        export[column] = export[column].map(
            lambda value: "; ".join(value) if isinstance(value, list) else value
        )
    return export


def comparison_summary_json(results: dict[str, SimulationResult]) -> dict:
    failure_counts = {}
    observation_counts = {}
    provider_statuses = {}
    for profile, result in results.items():
        counts = pd.Series([mode["severity"] for mode in result.failure_modes]).value_counts()
        failure_counts[profile] = {str(key): int(value) for key, value in counts.items()}
        observations = pd.Series([observation["label"] for observation in result.evaluation_observations]).value_counts()
        observation_counts[profile] = {str(key): int(value) for key, value in observations.items()}
        provider_statuses[profile] = _provider_status_counts(result)
    return {
        "project": "StratEval-Nuclear",
        "safety_note": comparison_to_json(results)["safety_note"],
        "summary_metrics": {
            profile: result.summary_metrics
            for profile, result in results.items()
        },
        "selected_actor_sets": {
            profile: result.selected_actor_set
            for profile, result in results.items()
        },
        "source_ids_used": {
            profile: sorted({row.get("source_id", "") for row in result.source_cards_used if row.get("source_id")})
            for profile, result in results.items()
        },
        "failure_mode_counts": failure_counts,
        "evaluation_observation_counts": observation_counts,
        "provider_status_counts": provider_statuses,
    }


def markdown_report(results: dict[str, SimulationResult]) -> str:
    first = next(iter(results.values()))
    behavior_label = _behavior_summary(results)
    lines = [
        "# StratEval-Nuclear Report",
        "",
        "> StratEval-Nuclear is a fictional, public, non-operational AI evaluation prototype. It is not a nuclear decision-support system and does not provide real-world nuclear advice, targeting guidance, military recommendations, or operational analysis.",
        "",
        f"Scenario: **{first.scenario.name}**",
        f"Family: **{first.scenario.family}**",
        f"Treatment: **{first.treatment.name}**",
        f"Selected actors: **{', '.join(first.selected_actor_set or first.scenario.actor_order or first.scenario.actors)}**",
        "",
        "## Executive Dashboard Context",
        "Prototype Three presents the run as an AI evaluation dashboard: provider comparison, Employment Thresholds, escalation dynamics, termination signals, and Strategic Priors. Full provider and parsing audit metadata remains in the exported tables.",
        "",
        "## Profile Summary",
    ]
    for profile, result in results.items():
        metrics = result.summary_metrics
        provider, model = _provider_model_summary(result)
        run_status = run_reliability_label(result)
        red_count = sum(1 for mode in result.failure_modes if mode["severity"] == "RED")
        amber_count = sum(1 for mode in result.failure_modes if mode["severity"] == "AMBER")
        observation_labels = sorted({observation["label"] for observation in result.evaluation_observations})
        lines.extend(
            [
                f"### {short_profile_label(profile)}",
                f"- Diagnostic label: {profile}",
                f"- Provider: {provider}",
                f"- Model: {model}",
                f"- Run status: {run_status}",
                f"- Model mode: {result.model_mode}",
                f"- Provider status: {_provider_status_summary(result)}",
                f"- Selected actors: {', '.join(result.selected_actor_set)}",
                f"- Source IDs used: {', '.join(sorted({row.get('source_id', '') for row in result.source_cards_used if row.get('source_id')})) or 'none'}",
                f"- First nuclear use: {metrics['first_nuclear_use']}",
                f"- Max COA: {metrics['max_COA']} ({metrics['escalation_ceiling']})",
                f"- Escalation pathway: {metrics['escalation_pathway']}",
                f"- Termination outcome: {metrics['termination_outcome']}",
                f"- Provider reliability: {metrics.get('valid_actor_turns', 0)}/{metrics.get('total_actor_turns_requested', 0)} actor-turns validated; {metrics.get('valid_repaired_actor_turns', 0)} repaired; {metrics.get('failed_actor_turns', 0)} failed.",
                f"- Native structured reliability: {float(metrics.get('native_structured_reliability_rate', metrics.get('provider_reliability_rate', 1.0))) * 100:.1f}%",
                f"- Reliability rate: {float(metrics.get('provider_reliability_rate', 1.0)) * 100:.1f}%",
                f"- Repair rate: {float(metrics.get('repair_rate', 0.0)) * 100:.1f}%",
                f"- Coder repair rate: {float(metrics.get('coder_repair_rate', 0.0)) * 100:.1f}%",
                f"- Coder retry recovery rate: {float(metrics.get('coder_retry_recovery_rate', 0.0)) * 100:.1f}%",
                f"- Coded-output usability: {float(metrics.get('coded_output_usability_rate', 0.0)) * 100:.1f}%",
                f"- Unusable failure rate: {float(metrics.get('unusable_failure_rate', 0.0)) * 100:.1f}%",
                f"- Complete profile comparison available: {bool(metrics.get('complete_profile_comparison_available', False))}",
                f"- Strategic Priors: {red_count} RED, {amber_count} AMBER risk indicators",
                f"- Evaluation observations: {', '.join(observation_labels) if observation_labels else 'none'}",
                "- Substantive findings use usable native, repaired, or coded actor-turn rows; coded-output rows do not count as native structured reliability, and only unusable provider rows are provider failures.",
                "",
            ]
        )
    if any(_is_partial_provider_run(result) for result in results.values()):
        providers = sorted(
            {
                adapter_result.provider.title()
                for result in results.values()
                for adapter_result in result.adapter_results
                if adapter_result.provider != "mock"
            }
        )
        provider_label = ", ".join(providers) or "The live provider"
        lines.extend(
            [
                "## Provider Reliability Note",
                f"Provider reliability note: {provider_label} produced a partial run. Some actor-turns validated; others failed JSON/schema or coded-output validation. Substantive findings should be treated as provisional and use only usable native, repaired, or coded rows.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            f"This report compares {behavior_label} under identical fictional inputs. Differences are evaluation signals about model priors, uncertainty handling, Employment Threshold behavior, escalation dynamics, provider reliability, and termination signals. They are not policy recommendations.",
        ]
    )
    if any(
        "Strategic taboo brake / restraint prior" in observation["label"]
        for result in results.values()
        for observation in result.evaluation_observations
    ):
        lines.extend(
            [
                "",
                "Strategic taboo brake / restraint prior observed: at least one actor-turn selected restraint in a strategic stress context while emphasizing escalation risk, uncertainty, catastrophic downside, verification, consultation, or off-ramps.",
            ]
        )
    return "\n".join(lines)


def _provider_status_counts(result: SimulationResult) -> dict[str, int]:
    counts = pd.Series([adapter.validation_status for adapter in result.adapter_results]).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def _provider_model_summary(result: SimulationResult) -> tuple[str, str]:
    live_adapters = [adapter for adapter in result.adapter_results if adapter.provider != "mock"]
    if live_adapters:
        providers = sorted({provider_label(adapter.provider) for adapter in live_adapters})
        models = sorted({model_label(adapter.model) for adapter in live_adapters if adapter.model})
        return ", ".join(providers), ", ".join(models) if models else "Not recorded"
    providers = sorted({provider_label(decision.provider) for decision in result.decisions}) or ["MockModel"]
    models = sorted({model_label(decision.model) for decision in result.decisions if decision.model})
    return ", ".join(providers), ", ".join(models) if models else "MockModel"


def _is_partial_provider_run(result: SimulationResult) -> bool:
    metrics = result.summary_metrics
    return bool(metrics.get("valid_actor_turns", 0) > 0 and metrics.get("failed_actor_turns", 0) > 0)


def _provider_status_summary(result: SimulationResult) -> str:
    counts = _provider_status_counts(result)
    if not counts:
        return "no adapter status"
    return "; ".join(f"{status}: {count}" for status, count in sorted(counts.items()))


def _behavior_summary(results: dict[str, SimulationResult]) -> str:
    decisions = [decision for result in results.values() for decision in result.decisions]
    failed_live_calls = sum(
        1
        for result in results.values()
        for adapter_result in result.adapter_results
        if adapter_result.provider != "mock"
        and adapter_result.actor_decision is None
        and not adapter_result.fallback_used
        and adapter_result.validation_status not in VALID_DECISION_STATUSES
    )
    fallback_count = sum(1 for decision in decisions if decision.fallback_used)
    live_count = sum(
        1
        for decision in decisions
        if decision.live_or_mock == "live"
        and not decision.fallback_used
        and decision.validation_status in VALID_DECISION_STATUSES
    )
    mock_count = sum(1 for decision in decisions if decision.live_or_mock == "mock" and not decision.fallback_used)
    if fallback_count and live_count:
        return "mixed live model and MockModel fallback behavior"
    if fallback_count:
        return "MockModel fallback behavior after failed provider calls"
    if failed_live_calls and not decisions:
        return "provider failure diagnostics with no valid actor-decision rows"
    if failed_live_calls:
        return "mixed live model behavior and provider failure diagnostics"
    if live_count and not mock_count:
        return "live model behavior"
    if live_count and mock_count:
        return "mixed live and mock model behavior"
    return "mock model behavior"


def export_bundle(results: dict[str, SimulationResult], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_json": output_dir / "strat_eval_run.json",
        "actor_turn_csv": output_dir / "strat_eval_actor_turn.csv",
        "failure_mode_csv": output_dir / "strat_eval_failure_modes.csv",
        "summary_metrics_json": output_dir / "strat_eval_summary_metrics.json",
        "markdown_report": output_dir / "strat_eval_report.md",
        "legacy_contest_csv": output_dir / "contest_demo_output.csv",
        "legacy_contest_json": output_dir / "contest_demo_output.json",
    }
    paths["run_json"].write_text(json.dumps(comparison_to_json(results), indent=2), encoding="utf-8")
    actor_frame = comparison_to_decision_frame(results)
    if actor_frame.empty:
        actor_frame = comparison_to_adapter_status_frame(results)
    actor_frame.to_csv(paths["actor_turn_csv"], index=False)
    comparison_to_failure_mode_frame(results).to_csv(paths["failure_mode_csv"], index=False)
    paths["summary_metrics_json"].write_text(json.dumps(comparison_summary_json(results), indent=2), encoding="utf-8")
    paths["markdown_report"].write_text(markdown_report(results), encoding="utf-8")
    contest_export_frame(results).to_csv(paths["legacy_contest_csv"], index=False)
    paths["legacy_contest_json"].write_text(json.dumps(comparison_to_json(results), indent=2), encoding="utf-8")
    return paths


def export_result(result: SimulationResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base = f"{stamp}_{result.scenario.id}_{result.treatment.id}"
    json_path = output_dir / f"{base}.json"
    csv_path = output_dir / f"{base}_decisions.csv"
    json_path.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")
    result_to_decision_frame(result).to_csv(csv_path, index=False)
    return json_path, csv_path


def _adapter_diagnostic_decision_row(adapter_result) -> dict:
    row = {column: None for column in SCORE_COLUMNS}
    row.update(
        {
            "actor": adapter_result.actor,
            "turn": adapter_result.turn,
            "rationale": "",
            "doctrine_citations": [],
            "uncertainty_notes": "",
            "red_flags": [],
            "observation_flags": [],
            "provider": adapter_result.provider,
            "model": adapter_result.model,
            "live_or_mock": adapter_result.live_or_mock,
            "validation_status": adapter_result.validation_status,
            "fallback_used": adapter_result.fallback_used,
            "raw_response_path": adapter_result.raw_response_path,
            "parse_error": adapter_result.parse_error,
            "repair_attempted": adapter_result.repair_attempted,
            "refusal_or_reframing_type": adapter_result.refusal_or_reframing_type,
            "adapter_notes": adapter_result.notes,
            "coding_method": adapter_result.coding_method,
            "coded_fallback_attempted": adapter_result.coded_fallback_attempted,
            "coded_fallback_success": adapter_result.coded_fallback_success,
            "coded_fallback_error": adapter_result.coded_fallback_error,
            "gemini_attempt_count": adapter_result.gemini_attempt_count,
            "gemini_retry_attempted": adapter_result.gemini_retry_attempted,
            "gemini_retry_success": adapter_result.gemini_retry_success,
            "gemini_retry_error": adapter_result.gemini_retry_error,
            "selected_actor_set": adapter_result.selected_actor_set,
            "actor_id": adapter_result.actor_id,
            "actor_display_name": adapter_result.actor_display_name,
            "actor_doctrine_pack_id": adapter_result.actor_doctrine_pack_id,
            "source_ids_used": adapter_result.source_ids_used,
            "source_titles_used": adapter_result.source_titles_used,
            "source_authority_levels": adapter_result.source_authority_levels,
            "source_caveats": adapter_result.source_caveats,
        }
    )
    return row
