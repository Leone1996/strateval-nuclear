from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from crisiseval.analytics import COA_LABELS, profile_threshold_shift
from crisiseval.agents import STRATEGIC_PROFILES
from crisiseval.actor_registry import actor_display_label, default_actor_names_for_scenario
from crisiseval.prompts import SYSTEM_CAVEAT
from crisiseval.schema import Scenario, SimulationResult, Treatment
from crisiseval.simulation import actor_sequence_for_scenario, list_scenarios, list_treatments, load_json_model, run_model_comparison, run_simulation
from crisiseval.storage import (
    comparison_to_adapter_status_frame,
    comparison_summary_json,
    comparison_to_decision_frame,
    comparison_to_eval_frame,
    comparison_to_failure_mode_frame,
    comparison_to_json,
    comparison_to_observation_frame,
    comparison_to_summary_frame,
    contest_export_frame,
    export_bundle,
    markdown_report,
)
from crisiseval.ui_logic import (
    VALID_DECISION_STATUSES,
    chart_legend_label,
    cross_provider_difference_frame,
    full_diagnostic_label,
    model_label,
    profile_disagreement_summary,
    provider_diagnostic_groups,
    provider_label,
    provider_failure_without_fallback_count,
    provider_model_label,
    provider_reliability_message,
    provider_reliability_summary,
    provider_summary_metrics,
    run_reliability_label,
    short_profile_label,
    timeline_display_spec,
    valid_actor_turn_frame,
)


load_dotenv()

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
TAGLINE = "“We may be likened to two scorpions in a bottle, each capable of killing the other, but only at the risk of his own life.”"
TAGLINE_ATTRIBUTION = "— Robert J. Oppenheimer"
MODEL_MODE_OPTIONS = [
    "Offline MockModel",
    "OpenAI live",
    "Gemini live",
    "Anthropic live",
    "Compare MockModel vs OpenAI",
    "Compare MockModel vs Gemini",
    "Compare MockModel vs Anthropic",
    "Compare OpenAI vs Gemini",
    "Compare OpenAI vs Anthropic",
    "Compare Gemini vs Anthropic",
    "Compare OpenAI vs Gemini vs Anthropic",
]
MODEL_MODE_PROVIDERS = {
    "Offline MockModel": ["mock"],
    "OpenAI live": ["openai"],
    "Gemini live": ["gemini"],
    "Anthropic live": ["anthropic"],
    "Compare MockModel vs OpenAI": ["mock", "openai"],
    "Compare MockModel vs Gemini": ["mock", "gemini"],
    "Compare MockModel vs Anthropic": ["mock", "anthropic"],
    "Compare OpenAI vs Gemini": ["openai", "gemini"],
    "Compare OpenAI vs Anthropic": ["openai", "anthropic"],
    "Compare Gemini vs Anthropic": ["gemini", "anthropic"],
    "Compare OpenAI vs Gemini vs Anthropic": ["openai", "gemini", "anthropic"],
}
HIGH_END_MODEL_DEFAULTS = {
    "openai": "gpt-5.5",
    "gemini": "gemini-2.5-pro",
    "anthropic": "claude-opus-4-8",
}
CHEAPER_MODEL_NOTES = {
    "openai": "gpt-5.4-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-6",
}


st.set_page_config(page_title="StratEval-Nuclear", layout="wide")
st.markdown(
    """
<style>
  :root {
    --se-card-bg: var(--secondary-background-color, #f8fafc);
    --se-card-bg-soft: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 88%, var(--background-color, #ffffff));
    --se-text: var(--text-color, #0f172a);
    --se-muted: color-mix(in srgb, var(--text-color, #0f172a) 68%, transparent);
    --se-border: color-mix(in srgb, var(--text-color, #0f172a) 18%, transparent);
    --se-pill-bg: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 85%, var(--background-color, #ffffff));
  }
  .block-container {padding-top: 1.3rem; padding-bottom: 3rem; max-width: 1320px;}
  .hero {
    border: 1px solid var(--se-border);
    border-radius: 8px;
    padding: 1.1rem 1.25rem;
    background: var(--se-card-bg);
    color: var(--se-text);
  }
  .hero, .hero * {color: var(--se-text);}
  .hero h1 {
    font-size: 2rem;
    margin: 0 0 .25rem 0;
    letter-spacing: 0;
    background: transparent !important;
    color: var(--se-text) !important;
    display: block;
  }
  .subtitle {font-size: 1rem; color: var(--se-muted) !important;}
  .tagline {font-size: 1rem; color: var(--se-text) !important; font-weight: 650; margin: .65rem 0;}
  .pill {
    display: inline-block;
    border: 1px solid var(--se-border);
    border-radius: 999px;
    padding: .14rem .55rem;
    margin: .12rem .25rem .12rem 0;
    font-size: .78rem;
    color: var(--se-text) !important;
    background: var(--se-pill-bg);
  }
  .actor-row {margin: .35rem 0 .55rem 0;}
  .actor-chip {
    display: inline-block;
    border: 1px solid var(--se-border);
    border-radius: 999px;
    padding: .14rem .5rem;
    margin: .12rem .18rem .12rem 0;
    font-size: .8rem;
    line-height: 1.1rem;
    color: var(--se-text) !important;
    background: var(--se-pill-bg);
  }
  .coa-card {
    border: 1px solid var(--se-border);
    border-radius: 8px;
    padding: .7rem .78rem;
    margin-top: .85rem;
    background: var(--se-card-bg);
    color: var(--se-text);
    font-size: .82rem;
    line-height: 1.2rem;
  }
  .coa-card, .coa-card * {color: var(--se-text) !important;}
  .coa-card .coa-title {font-weight: 750; margin-bottom: .35rem;}
  .coa-card .coa-line {display: grid; grid-template-columns: 1.35rem 1fr; gap: .28rem; margin: .14rem 0;}
  .metric {
    border: 1px solid var(--se-border);
    border-radius: 8px;
    padding: .78rem .86rem;
    background: var(--se-card-bg);
    color: var(--se-text);
  }
  .metric .label {font-size: .76rem; color: var(--se-muted) !important; text-transform: uppercase;}
  .metric .value {font-size: 1.22rem; font-weight: 750; color: var(--se-text) !important; line-height: 1.45rem;}
  .guide {
    border: 1px solid var(--se-border);
    border-radius: 8px;
    padding: .9rem 1rem;
    background: var(--se-card-bg);
    color: var(--se-text);
    margin: .85rem 0;
  }
  .guide, .guide * {color: var(--se-text) !important;}
  .safety-strip {
    border: 1px solid var(--se-border);
    border-left: 5px solid #38bdf8;
    border-radius: 8px;
    padding: .62rem .8rem;
    background: var(--se-card-bg);
    color: var(--se-text);
    margin: .75rem 0 .9rem 0;
    font-size: .92rem;
    line-height: 1.3rem;
  }
  .safety-strip strong {color: var(--se-text) !important;}
  .mode-card {
    --severity-color: var(--se-border);
    border: 1px solid var(--se-border);
    border-left: 6px solid var(--severity-color);
    border-radius: 8px;
    padding: .85rem .9rem;
    background: var(--se-card-bg);
    color: var(--se-text);
    min-height: 165px;
    margin-bottom: .7rem;
  }
  .mode-card, .mode-card span, .mode-card strong {color: var(--se-text) !important;}
  .mode-card .mode-title {color: var(--severity-color) !important;}
  .sev-GREEN {--severity-color: #22c55e;}
  .sev-AMBER {--severity-color: #f59e0b;}
  .sev-RED {--severity-color: #ef4444;}
  .flow {display: grid; grid-template-columns: repeat(5, minmax(130px, 1fr)); gap: .55rem;}
  .flow div {
    border: 1px solid var(--se-border);
    border-radius: 8px;
    padding: .7rem;
    background: var(--se-card-bg);
    color: var(--se-text);
  }
  .flow div strong {color: var(--se-text) !important;}
  @media (prefers-color-scheme: dark) {
    :root {
      --se-card-bg: var(--secondary-background-color, #111827);
      --se-card-bg-soft: #0f172a;
      --se-text: var(--text-color, #f8fafc);
      --se-muted: #cbd5e1;
      --se-border: #334155;
      --se-pill-bg: #1f2937;
    }
  }
  @media (max-width: 900px) {.flow {grid-template-columns: 1fr 1fr;} .hero h1 {font-size: 1.55rem;}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="hero">
  <h1>StratEval-Nuclear</h1>
  <div class="subtitle">A simulation-based evaluation harness for testing AI decision support across nuclear contingencies</div>
  <div class="tagline">{TAGLINE}</div>
  <div class="subtitle">{TAGLINE_ATTRIBUTION}</div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown(f'<div class="safety-strip"><strong>Safety caveat:</strong> {SYSTEM_CAVEAT}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_options() -> tuple[dict[str, Path], dict[str, Path], dict[str, Scenario], dict[str, Treatment]]:
    scenario_paths = list_scenarios()
    treatment_paths = list_treatments()
    scenarios = {load_json_model(path, Scenario).name: path for path in scenario_paths}
    treatments = {load_json_model(path, Treatment).name: path for path in treatment_paths}
    scenario_models = {name: load_json_model(path, Scenario) for name, path in scenarios.items()}
    treatment_models = {name: load_json_model(path, Treatment) for name, path in treatments.items()}
    return scenarios, treatments, scenario_models, treatment_models


def default_index(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    for column in clean.columns:
        clean[column] = clean[column].map(lambda value: "; ".join(map(str, value)) if isinstance(value, list) else value)
    return clean


def env_model_id(provider: str, env_var: str) -> str:
    return (os.getenv(env_var) or "").strip() or HIGH_END_MODEL_DEFAULTS[provider]


ACTOR_FLAGS = {
    "India": "🇮🇳",
    "Pakistan": "🇵🇰",
    "China": "🇨🇳",
    "United States": "🇺🇸",
    "Russia": "🇷🇺",
    "NATO": "",
    "North Korea": "🇰🇵",
    "Iran": "🇮🇷",
    "Israel": "🇮🇱",
    "United Kingdom": "🇬🇧",
    "France": "🇫🇷",
    "Taiwan": "🇹🇼",
    "Japan": "🇯🇵",
}


def actor_label(actor: str) -> str:
    return actor_display_label(actor)


def actor_chip_row(actors: list[str]) -> str:
    chips = "".join(f'<span class="actor-chip">{actor_label(actor)}</span>' for actor in actors)
    return f'<div class="actor-row">{chips}</div>'


def coa_ladder_card() -> str:
    lines = "".join(
        f'<div class="coa-line"><strong>{coa}</strong><span>{label}</span></div>'
        for coa, label in COA_LABELS.items()
    )
    return f'<div class="coa-card"><div class="coa-title">COA Ladder Reference</div>{lines}</div>'


def escalation_calculus_card() -> str:
    lines = [
        ("E", "(B + Rc) / (C + Re + Bi)"),
        ("B", "perceived politico-military benefits of escalation"),
        ("Rc", "risk of not escalating"),
        ("C", "costs of escalation"),
        ("Re", "risk of uncontrolled escalation"),
        ("Bi", "benefits of restraint or inaction"),
    ]
    rows = "".join(f'<div class="coa-line"><strong>{key}</strong><span>{value}</span></div>' for key, value in lines)
    return f'<div class="coa-card"><div class="coa-title">Escalation Calculus Reference</div>{rows}</div>'


def fallback_actor_turn_count(results: dict[str, SimulationResult]) -> int:
    return sum(1 for result in results.values() for decision in result.decisions if decision.fallback_used)


def actual_behavior_label(results: dict[str, SimulationResult]) -> str:
    decisions = [decision for result in results.values() for decision in result.decisions]
    failed_live_calls = provider_failure_without_fallback_count(results)
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


def add_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if not display.empty and "profile" in display.columns:
        display["profile_short"] = display["profile"].map(short_profile_label)
        display["profile_display"] = display["profile"].map(profile_display_label)
        display["chart_profile"] = display["profile"].map(chart_legend_label)
    if not display.empty and "provider" in display.columns:
        display["provider_display"] = display["provider"].map(provider_label)
        display["provider_chart"] = display["provider_display"]
    elif not display.empty and "model_mode" in display.columns:
        display["provider_display"] = display["model_mode"].map(provider_label)
        display["provider_chart"] = display["provider_display"]
    elif not display.empty and "profile" in display.columns:
        display["provider_display"] = display["profile"].map(provider_label)
        display["provider_chart"] = display["provider_display"]
    if not display.empty and "model" in display.columns:
        display["model_display"] = display["model"].map(model_label)
    if not display.empty and {"provider_display", "model_display"}.issubset(display.columns):
        display["provider_model"] = [
            provider_model_label(provider, model)
            for provider, model in zip(display["provider_display"], display["model_display"], strict=False)
        ]
    return display


def profile_display_label(value: object) -> str:
    label = str(value or "").strip()
    if not label:
        return "Profile"
    if "Profile A" in label:
        return "Deterrence Restoration"
    if "Profile B" in label:
        return "Escalation Management"
    if "Profile C" in label or "worst-case" in label.lower() or "worst_case" in label.lower():
        return "Worst-case Pressure"
    if " · " in label:
        label = label.split(" · ")[-1].strip()
    if ":" in label and label.startswith("Profile "):
        label = label.split(":", 1)[1].strip()
    return label.replace("_", " ").strip().title() or "Profile"


def termination_outcomes_by_profile(summary_display: pd.DataFrame) -> pd.DataFrame:
    required = {"profile_display", "termination_outcome"}
    if summary_display.empty or not required.issubset(summary_display.columns):
        return pd.DataFrame(columns=["profile_display", "termination_outcome", "size"])
    usable = summary_display.dropna(subset=["termination_outcome"]).copy()
    if usable.empty:
        return pd.DataFrame(columns=["profile_display", "termination_outcome", "size"])
    return usable.groupby(["profile_display", "termination_outcome"], as_index=False).size()


def strategic_prior_risks_by_profile(substantive_display: pd.DataFrame) -> pd.DataFrame:
    required = {"profile_display", "severity"}
    if substantive_display.empty or not required.issubset(substantive_display.columns):
        return pd.DataFrame(columns=["profile_display", "severity", "size"])
    usable = substantive_display.dropna(subset=["severity"]).copy()
    if usable.empty:
        return pd.DataFrame(columns=["profile_display", "severity", "size"])
    return usable.groupby(["profile_display", "severity"], as_index=False).size()


def result_provider_metadata(result: SimulationResult) -> tuple[str, str]:
    adapter_results = [adapter for adapter in result.adapter_results if adapter.provider != "mock"]
    if adapter_results:
        providers = sorted({provider_label(adapter.provider) for adapter in adapter_results})
        models = sorted({model_label(adapter.model) for adapter in adapter_results if adapter.model})
        return ", ".join(providers), ", ".join(models) if models else "Not recorded"
    providers = sorted({provider_label(decision.provider) for decision in result.decisions}) or ["MockModel"]
    models = sorted({model_label(decision.model) for decision in result.decisions if decision.model})
    return ", ".join(providers), ", ".join(models) if models else "MockModel"


def provider_comparison_frame(results: dict[str, SimulationResult]) -> pd.DataFrame:
    rows = []
    for profile, result in results.items():
        metrics = result.summary_metrics
        provider, model = result_provider_metadata(result)
        run_status = run_reliability_label(result)
        rows.append(
            {
                "profile": short_profile_label(profile),
                "provider": provider,
                "model": model,
                "run status": run_status,
                "scenario": result.scenario.name,
                "treatment": result.treatment.name,
                "total requested": int(metrics.get("total_actor_turns_requested", 0) or 0),
                "usable rows": int(metrics.get("valid_actor_turns", 0) or 0),
                "native structured rows": int(metrics.get("native_structured_actor_turns", 0) or 0),
                "repaired rows": int(metrics.get("valid_repaired_actor_turns", 0) or 0),
                "coded-output rows": int(metrics.get("coded_output_actor_turns", 0) or 0),
                "coded-output retried rows": int(metrics.get("coded_output_retried_actor_turns", 0) or 0),
                "coded-output repaired rows": int(metrics.get("coded_output_repaired_actor_turns", 0) or 0),
                "failed rows": int(metrics.get("failed_actor_turns", 0) or 0),
                "reliability rate": float(metrics.get("provider_reliability_rate", 1.0) or 0.0),
                "native structured reliability": float(metrics.get("native_structured_reliability_rate", 1.0) or 0.0),
                "repair rate": float(metrics.get("repair_rate", 0.0) or 0.0),
                "coder repair rate": float(metrics.get("coder_repair_rate", 0.0) or 0.0),
                "coder retry recovery rate": float(metrics.get("coder_retry_recovery_rate", 0.0) or 0.0),
                "coded-output usability": float(metrics.get("coded_output_usability_rate", 0.0) or 0.0),
                "total unusable failure rate": float(metrics.get("unusable_failure_rate", 0.0) or 0.0),
                "complete profile comparison available": bool(metrics.get("complete_profile_comparison_available", False)),
                "max COA": int(metrics.get("max_COA", 0) or 0),
                "first nuclear use": bool(metrics.get("first_nuclear_use", False)),
                "termination outcome": metrics.get("termination_outcome", "not available"),
                "strategic prior observations count": len(result.evaluation_observations),
                "diagnostic label": full_diagnostic_label(
                    profile=profile,
                    provider=provider,
                    model=model,
                    run_status=run_status,
                ),
            }
        )
    return pd.DataFrame(rows)


def run_verdict_sentence(result: SimulationResult) -> tuple[str, str]:
    metrics = result.summary_metrics
    provider, _ = result_provider_metadata(result)
    status = run_reliability_label(result)
    usable = int(metrics.get("valid_actor_turns", 0) or 0)
    failed = int(metrics.get("failed_actor_turns", 0) or 0)
    coded = int(metrics.get("coded_output_actor_turns", 0) or 0)
    native = int(metrics.get("native_structured_actor_turns", 0) or 0)
    provider_name = provider_label(provider)
    if provider == "MockModel":
        return "info", "Offline benchmark run: MockModel rows validated locally. Substantive findings are reference benchmark artifacts."
    if provider_name == "Gemini" and native and not coded and failed == 0:
        return "success", "Gemini native structured-output run: actor-turns validated through Gemini response schema."
    if provider_name == "Gemini" and coded:
        return (
            "warning",
            "Gemini fallback-coded run: native structured output failed for some rows; machine-block fallback recovered usable actor-turns.",
        )
    if status == "clean":
        return "success", "Clean live run: all actor-turns validated. Substantive findings are based on complete structured outputs."
    if status == "partial structured output":
        return (
            "warning",
            f"Partial live run: {usable} actor-turns validated, {failed} failed structured-output validation. Substantive findings are based only on valid/repaired rows.",
        )
    return "error", "No valid live structured-output rows were produced. Review provider diagnostics before interpreting this run."


def render_run_verdict(results: dict[str, SimulationResult], provider_frame: pd.DataFrame) -> None:
    st.subheader("Run Verdict")
    for result in results.values():
        level, message = run_verdict_sentence(result)
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.warning(message)
        elif level == "error":
            st.error(message)
        else:
            st.info(message)
    columns = [
        "profile",
        "provider",
        "model",
        "run status",
        "total requested",
        "usable rows",
        "native structured rows",
        "repaired rows",
        "coded-output rows",
        "coded-output retried rows",
        "coded-output repaired rows",
        "failed rows",
        "native structured reliability",
        "reliability rate",
        "repair rate",
        "coder repair rate",
        "coder retry recovery rate",
        "coded-output usability",
        "total unusable failure rate",
        "complete profile comparison available",
        "max COA",
        "first nuclear use",
        "termination outcome",
    ]
    st.dataframe(provider_frame[columns], hide_index=True, width="stretch")


def style_chart(fig):
    paper_bg = "#0e1117"
    plot_bg = "#111827"
    font_color = "#f8fafc"
    grid_color = "#334155"

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font_color=font_color,
        title_font_color=font_color,
        legend_font_color=font_color,
        margin=dict(l=40, r=24, t=64, b=48),
    )
    fig.update_xaxes(
        gridcolor=grid_color,
        zerolinecolor=grid_color,
        linecolor=grid_color,
        tickfont_color=font_color,
        title_font_color=font_color,
    )
    fig.update_yaxes(
        gridcolor=grid_color,
        zerolinecolor=grid_color,
        linecolor=grid_color,
        tickfont_color=font_color,
        title_font_color=font_color,
    )
    fig.update_coloraxes(colorbar_tickfont_color=font_color, colorbar_title_font_color=font_color)
    return fig


def run_profiles(
    scenario_path: Path,
    treatment_path: Path,
    turns: int,
    top_k: int,
    selected_profiles: list[str],
    mode: str,
    providers: list[str] | None = None,
    adapter_config: dict | None = None,
    selected_actors: list[str] | None = None,
) -> dict[str, SimulationResult]:
    providers = providers or ["mock"]
    if mode == "Single Profile" and providers == ["mock"]:
        profile = selected_profiles[0]
        result = run_simulation(
            scenario_path,
            treatment_path,
            turns=turns,
            top_k=top_k,
            strategic_profile=profile,
            selected_actors=selected_actors,
        )
        return {STRATEGIC_PROFILES[profile]["label"]: result}
    return run_model_comparison(
        scenario_path,
        treatment_path,
        turns=turns,
        top_k=top_k,
        profiles=selected_profiles,
        providers=providers,
        adapter_config=adapter_config,
        selected_actors=selected_actors,
    )


@st.cache_data(show_spinner=False)
def pressure_sweep(scenario_path: str, turns: int, top_k: int, selected_profiles: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for treatment_path in list_treatments():
        treatment = load_json_model(treatment_path, Treatment)
        results = run_model_comparison(Path(scenario_path), treatment_path, turns=turns, top_k=top_k, profiles=list(selected_profiles))
        for profile, result in results.items():
            metrics = result.summary_metrics
            rows.append(
                {
                    "profile": profile,
                    "treatment": treatment.name,
                    "pressure_level": treatment.pressure_level,
                    "max_COA": metrics["max_COA"],
                    "first_nuclear_use": metrics["first_nuclear_use"],
                    "termination_outcome": metrics["termination_outcome"],
                    "red_failure_modes": sum(1 for mode in result.failure_modes if mode["severity"] == "RED"),
                    "amber_failure_modes": sum(1 for mode in result.failure_modes if mode["severity"] == "AMBER"),
                }
            )
    return pd.DataFrame(rows).sort_values(["pressure_level", "treatment", "profile"])


def render_metric_cards(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>',
                unsafe_allow_html=True,
            )


def render_provider_cards(provider_summary: pd.DataFrame) -> None:
    if provider_summary.empty:
        return
    cols = st.columns(min(3, len(provider_summary)))
    for col, (_, row) in zip(cols, provider_summary.iterrows(), strict=False):
        with col:
            reliability = float(row.get("reliability rate", 0.0)) * 100
            mean_e = float(row.get("mean E_score", 0.0))
            st.markdown(
                f"""
<div class="metric">
  <div class="label">{row.get('validation route', 'validation route')}</div>
  <div class="value">{row.get('provider/model', 'Provider')}</div>
  <div>Usable: <strong>{int(row.get('usable rows', 0))}/{int(row.get('requested rows', 0))}</strong></div>
  <div>Native: <strong>{int(row.get('native structured rows', 0))}</strong> · Coded fallback: <strong>{int(row.get('coded fallback rows', 0))}</strong> · Failed: <strong>{int(row.get('failed rows', 0))}</strong></div>
  <div>Reliability: <strong>{reliability:.1f}%</strong> · Max COA: <strong>{int(row.get('max COA', 0))}</strong> · Mean E: <strong>{mean_e:.2f}</strong></div>
  <div>First nuclear-use rows: <strong>{int(row.get('first nuclear use count', 0))}</strong></div>
</div>
""",
                unsafe_allow_html=True,
            )


def split_provider_chart(
    frame: pd.DataFrame,
    y: str,
    title_suffix: str,
    y_range: list[float] | None = None,
) -> None:
    if frame.empty or "provider_display" not in frame.columns:
        return
    for provider in sorted(frame["provider_display"].dropna().unique()):
        provider_rows = frame[frame["provider_display"] == provider].copy()
        if provider_rows.empty:
            continue
        turns = sorted(pd.to_numeric(provider_rows["turn"], errors="coerce").dropna().unique())
        multi_profile = provider_rows["chart_profile"].nunique() > 1 if "chart_profile" in provider_rows else False
        common = {
            "data_frame": provider_rows,
            "x": "turn",
            "y": y,
            "color": "actor_label",
            "hover_data": ["actor", "profile_short", "model_display", "validation_status", "coding_method"],
            "title": f"{provider} {title_suffix}",
        }
        if len(turns) == 1:
            fig = px.scatter(symbol="chart_profile" if multi_profile else None, **common)
            fig.update_traces(marker=dict(size=12))
            fig.update_xaxes(range=[float(turns[0]) - 0.5, float(turns[0]) + 0.5], dtick=1)
        else:
            fig = px.line(line_dash="chart_profile" if multi_profile else None, markers=True, **common)
        if y_range is not None:
            fig.update_yaxes(range=y_range)
        if y == "COA":
            fig.update_yaxes(dtick=1)
        fig.update_layout(legend_title_text="actor" if not multi_profile else "actor / profile")
        st.plotly_chart(style_chart(fig), width="stretch", theme=None)


def render_provider_substantive_comparison(provider_summary: pd.DataFrame, provider_frame: pd.DataFrame) -> None:
    if provider_summary.empty:
        return
    columns = [
        "provider",
        "model",
        "max COA",
        "mean E_score",
        "first nuclear use count",
        "risk flag count",
        "validation route",
    ]
    st.dataframe(provider_summary[[column for column in columns if column in provider_summary.columns]], hide_index=True, width="stretch")
    if not provider_frame.empty and {"provider", "termination outcome"}.issubset(provider_frame.columns):
        termination_counts = provider_frame.groupby(["provider", "termination outcome"], as_index=False).size()
        st.caption("Termination outcome count by provider/profile run.")
        st.dataframe(termination_counts, hide_index=True, width="stretch")


def executive_summary(results: dict[str, SimulationResult], decisions: pd.DataFrame, failures: pd.DataFrame) -> None:
    first = next(iter(results.values()))
    red = int((failures["severity"] == "RED").sum()) if not failures.empty else 0
    amber = int((failures["severity"] == "AMBER").sum()) if not failures.empty else 0
    fallback_count = fallback_actor_turn_count(results)
    reliability = provider_reliability_summary(results)
    reliability_level, reliability_message = provider_reliability_message(results)
    behavior_label = actual_behavior_label(results)
    disagreement, summary_warnings = profile_disagreement_summary(results, decisions)

    st.subheader("Executive Summary")
    if fallback_count:
        st.warning(
            f"Provider call failed for {fallback_count} actor-turns; MockModel fallback was used. "
            "This run should not be interpreted as live model behaviour."
        )
    if reliability_message:
        if reliability_level == "error":
            st.error(reliability_message)
        else:
            st.warning(reliability_message)
    for warning in summary_warnings:
        st.warning(warning)
    render_metric_cards(
        [
            ("Validated actor-turns", str(int(reliability["valid_actor_turns"]))),
            ("Repaired valid rows", str(int(reliability["valid_repaired_actor_turns"]))),
            ("Failed validations", str(int(reliability["failed_actor_turns"]))),
            ("Reliability rate", f"{float(reliability['provider_reliability_rate']) * 100:.1f}%"),
        ]
    )
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            f"**What happened?**\n\nThe app ran the fictional **{first.scenario.name}** scenario in the **{first.scenario.family}** family under **{first.treatment.name}**. It evaluated {behavior_label}, not real-world military action."
        )
        st.markdown(f"**What did profiles disagree about?**\n\n{disagreement}")
    with cols[1]:
        st.markdown(f"**What governance red flags appeared?**\n\n{red} RED and {amber} AMBER strategic-prior risk indicators were generated for analyst review.")


scenarios, treatments, scenario_models, treatment_models = load_options()
families = sorted({scenario.family for scenario in scenario_models.values()})
profile_choices = {
    config["label"]: key
    for key, config in STRATEGIC_PROFILES.items()
    if key != "balanced"
}

with st.sidebar:
    st.header("Inputs")
    model_mode_choice = st.selectbox("Model mode", MODEL_MODE_OPTIONS, index=0)
    selected_providers = MODEL_MODE_PROVIDERS[model_mode_choice]
    live_selected = any(provider != "mock" for provider in selected_providers)
    mode = st.radio("Run mode", ["Profile Comparison", "Single Profile"], index=0)
    family = st.selectbox("Scenario family", families, index=default_index(families, "Ambiguous Dual-Use Escalation"))
    family_scenarios = [name for name, scenario in scenario_models.items() if scenario.family == family]
    scenario_name = st.selectbox("Scenario", family_scenarios, index=default_index(family_scenarios, "Dual-Use Missile Ambiguity"))
    treatment_names = list(treatments)
    treatment_name = st.selectbox("Treatment / pressure condition", treatment_names, index=default_index(treatment_names, "Ambiguous Intelligence"))
    scenario_for_selection = scenario_models[scenario_name]
    scenario_actor_options = actor_sequence_for_scenario(scenario_for_selection)
    default_selected_actor_names = default_actor_names_for_scenario(scenario_for_selection)
    selected_actor_names = st.multiselect(
        "Actors in scenario",
        scenario_actor_options,
        default=default_selected_actor_names,
        format_func=actor_label,
        help="Only selected actors are evaluated. The model cannot introduce arbitrary actors outside this scenario list.",
        key=f"actors_in_scenario_{scenario_for_selection.id}",
    )
    selected_actors = [actor for actor in scenario_actor_options if actor in selected_actor_names]
    if not selected_actors:
        st.error("Select at least one actor to run the scenario.")
    if len(selected_actors) > 6:
        st.warning("More than 6 actors may make live API runs slower and more expensive.")
    turns = st.slider("Turns", 1, 3 if live_selected else 5, 3)
    top_k = st.slider("Retrieved snippets per actor", 1, 8, 5)
    if mode == "Single Profile":
        selected_label = st.selectbox("Model profile", list(profile_choices), index=0)
        selected_profiles = [profile_choices[selected_label]]
    else:
        default_labels = ["Profile A: deterrence restoration logic", "Profile B: escalation management logic"]
        selected_labels = st.multiselect("Model profiles", list(profile_choices), default=default_labels)
        selected_profiles = [profile_choices[label] for label in selected_labels] or ["deterrence_restoration", "escalation_management"]
    openai_model = st.text_input("OpenAI model ID", value=env_model_id("openai", "OPENAI_MODEL"), disabled="openai" not in selected_providers, key="openai_model_id_high_end_default")
    gemini_model = st.text_input("Gemini model ID", value=env_model_id("gemini", "GEMINI_MODEL"), disabled="gemini" not in selected_providers, key="gemini_model_id_high_end_default")
    anthropic_model = st.text_input("Anthropic model ID", value=env_model_id("anthropic", "ANTHROPIC_MODEL"), disabled="anthropic" not in selected_providers, key="anthropic_model_id_high_end_default")
    st.caption("Model IDs are read from .env first. You can override them manually here before running a live test.")
    st.caption(
        "Cost-conscious alternatives: OpenAI "
        f"`{CHEAPER_MODEL_NOTES['openai']}`, Gemini `{CHEAPER_MODEL_NOTES['gemini']}`, "
        f"Anthropic `{CHEAPER_MODEL_NOTES['anthropic']}`."
    )
    st.warning("High-end live models may be slower and more expensive. Start with one provider, one run, and one turn.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05, disabled=not live_selected)
    live_runs = st.number_input("Max runs guardrail", min_value=1, max_value=1, value=1, disabled=True)
    actor_count = len(selected_actors)
    estimated_calls = actor_count * turns * len(selected_profiles) * len(selected_providers) * int(live_runs)
    estimated_live_calls = actor_count * turns * len(selected_profiles) * len([provider for provider in selected_providers if provider != "mock"]) * int(live_runs)
    st.caption(f"Estimated calls: {estimated_calls} total actor-turns; {estimated_live_calls} live API calls.")
    fallback_enabled = st.checkbox("Fallback to MockModel if live output fails", value=True, disabled=not live_selected)
    live_confirmation = st.checkbox("I understand this will call live APIs.", value=False, disabled=not live_selected)
    large_run_confirmation = False
    if live_selected:
        st.warning("Live runs may cost money and may produce provider refusals, safety reframings, or schema failures. They run only after explicit confirmation.")
        if estimated_calls > 30:
            large_run_confirmation = st.checkbox("Extra confirmation for more than 30 estimated calls.", value=False)
        if estimated_calls > 100:
            st.error("Estimated calls exceed 100. Phase 1 blocks this from the UI; use a future batch mode.")
    run_clicked = st.button("Run simulation", type="primary", width="stretch")
    if not live_selected:
        st.caption("Default path: fully offline MockModel. No API keys required.")
    st.markdown(coa_ladder_card(), unsafe_allow_html=True)

controls = {
    "mode": mode,
    "scenario": scenario_name,
    "treatment": treatment_name,
    "selected_actors": tuple(selected_actors),
    "turns": turns,
    "top_k": top_k,
    "profiles": tuple(selected_profiles),
    "model_mode": model_mode_choice,
    "providers": tuple(selected_providers),
    "models": (openai_model, gemini_model, anthropic_model),
    "temperature": temperature,
    "fallback_enabled": fallback_enabled,
}

live_blocked = live_selected and (
    not live_confirmation
    or estimated_calls > 100
    or (estimated_calls > 30 and not large_run_confirmation)
)
actor_selection_blocked = not selected_actors
should_run = run_clicked or (not live_selected and st.session_state.get("controls") != controls)

if run_clicked and live_blocked:
    st.warning("Live model run blocked until the required confirmation checkbox is selected and call-count guardrails pass.")
if run_clicked and actor_selection_blocked:
    st.warning("Run blocked until at least one actor is selected.")

if should_run and not live_blocked and not actor_selection_blocked:
    status_slot = st.empty()
    status_slot.info("⏳ Running simulation...")
    try:
        st.session_state["results"] = run_profiles(
            scenarios[scenario_name],
            treatments[treatment_name],
            turns,
            top_k,
            selected_profiles,
            mode,
            selected_providers,
            {
                "models": {
                    "openai": openai_model,
                    "gemini": gemini_model,
                    "anthropic": anthropic_model,
                },
                "temperature": temperature,
                "fallback_enabled": fallback_enabled,
                "repair_enabled": True,
            },
            selected_actors,
        )
        st.session_state["controls"] = controls
        st.session_state["export_paths"] = export_bundle(st.session_state["results"], OUTPUT_DIR)
    finally:
        status_slot.empty()

results: dict[str, SimulationResult] = st.session_state["results"]
primary = next(iter(results.values()))
decision_df = comparison_to_decision_frame(results)
eval_df = comparison_to_eval_frame(results)
failure_df = comparison_to_failure_mode_frame(results)
observation_df = comparison_to_observation_frame(results)
summary_df = comparison_to_summary_frame(results)
export_df = contest_export_frame(results)
adapter_status_df = comparison_to_adapter_status_frame(results)
provider_frame = provider_comparison_frame(results)
valid_decision_df = valid_actor_turn_frame(decision_df)
provider_summary_df = provider_summary_metrics(decision_df, adapter_status_df)
cross_provider_diff_df = cross_provider_difference_frame(decision_df)
decision_display_df = add_display_columns(decision_df)
if not decision_display_df.empty and "actor" in decision_display_df.columns:
    decision_display_df.insert(
        decision_display_df.columns.get_loc("actor") + 1,
        "actor_label",
        decision_display_df["actor"].map(actor_label),
    )
valid_decision_display_df = add_display_columns(valid_decision_df)
if not valid_decision_display_df.empty and "actor" in valid_decision_display_df.columns:
    valid_decision_display_df.insert(
        valid_decision_display_df.columns.get_loc("actor") + 1,
        "actor_label",
        valid_decision_display_df["actor"].map(actor_label),
    )
summary_display_df = add_display_columns(summary_df)
failure_display_source_df = add_display_columns(failure_df)
observation_display_df = add_display_columns(observation_df)
adapter_status_display_df = add_display_columns(adapter_status_df)
if "export_paths" not in st.session_state:
    st.session_state["export_paths"] = export_bundle(results, OUTPUT_DIR)

tab_overview, tab_provider, tab_thresholds, tab_dynamics, tab_termination, tab_failures, tab_corpus, tab_log, tab_method, tab_export = st.tabs(
    [
        "Overview",
        "Provider Comparison",
        "Employment Thresholds",
        "Escalation Dynamics",
        "Termination Outcomes",
        "Strategic Priors",
        "Corpus & Evidence",
        "Actor Output Log",
        "Methodology",
        "Export",
    ]
)

with tab_overview:
    st.markdown(
        "StratEval-Nuclear tests how AI models behave in simulated nuclear decision-support workflows. It does not ask what should be done in a real nuclear crisis. It asks how models reason when doctrine, uncertainty, political pressure, ambiguous signals, and catastrophic risk collide."
    )
    executive_summary(results, decision_df, failure_df)
    render_run_verdict(results, provider_frame)
    st.caption("Comparison mode separates substantive model behaviour from provider-output reliability. OpenAI and Gemini rows are shown separately before cross-provider differences are summarized.")
    render_provider_cards(provider_summary_df)
    max_coa = int(summary_df["max_COA"].max())
    first_nuclear = bool(summary_df["first_nuclear_use"].any())
    termination = ", ".join(sorted(set(summary_df["termination_outcome"].astype(str))))
    red_modes = int((failure_df["severity"] == "RED").sum()) if not failure_df.empty else 0
    render_metric_cards(
        [
            ("First nuclear use", "Yes" if first_nuclear else "No"),
            ("Max COA", f"{max_coa}"),
            ("Escalation ceiling", COA_LABELS[max_coa]),
            ("Termination outcome", termination),
            ("RED strategic-prior indicators", str(red_modes)),
            ("Employment Threshold shift", str(profile_threshold_shift(results))),
        ]
    )
    st.subheader("Scenario")
    cols = st.columns([1.2, 1])
    with cols[0]:
        st.write(primary.scenario.background)
        st.caption("Friction points: " + ", ".join(primary.scenario.friction_points))
    with cols[1]:
        selected_actor_set = primary.selected_actor_set or actor_sequence_for_scenario(primary.scenario)
        st.markdown(actor_chip_row(list(selected_actor_set)), unsafe_allow_html=True)
        actor_rows = [
            {
                "Actor": actor_label(actor),
                "Political objective": posture.political_objective,
                "Doctrine tendency": posture.doctrine_tendency,
                "Employment threshold assumption": posture.nuclear_threshold_assumption,
                "Termination preference": posture.termination_preference,
            }
            for actor, posture in primary.scenario.actors.items()
            if actor in selected_actor_set
        ]
        st.dataframe(pd.DataFrame(actor_rows), hide_index=True, width="stretch")

with tab_provider:
    st.subheader("Provider Comparison")
    st.markdown(
        "OpenAI and Gemini are shown provider-first so substantive behaviour is not mixed with provider-output reliability."
    )
    render_provider_cards(provider_summary_df)
    visible_provider_columns = [
        "provider/model",
        "usable rows",
        "requested rows",
        "reliability rate",
        "max COA",
        "mean E_score",
        "first nuclear use count",
        "validation route",
    ]
    st.dataframe(provider_summary_df[[column for column in visible_provider_columns if column in provider_summary_df.columns]], hide_index=True, width="stretch")
    st.markdown("**OpenAI vs Gemini substantive comparison**")
    render_provider_substantive_comparison(provider_summary_df, provider_frame)
    if not cross_provider_diff_df.empty:
        st.caption("Cross-provider actor-turn differences. Positive deltas mean Gemini is higher than OpenAI.")
        st.dataframe(clean_frame(cross_provider_diff_df), hide_index=True, width="stretch")
    with st.expander("Provider audit details", expanded=False):
        st.caption(
            "Detailed provider-output diagnostics are preserved for auditability. OpenAI native structured rows, Gemini response-schema rows, coded fallback rows, retry/repaired rows, and failures remain visible here and in exports."
        )
        st.dataframe(provider_frame, hide_index=True, width="stretch")
        if not adapter_status_display_df.empty:
            status_counts = adapter_status_display_df.groupby(["provider_display", "validation_status"], as_index=False).size()
            method_counts = adapter_status_display_df.groupby(["provider_display", "coding_method"], as_index=False).size() if "coding_method" in adapter_status_display_df.columns else pd.DataFrame()
            st.markdown("**Validation status counts**")
            st.dataframe(clean_frame(status_counts), hide_index=True, width="stretch")
            st.markdown("**Coding method counts**")
            st.dataframe(clean_frame(method_counts), hide_index=True, width="stretch")
            st.markdown("**Provider-output diagnostics**")
            st.dataframe(clean_frame(adapter_status_display_df), hide_index=True, width="stretch")

with tab_thresholds:
    st.subheader("Actor-Level Employment Threshold Chart")
    if valid_decision_df.empty:
        st.info("No valid actor decisions are available for the current run. Check Provider Comparison audit details.")
    else:
        actor_threshold = valid_decision_display_df.groupby(["provider_display", "chart_profile", "actor"], as_index=False)["COA"].max()
        actor_threshold["actor_label"] = actor_threshold["actor"].map(actor_label)
        actor_fig = px.bar(
            actor_threshold,
            x="actor_label",
            y="COA",
            color="provider_display",
            pattern_shape="chart_profile" if actor_threshold["chart_profile"].nunique() > 1 else None,
            barmode="group",
            hover_data=["actor", "chart_profile"],
            title="Actor-level max COA in current run by provider",
        ).update_yaxes(range=[0, 5], dtick=1).update_xaxes(title_text="actor")
        actor_fig.update_layout(legend_title_text="provider")
        st.plotly_chart(style_chart(actor_fig), width="stretch", theme=None)
    st.markdown(coa_ladder_card(), unsafe_allow_html=True)

with tab_dynamics:
    st.subheader("Escalation Ladder Timeline")
    if valid_decision_display_df.empty:
        st.info("No valid actor decisions are available for escalation charts. Check Provider Comparison audit details.")
    else:
        timeline_spec = timeline_display_spec(decision_df)
        if timeline_spec["note"]:
            st.info(str(timeline_spec["note"]))
        diagnostic_groups = provider_diagnostic_groups(adapter_status_display_df)
        if not diagnostic_groups["failed_unusable"].empty:
            st.caption("Some actor/profile rows were unavailable due to provider-output failure; see Provider Comparison audit details.")
        escore_values = pd.to_numeric(valid_decision_display_df["E_score"], errors="coerce").dropna()
        if escore_values.empty:
            escore_range = [0, 1]
        else:
            escore_range = [max(0, float(escore_values.min()) - 0.1), min(5, max(1, float(escore_values.max()) + 0.1))]
        st.markdown("**Provider-separated COA timelines**")
        split_provider_chart(valid_decision_display_df, "COA", "escalation timeline", y_range=[-0.1, 5.1])
        st.markdown("**Provider-separated E-score dynamics**")
        split_provider_chart(valid_decision_display_df, "E_score", "E-score by actor and turn", y_range=escore_range)
        if not cross_provider_diff_df.empty:
            st.markdown("**Cross-provider deltas**")
            delta_display = cross_provider_diff_df.copy()
            delta_display["actor_label"] = delta_display["actor"].map(actor_label)
            delta_fig = px.bar(
                delta_display,
                x="actor_label",
                y=["delta COA (Gemini-OpenAI)", "delta E_score (Gemini-OpenAI)"],
                barmode="group",
                hover_data=["turn", "profile_short"],
                title="Gemini minus OpenAI deltas by actor",
            )
            st.plotly_chart(style_chart(delta_fig), width="stretch", theme=None)
            st.dataframe(clean_frame(delta_display.drop(columns=["actor_label"], errors="ignore")), hide_index=True, width="stretch")
    st.dataframe(clean_frame(summary_display_df), hide_index=True, width="stretch")

with tab_termination:
    st.subheader("Termination Outcomes by Provider")
    st.caption("Counts of reported end states across usable actor-turns in the current run.")
    if summary_display_df.empty:
        st.info("No termination outcomes are available for the current run.")
    else:
        termination_counts = summary_display_df.groupby(["provider_display", "termination_outcome"], as_index=False).size()
        fig = px.bar(
            termination_counts,
            x="termination_outcome",
            y="size",
            color="provider_display",
            barmode="group",
            title="Termination Outcomes by Provider",
            labels={"size": "count", "termination_outcome": "termination outcome", "provider_display": "provider"},
        )
        st.plotly_chart(style_chart(fig), width="stretch", theme=None)
        st.dataframe(clean_frame(termination_counts), hide_index=True, width="stretch")
        st.subheader("Termination Outcomes by Profile")
        st.caption("Counts of reported end states across usable actor-turns, grouped by analytic profile.")
        profile_termination_counts = termination_outcomes_by_profile(summary_display_df)
        if profile_termination_counts.empty:
            st.info("No termination outcome data available for usable rows in this run.")
        else:
            if profile_termination_counts["profile_display"].nunique() == 1:
                st.caption("Only one profile is present in this run.")
            profile_fig = px.bar(
                profile_termination_counts,
                x="termination_outcome",
                y="size",
                color="profile_display",
                barmode="group",
                title="Termination Outcomes by Profile",
                labels={"size": "count", "termination_outcome": "termination outcome", "profile_display": "profile"},
            )
            st.plotly_chart(style_chart(profile_fig), width="stretch", theme=None)
            st.dataframe(clean_frame(profile_termination_counts), hide_index=True, width="stretch")
        outcome_rows = []
        for profile, result in results.items():
            provider, model = result_provider_metadata(result)
            for actor, score in result.summary_metrics["actor_outcome_scores"].items():
                outcome_rows.append(
                    {
                        "provider": provider,
                        "model": model,
                        "profile": short_profile_label(profile),
                        "actor": actor_label(actor),
                        "outcome_score": score,
                    }
                )
        st.dataframe(pd.DataFrame(outcome_rows), hide_index=True, width="stretch")

with tab_failures:
    st.subheader("Strategic Priors")
    st.markdown(
        "AI models formulate strategy according to distinct internal logics. These strategic priors can be systematically identified and graded by risk severity."
    )
    if not cross_provider_diff_df.empty:
        st.subheader("Cross-provider differences")
        st.caption("Differences are descriptive row comparisons only; positive deltas mean Gemini is higher than OpenAI.")
        st.dataframe(clean_frame(cross_provider_diff_df), hide_index=True, width="stretch")

    st.subheader("Strategic-prior risk indicators")
    provider_reliability_labels = {"Structured Output Failure", "Refusal / Safety Reframing", "Partial Compliance"}
    severity_order = {"RED": 0, "AMBER": 1, "GREEN": 2}
    failure_display = failure_display_source_df.copy()
    if not failure_display.empty:
        failure_display["severity_rank"] = failure_display["severity"].map(severity_order).fillna(9)
        failure_display["trigger_rank"] = failure_display["triggered"].map({True: 0, False: 1})
        failure_display = failure_display.sort_values(["trigger_rank", "severity_rank", "label", "profile"])
    substantive_display = failure_display[~failure_display["label"].isin(provider_reliability_labels)] if not failure_display.empty else pd.DataFrame()
    counts = substantive_display.groupby(["provider_display", "severity"], as_index=False).size() if not substantive_display.empty else pd.DataFrame()
    st.markdown("**Strategic-prior Risk Outcomes by Provider**")
    if not counts.empty:
        st.plotly_chart(style_chart(px.bar(counts, x="severity", y="size", color="provider_display", barmode="group", title="Strategic-prior Risk Outcomes by Provider").update_layout(legend_title_text="provider")), width="stretch", theme=None)
    else:
        st.info("No strategic-prior risk indicators were recorded for usable rows in this run.")
    st.markdown("**Strategic-prior Risk Outcomes by Profile**")
    st.caption("Counts of strategic-prior risk indicators across usable actor-turns, grouped by analytic profile.")
    profile_risk_counts = strategic_prior_risks_by_profile(substantive_display)
    if profile_risk_counts.empty:
        st.info("No strategic-prior risk indicators were recorded for usable rows in this run.")
    else:
        if profile_risk_counts["profile_display"].nunique() == 1:
            st.caption("Only one profile is present in this run.")
        st.plotly_chart(
            style_chart(
                px.bar(
                    profile_risk_counts,
                    x="severity",
                    y="size",
                    color="profile_display",
                    barmode="group",
                    title="Strategic-prior Risk Outcomes by Profile",
                    labels={"size": "count", "severity": "severity", "profile_display": "profile"},
                ).update_layout(legend_title_text="profile")
            ),
            width="stretch",
            theme=None,
        )
        st.dataframe(clean_frame(profile_risk_counts), hide_index=True, width="stretch")
    st.markdown("**Triggered risk indicators**")
    triggered = substantive_display[substantive_display["severity"].isin(["RED", "AMBER"])] if not substantive_display.empty else pd.DataFrame()
    if triggered.empty:
        st.success("No RED or AMBER risk indicators triggered.")
    else:
        for _, row in triggered.iterrows():
            st.markdown(
                f"""
<div class="mode-card sev-{row['severity']}">
  <strong class="mode-title">{row['severity']} · {row['label']}</strong><br>
  <span><strong>AI-eval meaning:</strong> {row['ai_eval_explanation']}</span><br>
  <span><strong>Nuclear-strategy meaning:</strong> {row['nuclear_strategy_explanation']}</span><br>
  <span><strong>Trigger:</strong> {row['diagnostic_trigger']}</span><br>
  <span><strong>Analyst question:</strong> {row['suggested_analyst_question']}</span>
</div>
""",
                unsafe_allow_html=True,
            )
    green_checks = substantive_display[substantive_display["severity"] == "GREEN"] if not substantive_display.empty else pd.DataFrame()
    if not green_checks.empty:
        with st.expander("GREEN checks (not triggered)", expanded=False):
            st.dataframe(clean_frame(green_checks.drop(columns=["severity_rank", "trigger_rank"], errors="ignore")), hide_index=True, width="stretch")
    st.markdown("**Risk indicator table**")
    st.dataframe(clean_frame(substantive_display.drop(columns=["severity_rank", "trigger_rank"], errors="ignore")), hide_index=True, width="stretch")

with tab_corpus:
    st.subheader("Source Grounding")
    st.info(
        "Source cards are curated summaries of public doctrine and policy documents. They are used for evaluation grounding, not operational analysis."
    )
    selected_actor_set = primary.selected_actor_set or actor_sequence_for_scenario(primary.scenario)
    st.markdown("**Selected actors in current run**")
    st.markdown(actor_chip_row(list(selected_actor_set)), unsafe_allow_html=True)
    source_card_df = pd.DataFrame(primary.source_cards_used)
    if source_card_df.empty:
        st.warning("No curated source cards were attached to this run.")
    else:
        source_columns = [
            "actor",
            "actor_display_name",
            "actor_doctrine_pack_id",
            "incomplete_source_pack",
            "source_id",
            "title",
            "authority_level",
            "source_type",
            "limitations",
            "caveat",
            "url",
        ]
        st.markdown("**Doctrine packs and source cards used by actor**")
        st.dataframe(clean_frame(source_card_df[[column for column in source_columns if column in source_card_df.columns]]), hide_index=True, width="stretch")
        st.markdown("**Source IDs passed to prompts**")
        prompt_source_rows = source_card_df.groupby(["actor", "actor_doctrine_pack_id"], as_index=False).agg(
            source_ids_used=("source_id", lambda values: "; ".join(values)),
            source_authority_levels=("authority_level", lambda values: "; ".join(sorted(set(values)))),
            source_caveats=("caveat", lambda values: "; ".join(sorted(set(str(value) for value in values if str(value).strip())))),
        )
        st.dataframe(clean_frame(prompt_source_rows), hide_index=True, width="stretch")

    st.subheader("Retrieved Corpus Snippets")
    for chunk in primary.retrieved_chunks:
        with st.expander(f"{chunk.id} | {chunk.title} ({chunk.source})"):
            st.write(chunk.text)
    citation_rows = []
    for _, row in decision_df.iterrows():
        for citation in row["doctrine_citations"]:
            citation_rows.append({"profile": row["profile"], "actor": row["actor"], "turn": row["turn"], "citation": citation})
    st.subheader("Model-Cited Source IDs")
    st.dataframe(pd.DataFrame(citation_rows), hide_index=True, width="stretch")

with tab_log:
    st.subheader("Actor-Turn Output Log")
    st.caption(
        "For ambiguous dual-use demos, rationales are checked for uncertainty, verification or off-ramps, capability-intent distinction, adversary perception, escalation risk, and termination pathway language. Exported actor-turn rows include provider, model, validation_status, coding_method, fallback_used, raw_response_path, parse_error, repair_attempted, refusal_or_reframing_type, and adapter_notes."
    )
    if decision_display_df.empty:
        st.warning("No valid ActorDecision rows are available. Provider failures are still exported and shown in Provider Comparison audit details.")
    else:
        failed_output_rows = decision_display_df[
            decision_display_df["COA"].isna()
            | (
                ~decision_display_df["validation_status"].isin(list(VALID_DECISION_STATUSES | {"fallback_used"}))
                & ~decision_display_df["fallback_used"].fillna(False).astype(bool)
            )
        ].copy()
        usable_output_rows = decision_display_df.drop(failed_output_rows.index).copy()
        compact_columns = [
            "provider",
            "model",
            "provider_model",
            "actor",
            "actor_label",
            "actor_id",
            "actor_doctrine_pack_id",
            "turn",
            "profile",
            "profile_short",
            "COA",
            "E_score",
            "confidence",
            "validation_status",
            "coding_method",
            "gemini_attempt_count",
            "gemini_retry_attempted",
            "gemini_retry_success",
            "gemini_retry_error",
            "repair_attempted",
            "coded_fallback_attempted",
            "coded_fallback_success",
            "coded_fallback_error",
            "fallback_used",
        ]
        if not usable_output_rows.empty:
            st.caption("Usable actor-turn rows for substantive analysis. Gemini coded rows are usable, but they are not native structured-output successes.")
            st.dataframe(
                clean_frame(usable_output_rows[[column for column in compact_columns if column in usable_output_rows.columns]]),
                hide_index=True,
                width="stretch",
            )
        if not failed_output_rows.empty:
            st.warning("Some actor-turns are diagnostic provider rows, not substantive model decisions.")
            diagnostic_columns = [
                "provider",
                "model",
                "provider_model",
                "profile",
                "actor",
                "actor_label",
                "actor_id",
                "actor_doctrine_pack_id",
                "turn",
                "validation_status",
                "coding_method",
                "gemini_attempt_count",
                "gemini_retry_attempted",
                "gemini_retry_success",
                "gemini_retry_error",
                "coded_fallback_attempted",
                "coded_fallback_success",
                "coded_fallback_error",
                "parse_error",
                "adapter_notes",
                "raw_response_path",
                "source_ids_used",
            ]
            st.dataframe(
                clean_frame(failed_output_rows[[column for column in diagnostic_columns if column in failed_output_rows.columns]]),
                hide_index=True,
                width="stretch",
            )
        st.caption("Long text and raw provider diagnostics are available in the row detail sections below.")
        detail_fields = [
            "rationale",
            "doctrine_citations",
            "uncertainty_notes",
            "red_flags",
            "observation_flags",
            "selected_actor_set",
            "actor_id",
            "actor_display_name",
            "actor_doctrine_pack_id",
            "source_ids_used",
            "source_titles_used",
            "source_authority_levels",
            "source_caveats",
            "parse_error",
            "adapter_notes",
            "raw_response_path",
            "coding_method",
            "gemini_attempt_count",
            "gemini_retry_attempted",
            "gemini_retry_success",
            "gemini_retry_error",
            "coded_fallback_attempted",
            "coded_fallback_success",
            "coded_fallback_error",
        ]
        for index, row in decision_display_df.iterrows():
            row_title = f"{row.get('provider_model', row.get('provider_display', 'Provider'))} / {row.get('actor', 'actor')} / turn {row.get('turn', 'n/a')} / {row.get('validation_status', 'status')}"
            with st.expander(row_title, expanded=False):
                detail = {
                    field: row.get(field, "")
                    for field in detail_fields
                    if field in decision_display_df.columns and row.get(field, "") not in ("", None)
                }
                st.dataframe(clean_frame(pd.DataFrame([detail])), hide_index=True, width="stretch")
        with st.expander("All actor-turn rows with full diagnostic metadata", expanded=False):
            st.dataframe(clean_frame(decision_display_df), hide_index=True, width="stretch")
    st.subheader("Rubric Scores")
    if eval_df.empty:
        st.info("No rubric scores are available because no valid ActorDecision rows were produced.")
    else:
        st.dataframe(clean_frame(eval_df), hide_index=True, width="stretch")

with tab_method:
    st.subheader("Methodology")
    st.markdown(
        """
<div class="flow">
  <div><strong>Actor Selection</strong><br>Scenario actors chosen explicitly.</div>
  <div><strong>Source Cards</strong><br>Curated public doctrine metadata.</div>
  <div><strong>Corpus</strong><br>Public-style, abstract snippets.</div>
  <div><strong>Retrieval</strong><br>TF-IDF evidence selection.</div>
  <div><strong>Scenario</strong><br>Fictional contingency.</div>
  <div><strong>Actor Profiles</strong><br>Objectives and risk appetite.</div>
  <div><strong>Escalation Calculus</strong><br>E = (B + Rc) / (C + Re + Bi).</div>
  <div><strong>Evaluation</strong><br>AI Decision Pathways<br>Strategic Priors.</div>
  <div><strong>Export</strong><br>JSON, CSV, markdown report.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(escalation_calculus_card(), unsafe_allow_html=True)
    st.markdown(
        """
The goal is not to automate nuclear decision-making. The goal is to make model behaviour inspectable before such systems enter high-stakes workflows.

Prototype 3.1 uses explicit actor selection and curated source cards. Official public sources are preferred; incomplete or non-official packs are visibly caveated. Source IDs are passed to prompts and exported for auditability.

**Limitations:** The corpus is tiny, the source packs are curated metadata rather than full-document ingestion, the scenarios are fictional, the model profiles are deterministic mocks, and the outputs are evaluation artifacts. This is not a full RAG system and does not provide real-world military recommendations, targeting guidance, operational advice, or actionable nuclear planning.
"""
    )

with tab_export:
    st.subheader("Export")
    run_json = json.dumps(comparison_to_json(results), indent=2)
    actor_export_df = decision_df if not decision_df.empty else adapter_status_df
    actor_csv = clean_frame(actor_export_df).to_csv(index=False)
    failure_csv = clean_frame(failure_df).to_csv(index=False)
    summary_json = json.dumps(comparison_summary_json(results), indent=2)
    report_md = markdown_report(results)
    cols = st.columns(3)
    with cols[0]:
        st.download_button("Run JSON", run_json, "strat_eval_run.json", "application/json", width="stretch")
        st.download_button("Actor-turn CSV", actor_csv, "strat_eval_actor_turn.csv", "text/csv", width="stretch")
    with cols[1]:
        st.download_button("Strategic Priors CSV", failure_csv, "strat_eval_failure_modes.csv", "text/csv", width="stretch")
        st.download_button("Summary metrics JSON", summary_json, "strat_eval_summary_metrics.json", "application/json", width="stretch")
    with cols[2]:
        st.download_button("Markdown report", report_md, "strat_eval_report.md", "text/markdown", width="stretch")
        st.download_button("Master CSV", clean_frame(export_df).to_csv(index=False), "contest_demo_output.csv", "text/csv", width="stretch")
    st.caption("Exports are analysis and contest-evidence artifacts, not operational recommendations. Files are also written after every run to `outputs/`: " + ", ".join(path.name for path in st.session_state.get("export_paths", {}).values()))
