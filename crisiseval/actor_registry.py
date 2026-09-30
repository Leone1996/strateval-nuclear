from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schema import CorpusChunk, Scenario


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = PROJECT_ROOT / "corpus"
SOURCE_CARD_DIR = REGISTRY_DIR / "source_cards"

ALLOWED_AUTHORITY_LEVELS = {
    "official",
    "official_translation",
    "authoritative_secondary",
    "caveated_secondary",
    "opaque_or_undeclared",
}
ALLOWED_SOURCE_TYPES = {
    "declaratory_policy",
    "defence_strategy",
    "nuclear_posture",
    "alliance_policy",
    "command_authority",
    "legal_text",
    "official_speech",
    "white_paper",
    "authoritative_analysis",
    "opacity_caveat",
}
REQUIRED_SOURCE_CARD_FIELDS = {
    "source_id",
    "title",
    "actor_scope",
    "issuing_body",
    "year",
    "authority_level",
    "source_type",
    "url",
    "reliability_note",
    "limitations",
    "tags",
    "summary_bullets",
    "doctrinal_priors",
    "safety_note",
}


@lru_cache
def load_actor_registry() -> dict[str, dict[str, Any]]:
    payload = _read_json(REGISTRY_DIR / "actor_registry.json")
    return {actor["display_name"]: actor for actor in payload.get("actors", [])}


@lru_cache
def load_actor_registry_by_id() -> dict[str, dict[str, Any]]:
    return {actor["actor_id"]: actor for actor in load_actor_registry().values()}


@lru_cache
def load_doctrine_packs() -> dict[str, dict[str, Any]]:
    payload = _read_json(REGISTRY_DIR / "doctrine_packs.json")
    return {pack["pack_id"]: pack for pack in payload.get("packs", [])}


@lru_cache
def load_source_registry() -> list[str]:
    payload = _read_json(REGISTRY_DIR / "source_registry.json")
    return list(payload.get("sources", []))


@lru_cache
def load_source_cards() -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    for path in sorted(SOURCE_CARD_DIR.glob("*.json")):
        card = _read_json(path)
        validate_source_card(card)
        cards[card["source_id"]] = card
    return cards


def validate_source_card(card: dict[str, Any]) -> None:
    missing = REQUIRED_SOURCE_CARD_FIELDS - set(card)
    if missing:
        raise ValueError(f"Source card {card.get('source_id', '<unknown>')} missing fields: {sorted(missing)}")
    if card["authority_level"] not in ALLOWED_AUTHORITY_LEVELS:
        raise ValueError(f"Invalid authority_level for {card['source_id']}: {card['authority_level']}")
    if card["source_type"] not in ALLOWED_SOURCE_TYPES:
        raise ValueError(f"Invalid source_type for {card['source_id']}: {card['source_type']}")
    if "wikipedia" in str(card.get("url", "")).lower():
        raise ValueError(f"Wikipedia source card is not allowed: {card['source_id']}")
    if card["authority_level"] not in {"official", "official_translation"} and not str(card.get("limitations", "")).strip():
        raise ValueError(f"Non-official source card requires limitations: {card['source_id']}")


def actor_record(actor_name: str) -> dict[str, Any]:
    registry = load_actor_registry()
    if actor_name in registry:
        return registry[actor_name]
    actor_id = slugify_actor(actor_name)
    return {
        "actor_id": actor_id,
        "display_name": actor_name,
        "short_label": actor_name,
        "flag": "",
        "nuclear_status": "scenario_only",
        "default_doctrine_pack_id": "scenario_only_incomplete_source_pack",
        "default_role_tags": ["scenario_context"],
        "caveat": "Scenario actor has no curated doctrine pack in Prototype 3.1.",
    }


def actor_display_label(actor_name: str) -> str:
    record = actor_record(actor_name)
    flag = record.get("flag", "")
    return f"{record.get('display_name', actor_name)} {flag}".strip()


def actor_sequence_for_selection(scenario: Scenario, selected_actors: list[str] | None = None) -> list[str]:
    ordered = scenario.actor_order or list(scenario.actors)
    if selected_actors is None:
        return ordered
    invalid = [actor for actor in selected_actors if actor not in scenario.actors]
    if invalid:
        raise ValueError(f"Selected actors are not present in scenario: {invalid}")
    selected = [actor for actor in selected_actors if actor in scenario.actors]
    if not selected:
        raise ValueError("At least one selected actor must belong to the scenario.")
    return [actor for actor in ordered if actor in selected]


def default_actor_names_for_scenario(scenario: Scenario) -> list[str]:
    return actor_sequence_for_selection(scenario)


def source_cards_for_actor(actor_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    record = actor_record(actor_name)
    pack = load_doctrine_packs().get(record["default_doctrine_pack_id"]) or _fallback_pack(record)
    cards = load_source_cards()
    return pack, [cards[source_id] for source_id in pack.get("source_ids", []) if source_id in cards]


def source_chunks_for_actor(actor_name: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[CorpusChunk]]:
    pack, cards = source_cards_for_actor(actor_name)
    chunks = [source_card_to_chunk(card) for card in cards]
    return pack, cards, chunks


def source_card_to_chunk(card: dict[str, Any]) -> CorpusChunk:
    summary = " ".join(card.get("summary_bullets", [])[:2])
    caveat = card.get("caveat") or card.get("limitations", "")
    text = (
        f"Authority: {card['authority_level']}. "
        f"Summary: {summary} "
        f"Caveat: {caveat}"
    ).strip()
    return CorpusChunk(
        id=card["source_id"],
        source=f"source_card:{card['source_id']}",
        title=card["title"],
        text=text,
    )


def source_prompt_lines(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return "- No curated source card available; treat doctrine grounding as incomplete."
    lines = []
    for card in cards:
        summary = " ".join(card.get("summary_bullets", [])[:1])
        caveat = card.get("caveat") or card.get("limitations", "")
        lines.append(
            f"- {card['source_id']} | {card['title']} | authority={card['authority_level']} | "
            f"summary={summary} | caveat={caveat}"
        )
    return "\n".join(lines)


def source_metadata_for_decision(
    selected_actors: list[str],
    actor_name: str,
    pack: dict[str, Any],
    cards: list[dict[str, Any]],
) -> dict[str, Any]:
    record = actor_record(actor_name)
    return {
        "selected_actor_set": list(selected_actors),
        "actor_id": record["actor_id"],
        "actor_display_name": record["display_name"],
        "actor_doctrine_pack_id": pack["pack_id"],
        "source_ids_used": [card["source_id"] for card in cards],
        "source_titles_used": [card["title"] for card in cards],
        "source_authority_levels": [card["authority_level"] for card in cards],
        "source_caveats": [card.get("caveat") or card.get("limitations", "") for card in cards],
    }


def attach_source_metadata(target: Any, metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        setattr(target, key, value)


def source_card_rows_for_actor(actor_name: str) -> list[dict[str, Any]]:
    pack, cards = source_cards_for_actor(actor_name)
    rows = []
    for card in cards:
        rows.append(
            {
                "actor": actor_name,
                "actor_display_name": actor_record(actor_name)["display_name"],
                "actor_doctrine_pack_id": pack["pack_id"],
                "incomplete_source_pack": bool(pack.get("incomplete_source_pack", False)),
                "source_id": card["source_id"],
                "title": card["title"],
                "authority_level": card["authority_level"],
                "source_type": card["source_type"],
                "url": card["url"],
                "limitations": card["limitations"],
                "caveat": card.get("caveat", ""),
            }
        )
    return rows


def slugify_actor(actor_name: str) -> str:
    return actor_name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _fallback_pack(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "pack_id": record.get("default_doctrine_pack_id", "scenario_only_incomplete_source_pack"),
        "actor_id": record["actor_id"],
        "display_name": f"{record['display_name']} scenario-only fallback pack",
        "hero_ready": False,
        "incomplete_source_pack": True,
        "source_ids": [],
        "pack_caveat": record["caveat"],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
