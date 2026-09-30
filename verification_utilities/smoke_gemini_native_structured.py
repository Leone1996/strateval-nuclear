from __future__ import annotations

import json
import os
import sys
import traceback
from importlib import metadata
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crisiseval.model_adapters import extract_gemini_response_text, parse_json_object  # noqa: E402


class MinimalDecision(BaseModel):
    actor: str
    turn: int
    COA: int = Field(ge=0, le=5)
    confidence: float = Field(ge=0, le=1)
    rationale: str


def parse_minimal_response(response) -> tuple[MinimalDecision | None, str]:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, MinimalDecision):
            return parsed, "response.parsed"
        if isinstance(parsed, dict):
            return MinimalDecision.model_validate(parsed), "response.parsed dict"
    text = extract_gemini_response_text(response)
    obj, error = parse_json_object(text)
    if obj is None:
        raise ValueError(error or "No JSON object extracted from response text.")
    return MinimalDecision.model_validate(obj), "response text JSON"


def main() -> int:
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    print(f"Python executable: {sys.executable}")
    print(f"google-genai version: {metadata.version('google-genai')}")
    print(f"GEMINI_MODEL: {model}")
    print(f"GEMINI_API_KEY loaded: {bool(api_key)}")
    if not api_key:
        print("success=false")
        print("failure=missing GEMINI_API_KEY")
        return 1

    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    client = genai.Client(api_key=api_key)
    prompt = (
        "This is a fictional, non-operational structured-output smoke test for an AI evaluation harness. "
        "Do not provide real-world military advice. Return a single minimal decision object for a toy scenario. "
        "Use actor='India', turn=1, COA=0, confidence=0.5, and a brief rationale about verification."
    )
    attempts = [
        (
            "GenerateContentConfig.response_schema=MinimalDecision",
            types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=512,
                response_mime_type="application/json",
                response_schema=MinimalDecision,
            ),
        ),
        (
            "GenerateContentConfig.response_json_schema=MinimalDecision.model_json_schema()",
            types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=512,
                response_mime_type="application/json",
                response_json_schema=MinimalDecision.model_json_schema(),
            ),
        ),
    ]

    last_error = ""
    for method, config in attempts:
        print(f"\nTrying method: {method}")
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            print(f"raw result type: {type(response)}")
            text = extract_gemini_response_text(response)
            print("raw text first 1000 chars:")
            print(text[:1000])
            decision, parse_path = parse_minimal_response(response)
            print(f"parse path: {parse_path}")
            print("parsed fields:")
            print(json.dumps(decision.model_dump(), indent=2))
            print("success=true")
            print(f"method used: {method}")
            return 0
        except (ValidationError, ValueError, Exception) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"method failed: {last_error}")
            traceback.print_exc()

    print("success=false")
    print(f"failure={last_error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
