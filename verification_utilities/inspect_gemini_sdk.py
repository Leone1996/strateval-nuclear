from __future__ import annotations

import inspect
import sys
from importlib import metadata


def safe_signature(value) -> str:
    try:
        return str(inspect.signature(value))
    except Exception as exc:
        return f"<signature unavailable: {exc}>"


def main() -> int:
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception as exc:
        print(f"Failed to import google-genai: {exc}")
        return 1

    try:
        version = metadata.version("google-genai")
    except metadata.PackageNotFoundError:
        version = "not found"

    print(f"Python executable: {sys.executable}")
    print(f"google-genai version: {version}")
    print(f"google.genai import path: {getattr(genai, '__file__', '<unknown>')}")
    print(f"google.genai.types import path: {getattr(types, '__file__', '<unknown>')}")

    print("\nClient inspection:")
    print(f"genai.Client: {genai.Client}")
    print(f"genai.Client signature: {safe_signature(genai.Client)}")
    try:
        client = genai.Client(api_key="inspect-only")
        print(f"client.models type: {type(client.models)}")
        relevant_model_methods = [
            name
            for name in dir(client.models)
            if "generate" in name.lower() or "content" in name.lower()
        ]
        print(f"client.models relevant methods: {relevant_model_methods}")
        if hasattr(client.models, "generate_content"):
            print(f"client.models.generate_content signature: {safe_signature(client.models.generate_content)}")
    except Exception as exc:
        print(f"Could not instantiate inspect-only client: {exc}")

    print("\nGenerateContentConfig inspection:")
    config_cls = getattr(types, "GenerateContentConfig", None)
    print(f"types.GenerateContentConfig: {config_cls}")
    if config_cls is not None:
        print(f"GenerateContentConfig signature: {safe_signature(config_cls)}")
        fields = {}
        if hasattr(config_cls, "model_fields"):
            fields = getattr(config_cls, "model_fields")
        elif hasattr(config_cls, "__fields__"):
            fields = getattr(config_cls, "__fields__")
        field_names = sorted(fields) if isinstance(fields, dict) else []
        print(f"GenerateContentConfig field count: {len(field_names)}")
        schema_related = [
            name
            for name in field_names
            if "schema" in name.lower()
            or "mime" in name.lower()
            or "response" in name.lower()
            or "format" in name.lower()
        ]
        print(f"GenerateContentConfig schema/response fields: {schema_related}")
        for candidate in [
            "response_mime_type",
            "response_schema",
            "response_json_schema",
            "response_format",
        ]:
            print(f"supports {candidate}: {candidate in field_names}")

    print("\nOther relevant type objects:")
    for name in [
        "Schema",
        "Type",
        "GenerateContentResponse",
        "GenerateContentConfig",
        "HttpOptions",
    ]:
        value = getattr(types, name, None)
        print(f"types.{name}: {value}")

    likely = []
    if config_cls is not None:
        field_names = sorted(getattr(config_cls, "model_fields", {}) or getattr(config_cls, "__fields__", {}) or {})
        if "response_schema" in field_names and "response_mime_type" in field_names:
            likely.append("models.generate_content(..., config=types.GenerateContentConfig(response_mime_type='application/json', response_schema=<schema>))")
        if "response_json_schema" in field_names and "response_mime_type" in field_names:
            likely.append("models.generate_content(..., config=types.GenerateContentConfig(response_mime_type='application/json', response_json_schema=<json schema>))")
        if "response_format" in field_names:
            likely.append("models.generate_content(..., config=types.GenerateContentConfig(response_format=<format>))")
    print("\nLikely structured-output interface:")
    if likely:
        for item in likely:
            print(f"- {item}")
    else:
        print("- No obvious provider-native structured-output interface found in local SDK config fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
