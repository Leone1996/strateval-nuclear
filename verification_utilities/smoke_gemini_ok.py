from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    print(f"Python executable: {sys.executable}")
    print(f"GEMINI_MODEL: {model}")
    print(f"GEMINI_API_KEY loaded: {bool(api_key)}")
    if not api_key:
        print("failure_category = missing key")
        return 1

    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents="Reply with exactly OK")
        text = getattr(response, "text", "") or str(response)
        print("raw minimal response first 1000 chars:")
        print(text[:1000])
        ok = text.strip() == "OK"
        print(f"minimal_status = {'valid' if ok else 'unexpected_response'}")
        if not ok:
            print("failure_category = malformed minimal response")
            return 1
        return 0
    except Exception as exc:
        error = str(exc).lower()
        if "quota" in error or "billing" in error or "resource_exhausted" in error:
            print("failure_category = quota/billing error")
        elif "not found" in error or ("model" in error and "not" in error):
            print("failure_category = invalid model")
        else:
            print("failure_category = sdk/api syntax or provider error")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
