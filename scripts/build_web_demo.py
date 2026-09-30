"""Build static browser fixtures from the preserved offline Python harness.

Run from the repository root: python scripts/build_web_demo.py
No provider APIs, keys or live model calls are used.
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crisiseval.simulation import list_scenarios, list_treatments, run_model_comparison


def main():
    scenarios = [json.loads(p.read_text()) for p in list_scenarios()]
    treatments = [json.loads(p.read_text()) for p in list_treatments()]
    cases = {}
    for sp in list_scenarios():
        for tp in list_treatments():
            results = run_model_comparison(sp, tp, turns=3, top_k=5)
            cases[f"{sp.stem}/{tp.stem}"] = {
                label: {
                    "decisions": [d.model_dump() for d in result.decisions],
                    "summary": result.summary_metrics,
                    "risks": result.failure_modes,
                    "sources": result.source_cards_used,
                }
                for label, result in results.items()
            }
    output = ROOT / "docs" / "data.json.gz"
    output.parent.mkdir(exist_ok=True)
    raw = json.dumps({
        "mode": "Precomputed deterministic MockModel outputs",
        "turns": 3,
        "top_k": 5,
        "scenarios": scenarios,
        "treatments": treatments,
        "cases": cases,
    }, separators=(",", ":")).encode()
    output.write_bytes(gzip.compress(raw, mtime=0))
    print(f"Built {len(cases)} scenario/treatment cases: {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
