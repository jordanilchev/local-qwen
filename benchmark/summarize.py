#!/usr/bin/env python3
"""Build fair comparison tables from a single benchmark session_id."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from benchmark._lib import CODING_PROMPTS


def _avg_median_decode(results: list[dict]) -> float:
    by_prompt = {r["prompt"]: r["decode_tps_median"] for r in results}
    vals = [by_prompt[pname] for pname, _ in CODING_PROMPTS if pname in by_prompt]
    return statistics.mean(vals) if vals else 0.0


def _avg_median_ttft(results: list[dict]) -> float:
    by_prompt = {r["prompt"]: r["ttft_ms_median"] for r in results}
    vals = [by_prompt[pname] for pname, _ in CODING_PROMPTS if pname in by_prompt]
    return statistics.mean(vals) if vals else 0.0


def load_session_results(session_id: str, results_dir: Path) -> list[dict]:
    matched: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("session_"):
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("session_id") == session_id:
            matched.append(data)
    return matched


def format_table(payloads: list[dict], *, baseline_method: str = "ollama") -> str:
    by_family: dict[str, list[dict]] = {}
    for p in payloads:
        fid = p.get("family_id") or "unknown"
        by_family.setdefault(fid, []).append(p)

    lines: list[str] = []
    for family_id, entries in sorted(by_family.items()):
        baseline_decode = None
        family_baseline = baseline_method
        for e in entries:
            if e["method"] == family_baseline:
                baseline_decode = _avg_median_decode(e["results"])
                break
        if baseline_decode is None:
            for e in entries:
                if e["method"] == "plain-mlx":
                    baseline_decode = _avg_median_decode(e["results"])
                    family_baseline = "plain-mlx"
                    break

        lines.append(f"\n### {family_id}")
        lines.append("")
        lines.append("| Method | Avg tok/s | TTFT (ms) | vs baseline | Parity |")
        lines.append("|--------|----------:|----------:|------------:|--------|")
        for e in sorted(entries, key=lambda x: -_avg_median_decode(x["results"])):
            decode = _avg_median_decode(e["results"])
            ttft = _avg_median_ttft(e["results"])
            vs = decode / baseline_decode if baseline_decode and baseline_decode > 0 else 1.0
            parity_flags = []
            for r in e["results"]:
                if r.get("parity_match") is False:
                    parity_flags.append(r["prompt"])
            parity = "✓" if not parity_flags else f"✗ {','.join(parity_flags)}"
            if e["method"] == family_baseline:
                parity = "—"
            lines.append(
                f"| {e['method']} | {decode:.1f} | {ttft:.0f} | {vs:.2f}× | {parity} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark session JSONs")
    parser.add_argument("--session-id", required=True, help="session_id shared by all methods")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    args = parser.parse_args()

    payloads = load_session_results(args.session_id, args.results_dir)
    if not payloads:
        raise SystemExit(f"No JSON results with session_id={args.session_id!r}")

    print(f"Session {args.session_id}: {len(payloads)} method files")
    print(format_table(payloads))


if __name__ == "__main__":
    main()
