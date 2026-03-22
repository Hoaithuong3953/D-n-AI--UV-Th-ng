#!/usr/bin/env python3
"""
Benchmark IntentDetector against labeled data using the real Gemini API (no LLM mock).

Usage (from project root, .env must contain GEMINI_API_KEY):
    python scripts/benchmark_intent.py
    python scripts/benchmark_intent.py --data data/benchmark_intent.csv

Outputs: accuracy, per-method counts, confusion-style wrong rows, latency stats.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai import GeminiClient, SYSTEM_PROMPT
from config import settings
from domain import Intent, IntentDetectionMethod
from services.support.intent_detector import IntentDetector


def load_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "text" not in reader.fieldnames or "expected" not in reader.fieldnames:
            raise SystemExit("CSV must have columns: text,expected (values: chat|roadmap)")
        for row in reader:
            text = (row.get("text") or "").strip()
            exp = (row.get("expected") or "").strip().lower()
            if not text:
                continue
            if exp not in ("chat", "roadmap"):
                raise SystemExit(f"Invalid expected label: {exp!r} (use chat or roadmap)")
            rows.append((text, exp))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Intent benchmark (real Gemini API)")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "benchmark_intent.csv",
        help="CSV with columns text,expected",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.8,
        help="Seconds between API calls (rate limit friendly)",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"File not found: {args.data}")

    rows = load_rows(args.data)
    if not rows:
        raise SystemExit("No rows to evaluate")

    client = GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model_name=settings.GEMINI_MODEL,
        request_timeout=60,
        stream_timeout=120,
        system_prompt=SYSTEM_PROMPT,
    )
    detector = IntentDetector(llm_client=client)

    correct = 0
    latencies: list[float] = []
    method_keyword = 0
    method_llm = 0
    wrong: list[tuple[int, str, str, str, str]] = []

    for i, (text, expected) in enumerate(rows, start=1):
        want = Intent.ROADMAP if expected == "roadmap" else Intent.CHAT
        t0 = time.perf_counter()
        result = detector.detect(text)
        latencies.append(time.perf_counter() - t0)

        if result.method == IntentDetectionMethod.KEYWORD:
            method_keyword += 1
        else:
            method_llm += 1

        if result.intent == want:
            correct += 1
        else:
            wrong.append(
                (
                    i,
                    text[:80] + ("…" if len(text) > 80 else ""),
                    expected,
                    result.intent.value,
                    result.method.value,
                )
            )

        if args.sleep > 0 and i < len(rows):
            time.sleep(args.sleep)

    n = len(rows)
    acc = correct / n
    print("=== Intent benchmark (real API) ===")
    print(f"Model: {settings.GEMINI_MODEL}")
    print(f"Samples: {n}")
    print(f"Accuracy: {correct}/{n} = {acc:.2%}")
    print(f"Keyword path: {method_keyword} ({method_keyword/n:.1%})")
    print(f"LLM fallback path: {method_llm} ({method_llm/n:.1%})")
    if latencies:
        print(f"Latency mean: {statistics.mean(latencies)*1000:.0f} ms")
        print(f"Latency median: {statistics.median(latencies)*1000:.0f} ms")
        if len(latencies) >= 2:
            print(f"Latency stdev: {statistics.stdev(latencies)*1000:.0f} ms")
    if wrong:
        print(f"\nWrong ({len(wrong)}):")
        for idx, snippet, exp, got, meth in wrong:
            print(f"  #{idx} expected={exp} got={got} method={meth} | {snippet!r}")


if __name__ == "__main__":
    main()
