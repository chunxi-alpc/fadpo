#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from common import append_jsonl, extract_json, iso_utc_now, load_jsonl, normalize_text, output_path, write_json
from experiment_utils import write_csv

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover
    AsyncOpenAI = None


SYSTEM_PROMPT = """You are a strict blind judge for medical reasoning.

You will compare Response A and Response B for the same case.
Ignore style preferences unless they affect clinical reasoning quality.
The final diagnosis is the same for both systems, so focus on process quality.

Return one JSON object only:
{
  "criteria": {
    "hallucination_freedom": {"winner": "A|B|tie", "reason": "..."},
    "logical_coherence": {"winner": "A|B|tie", "reason": "..."},
    "overall_preference": {"winner": "A|B|tie", "reason": "..."}
  },
  "summary": "one-sentence summary"
}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run blind pairwise judging on same-answer reasoning pairs."
    )
    parser.add_argument(
        "--input-file",
        default=output_path("process_metrics", "same_answer_subset.jsonl"),
        help="Pairwise input JSONL, usually the same-answer subset.",
    )
    parser.add_argument(
        "--output-file",
        default=output_path("gpt_judge", "judgments.jsonl"),
        help="Where to append structured judgments JSONL.",
    )
    parser.add_argument(
        "--win-rates-output",
        default=output_path("gpt_judge", "win_rates.csv"),
        help="Where to write aggregated win-rate CSV.",
    )
    parser.add_argument(
        "--prompt-output",
        default=output_path("gpt_judge", "prompt.txt"),
        help="Where to write the judge prompt template.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="Judge model name.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", ""),
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key for the judge model.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge temperature.")
    parser.add_argument("--max-tokens", type=int, default=900, help="Max completion tokens.")
    parser.add_argument("--max-concurrency", type=int, default=8, help="Concurrent judge requests.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for A/B swapping.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Optional sample cap.")
    return parser.parse_args()


def build_user_prompt(row: Dict[str, Any], show_left_as_a: bool) -> str:
    if show_left_as_a:
        response_a = normalize_text(row.get("left_reasoning"))
        response_b = normalize_text(row.get("right_reasoning"))
    else:
        response_a = normalize_text(row.get("right_reasoning"))
        response_b = normalize_text(row.get("left_reasoning"))
    return (
        f"Case:\n{normalize_text(row.get('prompt'))}\n\n"
        f"Reference final diagnosis:\n{normalize_text(row.get('gold_answer'))}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n"
    )


def winner_to_canonical(winner: str, show_left_as_a: bool) -> str:
    winner = normalize_text(winner).lower()
    if winner == "tie":
        return "tie"
    if show_left_as_a:
        return "left" if winner == "a" else "right"
    return "right" if winner == "a" else "left"


async def judge_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    row: Dict[str, Any],
    show_left_as_a: bool,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    async with semaphore:
        response = await client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(row, show_left_as_a)},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    raw_output = response.choices[0].message.content or ""
    payload = extract_json(raw_output)
    criteria = payload.get("criteria") or {}
    normalized = {}
    for key in ("hallucination_freedom", "logical_coherence", "overall_preference"):
        item = criteria.get(key) or {}
        canonical = winner_to_canonical(item.get("winner", "tie"), show_left_as_a)
        normalized[key] = {
            "winner": canonical,
            "presented_winner": normalize_text(item.get("winner")).lower() or "tie",
            "reason": normalize_text(item.get("reason")),
        }
    return {
        "id": row.get("id"),
        "timestamp": iso_utc_now(),
        "left_model_id": row.get("left_model_id"),
        "right_model_id": row.get("right_model_id"),
        "prompt": row.get("prompt"),
        "gold_answer": row.get("gold_answer"),
        "presented_order": "left_as_a" if show_left_as_a else "right_as_a",
        "judge_model_requested": args.model,
        "judge_model_snapshot": response.model,
        "criteria": normalized,
        "summary": normalize_text(payload.get("summary")),
        "raw_output": raw_output,
    }


def aggregate_win_rates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Counter] = defaultdict(Counter)
    meta: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        meta["global"] = {
            "left_model_id": row.get("left_model_id"),
            "right_model_id": row.get("right_model_id"),
        }
        for criterion, item in (row.get("criteria") or {}).items():
            grouped[criterion][str(item.get("winner"))] += 1

    summary = []
    for criterion, counter in sorted(grouped.items()):
        total = sum(counter.values())
        left_wins = counter.get("left", 0)
        right_wins = counter.get("right", 0)
        ties = counter.get("tie", 0)
        summary.append(
            {
                "criterion": criterion,
                "left_model_id": meta.get("global", {}).get("left_model_id", "left_model"),
                "right_model_id": meta.get("global", {}).get("right_model_id", "right_model"),
                "n": total,
                "left_win_rate": round(left_wins / max(total, 1), 6),
                "tie_rate": round(ties / max(total, 1), 6),
                "right_win_rate": round(right_wins / max(total, 1), 6),
            }
        )
    return summary


async def main_async(args: argparse.Namespace) -> None:
    if AsyncOpenAI is None:
        raise RuntimeError("Missing openai dependency. Install openai>=1.30.0 first.")
    if not args.api_key:
        raise RuntimeError("OPENAI_API_KEY or --api-key is required for pairwise judging.")

    rows = load_jsonl(args.input_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    existing_rows = load_jsonl(args.output_file) if Path(args.output_file).exists() else []
    existing_ids = {str(row.get("id")) for row in existing_rows}
    rows = [row for row in rows if str(row.get("id")) not in existing_ids]

    randomizer = random.Random(args.seed)
    order_map = {str(row.get("id")): randomizer.random() < 0.5 for row in rows}

    prompt_text = SYSTEM_PROMPT + "\n\n=== USER PROMPT TEMPLATE ===\n" + build_user_prompt(
        rows[0] if rows else {"prompt": "<case>", "gold_answer": "<answer>", "left_reasoning": "<A>", "right_reasoning": "<B>"},
        True,
    )
    write_json(args.prompt_output.replace(".txt", ".json"), {"system_prompt": SYSTEM_PROMPT})
    os.makedirs(os.path.dirname(args.prompt_output), exist_ok=True)
    with open(args.prompt_output, "w", encoding="utf-8") as handle:
        handle.write(prompt_text)

    client_kwargs = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = AsyncOpenAI(**client_kwargs)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    tasks = [judge_one(client, semaphore, row, order_map[str(row.get("id"))], args) for row in rows]

    results: List[Dict[str, Any]] = []
    for future in asyncio.as_completed(tasks):
        result = await future
        append_jsonl(args.output_file, result)
        results.append(result)

    all_results = existing_rows + results
    write_csv(args.win_rates_output, aggregate_win_rates(all_results))
    print(f"New judgments: {len(results)}")
    print(f"Total judgments: {len(all_results)}")
    print(f"JSONL: {args.output_file}")
    print(f"Win rates: {args.win_rates_output}")
    print(f"Prompt: {args.prompt_output}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
