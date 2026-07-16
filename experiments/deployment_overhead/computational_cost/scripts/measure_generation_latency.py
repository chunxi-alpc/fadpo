#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Template
from openai import OpenAI
from transformers import AutoTokenizer


def load_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def build_prompt(row: Dict[str, Any]) -> str:
    options = row.get("options") or {}
    option_str = "\n".join(f"{label}. {text}" for label, text in options.items())
    return (
        "Please answer the following multiple-choice question. "
        "If exactly one option is correct, conclude with 'The answer is A.'.\n"
        f"{row['question']}\n{option_str}\n"
    )


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure single-generation OpenAI-compatible latency.")
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.eval_file), limit=args.limit + args.warmup)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, padding_side="left")
    if tokenizer.chat_template is None:
        raise RuntimeError(f"Tokenizer has no chat template: {args.tokenizer}")
    template = Template(tokenizer.chat_template)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows):
            prompt = build_prompt(row)
            rendered = template.render(
                messages=[{"role": "user", "content": prompt}],
                bos_token=tokenizer.bos_token,
                add_generation_prompt=True,
            )
            input_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
            start = time.perf_counter()
            response = client.completions.create(
                model=args.model,
                prompt=rendered,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            elapsed = time.perf_counter() - start
            output = response.choices[0].text
            output_tokens = len(tokenizer.encode(output, add_special_tokens=False))
            record = {
                "index": idx,
                "recorded": idx >= args.warmup,
                "latency_seconds": elapsed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "question": row.get("question", ""),
                "answer_idx": row.get("answer_idx") or row.get("label"),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if idx >= args.warmup:
                records.append(record)

    latencies = [float(r["latency_seconds"]) for r in records]
    output_tokens = [int(r["output_tokens"]) for r in records]
    input_tokens = [int(r["input_tokens"]) for r in records]
    summary = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "eval_file": args.eval_file,
        "recorded_requests": len(records),
        "warmup_requests": args.warmup,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "mean_latency_seconds": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_seconds": statistics.median(latencies) if latencies else 0.0,
        "p90_latency_seconds": percentile(latencies, 0.90),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "mean_output_tokens": statistics.mean(output_tokens) if output_tokens else 0.0,
        "p95_output_tokens": percentile(output_tokens, 0.95),
        "mean_input_tokens": statistics.mean(input_tokens) if input_tokens else 0.0,
    }
    write_json(Path(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
