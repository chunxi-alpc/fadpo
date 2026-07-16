#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import httpx

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import append_jsonl, ensure_parent, load_jsonl, normalize_text, read_json
else:
    from .common import append_jsonl, ensure_parent, load_jsonl, normalize_text, read_json

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover
    AsyncOpenAI = None


SYSTEM_PROMPT = """You are a careful medical reasoning assistant.

Read the clinical case carefully, reason step by step, and end with a clearly marked final diagnosis.
Do not invent findings that are not supported by the case.
Use this exact final line format:
Final Answer: <diagnosis>
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visible reasoning traces for case-based medical evaluation data."
    )
    parser.add_argument("--input-file", required=True, help="Normalized JSONL/JSON case file.")
    parser.add_argument(
        "--output-file",
        default="outputs/f1_f4_natural_error/medcase_generations.jsonl",
        help="Where to append generation results.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", ""),
        help="OpenAI-compatible model name.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API key.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds for the OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--request-max-retries",
        type=int,
        default=2,
        help="How many times to retry a failed request before surfacing an error.",
    )
    parser.add_argument(
        "--request-retry-backoff",
        type=float,
        default=2.0,
        help="Base backoff in seconds between request retries.",
    )
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_records(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(file_path)
    if suffix == ".json":
        payload = read_json(file_path)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "records", "items", "examples"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            return [payload]
    raise ValueError(f"Unsupported input format: {file_path.suffix}")


def completed_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    return {normalize_text(row.get("id") or row.get("source_id")) for row in load_jsonl(path)}


def build_prompt(row: Dict[str, Any]) -> str:
    case_text = normalize_text(row.get("patient_context") or row.get("prompt") or row.get("case_prompt"))
    article_text = normalize_text(row.get("article_text"))
    if article_text:
        return f"Clinical Case:\n{case_text}\n\nOptional Source Text:\n{article_text}\n"
    return f"Clinical Case:\n{case_text}\n"


def strip_markdown_artifacts(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"</s>\s*$", "", cleaned).strip()
    cleaned = re.sub(r"^\s*```(?:json|text|markdown)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


def parse_output(text: str) -> Dict[str, str]:
    cleaned = strip_markdown_artifacts(text)
    answer = ""
    reasoning = cleaned
    matches = list(re.finditer(r"(?is)(?:^|\n)\s*Final Answer\s*:\s*(.+?)\s*$", cleaned))
    if matches:
        answer = normalize_text(matches[-1].group(1))
        reasoning = cleaned[: matches[-1].start()].strip()
    return {
        "output": cleaned,
        "predicted_reasoning": reasoning,
        "predicted_answer_text": answer,
    }


async def create_completion_with_retries(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    messages: List[Dict[str, str]],
):
    max_retries = max(int(args.request_max_retries), 0)
    base_backoff = max(float(args.request_retry_backoff), 0.0)
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:  # pragma: no cover - backend behavior is environment-specific
            last_exc = exc
            if attempt >= max_retries:
                raise
            if base_backoff > 0:
                await asyncio.sleep(base_backoff * (2**attempt))
    raise RuntimeError(f"Completion failed without an exception: {last_exc}")


async def generate_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    row: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(row)},
    ]
    async with semaphore:
        try:
            response = await create_completion_with_retries(client, args, messages)
        except Exception as exc:
            row_id = normalize_text(row.get("id") or row.get("source_id")) or "<unknown>"
            raise RuntimeError(f"Generation failed for `{row_id}`: {exc}") from exc
    text = response.choices[0].message.content or ""
    parsed = parse_output(text)
    output_row = dict(row)
    output_row.update(parsed)
    output_row["model_id"] = args.model
    output_row["generation_model_snapshot"] = response.model
    return output_row


async def main_async(args: argparse.Namespace) -> None:
    if AsyncOpenAI is None:
        raise RuntimeError("Missing openai dependency. Install openai>=1.30.0 first.")
    if not args.model:
        raise RuntimeError("--model or OPENAI_MODEL is required.")

    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    rows = load_records(args.input_file)
    if args.start > 0 or args.end > 0:
        end = None if args.end < 0 else args.end
        rows = rows[args.start:end]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    seen = completed_ids(args.output_file) if not args.overwrite else set()
    rows = [row for row in rows if normalize_text(row.get("id") or row.get("source_id")) not in seen]

    client_kwargs = {
        "api_key": args.api_key,
        "max_retries": 0,
        "timeout": args.request_timeout,
    }
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    http_client = httpx.AsyncClient(
        trust_env=False,
        timeout=httpx.Timeout(args.request_timeout),
    )
    client_kwargs["http_client"] = http_client
    client = AsyncOpenAI(**client_kwargs)
    semaphore = asyncio.Semaphore(args.max_concurrency)

    tasks = [generate_one(client, semaphore, row, args) for row in rows]
    written = 0
    failures: List[str] = []
    ensure_parent(args.output_file)
    try:
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
            except Exception as exc:
                failures.append(str(exc))
                continue
            append_jsonl(args.output_file, result)
            written += 1
    finally:
        await http_client.aclose()

    if failures:
        raise RuntimeError(
            f"{len(failures)} generation request(s) failed for {args.output_file}. First error: {failures[0]}"
        )

    print(f"Wrote {written} rows to {args.output_file}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
