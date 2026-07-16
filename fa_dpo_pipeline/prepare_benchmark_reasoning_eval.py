#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import ensure_parent, load_jsonl, normalize_answer, normalize_text, read_json, write_jsonl
else:
    from .common import ensure_parent, load_jsonl, normalize_answer, normalize_text, read_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize external medical MCQ benchmarks into a shared JSONL format for F1-F4 study."
    )
    parser.add_argument("--input-file", required=True, help="Raw benchmark file: json/jsonl/csv.")
    parser.add_argument(
        "--output-file",
        required=True,
        help="Normalized JSONL output path.",
    )
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Optional dataset name override. Defaults to benchmark field or input stem.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--max-samples", type=int, default=-1)
    return parser.parse_args()


def flatten_json_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [row if isinstance(row, dict) else {"value": row} for row in payload]
    if isinstance(payload, dict):
        for key in ("data", "records", "items", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                rows: List[Dict[str, Any]] = []
                for row in value:
                    rows.append(row if isinstance(row, dict) else {"value": row})
                return rows
        if payload and all(isinstance(v, list) for v in payload.values()):
            rows = []
            for source, items in payload.items():
                for row in items:
                    item = row if isinstance(row, dict) else {"value": row}
                    if "source" not in item:
                        item = dict(item)
                        item["source"] = source
                    rows.append(item)
            return rows
        return [payload]
    raise ValueError("Unsupported JSON structure: expected list or dict.")


def load_records(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(file_path)
    if suffix == ".json":
        return flatten_json_payload(read_json(file_path))
    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported input format: {file_path.suffix}")


def normalize_options(options: Any) -> Dict[str, str]:
    if isinstance(options, Mapping):
        normalized: Dict[str, str] = {}
        for key, value in options.items():
            text = normalize_text(value)
            if text:
                normalized[str(key).strip().upper()] = text
        return normalized
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        normalized = {}
        for idx, value in enumerate(options):
            text = normalize_text(value)
            if text:
                normalized[chr(65 + idx)] = text
        return normalized
    if isinstance(options, str):
        stripped = options.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                return normalize_options(parsed)
    return {}


def detect_gold_choice_label(options: Mapping[str, str], answer_text: str, row: Dict[str, Any]) -> str:
    explicit = normalize_text(row.get("answer_idx") or row.get("label")).upper()
    if explicit:
        return explicit
    normalized_answer = normalize_answer(answer_text)
    if not normalized_answer:
        return ""
    for label, option_text in options.items():
        if normalize_answer(option_text) == normalized_answer:
            return str(label).upper()
    return ""


def build_example_id(
    row: Dict[str, Any],
    dataset_name: str,
    subset: str,
    source: str,
    index: int,
) -> str:
    for field in ("id", "example_id", "qid", "uid"):
        value = normalize_text(row.get(field))
        if value:
            return value
    parts = [dataset_name or "dataset"]
    if subset:
        parts.append(subset)
    elif source:
        parts.append(source)
    parts.append(str(index))
    return "__".join(part.replace(" ", "_") for part in parts)


def normalize_row(row: Dict[str, Any], index: int, input_file: str, dataset_override: str) -> Dict[str, Any]:
    input_stem = Path(input_file).stem
    source = normalize_text(row.get("source") or row.get("source_dataset"))
    subset = normalize_text(row.get("subset") or row.get("meta_info") or source)
    dataset_name = normalize_text(dataset_override or row.get("benchmark") or row.get("dataset_name") or input_stem)
    question = normalize_text(row.get("question") or row.get("prompt") or row.get("input_str"))
    options = normalize_options(row.get("options"))
    answer_text = normalize_text(row.get("reference_answer") or row.get("answer"))
    answer_label = detect_gold_choice_label(options, answer_text, row)
    example_id = build_example_id(row, dataset_name, subset, source, index)
    source_id = normalize_text(row.get("id") or row.get("example_id"))

    output_row = dict(row)
    output_row.update(
        {
            "id": example_id,
            "source_id": source_id or example_id,
            "dataset_name": dataset_name,
            "benchmark": normalize_text(row.get("benchmark")) or dataset_name,
            "source": source or dataset_name,
            "subset": subset,
            "question": question,
            "prompt": question,
            "options": options,
            "answer": answer_text,
            "reference_answer": answer_text,
            "answer_idx": answer_label,
            "reference_answer_label": answer_label,
            "preferred_analysis_profile": "mcq_f4",
        }
    )
    return output_row


def main() -> None:
    args = parse_args()
    rows = load_records(args.input_file)
    if args.start > 0 or args.end > 0:
        end = None if args.end < 0 else args.end
        rows = rows[args.start:end]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    normalized_rows = [
        normalize_row(row=row, index=index, input_file=args.input_file, dataset_override=args.dataset_name)
        for index, row in enumerate(rows)
    ]
    ensure_parent(args.output_file)
    write_jsonl(args.output_file, normalized_rows)
    print(f"Wrote {len(normalized_rows)} rows to {args.output_file}")


if __name__ == "__main__":
    main()
