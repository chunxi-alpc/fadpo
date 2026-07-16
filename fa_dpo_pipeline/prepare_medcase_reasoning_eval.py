#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import ensure_parent, normalize_text, write_jsonl
else:
    from .common import ensure_parent, normalize_text, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MedCaseReasoning CSV into a normalized JSONL evaluation file "
            "for F1-F4 natural-error analysis and generation."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="hf_datasets/MedCaseReasoning/medcasereasoning_core.csv",
        help="Path to medcasereasoning_core.csv.",
    )
    parser.add_argument(
        "--output-file",
        default="result/f1_f4_natural_error/medcase_test.jsonl",
        help="Normalized JSONL output path.",
    )
    parser.add_argument(
        "--splits",
        default="test",
        help="Comma-separated splits to keep, e.g. `val,test`.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Optional cap after split filtering.",
    )
    parser.add_argument(
        "--include-article-text",
        action="store_true",
        help="Keep full article text as `article_text` and `evidence`.",
    )
    return parser.parse_args()


def parse_csv_fields(raw: str) -> List[str]:
    return [part.strip() for part in normalize_text(raw).split(",") if part.strip()]


def load_rows(path: str) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_row(row: Dict[str, Any], row_index: int, include_article_text: bool) -> Dict[str, Any]:
    source_id = normalize_text(row.get("")) or normalize_text(row.get("pmcid")) or str(row_index)
    article_text = normalize_text(row.get("text"))
    patient_context = normalize_text(row.get("case_prompt"))
    reasoning = normalize_text(row.get("diagnostic_reasoning"))
    answer = normalize_text(row.get("final_diagnosis"))
    output = f"{reasoning}\n\nFinal Answer: {answer}" if reasoning and answer else reasoning or answer

    normalized = {
        "id": source_id,
        "source_id": source_id,
        "dataset_name": "MedCaseReasoning",
        "split": normalize_text(row.get("split")),
        "pmcid": normalize_text(row.get("pmcid")),
        "title": normalize_text(row.get("title")),
        "journal": normalize_text(row.get("journal")),
        "article_link": normalize_text(row.get("article_link")),
        "publication_date": normalize_text(row.get("publication_date")),
        "patient_context": patient_context,
        "prompt": patient_context,
        "reference_reasoning": reasoning,
        "reasoning": reasoning,
        "reference_answer": answer,
        "answer": answer,
        "output": output,
    }

    if include_article_text:
        normalized["article_text"] = article_text
        normalized["evidence"] = article_text
    else:
        normalized["article_text"] = article_text
        normalized["evidence"] = ""

    return normalized


def filter_rows(rows: Iterable[Dict[str, Any]], splits: List[str]) -> List[Dict[str, Any]]:
    allowed = {split.lower() for split in splits}
    return [row for row in rows if normalize_text(row.get("split")).lower() in allowed]


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    splits = parse_csv_fields(args.splits)
    if splits:
        rows = filter_rows(rows, splits)
    normalized_rows = [
        normalize_row(row=row, row_index=index, include_article_text=args.include_article_text)
        for index, row in enumerate(rows)
    ]
    if args.max_samples > 0:
        normalized_rows = normalized_rows[: args.max_samples]

    output_path = ensure_parent(args.output_file)
    write_jsonl(output_path, normalized_rows)
    print(f"Wrote {len(normalized_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
