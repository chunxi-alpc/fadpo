#!/usr/bin/env python3
"""
Merge a base negative pool with a validated delta pool and emit refreshed stats.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fa_dpo_pipeline.common import data_path, result_path


RAW_POSITIVE_COUNT = 13092


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a base negative JSONL pool with a delta JSONL pool after "
            "checking ID collisions, and write refreshed summary statistics."
        )
    )
    parser.add_argument(
        "--base-file",
        default=data_path("result", "qwen-next", "medcase_unfaithful_negatives.strict_clean.jsonl"),
        help="Base negative pool JSONL.",
    )
    parser.add_argument(
        "--delta-file",
        default=data_path(
            "result",
            "qwen-next",
            "medcase_unfaithful_negatives.f1_rerun.rescued_unresolved.strict_clean.jsonl",
        ),
        help="Validated delta JSONL to append to the base pool.",
    )
    parser.add_argument(
        "--output-file",
        default=result_path(
            "qwen-next",
            "medcase_unfaithful_negatives.strict_clean.plus_f1_rescued.jsonl",
        ),
        help="Merged output JSONL.",
    )
    parser.add_argument(
        "--summary-file",
        default=result_path(
            "qwen-next",
            "medcase_unfaithful_negatives.strict_clean.plus_f1_rescued.summary.json",
        ),
        help="Where to write the refreshed summary JSON.",
    )
    parser.add_argument(
        "--raw-positive-count",
        type=int,
        default=RAW_POSITIVE_COUNT,
        help="Raw positive count used to report average negatives per raw positive.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_rows(rows: List[Dict[str, Any]], raw_positive_count: int) -> Dict[str, Any]:
    by_error = Counter(str(row.get("error_symbol")) for row in rows)
    by_answer_policy = Counter(bool(row.get("answer_preserved")) for row in rows)
    unique_sources = {str(row.get("source_id")) for row in rows}
    return {
        "rows": len(rows),
        "unique_sources": len(unique_sources),
        "avg_per_raw_positive": round(len(rows) / max(raw_positive_count, 1), 4),
        "by_error": dict(sorted(by_error.items())),
        "by_answer_preserved": dict(sorted((str(k), v) for k, v in by_answer_policy.items())),
    }


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_file)
    delta_path = Path(args.delta_file)
    output_path = Path(args.output_file)
    summary_path = Path(args.summary_file)

    base_rows = read_jsonl(base_path)
    delta_rows = read_jsonl(delta_path)

    base_ids = {str(row.get("id")) for row in base_rows}
    delta_ids = [str(row.get("id")) for row in delta_rows]
    duplicate_delta_ids = [row_id for row_id, count in Counter(delta_ids).items() if count > 1]
    collided_ids = sorted(base_ids.intersection(delta_ids))

    if duplicate_delta_ids:
        raise ValueError(f"Delta file has duplicate ids: {duplicate_delta_ids[:20]}")
    if collided_ids:
        raise ValueError(f"Delta file collides with base ids: {collided_ids[:20]}")

    merged_rows = list(base_rows) + list(delta_rows)
    write_jsonl(output_path, merged_rows)

    base_summary = summarize_rows(base_rows, args.raw_positive_count)
    delta_summary = summarize_rows(delta_rows, args.raw_positive_count)
    merged_summary = summarize_rows(merged_rows, args.raw_positive_count)

    base_sources = {str(row.get("source_id")) for row in base_rows}
    delta_sources = {str(row.get("source_id")) for row in delta_rows}

    summary = {
        "base_file": str(base_path),
        "delta_file": str(delta_path),
        "output_file": str(output_path),
        "base_summary": base_summary,
        "delta_summary": delta_summary,
        "merged_summary": merged_summary,
        "new_source_ids_introduced": len(delta_sources - base_sources),
        "source_ids_already_present_in_base": len(delta_sources & base_sources),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Base rows: {len(base_rows)}")
    print(f"Delta rows: {len(delta_rows)}")
    print(f"Merged rows: {len(merged_rows)} -> {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
