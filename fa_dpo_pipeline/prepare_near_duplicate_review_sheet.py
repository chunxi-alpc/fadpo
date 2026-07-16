#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict, List

from common import output_path
from experiment_utils import load_records, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert near-duplicate JSONL candidates into a reviewer-friendly CSV sheet."
    )
    parser.add_argument(
        "--input-file",
        default=output_path("data_fairness", "near_duplicate_candidates.jsonl"),
        help="Near-duplicate candidate JSONL from check_overlap_neardup.py.",
    )
    parser.add_argument(
        "--output-file",
        default=output_path("data_fairness", "near_duplicate_review.csv"),
        help="Reviewer-facing CSV to fill manually.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_records(args.input_file)
    review_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        review_rows.append(
            {
                "review_id": f"nd-{index:04d}",
                "eval_dataset": row.get("eval_dataset", ""),
                "eval_id": row.get("eval_id", ""),
                "train_id": row.get("train_id", ""),
                "question_similarity": row.get("question_similarity", ""),
                "answer_similarity": row.get("answer_similarity", ""),
                "eval_prompt": row.get("eval_prompt", ""),
                "train_prompt": row.get("train_prompt", ""),
                "eval_answer": row.get("eval_answer", ""),
                "train_answer": row.get("train_answer", ""),
                "review_decision": "",
                "confidence": "",
                "notes": "",
            }
        )
    write_csv(
        args.output_file,
        review_rows,
        fieldnames=[
            "review_id",
            "eval_dataset",
            "eval_id",
            "train_id",
            "question_similarity",
            "answer_similarity",
            "eval_prompt",
            "train_prompt",
            "eval_answer",
            "train_answer",
            "review_decision",
            "confidence",
            "notes",
        ],
    )
    print(f"Review rows: {len(review_rows)}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
