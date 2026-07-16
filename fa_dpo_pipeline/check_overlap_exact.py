#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import data_path, normalize_answer, output_path
from experiment_utils import canonical_prediction_rows, load_records, normalize_space, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check exact and normalized overlap between training data and evaluation benchmarks."
    )
    parser.add_argument(
        "--train-file",
        default=data_path("hf_datasets", "MedCaseReasoning", "medcasereasoning_core.csv"),
        help="Training/reference dataset file.",
    )
    parser.add_argument(
        "--eval-file",
        action="append",
        required=True,
        help="Evaluation dataset file. Repeat this flag for multiple benchmarks.",
    )
    parser.add_argument(
        "--output-file",
        default=output_path("data_fairness", "exact_overlap.csv"),
        help="Where to write the overlap CSV.",
    )
    return parser.parse_args()


def canonical_dataset(path: str, label: str) -> List[Dict[str, Any]]:
    rows = canonical_prediction_rows(load_records(path), model_id=label)
    return rows


def main() -> None:
    args = parse_args()
    train_rows = canonical_dataset(args.train_file, "train")

    exact_lookup: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    norm_lookup: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in train_rows:
        exact_key = (normalize_space(row.get("prompt")), normalize_space(row.get("final_answer") or row.get("gold_answer")))
        norm_key = (normalize_answer(row.get("prompt")), normalize_answer(row.get("final_answer") or row.get("gold_answer")))
        exact_lookup.setdefault(exact_key, []).append(row)
        norm_lookup.setdefault(norm_key, []).append(row)

    overlaps: List[Dict[str, Any]] = []
    for eval_path in args.eval_file:
        label = Path(eval_path).stem
        eval_rows = canonical_dataset(eval_path, label)
        for row in eval_rows:
            exact_key = (normalize_space(row.get("prompt")), normalize_space(row.get("final_answer") or row.get("gold_answer")))
            norm_key = (normalize_answer(row.get("prompt")), normalize_answer(row.get("final_answer") or row.get("gold_answer")))
            exact_matches = exact_lookup.get(exact_key, [])
            norm_matches = norm_lookup.get(norm_key, [])
            if not exact_matches and not norm_matches:
                continue
            match_pool = exact_matches or norm_matches
            for train_row in match_pool:
                overlaps.append(
                    {
                        "eval_dataset": label,
                        "eval_id": row.get("id"),
                        "train_id": train_row.get("id"),
                        "question_exact": bool(exact_matches),
                        "question_answer_normalized": bool(norm_matches),
                        "eval_prompt": row.get("prompt"),
                        "train_prompt": train_row.get("prompt"),
                        "eval_answer": row.get("final_answer") or row.get("gold_answer"),
                        "train_answer": train_row.get("final_answer") or train_row.get("gold_answer"),
                    }
                )

    write_csv(args.output_file, overlaps)
    print(f"Overlap rows: {len(overlaps)}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
