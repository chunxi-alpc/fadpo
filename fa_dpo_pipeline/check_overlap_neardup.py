#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import data_path, output_path
from experiment_utils import (
    canonical_prediction_rows,
    load_records,
    text_match_score,
    write_csv,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve near-duplicate train/eval question pairs for manual fairness review."
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
        help="Evaluation dataset file. Repeat for multiple benchmarks.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many nearest train questions to keep per eval question.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Minimum lexical similarity to export a candidate.",
    )
    parser.add_argument(
        "--candidates-output",
        default=output_path("data_fairness", "near_duplicate_candidates.jsonl"),
        help="Where to write candidate pairs JSONL.",
    )
    parser.add_argument(
        "--summary-output",
        default=output_path("data_fairness", "near_duplicate_summary.csv"),
        help="Where to write summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = canonical_prediction_rows(load_records(args.train_file), model_id="train")
    train_prompts = [row.get("prompt", "") for row in train_rows]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    train_matrix = vectorizer.fit_transform(train_prompts)

    all_candidates: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for eval_path in args.eval_file:
        label = Path(eval_path).stem
        eval_rows = canonical_prediction_rows(load_records(eval_path), model_id=label)
        eval_matrix = vectorizer.transform([row.get("prompt", "") for row in eval_rows])
        similarity = cosine_similarity(eval_matrix, train_matrix)
        dataset_candidates = 0

        for eval_index, eval_row in enumerate(eval_rows):
            row_scores = similarity[eval_index]
            if row_scores.size == 0:
                continue
            top_indices = np.argsort(row_scores)[::-1][: args.top_k]
            for train_index in top_indices:
                score = float(row_scores[train_index])
                if score < args.similarity_threshold:
                    continue
                train_row = train_rows[int(train_index)]
                answer_score = text_match_score(
                    eval_row.get("final_answer") or eval_row.get("gold_answer"),
                    train_row.get("final_answer") or train_row.get("gold_answer"),
                )
                all_candidates.append(
                    {
                        "eval_dataset": label,
                        "eval_id": eval_row.get("id"),
                        "train_id": train_row.get("id"),
                        "question_similarity": round(score, 6),
                        "answer_similarity": round(answer_score, 6),
                        "eval_prompt": eval_row.get("prompt"),
                        "train_prompt": train_row.get("prompt"),
                        "eval_answer": eval_row.get("final_answer") or eval_row.get("gold_answer"),
                        "train_answer": train_row.get("final_answer") or train_row.get("gold_answer"),
                    }
                )
                dataset_candidates += 1

        summary_rows.append(
            {
                "eval_dataset": label,
                "eval_questions": len(eval_rows),
                "candidate_pairs": dataset_candidates,
                "similarity_threshold": args.similarity_threshold,
                "top_k": args.top_k,
            }
        )

    write_jsonl(args.candidates_output, all_candidates)
    write_csv(args.summary_output, summary_rows)
    print(f"Candidate pairs: {len(all_candidates)}")
    print(f"Candidates: {args.candidates_output}")
    print(f"Summary: {args.summary_output}")


if __name__ == "__main__":
    main()
