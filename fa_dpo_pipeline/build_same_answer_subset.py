#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict

from common import normalize_answer, normalize_text, output_path, write_jsonl
from experiment_utils import canonical_prediction_rows, load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the same-answer subset where two systems both answer correctly."
    )
    parser.add_argument("--left-file", required=True, help="Prediction file for model/system A.")
    parser.add_argument("--right-file", required=True, help="Prediction file for model/system B.")
    parser.add_argument("--gold-file", default="", help="Optional gold/reference file.")
    parser.add_argument("--left-model-id", default="fa_dpo", help="Label for the left system.")
    parser.add_argument("--right-model-id", default="standard_dpo", help="Label for the right system.")
    parser.add_argument("--left-id-field", default="", help="Optional explicit id field for the left file.")
    parser.add_argument("--right-id-field", default="", help="Optional explicit id field for the right file.")
    parser.add_argument("--gold-id-field", default="", help="Optional explicit id field for the gold file.")
    parser.add_argument("--left-prompt-field", default="", help="Optional explicit prompt field for the left file.")
    parser.add_argument("--right-prompt-field", default="", help="Optional explicit prompt field for the right file.")
    parser.add_argument("--gold-prompt-field", default="", help="Optional explicit prompt field for the gold file.")
    parser.add_argument("--left-reasoning-field", default="", help="Optional explicit reasoning field for the left file.")
    parser.add_argument("--right-reasoning-field", default="", help="Optional explicit reasoning field for the right file.")
    parser.add_argument("--left-answer-field", default="", help="Optional explicit answer field for the left file.")
    parser.add_argument("--right-answer-field", default="", help="Optional explicit answer field for the right file.")
    parser.add_argument("--gold-answer-field", default="", help="Optional explicit gold-answer field.")
    parser.add_argument(
        "--min-reasoning-chars",
        type=int,
        default=0,
        help="Optional complexity filter: require each reasoning chain to be at least this long.",
    )
    parser.add_argument(
        "--output-file",
        default=output_path("process_metrics", "same_answer_subset.jsonl"),
        help="Where to write the canonical same-answer subset JSONL.",
    )
    return parser.parse_args()


def build_gold_lookup(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    if not args.gold_file:
        return {}
    rows = load_records(args.gold_file)
    gold_rows = canonical_prediction_rows(
        rows,
        model_id="gold",
        id_field=args.gold_id_field or None,
        prompt_field=args.gold_prompt_field or None,
        answer_field=args.gold_answer_field or None,
        gold_answer_field=args.gold_answer_field or None,
    )
    return {row["id"]: row for row in gold_rows if row["id"]}


def main() -> None:
    args = parse_args()
    left_rows = canonical_prediction_rows(
        load_records(args.left_file),
        model_id=args.left_model_id,
        id_field=args.left_id_field or None,
        prompt_field=args.left_prompt_field or None,
        reasoning_field=args.left_reasoning_field or None,
        answer_field=args.left_answer_field or None,
    )
    right_rows = canonical_prediction_rows(
        load_records(args.right_file),
        model_id=args.right_model_id,
        id_field=args.right_id_field or None,
        prompt_field=args.right_prompt_field or None,
        reasoning_field=args.right_reasoning_field or None,
        answer_field=args.right_answer_field or None,
    )
    left_lookup = {row["id"]: row for row in left_rows if row["id"]}
    right_lookup = {row["id"]: row for row in right_rows if row["id"]}
    gold_lookup = build_gold_lookup(args)

    subset = []
    for row_id in sorted(set(left_lookup) & set(right_lookup)):
        left = left_lookup[row_id]
        right = right_lookup[row_id]
        gold_answer = (
            gold_lookup.get(row_id, {}).get("final_answer")
            or gold_lookup.get(row_id, {}).get("gold_answer")
            or left.get("gold_answer")
            or right.get("gold_answer")
        )
        if not gold_answer:
            continue
        if len(left.get("reasoning", "")) < args.min_reasoning_chars:
            continue
        if len(right.get("reasoning", "")) < args.min_reasoning_chars:
            continue
        left_correct = normalize_answer(left.get("final_answer")) == normalize_answer(gold_answer)
        right_correct = normalize_answer(right.get("final_answer")) == normalize_answer(gold_answer)
        if not (left_correct and right_correct):
            continue
        subset.append(
            {
                "id": row_id,
                "prompt": gold_lookup.get(row_id, {}).get("prompt") or left.get("prompt") or right.get("prompt"),
                "gold_answer": gold_answer,
                "left_model_id": args.left_model_id,
                "right_model_id": args.right_model_id,
                "left_reasoning": left.get("reasoning"),
                "left_final_answer": left.get("final_answer"),
                "right_reasoning": right.get("reasoning"),
                "right_final_answer": right.get("final_answer"),
                "left_correct": left_correct,
                "right_correct": right_correct,
                "left_reasoning_chars": len(normalize_text(left.get("reasoning"))),
                "right_reasoning_chars": len(normalize_text(right.get("reasoning"))),
            }
        )

    write_jsonl(args.output_file, subset)
    print(f"Subset rows: {len(subset)}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
