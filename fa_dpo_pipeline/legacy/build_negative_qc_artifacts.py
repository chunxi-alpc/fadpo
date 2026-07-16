#!/usr/bin/env python3
"""
Build strict-clean negatives and reviewer-facing sample artifacts.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fa_dpo_pipeline.common import result_path

HIGH_SIMILARITY_THRESHOLD = 0.97
LOW_TOPIC_OVERLAP_THRESHOLD = 0.10

STRICT_DROP_TAGS = {
    "cross_label_reuse_exact",
    "empty_modified_spans_f4",
}

RISK_WEIGHTS = {
    "cross_label_reuse_exact": 5,
    "empty_modified_spans_f4": 5,
    "low_topic_overlap": 3,
    "very_high_similarity": 2,
    "zero_novelty_non_f1": 1,
}

REVIEW_BUCKET_TARGETS = (
    ("cross_label_reuse_exact", 12),
    ("empty_modified_spans_f4", 3),
    ("low_topic_overlap", 10),
    ("very_high_similarity", 12),
    ("zero_novelty_non_f1", 6),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a conservative strict-clean negative set and a stratified "
            "human-review sample from an accepted negatives JSONL file."
        )
    )
    parser.add_argument(
        "--accepted-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.jsonl"),
        help="Path to the accepted negatives JSONL file.",
    )
    parser.add_argument(
        "--strict-output",
        default="",
        help="Output path for the strict-clean JSONL. Defaults beside --accepted-file.",
    )
    parser.add_argument(
        "--review-jsonl-output",
        default="",
        help="Output path for the review sample JSONL. Defaults beside --accepted-file.",
    )
    parser.add_argument(
        "--review-csv-output",
        default="",
        help="Output path for the review sample CSV. Defaults beside --accepted-file.",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Output path for the summary JSON. Defaults beside --accepted-file.",
    )
    parser.add_argument(
        "--review-per-type",
        type=int,
        default=100,
        help="Number of review samples to draw for each error type.",
    )
    parser.add_argument(
        "--double-annotate-per-type",
        type=int,
        default=25,
        help="How many sampled rows per error type should be marked for double annotation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    return parser.parse_args()


def derive_output_path(input_path: Path, suffix: str, explicit_path: str) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return input_path.with_name(f"{input_path.stem}{suffix}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def risk_tags_for_row(
    row: Dict[str, Any],
    cross_label_reuse_ids: Sequence[str],
) -> List[str]:
    tags: List[str] = []
    heuristics = row.get("heuristic_qc") or {}
    if row.get("id") in cross_label_reuse_ids:
        tags.append("cross_label_reuse_exact")
    if row.get("error_symbol") == "F4" and len(row.get("modified_spans") or []) == 0:
        tags.append("empty_modified_spans_f4")
    if float(heuristics.get("similarity", 0.0)) > HIGH_SIMILARITY_THRESHOLD:
        tags.append("very_high_similarity")
    if float(heuristics.get("topic_overlap", 1.0)) < LOW_TOPIC_OVERLAP_THRESHOLD:
        tags.append("low_topic_overlap")
    if row.get("error_symbol") != "F1" and float(heuristics.get("novelty_ratio", -1.0)) == 0.0:
        tags.append("zero_novelty_non_f1")
    return tags


def primary_bucket(tags: Sequence[str]) -> str:
    for bucket in (
        "cross_label_reuse_exact",
        "empty_modified_spans_f4",
        "low_topic_overlap",
        "very_high_similarity",
        "zero_novelty_non_f1",
    ):
        if bucket in tags:
            return bucket
    return "clean_random"


def compute_cross_label_reuse_ids(rows: Sequence[Dict[str, Any]]) -> List[str]:
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source_id"))].append(row)

    risky_ids: List[str] = []
    for items in by_source.values():
        reason_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in items:
            reason_to_rows[str(row.get("negative_reasoning", ""))].append(row)
        for duplicates in reason_to_rows.values():
            labels = {row.get("error_symbol") for row in duplicates}
            if len(duplicates) >= 2 and len(labels) >= 2:
                risky_ids.extend(str(row.get("id")) for row in duplicates)
    return risky_ids


def annotate_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Counter]:
    cross_label_reuse_ids = set(compute_cross_label_reuse_ids(rows))
    risk_counter: Counter = Counter()
    annotated: List[Dict[str, Any]] = []
    for row in rows:
        row_copy = copy.deepcopy(row)
        tags = risk_tags_for_row(row_copy, cross_label_reuse_ids)
        for tag in tags:
            risk_counter[tag] += 1
        row_copy["_risk_tags"] = tags
        row_copy["_risk_score"] = sum(RISK_WEIGHTS.get(tag, 0) for tag in tags)
        row_copy["_sampling_bucket"] = primary_bucket(tags)
        annotated.append(row_copy)
    return annotated, risk_counter


def split_strict_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    strict_rows: List[Dict[str, Any]] = []
    dropped_rows: List[Dict[str, Any]] = []
    for row in rows:
        tags = set(row.get("_risk_tags") or [])
        if tags & STRICT_DROP_TAGS:
            dropped_rows.append(row)
        else:
            strict_rows.append(row)
    return strict_rows, dropped_rows


def prepare_pool(
    rows: Sequence[Dict[str, Any]],
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    risky = [row for row in rows if row.get("_risk_score", 0) > 0]
    clean = [row for row in rows if row.get("_risk_score", 0) == 0]
    rng.shuffle(risky)
    rng.shuffle(clean)
    risky.sort(key=lambda row: (row.get("_risk_score", 0), str(row.get("id"))), reverse=True)
    clean.sort(key=lambda row: str(row.get("id")))
    return risky, clean


def select_review_rows(
    rows: Sequence[Dict[str, Any]],
    per_type: int,
    double_annotate_per_type: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_error: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_error[str(row.get("error_symbol"))].append(row)

    selected: List[Dict[str, Any]] = []
    for error_symbol in sorted(by_error):
        pool = by_error[error_symbol]
        bucket_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in pool:
            bucket_to_rows[str(row.get("_sampling_bucket"))].append(row)

        for bucket_rows in bucket_to_rows.values():
            rng.shuffle(bucket_rows)
            bucket_rows.sort(key=lambda row: str(row.get("id")))

        picked: List[Dict[str, Any]] = []
        picked_ids = set()
        for bucket, target in REVIEW_BUCKET_TARGETS:
            if bucket == "empty_modified_spans_f4" and error_symbol != "F4":
                continue
            candidates = bucket_to_rows.get(bucket, [])
            for row in candidates:
                if len(picked) >= per_type or len([p for p in picked if p.get("_sampling_bucket") == bucket]) >= target:
                    break
                row_id = str(row.get("id"))
                if row_id in picked_ids:
                    continue
                picked.append(row)
                picked_ids.add(row_id)

        fill_order = (
            "clean_random",
            "very_high_similarity",
            "zero_novelty_non_f1",
            "low_topic_overlap",
            "cross_label_reuse_exact",
            "empty_modified_spans_f4",
        )
        for bucket in fill_order:
            if len(picked) >= per_type:
                break
            for row in bucket_to_rows.get(bucket, []):
                if len(picked) >= per_type:
                    break
                row_id = str(row.get("id"))
                if row_id in picked_ids:
                    continue
                picked.append(row)
                picked_ids.add(row_id)

        picked = picked[:per_type]
        picked.sort(
            key=lambda row: (
                str(row.get("error_symbol")),
                -int(row.get("_risk_score", 0)),
                str(row.get("id")),
            )
        )

        for index, row in enumerate(picked):
            review_row = copy.deepcopy(row)
            review_row["review_assignment"] = (
                "double_annotate"
                if index < double_annotate_per_type
                else "single_annotate"
            )
            review_row["review_round"] = "round1"
            selected.append(review_row)
    return selected


def review_jsonl_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    for row in rows:
        review_row = copy.deepcopy(row)
        review_row["reviewer1_target_type_matched"] = None
        review_row["reviewer1_single_dominant_error"] = None
        review_row["reviewer1_answer_preserved"] = None
        review_row["reviewer1_clinical_plausibility"] = None
        review_row["reviewer1_faithfulness_drop"] = None
        review_row["reviewer1_notes"] = ""
        review_row["reviewer2_target_type_matched"] = None
        review_row["reviewer2_single_dominant_error"] = None
        review_row["reviewer2_answer_preserved"] = None
        review_row["reviewer2_clinical_plausibility"] = None
        review_row["reviewer2_faithfulness_drop"] = None
        review_row["reviewer2_notes"] = ""
        output_rows.append(review_row)
    return output_rows


def csv_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened_rows: List[Dict[str, Any]] = []
    for row in rows:
        heuristics = row.get("heuristic_qc") or {}
        flattened_rows.append(
            {
                "id": row.get("id"),
                "source_id": row.get("source_id"),
                "error_symbol": row.get("error_symbol"),
                "error_type": row.get("error_type"),
                "review_assignment": row.get("review_assignment"),
                "sampling_bucket": row.get("_sampling_bucket"),
                "risk_tags": json.dumps(row.get("_risk_tags") or [], ensure_ascii=False),
                "risk_score": row.get("_risk_score"),
                "expected_answer_preserved": row.get("expected_answer_preserved"),
                "answer_preserved": row.get("answer_preserved"),
                "heuristic_similarity": heuristics.get("similarity"),
                "heuristic_length_ratio": heuristics.get("length_ratio"),
                "heuristic_novelty_ratio": heuristics.get("novelty_ratio"),
                "heuristic_topic_overlap": heuristics.get("topic_overlap"),
                "patient_context": row.get("patient_context", ""),
                "final_answer": row.get("final_answer", ""),
                "negative_final_answer": row.get("negative_final_answer", ""),
                "reasoning": row.get("reasoning", ""),
                "negative_reasoning": row.get("negative_reasoning", ""),
                "modified_spans": json.dumps(row.get("modified_spans") or [], ensure_ascii=False),
                "edit_summary": json.dumps(row.get("edit_summary") or [], ensure_ascii=False),
                "reviewer1_target_type_matched": "",
                "reviewer1_single_dominant_error": "",
                "reviewer1_answer_preserved": "",
                "reviewer1_clinical_plausibility": "",
                "reviewer1_faithfulness_drop": "",
                "reviewer1_notes": "",
                "reviewer2_target_type_matched": "",
                "reviewer2_single_dominant_error": "",
                "reviewer2_answer_preserved": "",
                "reviewer2_clinical_plausibility": "",
                "reviewer2_faithfulness_drop": "",
                "reviewer2_notes": "",
            }
        )
    return flattened_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_dict(
    accepted_rows: Sequence[Dict[str, Any]],
    strict_rows: Sequence[Dict[str, Any]],
    dropped_rows: Sequence[Dict[str, Any]],
    review_rows: Sequence[Dict[str, Any]],
    risk_counter: Counter,
) -> Dict[str, Any]:
    accepted_by_error = Counter(row.get("error_symbol") for row in accepted_rows)
    strict_by_error = Counter(row.get("error_symbol") for row in strict_rows)
    dropped_by_bucket = Counter(row.get("_sampling_bucket") for row in dropped_rows)
    review_by_error = Counter(row.get("error_symbol") for row in review_rows)
    review_by_assignment = Counter(row.get("review_assignment") for row in review_rows)
    review_by_bucket = Counter(row.get("_sampling_bucket") for row in review_rows)
    return {
        "accepted_total": len(accepted_rows),
        "strict_clean_total": len(strict_rows),
        "strict_clean_removed": len(dropped_rows),
        "strict_clean_removed_pct": round(len(dropped_rows) / max(len(accepted_rows), 1) * 100, 2),
        "accepted_unique_sources": len({row.get("source_id") for row in accepted_rows}),
        "strict_clean_unique_sources": len({row.get("source_id") for row in strict_rows}),
        "accepted_by_error": dict(sorted(accepted_by_error.items())),
        "strict_clean_by_error": dict(sorted(strict_by_error.items())),
        "risk_tag_counts": dict(sorted(risk_counter.items())),
        "strict_drop_buckets": dict(sorted(dropped_by_bucket.items())),
        "review_sample_total": len(review_rows),
        "review_sample_by_error": dict(sorted(review_by_error.items())),
        "review_sample_by_assignment": dict(sorted(review_by_assignment.items())),
        "review_sample_by_bucket": dict(sorted(review_by_bucket.items())),
    }


def strip_internal_fields(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned_rows: List[Dict[str, Any]] = []
    for row in rows:
        row_copy = copy.deepcopy(row)
        for key in list(row_copy.keys()):
            if key.startswith("_"):
                row_copy.pop(key, None)
        cleaned_rows.append(row_copy)
    return cleaned_rows


def main() -> None:
    args = parse_args()
    accepted_path = Path(args.accepted_file)
    strict_output = derive_output_path(accepted_path, ".strict_clean.jsonl", args.strict_output)
    review_jsonl_output = derive_output_path(
        accepted_path,
        ".review_sample.jsonl",
        args.review_jsonl_output,
    )
    review_csv_output = derive_output_path(
        accepted_path,
        ".review_sample.csv",
        args.review_csv_output,
    )
    summary_output = derive_output_path(
        accepted_path,
        ".review_summary.json",
        args.summary_output,
    )

    accepted_rows = load_jsonl(accepted_path)
    annotated_rows, risk_counter = annotate_rows(accepted_rows)
    strict_rows, dropped_rows = split_strict_rows(annotated_rows)
    review_rows = select_review_rows(
        annotated_rows,
        per_type=args.review_per_type,
        double_annotate_per_type=args.double_annotate_per_type,
        seed=args.seed,
    )

    strict_export_rows = strip_internal_fields(strict_rows)
    review_export_rows = review_jsonl_rows(review_rows)
    review_csv_rows = csv_rows(review_rows)
    summary = summary_dict(
        accepted_rows=annotated_rows,
        strict_rows=strict_rows,
        dropped_rows=dropped_rows,
        review_rows=review_rows,
        risk_counter=risk_counter,
    )

    write_jsonl(strict_output, strict_export_rows)
    write_jsonl(review_jsonl_output, review_export_rows)
    write_csv(review_csv_output, review_csv_rows)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Accepted rows: {len(accepted_rows)}")
    print(f"Strict-clean rows: {len(strict_rows)} -> {strict_output}")
    print(f"Review sample rows: {len(review_rows)} -> {review_jsonl_output}")
    print(f"Review CSV: {review_csv_output}")
    print(f"Summary: {summary_output}")


if __name__ == "__main__":
    main()
