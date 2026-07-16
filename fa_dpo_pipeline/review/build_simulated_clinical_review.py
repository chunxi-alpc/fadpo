#!/usr/bin/env python3
"""
Build a simulated "returned from clinicians" review file from the stage-2 review CSV.

This is meant to show what a completed medical review artifact should look like
once reviewer fields, adjudication fields, and curation decisions are all filled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fa_dpo_pipeline.common import data_path, result_path

DECISION_QUOTAS = {
    "accept": 305,
    "minor_revision": 57,
    "reject": 38,
}

PLAUSIBILITY_DISTS = {
    "accept": {5: 154, 4: 151},
    "minor_revision": {4: 13, 3: 44},
    "reject": {2: 36, 1: 2},
}

FAITHFULNESS_DISTS = {
    "accept": {5: 39, 4: 266},
    "minor_revision": {4: 5, 3: 52},
    "reject": {2: 23, 1: 15},
}

BUCKET_WEIGHTS = {
    "clean_random": 4.0,
    "zero_novelty_non_f1": 3.2,
    "very_high_similarity": 2.4,
    "low_topic_overlap": 1.7,
    "cross_label_reuse_exact": -0.6,
    "empty_modified_spans_f4": -1.4,
}

ERROR_WEIGHTS = {
    "F1": 0.75,
    "F2": 0.25,
    "F3": 0.45,
    "F4": -0.35,
}

DOUBLE_ANNOTATE_FIELDS = (
    "target_type_matched",
    "single_dominant_error",
    "answer_preserved",
    "clinical_plausibility",
    "faithfulness_drop",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a completed clinician-returned review file from the stage-2 "
            "medical review sample CSV."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=data_path("result", "qwen-next-stage2", "medcase_unfaithful_negatives.review_sample.csv"),
        help="Outgoing review CSV that was sent to clinicians.",
    )
    parser.add_argument(
        "--output-csv",
        default=result_path("qwen-next-stage2", "medcase_unfaithful_negatives.review_sample.annotated.simulated.csv"),
        help="Completed review CSV output path.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=result_path("qwen-next-stage2", "medcase_unfaithful_negatives.review_sample.annotated.simulated.jsonl"),
        help="Machine-readable completed review JSONL output path.",
    )
    parser.add_argument(
        "--output-summary",
        default=result_path("qwen-next-stage2", "medcase_unfaithful_negatives.review_sample.annotated.simulated.summary.json"),
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--review-batch-id",
        default="qwen-next-stage2-round1-simulated",
        help="Review batch identifier to stamp into every row.",
    )
    parser.add_argument(
        "--review-date",
        default="2026-03-28",
        help="Review completion date to stamp into every row (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for reviewer-2 disagreement simulation.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def stable_unit_float(value: str) -> float:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def parse_bool(value: Any) -> bool | None:
    if value in ("", None):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def parse_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    return int(value)


def parse_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def parse_json_text(value: str) -> Any:
    if value in ("", None):
        return []
    return json.loads(value)


def quality_score(row: Dict[str, str]) -> float:
    bucket = row.get("sampling_bucket", "")
    error_symbol = row.get("error_symbol", "")
    similarity = parse_float(row.get("heuristic_similarity")) or 0.0
    topic_overlap = parse_float(row.get("heuristic_topic_overlap")) or 0.0
    novelty_ratio = parse_float(row.get("heuristic_novelty_ratio")) or 0.0
    risk_score = parse_int(row.get("risk_score")) or 0
    expected_answer_preserved = 1.0 if parse_bool(row.get("expected_answer_preserved")) else 0.0
    return (
        BUCKET_WEIGHTS.get(bucket, 0.0)
        + ERROR_WEIGHTS.get(error_symbol, 0.0)
        + topic_overlap * 1.5
        + novelty_ratio * 2.0
        + expected_answer_preserved * 0.2
        - max(similarity - 0.92, 0.0) * 6.5
        - risk_score * 0.18
        + stable_unit_float(str(row.get("id", ""))) * 0.01
    )


def decision_sort_key(row: Dict[str, Any]) -> tuple[float, str]:
    return (float(row["_quality_score"]), str(row["id"]))


def set_ranked_decisions(rows: List[Dict[str, Any]]) -> None:
    ranked = sorted(rows, key=decision_sort_key, reverse=True)
    cursor = 0
    for decision in ("accept", "minor_revision", "reject"):
        for row in ranked[cursor : cursor + DECISION_QUOTAS[decision]]:
            row["adjudicated_final_decision"] = decision
        cursor += DECISION_QUOTAS[decision]


def assign_boolean_fields(rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["adjudicated_final_decision"]].append(row)
    for items in grouped.values():
        items.sort(key=decision_sort_key, reverse=True)

    for row in grouped["accept"]:
        row["adjudicated_target_type_matched"] = True
        row["adjudicated_single_dominant_error"] = True
        row["adjudicated_answer_preserved"] = True

    for index, row in enumerate(grouped["minor_revision"]):
        row["adjudicated_target_type_matched"] = index < 29
        row["adjudicated_single_dominant_error"] = index < 18
        row["adjudicated_answer_preserved"] = True

    for index, row in enumerate(grouped["reject"]):
        row["adjudicated_target_type_matched"] = False
        row["adjudicated_single_dominant_error"] = False
        row["adjudicated_answer_preserved"] = index < 16


def assign_ranked_scores(rows: List[Dict[str, Any]], field: str, dists: Dict[str, Dict[int, int]]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["adjudicated_final_decision"]].append(row)
    for items in grouped.values():
        items.sort(key=decision_sort_key, reverse=True)

    for decision, score_counts in dists.items():
        items = grouped[decision]
        cursor = 0
        for score in sorted(score_counts, reverse=True):
            count = score_counts[score]
            for row in items[cursor : cursor + count]:
                row[field] = score
            cursor += count


def reviewer1_note(row: Dict[str, Any]) -> str:
    decision = row["adjudicated_final_decision"]
    bucket = row["sampling_bucket"]
    if decision == "accept":
        if bucket == "clean_random":
            return ""
        if bucket == "very_high_similarity":
            return "Target error is valid, but the rewrite remains close to the original reasoning."
        if bucket == "zero_novelty_non_f1":
            return "Acceptable overall; novelty is limited but the intended error remains identifiable."
        return "Accept after confirming the intended error stays localized and clinically readable."
    if decision == "minor_revision":
        if bucket == "very_high_similarity":
            return "Main error is visible, but the rewrite is too close to the source reasoning and should be sharpened."
        if bucket == "low_topic_overlap":
            return "Error direction is partly correct, but the rewrite drifts slightly away from the central case evidence."
        return "Usable after a small rewrite to make the dominant error cleaner and more isolated."
    if bucket == "cross_label_reuse_exact":
        return "Reject because the reasoning appears cross-label reusable and does not support a trustworthy target-type judgment."
    if bucket == "empty_modified_spans_f4":
        return "Reject because the F4 edit is too weak to establish a real reasoning-answer mismatch."
    if bucket == "low_topic_overlap":
        return "Reject because the rewritten reasoning drifts away from the clinical evidence and target error is unclear."
    return "Reject because the intended error is not realized cleanly enough for dataset release."


def adjudicated_note(row: Dict[str, Any]) -> str:
    decision = row["adjudicated_final_decision"]
    bucket = row["sampling_bucket"]
    if decision == "accept":
        if bucket == "clean_random":
            return "Consensus: retain as-is."
        return "Consensus: retain after confirming the heuristic risk tag does not invalidate the sample."
    if decision == "minor_revision":
        return "Consensus: keep only after minor wording cleanup to isolate the dominant error."
    if bucket == "cross_label_reuse_exact":
        return "Consensus: drop from release because cross-label reuse undermines error-type control."
    if bucket == "empty_modified_spans_f4":
        return "Consensus: drop from release because the intended F4 perturbation is not materially realized."
    return "Consensus: drop from release because quality issues are too large for minor revision."


def curation_action_for(decision: str) -> str:
    if decision == "accept":
        return "keep"
    if decision == "minor_revision":
        return "revise_then_keep"
    return "drop"


def rubric_decision(
    target_type_matched: bool,
    single_dominant_error: bool,
    answer_preserved: bool,
    clinical_plausibility: int,
    faithfulness_drop: int,
) -> str:
    if (
        target_type_matched
        and single_dominant_error
        and answer_preserved
        and clinical_plausibility >= 4
        and faithfulness_drop >= 3
    ):
        return "accept"
    if (
        not target_type_matched
        and not single_dominant_error
        and (clinical_plausibility <= 2 or faithfulness_drop <= 2)
    ) or not answer_preserved:
        return "reject"
    return "minor_revision"


def maybe_flip_boolean(
    rng: random.Random,
    key: str,
    value: bool,
    base_agree: float,
) -> bool:
    threshold = base_agree + stable_unit_float(key) * 0.04 - 0.02
    return value if rng.random() <= threshold else (not value)


def noisy_score(
    rng: random.Random,
    key: str,
    value: int,
    minimum: int = 1,
    maximum: int = 5,
) -> int:
    pivot = stable_unit_float(key)
    if pivot < 0.12:
        delta = -1
    elif pivot > 0.88:
        delta = 1
    else:
        delta = 0
    if rng.random() < 0.82:
        delta = 0
    return max(minimum, min(maximum, value + delta))


def reviewer2_note(row: Dict[str, Any], decision: str) -> str:
    if decision == row["adjudicated_final_decision"]:
        if decision == "accept":
            return ""
        if decision == "minor_revision":
            return "Would keep after a light wording cleanup."
        return "Not suitable for release in the current form."
    if decision == "accept":
        return "I am slightly more permissive; the target error still looks clinically readable."
    if decision == "minor_revision":
        return "Borderline sample; I would revise instead of making a final keep/drop decision."
    return "I am less confident that the target error is controlled well enough for release."


def simulate_reviewer_fields(rows: List[Dict[str, Any]], seed: int) -> None:
    rng = random.Random(seed)
    for row in rows:
        row["reviewer1_target_type_matched"] = row["adjudicated_target_type_matched"]
        row["reviewer1_single_dominant_error"] = row["adjudicated_single_dominant_error"]
        row["reviewer1_answer_preserved"] = row["adjudicated_answer_preserved"]
        row["reviewer1_clinical_plausibility"] = row["adjudicated_clinical_plausibility"]
        row["reviewer1_faithfulness_drop"] = row["adjudicated_faithfulness_drop"]
        row["reviewer1_final_decision"] = row["adjudicated_final_decision"]
        row["reviewer1_notes"] = reviewer1_note(row)

        if row["review_assignment"] != "double_annotate":
            row["reviewer2_target_type_matched"] = ""
            row["reviewer2_single_dominant_error"] = ""
            row["reviewer2_answer_preserved"] = ""
            row["reviewer2_clinical_plausibility"] = ""
            row["reviewer2_faithfulness_drop"] = ""
            row["reviewer2_final_decision"] = ""
            row["reviewer2_notes"] = ""
            row["adjudication_required"] = False
            continue

        bucket = row["sampling_bucket"]
        agreement_penalty = {
            "clean_random": 0.0,
            "zero_novelty_non_f1": 0.02,
            "very_high_similarity": 0.05,
            "low_topic_overlap": 0.08,
            "cross_label_reuse_exact": 0.10,
            "empty_modified_spans_f4": 0.12,
        }.get(bucket, 0.04)

        reviewer2_target = maybe_flip_boolean(
            rng,
            f"{row['id']}::r2::target",
            row["adjudicated_target_type_matched"],
            0.97 - agreement_penalty / 2.0,
        )
        reviewer2_single = maybe_flip_boolean(
            rng,
            f"{row['id']}::r2::single",
            row["adjudicated_single_dominant_error"],
            0.95 - agreement_penalty / 2.0,
        )
        reviewer2_answer = maybe_flip_boolean(
            rng,
            f"{row['id']}::r2::answer",
            row["adjudicated_answer_preserved"],
            0.95 - agreement_penalty / 2.0,
        )
        reviewer2_plausibility = noisy_score(
            rng,
            f"{row['id']}::r2::plausibility",
            row["adjudicated_clinical_plausibility"],
        )
        reviewer2_faithfulness = noisy_score(
            rng,
            f"{row['id']}::r2::faithfulness",
            row["adjudicated_faithfulness_drop"],
        )
        reviewer2_decision = rubric_decision(
            target_type_matched=reviewer2_target,
            single_dominant_error=reviewer2_single,
            answer_preserved=reviewer2_answer,
            clinical_plausibility=reviewer2_plausibility,
            faithfulness_drop=reviewer2_faithfulness,
        )

        row["reviewer2_target_type_matched"] = reviewer2_target
        row["reviewer2_single_dominant_error"] = reviewer2_single
        row["reviewer2_answer_preserved"] = reviewer2_answer
        row["reviewer2_clinical_plausibility"] = reviewer2_plausibility
        row["reviewer2_faithfulness_drop"] = reviewer2_faithfulness
        row["reviewer2_final_decision"] = reviewer2_decision
        row["reviewer2_notes"] = reviewer2_note(row, reviewer2_decision)

        row["adjudication_required"] = any(
            (
                row["reviewer1_target_type_matched"] != row["reviewer2_target_type_matched"],
                row["reviewer1_single_dominant_error"] != row["reviewer2_single_dominant_error"],
                row["reviewer1_answer_preserved"] != row["reviewer2_answer_preserved"],
                row["reviewer1_final_decision"] != row["reviewer2_final_decision"],
            )
        )


def enrich_rows(rows: List[Dict[str, str]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        new_row: Dict[str, Any] = dict(row)
        new_row["_quality_score"] = quality_score(row)
        new_row["review_batch_id"] = args.review_batch_id
        new_row["review_round"] = "round1"
        new_row["annotation_status"] = "simulated_complete_example"
        new_row["review_completed"] = True
        new_row["review_completed_at"] = args.review_date
        enriched.append(new_row)

    set_ranked_decisions(enriched)
    assign_boolean_fields(enriched)
    assign_ranked_scores(enriched, "adjudicated_clinical_plausibility", PLAUSIBILITY_DISTS)
    assign_ranked_scores(enriched, "adjudicated_faithfulness_drop", FAITHFULNESS_DISTS)

    for row in enriched:
        row["adjudicated_notes"] = adjudicated_note(row)
        row["curation_action"] = curation_action_for(row["adjudicated_final_decision"])

    simulate_reviewer_fields(enriched, seed=args.seed)
    return enriched


def output_fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    base_fields = [
        "review_batch_id",
        "review_round",
        "annotation_status",
        "review_completed",
        "review_completed_at",
        "id",
        "source_id",
        "error_symbol",
        "error_type",
        "review_assignment",
        "sampling_bucket",
        "risk_tags",
        "risk_score",
        "expected_answer_preserved",
        "answer_preserved",
        "heuristic_similarity",
        "heuristic_length_ratio",
        "heuristic_novelty_ratio",
        "heuristic_topic_overlap",
        "patient_context",
        "final_answer",
        "negative_final_answer",
        "reasoning",
        "negative_reasoning",
        "modified_spans",
        "edit_summary",
        "reviewer1_target_type_matched",
        "reviewer1_single_dominant_error",
        "reviewer1_answer_preserved",
        "reviewer1_clinical_plausibility",
        "reviewer1_faithfulness_drop",
        "reviewer1_final_decision",
        "reviewer1_notes",
        "reviewer2_target_type_matched",
        "reviewer2_single_dominant_error",
        "reviewer2_answer_preserved",
        "reviewer2_clinical_plausibility",
        "reviewer2_faithfulness_drop",
        "reviewer2_final_decision",
        "reviewer2_notes",
        "adjudication_required",
        "adjudicated_target_type_matched",
        "adjudicated_single_dominant_error",
        "adjudicated_answer_preserved",
        "adjudicated_clinical_plausibility",
        "adjudicated_faithfulness_drop",
        "adjudicated_final_decision",
        "adjudicated_notes",
        "curation_action",
    ]
    extra_fields = [
        key
        for key in rows[0].keys()
        if key not in set(base_fields) and not key.startswith("_")
    ]
    return base_fields + extra_fields


def row_for_csv(row: Dict[str, Any], fieldnames: Sequence[str]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for field in fieldnames:
        value = row.get(field, "")
        if isinstance(value, list):
            output[field] = json.dumps(value, ensure_ascii=False)
        else:
            output[field] = value
    return output


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = output_fieldnames(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_for_csv(row, fieldnames))


def jsonl_row(row: Dict[str, Any]) -> Dict[str, Any]:
    converted = {
        key: value
        for key, value in row.items()
        if not key.startswith("_")
    }
    converted["risk_score"] = parse_int(converted.get("risk_score"))
    converted["expected_answer_preserved"] = parse_bool(converted.get("expected_answer_preserved"))
    converted["answer_preserved"] = parse_bool(converted.get("answer_preserved"))
    converted["heuristic_similarity"] = parse_float(converted.get("heuristic_similarity"))
    converted["heuristic_length_ratio"] = parse_float(converted.get("heuristic_length_ratio"))
    converted["heuristic_novelty_ratio"] = parse_float(converted.get("heuristic_novelty_ratio"))
    converted["heuristic_topic_overlap"] = parse_float(converted.get("heuristic_topic_overlap"))
    converted["risk_tags"] = parse_json_text(converted.get("risk_tags"))
    converted["modified_spans"] = parse_json_text(converted.get("modified_spans"))
    converted["edit_summary"] = parse_json_text(converted.get("edit_summary"))

    bool_fields = [
        "review_completed",
        "adjudication_required",
        "reviewer1_target_type_matched",
        "reviewer1_single_dominant_error",
        "reviewer1_answer_preserved",
        "reviewer2_target_type_matched",
        "reviewer2_single_dominant_error",
        "reviewer2_answer_preserved",
        "adjudicated_target_type_matched",
        "adjudicated_single_dominant_error",
        "adjudicated_answer_preserved",
    ]
    int_fields = [
        "reviewer1_clinical_plausibility",
        "reviewer1_faithfulness_drop",
        "reviewer2_clinical_plausibility",
        "reviewer2_faithfulness_drop",
        "adjudicated_clinical_plausibility",
        "adjudicated_faithfulness_drop",
    ]
    for field in bool_fields:
        converted[field] = parse_bool(converted.get(field))
    for field in int_fields:
        converted[field] = parse_int(converted.get(field))
    return converted


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonl_row(row), ensure_ascii=False) + "\n")


def cohen_kappa(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    if not left or len(left) != len(right):
        return None
    n_items = len(left)
    labels = sorted(set(left) | set(right))
    observed = sum(1 for l_value, r_value in zip(left, right) if l_value == r_value) / n_items
    expected = 0.0
    left_counter = Counter(left)
    right_counter = Counter(right)
    for label in labels:
        expected += (left_counter[label] / n_items) * (right_counter[label] / n_items)
    if expected >= 1.0:
        return 1.0
    return round((observed - expected) / (1.0 - expected), 3)


def summarize(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    decision_counts = Counter(row["adjudicated_final_decision"] for row in rows)
    action_counts = Counter(row["curation_action"] for row in rows)
    by_error: Dict[str, Dict[str, Any]] = {}
    by_bucket: Dict[str, Dict[str, Any]] = {}

    for field_name, container_key in (
        ("error_symbol", by_error),
        ("sampling_bucket", by_bucket),
    ):
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[field_name])].append(row)
        for group_name, items in grouped.items():
            container_key[group_name] = {
                "rows": len(items),
                "decision_counts": dict(sorted(Counter(row["adjudicated_final_decision"] for row in items).items())),
            }

    double_rows = [row for row in rows if row["review_assignment"] == "double_annotate"]
    target_left = [row["reviewer1_target_type_matched"] for row in double_rows]
    target_right = [row["reviewer2_target_type_matched"] for row in double_rows]
    single_left = [row["reviewer1_single_dominant_error"] for row in double_rows]
    single_right = [row["reviewer2_single_dominant_error"] for row in double_rows]
    answer_left = [row["reviewer1_answer_preserved"] for row in double_rows]
    answer_right = [row["reviewer2_answer_preserved"] for row in double_rows]
    decision_left = [row["reviewer1_final_decision"] for row in double_rows]
    decision_right = [row["reviewer2_final_decision"] for row in double_rows]

    summary = {
        "annotation_status": "simulated_complete_example",
        "source_review_file": str(Path(args.input_csv)),
        "review_batch_id": args.review_batch_id,
        "review_completed_at": args.review_date,
        "rows": len(rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "curation_action_counts": dict(sorted(action_counts.items())),
        "adjudicated_metrics": {
            "target_type_matched_true": sum(1 for row in rows if row["adjudicated_target_type_matched"]),
            "single_dominant_error_true": sum(1 for row in rows if row["adjudicated_single_dominant_error"]),
            "answer_preserved_true": sum(1 for row in rows if row["adjudicated_answer_preserved"]),
            "mean_clinical_plausibility": round(
                sum(int(row["adjudicated_clinical_plausibility"]) for row in rows) / max(len(rows), 1),
                3,
            ),
            "mean_faithfulness_drop": round(
                sum(int(row["adjudicated_faithfulness_drop"]) for row in rows) / max(len(rows), 1),
                3,
            ),
        },
        "double_annotate_rows": len(double_rows),
        "agreement": {
            "target_type_matched_kappa": cohen_kappa(target_left, target_right),
            "single_dominant_error_kappa": cohen_kappa(single_left, single_right),
            "answer_preserved_kappa": cohen_kappa(answer_left, answer_right),
            "final_decision_kappa": cohen_kappa(decision_left, decision_right),
        },
        "by_error_symbol": dict(sorted(by_error.items())),
        "by_sampling_bucket": dict(sorted(by_bucket.items())),
        "notes": [
            "This is a simulated completed-review artifact built to illustrate the expected returned-file schema.",
            "Real clinician annotations should overwrite this artifact and remove the simulated status label.",
        ],
    }
    return summary


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_jsonl = Path(args.output_jsonl)
    output_summary = Path(args.output_summary)

    rows = load_csv(input_csv)
    if len(rows) != 400:
        raise ValueError(f"Expected 400 review rows, found {len(rows)}")

    enriched_rows = enrich_rows(rows, args)
    write_csv(output_csv, enriched_rows)
    write_jsonl(output_jsonl, enriched_rows)
    summary = summarize(enriched_rows, args)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Completed review CSV: {output_csv}")
    print(f"Completed review JSONL: {output_jsonl}")
    print(f"Completed review summary: {output_summary}")
    print(json.dumps(summary["decision_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
