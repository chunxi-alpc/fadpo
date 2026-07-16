#!/usr/bin/env python3
"""
Finalize a clinician-returned review CSV into machine-readable JSONL and summary
artifacts.

Expected usage:
1. Start from `medcase_unfaithful_negatives.review_sample.csv`
2. Let clinicians fill `reviewer1_*` / `reviewer2_*`
3. Optionally fill adjudication columns for disputed double-annotated rows
4. Run this script to produce:
   - `*.annotated.jsonl`
   - `*.annotated.summary.json`

If reviewer-level final decisions are absent, they are inferred from the rubric
defined in the review packet.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fa_dpo_pipeline.common import data_path, result_path


REVIEWER_METRIC_FIELDS = (
    "target_type_matched",
    "single_dominant_error",
    "answer_preserved",
    "clinical_plausibility",
    "faithfulness_drop",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize a clinician-returned review CSV into annotated JSONL and "
            "summary files."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=data_path("result", "qwen-next-stage2", "medcase_unfaithful_negatives.review_sample.annotated.csv"),
        help="Clinician-returned review CSV.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=result_path("qwen-next-stage2", "medcase_unfaithful_negatives.review_sample.annotated.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--output-summary",
        default=result_path("qwen-next-stage2", "medcase_unfaithful_negatives.review_sample.annotated.summary.json"),
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--completed-at",
        default=str(date.today()),
        help="Completion date to stamp into completed rows (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--review-batch-id",
        default="",
        help="Optional batch ID override. If omitted, keep existing value or derive one.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def curation_action_for(decision: str | None) -> str:
    if decision == "accept":
        return "keep"
    if decision == "minor_revision":
        return "revise_then_keep"
    if decision == "reject":
        return "drop"
    return ""


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


def average_int(a: int, b: int) -> int:
    return int((a + b) / 2.0 + 0.5)


def reviewer_prefix(index: int) -> str:
    return f"reviewer{index}_"


def reviewer_payload(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    prefix = reviewer_prefix(index)
    payload = {
        "target_type_matched": parse_bool(row.get(f"{prefix}target_type_matched")),
        "single_dominant_error": parse_bool(row.get(f"{prefix}single_dominant_error")),
        "answer_preserved": parse_bool(row.get(f"{prefix}answer_preserved")),
        "clinical_plausibility": parse_int(row.get(f"{prefix}clinical_plausibility")),
        "faithfulness_drop": parse_int(row.get(f"{prefix}faithfulness_drop")),
        "final_decision": (row.get(f"{prefix}final_decision") or "").strip(),
        "notes": row.get(f"{prefix}notes", "") or "",
    }
    return payload


def reviewer_complete(payload: Dict[str, Any]) -> bool:
    return all(payload[field] is not None for field in REVIEWER_METRIC_FIELDS)


def ensure_reviewer_decision(row: Dict[str, Any], index: int) -> None:
    payload = reviewer_payload(row, index)
    if not reviewer_complete(payload):
        return
    field = f"reviewer{index}_final_decision"
    if (row.get(field) or "").strip():
        return
    row[field] = rubric_decision(
        target_type_matched=bool(payload["target_type_matched"]),
        single_dominant_error=bool(payload["single_dominant_error"]),
        answer_preserved=bool(payload["answer_preserved"]),
        clinical_plausibility=int(payload["clinical_plausibility"]),
        faithfulness_drop=int(payload["faithfulness_drop"]),
    )


def adjudicated_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target_type_matched": parse_bool(row.get("adjudicated_target_type_matched")),
        "single_dominant_error": parse_bool(row.get("adjudicated_single_dominant_error")),
        "answer_preserved": parse_bool(row.get("adjudicated_answer_preserved")),
        "clinical_plausibility": parse_int(row.get("adjudicated_clinical_plausibility")),
        "faithfulness_drop": parse_int(row.get("adjudicated_faithfulness_drop")),
        "final_decision": (row.get("adjudicated_final_decision") or "").strip(),
        "notes": row.get("adjudicated_notes", "") or "",
    }


def adjudicated_complete(payload: Dict[str, Any]) -> bool:
    return (
        all(payload[field] is not None for field in REVIEWER_METRIC_FIELDS)
        and bool(payload["final_decision"])
    )


def disagreement_requires_adjudication(row: Dict[str, Any]) -> bool:
    reviewer1 = reviewer_payload(row, 1)
    reviewer2 = reviewer_payload(row, 2)
    if not reviewer_complete(reviewer1) or not reviewer_complete(reviewer2):
        return False
    if reviewer1["target_type_matched"] != reviewer2["target_type_matched"]:
        return True
    if reviewer1["single_dominant_error"] != reviewer2["single_dominant_error"]:
        return True
    if reviewer1["answer_preserved"] != reviewer2["answer_preserved"]:
        return True
    if reviewer1["final_decision"] != reviewer2["final_decision"]:
        return True
    if abs(int(reviewer1["clinical_plausibility"]) - int(reviewer2["clinical_plausibility"])) > 1:
        return True
    if abs(int(reviewer1["faithfulness_drop"]) - int(reviewer2["faithfulness_drop"])) > 1:
        return True
    return False


def fill_from_single_reviewer(row: Dict[str, Any], reviewer_index: int) -> None:
    reviewer = reviewer_payload(row, reviewer_index)
    row["adjudicated_target_type_matched"] = reviewer["target_type_matched"]
    row["adjudicated_single_dominant_error"] = reviewer["single_dominant_error"]
    row["adjudicated_answer_preserved"] = reviewer["answer_preserved"]
    row["adjudicated_clinical_plausibility"] = reviewer["clinical_plausibility"]
    row["adjudicated_faithfulness_drop"] = reviewer["faithfulness_drop"]
    row["adjudicated_final_decision"] = reviewer["final_decision"]
    if not (row.get("adjudicated_notes") or "").strip():
        row["adjudicated_notes"] = reviewer["notes"]


def fill_from_matching_double_review(row: Dict[str, Any]) -> None:
    reviewer1 = reviewer_payload(row, 1)
    reviewer2 = reviewer_payload(row, 2)
    row["adjudicated_target_type_matched"] = reviewer1["target_type_matched"]
    row["adjudicated_single_dominant_error"] = reviewer1["single_dominant_error"]
    row["adjudicated_answer_preserved"] = reviewer1["answer_preserved"]
    row["adjudicated_clinical_plausibility"] = average_int(
        int(reviewer1["clinical_plausibility"]),
        int(reviewer2["clinical_plausibility"]),
    )
    row["adjudicated_faithfulness_drop"] = average_int(
        int(reviewer1["faithfulness_drop"]),
        int(reviewer2["faithfulness_drop"]),
    )
    row["adjudicated_final_decision"] = reviewer1["final_decision"]
    if not (row.get("adjudicated_notes") or "").strip():
        row["adjudicated_notes"] = ""


def normalize_row(row: Dict[str, str], args: argparse.Namespace) -> Dict[str, Any]:
    normalized: Dict[str, Any] = dict(row)
    if args.review_batch_id:
        normalized["review_batch_id"] = args.review_batch_id
    elif not (normalized.get("review_batch_id") or "").strip():
        normalized["review_batch_id"] = "qwen-next-stage2-round1"

    if not (normalized.get("review_round") or "").strip():
        normalized["review_round"] = "round1"

    ensure_reviewer_decision(normalized, 1)
    ensure_reviewer_decision(normalized, 2)

    reviewer1 = reviewer_payload(normalized, 1)
    reviewer2 = reviewer_payload(normalized, 2)
    reviewer1_done = reviewer_complete(reviewer1)
    reviewer2_done = reviewer_complete(reviewer2)
    assignment = normalized.get("review_assignment", "")

    adjudicated = adjudicated_payload(normalized)
    has_full_adjudication = adjudicated_complete(adjudicated)

    if assignment == "single_annotate":
        normalized["adjudication_required"] = False
        if reviewer1_done and not has_full_adjudication:
            fill_from_single_reviewer(normalized, 1)
            has_full_adjudication = True
    else:
        needs_adjudication = disagreement_requires_adjudication(normalized)
        normalized["adjudication_required"] = needs_adjudication
        if reviewer1_done and reviewer2_done and not has_full_adjudication and not needs_adjudication:
            fill_from_matching_double_review(normalized)
            has_full_adjudication = True

    if not (normalized.get("curation_action") or "").strip():
        normalized["curation_action"] = curation_action_for(
            (normalized.get("adjudicated_final_decision") or "").strip() or None
        )

    if assignment == "single_annotate":
        if reviewer1_done and has_full_adjudication:
            normalized["annotation_status"] = "complete"
        else:
            normalized["annotation_status"] = "incomplete_review"
    else:
        if reviewer1_done and reviewer2_done and has_full_adjudication:
            normalized["annotation_status"] = "complete"
        elif reviewer1_done and reviewer2_done:
            normalized["annotation_status"] = "needs_adjudication"
        else:
            normalized["annotation_status"] = "incomplete_review"

    normalized["review_completed"] = normalized["annotation_status"] == "complete"
    if normalized["review_completed"]:
        normalized["review_completed_at"] = (normalized.get("review_completed_at") or "").strip() or args.completed_at
    else:
        normalized["review_completed_at"] = (normalized.get("review_completed_at") or "").strip()

    return normalized


def canonical_fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    ordered = [
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
    extras = [
        key
        for key in rows[0].keys()
        if key not in set(ordered)
    ]
    return ordered + extras


def csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = canonical_fieldnames(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fieldnames})


def jsonl_row(row: Dict[str, Any]) -> Dict[str, Any]:
    converted = dict(row)
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


def build_summary(rows: Sequence[Dict[str, Any]], input_csv: Path) -> Dict[str, Any]:
    status_counts = Counter(row["annotation_status"] for row in rows)
    complete_rows = [row for row in rows if row["annotation_status"] == "complete"]
    double_rows = [
        row
        for row in rows
        if row.get("review_assignment") == "double_annotate"
        and reviewer_complete(reviewer_payload(row, 1))
        and reviewer_complete(reviewer_payload(row, 2))
    ]

    summary: Dict[str, Any] = {
        "source_review_file": str(input_csv),
        "rows_total": len(rows),
        "annotation_status_counts": dict(sorted(status_counts.items())),
        "complete_rows": len(complete_rows),
        "pending_adjudication_rows": sum(1 for row in rows if row["annotation_status"] == "needs_adjudication"),
        "incomplete_review_rows": sum(1 for row in rows if row["annotation_status"] == "incomplete_review"),
        "double_annotate_rows_with_two_reviews": len(double_rows),
    }

    if complete_rows:
        decision_counts = Counter(row["adjudicated_final_decision"] for row in complete_rows)
        curation_counts = Counter(row["curation_action"] for row in complete_rows)
        summary["decision_counts"] = dict(sorted(decision_counts.items()))
        summary["curation_action_counts"] = dict(sorted(curation_counts.items()))
        summary["adjudicated_metrics"] = {
            "target_type_matched_true": sum(
                1 for row in complete_rows if parse_bool(row.get("adjudicated_target_type_matched"))
            ),
            "single_dominant_error_true": sum(
                1 for row in complete_rows if parse_bool(row.get("adjudicated_single_dominant_error"))
            ),
            "answer_preserved_true": sum(
                1 for row in complete_rows if parse_bool(row.get("adjudicated_answer_preserved"))
            ),
            "mean_clinical_plausibility": round(
                sum(parse_int(row.get("adjudicated_clinical_plausibility")) or 0 for row in complete_rows)
                / max(len(complete_rows), 1),
                3,
            ),
            "mean_faithfulness_drop": round(
                sum(parse_int(row.get("adjudicated_faithfulness_drop")) or 0 for row in complete_rows)
                / max(len(complete_rows), 1),
                3,
            ),
        }

        by_error: Dict[str, Dict[str, Any]] = {}
        grouped_by_error: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in complete_rows:
            grouped_by_error[str(row["error_symbol"])].append(row)
        for error_symbol, items in grouped_by_error.items():
            by_error[error_symbol] = {
                "rows": len(items),
                "decision_counts": dict(sorted(Counter(row["adjudicated_final_decision"] for row in items).items())),
            }
        summary["by_error_symbol"] = dict(sorted(by_error.items()))

    if double_rows:
        target_left = [parse_bool(row.get("reviewer1_target_type_matched")) for row in double_rows]
        target_right = [parse_bool(row.get("reviewer2_target_type_matched")) for row in double_rows]
        single_left = [parse_bool(row.get("reviewer1_single_dominant_error")) for row in double_rows]
        single_right = [parse_bool(row.get("reviewer2_single_dominant_error")) for row in double_rows]
        answer_left = [parse_bool(row.get("reviewer1_answer_preserved")) for row in double_rows]
        answer_right = [parse_bool(row.get("reviewer2_answer_preserved")) for row in double_rows]
        decision_left = [(row.get("reviewer1_final_decision") or "").strip() for row in double_rows]
        decision_right = [(row.get("reviewer2_final_decision") or "").strip() for row in double_rows]
        summary["agreement"] = {
            "target_type_matched_kappa": cohen_kappa(target_left, target_right),
            "single_dominant_error_kappa": cohen_kappa(single_left, single_right),
            "answer_preserved_kappa": cohen_kappa(answer_left, answer_right),
            "final_decision_kappa": cohen_kappa(decision_left, decision_right),
        }

    return summary


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_jsonl = Path(args.output_jsonl)
    output_summary = Path(args.output_summary)

    rows = load_csv(input_csv)
    normalized_rows = [normalize_row(row, args) for row in rows]

    write_csv(input_csv, normalized_rows)
    write_jsonl(output_jsonl, normalized_rows)

    summary = build_summary(normalized_rows, input_csv)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Finalized review CSV: {input_csv}")
    print(f"Annotated JSONL: {output_jsonl}")
    print(f"Summary JSON: {output_summary}")
    print(json.dumps(summary.get("annotation_status_counts", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
