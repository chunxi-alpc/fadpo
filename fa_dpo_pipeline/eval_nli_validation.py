#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from common import normalize_text, output_path, write_json
from experiment_utils import load_records, write_csv


LABEL_MAP = {
    "entail": "entail",
    "entailed": "entail",
    "entailment": "entail",
    "support": "entail",
    "supported": "entail",
    "neutral": "neutral",
    "unknown": "neutral",
    "unclear": "neutral",
    "contradict": "contradict",
    "contradiction": "contradict",
    "contradicted": "contradict",
    "refute": "contradict",
    "refuted": "contradict",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate manually annotated NLI validation results against verifier predictions."
    )
    parser.add_argument(
        "--annotation-file",
        default=output_path("nli_validation", "annotation.csv"),
        help="Annotation CSV filled by human reviewers.",
    )
    parser.add_argument(
        "--sample-file",
        default=output_path("nli_validation", "sample.jsonl"),
        help="Sample JSONL exported by sample_nli_validation.py.",
    )
    parser.add_argument(
        "--metrics-output",
        default=output_path("nli_validation", "metrics.json"),
        help="Where to write summary metrics JSON.",
    )
    parser.add_argument(
        "--confusion-output",
        default=output_path("nli_validation", "confusion_matrix.csv"),
        help="Where to write the confusion matrix CSV.",
    )
    parser.add_argument(
        "--errors-output",
        default=output_path("nli_validation", "error_cases.md"),
        help="Where to write representative error cases.",
    )
    parser.add_argument(
        "--calibration-output",
        default=output_path("nli_validation", "calibration_metrics.json"),
        help="Where to write calibration metrics JSON.",
    )
    parser.add_argument(
        "--route-breakdown-output",
        default=output_path("nli_validation", "route_breakdown.csv"),
        help="Where to write route-wise metrics CSV.",
    )
    parser.add_argument(
        "--route-confusion-output",
        default=output_path("nli_validation", "route_confusion_matrix.csv"),
        help="Where to write the route-wise confusion-matrix CSV.",
    )
    parser.add_argument(
        "--reliability-output",
        default=output_path("nli_validation", "reliability_curve.csv"),
        help="Where to write the reliability-curve CSV.",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Number of bins for expected calibration error.",
    )
    parser.add_argument(
        "--split",
        choices=["all", "evaluation", "calibration"],
        default="evaluation",
        help="Which split to evaluate when `calibration_split` is present.",
    )
    parser.add_argument(
        "--max-error-cases",
        type=int,
        default=20,
        help="Maximum number of representative mismatches to export.",
    )
    return parser.parse_args()


def canonical_label(value: Any) -> str:
    text = normalize_text(value).lower()
    if text not in LABEL_MAP:
        return ""
    return LABEL_MAP[text]


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def route_bucket(row: Dict[str, Any]) -> str:
    text = normalize_text(row.get("route_bucket"))
    if text:
        return text
    role = normalize_text(row.get("aru_role")).upper()
    if role == "O":
        return "observation_context"
    if role == "W":
        return "warrant_knowledge"
    if role == "D":
        return "differentiation_combined"
    if role == "C":
        return "claim_history"
    return normalize_text(row.get("verifier_route")) or "unknown"


def probabilities_for_row(row: Dict[str, Any]) -> Dict[str, float]:
    probs = {
        "entail": safe_float(row.get("entailment")),
        "neutral": safe_float(row.get("neutral")),
        "contradict": safe_float(row.get("contradiction")),
    }
    total = sum(probs.values())
    if total <= 0.0:
        predicted = canonical_label(row.get("predicted_label"))
        if predicted:
            return {label: 1.0 if label == predicted else 0.0 for label in probs}
        return {label: 0.0 for label in probs}
    return {label: value / total for label, value in probs.items()}


def has_probability_fields(row: Dict[str, Any]) -> bool:
    return any(normalize_text(row.get(key)) for key in ("entailment", "neutral", "contradiction"))


def probability_argmax_label(probs: Dict[str, float]) -> str:
    if not probs:
        return ""
    return max(probs, key=probs.get)


def predicted_confidence(row: Dict[str, Any], probs: Dict[str, float]) -> float:
    return max(probs.values()) if probs else 0.0


def multiclass_brier(rows: List[Dict[str, Any]], labels: List[str]) -> float:
    prob_rows = [row for row in rows if row.get("_has_probs")]
    if not prob_rows:
        return 0.0
    total = 0.0
    for row in prob_rows:
        probs = row["_probs"]
        gold = row["human_label"]
        row_score = 0.0
        for label in labels:
            target = 1.0 if gold == label else 0.0
            row_score += (probs.get(label, 0.0) - target) ** 2
        total += row_score
    return total / len(prob_rows)


def calibration_curve(rows: List[Dict[str, Any]], num_bins: int) -> List[Dict[str, Any]]:
    prob_rows = [row for row in rows if row.get("_has_probs")]
    if not prob_rows:
        return []
    bins = max(int(num_bins), 1)
    curve: List[Dict[str, Any]] = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = []
        for row in prob_rows:
            confidence = row["_predicted_confidence"]
            if index == bins - 1:
                in_bin = lower <= confidence <= upper
            else:
                in_bin = lower <= confidence < upper
            if in_bin:
                members.append(row)
        if not members:
            curve.append(
                {
                    "bin_index": index,
                    "lower_bound": round(lower, 6),
                    "upper_bound": round(upper, 6),
                    "count": 0,
                    "avg_confidence": 0.0,
                    "accuracy": 0.0,
                    "gap": 0.0,
                }
            )
            continue
        avg_conf = sum(row["_predicted_confidence"] for row in members) / len(members)
        accuracy = sum(1.0 for row in members if row["human_label"] == row["_prob_argmax_label"]) / len(members)
        curve.append(
            {
                "bin_index": index,
                "lower_bound": round(lower, 6),
                "upper_bound": round(upper, 6),
                "count": len(members),
                "avg_confidence": round(avg_conf, 6),
                "accuracy": round(accuracy, 6),
                "gap": round(abs(avg_conf - accuracy), 6),
            }
        )
    return curve


def expected_calibration_error(rows: List[Dict[str, Any]], num_bins: int) -> float:
    prob_rows = [row for row in rows if row.get("_has_probs")]
    if not prob_rows:
        return 0.0
    curve = calibration_curve(rows, num_bins)
    total = len(prob_rows)
    return sum((row["count"] / total) * row["gap"] for row in curve)


def summarize_rows(rows: List[Dict[str, Any]], labels: List[str], ece_bins: int) -> Dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "n_with_probs": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "ece": 0.0,
            "brier": 0.0,
            "label_prob_mismatch_count": 0,
            "class_wise_f1": {label: 0.0 for label in labels},
            "support": {label: 0 for label in labels},
        }
    y_true = [row["human_label"] for row in rows]
    y_pred = [row["predicted_label"] for row in rows]
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    prob_rows = [row for row in rows if row.get("_has_probs")]
    return {
        "n": len(rows),
        "n_with_probs": len(prob_rows),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 6),
        "ece": round(float(expected_calibration_error(rows, ece_bins)), 6),
        "brier": round(float(multiclass_brier(rows, labels)), 6),
        "label_prob_mismatch_count": sum(
            1 for row in prob_rows if row.get("_prob_argmax_label") and row.get("_prob_argmax_label") != row.get("predicted_label")
        ),
        "class_wise_f1": {label: round(float(report.get(label, {}).get("f1-score", 0.0)), 6) for label in labels},
        "support": {label: int(report.get(label, {}).get("support", 0)) for label in labels},
    }


def confusion_rows_for_group(rows: List[Dict[str, Any]], labels: List[str], group_name: str) -> List[Dict[str, Any]]:
    if not rows:
        return []
    y_true = [row["human_label"] for row in rows]
    y_pred = [row["predicted_label"] for row in rows]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    long_rows: List[Dict[str, Any]] = []
    for gold_index, gold_label in enumerate(labels):
        for pred_index, pred_label in enumerate(labels):
            long_rows.append(
                {
                    "route": group_name,
                    "gold_label": gold_label,
                    "predicted_label": pred_label,
                    "count": int(matrix[gold_index][pred_index]),
                }
            )
    return long_rows


def main() -> None:
    args = parse_args()
    annotations = load_records(args.annotation_file)
    samples = {str(row.get("sample_id")): row for row in load_records(args.sample_file)}

    rows: List[Dict[str, Any]] = []
    for row in annotations:
        human = canonical_label(row.get("human_label"))
        predicted = canonical_label(row.get("predicted_label"))
        if not human or not predicted:
            continue
        merged = dict(samples.get(str(row.get("sample_id")), {}))
        merged.update(row)
        split_name = normalize_text(merged.get("calibration_split")).lower()
        if args.split != "all" and split_name and split_name != args.split:
            continue
        merged["human_label"] = human
        merged["predicted_label"] = predicted
        merged["route_bucket"] = route_bucket(merged)
        merged["_has_probs"] = has_probability_fields(merged)
        merged["_probs"] = probabilities_for_row(merged)
        merged["_prob_argmax_label"] = probability_argmax_label(merged["_probs"])
        merged["_predicted_confidence"] = predicted_confidence(merged, merged["_probs"])
        rows.append(merged)

    labels = ["entail", "neutral", "contradict"]
    y_true = [row["human_label"] for row in rows]
    y_pred = [row["predicted_label"] for row in rows]

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = summarize_rows(rows, labels, args.ece_bins)
    metrics["evaluated_split"] = args.split
    split_values = [normalize_text(row.get("calibration_split")) or "unspecified" for row in rows]
    prob_rows = [row for row in rows if row.get("_has_probs")]
    metrics["route_counts"] = {
        route: sum(1 for row in rows if row.get("route_bucket") == route)
        for route in sorted({row.get("route_bucket", "unknown") for row in rows})
    }
    metrics["split_counts"] = {
        split: sum(1 for value in split_values if value == split)
        for split in sorted(set(split_values))
    }
    write_json(args.metrics_output, metrics)
    write_json(
        args.calibration_output,
        {
            "n": metrics["n"],
            "n_with_probs": metrics["n_with_probs"],
            "evaluated_split": args.split,
            "ece_bins": args.ece_bins,
            "ece": metrics["ece"],
            "brier": metrics["brier"],
            "label_prob_mismatch_count": metrics["label_prob_mismatch_count"],
            "avg_predicted_confidence": round(
                float(sum(row["_predicted_confidence"] for row in prob_rows) / len(prob_rows)),
                6,
            )
            if prob_rows
            else 0.0,
        },
    )

    matrix_rows = []
    for gold_index, gold_label in enumerate(labels):
        row = {"gold_label": gold_label}
        for pred_index, pred_label in enumerate(labels):
            row[pred_label] = int(matrix[gold_index][pred_index]) if rows else 0
        matrix_rows.append(row)
    write_csv(args.confusion_output, matrix_rows, fieldnames=["gold_label", *labels])

    reliability_rows = calibration_curve(rows, args.ece_bins)
    write_csv(
        args.reliability_output,
        reliability_rows,
        fieldnames=["bin_index", "lower_bound", "upper_bound", "count", "avg_confidence", "accuracy", "gap"],
    )

    route_rows = []
    route_confusion_rows = []
    for route in sorted({row.get("route_bucket", "unknown") for row in rows}):
        route_subset = [row for row in rows if row.get("route_bucket") == route]
        summary = summarize_rows(route_subset, labels, args.ece_bins)
        route_rows.append(
            {
                "route": route,
                "n": summary["n"],
                "accuracy": summary["accuracy"],
                "macro_f1": summary["macro_f1"],
                "ece": summary["ece"],
                "brier": summary["brier"],
                "entail_f1": summary["class_wise_f1"]["entail"],
                "neutral_f1": summary["class_wise_f1"]["neutral"],
                "contradict_f1": summary["class_wise_f1"]["contradict"],
            }
        )
        route_confusion_rows.extend(confusion_rows_for_group(route_subset, labels, route))
    write_csv(
        args.route_breakdown_output,
        route_rows,
        fieldnames=["route", "n", "accuracy", "macro_f1", "ece", "brier", "entail_f1", "neutral_f1", "contradict_f1"],
    )
    write_csv(
        args.route_confusion_output,
        route_confusion_rows,
        fieldnames=["route", "gold_label", "predicted_label", "count"],
    )

    mismatches = [row for row in rows if row["human_label"] != row["predicted_label"]]
    mismatches = mismatches[: args.max_error_cases]
    lines = ["# NLI Error Cases", ""]
    for row in mismatches:
        lines.extend(
            [
                f"## {row.get('sample_id')}",
                f"- Gold: `{row.get('human_label')}`",
                f"- Predicted: `{row.get('predicted_label')}`",
                f"- ARU role: `{row.get('aru_role')}`",
                f"- Route: `{row.get('route_bucket')}`",
                f"- Predicted confidence: `{row.get('_predicted_confidence', 0.0):.4f}`",
                f"- Split: `{normalize_text(row.get('calibration_split')) or 'unspecified'}`",
                f"- ARU: {normalize_text(row.get('aru_text'))}",
                f"- Evidence: {normalize_text(row.get('evidence_text'))}",
                f"- Note: {normalize_text(row.get('notes'))}",
                "",
            ]
        )
    Path(args.errors_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.errors_output).write_text("\n".join(lines), encoding="utf-8")

    print(f"Evaluated rows: {len(rows)}")
    print(f"Metrics: {args.metrics_output}")
    print(f"Calibration metrics: {args.calibration_output}")
    print(f"Route breakdown: {args.route_breakdown_output}")
    print(f"Route confusion matrix: {args.route_confusion_output}")
    print(f"Reliability curve: {args.reliability_output}")
    print(f"Confusion matrix: {args.confusion_output}")
    print(f"Error cases: {args.errors_output}")


if __name__ == "__main__":
    main()
