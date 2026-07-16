#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
LABELS = ["entailment", "neutral", "contradiction"]

REPORTED_HELDOUT = {
    "n": 160,
    "accuracy": "0.838",
    "macro_f1": "0.827",
    "entailment_prf": "0.882 / 0.938 / 0.909",
    "neutral_prf": "0.793 / 0.793 / 0.793",
    "contradiction_prf": "0.824 / 0.737 / 0.778",
    "ece": "0.064",
    "brier": "0.213",
    "kappa": "0.74",
    "confusion": {
        "entailment": {"entailment": 60, "neutral": 4, "contradiction": 0},
        "neutral": {"entailment": 6, "neutral": 46, "contradiction": 6},
        "contradiction": {"entailment": 2, "neutral": 8, "contradiction": 28},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute label-level and training-level NLI false-positive diagnostics."
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=EXPERIMENT_DIR / "inputs/nli_validation_annotation.csv",
        help="Optional row-level NLI validation annotation CSV.",
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=EXPERIMENT_DIR / "inputs/nli_validation_items.jsonl",
        help="Optional row-level NLI validation sample JSONL to merge by sample_id/item_id.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=EXPERIMENT_DIR / "outputs/false_positive_metrics.csv",
        help="Output metrics CSV.",
    )
    parser.add_argument(
        "--tables-output",
        type=Path,
        default=EXPERIMENT_DIR / "outputs/nli_false_positive_tables.tex",
        help="Output LaTeX table file.",
    )
    parser.add_argument(
        "--use-reported-heldout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If row-level files are absent, emit the already reported held-out Study A metrics.",
    )
    parser.add_argument(
        "--split",
        choices=["all", "evaluation", "calibration"],
        default="evaluation",
        help="Which row-level split to use when validation annotation files are present.",
    )
    parser.add_argument("--risk-threshold", type=float, default=0.1)
    parser.add_argument("--entailment-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--top-label-gap-threshold", type=float, default=0.1)
    parser.add_argument("--contradiction-threshold", type=float, default=0.7)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def canonical_label(value: Any) -> str:
    text = normalize_text(value).lower().replace("-", "_")
    if "contra" in text:
        return "contradiction"
    if "entail" in text or text in {"support", "supported"}:
        return "entailment"
    if "neutral" in text or text in {"unknown", "unclear"}:
        return "neutral"
    return ""


def probability(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in row and normalize_text(row.get(key)) != "":
            return safe_float(row.get(key))
    return 0.0


def top_gap(row: dict[str, Any]) -> float:
    probs = [
        probability(row, "p_ent", "entailment"),
        probability(row, "p_neu", "neutral"),
        probability(row, "p_con", "contradiction"),
    ]
    probs.sort(reverse=True)
    return probs[0] - probs[1] if len(probs) >= 2 else 0.0


def risk_score(row: dict[str, Any]) -> float:
    return probability(row, "risk_score", "score", "aru_score")


def training_high_risk(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        risk_score(row) > args.risk_threshold
        and probability(row, "p_ent", "entailment") < args.entailment_confidence_threshold
        and top_gap(row) >= args.top_label_gap_threshold
    )


def contradiction_dominant(row: dict[str, Any], args: argparse.Namespace) -> bool:
    pred = canonical_label(row.get("predicted_label") or row.get("verifier_pred_label"))
    return pred == "contradiction" or probability(row, "p_con", "contradiction") >= args.contradiction_threshold


def format_decimal(value: float) -> str:
    return f"{value:.3f}"


def row_probabilities(row: dict[str, Any]) -> dict[str, float]:
    probs = {
        "entailment": probability(row, "p_ent", "entailment"),
        "neutral": probability(row, "p_neu", "neutral"),
        "contradiction": probability(row, "p_con", "contradiction"),
    }
    total = sum(probs.values())
    if total <= 0.0:
        pred = row.get("_predicted_label") or canonical_label(row.get("predicted_label") or row.get("verifier_pred_label"))
        return {label: 1.0 if label == pred else 0.0 for label in LABELS}
    return {label: value / total for label, value in probs.items()}


def class_prf(confusion: dict[str, dict[str, int]], label: str) -> tuple[float, float, float]:
    tp = confusion[label][label]
    fp = sum(confusion[gold][label] for gold in LABELS if gold != label)
    fn = sum(confusion[label][pred] for pred in LABELS if pred != label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1


def multiclass_brier(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        probs = row_probabilities(row)
        gold = row["_human_label"]
        total += sum((probs[label] - (1.0 if label == gold else 0.0)) ** 2 for label in LABELS)
    return total / len(rows)


def expected_calibration_error(rows: list[dict[str, Any]], bins: int = 10) -> float:
    if not rows:
        return 0.0
    total = len(rows)
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = []
        for row in rows:
            confidence = max(row_probabilities(row).values())
            if index == bins - 1:
                in_bin = lower <= confidence <= upper
            else:
                in_bin = lower <= confidence < upper
            if in_bin:
                members.append((row, confidence))
        if not members:
            continue
        avg_confidence = sum(confidence for _, confidence in members) / len(members)
        accuracy = sum(1 for row, _ in members if row["_human_label"] == row["_predicted_label"]) / len(members)
        ece += (len(members) / total) * abs(avg_confidence - accuracy)
    return ece


def row_level_display_summary(
    rows: list[dict[str, Any]],
    confusion: dict[str, dict[str, int]],
) -> dict[str, str]:
    n = len(rows)
    correct = sum(confusion[label][label] for label in LABELS)
    f1s = {}
    prf_strings = {}
    for label in LABELS:
        precision, recall, f1 = class_prf(confusion, label)
        f1s[label] = f1
        prf_strings[label] = f"{format_decimal(precision)} / {format_decimal(recall)} / {format_decimal(f1)}"
    macro_f1 = sum(f1s.values()) / len(LABELS) if LABELS else 0.0
    return {
        "accuracy": format_decimal(correct / n if n else 0.0),
        "macro_f1": format_decimal(macro_f1),
        "entailment_prf": prf_strings["entailment"],
        "neutral_prf": prf_strings["neutral"],
        "contradiction_prf": prf_strings["contradiction"],
        "ece": format_decimal(expected_calibration_error(rows)),
        "brier": format_decimal(multiclass_brier(rows)),
        "kappa": "NA",
    }


def reported_display_summary() -> dict[str, str]:
    return {
        "accuracy": str(REPORTED_HELDOUT["accuracy"]),
        "macro_f1": str(REPORTED_HELDOUT["macro_f1"]),
        "entailment_prf": str(REPORTED_HELDOUT["entailment_prf"]),
        "neutral_prf": str(REPORTED_HELDOUT["neutral_prf"]),
        "contradiction_prf": str(REPORTED_HELDOUT["contradiction_prf"]),
        "ece": str(REPORTED_HELDOUT["ece"]),
        "brier": str(REPORTED_HELDOUT["brier"]),
        "kappa": f"$\\kappa={REPORTED_HELDOUT['kappa']}$",
    }


def format_fraction(numerator: int | str, denominator: int | str) -> str:
    if isinstance(numerator, str) or isinstance(denominator, str):
        return "NA"
    if denominator == 0:
        return "NA"
    percent = (Decimal(numerator) * Decimal(100) / Decimal(denominator)).quantize(
        Decimal("0.1"),
        rounding=ROUND_HALF_UP,
    )
    return f"{numerator}/{denominator} ({percent}%)"


def metric_row(
    study: str,
    metric: str,
    numerator: int | str,
    denominator: int | str,
    status: str,
    notes: str = "",
) -> dict[str, Any]:
    if isinstance(numerator, int) and isinstance(denominator, int) and denominator:
        rate = numerator / denominator
        formatted = format_fraction(numerator, denominator)
    else:
        rate = "NA"
        formatted = "NA"
    return {
        "study": study,
        "metric": metric,
        "numerator": numerator,
        "denominator": denominator,
        "rate": rate if isinstance(rate, str) else round(rate, 6),
        "formatted": formatted,
        "status": status,
        "notes": notes,
    }


def merge_row_level_inputs(annotation_file: Path, sample_file: Path, split: str) -> list[dict[str, Any]]:
    annotations = load_csv(annotation_file)
    sample_rows = load_jsonl(sample_file)
    sample_lookup = {}
    for row in sample_rows:
        key = normalize_text(row.get("sample_id") or row.get("item_id"))
        if key:
            sample_lookup[key] = row

    merged_rows: list[dict[str, Any]] = []
    for row in annotations:
        key = normalize_text(row.get("sample_id") or row.get("item_id"))
        merged = dict(sample_lookup.get(key, {}))
        merged.update(row)
        split_name = normalize_text(merged.get("calibration_split")).lower()
        if split != "all" and split_name and split_name != split:
            continue
        human = canonical_label(merged.get("human_label") or merged.get("final_human_label"))
        pred = canonical_label(merged.get("predicted_label") or merged.get("verifier_pred_label"))
        if not human or not pred:
            continue
        merged["_human_label"] = human
        merged["_predicted_label"] = pred
        merged_rows.append(merged)
    return merged_rows


def compute_row_level_metrics(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    gold_entail = [row for row in rows if row["_human_label"] == "entailment"]
    entail_to_non = [row for row in gold_entail if row["_predicted_label"] in {"neutral", "contradiction"}]
    entail_to_con = [row for row in gold_entail if row["_predicted_label"] == "contradiction"]
    entail_to_mask = [row for row in gold_entail if training_high_risk(row, args)]
    entail_to_con_mask = [row for row in gold_entail if training_high_risk(row, args) and contradiction_dominant(row, args)]
    denom = len(gold_entail)
    return [
        metric_row("Study A", "gold_entailment_cases", denom, len(rows), "computed_from_row_level_file"),
        metric_row("Study A", "label_level_entailment_to_non_entailment", len(entail_to_non), denom, "computed_from_row_level_file"),
        metric_row("Study A", "label_level_entailment_to_contradiction", len(entail_to_con), denom, "computed_from_row_level_file"),
        metric_row("Study C", "training_level_entailment_to_high_risk_mask", len(entail_to_mask), denom, "computed_from_row_level_file"),
        metric_row("Study C", "training_level_entailment_to_contradiction_risk_mask", len(entail_to_con_mask), denom, "computed_from_row_level_file"),
    ]


def reported_metrics() -> list[dict[str, Any]]:
    confusion = REPORTED_HELDOUT["confusion"]
    gold_entail = sum(confusion["entailment"].values())
    entail_to_non = confusion["entailment"]["neutral"] + confusion["entailment"]["contradiction"]
    entail_to_con = confusion["entailment"]["contradiction"]
    return [
        metric_row("Study A", "heldout_aru_premise_pairs", REPORTED_HELDOUT["n"], REPORTED_HELDOUT["n"], "reported_in_manuscript"),
        metric_row("Study A", "gold_entailment_cases", gold_entail, REPORTED_HELDOUT["n"], "reported_in_manuscript"),
        metric_row("Study A", "label_level_entailment_to_non_entailment", entail_to_non, gold_entail, "reported_in_manuscript"),
        metric_row("Study A", "label_level_entailment_to_contradiction", entail_to_con, gold_entail, "reported_in_manuscript"),
    ]


def confusion_from_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for row in rows:
        matrix[row["_human_label"]][row["_predicted_label"]] += 1
    return matrix


def selected_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    for row in rows:
        if row["metric"] == metric:
            return row
    return {"formatted": "NA", "numerator": "NA", "denominator": "NA"}


def render_tables(metrics: list[dict[str, Any]], row_level_rows: list[dict[str, Any]]) -> str:
    if row_level_rows:
        confusion = confusion_from_rows(row_level_rows)
        heldout_n = len(row_level_rows)
        table_status = "computed from row-level files"
        gold_entail_n = sum(confusion["entailment"].values())
        display = row_level_display_summary(row_level_rows, confusion)
    else:
        confusion = REPORTED_HELDOUT["confusion"]
        heldout_n = REPORTED_HELDOUT["n"]
        table_status = "reported manuscript values"
        gold_entail_n = sum(confusion["entailment"].values())
        display = reported_display_summary()

    label_non = selected_metric(metrics, "label_level_entailment_to_non_entailment")["formatted"]
    label_con = selected_metric(metrics, "label_level_entailment_to_contradiction")["formatted"]
    threshold_caption = "Label-level false-positive diagnostics on the held-out in-pipeline medical evaluation split."
    threshold_rows = [
        f"Gold entailment cases & {gold_entail_n} \\\\",
        f"Label-level entailment $\\rightarrow$ non-entailment & {label_non} \\\\",
        f"Label-level entailment $\\rightarrow$ contradiction & {label_con} \\\\",
    ]
    if row_level_rows:
        train_mask = selected_metric(metrics, "training_level_entailment_to_high_risk_mask")["formatted"]
        train_con_mask = selected_metric(metrics, "training_level_entailment_to_contradiction_risk_mask")["formatted"]
        threshold_caption = "Label-level and training-level false-positive diagnostics under the Fa-DPO high-risk masking rule."
        threshold_rows.extend(
            [
                f"Training-level entailment $\\rightarrow$ high-risk mask & {train_mask} \\\\",
                f"Training-level entailment $\\rightarrow$ contradiction-risk mask & {train_con_mask} \\\\",
            ]
        )

    return "\n".join(
        [
            "% Auto-generated by analyze_false_positive_metrics.py.",
            f"% Status: {table_status}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\footnotesize",
            f"\\caption{{Verifier-level NLI validation and false-positive diagnostics on the held-out in-pipeline medical evaluation split ($n={heldout_n}$).}}",
            "\\label{tab:nli_false_positive_validation}",
            "\\begin{tabular}{lc}",
            "\\toprule",
            "\\textbf{Metric} & \\textbf{Statistic} \\\\",
            "\\midrule",
            f"Held-out ARU-premise pairs & {heldout_n} \\\\",
            f"Accuracy & {display['accuracy']} \\\\",
            f"Macro-F1 & {display['macro_f1']} \\\\",
            f"Entailment precision / recall / F1 & {display['entailment_prf']} \\\\",
            f"Neutral precision / recall / F1 & {display['neutral_prf']} \\\\",
            f"Contradiction precision / recall / F1 & {display['contradiction_prf']} \\\\",
            f"Gold-entailment $\\rightarrow$ non-entailment rate & {label_non} \\\\",
            f"Gold-entailment $\\rightarrow$ contradiction rate & {label_con} \\\\",
            f"Expected calibration error & {display['ece']} \\\\",
            f"Brier score & {display['brier']} \\\\",
            f"Double-annotation agreement & {display['kappa']} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\footnotesize",
            f"\\caption{{Verifier-level NLI confusion matrix on the held-out in-pipeline medical evaluation split ($n={heldout_n}$). Rows denote human labels and columns denote verifier predictions.}}",
            "\\label{tab:nli_false_positive_confusion}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "\\textbf{Gold $\\backslash$ Pred} & \\textbf{Entailment} & \\textbf{Neutral} & \\textbf{Contradiction} \\\\",
            "\\midrule",
            f"Entailment & {confusion['entailment']['entailment']} & {confusion['entailment']['neutral']} & {confusion['entailment']['contradiction']} \\\\",
            f"Neutral & {confusion['neutral']['entailment']} & {confusion['neutral']['neutral']} & {confusion['neutral']['contradiction']} \\\\",
            f"Contradiction & {confusion['contradiction']['entailment']} & {confusion['contradiction']['neutral']} & {confusion['contradiction']['contradiction']} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\footnotesize",
            f"\\caption{{{threshold_caption}}}",
            "\\label{tab:threshold_level_false_positive}",
            "\\begin{tabular}{lc}",
            "\\toprule",
            "\\textbf{Metric} & \\textbf{Statistic} \\\\",
            "\\midrule",
            *threshold_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    row_level_rows: list[dict[str, Any]] = []
    if args.annotation_file.exists():
        row_level_rows = merge_row_level_inputs(args.annotation_file, args.sample_file, args.split)
        if row_level_rows:
            metrics = compute_row_level_metrics(row_level_rows, args)
        elif args.use_reported_heldout:
            metrics = reported_metrics()
        else:
            raise ValueError(
                f"No usable row-level annotations in {args.annotation_file} for split={args.split}."
            )
    elif args.use_reported_heldout:
        metrics = reported_metrics()
    else:
        raise FileNotFoundError(f"Missing annotation file: {args.annotation_file}")

    fieldnames = ["study", "metric", "numerator", "denominator", "rate", "formatted", "status", "notes"]
    write_csv(args.output_csv, metrics, fieldnames)
    args.tables_output.parent.mkdir(parents=True, exist_ok=True)
    args.tables_output.write_text(render_tables(metrics, row_level_rows), encoding="utf-8")

    status_counts = Counter(row["status"] for row in metrics)
    print(f"Wrote metrics: {args.output_csv}")
    print(f"Wrote tables: {args.tables_output}")
    print(f"Metric statuses: {dict(status_counts)}")


if __name__ == "__main__":
    main()
