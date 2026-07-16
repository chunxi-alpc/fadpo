#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_ai_pilot_preannotations import ROLE_MAP, build_candidates, match_span, normalize_text


ROLES = ["Observation", "Warrant", "Differentiation", "Claim"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def has_bool_field(row: dict[str, Any], field: str) -> bool:
    return str(row.get(field) if row.get(field) is not None else "").strip() != ""


def parse_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_role(value: str) -> str:
    value = (value or "").strip()
    return ROLE_MAP.get(value, value)


def token_offsets(text: str) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    start: int | None = None
    for idx, char in enumerate(text):
        if char.isspace():
            if start is not None:
                offsets.append((start, idx))
                start = None
            continue
        if start is None:
            start = idx
    if start is not None:
        offsets.append((start, len(text)))
    return offsets


def char_span_to_token_span(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    indices = [idx for idx, (tok_start, tok_end) in enumerate(offsets) if tok_start < end and tok_end > start]
    if not indices:
        return None
    return min(indices), max(indices) + 1


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    inter = max(0, right - left)
    if inter <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def build_pred_records(items: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items_by_id = {str(row["item_id"]): row for row in items}
    records: list[dict[str, Any]] = []
    for pred in predictions:
        item_id = str(pred.get("id"))
        item = items_by_id[item_id]
        response = item.get("response_text", "")
        final_answer_norm = normalize_text(item.get("final_answer", ""))
        candidates = build_candidates(response)
        cursor = 0
        for idx, aru in enumerate(pred.get("arus") or [], 1):
            text = str(aru.get("text") or "")
            explicit_start = parse_int(aru.get("char_start"))
            explicit_end = parse_int(aru.get("char_end"))
            if (
                explicit_start is not None
                and explicit_end is not None
                and 0 <= explicit_start < explicit_end <= len(response)
            ):
                start = explicit_start
                end = explicit_end
                span_text = response[start:end]
                score = 1.0
                method = "provided_char_span"
                if end >= cursor:
                    cursor = end
            else:
                start, end, span_text, score, method = match_span(response, text, candidates, cursor)
                if start is not None and end is not None and end >= cursor:
                    cursor = end
            role = normalize_role(str(aru.get("role_label") or aru.get("type") or ""))
            if has_bool_field(aru, "is_final_answer_claim"):
                is_final_answer_claim = parse_bool(aru.get("is_final_answer_claim"))
            else:
                is_final_answer_claim = (
                    role == "Claim"
                    and bool(final_answer_norm)
                    and (
                        final_answer_norm in normalize_text(text)
                        or final_answer_norm in normalize_text(span_text)
                    )
                )
            if has_bool_field(aru, "mask_eligible"):
                mask_eligible = parse_bool(aru.get("mask_eligible"))
            else:
                mask_eligible = role in {"Warrant", "Differentiation", "Claim"}
            records.append(
                {
                    "item_id": item_id,
                    "aru_id": f"P_{idx:03d}",
                    "char_start": start,
                    "char_end": end,
                    "span_text": span_text,
                    "role": role,
                    "is_final_answer_claim": is_final_answer_claim,
                    "mask_eligible": mask_eligible,
                    "match_method": method,
                    "match_score": score,
                    "source_text": text,
                }
            )
    return records


def build_gold_records(annotations: list[dict[str, str]], item_response: dict[str, str]) -> tuple[list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    qc: Counter[str] = Counter()
    for idx, row in enumerate(annotations, 1):
        item_id = str(row.get("item_id") or "").strip()
        if not item_id:
            qc["blank_item_id"] += 1
            continue
        response = row.get("response_text") or item_response.get(item_id, "")
        use_final_span = bool(str(row.get("final_char_start") or "").strip() and str(row.get("final_char_end") or "").strip())
        start = parse_int(row.get("final_char_start" if use_final_span else "char_start"))
        end = parse_int(row.get("final_char_end" if use_final_span else "char_end"))
        role = normalize_role(row.get("final_role_label") if row.get("final_role_label") else row.get("role_label", ""))
        is_final = parse_bool(
            row.get("final_is_final_answer_claim")
            if str(row.get("final_is_final_answer_claim") or "").strip()
            else row.get("is_final_answer_claim")
        )
        mask = parse_bool(
            row.get("final_mask_eligible")
            if str(row.get("final_mask_eligible") or "").strip()
            else row.get("mask_eligible")
        )
        span_text = row.get("span_text") or ""
        valid_span = start is not None and end is not None and 0 <= start < end <= len(response)
        if not valid_span:
            qc["invalid_or_blank_span"] += 1
        else:
            observed = response[start:end]
            if span_text and normalize_text(observed) != normalize_text(span_text):
                qc["span_text_mismatch"] += 1
        if role not in ROLES:
            qc["invalid_role"] += 1
        records.append(
            {
                "item_id": item_id,
                "aru_id": row.get("aru_id") or f"G_{idx:05d}",
                "char_start": start,
                "char_end": end,
                "span_text": span_text,
                "role": role,
                "is_final_answer_claim": is_final,
                "mask_eligible": mask,
                "valid_span": valid_span,
            }
        )
    return records, qc


def add_token_spans(records: list[dict[str, Any]], item_response: dict[str, str]) -> int:
    invalid = 0
    offsets_by_item = {item_id: token_offsets(text) for item_id, text in item_response.items()}
    for record in records:
        start = record.get("char_start")
        end = record.get("char_end")
        item_id = record.get("item_id")
        if start is None or end is None or item_id not in offsets_by_item:
            record["token_span"] = None
            invalid += 1
            continue
        token_span = char_span_to_token_span(offsets_by_item[item_id], start, end)
        record["token_span"] = token_span
        if token_span is None:
            invalid += 1
    return invalid


def max_cardinality_match(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    threshold: float,
    predicate=lambda _g, _p: True,
) -> list[tuple[int, int, float]]:
    edges: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for gi, g in enumerate(gold):
        if g.get("token_span") is None:
            continue
        for pi, p in enumerate(pred):
            if p.get("token_span") is None:
                continue
            if not predicate(g, p):
                continue
            score = iou(g["token_span"], p["token_span"])
            if score >= threshold:
                edges[gi].append((pi, score))
        edges[gi].sort(key=lambda item: item[1], reverse=True)

    pred_to_gold: dict[int, int] = {}

    def dfs(gi: int, seen: set[int]) -> bool:
        for pi, _score in edges.get(gi, []):
            if pi in seen:
                continue
            seen.add(pi)
            if pi not in pred_to_gold or dfs(pred_to_gold[pi], seen):
                pred_to_gold[pi] = gi
                return True
        return False

    for gi in sorted(edges, key=lambda key: len(edges[key])):
        dfs(gi, set())

    pairs: list[tuple[int, int, float]] = []
    for pi, gi in pred_to_gold.items():
        score = iou(gold[gi]["token_span"], pred[pi]["token_span"])
        pairs.append((gi, pi, score))
    pairs.sort()
    return pairs


def prf(tp: int, pred_total: int, gold_total: int) -> tuple[float, float, float]:
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def macro_f1(labels: list[str], gold_labels: list[str], pred_labels: list[str]) -> float:
    values: list[float] = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold_labels, pred_labels))
        fp = sum(g != label and p == label for g, p in zip(gold_labels, pred_labels))
        fn = sum(g == label and p != label for g, p in zip(gold_labels, pred_labels))
        _p, _r, f1 = prf(tp, tp + fp, tp + fn)
        values.append(f1)
    return sum(values) / len(values) if values else 0.0


def compute_metrics(
    gold_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
    threshold: float,
) -> dict[str, float | int]:
    gold_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pred_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in gold_records:
        gold_by_item[record["item_id"]].append(record)
    for record in pred_records:
        pred_by_item[record["item_id"]].append(record)

    matches: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for item_id in sorted(set(gold_by_item) | set(pred_by_item)):
        gold = gold_by_item.get(item_id, [])
        pred = pred_by_item.get(item_id, [])
        for gi, pi, score in max_cardinality_match(gold, pred, threshold):
            matches.append((gold[gi], pred[pi], score))

    tp = len(matches)
    precision, recall, f1 = prf(tp, len(pred_records), len(gold_records))
    role_gold = [g["role"] for g, _p, _s in matches]
    role_pred = [p["role"] for _g, p, _s in matches]
    role_acc = sum(g == p for g, p in zip(role_gold, role_pred)) / len(matches) if matches else 0.0
    role_macro = macro_f1(ROLES, role_gold, role_pred) if matches else 0.0

    def subset_metric(field: str) -> tuple[float, float, float]:
        gold_subset = [record for record in gold_records if record.get(field)]
        pred_subset = [record for record in pred_records if record.get(field)]
        subset_matches = 0
        gb: dict[str, list[dict[str, Any]]] = defaultdict(list)
        pb: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in gold_subset:
            gb[record["item_id"]].append(record)
        for record in pred_subset:
            pb[record["item_id"]].append(record)
        for item_id in sorted(set(gb) | set(pb)):
            subset_matches += len(max_cardinality_match(gb.get(item_id, []), pb.get(item_id, []), threshold))
        return prf(subset_matches, len(pred_subset), len(gold_subset))

    final_p, final_r, final_f1 = subset_metric("is_final_answer_claim")
    mask_p, mask_r, mask_f1 = subset_metric("mask_eligible")
    return {
        "iou_threshold": threshold,
        "gold_spans": len(gold_records),
        "predicted_spans": len(pred_records),
        "matched_spans": tp,
        "span_precision": precision,
        "span_recall": recall,
        "span_f1": f1,
        "role_accuracy": role_acc,
        "role_macro_f1": role_macro,
        "final_claim_precision": final_p,
        "final_claim_recall": final_r,
        "final_claim_f1": final_f1,
        "mask_eligible_precision": mask_p,
        "mask_eligible_recall": mask_r,
        "mask_eligible_f1": mask_f1,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ARU parser validation annotations.")
    parser.add_argument("--items", type=Path, default=Path("experiments/verifier_validation/aru_parser_validation/samples/parser_validation_items_100.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("experiments/verifier_validation/aru_parser_validation/outputs/parser_predictions_100.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("experiments/verifier_validation/aru_parser_validation/annotations/final_clinician_annotations.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/verifier_validation/aru_parser_validation/outputs"))
    args = parser.parse_args()

    items = load_jsonl(args.items)
    predictions = load_jsonl(args.predictions)
    annotations = load_csv(args.annotations)
    item_response = {str(item["item_id"]): item.get("response_text", "") for item in items}

    pred_records = build_pred_records(items, predictions)
    gold_records, qc_counts = build_gold_records(annotations, item_response)
    pred_invalid_token = add_token_spans(pred_records, item_response)
    gold_invalid_token = add_token_spans(gold_records, item_response)

    annotator_counts = Counter(row.get("annotator_id", "") for row in annotations)
    final_field_rows = sum(
        bool(str(row.get("final_char_start") or "").strip() and str(row.get("final_char_end") or "").strip())
        for row in annotations
    )
    adjudication_status_rows = sum(
        str(row.get("adjudication_status") or "").strip().lower() in {"agreement", "adjudicated"}
        for row in annotations
    )
    non_clinician_marker_rows = sum(
        any(
            marker in str(row.get("annotator_id", ""))
            for marker in ["AI_PILOT", "AI_SIMULATION", "NOT_CLINICIAN"]
        )
        for row in annotations
    )
    needs_human_review_rows = sum(str(row.get("needs_human_review") or "").strip().lower() == "true" for row in annotations)
    role_map_for_qc = {"O": "Observation", "W": "Warrant", "D": "Differentiation", "C": "Claim"}
    role_equals_predicted_rows = 0
    span_equals_predicted_rows = 0
    for row in annotations:
        predicted_role = role_map_for_qc.get(str(row.get("predicted_role") or ""), str(row.get("predicted_role") or ""))
        if str(row.get("role_label") or "") == predicted_role:
            role_equals_predicted_rows += 1
        if normalize_text(row.get("span_text") or "") == normalize_text(row.get("predicted_aru_text") or ""):
            span_equals_predicted_rows += 1
    valid_for_rebuttal = (
        bool(annotations)
        and non_clinician_marker_rows == 0
        and qc_counts["invalid_or_blank_span"] == 0
        and qc_counts["invalid_role"] == 0
        and final_field_rows == len(annotations)
        and adjudication_status_rows == len(annotations)
        and len({row.get("item_id") for row in annotations}) == len(items)
    )

    metrics_rows = [compute_metrics(gold_records, pred_records, threshold) for threshold in [0.5, 0.7]]
    write_csv(args.output_dir / "parser_validation_metrics.csv", metrics_rows)

    pred_method_counts = Counter(record["match_method"] for record in pred_records)
    role_counts_gold = Counter(record["role"] for record in gold_records)
    role_counts_pred = Counter(record["role"] for record in pred_records)
    error_rows = [
        {"item": "valid_for_rebuttal", "value": str(valid_for_rebuttal).lower()},
        {"item": "annotation_rows", "value": len(annotations)},
        {"item": "unique_annotation_items", "value": len({row.get("item_id") for row in annotations})},
        {"item": "non_clinician_marker_rows", "value": non_clinician_marker_rows},
        {"item": "needs_human_review_rows", "value": needs_human_review_rows},
        {"item": "role_equals_predicted_rows", "value": role_equals_predicted_rows},
        {"item": "span_equals_predicted_text_rows", "value": span_equals_predicted_rows},
        {"item": "final_adjudicated_span_rows", "value": final_field_rows},
        {"item": "agreement_or_adjudicated_status_rows", "value": adjudication_status_rows},
        {"item": "invalid_or_blank_gold_spans", "value": qc_counts["invalid_or_blank_span"]},
        {"item": "span_text_mismatch_rows", "value": qc_counts["span_text_mismatch"]},
        {"item": "invalid_gold_roles", "value": qc_counts["invalid_role"]},
        {"item": "gold_token_alignment_failures", "value": gold_invalid_token},
        {"item": "pred_token_alignment_failures", "value": pred_invalid_token},
        {"item": "prediction_match_methods", "value": dict(pred_method_counts)},
        {"item": "gold_role_counts", "value": dict(role_counts_gold)},
        {"item": "pred_role_counts", "value": dict(role_counts_pred)},
        {"item": "annotator_counts", "value": dict(annotator_counts)},
    ]
    write_csv(args.output_dir / "parser_error_breakdown.csv", error_rows)

    report_path = args.output_dir / "parser_validation_results.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# ARU Parser Validation Results\n\n")
        handle.write("## Validity Status\n\n")
        if valid_for_rebuttal:
            handle.write("Status: valid for reviewer-facing reporting after manual review.\n\n")
        else:
            handle.write("Status: not valid for reviewer-facing clinician-validation reporting yet.\n\n")
        handle.write("Reasons checked by the script:\n\n")
        handle.write(f"- Non-clinician marker rows: `{non_clinician_marker_rows}`\n")
        handle.write(f"- Rows still marked as needing human review: `{needs_human_review_rows}`\n")
        handle.write(f"- Rows whose role label equals the parser-predicted role: `{role_equals_predicted_rows}` / `{len(annotations)}`\n")
        handle.write(f"- Rows whose span text exactly equals the parser-predicted ARU text: `{span_equals_predicted_rows}` / `{len(annotations)}`\n")
        handle.write(f"- Invalid or blank gold spans: `{qc_counts['invalid_or_blank_span']}`\n")
        handle.write(f"- Invalid role labels: `{qc_counts['invalid_role']}`\n")
        handle.write(f"- Final adjudicated span rows: `{final_field_rows}`\n")
        handle.write(f"- Rows marked agreement/adjudicated: `{adjudication_status_rows}`\n")
        handle.write(f"- Annotator counts: `{dict(annotator_counts)}`\n")
        handle.write("\n## Metrics\n\n")
        handle.write("These metrics are computed for pipeline debugging. If the validity status above is not valid, do not report them as clinician-adjudicated results.\n\n")
        handle.write("| IoU | Span P | Span R | Span F1 | Role Acc | Role Macro-F1 | Final Claim F1 | Mask Eligible F1 |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in metrics_rows:
            handle.write(
                f"| {fmt(row['iou_threshold'])} | {fmt(row['span_precision'])} | {fmt(row['span_recall'])} | "
                f"{fmt(row['span_f1'])} | {fmt(row['role_accuracy'])} | {fmt(row['role_macro_f1'])} | "
                f"{fmt(row['final_claim_f1'])} | {fmt(row['mask_eligible_f1'])} |\n"
            )
        handle.write("\n## Implementation Notes\n\n")
        handle.write("- Token spans use whitespace tokenization because the project tokenizer dependency is unavailable in the current environment.\n")
        handle.write("- Parser predictions do not include original `char_start`/`char_end`; the script aligns predicted ARU text back to `response_text` before scoring.\n")
        handle.write("- Prediction alignment methods are summarized in `parser_error_breakdown.csv`.\n")

    print(report_path)
    print(args.output_dir / "parser_validation_metrics.csv")
    print(args.output_dir / "parser_error_breakdown.csv")
    print("valid_for_rebuttal", valid_for_rebuttal)
    print("metrics", metrics_rows)


if __name__ == "__main__":
    main()
