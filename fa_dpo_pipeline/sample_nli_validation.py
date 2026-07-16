#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from common import load_jsonl, normalize_text, output_path, result_path, write_json, write_jsonl
from experiment_utils import write_csv


ROUTE_ORDER = ("observation_context", "warrant_knowledge", "differentiation_combined", "claim_history")
LABEL_ORDER = ("entail", "neutral", "contradict", "unknown")
CONFIDENCE_ORDER = ("high", "medium", "low")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample stratified ARU-evidence pairs for manual NLI verifier validation."
    )
    parser.add_argument(
        "--aru-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.strict_clean.arus.with_evidence.jsonl"),
        help="ARU file with attached evidence from stage 05.",
    )
    parser.add_argument(
        "--score-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.strict_clean.scores.jsonl"),
        help="Optional verifier score file from stage 06.",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=150,
        help="Total number of ARU-evidence pairs to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--calibration-ratio",
        type=float,
        default=0.25,
        help="Fraction of sampled rows assigned to the calibration split.",
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.8,
        help="Lower bound for the `high` confidence band.",
    )
    parser.add_argument(
        "--medium-confidence-threshold",
        type=float,
        default=0.6,
        help="Lower bound for the `medium` confidence band.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=output_path("nli_validation", "sample.jsonl"),
        help="Where to write the sampled JSONL.",
    )
    parser.add_argument(
        "--output-csv",
        default=output_path("nli_validation", "annotation.csv"),
        help="Where to write the annotation CSV.",
    )
    parser.add_argument(
        "--manifest-jsonl",
        default=output_path("nli_validation", "sample_manifest.jsonl"),
        help="Where to write the compact sample manifest JSONL.",
    )
    parser.add_argument(
        "--quota-report-json",
        default=output_path("nli_validation", "sampling_quota.json"),
        help="Where to write the explicit sampling-quota report JSON.",
    )
    return parser.parse_args()


def load_score_lookup(path: str) -> Dict[str, Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    rows = load_jsonl(file_path)
    return {str(row.get("id")): row for row in rows}


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def route_bucket(node_type: str) -> str:
    if node_type == "O":
        return "observation_context"
    if node_type == "W":
        return "warrant_knowledge"
    if node_type == "D":
        return "differentiation_combined"
    return "claim_history"


def route_sort_key(route: str) -> tuple[int, str]:
    try:
        return (ROUTE_ORDER.index(route), route)
    except ValueError:
        return (len(ROUTE_ORDER), route)


def label_sort_key(label: str) -> tuple[int, str]:
    try:
        return (LABEL_ORDER.index(label), label)
    except ValueError:
        return (len(LABEL_ORDER), label)


def confidence_sort_key(label: str) -> tuple[int, str]:
    try:
        return (CONFIDENCE_ORDER.index(label), label)
    except ValueError:
        return (len(CONFIDENCE_ORDER), label)


def confidence_band(score: float, args: argparse.Namespace) -> str:
    if score >= args.high_confidence_threshold:
        return "high"
    if score >= args.medium_confidence_threshold:
        return "medium"
    return "low"


def difficulty_band(confidence: float, args: argparse.Namespace) -> str:
    if confidence >= args.high_confidence_threshold:
        return "easy"
    if confidence >= args.medium_confidence_threshold:
        return "medium"
    return "hard"


def normalized_predicted_label(value: str) -> str:
    text = normalize_text(value).lower()
    if text in {"entail", "neutral", "contradict"}:
        return text
    return "unknown"


def fallback_evidence(row: Dict[str, Any], arus: List[Dict[str, Any]], node_index: int, node_type: str, node: Dict[str, Any]) -> Dict[str, str]:
    if node_type in {"W", "D"}:
        docs = node.get("retrieved_evidence") or []
        if isinstance(docs, list) and docs:
            return {"route": "retrieved_evidence", "evidence_text": normalize_text(docs[0])}
        return {"route": "retrieved_evidence", "evidence_text": ""}
    if node_type == "O":
        return {"route": "patient_context", "evidence_text": normalize_text(row.get("patient_context"))}
    history = " ".join(normalize_text(item.get("text")) for item in arus[:node_index] if normalize_text(item.get("text")))
    return {"route": "reasoning_history", "evidence_text": history}


def build_candidates(
    aru_rows: List[Dict[str, Any]],
    score_lookup: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for row in aru_rows:
        row_id = str(row.get("id"))
        score_row = score_lookup.get(row_id, {})
        detail_lookup = {
            int(item.get("id", -1)): item for item in (score_row.get("metrics", {}) or {}).get("node_details", []) if item.get("id") is not None
        }
        arus = row.get("arus") or []
        for index, node in enumerate(arus):
            node_text = normalize_text(node.get("text"))
            node_type = str(node.get("type", "C")).upper()[:1] or "C"
            if not node_text:
                continue
            detail = detail_lookup.get(index, {})
            fallback = fallback_evidence(row, arus, index, node_type, node)
            evidence_text = normalize_text(detail.get("selected_premise")) or fallback["evidence_text"]
            probabilities = {
                "entail": safe_float(detail.get("entailment")),
                "neutral": safe_float(detail.get("neutral")),
                "contradict": safe_float(detail.get("contradiction")),
            }
            predicted_label = normalize_text(detail.get("decision_label")) or normalize_text(detail.get("argmax_label"))
            if not predicted_label and any(probabilities.values()):
                predicted_label = max(probabilities, key=probabilities.get)
            predicted_label = normalized_predicted_label(predicted_label)
            confidence = max(probabilities.values()) if probabilities else 0.0
            conf_band = confidence_band(confidence, args)
            candidates.append(
                {
                    "sample_id": f"{row_id}::aru{index}",
                    "id": row_id,
                    "source_id": row.get("source_id"),
                    "error_symbol": row.get("error_symbol"),
                    "aru_id": index,
                    "aru_role": node_type,
                    "route_bucket": route_bucket(node_type),
                    "aru_text": node_text,
                    "verifier_route": normalize_text(detail.get("route")) or fallback["route"],
                    "evidence_text": evidence_text,
                    "patient_context": normalize_text(row.get("patient_context")),
                    "predicted_label": predicted_label,
                    "argmax_label": normalize_text(detail.get("argmax_label")),
                    "entailment": probabilities["entail"],
                    "neutral": probabilities["neutral"],
                    "contradiction": probabilities["contradict"],
                    "predicted_confidence": round(confidence, 6),
                    "confidence_band": conf_band,
                    "predicted_confidence_band": conf_band,
                    "difficulty_band": difficulty_band(confidence, args),
                    "is_terminal_claim": bool(node_type == "C" and index == len(arus) - 1),
                    "score": detail.get("score", ""),
                }
            )
    return candidates


def proportional_targets(
    total: int,
    capacities: Dict[Any, int],
    ordered_keys: Sequence[Any],
    *,
    min_per_non_empty: int = 0,
) -> Dict[Any, int]:
    targets = {key: 0 for key in capacities}
    if total <= 0:
        return targets

    eligible = [key for key in ordered_keys if capacities.get(key, 0) > 0]
    if not eligible:
        return targets

    remaining = total
    if min_per_non_empty > 0:
        for key in eligible:
            if remaining <= 0:
                break
            base = min(min_per_non_empty, capacities[key])
            targets[key] += base
            remaining -= base

    if remaining <= 0:
        return targets

    residual_capacity = {key: max(capacities[key] - targets[key], 0) for key in eligible}
    residual_total = sum(residual_capacity.values())
    if residual_total <= 0:
        return targets

    remainders: List[Tuple[float, int, int, Any]] = []
    for index, key in enumerate(eligible):
        exact = remaining * residual_capacity[key] / residual_total
        whole = min(int(exact), residual_capacity[key])
        targets[key] += whole
        remainders.append((exact - whole, residual_capacity[key], -index, key))

    leftover = total - sum(targets.values())
    for _, _, _, key in sorted(remainders, reverse=True):
        if leftover <= 0:
            break
        if targets[key] >= capacities[key]:
            continue
        targets[key] += 1
        leftover -= 1
    return targets


def shuffle_grouped_rows(rows: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    shuffled = rows[:]
    rng.shuffle(shuffled)
    return shuffled


def pop_rows(rows: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    selected = rows[:count]
    del rows[:count]
    return selected


def label_key(row: Dict[str, Any]) -> str:
    return normalized_predicted_label(str(row.get("predicted_label")))


def subgroup_key(row: Dict[str, Any]) -> tuple[str, str]:
    return (
        normalize_text(row.get("predicted_confidence_band")) or normalize_text(row.get("confidence_band")) or "unknown",
        "terminal" if bool(row.get("is_terminal_claim")) else "non_terminal",
    )


def stratified_sample(rows: List[Dict[str, Any]], total_samples: int, seed: int) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    by_route: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_route[str(row.get("route_bucket"))].append(row)

    routes = [route for route in ROUTE_ORDER if by_route.get(route)]
    if not routes:
        return [], {"requested_total": total_samples, "sampled_total": 0, "route_targets": {}, "label_targets": {}, "subgroup_targets": {}}

    sampled: List[Dict[str, Any]] = []
    route_capacities = {route: len(by_route[route]) for route in routes}
    route_targets = proportional_targets(total_samples, route_capacities, routes, min_per_non_empty=1)
    label_target_report: Dict[str, Dict[str, int]] = {}
    subgroup_target_report: Dict[str, Dict[str, int]] = {}

    for route in routes:
        route_rows = shuffle_grouped_rows(by_route[route], rng)
        route_target = route_targets.get(route, 0)
        if route_target <= 0:
            label_target_report[route] = {}
            subgroup_target_report[route] = {}
            continue

        by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in route_rows:
            by_label[label_key(row)].append(row)
        labels = [label for label in LABEL_ORDER if by_label.get(label)]
        label_capacities = {label: len(by_label[label]) for label in labels}
        label_targets = proportional_targets(route_target, label_capacities, labels, min_per_non_empty=1 if route_target >= len(labels) else 0)
        label_target_report[route] = label_targets
        subgroup_target_report[route] = {}

        for label in labels:
            label_rows = shuffle_grouped_rows(by_label[label], rng)
            label_target = label_targets.get(label, 0)
            by_subgroup: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
            for row in label_rows:
                by_subgroup[subgroup_key(row)].append(row)
            subgroup_order = sorted(by_subgroup, key=lambda item: (confidence_sort_key(item[0]), item[1]))
            subgroup_capacities = {key: len(by_subgroup[key]) for key in subgroup_order}
            subgroup_targets = proportional_targets(label_target, subgroup_capacities, subgroup_order)
            for subgroup in subgroup_order:
                subgroup_name = f"{label}|{subgroup[0]}|{subgroup[1]}"
                subgroup_target_report[route][subgroup_name] = subgroup_targets.get(subgroup, 0)
                subgroup_rows = shuffle_grouped_rows(by_subgroup[subgroup], rng)
                sampled.extend(pop_rows(subgroup_rows, subgroup_targets.get(subgroup, 0)))

    sampled_ids = {row["sample_id"] for row in sampled}
    remaining = max(total_samples - len(sampled), 0)
    residue = shuffle_grouped_rows([row for row in rows if row["sample_id"] not in sampled_ids], rng)
    sampled.extend(pop_rows(residue, remaining))
    sampled.sort(key=lambda item: (item["route_bucket"], item["aru_role"], item["sample_id"]))
    quota_report = {
        "requested_total": total_samples,
        "sampled_total": len(sampled[:total_samples]),
        "route_targets": route_targets,
        "label_targets": label_target_report,
        "subgroup_targets": subgroup_target_report,
    }
    return sampled[:total_samples], quota_report


def assign_calibration_split(sampled: List[Dict[str, Any]], ratio: float, seed: int) -> None:
    if ratio <= 0.0 or len(sampled) <= 1:
        for row in sampled:
            row["calibration_split"] = "evaluation"
        return

    target_total = int(round(len(sampled) * ratio))
    target_total = max(1, min(target_total, len(sampled) - 1))
    rng = random.Random(seed)
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in sampled:
        grouped[(str(row.get("route_bucket")), str(row.get("predicted_label")))].append(row)

    calibration_rows: List[Dict[str, Any]] = []
    for key, group_rows in grouped.items():
        rng.shuffle(group_rows)
        if len(group_rows) <= 1:
            continue
        target = int(round(len(group_rows) * ratio))
        target = max(target, 1)
        target = min(target, len(group_rows) - 1)
        calibration_rows.extend(group_rows[:target])

    if len(calibration_rows) > target_total:
        rng.shuffle(calibration_rows)
        calibration_rows = calibration_rows[:target_total]
    elif len(calibration_rows) < target_total:
        selected_ids = {row["sample_id"] for row in calibration_rows}
        fallback = [row for row in sampled if row["sample_id"] not in selected_ids]
        rng.shuffle(fallback)
        calibration_rows.extend(fallback[: target_total - len(calibration_rows)])

    calibration_ids = {row["sample_id"] for row in calibration_rows}

    for row in sampled:
        row["calibration_split"] = "calibration" if row["sample_id"] in calibration_ids else "evaluation"


def annotation_rows(sampled: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in sampled:
        rows.append(
            {
                "sample_id": row["sample_id"],
                "id": row["id"],
                "source_id": row.get("source_id", ""),
                "error_symbol": row.get("error_symbol", ""),
                "aru_id": row["aru_id"],
                "aru_role": row["aru_role"],
                "route_bucket": row.get("route_bucket", ""),
                "verifier_route": row["verifier_route"],
                "aru_text": row["aru_text"],
                "evidence_text": row["evidence_text"],
                "predicted_label": row.get("predicted_label", ""),
                "predicted_confidence": row.get("predicted_confidence", ""),
                "confidence_band": row.get("confidence_band", ""),
                "predicted_confidence_band": row.get("predicted_confidence_band", ""),
                "difficulty_band": row.get("difficulty_band", ""),
                "entailment": row.get("entailment", ""),
                "neutral": row.get("neutral", ""),
                "contradiction": row.get("contradiction", ""),
                "is_terminal_claim": row.get("is_terminal_claim", False),
                "calibration_split": row.get("calibration_split", "evaluation"),
                "human_label": "",
                "notes": "",
            }
        )
    return rows


def manifest_rows(sampled: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in sampled:
        rows.append(
            {
                "sample_id": row["sample_id"],
                "source_id": row.get("source_id", ""),
                "aru_id": row.get("aru_id", ""),
                "aru_role": row.get("aru_role", ""),
                "verifier_route": row.get("verifier_route", ""),
                "route_bucket": row.get("route_bucket", ""),
                "predicted_label": row.get("predicted_label", ""),
                "predicted_confidence_band": row.get("predicted_confidence_band", ""),
                "difficulty_band": row.get("difficulty_band", ""),
                "is_terminal_claim": row.get("is_terminal_claim", False),
                "calibration_split": row.get("calibration_split", "evaluation"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    aru_rows = load_jsonl(args.aru_file)
    score_lookup = load_score_lookup(args.score_file)
    candidates = build_candidates(aru_rows, score_lookup, args)
    sampled, quota_report = stratified_sample(candidates, args.total_samples, args.seed)
    assign_calibration_split(sampled, args.calibration_ratio, args.seed)
    write_jsonl(args.output_jsonl, sampled)
    write_jsonl(args.manifest_jsonl, manifest_rows(sampled))
    write_json(args.quota_report_json, quota_report)
    write_csv(
        args.output_csv,
        annotation_rows(sampled),
        fieldnames=[
            "sample_id",
            "id",
            "source_id",
            "error_symbol",
            "aru_id",
            "aru_role",
            "route_bucket",
            "verifier_route",
            "aru_text",
            "evidence_text",
            "predicted_label",
            "predicted_confidence",
            "confidence_band",
            "predicted_confidence_band",
            "difficulty_band",
            "entailment",
            "neutral",
            "contradiction",
            "is_terminal_claim",
            "calibration_split",
            "human_label",
            "notes",
        ],
    )
    print(f"Candidates: {len(candidates)}")
    print(f"Sampled rows: {len(sampled)}")
    print(f"JSONL: {args.output_jsonl}")
    print(f"Manifest: {args.manifest_jsonl}")
    print(f"Quota report: {args.quota_report_json}")
    print(f"CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
