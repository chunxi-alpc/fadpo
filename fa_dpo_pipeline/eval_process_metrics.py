#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict, List

from common import load_jsonl, output_path, write_jsonl
from experiment_utils import summarize_numeric, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute same-answer process-faithfulness metrics from verifier outputs."
    )
    parser.add_argument(
        "--subset-file",
        default=output_path("process_metrics", "same_answer_subset.jsonl"),
        help="Canonical same-answer subset built by build_same_answer_subset.py.",
    )
    parser.add_argument("--left-score-file", required=True, help="Verifier score file for the left model.")
    parser.add_argument("--right-score-file", required=True, help="Verifier score file for the right model.")
    parser.add_argument(
        "--metrics-output",
        default=output_path("process_metrics", "metrics.csv"),
        help="Where to write aggregate metrics CSV.",
    )
    parser.add_argument(
        "--per-example-output",
        default=output_path("process_metrics", "per_example_metrics.jsonl"),
        help="Where to write per-example metrics JSONL.",
    )
    return parser.parse_args()


def load_score_lookup(path: str) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("id")): row for row in load_jsonl(path)}


def example_metrics(row_id: str, model_id: str, score_row: Dict[str, Any]) -> Dict[str, Any]:
    details = (score_row.get("metrics") or {}).get("node_details", [])
    if not details:
        return {
            "id": row_id,
            "model_id": model_id,
            "total_nodes": 0,
            "support_ratio": 0.0,
            "contradiction_rate": 0.0,
            "unsupported_warrant_ratio": 0.0,
            "claim_inconsistency_rate": 0.0,
        }

    total_nodes = len(details)
    supported = sum(1 for item in details if item.get("decision_label") == "entail")
    contradictory = sum(1 for item in details if item.get("decision_label") == "contradict")
    warrant_nodes = [item for item in details if str(item.get("type")) == "W"]
    claim_nodes = [item for item in details if str(item.get("type")) == "C"]
    unsupported_warrants = sum(1 for item in warrant_nodes if item.get("decision_label") != "entail")
    inconsistent_claims = sum(1 for item in claim_nodes if item.get("decision_label") == "contradict")

    return {
        "id": row_id,
        "model_id": model_id,
        "total_nodes": total_nodes,
        "support_ratio": round(supported / max(total_nodes, 1), 6),
        "contradiction_rate": round(contradictory / max(total_nodes, 1), 6),
        "unsupported_warrant_ratio": round(unsupported_warrants / max(len(warrant_nodes), 1), 6),
        "claim_inconsistency_rate": round(inconsistent_claims / max(len(claim_nodes), 1), 6),
    }


def aggregate_model(rows: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
    metrics = {
        "support_ratio": [float(row["support_ratio"]) for row in rows],
        "contradiction_rate": [float(row["contradiction_rate"]) for row in rows],
        "unsupported_warrant_ratio": [float(row["unsupported_warrant_ratio"]) for row in rows],
        "claim_inconsistency_rate": [float(row["claim_inconsistency_rate"]) for row in rows],
    }
    summary = {"model_id": model_id, "n": len(rows)}
    for name, values in metrics.items():
        stats = summarize_numeric(values)
        summary[f"{name}_mean"] = stats["mean"]
        summary[f"{name}_std"] = stats["std"]
    return summary


def main() -> None:
    args = parse_args()
    subset = load_jsonl(args.subset_file)
    left_lookup = load_score_lookup(args.left_score_file)
    right_lookup = load_score_lookup(args.right_score_file)

    per_example: List[Dict[str, Any]] = []
    left_model_id = subset[0].get("left_model_id", "left_model") if subset else "left_model"
    right_model_id = subset[0].get("right_model_id", "right_model") if subset else "right_model"
    for row in subset:
        row_id = str(row.get("id"))
        if row_id in left_lookup:
            per_example.append(example_metrics(row_id, left_model_id, left_lookup[row_id]))
        if row_id in right_lookup:
            per_example.append(example_metrics(row_id, right_model_id, right_lookup[row_id]))

    write_jsonl(args.per_example_output, per_example)
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in per_example:
        by_model.setdefault(str(row.get("model_id")), []).append(row)
    summary_rows = [aggregate_model(rows, model_id) for model_id, rows in sorted(by_model.items())]
    write_csv(args.metrics_output, summary_rows)

    print(f"Per-example rows: {len(per_example)}")
    print(f"Metrics: {args.metrics_output}")
    print(f"Per-example output: {args.per_example_output}")


if __name__ == "__main__":
    main()
