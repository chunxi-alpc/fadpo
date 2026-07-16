#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from common import load_jsonl, normalize_text, output_path, result_path
from experiment_utils import match_any_text, parse_list_field, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality for verifier-side evidence routing."
    )
    parser.add_argument(
        "--aru-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.strict_clean.arus.with_evidence.jsonl"),
        help="ARU file with retrieval candidates from stage 05.",
    )
    parser.add_argument(
        "--score-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.strict_clean.scores.jsonl"),
        help="Optional verifier score file from stage 06 for silver support coverage.",
    )
    parser.add_argument(
        "--targets-file",
        default="",
        help="Optional gold target CSV/JSONL with id, aru_id, and gold_text / gold_doc_id.",
    )
    parser.add_argument(
        "--output-file",
        default=output_path("retrieval_protocol", "retrieval_quality.csv"),
        help="Where to write the retrieval quality summary CSV.",
    )
    return parser.parse_args()


def build_aru_lookup(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id"))
        for index, node in enumerate(row.get("arus") or []):
            lookup[(row_id, index)] = node
    return lookup


def summarize_group(rows: List[Dict[str, Any]], *, value_key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get(value_key, 0.0)) for row in rows) / len(rows), 6)


def load_targets(path: str) -> List[Dict[str, Any]]:
    if not path:
        return []
    if path.endswith(".jsonl"):
        return load_jsonl(path)
    if path.endswith(".csv"):
        import csv

        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported targets file: {path}")


def candidate_texts(node: Dict[str, Any], k: int) -> List[str]:
    candidates = node.get("retrieval_candidates") or []
    if candidates:
        return [normalize_text(item.get("text")) for item in candidates[:k] if normalize_text(item.get("text"))]
    docs = node.get("retrieved_evidence") or []
    return [normalize_text(item) for item in docs[:k] if normalize_text(item)]


def gold_target_texts(row: Dict[str, Any]) -> List[str]:
    targets = []
    targets.extend(parse_list_field(row.get("gold_text")))
    targets.extend(parse_list_field(row.get("gold_substrings")))
    return [normalize_text(item) for item in targets if normalize_text(item)]


def gold_target_ids(row: Dict[str, Any]) -> List[str]:
    return [normalize_text(item) for item in parse_list_field(row.get("gold_doc_id")) if normalize_text(item)]


def compute_gold_matches(aru_lookup: Dict[Tuple[str, int], Dict[str, Any]], targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in targets:
        key = (str(row.get("id")), int(row.get("aru_id", 0)))
        node = aru_lookup.get(key)
        if node is None:
            continue
        candidates = node.get("retrieval_candidates") or []
        target_ids = set(gold_target_ids(row))
        target_texts = gold_target_texts(row)

        def match_at(k: int) -> float:
            top_candidates = candidates[:k]
            if target_ids:
                candidate_ids = {
                    normalize_text(item.get("corpus_id")) or normalize_text(item.get("doc_id")) for item in top_candidates
                }
                if candidate_ids & target_ids:
                    return 1.0
            if target_texts and match_any_text(candidate_texts(node, k), target_texts, min_score=0.75):
                return 1.0
            return 0.0

        route = normalize_text(row.get("route")) or str(node.get("type", "")).upper()[:1] or "unknown"
        rows.append(
            {
                "route": route,
                "hit_at_5": match_at(5),
                "hit_at_20": match_at(20),
            }
        )
    return rows


def compute_silver_matches(
    aru_lookup: Dict[Tuple[str, int], Dict[str, Any]],
    score_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for score_row in score_rows:
        row_id = str(score_row.get("id"))
        for detail in (score_row.get("metrics") or {}).get("node_details", []):
            route = normalize_text(detail.get("route"))
            if route not in {"retrieved_evidence", "observation_fallback"}:
                continue
            key = (row_id, int(detail.get("id", 0)))
            node = aru_lookup.get(key, {})
            candidate_count = len(node.get("retrieval_candidates") or node.get("retrieved_evidence") or [])
            rows.append(
                {
                    "route": route,
                    "supported": 1.0 if normalize_text(detail.get("decision_label")) == "entail" else 0.0,
                    "candidate_count": candidate_count,
                }
            )
    return rows


def summary_rows(gold_rows: List[Dict[str, Any]], silver_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for row in gold_rows:
        grouped[("gold", str(row.get("route")))].append(row)
        grouped[("gold", "overall")].append(row)
    for row in silver_rows:
        grouped[("silver", str(row.get("route")))].append(row)
        grouped[("silver", "overall")].append(row)

    summary: List[Dict[str, Any]] = []
    for (mode, route), rows in sorted(grouped.items()):
        record = {
            "evaluation_mode": mode,
            "route": route,
            "n": len(rows),
        }
        if mode == "gold":
            record["hit_at_5"] = summarize_group(rows, value_key="hit_at_5")
            record["hit_at_20"] = summarize_group(rows, value_key="hit_at_20")
        else:
            record["support_coverage"] = summarize_group(rows, value_key="supported")
            record["avg_candidate_count"] = round(
                sum(float(item.get("candidate_count", 0.0)) for item in rows) / max(len(rows), 1), 6
            )
        summary.append(record)
    return summary


def main() -> None:
    args = parse_args()
    aru_rows = load_jsonl(args.aru_file)
    aru_lookup = build_aru_lookup(aru_rows)
    score_rows = load_jsonl(args.score_file) if args.score_file else []
    gold_rows = compute_gold_matches(aru_lookup, load_targets(args.targets_file)) if args.targets_file else []
    silver_rows = compute_silver_matches(aru_lookup, score_rows) if score_rows else []
    rows = summary_rows(gold_rows, silver_rows)
    write_csv(args.output_file, rows)
    print(f"Summary rows: {len(rows)}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
