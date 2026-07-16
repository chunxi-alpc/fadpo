#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fa_dpo_pipeline.common import load_jsonl, normalize_text


FRAGMENT_PATTERNS = (
    re.compile(r"^(?:of|but|and|because|by|after|in the setting of)\b", re.I),
    re.compile(r"^a differential diagnosis\.?$", re.I),
    re.compile(r"^differential diagnoses\.?$", re.I),
    re.compile(r"^differential diagnosis\.?$", re.I),
    re.compile(r"^the findings\.?$", re.I),
    re.compile(r"^the immunohistochemical profile\.?$", re.I),
    re.compile(r"\b(?:was|were|is|are)\.?$", re.I),
)

QUERY_PATTERN_NAMES = (
    "differential diagnosis",
    "clinical features",
    "modality/procedure-heavy",
    "claimish",
    "short<=3",
)

MODALITY_TERMS = (
    "mri",
    "pet/ct",
    "mrcp",
    "ultrasound",
    "biopsy",
    "histology",
    "histopath",
    "immunohist",
    "endoscopy",
    "radiograph",
    "x-ray",
    "xray",
    "fna",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-signal QC regression summary for stage-04/05/06 smoke runs."
    )
    parser.add_argument("--arus-file", required=True, help="Stage-04 ARU JSONL.")
    parser.add_argument("--evidence-file", required=True, help="Stage-05 ARU-with-evidence JSONL.")
    parser.add_argument("--scores-file", required=True, help="Stage-06 score JSONL.")
    parser.add_argument(
        "--high-risk-threshold",
        type=float,
        default=0.9,
        help="Threshold for counting high-risk O nodes.",
    )
    parser.add_argument(
        "--selected-premise-contradiction-threshold",
        type=float,
        default=0.1,
        help="Threshold for considering the selected premise non-contradictory.",
    )
    parser.add_argument(
        "--top-k-samples",
        type=int,
        default=5,
        help="How many highest-risk samples to summarize.",
    )
    parser.add_argument(
        "--max-example-count",
        type=int,
        default=8,
        help="How many examples to print for each heuristic bucket.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Optional file to write the same summary text.",
    )
    return parser.parse_args()


def iter_nodes(rows: Iterable[Dict[str, Any]]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for row in rows:
        for node in row.get("arus") or []:
            if isinstance(node, dict):
                yield row, node


def normalize_node_type(value: Any) -> str:
    return (normalize_text(value).upper()[:1] or "C")


def is_fragmentary_observation(text: str) -> bool:
    clean = normalize_text(text)
    if not clean:
        return False
    lowered = clean.lower()
    return any(pattern.search(clean) for pattern in FRAGMENT_PATTERNS) or lowered in {
        "differential diagnoses.",
        "a differential diagnosis.",
        "the findings.",
        "the immunohistochemical profile.",
    }


def query_pattern_bucket(query: str) -> str | None:
    lowered = normalize_text(query).lower()
    if not lowered:
        return None
    if "differential diagnosis" in lowered:
        return "differential diagnosis"
    if "clinical features" in lowered:
        return "clinical features"
    if (
        "pet/ct" in lowered
        or re.search(r"\bct\b", lowered)
        or re.search(r"\bpet\b", lowered)
        or any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in MODALITY_TERMS if term not in {"pet/ct"})
    ):
        return "modality/procedure-heavy"
    if re.search(r"\b(was|were|is|are)\b", lowered):
        return "claimish"
    if len(lowered.split()) <= 3:
        return "short<=3"
    return None


def preview(text: Any, limit: int = 180) -> str:
    clean = normalize_text(text)
    if len(clean) <= limit:
        return clean
    return clean[: max(limit - 3, 0)].rstrip() + "..."


def main() -> None:
    args = parse_args()

    arus_rows = load_jsonl(args.arus_file)
    evidence_rows = load_jsonl(args.evidence_file)
    score_rows = load_jsonl(args.scores_file)

    arus_by_id = {str(row.get("id")): row for row in arus_rows if row.get("id") is not None}
    evidence_by_id = {str(row.get("id")): row for row in evidence_rows if row.get("id") is not None}
    score_by_id = {str(row.get("id")): row for row in score_rows if row.get("id") is not None}

    common_ids = sorted(set(arus_by_id) & set(evidence_by_id) & set(score_by_id))
    missing = {
        "arus_only": len(set(arus_by_id) - set(common_ids)),
        "evidence_only": len(set(evidence_by_id) - set(common_ids)),
        "scores_only": len(set(score_by_id) - set(common_ids)),
    }

    type_counter = Counter()
    node_mismatch_rows = []
    o_total = 0
    o_high_risk = 0
    o_fragmentary = 0
    o_ctx_conflict = 0
    o_ctx_conflict_nonconflicting_selected = 0
    o_fragment_examples: List[str] = []
    o_ctx_examples: List[str] = []

    query_counts = Counter()
    query_examples: Dict[str, List[str]] = {name: [] for name in QUERY_PATTERN_NAMES}

    for sample_id in common_ids:
        score_row = score_by_id[sample_id]
        evidence_row = evidence_by_id[sample_id]
        score_nodes = score_row.get("metrics", {}).get("node_details") or []
        evidence_nodes = evidence_row.get("arus") or []
        if len(score_nodes) != len(evidence_nodes):
            node_mismatch_rows.append(
                f"{sample_id}: evidence_nodes={len(evidence_nodes)} score_nodes={len(score_nodes)}"
            )

        for node in score_nodes:
            node_type = normalize_node_type(node.get("type"))
            type_counter[node_type] += 1
            if node_type != "O":
                continue
            o_total += 1
            score = float(node.get("score", 0.0))
            if score >= args.high_risk_threshold:
                o_high_risk += 1
            text = normalize_text(node.get("text"))
            if is_fragmentary_observation(text):
                o_fragmentary += 1
                if len(o_fragment_examples) < args.max_example_count:
                    o_fragment_examples.append(f"{sample_id}: {text}")
            tag = normalize_text(node.get("tag"))
            selected_contra = float(node.get("selected_premise_contradiction", 0.0) or 0.0)
            if tag.startswith("CtxConflict"):
                o_ctx_conflict += 1
                if selected_contra < args.selected_premise_contradiction_threshold:
                    o_ctx_conflict_nonconflicting_selected += 1
                    if len(o_ctx_examples) < args.max_example_count:
                        premise = preview(node.get("selected_premise"), 120)
                        o_ctx_examples.append(
                            f"{sample_id}: {text} | selected_premise={premise} | selected_contra={selected_contra:.4f}"
                        )

        for node in evidence_nodes:
            if not isinstance(node, dict):
                continue
            node_type = normalize_node_type(node.get("type"))
            if node_type not in {"W", "D"}:
                continue
            bucket = query_pattern_bucket(node.get("retrieval_query"))
            if bucket is None:
                continue
            query_counts[bucket] += 1
            if len(query_examples[bucket]) < args.max_example_count:
                query_examples[bucket].append(
                    f"{sample_id}: {preview(node.get('retrieval_query'), 140)} | {preview(node.get('text'), 100)}"
                )

    top_samples = sorted(
        (score_by_id[sample_id] for sample_id in common_ids),
        key=lambda row: float(row.get("unfaithfulness_score", 0.0)),
        reverse=True,
    )[: max(args.top_k_samples, 0)]

    lines: List[str] = []
    lines.append("QC Regression Summary")
    lines.append(f"Files: {args.arus_file}, {args.evidence_file}, {args.scores_file}")
    lines.append(
        f"Rows aligned: {len(common_ids)} | arus_only={missing['arus_only']} "
        f"evidence_only={missing['evidence_only']} scores_only={missing['scores_only']}"
    )
    if node_mismatch_rows:
        lines.append(f"Node count mismatches: {len(node_mismatch_rows)}")
        for item in node_mismatch_rows[: min(len(node_mismatch_rows), args.max_example_count)]:
            lines.append(f"  - {item}")
    else:
        lines.append("Node count mismatches: 0")

    lines.append(
        "Node counts: "
        + ", ".join(f"{key}={type_counter.get(key, 0)}" for key in ["O", "C", "W", "D"])
    )
    lines.append(
        f"High-risk O nodes (score>={args.high_risk_threshold}): {o_high_risk}/{o_total}"
    )
    lines.append(f"Fragmentary O nodes: {o_fragmentary}/{o_total}")
    for item in o_fragment_examples:
        lines.append(f"  - {item}")
    lines.append(
        "CtxConflict O with non-conflicting selected premise "
        f"(selected_premise_contradiction<{args.selected_premise_contradiction_threshold}): "
        f"{o_ctx_conflict_nonconflicting_selected}/{o_ctx_conflict}"
    )
    for item in o_ctx_examples:
        lines.append(f"  - {item}")
    lines.append("D/W query patterns:")
    for name in QUERY_PATTERN_NAMES:
        lines.append(f"  - {name}: {query_counts.get(name, 0)}")
        for item in query_examples[name]:
            lines.append(f"    * {item}")

    lines.append("Top risky samples:")
    for row in top_samples:
        sample_id = str(row.get("id"))
        lines.append(
            f"  - {sample_id}: score={float(row.get('unfaithfulness_score', 0.0)):.4f} "
            f"margin={float(row.get('dpo_margin', 0.0)):.4f}"
        )
        nodes = row.get("metrics", {}).get("node_details") or []
        top_nodes = sorted(nodes, key=lambda node: float(node.get("score", 0.0)), reverse=True)[:3]
        for node in top_nodes:
            lines.append(
                f"    * {normalize_node_type(node.get('type'))} "
                f"{float(node.get('score', 0.0)):.4f} {normalize_text(node.get('tag'))} | "
                f"{preview(node.get('text'), 120)}"
            )

    summary = "\n".join(lines) + "\n"
    print(summary, end="")
    if args.output_file:
        Path(args.output_file).expanduser().write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
