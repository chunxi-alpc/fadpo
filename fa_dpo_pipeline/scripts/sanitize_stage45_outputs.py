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

from fa_dpo_pipeline.common import load_jsonl, normalize_text, write_json, write_jsonl


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
        description="Targeted salvage for stage-04/05 outputs before stage 06."
    )
    parser.add_argument(
        "--arus-file",
        required=True,
        help="Stage-04 ARU JSONL.",
    )
    parser.add_argument(
        "--evidence-file",
        required=True,
        help="Stage-05 ARU-with-evidence JSONL.",
    )
    parser.add_argument(
        "--output-arus-file",
        default="",
        help="Where to write sanitized stage-04 output. If empty, no file is written.",
    )
    parser.add_argument(
        "--output-evidence-file",
        default="",
        help="Where to write sanitized stage-05 output. If empty, no file is written.",
    )
    parser.add_argument(
        "--output-summary-file",
        default="",
        help="Optional JSON summary output.",
    )
    parser.add_argument(
        "--mode",
        choices=["salvage", "filter"],
        default="salvage",
        help="salvage: rewrite/drop bad nodes; filter: only keep clean samples.",
    )
    parser.add_argument(
        "--min-clean-ratio",
        type=float,
        default=0.92,
        help="Minimum clean-node ratio for keeping a sample in filter mode.",
    )
    parser.add_argument(
        "--max-fragment-o-ratio",
        type=float,
        default=0.04,
        help="Maximum allowed fragmentary-O ratio for keeping a sample in filter mode.",
    )
    parser.add_argument(
        "--max-bad-query-ratio",
        type=float,
        default=0.18,
        help="Maximum allowed bad-query ratio among W/D nodes for keeping a sample in filter mode.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only print summary without writing outputs.",
    )
    parser.add_argument(
        "--max-example-count",
        type=int,
        default=8,
        help="How many example rows to show per bucket.",
    )
    return parser.parse_args()


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


def node_type(node: Dict[str, Any]) -> str:
    return (normalize_text(node.get("type")).upper()[:1] or "C")


def clean_query_value(value: Any) -> str:
    return normalize_text(value)


def salvage_o_node(node: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str]:
    text = normalize_text(node.get("text"))
    if not text:
        return node, True, "empty_o"
    if is_fragmentary_observation(text):
        node = dict(node)
        node["type"] = "C"
        node["retrieval_query"] = ""
        node["rerank_query"] = ""
        node["retrieval_query_source"] = ""
        node["retrieval_query_variants"] = []
        node["retrieval_candidates"] = []
        node["retrieved_evidence"] = []
        return node, True, "fragment_o"
    return node, False, ""


def salvage_wd_node(node: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str]:
    node = dict(node)
    query = clean_query_value(node.get("retrieval_query"))
    bucket = query_pattern_bucket(query)
    if bucket is not None:
        node["query_quality"] = bucket
        node["query_needs_review"] = True
        return node, True, bucket
    node["query_quality"] = "ok"
    node["query_needs_review"] = False
    return node, False, "ok"


def sanitize_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    row = dict(row)
    nodes = row.get("arus") or []
    if not isinstance(nodes, list):
        nodes = []

    stats = Counter()
    sanitized_nodes: List[Dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        kind = node_type(node)
        if kind == "O":
            stats["o_total"] += 1
            fixed, changed, reason = salvage_o_node(node)
            if changed:
                stats[f"o_{reason}"] += 1
            sanitized_nodes.append(fixed)
            continue
        if kind in {"W", "D"}:
            stats[f"{kind.lower()}_total"] += 1
            fixed, changed, reason = salvage_wd_node(node)
            if changed:
                stats[f"wd_{reason}"] += 1
            sanitized_nodes.append(fixed)
            continue
        stats["c_total"] += 1
        sanitized_nodes.append(node)

    row["arus"] = sanitized_nodes
    row["sanitization"] = {
        "o_total": stats["o_total"],
        "o_fragment_o": stats["o_fragment_o"],
        "wd_total": stats["w_total"] + stats["d_total"],
        "wd_query_needs_review": sum(v for k, v in stats.items() if k.startswith("wd_") and k != "wd_total"),
    }
    return row, dict(stats)


def sample_is_clean(row: Dict[str, Any], min_clean_ratio: float, max_fragment_ratio: float, max_bad_query_ratio: float) -> Tuple[bool, Dict[str, Any]]:
    nodes = row.get("arus") or []
    total = 0
    clean = 0
    frag_o = 0
    wd_total = 0
    bad_query = 0
    bad_query_examples: List[str] = []

    for node in nodes:
        if not isinstance(node, dict):
            continue
        kind = node_type(node)
        total += 1
        text = normalize_text(node.get("text"))
        if kind == "O" and is_fragmentary_observation(text):
            frag_o += 1
        elif kind == "O":
            clean += 1
        elif kind in {"W", "D"}:
            wd_total += 1
            q = clean_query_value(node.get("retrieval_query"))
            bucket = query_pattern_bucket(q)
            if bucket is not None:
                bad_query += 1
                if len(bad_query_examples) < 4:
                    bad_query_examples.append(f"{kind}: {bucket}: {q}")
            else:
                clean += 1
        else:
            clean += 1

    clean_ratio = clean / max(total, 1)
    frag_ratio = frag_o / max(total, 1)
    bad_query_ratio = bad_query / max(wd_total, 1) if wd_total else 0.0
    keep = (
        clean_ratio >= min_clean_ratio
        and frag_ratio <= max_fragment_ratio
        and bad_query_ratio <= max_bad_query_ratio
    )
    return keep, {
        "total": total,
        "clean": clean,
        "frag_o": frag_o,
        "wd_total": wd_total,
        "bad_query": bad_query,
        "clean_ratio": round(clean_ratio, 4),
        "frag_ratio": round(frag_ratio, 4),
        "bad_query_ratio": round(bad_query_ratio, 4),
        "bad_query_examples": bad_query_examples,
    }


def main() -> None:
    args = parse_args()

    arus_rows = load_jsonl(args.arus_file)
    evidence_rows = load_jsonl(args.evidence_file)
    arus_by_id = {str(row.get("id")): row for row in arus_rows if row.get("id") is not None}
    evidence_by_id = {str(row.get("id")): row for row in evidence_rows if row.get("id") is not None}
    common_ids = sorted(set(arus_by_id) & set(evidence_by_id))

    sanitized_arus: List[Dict[str, Any]] = []
    sanitized_evidence: List[Dict[str, Any]] = []
    keep_ids: List[str] = []
    row_stats: List[Dict[str, Any]] = []
    counters = Counter()
    example_kept: List[str] = []
    example_dropped: List[str] = []

    for row_id in common_ids:
        arus_row, arus_counts = sanitize_row(arus_by_id[row_id])
        evidence_row, evidence_counts = sanitize_row(evidence_by_id[row_id])

        keep, sample_stats = sample_is_clean(
            evidence_row,
            min_clean_ratio=args.min_clean_ratio,
            max_fragment_ratio=args.max_fragment_o_ratio,
            max_bad_query_ratio=args.max_bad_query_ratio,
        )

        sample_stats["id"] = row_id
        row_stats.append(sample_stats)

        if args.mode == "salvage":
            sanitized_arus.append(arus_row)
            sanitized_evidence.append(evidence_row)
            keep_ids.append(row_id)
            if len(example_kept) < args.max_example_count:
                example_kept.append(
                    f"{row_id}: clean_ratio={sample_stats['clean_ratio']:.4f} frag={sample_stats['frag_ratio']:.4f} bad_query={sample_stats['bad_query_ratio']:.4f}"
                )
        else:
            if keep:
                sanitized_arus.append(arus_row)
                sanitized_evidence.append(evidence_row)
                keep_ids.append(row_id)
                counters["kept_rows"] += 1
                if len(example_kept) < args.max_example_count:
                    example_kept.append(
                        f"{row_id}: clean_ratio={sample_stats['clean_ratio']:.4f} frag={sample_stats['frag_ratio']:.4f} bad_query={sample_stats['bad_query_ratio']:.4f}"
                    )
            else:
                counters["dropped_rows"] += 1
                if len(example_dropped) < args.max_example_count:
                    example_dropped.append(
                        f"{row_id}: clean_ratio={sample_stats['clean_ratio']:.4f} frag={sample_stats['frag_ratio']:.4f} bad_query={sample_stats['bad_query_ratio']:.4f}"
                    )

        counters["rows"] += 1
        counters["o_total"] += arus_counts.get("o_total", 0)
        counters["o_fragment_o"] += arus_counts.get("o_fragment_o", 0)
        counters["wd_total"] += arus_counts.get("w_total", 0) + arus_counts.get("d_total", 0)
        counters["wd_flagged"] += sum(v for k, v in evidence_counts.items() if k.startswith("wd_") and k != "wd_total")

    summary = {
        "mode": args.mode,
        "source_rows": len(common_ids),
        "written_rows": len(keep_ids),
        "dropped_rows": len(common_ids) - len(keep_ids),
        "o_total": counters["o_total"],
        "o_fragment_o": counters["o_fragment_o"],
        "wd_total": counters["wd_total"],
        "wd_flagged": counters["wd_flagged"],
        "min_clean_ratio": args.min_clean_ratio,
        "max_fragment_o_ratio": args.max_fragment_o_ratio,
        "max_bad_query_ratio": args.max_bad_query_ratio,
        "kept_examples": example_kept,
        "dropped_examples": example_dropped,
    }

    lines = [
        "Stage 04/05 Sanitization Summary",
        f"Mode: {args.mode}",
        f"Rows aligned: {len(common_ids)}",
        f"Written rows: {len(keep_ids)}",
        f"Dropped rows: {len(common_ids) - len(keep_ids)}",
        f"O nodes total: {summary['o_total']}",
        f"Fragmentary O nodes flagged: {summary['o_fragment_o']}",
        f"W/D nodes total: {summary['wd_total']}",
        f"W/D nodes flagged for query review: {summary['wd_flagged']}",
        f"Thresholds: min_clean_ratio={args.min_clean_ratio} max_fragment_o_ratio={args.max_fragment_o_ratio} max_bad_query_ratio={args.max_bad_query_ratio}",
        "Kept examples:",
    ]
    lines.extend(f"  - {item}" for item in example_kept[: args.max_example_count])
    if args.mode == "filter":
        lines.append("Dropped examples:")
        lines.extend(f"  - {item}" for item in example_dropped[: args.max_example_count])

    summary_text = "\n".join(lines) + "\n"
    print(summary_text, end="")

    if args.dry_run:
        return

    if args.output_arus_file:
        write_jsonl(args.output_arus_file, sanitized_arus)
    if args.output_evidence_file:
        write_jsonl(args.output_evidence_file, sanitized_evidence)
    if args.output_summary_file:
        write_json(args.output_summary_file, summary)


if __name__ == "__main__":
    main()
