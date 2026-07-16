#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from random import Random
from typing import Any, Dict, Iterable, Iterator, List

from common import normalize_text, output_path, result_path, write_json, write_jsonl
from experiment_utils import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-sample W/D retrieval examples from a large stage-05 JSONL for manual or LLM review."
    )
    parser.add_argument(
        "--aru-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.strict_clean.arus.with_evidence.jsonl"),
        help="Stage-05 ARU JSONL with attached evidence.",
    )
    parser.add_argument(
        "--roles",
        default="W,D",
        help="Comma-separated ARU roles to sample. Default: `W,D`.",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=120,
        help="Total examples to sample across buckets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible reservoir sampling.",
    )
    parser.add_argument(
        "--max-evidence-per-node",
        type=int,
        default=3,
        help="How many retained evidence texts to keep per sampled node.",
    )
    parser.add_argument(
        "--max-candidates-per-node",
        type=int,
        default=5,
        help="How many retrieval candidates to keep per sampled node.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=1200,
        help="How many patient-context characters to keep in the sample payload.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=700,
        help="How many characters to keep for ARU/evidence/candidate text fields.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=output_path("retrieval_review", "sample.jsonl"),
        help="Where to write the sampled JSONL.",
    )
    parser.add_argument(
        "--output-csv",
        default=output_path("retrieval_review", "sample.csv"),
        help="Where to write the flattened CSV.",
    )
    parser.add_argument(
        "--output-summary",
        default=output_path("retrieval_review", "summary.json"),
        help="Where to write the summary JSON.",
    )
    return parser.parse_args()


def truncate_text(text: Any, limit: int) -> str:
    clean = normalize_text(text)
    if limit <= 0 or len(clean) <= limit:
        return clean
    return clean[: max(limit - 3, 0)].rstrip() + "..."


def parse_roles(value: str) -> List[str]:
    roles = []
    for item in normalize_text(value).split(","):
        role = item.strip().upper()[:1]
        if role and role not in roles:
            roles.append(role)
    return roles


def evidence_texts(value: Any) -> List[str]:
    if isinstance(value, list):
        texts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                text = normalize_text(item.get("text"))
            else:
                text = normalize_text(item)
            if text:
                texts.append(text)
        return texts
    if isinstance(value, dict):
        text = normalize_text(value.get("text"))
        return [text] if text else []
    text = normalize_text(value)
    return [text] if text else []


def candidate_bucket(role: str, retained_count: int) -> str:
    suffix = "with_evidence" if retained_count > 0 else "no_evidence"
    return f"{role}_{suffix}"


def format_candidate_preview(candidate: Dict[str, Any]) -> str:
    title = normalize_text(candidate.get("title"))
    text = normalize_text(candidate.get("text"))
    reranker_rank = candidate.get("reranker_rank")
    retriever_rank = candidate.get("retriever_rank")
    prefix_parts = []
    if reranker_rank not in {None, ""}:
        prefix_parts.append(f"rr={reranker_rank}")
    if retriever_rank not in {None, ""}:
        prefix_parts.append(f"tr={retriever_rank}")
    prefix = f"[{', '.join(prefix_parts)}] " if prefix_parts else ""
    if title and text:
        return f"{prefix}{title}: {text}"
    return prefix + (title or text)


def iter_jsonl_rows(path: str, scan_state: Dict[str, Any] | None = None) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if scan_state is not None:
                    scan_state["invalid_json_lines"] = int(scan_state.get("invalid_json_lines", 0)) + 1
                    preview = scan_state.setdefault("invalid_json_line_numbers", [])
                    if len(preview) < 10:
                        preview.append(line_number)
                continue
            if isinstance(row, dict):
                yield row


def iter_review_candidates(path: str, args: argparse.Namespace, roles: Iterable[str]) -> Iterator[Dict[str, Any]]:
    yield from iter_review_candidates_with_state(path, args, roles, scan_state=None)


def iter_review_candidates_with_state(
    path: str,
    args: argparse.Namespace,
    roles: Iterable[str],
    scan_state: Dict[str, Any] | None,
) -> Iterator[Dict[str, Any]]:
    selected_roles = set(roles)
    for row_index, row in enumerate(iter_jsonl_rows(path, scan_state=scan_state)):
        row_id = str(row.get("id", row_index))
        patient_context = truncate_text(row.get("patient_context"), args.max_context_chars)
        arus = row.get("arus") or []
        if not isinstance(arus, list):
            continue
        for aru_index, node in enumerate(arus):
            role = str(node.get("type", "C")).upper()[:1] or "C"
            if role not in selected_roles:
                continue
            aru_text = normalize_text(node.get("text"))
            if not aru_text:
                continue

            retained_evidence_full = evidence_texts(node.get("retrieved_evidence"))
            retrieval_candidates_raw = node.get("retrieval_candidates") or []
            if not isinstance(retrieval_candidates_raw, list):
                retrieval_candidates_raw = []

            retrieval_candidates = []
            for item in retrieval_candidates_raw[: max(args.max_candidates_per_node, 0)]:
                if isinstance(item, dict):
                    retrieval_candidates.append(
                        {
                            "doc_id": item.get("doc_id"),
                            "corpus_id": item.get("corpus_id"),
                            "title": truncate_text(item.get("title"), args.max_text_chars),
                            "text": truncate_text(item.get("text"), args.max_text_chars),
                            "retriever_rank": item.get("retriever_rank"),
                            "retriever_score": item.get("retriever_score"),
                            "reranker_rank": item.get("reranker_rank"),
                            "reranker_score": item.get("reranker_score"),
                        }
                    )
                else:
                    retrieval_candidates.append(
                        {
                            "doc_id": "",
                            "corpus_id": "",
                            "title": "",
                            "text": truncate_text(item, args.max_text_chars),
                            "retriever_rank": "",
                            "retriever_score": "",
                            "reranker_rank": "",
                            "reranker_score": "",
                        }
                    )

            retained_evidence = [
                truncate_text(text, args.max_text_chars)
                for text in retained_evidence_full[: max(args.max_evidence_per_node, 0)]
            ]

            yield {
                "sample_id": f"{row_id}::aru{aru_index}",
                "id": row_id,
                "source_id": row.get("source_id"),
                "error_symbol": row.get("error_symbol"),
                "row_index": row_index,
                "aru_id": aru_index,
                "aru_role": role,
                "bucket": candidate_bucket(role, len(retained_evidence_full)),
                "patient_context": patient_context,
                "aru_text": truncate_text(aru_text, args.max_text_chars),
                "retrieval_query": truncate_text(
                    node.get("retrieval_query") or aru_text,
                    args.max_text_chars,
                ),
                "retrieval_query_source": normalize_text(node.get("retrieval_query_source")) or (
                    "legacy_missing" if role in {"W", "D"} else ""
                ),
                "retrieval_query_variants": [
                    truncate_text(text, args.max_text_chars)
                    for text in (
                        node.get("retrieval_query_variants")
                        or ([aru_text] if role in {"W", "D"} else [])
                    )[: max(args.max_candidates_per_node, 0)]
                ],
                "retrieved_evidence_count": len(retained_evidence_full),
                "retrieval_candidate_count": len(retrieval_candidates_raw),
                "retrieved_evidence": retained_evidence,
                "retrieval_candidates": retrieval_candidates,
            }


def count_buckets(path: str, args: argparse.Namespace, roles: List[str], scan_state: Dict[str, Any] | None = None) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in iter_review_candidates_with_state(path, args, roles, scan_state=scan_state):
        counter[str(item["bucket"])] += 1
    return counter


def allocate_quotas(bucket_counts: Counter[str], total_samples: int) -> Dict[str, int]:
    active_buckets = [bucket for bucket, count in sorted(bucket_counts.items()) if count > 0]
    if not active_buckets or total_samples <= 0:
        return {bucket: 0 for bucket in active_buckets}

    quotas = {bucket: 0 for bucket in active_buckets}
    base = total_samples // len(active_buckets)
    for bucket in active_buckets:
        quotas[bucket] = min(bucket_counts[bucket], base)

    remaining = total_samples - sum(quotas.values())
    while remaining > 0:
        progressed = False
        for bucket in active_buckets:
            if quotas[bucket] >= bucket_counts[bucket]:
                continue
            quotas[bucket] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return quotas


def reservoir_sample(path: str, args: argparse.Namespace, roles: List[str], quotas: Dict[str, int]) -> List[Dict[str, Any]]:
    rng = Random(args.seed)
    seen: Counter[str] = Counter()
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in iter_review_candidates(path, args, roles):
        bucket = str(item["bucket"])
        quota = quotas.get(bucket, 0)
        if quota <= 0:
            continue
        seen[bucket] += 1
        current = buckets[bucket]
        if len(current) < quota:
            current.append(item)
            continue
        replacement_index = rng.randint(0, seen[bucket] - 1)
        if replacement_index < quota:
            current[replacement_index] = item

    merged: List[Dict[str, Any]] = []
    for bucket in sorted(buckets):
        merged.extend(sorted(buckets[bucket], key=lambda item: str(item.get("sample_id"))))
    return merged


def flatten_for_csv(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for row in rows:
        flattened.append(
            {
                "sample_id": row.get("sample_id", ""),
                "id": row.get("id", ""),
                "source_id": row.get("source_id", ""),
                "error_symbol": row.get("error_symbol", ""),
                "aru_id": row.get("aru_id", ""),
                "aru_role": row.get("aru_role", ""),
                "bucket": row.get("bucket", ""),
                "retrieved_evidence_count": row.get("retrieved_evidence_count", 0),
                "retrieval_candidate_count": row.get("retrieval_candidate_count", 0),
                "aru_text": row.get("aru_text", ""),
                "retrieval_query": row.get("retrieval_query", ""),
                "retrieval_query_source": row.get("retrieval_query_source", ""),
                "retrieval_query_variants": " || ".join(row.get("retrieval_query_variants") or []),
                "patient_context": row.get("patient_context", ""),
                "retrieved_evidence_preview": " || ".join(row.get("retrieved_evidence") or []),
                "retrieval_candidates_preview": " || ".join(
                    format_candidate_preview(candidate) for candidate in (row.get("retrieval_candidates") or [])
                ),
                "review_label": "",
                "review_notes": "",
            }
        )
    return flattened


def build_summary(
    bucket_counts: Counter[str],
    quotas: Dict[str, int],
    sampled_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    scan_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    sampled_counts: Counter[str] = Counter(str(row.get("bucket")) for row in sampled_rows)
    return {
        "aru_file": args.aru_file,
        "roles": parse_roles(args.roles),
        "total_candidates": int(sum(bucket_counts.values())),
        "requested_samples": int(args.total_samples),
        "sampled_rows": len(sampled_rows),
        "bucket_counts": {bucket: int(bucket_counts[bucket]) for bucket in sorted(bucket_counts)},
        "bucket_quotas": {bucket: int(quotas.get(bucket, 0)) for bucket in sorted(bucket_counts)},
        "sampled_bucket_counts": {bucket: int(sampled_counts[bucket]) for bucket in sorted(sampled_counts)},
        "seed": args.seed,
        "max_evidence_per_node": args.max_evidence_per_node,
        "max_candidates_per_node": args.max_candidates_per_node,
        "max_context_chars": args.max_context_chars,
        "max_text_chars": args.max_text_chars,
        "invalid_json_lines": int((scan_state or {}).get("invalid_json_lines", 0)),
        "invalid_json_line_numbers": list((scan_state or {}).get("invalid_json_line_numbers", [])),
    }


def main() -> None:
    args = parse_args()
    roles = parse_roles(args.roles)
    if not roles:
        raise ValueError("No valid ARU roles were parsed from `--roles`.")

    scan_state: Dict[str, Any] = {"invalid_json_lines": 0, "invalid_json_line_numbers": []}
    bucket_counts = count_buckets(args.aru_file, args, roles, scan_state=scan_state)
    quotas = allocate_quotas(bucket_counts, args.total_samples)
    sampled_rows = reservoir_sample(args.aru_file, args, roles, quotas)

    write_jsonl(args.output_jsonl, sampled_rows)
    write_csv(
        args.output_csv,
        flatten_for_csv(sampled_rows),
        fieldnames=[
            "sample_id",
            "id",
            "source_id",
            "error_symbol",
            "aru_id",
            "aru_role",
            "bucket",
            "retrieved_evidence_count",
            "retrieval_candidate_count",
            "aru_text",
            "retrieval_query",
            "retrieval_query_source",
            "retrieval_query_variants",
            "patient_context",
            "retrieved_evidence_preview",
            "retrieval_candidates_preview",
            "review_label",
            "review_notes",
        ],
    )
    summary = build_summary(bucket_counts, quotas, sampled_rows, args, scan_state=scan_state)
    write_json(args.output_summary, summary)

    print(f"Total candidates: {summary['total_candidates']}")
    print(f"Requested samples: {summary['requested_samples']}")
    print(f"Sampled rows: {summary['sampled_rows']}")
    print(f"Buckets: {summary['bucket_counts']}")
    if summary["invalid_json_lines"]:
        print(
            "Skipped invalid JSON lines: "
            f"{summary['invalid_json_lines']} at {summary['invalid_json_line_numbers']}"
        )
    print(f"JSONL: {args.output_jsonl}")
    print(f"CSV: {args.output_csv}")
    print(f"Summary: {args.output_summary}")


if __name__ == "__main__":
    main()
