#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fa_dpo_pipeline.common import (
    artifact_path,
    ensure_parent,
    iso_utc_now,
    read_jsonl,
    safe_float,
    write_jsonl,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 60-row manual-review pack for stage-06 by sampling high-risk, "
            "mid-band, and low-risk rows and extracting matching rows from score/evidence/negative files."
        )
    )
    parser.add_argument("--score-file", required=True, help="Stage-06 faithfulness_scores.jsonl")
    parser.add_argument("--evidence-file", required=True, help="Stage-05/06 ARU-with-evidence JSONL")
    parser.add_argument("--negatives-file", required=True, help="Original negatives JSONL aligned by id")
    parser.add_argument(
        "--output-dir",
        default=artifact_path("fa_dpo_pipeline", "manual_review_packs", "score06_sample60"),
        help="Directory where the review pack will be written.",
    )
    parser.add_argument("--aru-file", default="", help="Optional stage-04 ARU JSONL to subset.")
    parser.add_argument(
        "--score-metadata-file",
        default="",
        help="Optional stage-06 metadata JSON to copy into the pack.",
    )
    parser.add_argument(
        "--stage05-metadata-file",
        default="",
        help="Optional stage-05 metadata JSON to copy into the pack.",
    )
    parser.add_argument("--count-per-bucket", type=int, default=20, help="Rows to sample for each bucket.")
    parser.add_argument("--high-threshold", type=float, default=0.9, help="High-risk threshold.")
    parser.add_argument("--mid-min", type=float, default=0.45, help="Lower bound for mid-band sampling.")
    parser.add_argument("--mid-max", type=float, default=0.55, help="Upper bound for mid-band sampling.")
    parser.add_argument("--low-threshold", type=float, default=0.15, help="Low-risk threshold.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--allow-short-buckets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow fewer than count-per-bucket rows when a bucket is too small.",
    )
    parser.add_argument(
        "--metadata-file",
        default=artifact_path("fa_dpo_pipeline", "run_metadata", "build_score06_manual_review_pack.json"),
        help="Where to write run metadata JSON.",
    )
    return parser.parse_args()


def sample_bucket(
    rows: Sequence[Dict[str, Any]],
    count: int,
    rng: random.Random,
    bucket_name: str,
    allow_short: bool,
) -> List[Dict[str, Any]]:
    if len(rows) < count and not allow_short:
        raise RuntimeError(
            f"Bucket '{bucket_name}' only has {len(rows)} rows, fewer than requested {count}. "
            "Use --allow-short-buckets if this is intentional."
        )
    actual_count = min(count, len(rows))
    if actual_count == 0:
        return []
    return rng.sample(list(rows), actual_count)


def order_bucket_rows(bucket_name: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def score(row: Dict[str, Any]) -> float:
        return safe_float(row.get("unfaithfulness_score"))

    if bucket_name == "high_risk":
        return sorted(rows, key=lambda row: (-score(row), str(row.get("id"))))
    if bucket_name == "mid_band":
        return sorted(rows, key=lambda row: (abs(score(row) - 0.5), score(row), str(row.get("id"))))
    return sorted(rows, key=lambda row: (score(row), str(row.get("id"))))


def build_manifest_rows(sampled_by_bucket: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    manifest_rows: List[Dict[str, Any]] = []
    for bucket_name in ("high_risk", "mid_band", "low_risk"):
        ordered_rows = order_bucket_rows(bucket_name, sampled_by_bucket.get(bucket_name, []))
        for idx, row in enumerate(ordered_rows, start=1):
            metrics = row.get("metrics") or {}
            manifest_rows.append(
                {
                    "review_bucket": bucket_name,
                    "bucket_rank": idx,
                    "id": str(row.get("id")),
                    "source_id": row.get("source_id"),
                    "error_symbol": row.get("error_symbol"),
                    "unfaithfulness_score": safe_float(row.get("unfaithfulness_score")),
                    "dpo_margin": safe_float(row.get("dpo_margin")),
                    "max_risk": safe_float(metrics.get("max_risk")),
                    "avg_defect": safe_float(metrics.get("avg_defect")),
                    "coverage_ratio": safe_float(metrics.get("coverage_ratio")),
                    "penalty_score": safe_float(metrics.get("penalty_score")),
                }
            )
    return manifest_rows


def extract_rows_by_id(path: str, ordered_ids: Sequence[str]) -> tuple[List[Dict[str, Any]], List[str]]:
    target_ids = set(ordered_ids)
    found: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        row_id = str(row.get("id"))
        if row_id in target_ids and row_id not in found:
            found[row_id] = row
            if len(found) == len(target_ids):
                break
    ordered_rows = [found[row_id] for row_id in ordered_ids if row_id in found]
    missing_ids = [row_id for row_id in ordered_ids if row_id not in found]
    return ordered_rows, missing_ids


def copy_optional_file(path_text: str, output_dir: Path) -> str:
    if not path_text:
        return ""
    source = Path(path_text)
    if not source.exists():
        return ""
    target = output_dir / source.name
    ensure_parent(target)
    shutil.copy2(source, target)
    return str(target)


def write_manifest_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "review_bucket",
        "bucket_rank",
        "id",
        "source_id",
        "error_symbol",
        "unfaithfulness_score",
        "dpo_margin",
        "max_risk",
        "avg_defect",
        "coverage_ratio",
        "penalty_score",
    ]
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(
    path: Path,
    args: argparse.Namespace,
    manifest_rows: Sequence[Dict[str, Any]],
    missing_by_file: Dict[str, List[str]],
    copied_files: Dict[str, str],
) -> None:
    bucket_counts: Dict[str, int] = {}
    for row in manifest_rows:
        bucket = str(row.get("review_bucket"))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    lines = [
        "# Score06 Manual Review Pack",
        "",
        f"- created_at: `{iso_utc_now()}`",
        f"- score_file: `{args.score_file}`",
        f"- evidence_file: `{args.evidence_file}`",
        f"- negatives_file: `{args.negatives_file}`",
    ]
    if args.aru_file:
        lines.append(f"- aru_file: `{args.aru_file}`")
    lines.extend(
        [
            "",
            "## Sample Counts",
            "",
            f"- high_risk (`score >= {args.high_threshold}`): {bucket_counts.get('high_risk', 0)}",
            f"- mid_band (`{args.mid_min} <= score <= {args.mid_max}`): {bucket_counts.get('mid_band', 0)}",
            f"- low_risk (`score <= {args.low_threshold}`): {bucket_counts.get('low_risk', 0)}",
            f"- total: {len(manifest_rows)}",
            "",
            "## Missing IDs",
            "",
        ]
    )
    for file_label, missing_ids in missing_by_file.items():
        lines.append(f"- {file_label}: {len(missing_ids)} missing")
        if missing_ids:
            preview = ", ".join(missing_ids[:10])
            lines.append(f"  ids: `{preview}`")
    lines.extend(["", "## Copied Metadata", ""])
    for label, copied_path in copied_files.items():
        lines.append(f"- {label}: `{copied_path or 'not copied'}`")
    ensure_parent(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started_at = iso_utc_now()

    score_rows = list(read_jsonl(args.score_file))
    high_rows = [row for row in score_rows if safe_float(row.get("unfaithfulness_score")) >= args.high_threshold]
    mid_rows = [
        row
        for row in score_rows
        if args.mid_min <= safe_float(row.get("unfaithfulness_score")) <= args.mid_max
    ]
    low_rows = [row for row in score_rows if safe_float(row.get("unfaithfulness_score")) <= args.low_threshold]

    rng = random.Random(args.seed)
    sampled_by_bucket = {
        "high_risk": sample_bucket(high_rows, args.count_per_bucket, rng, "high_risk", args.allow_short_buckets),
        "mid_band": sample_bucket(mid_rows, args.count_per_bucket, rng, "mid_band", args.allow_short_buckets),
        "low_risk": sample_bucket(low_rows, args.count_per_bucket, rng, "low_risk", args.allow_short_buckets),
    }

    manifest_rows = build_manifest_rows(sampled_by_bucket)
    ordered_ids = [str(row["id"]) for row in manifest_rows]
    sample_tag = f"sample{len(manifest_rows)}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_selected, missing_scores = extract_rows_by_id(args.score_file, ordered_ids)
    evidence_selected, missing_evidence = extract_rows_by_id(args.evidence_file, ordered_ids)
    negatives_selected, missing_negatives = extract_rows_by_id(args.negatives_file, ordered_ids)

    write_jsonl(output_dir / "sample_manifest.jsonl", manifest_rows)
    write_manifest_csv(output_dir / "sample_manifest.csv", manifest_rows)
    (output_dir / "sampled_ids.txt").write_text("\n".join(ordered_ids) + "\n", encoding="utf-8")
    score_subset_path = output_dir / f"faithfulness_scores.{sample_tag}.jsonl"
    evidence_subset_path = output_dir / f"arus_with_evidence.{sample_tag}.jsonl"
    negatives_subset_path = output_dir / f"negatives.{sample_tag}.jsonl"
    write_jsonl(score_subset_path, score_selected)
    write_jsonl(evidence_subset_path, evidence_selected)
    write_jsonl(negatives_subset_path, negatives_selected)

    missing_by_file = {
        "score_file": missing_scores,
        "evidence_file": missing_evidence,
        "negatives_file": missing_negatives,
    }

    copied_files = {
        "score_metadata": copy_optional_file(args.score_metadata_file, output_dir),
        "stage05_metadata": copy_optional_file(args.stage05_metadata_file, output_dir),
    }

    if args.aru_file:
        aru_selected, missing_aru = extract_rows_by_id(args.aru_file, ordered_ids)
        aru_subset_path = output_dir / f"arus.{sample_tag}.jsonl"
        write_jsonl(aru_subset_path, aru_selected)
        missing_by_file["aru_file"] = missing_aru
    else:
        aru_subset_path = None
        missing_by_file["aru_file"] = []

    write_summary(
        output_dir / "review_pack_summary.md",
        args=args,
        manifest_rows=manifest_rows,
        missing_by_file=missing_by_file,
        copied_files=copied_files,
    )

    metadata_path = write_run_metadata(
        stage_name="build_score06_manual_review_pack",
        args=args,
        inputs={
            "score_file": args.score_file,
            "evidence_file": args.evidence_file,
            "negatives_file": args.negatives_file,
            "aru_file": args.aru_file,
            "score_metadata_file": args.score_metadata_file,
            "stage05_metadata_file": args.stage05_metadata_file,
        },
        outputs={
            "output_dir": output_dir,
            "manifest_jsonl": output_dir / "sample_manifest.jsonl",
            "manifest_csv": output_dir / "sample_manifest.csv",
            "score_subset": score_subset_path,
            "evidence_subset": evidence_subset_path,
            "negatives_subset": negatives_subset_path,
            "aru_subset": aru_subset_path,
        },
        stats={
            "score_rows_total": len(score_rows),
            "high_candidates": len(high_rows),
            "mid_candidates": len(mid_rows),
            "low_candidates": len(low_rows),
            "sampled_rows_total": len(manifest_rows),
            "sampled_high_risk": len(sampled_by_bucket["high_risk"]),
            "sampled_mid_band": len(sampled_by_bucket["mid_band"]),
            "sampled_low_risk": len(sampled_by_bucket["low_risk"]),
            "missing_score_rows": len(missing_scores),
            "missing_evidence_rows": len(missing_evidence),
            "missing_negative_rows": len(missing_negatives),
            "missing_aru_rows": len(missing_by_file["aru_file"]),
        },
        metadata_file=args.metadata_file,
        started_at=started_at,
        finished_at=iso_utc_now(),
    )

    print(f"Output dir: {output_dir}")
    print(f"Manifest JSONL: {output_dir / 'sample_manifest.jsonl'}")
    print(f"Manifest CSV: {output_dir / 'sample_manifest.csv'}")
    print(f"Score subset: {score_subset_path}")
    print(f"Evidence subset: {evidence_subset_path}")
    print(f"Negatives subset: {negatives_subset_path}")
    if aru_subset_path is not None:
        print(f"ARU subset: {aru_subset_path}")
    print(f"Summary: {output_dir / 'review_pack_summary.md'}")
    print(f"Run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
