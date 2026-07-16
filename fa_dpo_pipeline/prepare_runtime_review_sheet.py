#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import artifact_path, output_path, read_json
from experiment_utils import write_csv


SAMPLE_COUNT_KEYS = (
    "accepted_rows",
    "processed_rows",
    "scored_rows",
    "train_rows",
    "total_rows",
    "indexed_documents",
    "kept_docs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert run metadata JSON files into a reviewer-friendly runtime checklist CSV."
    )
    parser.add_argument(
        "--metadata-dir",
        default=artifact_path("fa_dpo_pipeline", "run_metadata"),
        help="Directory containing per-stage run metadata JSON files.",
    )
    parser.add_argument(
        "--output-file",
        default=output_path("cost_scalability", "runtime_review.csv"),
        help="Reviewer-facing CSV to fill manually.",
    )
    return parser.parse_args()


def infer_sample_count(stats: Dict[str, Any]) -> Any:
    for key in SAMPLE_COUNT_KEYS:
        if key in stats:
            return stats[key]
    return ""


def flatten_paths(mapping: Dict[str, Any]) -> str:
    parts = []
    for key, value in sorted(mapping.items()):
        parts.append(f"{key}={value}")
    return " | ".join(parts)


def gpu_summary(hardware: Dict[str, Any]) -> str:
    gpus = hardware.get("gpus") or []
    if gpus:
        names = [str(item.get("name", "")) for item in gpus if item.get("name")]
        return f"{hardware.get('gpu_count', len(names))}x " + ", ".join(names)
    if hardware.get("gpu_count") is not None:
        return str(hardware.get("gpu_count"))
    return ""


def main() -> None:
    args = parse_args()
    metadata_dir = Path(args.metadata_dir)
    rows: List[Dict[str, Any]] = []
    for index, path in enumerate(sorted(metadata_dir.glob("*.json")), start=1):
        payload = read_json(path)
        stats = payload.get("stats") or {}
        rows.append(
            {
                "review_id": f"rt-{index:03d}",
                "stage_name": payload.get("stage_name", ""),
                "status": payload.get("status", ""),
                "started_at": payload.get("started_at", ""),
                "finished_at": payload.get("finished_at", ""),
                "elapsed_seconds": payload.get("elapsed_seconds", ""),
                "sample_count": infer_sample_count(stats),
                "gpu_summary": gpu_summary(payload.get("hardware") or {}),
                "input_paths": flatten_paths(payload.get("inputs") or {}),
                "output_paths": flatten_paths(payload.get("outputs") or {}),
                "confirm_hardware": "",
                "confirm_paths": "",
                "confirm_runtime": "",
                "notes": "",
            }
        )
    write_csv(
        args.output_file,
        rows,
        fieldnames=[
            "review_id",
            "stage_name",
            "status",
            "started_at",
            "finished_at",
            "elapsed_seconds",
            "sample_count",
            "gpu_summary",
            "input_paths",
            "output_paths",
            "confirm_hardware",
            "confirm_paths",
            "confirm_runtime",
            "notes",
        ],
    )
    print(f"Review rows: {len(rows)}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
