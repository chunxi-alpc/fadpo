#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import artifact_path, output_path, read_json, write_json
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
        description="Aggregate per-stage runtime and scaling metadata for the Fa-DPO pipeline."
    )
    parser.add_argument(
        "--metadata-dir",
        default=artifact_path("fa_dpo_pipeline", "run_metadata"),
        help="Directory containing per-stage metadata JSON files.",
    )
    parser.add_argument(
        "--summary-output",
        default=output_path("cost_scalability", "cost_summary.csv"),
        help="Where to write the stage-level summary CSV.",
    )
    parser.add_argument(
        "--hardware-output",
        default=output_path("cost_scalability", "hardware.json"),
        help="Where to write aggregated hardware info JSON.",
    )
    parser.add_argument(
        "--stats-output",
        default=output_path("cost_scalability", "per_stage_stats.json"),
        help="Where to write the full per-stage metadata JSON.",
    )
    return parser.parse_args()


def infer_sample_count(stats: Dict[str, Any]) -> float:
    for key in SAMPLE_COUNT_KEYS:
        if key in stats:
            try:
                return float(stats[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def main() -> None:
    args = parse_args()
    metadata_dir = Path(args.metadata_dir)
    stage_payloads: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    hardware: Dict[str, Any] = {"stages": {}}

    for path in sorted(metadata_dir.glob("*.json")):
        payload = read_json(path)
        stage_payloads.append(payload)
        stats = payload.get("stats") or {}
        elapsed = float(payload.get("elapsed_seconds") or 0.0)
        sample_count = infer_sample_count(stats)
        throughput = sample_count / (elapsed / 3600.0) if elapsed > 0 and sample_count > 0 else 0.0
        summary_rows.append(
            {
                "stage_name": payload.get("stage_name"),
                "status": payload.get("status"),
                "elapsed_seconds": round(elapsed, 4),
                "elapsed_hours": round(elapsed / 3600.0, 6),
                "sample_count": round(sample_count, 4),
                "throughput_per_hour": round(throughput, 6),
                "total_nli_pairs": stats.get("total_nli_pairs", ""),
                "unique_queries": stats.get("unique_queries", ""),
                "avg_arus_per_sample": stats.get("avg_arus_per_sample", ""),
            }
        )
        hardware["stages"][str(payload.get("stage_name"))] = payload.get("hardware") or {}

    write_csv(args.summary_output, summary_rows)
    write_json(args.hardware_output, hardware)
    write_json(args.stats_output, stage_payloads)
    print(f"Stages: {len(summary_rows)}")
    print(f"Summary: {args.summary_output}")
    print(f"Hardware: {args.hardware_output}")
    print(f"Per-stage stats: {args.stats_output}")


if __name__ == "__main__":
    main()
