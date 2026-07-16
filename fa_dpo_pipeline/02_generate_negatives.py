#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from common import artifact_path, count_jsonl, iso_utc_now, result_path, run_python_script, write_run_metadata


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Stage-1 wrapper for controlled negative generation."
    )
    parser.add_argument(
        "--output-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.jsonl"),
        help="Accepted negatives JSONL output path.",
    )
    parser.add_argument(
        "--source",
        default="medcase",
        help="Input source passed through to generate_bad_reasoning.py.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split passed through to generate_bad_reasoning.py.",
    )
    parser.add_argument(
        "--metadata-file",
        default=artifact_path("fa_dpo_pipeline", "run_metadata", "02_generate_negatives.json"),
        help="Where to write run metadata JSON.",
    )
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def main() -> None:
    args, extra_args = parse_args()
    started_at = iso_utc_now()
    passthrough = [
        "--source",
        args.source,
        "--split",
        args.split,
        "--output-file",
        args.output_file,
        *extra_args,
    ]
    run_python_script("fa_dpo_pipeline/legacy/generate_bad_reasoning.py", passthrough)
    output_path = Path(args.output_file)
    metadata_path = write_run_metadata(
        stage_name="02_generate_negatives",
        args=args,
        inputs={"source": args.source, "split": args.split},
        outputs={
            "output_file": output_path,
            "attempts_file": output_path.with_suffix(".attempts.jsonl"),
            "review_file": output_path.with_suffix(".review.jsonl"),
            "rejected_file": output_path.with_suffix(".rejected.jsonl"),
            "infeasible_file": output_path.with_suffix(".infeasible.jsonl"),
            "failures_file": output_path.with_suffix(".failures.jsonl"),
            "progress_file": output_path.with_suffix(".progress.json"),
        },
        stats={
            "accepted_rows": count_jsonl(output_path),
            "attempt_rows": count_jsonl(output_path.with_suffix(".attempts.jsonl")),
            "review_rows": count_jsonl(output_path.with_suffix(".review.jsonl")),
            "rejected_rows": count_jsonl(output_path.with_suffix(".rejected.jsonl")),
            "infeasible_rows": count_jsonl(output_path.with_suffix(".infeasible.jsonl")),
            "failure_rows": count_jsonl(output_path.with_suffix(".failures.jsonl")),
        },
        metadata_file=args.metadata_file,
        started_at=started_at,
        finished_at=iso_utc_now(),
    )
    print(f"Run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
