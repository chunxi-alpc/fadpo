#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import List

from common import result_path, run_python_script


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Stage-2 wrapper for strict-clean filtering and review artifacts."
    )
    parser.add_argument(
        "--accepted-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.jsonl"),
        help="Accepted negatives JSONL from stage 02.",
    )
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def main() -> None:
    args, extra_args = parse_args()
    passthrough = ["--accepted-file", args.accepted_file, *extra_args]
    run_python_script("fa_dpo_pipeline/legacy/build_negative_qc_artifacts.py", passthrough)


if __name__ == "__main__":
    main()
