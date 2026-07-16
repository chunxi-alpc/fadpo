#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from scipy.stats import ttest_rel

from common import output_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate same-backbone multiseed results and run paired t-tests."
    )
    parser.add_argument("--input-file", required=True, help="CSV or JSONL table with model/seed/metric columns.")
    parser.add_argument("--baseline-model", required=True, help="Baseline model label for paired t-tests.")
    parser.add_argument("--target-model", required=True, help="Target model label for paired t-tests.")
    parser.add_argument("--model-col", default="model", help="Column name for the model setting.")
    parser.add_argument("--seed-col", default="seed", help="Column name for the seed.")
    parser.add_argument(
        "--group-cols",
        default="setting",
        help="Comma-separated grouping columns besides model and seed. Use a single constant column if needed.",
    )
    parser.add_argument(
        "--summary-output",
        default=output_path("significance", "multiseed_results.csv"),
        help="Where to write the mean/std summary CSV.",
    )
    parser.add_argument(
        "--ttest-output",
        default=output_path("significance", "ttest_summary.csv"),
        help="Where to write the paired t-test CSV.",
    )
    return parser.parse_args()


def load_frame(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    if file_path.suffix.lower() == ".jsonl":
        return pd.read_json(file_path, lines=True)
    raise ValueError(f"Unsupported input format: {path}")


def main() -> None:
    args = parse_args()
    frame = load_frame(args.input_file)
    group_cols = [col.strip() for col in args.group_cols.split(",") if col.strip()]
    if group_cols == ["setting"] and "setting" not in frame.columns:
        frame["setting"] = "default"
    numeric_cols = [
        column
        for column in frame.columns
        if column not in set(group_cols + [args.model_col, args.seed_col]) and pd.api.types.is_numeric_dtype(frame[column])
    ]

    summary_rows: List[dict] = []
    for keys, subset in frame.groupby(group_cols + [args.model_col], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols + [args.model_col], keys))
        for metric in numeric_cols:
            summary_rows.append(
                {
                    **key_map,
                    "metric": metric,
                    "n_seeds": int(subset[args.seed_col].nunique()),
                    "mean": round(float(subset[metric].mean()), 6),
                    "std": round(float(subset[metric].std(ddof=0)), 6),
                }
            )
    pd.DataFrame(summary_rows).to_csv(args.summary_output, index=False)

    ttest_rows: List[dict] = []
    for keys, subset in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        baseline = subset[subset[args.model_col] == args.baseline_model]
        target = subset[subset[args.model_col] == args.target_model]
        merged = baseline.merge(target, on=args.seed_col, suffixes=("_baseline", "_target"))
        if merged.empty:
            continue
        for metric in numeric_cols:
            left = merged[f"{metric}_baseline"]
            right = merged[f"{metric}_target"]
            stat = ttest_rel(right, left, nan_policy="omit")
            ttest_rows.append(
                {
                    **key_map,
                    "metric": metric,
                    "n_pairs": int(len(merged)),
                    "baseline_model": args.baseline_model,
                    "target_model": args.target_model,
                    "baseline_mean": round(float(left.mean()), 6),
                    "target_mean": round(float(right.mean()), 6),
                    "mean_delta": round(float((right - left).mean()), 6),
                    "t_stat": round(float(stat.statistic), 6) if stat.statistic == stat.statistic else "",
                    "p_value": round(float(stat.pvalue), 6) if stat.pvalue == stat.pvalue else "",
                }
            )
    pd.DataFrame(ttest_rows).to_csv(args.ttest_output, index=False)

    print(f"Metrics: {args.summary_output}")
    print(f"T-test: {args.ttest_output}")


if __name__ == "__main__":
    main()
