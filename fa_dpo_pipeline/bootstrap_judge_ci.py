#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List

from common import load_jsonl, output_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for pairwise judge win rates."
    )
    parser.add_argument(
        "--judgments-file",
        default=output_path("gpt_judge", "judgments.jsonl"),
        help="Structured judgments JSONL from run_pairwise_judge.py.",
    )
    parser.add_argument(
        "--output-file",
        default=output_path("gpt_judge", "bootstrap_ci.json"),
        help="Where to write the bootstrap summary JSON.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="How many bootstrap resamples to draw.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def bootstrap_distribution(outcomes: List[str], label: str, rng: random.Random, samples: int) -> List[float]:
    if not outcomes:
        return []
    distribution = []
    for _ in range(samples):
        draw = [rng.choice(outcomes) for _ in outcomes]
        counter = Counter(draw)
        distribution.append(counter.get(label, 0) / len(draw))
    return distribution


def main() -> None:
    args = parse_args()
    judgments = load_jsonl(args.judgments_file)
    grouped: Dict[str, List[str]] = defaultdict(list)
    meta: Dict[str, Any] = {}
    for row in judgments:
        meta["left_model_id"] = row.get("left_model_id")
        meta["right_model_id"] = row.get("right_model_id")
        for criterion, item in (row.get("criteria") or {}).items():
            grouped[criterion].append(str(item.get("winner")))

    rng = random.Random(args.seed)
    result: Dict[str, Any] = {
        "left_model_id": meta.get("left_model_id"),
        "right_model_id": meta.get("right_model_id"),
        "bootstrap_samples": args.bootstrap_samples,
        "criteria": {},
    }
    for criterion, outcomes in sorted(grouped.items()):
        counter = Counter(outcomes)
        criterion_result = {"n": len(outcomes), "point_estimates": {}, "ci_95": {}}
        for label in ("left", "tie", "right"):
            point = counter.get(label, 0) / max(len(outcomes), 1)
            dist = bootstrap_distribution(outcomes, label, rng, args.bootstrap_samples)
            criterion_result["point_estimates"][label] = round(point, 6)
            criterion_result["ci_95"][label] = {
                "low": round(quantile(dist, 0.025), 6),
                "high": round(quantile(dist, 0.975), 6),
            }
        result["criteria"][criterion] = criterion_result

    write_json(args.output_file, result)
    print(f"Bootstrap criteria: {len(result['criteria'])}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
