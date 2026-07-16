#!/usr/bin/env python3
"""Compute paired bootstrap CIs and paired tests for R1.13/R2.10.

The script expects paired, question-level logs. It intentionally does not try to
recover uncertainty from aggregate tables in the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PROCESS_METRICS = (
    "support_rate",
    "contradiction_rate",
    "unsupported_medical_reasoning_rate",
    "claim_inconsistency_rate",
)


def read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "correct"}


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(ordered[low])
    weight = pos - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def bootstrap_two_sided_pvalue(samples: Sequence[float]) -> float:
    if not samples:
        return float("nan")
    n = len(samples)
    le_zero = sum(1 for value in samples if value <= 0.0)
    ge_zero = sum(1 for value in samples if value >= 0.0)
    return min(1.0, 2.0 * min((le_zero + 1) / (n + 1), (ge_zero + 1) / (n + 1)))


def paired_bootstrap_summary(
    deltas: Sequence[float],
    n_resamples: int,
    seed: int,
) -> Tuple[float, float, float, float, List[float]]:
    if not deltas:
        return float("nan"), float("nan"), float("nan"), float("nan"), []
    rng = random.Random(seed)
    n = len(deltas)
    point = sum(deltas) / n
    samples: List[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        samples.append(total / n)
    return point, percentile(samples, 0.025), percentile(samples, 0.975), bootstrap_two_sided_pvalue(samples), samples


def paired_bootstrap_ci(deltas: Sequence[float], n_resamples: int, seed: int) -> Tuple[float, float, float]:
    point, low, high, _, _ = paired_bootstrap_summary(deltas, n_resamples, seed)
    return point, low, high


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    adjusted = [float("nan")] * len(pvalues)
    indexed = [
        (idx, pvalue)
        for idx, pvalue in enumerate(pvalues)
        if isinstance(pvalue, (int, float)) and math.isfinite(float(pvalue))
    ]
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running_max = 0.0
    for rank, (idx, pvalue) in enumerate(indexed):
        raw_adjusted = (m - rank) * float(pvalue)
        running_max = max(running_max, raw_adjusted)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail_to = min(b, c)
    tail = sum(math.comb(discordant, k) for k in range(tail_to + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def rank_abs_values(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(abs(value) for value in values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][1] == indexed[pos][1]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for original_idx, _ in indexed[pos:end]:
            ranks[original_idx] = avg_rank
        pos = end
    return ranks


def exact_wilcoxon(deltas: Sequence[float]) -> Dict[str, Any]:
    nonzero = [value for value in deltas if value != 0]
    if not nonzero:
        return {"n_nonzero": 0, "w_plus": 0.0, "p_two_sided": 1.0, "p_greater": 1.0}
    ranks = rank_abs_values(nonzero)
    observed = sum(rank for rank, delta in zip(ranks, nonzero) if delta > 0)
    total_rank = sum(ranks)
    all_w = []
    for signs in product((0, 1), repeat=len(nonzero)):
        all_w.append(sum(rank for rank, sign in zip(ranks, signs) if sign))
    center_distance = abs(observed - total_rank / 2.0)
    p_two = sum(1 for value in all_w if abs(value - total_rank / 2.0) >= center_distance) / len(all_w)
    p_greater = sum(1 for value in all_w if value >= observed) / len(all_w)
    return {
        "n_nonzero": len(nonzero),
        "w_plus": observed,
        "p_two_sided": min(1.0, p_two),
        "p_greater": min(1.0, p_greater),
    }


def normalize_system(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"standard", "std", "standard_dpo", "standard dpo", "standard dpo-retained"}:
        return "standard"
    if text in {"fadpo", "fa-dpo", "fa_dpo", "fa-dpo-retained"}:
        return "fadpo"
    return text


def process_rows_from_long(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        backbone = str(row.get("backbone", "")).strip()
        case_id = str(row.get("case_id", row.get("question_id", ""))).strip()
        system = normalize_system(row.get("system", row.get("method", "")))
        if not backbone or not case_id or system not in {"standard", "fadpo"}:
            continue
        grouped[(backbone, case_id)][system] = row

    wide_rows: List[Dict[str, Any]] = []
    for (backbone, case_id), systems in grouped.items():
        if "standard" not in systems or "fadpo" not in systems:
            continue
        row: Dict[str, Any] = {"backbone": backbone, "case_id": case_id}
        for metric in PROCESS_METRICS:
            row[f"standard_{metric}"] = systems["standard"].get(metric)
            row[f"fadpo_{metric}"] = systems["fadpo"].get(metric)
        for count_key in ("standard_aru_count", "fadpo_aru_count", "aru_count"):
            if count_key in systems["standard"] or count_key in systems["fadpo"]:
                row[count_key] = systems["standard"].get(count_key, systems["fadpo"].get(count_key))
        wide_rows.append(row)
    return wide_rows


def summarize_process(path: Path, output_dir: Path, n_resamples: int, seed: int) -> None:
    raw_rows = read_rows(path)
    rows = process_rows_from_long(raw_rows) if any("system" in row or "method" in row for row in raw_rows) else raw_rows
    by_backbone: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        backbone = str(row.get("backbone", "")).strip()
        if backbone:
            by_backbone[backbone].append(row)

    summary: List[Dict[str, Any]] = []
    summary_by_backbone: Dict[str, Dict[str, Any]] = {}
    metric_rows: List[Dict[str, Any]] = []
    for backbone, group in sorted(by_backbone.items()):
        out: Dict[str, Any] = {"backbone": backbone, "n_cases": len(group), "responses": 2 * len(group)}
        aru_counts = [as_float(row.get("aru_count")) for row in group]
        std_aru_counts = [as_float(row.get("standard_aru_count")) for row in group]
        fadpo_aru_counts = [as_float(row.get("fadpo_aru_count")) for row in group]
        if any(value is not None for value in aru_counts):
            out["arus_total"] = int(sum(value or 0 for value in aru_counts))
        elif any(value is not None for value in std_aru_counts + fadpo_aru_counts):
            out["arus_total"] = int(sum(value or 0 for value in std_aru_counts + fadpo_aru_counts))
        else:
            out["arus_total"] = "NA"

        for metric in PROCESS_METRICS:
            deltas: List[float] = []
            for row in group:
                std = as_float(row.get(f"standard_{metric}"))
                fadpo = as_float(row.get(f"fadpo_{metric}"))
                if std is not None and fadpo is not None:
                    deltas.append(fadpo - std)
            point, low, high, pvalue, _ = paired_bootstrap_summary(deltas, n_resamples, seed)
            out[f"{metric}_delta"] = point
            out[f"{metric}_ci_low"] = low
            out[f"{metric}_ci_high"] = high
            out[f"{metric}_bootstrap_p"] = pvalue
            metric_rows.append(
                {
                    "backbone": backbone,
                    "metric": metric,
                    "n_cases": len(deltas),
                    "delta": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_p": pvalue,
                }
            )
        summary.append(out)
        summary_by_backbone[backbone] = out

    adjusted_process = holm_adjust([float(row["bootstrap_p"]) for row in metric_rows])
    for row, adjusted_p in zip(metric_rows, adjusted_process):
        row["holm_p"] = adjusted_p
    for backbone, out in summary_by_backbone.items():
        row_pvalues = [
            float(row["holm_p"])
            for row in metric_rows
            if row["backbone"] == backbone and math.isfinite(float(row["holm_p"]))
        ]
        out["max_holm_p"] = max(row_pvalues) if row_pvalues else "NA"

    write_csv(output_dir / "process_paired_summary.csv", summary)
    write_csv(output_dir / "process_bootstrap_ci.csv", metric_rows)


def summarize_accuracy(path: Path, output_dir: Path, n_resamples: int, seed: int) -> None:
    rows = read_rows(path)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        backbone = str(row.get("backbone", "")).strip()
        benchmark = str(row.get("benchmark", "")).strip()
        if backbone and benchmark:
            grouped[(backbone, benchmark)].append(row)

    accuracy_rows: List[Dict[str, Any]] = []
    mcnemar_rows: List[Dict[str, Any]] = []
    macro_rows: List[Dict[str, Any]] = []
    macro_deltas_by_backbone: Dict[str, float] = {}

    for (backbone, benchmark), group in sorted(grouped.items()):
        deltas = []
        b = 0
        c = 0
        std_correct_total = 0
        fadpo_correct_total = 0
        for row in group:
            std_correct = as_bool(row.get("standard_correct", row.get("std_correct")))
            fadpo_correct = as_bool(row.get("fadpo_correct", row.get("fa_dpo_correct")))
            std_correct_total += int(std_correct)
            fadpo_correct_total += int(fadpo_correct)
            deltas.append(float(fadpo_correct) - float(std_correct))
            if std_correct and not fadpo_correct:
                b += 1
            elif (not std_correct) and fadpo_correct:
                c += 1
        point, low, high, pvalue, _ = paired_bootstrap_summary(deltas, n_resamples, seed)
        accuracy_rows.append(
            {
                "backbone": backbone,
                "benchmark": benchmark,
                "n_questions": len(group),
                "standard_accuracy": std_correct_total / len(group) if group else float("nan"),
                "fadpo_accuracy": fadpo_correct_total / len(group) if group else float("nan"),
                "delta": point,
                "ci_low": low,
                "ci_high": high,
                "bootstrap_p": pvalue,
            }
        )
        mcnemar_rows.append(
            {
                "backbone": backbone,
                "benchmark": benchmark,
                "n_questions": len(group),
                "standard_correct_fadpo_wrong": b,
                "standard_wrong_fadpo_correct": c,
                "exact_mcnemar_p": exact_mcnemar_pvalue(b, c),
            }
        )

    adjusted_mcnemar = holm_adjust([float(row["exact_mcnemar_p"]) for row in mcnemar_rows])
    for row, adjusted_p in zip(mcnemar_rows, adjusted_mcnemar):
        row["holm_p"] = adjusted_p

    by_backbone_benchmark: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    for (backbone, benchmark), group in grouped.items():
        by_backbone_benchmark[backbone][benchmark] = group

    for backbone, benchmark_groups in sorted(by_backbone_benchmark.items()):
        benchmark_summaries: List[Tuple[str, float, float, List[float]]] = []
        for benchmark, group in sorted(benchmark_groups.items()):
            std_values = []
            fadpo_values = []
            deltas = []
            for row in group:
                std_correct = float(as_bool(row.get("standard_correct", row.get("std_correct"))))
                fadpo_correct = float(as_bool(row.get("fadpo_correct", row.get("fa_dpo_correct"))))
                std_values.append(std_correct)
                fadpo_values.append(fadpo_correct)
                deltas.append(fadpo_correct - std_correct)
            if deltas:
                benchmark_summaries.append(
                    (
                        benchmark,
                        sum(std_values) / len(std_values),
                        sum(fadpo_values) / len(fadpo_values),
                        deltas,
                    )
                )
        if not benchmark_summaries:
            continue
        standard_macro = sum(item[1] for item in benchmark_summaries) / len(benchmark_summaries)
        fadpo_macro = sum(item[2] for item in benchmark_summaries) / len(benchmark_summaries)
        point = fadpo_macro - standard_macro
        rng = random.Random(seed)
        samples: List[float] = []
        for _ in range(n_resamples):
            replicate_deltas = []
            for _, _, _, deltas in benchmark_summaries:
                n = len(deltas)
                total = 0.0
                for _ in range(n):
                    total += deltas[rng.randrange(n)]
                replicate_deltas.append(total / n)
            samples.append(sum(replicate_deltas) / len(replicate_deltas))
        low = percentile(samples, 0.025)
        high = percentile(samples, 0.975)
        pvalue = bootstrap_two_sided_pvalue(samples)
        macro_deltas_by_backbone[backbone] = point
        macro_rows.append(
            {
                "backbone": backbone,
                "n_benchmarks": len(benchmark_summaries),
                "standard_macro_accuracy": standard_macro,
                "fadpo_macro_accuracy": fadpo_macro,
                "delta": point,
                "ci_low": low,
                "ci_high": high,
                "bootstrap_p": pvalue,
                "bootstrap_design": "benchmark_stratified_question_level",
            }
        )

    write_csv(output_dir / "accuracy_bootstrap_ci.csv", accuracy_rows)
    write_csv(output_dir / "macro_accuracy_bootstrap_ci.csv", macro_rows)
    write_csv(output_dir / "mcnemar_tests.csv", mcnemar_rows)
    write_json(
        output_dir / "backbone_wilcoxon.json",
        {
            "macro_average_deltas": macro_deltas_by_backbone,
            "exact_wilcoxon_signed_rank": exact_wilcoxon(list(macro_deltas_by_backbone.values())),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-file", type=Path, help="CSV/JSONL paired process metrics.")
    parser.add_argument("--accuracy-file", type=Path, help="CSV/JSONL paired correctness logs.")
    parser.add_argument("--output-dir", type=Path, default=Path("analyses/statistical_ci/outputs"))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.process_file and not args.accuracy_file:
        raise SystemExit("Provide --process-file, --accuracy-file, or both.")
    if args.process_file:
        summarize_process(args.process_file, args.output_dir, args.resamples, args.seed)
    if args.accuracy_file:
        summarize_accuracy(args.accuracy_file, args.output_dir, args.resamples, args.seed)


if __name__ == "__main__":
    main()
