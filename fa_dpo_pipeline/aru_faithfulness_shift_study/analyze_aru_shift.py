#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import ensure_parent, load_jsonl, normalize_text, output_path, write_json, write_jsonl
    from experiment_utils import write_csv
else:
    from ..common import ensure_parent, load_jsonl, normalize_text, output_path, write_json, write_jsonl
    from ..experiment_utils import write_csv


ROLE_ORDER = ("ALL", "O", "W", "D", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze before/after ARU-level unfaithfulness shifts on a paired same-answer subset."
        )
    )
    parser.add_argument(
        "--subset-file",
        required=True,
        help="Same-answer subset JSONL produced by build_same_answer_subset.py.",
    )
    parser.add_argument(
        "--before-score-file",
        required=True,
        help="Verifier score JSONL for the pre-method system (for example Standard DPO).",
    )
    parser.add_argument(
        "--after-score-file",
        required=True,
        help="Verifier score JSONL for the post-method system (for example Fa-DPO).",
    )
    parser.add_argument(
        "--before-label",
        default="Standard DPO",
        help="Display label for the pre-method system.",
    )
    parser.add_argument(
        "--after-label",
        default="Fa-DPO",
        help="Display label for the post-method system.",
    )
    parser.add_argument(
        "--output-dir",
        default=output_path("aru_faithfulness_shift_study"),
        help="Directory for reports and machine-readable outputs.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Number of question-level bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for bootstrap resampling.",
    )
    return parser.parse_args()


def load_subset_ids(path: str) -> List[str]:
    seen = set()
    ordered_ids: List[str] = []
    for row in load_jsonl(path):
        row_id = normalize_text(row.get("id"))
        if not row_id or row_id in seen:
            continue
        seen.add(row_id)
        ordered_ids.append(row_id)
    return ordered_ids


def load_score_lookup(path: str) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(path):
        row_id = normalize_text(row.get("id"))
        if row_id:
            lookup[row_id] = row
    return lookup


def empty_role_counter() -> Dict[str, int]:
    return {
        "total": 0,
        "entail": 0,
        "neutral": 0,
        "contradict": 0,
        "other": 0,
        "unfaithful": 0,
    }


def node_role(node: Mapping[str, Any]) -> str:
    return normalize_text(node.get("type")).upper()[:1]


def node_label(node: Mapping[str, Any]) -> str:
    return normalize_text(node.get("decision_label")).lower()


def collect_role_counts(score_row: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
    details = ((score_row.get("metrics") or {}).get("node_details") or [])
    counts = {role: empty_role_counter() for role in ROLE_ORDER}

    for node in details:
        role = node_role(node)
        if role not in {"O", "W", "D", "C"}:
            continue
        label = node_label(node)
        for bucket in ("ALL", role):
            counts[bucket]["total"] += 1
            if label == "entail":
                counts[bucket]["entail"] += 1
            elif label == "neutral":
                counts[bucket]["neutral"] += 1
                counts[bucket]["unfaithful"] += 1
            elif label == "contradict":
                counts[bucket]["contradict"] += 1
                counts[bucket]["unfaithful"] += 1
            else:
                counts[bucket]["other"] += 1
                counts[bucket]["unfaithful"] += 1
    return counts


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def exact_two_sided_sign_pvalue(improved: int, worsened: int) -> float:
    non_ties = improved + worsened
    if non_ties == 0:
        return 1.0
    smaller_tail = min(improved, worsened)
    tail_mass = sum(math.comb(non_ties, k) for k in range(smaller_tail + 1)) / (2**non_ties)
    return round(min(1.0, 2.0 * tail_mass), 6)


def bootstrap_mean_ci(values: Sequence[float], *, samples: int, seed: int) -> Tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    count = len(values)
    means: List[float] = []
    for _ in range(samples):
        sampled = [values[rng.randrange(count)] for _ in range(count)]
        means.append(sum(sampled) / count)
    means.sort()
    low_index = max(0, min(len(means) - 1, int(0.025 * len(means))))
    high_index = max(0, min(len(means) - 1, int(0.975 * len(means))))
    return round(means[low_index], 6), round(means[high_index], 6)


def build_per_example_rows(
    subset_ids: Sequence[str],
    before_lookup: Mapping[str, Mapping[str, Any]],
    after_lookup: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    rows: List[Dict[str, Any]] = []
    missing_before: List[str] = []
    missing_after: List[str] = []

    for row_id in subset_ids:
        before_row = before_lookup.get(row_id)
        after_row = after_lookup.get(row_id)
        if before_row is None:
            missing_before.append(row_id)
        if after_row is None:
            missing_after.append(row_id)
        if before_row is None or after_row is None:
            continue

        before_counts = collect_role_counts(before_row)
        after_counts = collect_role_counts(after_row)
        for role in ROLE_ORDER:
            before_role = before_counts[role]
            after_role = after_counts[role]
            before_rate = safe_rate(before_role["unfaithful"], before_role["total"])
            after_rate = safe_rate(after_role["unfaithful"], after_role["total"])
            rows.append(
                {
                    "id": row_id,
                    "role": role,
                    "before_total_arus": before_role["total"],
                    "before_unfaithful_arus": before_role["unfaithful"],
                    "before_entail_arus": before_role["entail"],
                    "before_neutral_arus": before_role["neutral"],
                    "before_contradict_arus": before_role["contradict"],
                    "before_other_arus": before_role["other"],
                    "before_unfaithful_rate": before_rate,
                    "after_total_arus": after_role["total"],
                    "after_unfaithful_arus": after_role["unfaithful"],
                    "after_entail_arus": after_role["entail"],
                    "after_neutral_arus": after_role["neutral"],
                    "after_contradict_arus": after_role["contradict"],
                    "after_other_arus": after_role["other"],
                    "after_unfaithful_rate": after_rate,
                    "delta_unfaithful_rate": (
                        round(after_rate - before_rate, 6)
                        if before_rate is not None and after_rate is not None
                        else None
                    ),
                }
            )

    return rows, {"missing_before": missing_before, "missing_after": missing_after}


def summarize_role(
    role: str,
    role_rows: Sequence[Mapping[str, Any]],
    *,
    before_label: str,
    after_label: str,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    paired_rows = [
        row
        for row in role_rows
        if int(row.get("before_total_arus", 0)) > 0 and int(row.get("after_total_arus", 0)) > 0
    ]

    before_total = sum(int(row.get("before_total_arus", 0)) for row in paired_rows)
    before_unfaithful = sum(int(row.get("before_unfaithful_arus", 0)) for row in paired_rows)
    before_neutral = sum(int(row.get("before_neutral_arus", 0)) for row in paired_rows)
    before_contradict = sum(int(row.get("before_contradict_arus", 0)) for row in paired_rows)

    after_total = sum(int(row.get("after_total_arus", 0)) for row in paired_rows)
    after_unfaithful = sum(int(row.get("after_unfaithful_arus", 0)) for row in paired_rows)
    after_neutral = sum(int(row.get("after_neutral_arus", 0)) for row in paired_rows)
    after_contradict = sum(int(row.get("after_contradict_arus", 0)) for row in paired_rows)

    before_rate = safe_rate(before_unfaithful, before_total) if paired_rows else None
    after_rate = safe_rate(after_unfaithful, after_total) if paired_rows else None
    before_neutral_rate = safe_rate(before_neutral, before_total) if paired_rows else None
    before_contradict_rate = safe_rate(before_contradict, before_total) if paired_rows else None
    after_neutral_rate = safe_rate(after_neutral, after_total) if paired_rows else None
    after_contradict_rate = safe_rate(after_contradict, after_total) if paired_rows else None

    deltas = [float(row["delta_unfaithful_rate"]) for row in paired_rows if row.get("delta_unfaithful_rate") is not None]
    improved = sum(1 for delta in deltas if delta < 0)
    worsened = sum(1 for delta in deltas if delta > 0)
    ties = sum(1 for delta in deltas if delta == 0)
    ci_low, ci_high = bootstrap_mean_ci(deltas, samples=bootstrap_samples, seed=seed)

    return {
        "role": role,
        "paired_examples": len(paired_rows),
        "before_label": before_label,
        "after_label": after_label,
        "before_total_arus": before_total,
        "before_unfaithful_arus": before_unfaithful,
        "before_unfaithful_rate": before_rate,
        "before_neutral_rate": before_neutral_rate,
        "before_contradict_rate": before_contradict_rate,
        "after_total_arus": after_total,
        "after_unfaithful_arus": after_unfaithful,
        "after_unfaithful_rate": after_rate,
        "after_neutral_rate": after_neutral_rate,
        "after_contradict_rate": after_contradict_rate,
        "pooled_delta_rate": round(after_rate - before_rate, 6) if before_rate is not None and after_rate is not None else None,
        "mean_before_rate": round(mean(float(row["before_unfaithful_rate"]) for row in paired_rows), 6)
        if paired_rows
        else None,
        "mean_after_rate": round(mean(float(row["after_unfaithful_rate"]) for row in paired_rows), 6)
        if paired_rows
        else None,
        "mean_delta_rate": round(mean(deltas), 6) if deltas else None,
        "median_delta_rate": round(median(deltas), 6) if deltas else None,
        "improved_examples": improved,
        "worsened_examples": worsened,
        "tied_examples": ties,
        "sign_test_pvalue": exact_two_sided_sign_pvalue(improved, worsened) if deltas else None,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }


def pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:.1f}"


def delta_pp(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:+.1f}"


def format_pvalue(value: float | None) -> str:
    if value is None:
        return "--"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def render_report(
    args: argparse.Namespace,
    subset_ids: Sequence[str],
    matched_questions: int,
    coverage: Mapping[str, Sequence[str]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# ARU Faithfulness Shift Report",
        "",
        "## Inputs",
        "",
        f"- subset file: `{args.subset_file}`",
        f"- before score file: `{args.before_score_file}`",
        f"- after score file: `{args.after_score_file}`",
        f"- before label: `{args.before_label}`",
        f"- after label: `{args.after_label}`",
        "",
        "## Design",
        "",
        "- subset restriction: same-answer hard subset",
        "- unfaithful ARU definition: `decision_label != entail`",
        "- descriptive rate: pooled ARU proportion within the paired-question set",
        "- inferential unit: question-level paired role-wise unfaithful rate",
        "- statistical test: two-sided paired sign test",
        f"- uncertainty: {args.bootstrap_samples} bootstrap resamples for the mean paired delta",
        "",
        "## Coverage",
        "",
        f"- requested subset questions: {len(subset_ids)}",
        f"- matched questions with both score rows: {matched_questions}",
        f"- missing before rows: {len(coverage.get('missing_before', []))}",
        f"- missing after rows: {len(coverage.get('missing_after', []))}",
        "",
        "## Role Summary",
        "",
        "| Role | Paired N | Before (%) | After (%) | Delta (pp) | 95% CI (mean delta, pp) | p-value |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]

    for row in summary_rows:
        ci_text = "--"
        if row.get("bootstrap_ci_low") is not None and row.get("bootstrap_ci_high") is not None:
            ci_text = (
                f"[{delta_pp(row.get('bootstrap_ci_low'))}, "
                f"{delta_pp(row.get('bootstrap_ci_high'))}]"
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("role", "")),
                    str(row.get("paired_examples", 0)),
                    pct(row.get("before_unfaithful_rate")),
                    pct(row.get("after_unfaithful_rate")),
                    delta_pp(row.get("pooled_delta_rate")),
                    ci_text,
                    format_pvalue(row.get("sign_test_pvalue")),
                ]
            )
            + " |"
        )

    if coverage.get("missing_before"):
        lines.extend(
            [
                "",
                "## Missing Before Rows",
                "",
                f"`{coverage['missing_before'][:20]}`",
            ]
        )
    if coverage.get("missing_after"):
        lines.extend(
            [
                "",
                "## Missing After Rows",
                "",
                f"`{coverage['missing_after'][:20]}`",
            ]
        )
    return "\n".join(lines) + "\n"


def render_paper_table(summary_rows: Sequence[Mapping[str, Any]], before_label: str, after_label: str) -> str:
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\footnotesize",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    \\caption{ARU-level unfaithfulness shift on the same-answer hard subset. "
        "An ARU is counted as unfaithful when its routed verifier label is non-entailment. "
        "The percentage columns report pooled ARU proportions, while the confidence interval and $p$-value are computed from question-level paired rate differences to avoid treating non-aligned ARUs as independent samples.}",
        "    \\label{tab:aru_unfaithfulness_shift_placeholder}",
        "    \\begin{adjustbox}{max width=\\columnwidth}",
        "    \\begin{tabular}{lccccc}",
        "        \\toprule",
        f"        \\textbf{{Role}} & \\textbf{{{before_label} (\\%)}} & \\textbf{{{after_label} (\\%)}} & \\textbf{{$\\Delta$ (pp)}} & \\textbf{{95\\% CI}} & \\textbf{{$p$-value}} \\\\",
        "        \\midrule",
    ]

    for row in summary_rows:
        ci_text = "--"
        if row.get("bootstrap_ci_low") is not None and row.get("bootstrap_ci_high") is not None:
            ci_text = (
                f"[{delta_pp(row.get('bootstrap_ci_low'))}, "
                f"{delta_pp(row.get('bootstrap_ci_high'))}]"
            )
        lines.append(
            "        "
            + " & ".join(
                [
                    str(row.get("role", "")),
                    pct(row.get("before_unfaithful_rate")),
                    pct(row.get("after_unfaithful_rate")),
                    delta_pp(row.get("pooled_delta_rate")),
                    ci_text,
                    format_pvalue(row.get("sign_test_pvalue")),
                ]
            )
            + " \\\\"
        )

    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "    \\end{adjustbox}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    subset_ids = load_subset_ids(args.subset_file)
    before_lookup = load_score_lookup(args.before_score_file)
    after_lookup = load_score_lookup(args.after_score_file)

    per_example_rows, coverage = build_per_example_rows(subset_ids, before_lookup, after_lookup)
    role_summary_rows: List[Dict[str, Any]] = []
    for role_index, role in enumerate(ROLE_ORDER):
        role_rows = [row for row in per_example_rows if row.get("role") == role]
        role_summary_rows.append(
            summarize_role(
                role,
                role_rows,
                before_label=args.before_label,
                after_label=args.after_label,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + role_index,
            )
        )

    matched_questions = len({row["id"] for row in per_example_rows if row.get("role") == "ALL"})
    output_dir = ensure_parent(Path(args.output_dir) / "summary.json").parent
    per_example_path = output_dir / "per_example_rates.jsonl"
    summary_json_path = output_dir / "summary.json"
    summary_csv_path = output_dir / "role_summary.csv"
    report_path = output_dir / "report.md"
    paper_table_path = output_dir / "paper_table.tex"

    report_text = render_report(args, subset_ids, matched_questions, coverage, role_summary_rows)
    paper_table_text = render_paper_table(role_summary_rows, args.before_label, args.after_label)

    write_jsonl(per_example_path, per_example_rows)
    write_csv(summary_csv_path, role_summary_rows)
    write_json(
        summary_json_path,
        {
            "inputs": {
                "subset_file": args.subset_file,
                "before_score_file": args.before_score_file,
                "after_score_file": args.after_score_file,
                "before_label": args.before_label,
                "after_label": args.after_label,
            },
            "coverage": {
                "requested_subset_questions": len(subset_ids),
                "matched_questions": matched_questions,
                "missing_before": list(coverage.get("missing_before", [])),
                "missing_after": list(coverage.get("missing_after", [])),
            },
            "role_summary": role_summary_rows,
        },
    )
    report_path.write_text(report_text, encoding="utf-8")
    paper_table_path.write_text(paper_table_text, encoding="utf-8")

    print(f"Per-example output: {per_example_path}")
    print(f"Role summary CSV: {summary_csv_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Report: {report_path}")
    print(f"Paper table: {paper_table_path}")


if __name__ == "__main__":
    main()
