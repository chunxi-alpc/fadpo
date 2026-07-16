#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import ensure_parent, normalize_text, read_json, safe_float, write_json
else:
    from .common import ensure_parent, normalize_text, read_json, safe_float, write_json


ERROR_TYPES = ("F1", "F2", "F3", "F4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple F1-F4 summary JSON files into study-level CSV/Markdown artifacts."
    )
    parser.add_argument(
        "--summary-file",
        action="append",
        default=[],
        help="Can be passed multiple times.",
    )
    parser.add_argument(
        "--summary-glob",
        action="append",
        default=[],
        help="Glob pattern(s) used to discover summary JSON files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/f1_f4_natural_error/study_aggregate",
        help="Prefix for .json/.csv/.md outputs.",
    )
    return parser.parse_args()


def discover_summary_files(args: argparse.Namespace) -> List[str]:
    paths = []
    seen = set()
    for path in args.summary_file:
        resolved = str(Path(path))
        if resolved not in seen:
            paths.append(resolved)
            seen.add(resolved)
    for pattern in args.summary_glob:
        for path in sorted(Path().glob(pattern)):
            resolved = str(path)
            if resolved not in seen:
                paths.append(resolved)
                seen.add(resolved)
    if not paths:
        raise ValueError("No summary files found. Pass --summary-file or --summary-glob.")
    return paths


def join_values(values: Sequence[str]) -> str:
    cleaned = [normalize_text(value) for value in values if normalize_text(value)]
    return "|".join(cleaned)


def parse_group_key(group_key: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for part in group_key.split("|"):
        piece = part.strip()
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        parsed[normalize_text(key)] = normalize_text(value)
    return parsed


def collect_group_values(summary: Dict[str, Any], field: str) -> List[str]:
    values = set()
    for item in summary.get("group_summaries", []):
        parsed = parse_group_key(normalize_text(item.get("group")))
        value = normalize_text(parsed.get(field))
        if value and value != "NA":
            values.add(value)
    return sorted(values)


def metric_row(summary: Dict[str, Any], summary_file: str) -> Dict[str, Any]:
    overall = summary.get("overall", {})
    model_ids = summary.get("available_model_ids", []) or collect_group_values(summary, "model_id")
    dataset_names = summary.get("available_datasets", []) or collect_group_values(summary, "dataset_name")
    if not dataset_names:
        dataset_names = [Path(normalize_text(summary.get("input_file")) or summary_file).stem]
    profiles = summary.get("analysis_profiles", {})
    row: Dict[str, Any] = {
        "summary_file": summary_file,
        "input_file": normalize_text(summary.get("input_file")),
        "annotation_mode": normalize_text(summary.get("annotation_mode")),
        "dataset_name": join_values(dataset_names),
        "model_id": join_values(model_ids),
        "analysis_profiles": join_values(list(profiles.keys())),
        "count": overall.get("count", 0),
        "answer_exact_match_rate": overall.get("answer_exact_match_rate"),
        "any_faithfulness_error_rate": overall.get("any_faithfulness_error_rate"),
        "no_faithfulness_error_rate": overall.get("no_faithfulness_error_rate"),
        "single_dominant_error_rate": overall.get("single_dominant_error_rate"),
        "answer_supported_rate": overall.get("answer_supported_rate"),
        "mean_clinical_plausibility": overall.get("mean_clinical_plausibility"),
        "mean_confidence": overall.get("mean_confidence"),
    }
    for error_type in ERROR_TYPES:
        row[f"dominant_{error_type}_rate"] = overall.get("dominant_error_rates", {}).get(error_type)
        row[f"presence_{error_type}_rate"] = overall.get("presence_rates", {}).get(error_type)
    return row


def group_metric_rows(summary: Dict[str, Any], summary_file: str) -> List[Dict[str, Any]]:
    top_level = metric_row(summary, summary_file)
    rows = []
    for item in summary.get("group_summaries", []):
        parsed_group = parse_group_key(normalize_text(item.get("group")))
        row = dict(top_level)
        row.update(
            {
                "group": normalize_text(item.get("group")),
                "group_count": item.get("count", 0),
                "group_answer_exact_match_rate": item.get("answer_exact_match_rate"),
                "group_single_dominant_error_rate": item.get("single_dominant_error_rate"),
                "group_answer_supported_rate": item.get("answer_supported_rate"),
                "group_mean_clinical_plausibility": item.get("mean_clinical_plausibility"),
                "group_mean_confidence": item.get("mean_confidence"),
            }
        )
        row.update(parsed_group)
        for error_type in ERROR_TYPES:
            row[f"group_dominant_{error_type}_rate"] = item.get("dominant_error_rates", {}).get(error_type)
            row[f"group_presence_{error_type}_rate"] = item.get("presence_rates", {}).get(error_type)
        rows.append(row)
    return rows


def safe_mean(values: Iterable[Any]) -> float | None:
    nums = [safe_float(value, default=float("nan")) for value in values]
    nums = [value for value in nums if value == value]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 6)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_parent(path)
    if not rows:
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_rate(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def render_markdown(summary_rows: List[Dict[str, Any]]) -> str:
    datasets = sorted({row["dataset_name"] for row in summary_rows if row.get("dataset_name")})
    models = sorted({row["model_id"] for row in summary_rows if row.get("model_id")})

    lines = [
        "# F1-F4 Study Aggregate",
        "",
        f"- Summary files: `{len(summary_rows)}`",
        f"- Distinct datasets: `{len(datasets)}`",
        f"- Distinct model groups: `{len(models)}`",
        "",
        "## Per-Run Overview",
        "",
        "| Dataset | Model | N | Accuracy | Any Error | Dominant F4 | Presence F4 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary_rows, key=lambda item: (item.get("dataset_name", ""), item.get("model_id", ""))):
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("dataset_name", ""),
                    row.get("model_id", ""),
                    str(row.get("count", 0)),
                    format_rate(row.get("answer_exact_match_rate")),
                    format_rate(row.get("any_faithfulness_error_rate")),
                    format_rate(row.get("dominant_F4_rate")),
                    format_rate(row.get("presence_F4_rate")),
                ]
            )
            + " |"
        )

    dataset_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    model_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        if row.get("dataset_name"):
            dataset_buckets[row["dataset_name"]].append(row)
        if row.get("model_id"):
            model_buckets[row["model_id"]].append(row)

    if dataset_buckets:
        lines.extend(
            [
                "",
                "## By Dataset",
                "",
                "| Dataset | Runs | Avg Accuracy | Avg Any Error | Avg Dominant F4 | Avg Presence F4 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for dataset_name, items in sorted(dataset_buckets.items()):
            lines.append(
                "| "
                + " | ".join(
                    [
                        dataset_name,
                        str(len(items)),
                        format_rate(safe_mean(item.get("answer_exact_match_rate") for item in items)),
                        format_rate(safe_mean(item.get("any_faithfulness_error_rate") for item in items)),
                        format_rate(safe_mean(item.get("dominant_F4_rate") for item in items)),
                        format_rate(safe_mean(item.get("presence_F4_rate") for item in items)),
                    ]
                )
                + " |"
            )

    if model_buckets:
        lines.extend(
            [
                "",
                "## By Model",
                "",
                "| Model | Runs | Avg Accuracy | Avg Any Error | Avg Dominant F4 | Avg Presence F4 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for model_id, items in sorted(model_buckets.items()):
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_id,
                        str(len(items)),
                        format_rate(safe_mean(item.get("answer_exact_match_rate") for item in items)),
                        format_rate(safe_mean(item.get("any_faithfulness_error_rate") for item in items)),
                        format_rate(safe_mean(item.get("dominant_F4_rate") for item in items)),
                        format_rate(safe_mean(item.get("presence_F4_rate") for item in items)),
                    ]
                )
                + " |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary_files = discover_summary_files(args)
    summaries = [(path, read_json(path)) for path in summary_files]

    summary_rows = [metric_row(summary, path) for path, summary in summaries]
    group_rows: List[Dict[str, Any]] = []
    for path, summary in summaries:
        group_rows.extend(group_metric_rows(summary, path))

    output_prefix = Path(args.output_prefix)
    payload = {
        "summary_files": summary_files,
        "summary_rows": summary_rows,
        "group_rows": group_rows,
    }
    write_json(str(output_prefix) + ".json", payload)
    write_csv(str(output_prefix) + ".csv", summary_rows)
    write_csv(str(output_prefix) + ".groups.csv", group_rows)
    ensure_parent(str(output_prefix) + ".md")
    Path(str(output_prefix) + ".md").write_text(render_markdown(summary_rows), encoding="utf-8")
    print(f"Wrote aggregate JSON: {output_prefix}.json")
    print(f"Wrote aggregate CSV: {output_prefix}.csv")
    print(f"Wrote aggregate group CSV: {output_prefix}.groups.csv")
    print(f"Wrote aggregate Markdown: {output_prefix}.md")


if __name__ == "__main__":
    main()
