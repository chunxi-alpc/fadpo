#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

if __package__ is None or __package__ == "":
    CURRENT_DIR = Path(__file__).resolve().parent
    PARENT_DIR = CURRENT_DIR.parent
    sys.path.insert(0, str(CURRENT_DIR))
    sys.path.insert(0, str(PARENT_DIR))
    from analyze_qa_faithfulness_errors import (
        ERROR_TYPES,
        NON_ERROR_TYPES,
        render_report,
        summarize_rows,
        write_group_summary_csv,
    )
    from common import ensure_parent, load_jsonl, normalize_text, write_json
    from study_lib import (
        analysis_master_path,
        example_id,
        get_stage_names,
        load_study_config,
        model_run_dir,
        sample_ids_from_file,
        select_entries,
        stage_report_dir,
        stage_sample_path,
    )
else:
    from ..analyze_qa_faithfulness_errors import (
        ERROR_TYPES,
        NON_ERROR_TYPES,
        render_report,
        summarize_rows,
        write_group_summary_csv,
    )
    from ..common import ensure_parent, load_jsonl, normalize_text, write_json
    from .study_lib import (
        analysis_master_path,
        example_id,
        get_stage_names,
        load_study_config,
        model_run_dir,
        sample_ids_from_file,
        select_entries,
        stage_report_dir,
        stage_sample_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build stage-level summaries and tables for the resumable F1-F4 natural-error study."
    )
    parser.add_argument("--config", required=True, help="Path to study config JSON.")
    parser.add_argument("--stage", required=True, help="Stage name, e.g. smoke/pilot/final.")
    parser.add_argument("--models", default="", help="Optional comma-separated model tags or ids.")
    parser.add_argument("--datasets", default="", help="Optional comma-separated dataset keys.")
    return parser.parse_args()


def parse_csv_fields(raw: str) -> List[str]:
    text = normalize_text(raw)
    return [part.strip() for part in text.split(",") if part.strip()]


def resolve_models(config: Mapping[str, Any], raw_models: str) -> List[Dict[str, Any]]:
    if not raw_models:
        return [dict(item) for item in config["models"]]
    wanted = set(parse_csv_fields(raw_models))
    selected = [
        dict(item)
        for item in config["models"]
        if normalize_text(item.get("tag")) in wanted or normalize_text(item.get("id")) in wanted
    ]
    if not selected:
        raise ValueError(f"No models matched {sorted(wanted)}.")
    return selected


def analysis_id(row: Mapping[str, Any]) -> str:
    analysis = row.get("faithfulness_analysis")
    if isinstance(analysis, Mapping):
        value = normalize_text(analysis.get("example_id"))
        if value:
            return value
    return example_id(row)


def filter_rows_by_ids(rows: Iterable[Mapping[str, Any]], allowed_ids: set[str]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows if analysis_id(row) in allowed_ids]


def answer_subset_rows(rows: Iterable[Mapping[str, Any]], exact_match: bool) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in rows:
        analysis = row.get("faithfulness_analysis")
        if not isinstance(analysis, Mapping):
            continue
        answer_match = analysis.get("answer_match")
        if not isinstance(answer_match, Mapping):
            continue
        if answer_match.get("available") is not True:
            continue
        if answer_match.get("exact_match") is exact_match:
            selected.append(dict(row))
    return selected


def optional_rate(summary: Dict[str, Any], key: str) -> Any:
    return summary.get("overall", {}).get(key)


def overall_to_flat_row(
    summary: Dict[str, Any],
    stage: str,
    model_cfg: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
    correct_summary: Dict[str, Any] | None,
    wrong_summary: Dict[str, Any] | None,
) -> Dict[str, Any]:
    overall = summary.get("overall", {})
    row = {
        "stage": stage,
        "model_tag": normalize_text(model_cfg.get("tag")),
        "model_id": normalize_text(model_cfg.get("id")),
        "dataset_key": normalize_text(dataset_cfg.get("key")),
        "dataset_name": normalize_text(dataset_cfg.get("dataset_name") or dataset_cfg.get("key")),
        "dataset_type": normalize_text(dataset_cfg.get("type")),
        "count": overall.get("count", 0),
        "answer_exact_match_rate": overall.get("answer_exact_match_rate"),
        "any_faithfulness_error_rate": overall.get("any_faithfulness_error_rate"),
        "no_faithfulness_error_rate": overall.get("no_faithfulness_error_rate"),
        "single_dominant_error_rate": overall.get("single_dominant_error_rate"),
        "answer_supported_rate": overall.get("answer_supported_rate"),
        "mean_clinical_plausibility": overall.get("mean_clinical_plausibility"),
        "mean_confidence": overall.get("mean_confidence"),
        "correct_subset_count": (correct_summary or {}).get("overall", {}).get("count", 0),
        "correct_subset_any_error_rate": optional_rate(correct_summary or {}, "any_faithfulness_error_rate"),
        "wrong_subset_count": (wrong_summary or {}).get("overall", {}).get("count", 0),
        "wrong_subset_any_error_rate": optional_rate(wrong_summary or {}, "any_faithfulness_error_rate"),
    }
    for error_type in ERROR_TYPES:
        row[f"dominant_{error_type}_rate"] = overall.get("dominant_error_rates", {}).get(error_type)
        row[f"presence_{error_type}_rate"] = overall.get("presence_rates", {}).get(error_type)
        row[f"correct_subset_presence_{error_type}_rate"] = (
            (correct_summary or {}).get("overall", {}).get("presence_rates", {}).get(error_type)
        )
        row[f"wrong_subset_presence_{error_type}_rate"] = (
            (wrong_summary or {}).get("overall", {}).get("presence_rates", {}).get(error_type)
        )
    return row


def combined_summary(rows: Sequence[Dict[str, Any]], label: str) -> Dict[str, Any] | None:
    if not rows:
        return None
    return summarize_rows(
        rows=rows,
        input_file=label,
        annotation_mode="stage_filtered",
        group_fields=["model_id", "dataset_name", "subset", "split"],
    )


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    output_path = ensure_parent(path)
    if not rows:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["empty"])
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_rate(value: Any) -> str:
    if value in ("", None):
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def render_stage_markdown(
    stage: str,
    overall_summary: Dict[str, Any] | None,
    run_rows: Sequence[Mapping[str, Any]],
    model_rows: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Stage Report",
        "",
        f"- Stage: `{stage}`",
        f"- Run count: `{len(run_rows)}`",
    ]
    if overall_summary:
        overall = overall_summary.get("overall", {})
        lines.extend(
            [
                f"- Total rows: `{overall.get('count', 0)}`",
                f"- Overall answer accuracy: `{format_rate(overall.get('answer_exact_match_rate'))}`",
                f"- Overall any-error rate: `{format_rate(overall.get('any_faithfulness_error_rate'))}`",
                f"- Overall F4 presence: `{format_rate(overall.get('presence_rates', {}).get('F4'))}`",
            ]
        )

    if run_rows:
        lines.extend(
            [
                "",
                "## Run-Level Overview",
                "",
                "| Model | Dataset | N | Accuracy | Any Error | Presence F4 | Dominant F4 | Correct-subset Any Error |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in run_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        normalize_text(row.get("model_tag")),
                        normalize_text(row.get("dataset_key")),
                        str(row.get("count", 0)),
                        format_rate(row.get("answer_exact_match_rate")),
                        format_rate(row.get("any_faithfulness_error_rate")),
                        format_rate(row.get("presence_F4_rate")),
                        format_rate(row.get("dominant_F4_rate")),
                        format_rate(row.get("correct_subset_any_error_rate")),
                    ]
                )
                + " |"
            )

    if model_rows:
        lines.extend(
            [
                "",
                "## By Model",
                "",
                "| Model | N | Accuracy | Any Error | Presence F4 | Dominant F4 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in model_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        normalize_text(row.get("model_tag")),
                        str(row.get("count", 0)),
                        format_rate(row.get("answer_exact_match_rate")),
                        format_rate(row.get("any_faithfulness_error_rate")),
                        format_rate(row.get("presence_F4_rate")),
                        format_rate(row.get("dominant_F4_rate")),
                    ]
                )
                + " |"
            )

    if dataset_rows:
        lines.extend(
            [
                "",
                "## By Dataset",
                "",
                "| Dataset | N | Accuracy | Any Error | Presence F4 | Dominant F4 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in dataset_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        normalize_text(row.get("dataset_key")),
                        str(row.get("count", 0)),
                        format_rate(row.get("answer_exact_match_rate")),
                        format_rate(row.get("any_faithfulness_error_rate")),
                        format_rate(row.get("presence_F4_rate")),
                        format_rate(row.get("dominant_F4_rate")),
                    ]
                )
                + " |"
            )

    return "\n".join(lines) + "\n"


def summary_to_slice_row(summary: Dict[str, Any], slice_name: str, slice_value: str) -> Dict[str, Any]:
    overall = summary.get("overall", {})
    row = {
        slice_name: slice_value,
        "count": overall.get("count", 0),
        "answer_exact_match_rate": overall.get("answer_exact_match_rate"),
        "any_faithfulness_error_rate": overall.get("any_faithfulness_error_rate"),
        "single_dominant_error_rate": overall.get("single_dominant_error_rate"),
    }
    for error_type in ERROR_TYPES:
        row[f"dominant_{error_type}_rate"] = overall.get("dominant_error_rates", {}).get(error_type)
        row[f"presence_{error_type}_rate"] = overall.get("presence_rates", {}).get(error_type)
    return row


def write_pivot_csv(
    rows: Sequence[Mapping[str, Any]],
    row_field: str,
    column_field: str,
    value_field: str,
    path: str | Path,
) -> None:
    row_keys = sorted({normalize_text(row.get(row_field)) for row in rows if normalize_text(row.get(row_field))})
    column_keys = sorted(
        {normalize_text(row.get(column_field)) for row in rows if normalize_text(row.get(column_field))}
    )
    output_rows = []
    for row_key in row_keys:
        pivot_row: Dict[str, Any] = {row_field: row_key}
        for column_key in column_keys:
            value = ""
            for row in rows:
                if normalize_text(row.get(row_field)) == row_key and normalize_text(row.get(column_field)) == column_key:
                    value = row.get(value_field, "")
                    break
            pivot_row[column_key] = value
        output_rows.append(pivot_row)
    write_csv(path, output_rows)


def run_stage_analysis(config_path: str, stage: str, models: str = "", datasets: str = "") -> None:
    config = load_study_config(config_path)
    stage_names = get_stage_names(config)
    if stage not in stage_names:
        raise ValueError(f"Unknown stage `{stage}`. Expected one of {stage_names}.")
    selected_models = resolve_models(config, models)

    selected_datasets = (
        select_entries(config["datasets"], datasets, "key")
        if datasets
        else [dict(item) for item in config["datasets"]]
    )

    stage_dir = stage_report_dir(config, stage)
    ensure_parent(stage_dir / "placeholder.txt")

    run_rows: List[Dict[str, Any]] = []
    run_payloads: List[Dict[str, Any]] = []

    for model_cfg in selected_models:
        for dataset_cfg in selected_datasets:
            stage_file = stage_sample_path(config, dataset_cfg, stage)
            master_file = analysis_master_path(config, model_cfg, dataset_cfg)
            if not stage_file.exists() or not master_file.exists():
                continue

            allowed_ids = set(sample_ids_from_file(stage_file))
            filtered_rows = filter_rows_by_ids(load_jsonl(master_file), allowed_ids)
            if not filtered_rows:
                continue

            summary = summarize_rows(
                rows=filtered_rows,
                input_file=str(master_file),
                annotation_mode="stage_filtered",
                group_fields=["model_id", "dataset_name", "subset", "split"],
            )
            correct_rows = answer_subset_rows(filtered_rows, exact_match=True)
            wrong_rows = answer_subset_rows(filtered_rows, exact_match=False)
            correct_summary = combined_summary(correct_rows, f"{master_file}#correct")
            wrong_summary = combined_summary(wrong_rows, f"{master_file}#wrong")

            run_dir = stage_dir / "runs" / normalize_text(model_cfg["tag"]) / normalize_text(dataset_cfg["tag"])
            ensure_parent(run_dir / "placeholder.txt")
            summary_path = run_dir / "summary.json"
            report_path = run_dir / "report.md"
            group_csv_path = run_dir / "groups.csv"
            write_json(summary_path, summary)
            write_group_summary_csv(group_csv_path, summary)
            report_path.write_text(render_report(summary), encoding="utf-8")
            if correct_summary:
                write_json(run_dir / "correct_subset.summary.json", correct_summary)
            if wrong_summary:
                write_json(run_dir / "wrong_subset.summary.json", wrong_summary)

            flat_row = overall_to_flat_row(
                summary=summary,
                stage=stage,
                model_cfg=model_cfg,
                dataset_cfg=dataset_cfg,
                correct_summary=correct_summary,
                wrong_summary=wrong_summary,
            )
            run_rows.append(flat_row)
            run_payloads.append(
                {
                    "model_cfg": model_cfg,
                    "dataset_cfg": dataset_cfg,
                    "rows": filtered_rows,
                    "summary": summary,
                }
            )

    model_rows: List[Dict[str, Any]] = []
    dataset_rows: List[Dict[str, Any]] = []
    overall_summary = combined_summary([row for payload in run_payloads for row in payload["rows"]], f"stage={stage}")

    for model_cfg in selected_models:
        rows = [row for payload in run_payloads if normalize_text(payload["model_cfg"]["tag"]) == normalize_text(model_cfg["tag"]) for row in payload["rows"]]
        summary = combined_summary(rows, f"stage={stage}|model={model_cfg['tag']}")
        if summary:
            row = summary_to_slice_row(summary, "model_tag", normalize_text(model_cfg["tag"]))
            row["model_id"] = normalize_text(model_cfg["id"])
            model_rows.append(row)
            model_dir = stage_dir / "models" / normalize_text(model_cfg["tag"])
            ensure_parent(model_dir / "placeholder.txt")
            write_json(model_dir / "summary.json", summary)
            (model_dir / "report.md").write_text(render_report(summary), encoding="utf-8")

    for dataset_cfg in selected_datasets:
        rows = [row for payload in run_payloads if normalize_text(payload["dataset_cfg"]["tag"]) == normalize_text(dataset_cfg["tag"]) for row in payload["rows"]]
        summary = combined_summary(rows, f"stage={stage}|dataset={dataset_cfg['tag']}")
        if summary:
            row = summary_to_slice_row(summary, "dataset_key", normalize_text(dataset_cfg["key"]))
            row["dataset_name"] = normalize_text(dataset_cfg.get("dataset_name") or dataset_cfg["key"])
            dataset_rows.append(row)
            dataset_dir = stage_dir / "datasets" / normalize_text(dataset_cfg["tag"])
            ensure_parent(dataset_dir / "placeholder.txt")
            write_json(dataset_dir / "summary.json", summary)
            (dataset_dir / "report.md").write_text(render_report(summary), encoding="utf-8")

    write_csv(stage_dir / "run_level_summary.csv", run_rows)
    write_csv(stage_dir / "model_level_summary.csv", model_rows)
    write_csv(stage_dir / "dataset_level_summary.csv", dataset_rows)
    if run_rows:
        write_pivot_csv(run_rows, "model_tag", "dataset_key", "answer_exact_match_rate", stage_dir / "accuracy_pivot.csv")
        write_pivot_csv(
            run_rows,
            "model_tag",
            "dataset_key",
            "any_faithfulness_error_rate",
            stage_dir / "any_error_pivot.csv",
        )
        write_pivot_csv(run_rows, "model_tag", "dataset_key", "presence_F4_rate", stage_dir / "presence_f4_pivot.csv")
        write_pivot_csv(run_rows, "model_tag", "dataset_key", "dominant_F4_rate", stage_dir / "dominant_f4_pivot.csv")

    payload = {
        "stage": stage,
        "run_rows": run_rows,
        "model_rows": model_rows,
        "dataset_rows": dataset_rows,
        "overall_summary": overall_summary,
    }
    write_json(stage_dir / "stage_payload.json", payload)
    if overall_summary:
        write_json(stage_dir / "overall.summary.json", overall_summary)
        (stage_dir / "overall.report.md").write_text(render_report(overall_summary), encoding="utf-8")

    markdown = render_stage_markdown(stage, overall_summary, run_rows, model_rows, dataset_rows)
    (stage_dir / "stage_report.md").write_text(markdown, encoding="utf-8")
    print(f"Wrote stage report to {stage_dir}")


def main() -> None:
    args = parse_args()
    run_stage_analysis(
        config_path=args.config,
        stage=args.stage,
        models=args.models,
        datasets=args.datasets,
    )


if __name__ == "__main__":
    main()
