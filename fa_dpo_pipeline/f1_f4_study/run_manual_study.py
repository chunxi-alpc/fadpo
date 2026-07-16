#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping

if __package__ is None or __package__ == "":
    CURRENT_DIR = Path(__file__).resolve().parent
    PARENT_DIR = CURRENT_DIR.parent
    sys.path.insert(0, str(CURRENT_DIR))
    sys.path.insert(0, str(PARENT_DIR))
    from analyze_qa_faithfulness_errors import (
        SYSTEM_PROMPT as JUDGE_SYSTEM_PROMPT,
        build_output_row,
        build_user_prompt,
        completed_ids as completed_analysis_ids,
        normalize_judge_payload,
        normalize_row,
        render_report,
        summarize_rows,
        write_group_summary_csv,
    )
    from common import ensure_parent, load_jsonl, normalize_text, write_json, write_jsonl
    from generate_case_reasoning_eval import (
        SYSTEM_PROMPT as CASE_GENERATION_SYSTEM_PROMPT,
        build_prompt as build_case_generation_prompt,
        parse_output as parse_case_output,
    )
    from generate_mcq_reasoning_eval import (
        SYSTEM_PROMPT as MCQ_GENERATION_SYSTEM_PROMPT,
        build_prompt as build_mcq_generation_prompt,
        get_options_map,
        parse_output as parse_mcq_output,
    )
    from run_study import prepare_dataset, resolve_datasets, resolve_models
    from study_lib import (
        analysis_master_group_csv_path,
        analysis_master_path,
        analysis_master_report_path,
        analysis_master_summary_path,
        ensure_sample_pool,
        generation_master_path,
        get_stage_names,
        load_study_config,
        manual_generation_jobs_path,
        manual_generation_results_path,
        manual_judge_jobs_path,
        manual_judge_results_path,
        materialize_stage_sample,
        model_run_dir,
        prepared_dataset_path,
        save_resolved_config,
        stage_prefix,
    )
else:
    from ..analyze_qa_faithfulness_errors import (
        SYSTEM_PROMPT as JUDGE_SYSTEM_PROMPT,
        build_output_row,
        build_user_prompt,
        completed_ids as completed_analysis_ids,
        normalize_judge_payload,
        normalize_row,
        render_report,
        summarize_rows,
        write_group_summary_csv,
    )
    from ..common import ensure_parent, load_jsonl, normalize_text, write_json, write_jsonl
    from ..generate_case_reasoning_eval import (
        SYSTEM_PROMPT as CASE_GENERATION_SYSTEM_PROMPT,
        build_prompt as build_case_generation_prompt,
        parse_output as parse_case_output,
    )
    from ..generate_mcq_reasoning_eval import (
        SYSTEM_PROMPT as MCQ_GENERATION_SYSTEM_PROMPT,
        build_prompt as build_mcq_generation_prompt,
        get_options_map,
        parse_output as parse_mcq_output,
    )
    from .run_study import prepare_dataset, resolve_datasets, resolve_models
    from .study_lib import (
        analysis_master_group_csv_path,
        analysis_master_path,
        analysis_master_report_path,
        analysis_master_summary_path,
        ensure_sample_pool,
        generation_master_path,
        get_stage_names,
        load_study_config,
        manual_generation_jobs_path,
        manual_generation_results_path,
        manual_judge_jobs_path,
        manual_judge_results_path,
        materialize_stage_sample,
        model_run_dir,
        prepared_dataset_path,
        save_resolved_config,
        stage_prefix,
    )


PHASES = (
    "prepare",
    "sample",
    "export_generate",
    "import_generate",
    "export_judge",
    "import_judge",
    "analyze",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the manual/non-HTTP backend for the F1-F4 study. "
            "This exports generation/judge job files, then imports assistant-produced results."
        )
    )
    parser.add_argument("--config", required=True, help="Path to study config JSON.")
    parser.add_argument("--stage", required=True, help="Stage name, e.g. smoke/pilot/final.")
    parser.add_argument(
        "--phases",
        default="prepare,sample,export_generate",
        help="Comma-separated phases to run.",
    )
    parser.add_argument("--models", default="", help="Optional comma-separated model tags or ids.")
    parser.add_argument("--datasets", default="", help="Optional comma-separated dataset keys.")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-sample-pool", action="store_true")
    parser.add_argument("--force-stage-files", action="store_true")
    parser.add_argument("--overwrite-exported-jobs", action="store_true")
    return parser.parse_args()


def parse_csv_fields(raw: str) -> List[str]:
    text = normalize_text(raw)
    return [part.strip() for part in text.split(",") if part.strip()]


def normalize_generation_job(row: Mapping[str, Any], dataset_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    dataset_type = normalize_text(dataset_cfg["type"])
    if dataset_type == "case":
        system_prompt = CASE_GENERATION_SYSTEM_PROMPT
        user_prompt = build_case_generation_prompt(dict(row))
    else:
        system_prompt = MCQ_GENERATION_SYSTEM_PROMPT
        user_prompt = build_mcq_generation_prompt(dict(row))
    return {
        "job_type": "generation",
        "example_id": normalize_text(row.get("id") or row.get("source_id")),
        "dataset_key": normalize_text(dataset_cfg["key"]),
        "dataset_type": dataset_type,
        "model_tag": normalize_text(model_cfg["tag"]),
        "model_id": normalize_text(model_cfg["id"]),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "source_row": dict(row),
        "response_requirements": {
            "required_fields": ["example_id", "output"],
            "optional_fields": ["predicted_reasoning", "predicted_answer_text", "predicted_answer_label"],
        },
    }


def minimal_args_namespace() -> SimpleNamespace:
    return SimpleNamespace(
        id_field="",
        question_field="",
        evidence_field="",
        reference_reasoning_field="",
        reference_answer_field="",
        prediction_field="",
        predicted_reasoning_field="",
        predicted_answer_field="",
    )


def normalize_judge_job(row: Mapping[str, Any], dataset_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    analysis_profile = "case_full" if normalize_text(dataset_cfg["type"]) == "case" else "mcq_f4"
    normalized = normalize_row(
        row=dict(row),
        index=0,
        group_fields=["model_id", "dataset_name", "subset", "split"],
        args=minimal_args_namespace(),
    )
    return {
        "job_type": "judge",
        "example_id": normalize_text(normalized["example_id"]),
        "dataset_key": normalize_text(dataset_cfg["key"]),
        "dataset_type": normalize_text(dataset_cfg["type"]),
        "model_tag": normalize_text(model_cfg["tag"]),
        "model_id": normalize_text(model_cfg["id"]),
        "analysis_profile": analysis_profile,
        "system_prompt": JUDGE_SYSTEM_PROMPT,
        "user_prompt": build_user_prompt(normalized, analysis_profile),
        "source_row": dict(row),
        "response_requirements": {
            "required_fields": [
                "example_id",
                "dominant_error_type",
                "has_f1",
                "has_f2",
                "has_f3",
                "has_f4",
                "single_dominant_error",
                "answer_supported_by_reasoning",
                "clinical_plausibility_score",
                "confidence_score",
                "short_rationale",
            ]
        },
    }


def sample_row_map(stage_file: str | Path) -> Dict[str, Dict[str, Any]]:
    rows = list(load_jsonl(stage_file))
    return {
        normalize_text(row.get("id") or row.get("source_id")): dict(row)
        for row in rows
        if normalize_text(row.get("id") or row.get("source_id"))
    }


def analysis_example_id(row: Mapping[str, Any]) -> str:
    analysis = row.get("faithfulness_analysis")
    if isinstance(analysis, Mapping):
        value = normalize_text(analysis.get("example_id"))
        if value:
            return value
    return normalize_text(row.get("id") or row.get("source_id") or row.get("example_id"))


def ordered_upsert_rows(
    existing_rows: List[Mapping[str, Any]],
    new_rows: List[Mapping[str, Any]],
    *,
    id_getter,
) -> List[Dict[str, Any]]:
    ordered_ids: List[str] = []
    row_map: Dict[str, Dict[str, Any]] = {}
    passthrough_rows: List[Dict[str, Any]] = []

    for row in list(existing_rows) + list(new_rows):
        row_id = normalize_text(id_getter(row))
        if not row_id:
            passthrough_rows.append(dict(row))
            continue
        if row_id not in row_map:
            ordered_ids.append(row_id)
        row_map[row_id] = dict(row)

    return [row_map[row_id] for row_id in ordered_ids] + passthrough_rows


def merge_generation_result(
    source_row: Mapping[str, Any],
    result_row: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = dict(source_row)
    output = normalize_text(result_row.get("output"))
    if not output:
        raise ValueError(f"Generation result for {result_row.get('example_id')} is missing `output`.")
    if normalize_text(dataset_cfg["type"]) == "case":
        parsed = parse_case_output(output)
    else:
        parsed = parse_mcq_output(output, get_options_map(source_row.get("options")))
    merged.update(parsed)
    for field in ("predicted_reasoning", "predicted_answer_text", "predicted_answer_label"):
        value = normalize_text(result_row.get(field))
        if value:
            merged[field] = value
    merged["model_id"] = normalize_text(result_row.get("model_id") or model_cfg["id"])
    merged["generation_model_snapshot"] = normalize_text(
        result_row.get("generation_model_snapshot") or f"manual::{model_cfg['tag']}"
    )
    return merged


def export_generation_jobs(
    config: Mapping[str, Any],
    stage: str,
    dataset_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    overwrite: bool,
) -> Path:
    stage_file = materialize_stage_sample(config, dataset_cfg, stage, ensure_sample_pool(config, dataset_cfg))
    stage_rows = list(load_jsonl(stage_file))
    master_file = generation_master_path(config, model_cfg, dataset_cfg)
    existing_ids = {
        normalize_text(row.get("id") or row.get("source_id"))
        for row in (load_jsonl(master_file) if master_file.exists() else [])
    }
    jobs = [
        normalize_generation_job(row, dataset_cfg, model_cfg)
        for row in stage_rows
        if normalize_text(row.get("id") or row.get("source_id")) not in existing_ids
    ]
    jobs_path = manual_generation_jobs_path(config, stage, model_cfg, dataset_cfg)
    if jobs_path.exists() and not overwrite and not jobs:
        return jobs_path
    ensure_parent(jobs_path)
    write_jsonl(jobs_path, jobs)
    print(f"Generation jobs: {jobs_path} ({len(jobs)} pending)")
    return jobs_path


def import_generation_results(
    config: Mapping[str, Any],
    stage: str,
    dataset_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
) -> int:
    stage_file = materialize_stage_sample(config, dataset_cfg, stage, ensure_sample_pool(config, dataset_cfg))
    stage_rows = sample_row_map(stage_file)
    results_file = manual_generation_results_path(config, stage, model_cfg, dataset_cfg)
    master_file = generation_master_path(config, model_cfg, dataset_cfg)
    if not results_file.exists():
        print(f"Skip generation import; no results file: {results_file}")
        return 0
    existing_rows = list(load_jsonl(master_file)) if master_file.exists() else []
    existing_ids = {
        normalize_text(row.get("id") or row.get("source_id"))
        for row in existing_rows
    }
    imported = 0
    imported_rows: List[Dict[str, Any]] = []
    for result_row in load_jsonl(results_file):
        row_id = normalize_text(result_row.get("example_id") or result_row.get("id") or result_row.get("source_id"))
        if not row_id or row_id not in stage_rows or row_id in existing_ids:
            continue
        merged = merge_generation_result(stage_rows[row_id], result_row, dataset_cfg, model_cfg)
        imported_rows.append(merged)
        existing_ids.add(row_id)
        imported += 1
    deduped_rows = ordered_upsert_rows(
        existing_rows,
        imported_rows,
        id_getter=lambda row: row.get("id") or row.get("source_id"),
    )
    write_jsonl(master_file, deduped_rows)
    print(f"Imported generation rows into {master_file}: {imported}")
    return imported


def export_judge_jobs(
    config: Mapping[str, Any],
    stage: str,
    dataset_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    overwrite: bool,
) -> Path:
    stage_file = materialize_stage_sample(config, dataset_cfg, stage, ensure_sample_pool(config, dataset_cfg))
    stage_ids = set(sample_row_map(stage_file).keys())
    generation_file = generation_master_path(config, model_cfg, dataset_cfg)
    if not generation_file.exists():
        raise FileNotFoundError(f"Missing generation master file: {generation_file}")
    pending_rows = [
        row
        for row in load_jsonl(generation_file)
        if normalize_text(row.get("id") or row.get("source_id")) in stage_ids
    ]
    analysis_file = analysis_master_path(config, model_cfg, dataset_cfg)
    done_ids = completed_analysis_ids(str(analysis_file))
    jobs = [
        normalize_judge_job(row, dataset_cfg, model_cfg)
        for row in pending_rows
        if normalize_text(row.get("id") or row.get("source_id")) not in done_ids
    ]
    jobs_path = manual_judge_jobs_path(config, stage, model_cfg, dataset_cfg)
    if jobs_path.exists() and not overwrite and not jobs:
        return jobs_path
    ensure_parent(jobs_path)
    write_jsonl(jobs_path, jobs)
    print(f"Judge jobs: {jobs_path} ({len(jobs)} pending)")
    return jobs_path


def import_judge_results(
    config: Mapping[str, Any],
    stage: str,
    dataset_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
) -> int:
    stage_file = materialize_stage_sample(config, dataset_cfg, stage, ensure_sample_pool(config, dataset_cfg))
    stage_ids = set(sample_row_map(stage_file).keys())
    generation_file = generation_master_path(config, model_cfg, dataset_cfg)
    if not generation_file.exists():
        raise FileNotFoundError(f"Missing generation master file: {generation_file}")
    generation_rows = {
        normalize_text(row.get("id") or row.get("source_id")): dict(row)
        for row in load_jsonl(generation_file)
        if normalize_text(row.get("id") or row.get("source_id")) in stage_ids
    }
    results_file = manual_judge_results_path(config, stage, model_cfg, dataset_cfg)
    analysis_file = analysis_master_path(config, model_cfg, dataset_cfg)
    if not results_file.exists():
        print(f"Skip judge import; no results file: {results_file}")
        return 0

    existing_rows = list(load_jsonl(analysis_file)) if analysis_file.exists() else []
    done_ids = completed_analysis_ids(str(analysis_file))
    imported = 0
    imported_rows: List[Dict[str, Any]] = []
    for result_row in load_jsonl(results_file):
        row_id = normalize_text(result_row.get("example_id") or result_row.get("id") or result_row.get("source_id"))
        if not row_id or row_id not in generation_rows or row_id in done_ids:
            continue
        source_row = generation_rows[row_id]
        analysis_profile = "case_full" if normalize_text(dataset_cfg["type"]) == "case" else "mcq_f4"
        normalized = normalize_row(
            row=dict(source_row),
            index=0,
            group_fields=["model_id", "dataset_name", "subset", "split"],
            args=minimal_args_namespace(),
        )
        payload = dict(result_row.get("faithfulness_analysis") or result_row)
        analysis = normalize_judge_payload(payload)
        analysis["status"] = "manual_judged"
        analysis["judge_model_requested"] = f"manual::{model_cfg['tag']}"
        analysis["judge_model_snapshot"] = f"manual::{model_cfg['tag']}"
        output_row = build_output_row(source_row, normalized, analysis_profile, analysis)
        imported_rows.append(output_row)
        done_ids.add(row_id)
        imported += 1

    all_rows = ordered_upsert_rows(
        existing_rows,
        imported_rows,
        id_getter=analysis_example_id,
    )
    if all_rows:
        write_jsonl(analysis_file, all_rows)
        summary = summarize_rows(
            rows=all_rows,
            input_file=str(generation_file),
            annotation_mode="manual",
            group_fields=["model_id", "dataset_name", "subset", "split"],
        )
        write_json(analysis_master_summary_path(config, model_cfg, dataset_cfg), summary)
        write_group_summary_csv(analysis_master_group_csv_path(config, model_cfg, dataset_cfg), summary)
        analysis_master_report_path(config, model_cfg, dataset_cfg).write_text(
            render_report(summary),
            encoding="utf-8",
        )

    print(f"Imported judge rows into {analysis_file}: {imported}")
    return imported


def run_stage_analysis(config_path: str, stage: str, models: str, datasets: str) -> None:
    if __package__ is None or __package__ == "":
        from analyze_stage_results import run_stage_analysis as _run_stage_analysis
    else:
        from .analyze_stage_results import run_stage_analysis as _run_stage_analysis

    _run_stage_analysis(config_path=config_path, stage=stage, models=models, datasets=datasets)


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    stage_names = get_stage_names(config)
    if args.stage not in stage_names:
        raise ValueError(f"Unknown stage `{args.stage}`. Expected one of {stage_names}.")

    phases = parse_csv_fields(args.phases)
    invalid_phases = [phase for phase in phases if phase not in PHASES]
    if invalid_phases:
        raise ValueError(f"Invalid phases: {invalid_phases}. Expected subset of {PHASES}.")

    selected_models = resolve_models(config, args.models)
    selected_datasets = resolve_datasets(config, args.datasets)
    resolved_config_path = save_resolved_config(config, args.config)

    if "prepare" in phases:
        for dataset_cfg in selected_datasets:
            prepare_dataset(config, dataset_cfg, force=args.force_prepare, dry_run=False)

    if "sample" in phases:
        for dataset_cfg in selected_datasets:
            sample_pool = ensure_sample_pool(config, dataset_cfg, force=args.force_sample_pool)
            for stage_name in stage_prefix(stage_names, args.stage):
                stage_file = materialize_stage_sample(
                    config,
                    dataset_cfg,
                    stage_name,
                    sample_pool,
                    force=args.force_stage_files,
                )
                print(f"Stage sample ready: {stage_file}")

    if "export_generate" in phases:
        for model_cfg in selected_models:
            for dataset_cfg in selected_datasets:
                ensure_parent(model_run_dir(config, model_cfg, dataset_cfg) / "placeholder.txt")
                export_generation_jobs(
                    config=config,
                    stage=args.stage,
                    dataset_cfg=dataset_cfg,
                    model_cfg=model_cfg,
                    overwrite=args.overwrite_exported_jobs,
                )

    if "import_generate" in phases:
        for model_cfg in selected_models:
            for dataset_cfg in selected_datasets:
                import_generation_results(config, args.stage, dataset_cfg, model_cfg)

    if "export_judge" in phases:
        for model_cfg in selected_models:
            for dataset_cfg in selected_datasets:
                export_judge_jobs(
                    config=config,
                    stage=args.stage,
                    dataset_cfg=dataset_cfg,
                    model_cfg=model_cfg,
                    overwrite=args.overwrite_exported_jobs,
                )

    if "import_judge" in phases:
        for model_cfg in selected_models:
            for dataset_cfg in selected_datasets:
                import_judge_results(config, args.stage, dataset_cfg, model_cfg)

    if "analyze" in phases:
        run_stage_analysis(
            config_path=str(resolved_config_path),
            stage=args.stage,
            models=args.models,
            datasets=args.datasets,
        )


if __name__ == "__main__":
    main()
