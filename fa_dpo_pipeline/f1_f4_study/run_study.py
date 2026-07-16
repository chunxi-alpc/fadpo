#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ is None or __package__ == "":
    CURRENT_DIR = Path(__file__).resolve().parent
    PARENT_DIR = CURRENT_DIR.parent
    sys.path.insert(0, str(CURRENT_DIR))
    sys.path.insert(0, str(PARENT_DIR))
    from common import ensure_parent, normalize_text
    from study_lib import (
        analysis_master_group_csv_path,
        analysis_master_path,
        analysis_master_report_path,
        analysis_master_summary_path,
        call_python,
        ensure_sample_pool,
        generation_master_path,
        get_stage_names,
        load_study_config,
        materialize_stage_sample,
        model_run_dir,
        prepared_dataset_path,
        save_resolved_config,
        select_entries,
        stage_prefix,
    )
else:
    from ..common import ensure_parent, normalize_text
    from .study_lib import (
        analysis_master_group_csv_path,
        analysis_master_path,
        analysis_master_report_path,
        analysis_master_summary_path,
        call_python,
        ensure_sample_pool,
        generation_master_path,
        get_stage_names,
        load_study_config,
        materialize_stage_sample,
        model_run_dir,
        prepared_dataset_path,
        save_resolved_config,
        select_entries,
        stage_prefix,
    )


PHASES = ("prepare", "sample", "generate", "judge", "analyze")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the resumable F1-F4 natural-error study end to end. "
            "Stages expand monotonically, and generation/judging reuses prior outputs."
        )
    )
    parser.add_argument("--config", required=True, help="Path to study config JSON.")
    parser.add_argument("--stage", required=True, help="Stage name, e.g. smoke/pilot/final.")
    parser.add_argument(
        "--phases",
        default="prepare,sample,generate,judge,analyze",
        help="Comma-separated phases to run.",
    )
    parser.add_argument("--models", default="", help="Optional comma-separated model tags or ids.")
    parser.add_argument("--datasets", default="", help="Optional comma-separated dataset keys.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-sample-pool", action="store_true")
    parser.add_argument("--force-stage-files", action="store_true")
    return parser.parse_args()


def parse_csv_fields(raw: str) -> List[str]:
    text = normalize_text(raw)
    return [part.strip() for part in text.split(",") if part.strip()]


def resolve_models(config: Dict[str, Any], raw_models: str) -> List[Dict[str, Any]]:
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


def resolve_datasets(config: Dict[str, Any], raw_datasets: str) -> List[Dict[str, Any]]:
    if not raw_datasets:
        return [dict(item) for item in config["datasets"]]
    return select_entries(config["datasets"], raw_datasets, "key")


def prepare_dataset(config: Dict[str, Any], dataset_cfg: Dict[str, Any], force: bool, dry_run: bool) -> None:
    output_file = prepared_dataset_path(config, dataset_cfg)
    if output_file.exists() and not force:
        print(f"Skip prepare: {output_file}")
        return

    ensure_parent(output_file)
    if dataset_cfg["type"] == "case":
        args = [
            "--input-csv",
            normalize_text(dataset_cfg["input_file"]),
            "--output-file",
            str(output_file),
            "--splits",
            normalize_text(dataset_cfg.get("splits") or "test"),
        ]
        if dataset_cfg.get("include_article_text"):
            args.append("--include-article-text")
        call_python("prepare_medcase_reasoning_eval.py", args, dry_run=dry_run)
        return

    args = [
        "--input-file",
        normalize_text(dataset_cfg["input_file"]),
        "--output-file",
        str(output_file),
    ]
    dataset_name = normalize_text(dataset_cfg.get("dataset_name"))
    if dataset_name:
        args.extend(["--dataset-name", dataset_name])
    call_python("prepare_benchmark_reasoning_eval.py", args, dry_run=dry_run)


def generate_for_run(
    config: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    stage: str,
    dry_run: bool,
) -> None:
    input_file = materialize_stage_sample(config, dataset_cfg, stage, ensure_sample_pool(config, dataset_cfg))
    output_file = generation_master_path(config, model_cfg, dataset_cfg)
    ensure_parent(output_file)
    script_name = "generate_case_reasoning_eval.py" if dataset_cfg["type"] == "case" else "generate_mcq_reasoning_eval.py"
    if dataset_cfg["type"] == "case":
        max_tokens = int(
            model_cfg.get("generation_case_max_tokens", config["defaults"]["generation_case_max_tokens"])
        )
    else:
        max_tokens = int(
            model_cfg.get("generation_mcq_max_tokens", config["defaults"]["generation_mcq_max_tokens"])
        )
    args = [
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
        "--model",
        normalize_text(model_cfg["id"]),
        "--base-url",
        normalize_text(model_cfg.get("base_url")),
        "--api-key",
        normalize_text(model_cfg.get("api_key")),
        "--temperature",
        str(config["defaults"]["generation_temperature"]),
        "--max-tokens",
        str(max_tokens),
        "--max-concurrency",
        str(model_cfg.get("max_concurrency", config["defaults"]["generation_max_concurrency"])),
    ]
    call_python(script_name, args, dry_run=dry_run)


def judge_for_run(
    config: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    dry_run: bool,
) -> None:
    input_file = generation_master_path(config, model_cfg, dataset_cfg)
    if not input_file.exists() and not dry_run:
        raise FileNotFoundError(f"Missing generation master file: {input_file}")
    output_file = analysis_master_path(config, model_cfg, dataset_cfg)
    summary_file = analysis_master_summary_path(config, model_cfg, dataset_cfg)
    report_file = analysis_master_report_path(config, model_cfg, dataset_cfg)
    group_file = analysis_master_group_csv_path(config, model_cfg, dataset_cfg)
    ensure_parent(output_file)
    analysis_profile = "case_full" if dataset_cfg["type"] == "case" else "mcq_f4"
    args = [
        "--input-file",
        str(input_file),
        "--annotation-mode",
        "llm",
        "--analysis-profile",
        analysis_profile,
        "--model",
        normalize_text(config["judge_model"]["id"]),
        "--base-url",
        normalize_text(config["judge_model"].get("base_url")),
        "--api-key",
        normalize_text(config["judge_model"].get("api_key")),
        "--temperature",
        str(config["defaults"]["judge_temperature"]),
        "--max-tokens",
        str(config["defaults"]["judge_max_tokens"]),
        "--max-concurrency",
        str(config["judge_model"].get("max_concurrency", config["defaults"]["judge_max_concurrency"])),
        "--min-reasoning-chars",
        str(config["defaults"]["judge_min_reasoning_chars"]),
        "--group-by",
        "model_id,dataset_name,subset,split",
        "--output-file",
        str(output_file),
        "--summary-file",
        str(summary_file),
        "--report-file",
        str(report_file),
        "--group-summary-csv",
        str(group_file),
    ]
    call_python("analyze_qa_faithfulness_errors.py", args, dry_run=dry_run)


def run_stage_analysis(config_path: str, stage: str, models: str, datasets: str, dry_run: bool) -> None:
    args = [
        "--config",
        config_path,
        "--stage",
        stage,
    ]
    if models:
        args.extend(["--models", models])
    if datasets:
        args.extend(["--datasets", datasets])
    call_python("f1_f4_study/analyze_stage_results.py", args, dry_run=dry_run)


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
            prepare_dataset(config, dataset_cfg, force=args.force_prepare, dry_run=args.dry_run)

    if "sample" in phases:
        for dataset_cfg in selected_datasets:
            if args.dry_run:
                prepared_file = prepared_dataset_path(config, dataset_cfg)
                if not prepared_file.exists():
                    print(f"Dry-run sample planning only: prepared file would be {prepared_file}")
                for stage_name in stage_prefix(stage_names, args.stage):
                    print(f"Dry-run stage sample target: {prepared_dataset_path(config, dataset_cfg).parent / 'stages' / f'{stage_name}.jsonl'}")
                continue

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

    if "generate" in phases:
        for model_cfg in selected_models:
            for dataset_cfg in selected_datasets:
                ensure_parent(model_run_dir(config, model_cfg, dataset_cfg) / "placeholder.txt")
                generate_for_run(config, dataset_cfg, model_cfg, stage=args.stage, dry_run=args.dry_run)

    if "judge" in phases:
        for model_cfg in selected_models:
            for dataset_cfg in selected_datasets:
                judge_for_run(config, dataset_cfg, model_cfg, dry_run=args.dry_run)

    if "analyze" in phases:
        run_stage_analysis(
            config_path=str(resolved_config_path),
            stage=args.stage,
            models=args.models,
            datasets=args.datasets,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
