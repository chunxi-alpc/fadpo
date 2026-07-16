#!/usr/bin/env python3
from __future__ import annotations

import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

if __package__ is None or __package__ == "":
    CURRENT_DIR = Path(__file__).resolve().parent
    PARENT_DIR = CURRENT_DIR.parent
    sys.path.insert(0, str(PARENT_DIR))
    from common import ensure_parent, load_jsonl, normalize_text, read_json, write_json, write_jsonl
else:
    from ..common import ensure_parent, load_jsonl, normalize_text, read_json, write_json, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def parse_csv_fields(raw: str) -> List[str]:
    text = normalize_text(raw)
    if not text or text.lower() in {"none", "null"}:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def slugify_tag(value: str) -> str:
    text = normalize_text(value) or "item"
    text = re.sub(r"[^\w.-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("._") or "item"


def load_study_config(path: str) -> Dict[str, Any]:
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("Study config must be a JSON object.")

    config = dict(config)
    config.setdefault("study_name", "f1_f4_natural_error_study")
    config.setdefault("output_root", "outputs/f1_f4_study")
    config.setdefault(
        "stages",
        {
            "smoke": {"description": "Pipeline shakeout run."},
            "pilot": {"description": "Trend-checking run."},
            "final": {"description": "Paper-ready run."},
        },
    )
    config.setdefault("defaults", {})
    defaults = config["defaults"]
    defaults.setdefault("sampling_seed", 20260409)
    defaults.setdefault("generation_temperature", 0.0)
    defaults.setdefault("generation_case_max_tokens", 2000)
    defaults.setdefault("generation_mcq_max_tokens", 1200)
    defaults.setdefault("generation_max_concurrency", 8)
    defaults.setdefault("judge_temperature", 0.0)
    defaults.setdefault("judge_max_tokens", 900)
    defaults.setdefault("judge_max_concurrency", 8)
    defaults.setdefault("judge_min_reasoning_chars", 30)

    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Study config must contain a non-empty `models` list.")
    for model in models:
        if not isinstance(model, dict) or not normalize_text(model.get("id")):
            raise ValueError("Each model entry must be an object with an `id`.")
        model.setdefault("tag", slugify_tag(model["id"]))
        model.setdefault("base_url", "http://localhost:8000/v1")
        model.setdefault("api_key", "EMPTY")
        model.setdefault("max_concurrency", defaults["generation_max_concurrency"])

    judge_model = config.get("judge_model")
    if not isinstance(judge_model, dict) or not normalize_text(judge_model.get("id")):
        raise ValueError("Study config must contain a `judge_model` object with an `id`.")
    judge_model = dict(judge_model)
    judge_model.setdefault("tag", slugify_tag(judge_model["id"]))
    judge_model.setdefault("base_url", "http://localhost:8000/v1")
    judge_model.setdefault("api_key", "EMPTY")
    judge_model.setdefault("max_concurrency", defaults["judge_max_concurrency"])
    config["judge_model"] = judge_model

    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Study config must contain a non-empty `datasets` list.")

    stage_names = get_stage_names(config)
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("Each dataset entry must be an object.")
        dataset.setdefault("sampling", {})
        dataset.setdefault("key", slugify_tag(dataset.get("dataset_name") or dataset.get("input_file") or "dataset"))
        dataset.setdefault("tag", slugify_tag(dataset["key"]))
        dataset.setdefault("sampling_seed", defaults["sampling_seed"])
        dataset_type = normalize_text(dataset.get("type")).lower()
        if dataset_type not in {"case", "mcq"}:
            raise ValueError(f"Dataset `{dataset['key']}` must set `type` to `case` or `mcq`.")
        dataset["type"] = dataset_type
        if not normalize_text(dataset.get("input_file")):
            raise ValueError(f"Dataset `{dataset['key']}` is missing `input_file`.")

        has_stage_sizes = isinstance(dataset.get("stage_sizes"), Mapping)
        has_stage_group_sizes = isinstance(dataset.get("stage_group_sizes"), Mapping)
        if not has_stage_sizes and not has_stage_group_sizes:
            raise ValueError(
                f"Dataset `{dataset['key']}` must define `stage_sizes` or `stage_group_sizes`."
            )
        if has_stage_sizes and has_stage_group_sizes:
            raise ValueError(
                f"Dataset `{dataset['key']}` must use only one of `stage_sizes` or `stage_group_sizes`."
            )

        if has_stage_sizes:
            validate_stage_sizes(dataset["key"], dataset["stage_sizes"], stage_names)
        else:
            validate_stage_group_sizes(dataset["key"], dataset["stage_group_sizes"], stage_names)

    return config


def get_stage_names(config: Mapping[str, Any]) -> List[str]:
    stages = config.get("stages", {})
    if isinstance(stages, Mapping):
        return [normalize_text(key) for key in stages.keys() if normalize_text(key)]
    raise ValueError("`stages` must be a JSON object keyed by stage name.")


def validate_stage_sizes(dataset_key: str, stage_sizes: Mapping[str, Any], stage_names: Sequence[str]) -> None:
    last = -1
    for stage in stage_names:
        if stage not in stage_sizes:
            raise ValueError(f"Dataset `{dataset_key}` is missing stage_sizes[{stage!r}].")
        value = int(stage_sizes[stage])
        if value < last:
            raise ValueError(
                f"Dataset `{dataset_key}` has non-monotonic stage sizes: {stage}={value} after {last}."
            )
        last = value


def validate_stage_group_sizes(
    dataset_key: str,
    stage_group_sizes: Mapping[str, Any],
    stage_names: Sequence[str],
) -> None:
    previous_by_group: Dict[str, int] = {}
    for stage in stage_names:
        if stage not in stage_group_sizes:
            raise ValueError(f"Dataset `{dataset_key}` is missing stage_group_sizes[{stage!r}].")
        group_sizes = stage_group_sizes[stage]
        if not isinstance(group_sizes, Mapping) or not group_sizes:
            raise ValueError(
                f"Dataset `{dataset_key}` stage_group_sizes[{stage!r}] must be a non-empty object."
            )
        for group_name, raw_value in group_sizes.items():
            value = int(raw_value)
            previous = previous_by_group.get(group_name, -1)
            if value < previous:
                raise ValueError(
                    f"Dataset `{dataset_key}` group `{group_name}` shrank at stage `{stage}`."
                )
            previous_by_group[group_name] = value


def study_root(config: Mapping[str, Any]) -> Path:
    return (REPO_ROOT / normalize_text(config["output_root"]) / normalize_text(config["study_name"])).resolve()


def dataset_work_dir(config: Mapping[str, Any], dataset_cfg: Mapping[str, Any]) -> Path:
    return study_root(config) / "datasets" / normalize_text(dataset_cfg["tag"])


def prepared_dataset_path(config: Mapping[str, Any], dataset_cfg: Mapping[str, Any]) -> Path:
    return dataset_work_dir(config, dataset_cfg) / "prepared.full.jsonl"


def sample_pool_path(config: Mapping[str, Any], dataset_cfg: Mapping[str, Any]) -> Path:
    return dataset_work_dir(config, dataset_cfg) / "sample_pool.json"


def stage_sample_path(config: Mapping[str, Any], dataset_cfg: Mapping[str, Any], stage: str) -> Path:
    return dataset_work_dir(config, dataset_cfg) / "stages" / f"{stage}.jsonl"


def model_run_dir(config: Mapping[str, Any], model_cfg: Mapping[str, Any], dataset_cfg: Mapping[str, Any]) -> Path:
    return study_root(config) / "runs" / normalize_text(model_cfg["tag"]) / normalize_text(dataset_cfg["tag"])


def generation_master_path(
    config: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
) -> Path:
    return model_run_dir(config, model_cfg, dataset_cfg) / "generations.master.jsonl"


def analysis_master_path(
    config: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
) -> Path:
    return model_run_dir(config, model_cfg, dataset_cfg) / "faithfulness.master.jsonl"


def analysis_master_summary_path(
    config: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
) -> Path:
    return model_run_dir(config, model_cfg, dataset_cfg) / "faithfulness.master.summary.json"


def analysis_master_report_path(
    config: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
) -> Path:
    return model_run_dir(config, model_cfg, dataset_cfg) / "faithfulness.master.report.md"


def analysis_master_group_csv_path(
    config: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
) -> Path:
    return model_run_dir(config, model_cfg, dataset_cfg) / "faithfulness.master.groups.csv"


def stage_report_dir(config: Mapping[str, Any], stage: str) -> Path:
    return study_root(config) / "stage_reports" / normalize_text(stage)


def manual_job_root(config: Mapping[str, Any], stage: str) -> Path:
    return study_root(config) / "manual_jobs" / normalize_text(stage)


def manual_generation_jobs_path(
    config: Mapping[str, Any],
    stage: str,
    model_cfg: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
) -> Path:
    return (
        manual_job_root(config, stage)
        / "generation"
        / normalize_text(model_cfg["tag"])
        / f"{normalize_text(dataset_cfg['tag'])}.jobs.jsonl"
    )


def manual_generation_results_path(
    config: Mapping[str, Any],
    stage: str,
    model_cfg: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
) -> Path:
    return (
        manual_job_root(config, stage)
        / "generation"
        / normalize_text(model_cfg["tag"])
        / f"{normalize_text(dataset_cfg['tag'])}.results.jsonl"
    )


def manual_judge_jobs_path(
    config: Mapping[str, Any],
    stage: str,
    model_cfg: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
) -> Path:
    return (
        manual_job_root(config, stage)
        / "judge"
        / normalize_text(model_cfg["tag"])
        / f"{normalize_text(dataset_cfg['tag'])}.jobs.jsonl"
    )


def manual_judge_results_path(
    config: Mapping[str, Any],
    stage: str,
    model_cfg: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
) -> Path:
    return (
        manual_job_root(config, stage)
        / "judge"
        / normalize_text(model_cfg["tag"])
        / f"{normalize_text(dataset_cfg['tag'])}.results.jsonl"
    )


def save_resolved_config(config: Mapping[str, Any], config_path: str) -> Path:
    output_path = study_root(config) / "resolved_config.json"
    payload = dict(config)
    payload["_source_config"] = str(Path(config_path).resolve())
    write_json(output_path, payload)
    return output_path


def select_entries(entries: Sequence[Mapping[str, Any]], raw_names: str, key_field: str) -> List[Dict[str, Any]]:
    if not raw_names:
        return [dict(item) for item in entries]
    wanted = set(parse_csv_fields(raw_names))
    selected = [dict(item) for item in entries if normalize_text(item.get(key_field)) in wanted]
    if not selected:
        raise ValueError(f"No entries matched {sorted(wanted)} by `{key_field}`.")
    return selected


def stage_prefix(stage_names: Sequence[str], target_stage: str) -> List[str]:
    if target_stage not in stage_names:
        raise ValueError(f"Unknown stage `{target_stage}`. Expected one of {stage_names}.")
    return list(stage_names[: stage_names.index(target_stage) + 1])


def run_command(cmd: Sequence[str], dry_run: bool = False) -> None:
    print("$", " ".join(str(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=REPO_ROOT, check=True)


def call_python(script_relative: str, args: Sequence[str], dry_run: bool = False) -> None:
    script_path = PACKAGE_ROOT / script_relative
    cmd = [sys.executable, str(script_path), *[str(item) for item in args]]
    run_command(cmd, dry_run=dry_run)


def example_id(row: Mapping[str, Any]) -> str:
    return normalize_text(row.get("id") or row.get("source_id") or row.get("example_id"))


def ordered_rows_by_id(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    row_map: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        row_id = example_id(row)
        if not row_id:
            raise ValueError("Every prepared row must have `id` or `source_id`.")
        if row_id in row_map:
            raise ValueError(f"Duplicate example id detected: {row_id}")
        row_map[row_id] = dict(row)
    return row_map


def build_sample_pool_manifest(
    prepared_file: str | Path,
    dataset_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    rows = list(load_jsonl(prepared_file))
    row_map = ordered_rows_by_id(rows)
    sampling_cfg = dataset_cfg.get("sampling", {}) or {}
    seed = int(dataset_cfg.get("sampling_seed", 0))
    group_field = normalize_text(sampling_cfg.get("group_field"))
    manifest: Dict[str, Any] = {
        "dataset_key": normalize_text(dataset_cfg["key"]),
        "prepared_file": str(Path(prepared_file).resolve()),
        "seed": seed,
        "group_field": group_field,
        "count": len(row_map),
    }

    if group_field:
        grouped_ids: Dict[str, List[str]] = defaultdict(list)
        for row_id, row in row_map.items():
            group_name = normalize_text(row.get(group_field)) or "NA"
            grouped_ids[group_name].append(row_id)
        ordered_ids_by_group: Dict[str, List[str]] = {}
        for group_name in sorted(grouped_ids):
            ids = sorted(grouped_ids[group_name])
            random.Random(f"{seed}:{group_name}").shuffle(ids)
            ordered_ids_by_group[group_name] = ids
        manifest["ordered_ids_by_group"] = ordered_ids_by_group
        manifest["counts_by_group"] = {key: len(value) for key, value in ordered_ids_by_group.items()}
    else:
        ordered_ids = sorted(row_map)
        random.Random(seed).shuffle(ordered_ids)
        manifest["ordered_ids"] = ordered_ids
    return manifest


def ensure_sample_pool(
    config: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    prepared_file = prepared_dataset_path(config, dataset_cfg)
    if not prepared_file.exists():
        raise FileNotFoundError(f"Prepared dataset file does not exist: {prepared_file}")
    pool_file = sample_pool_path(config, dataset_cfg)
    if pool_file.exists() and not force:
        return read_json(pool_file)
    manifest = build_sample_pool_manifest(prepared_file, dataset_cfg)
    write_json(pool_file, manifest)
    return manifest


def selected_ids_for_stage(
    dataset_cfg: Mapping[str, Any],
    sample_pool: Mapping[str, Any],
    stage: str,
) -> List[str]:
    if "stage_sizes" in dataset_cfg:
        count = int(dataset_cfg["stage_sizes"][stage])
        ordered_ids = list(sample_pool.get("ordered_ids", []))
        if count > len(ordered_ids):
            raise ValueError(
                f"Dataset `{dataset_cfg['key']}` requested {count} rows at stage `{stage}` "
                f"but only {len(ordered_ids)} are available."
            )
        return ordered_ids[:count]

    ordered_ids_by_group = sample_pool.get("ordered_ids_by_group", {})
    requested = dataset_cfg["stage_group_sizes"][stage]
    selected: List[str] = []
    for group_name in sorted(requested):
        count = int(requested[group_name])
        available_ids = list(ordered_ids_by_group.get(group_name, []))
        if count > len(available_ids):
            raise ValueError(
                f"Dataset `{dataset_cfg['key']}` group `{group_name}` requested {count} rows "
                f"at stage `{stage}` but only {len(available_ids)} are available."
            )
        selected.extend(available_ids[:count])
    return selected


def materialize_stage_sample(
    config: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
    stage: str,
    sample_pool: Mapping[str, Any],
    force: bool = False,
) -> Path:
    stage_file = stage_sample_path(config, dataset_cfg, stage)
    prepared_file = prepared_dataset_path(config, dataset_cfg)
    row_map = ordered_rows_by_id(load_jsonl(prepared_file))
    selected_ids = selected_ids_for_stage(dataset_cfg, sample_pool, stage)
    if stage_file.exists() and not force:
        existing_ids = [example_id(row) for row in load_jsonl(stage_file)]
        if existing_ids == selected_ids:
            return stage_file
    selected_rows = [row_map[row_id] for row_id in selected_ids]
    ensure_parent(stage_file)
    write_jsonl(stage_file, selected_rows)
    return stage_file


def sample_ids_from_file(path: str | Path) -> List[str]:
    return [example_id(row) for row in load_jsonl(path)]
