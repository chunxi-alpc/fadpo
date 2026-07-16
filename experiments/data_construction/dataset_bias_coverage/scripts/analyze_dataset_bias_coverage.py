#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional


FAILURE_TYPES = ("F1", "F2", "F3", "F4")
BACKGROUND_MASK_VALUE = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build retained-pool, source-coverage, taxonomy-retention, and audit-sample "
            "artifacts for the dataset-bias/coverage work package."
        )
    )
    parser.add_argument("--train-file", default="data/train_fadpo.jsonl")
    parser.add_argument("--eval-file", default="data/eval_fadpo.jsonl")
    parser.add_argument("--negatives-file", default="data/medcase_unfaithful_negatives.jsonl")
    parser.add_argument("--aru-file", default="data/medcase_unfaithful_negatives.sanitized.arus.jsonl")
    parser.add_argument("--source-metadata-file", default="")
    parser.add_argument("--model-sampled-negatives-file", default="")
    parser.add_argument(
        "--output-dir",
        default="experiments/data_construction/dataset_bias_coverage/outputs",
    )
    parser.add_argument("--source-positives", type=int, default=13092)
    parser.add_argument("--raw-attempts-overall", type=int, default=52368)
    parser.add_argument("--qc-candidates-overall", type=int, default=47393)
    parser.add_argument("--audit-sample-per-type", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260527)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_cell(row.get(field)) for field in fieldnames})


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def format_cell(value: Any) -> Any:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def pct(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator * 100.0


def safe_mean(values: Iterable[float]) -> Optional[float]:
    items = list(values)
    if not items:
        return None
    return mean(items)


def word_count(text: Any) -> int:
    return len(str(text or "").split())


class RetainedStats:
    def __init__(self) -> None:
        self.rows = 0
        self.sources: set[str] = set()
        self.chosen_words: List[int] = []
        self.rejected_words: List[int] = []
        self.mask_tokens: List[int] = []
        self.high_weight_tokens: List[int] = []
        self.margin: List[float] = []
        self.train_rows = 0
        self.eval_rows = 0
        self.total_mask_tokens = 0
        self.total_high_weight_tokens = 0

    def add(self, row: Mapping[str, Any], split: str) -> None:
        self.rows += 1
        source_id = str(row.get("source_id") or "")
        if source_id:
            self.sources.add(source_id)
        self.chosen_words.append(word_count(row.get("chosen")))
        self.rejected_words.append(word_count(row.get("rejected")))
        mask = row.get("topo_mask") or []
        mask_len = len(mask)
        high_len = sum(1 for value in mask if float(value) > BACKGROUND_MASK_VALUE)
        self.mask_tokens.append(mask_len)
        self.high_weight_tokens.append(high_len)
        self.total_mask_tokens += mask_len
        self.total_high_weight_tokens += high_len
        if row.get("margin") is not None:
            self.margin.append(float(row.get("margin")))
        if split == "train":
            self.train_rows += 1
        elif split == "eval":
            self.eval_rows += 1

    def as_row(self, failure_type: str, total_rows: int) -> Dict[str, Any]:
        return {
            "type": failure_type,
            "retained_pairs": self.rows,
            "retained_pair_pct": pct(self.rows, total_rows),
            "unique_sources": len(self.sources),
            "mean_chosen_words": safe_mean(self.chosen_words),
            "mean_rejected_words": safe_mean(self.rejected_words),
            "mean_mask_tokens": safe_mean(self.mask_tokens),
            "mean_high_weight_tokens": safe_mean(self.high_weight_tokens),
            "high_weight_token_share_pct": pct(
                self.total_high_weight_tokens, self.total_mask_tokens
            ),
            "mean_margin": safe_mean(self.margin),
            "train_pairs": self.train_rows,
            "eval_pairs": self.eval_rows,
        }


def collect_retained_stats(train_file: Path, eval_file: Path) -> tuple[Dict[str, RetainedStats], Dict[str, Any]]:
    by_type = {failure_type: RetainedStats() for failure_type in FAILURE_TYPES}
    overall = RetainedStats()
    source_type_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for split, path in (("train", train_file), ("eval", eval_file)):
        for row in read_jsonl(path):
            failure_type = str(row.get("error_symbol") or "")
            if failure_type not in by_type:
                continue
            by_type[failure_type].add(row, split)
            overall.add(row, split)
            source_id = str(row.get("source_id") or "")
            if source_id:
                source_type_counts[source_id][failure_type] += 1

    by_type["Overall"] = overall
    retained_context = {
        "total_retained_pairs": overall.rows,
        "total_unique_sources": len(overall.sources),
        "source_type_counts": source_type_counts,
    }
    return by_type, retained_context


def collect_answer_policy(negatives_file: Path) -> Dict[str, Counter[Any]]:
    counters: Dict[str, Counter[Any]] = defaultdict(Counter)
    for row in read_jsonl(negatives_file):
        failure_type = str(row.get("error_symbol") or "")
        if failure_type not in FAILURE_TYPES:
            continue
        counters[failure_type]["retained_pairs"] += 1
        answer_preserved = bool(row.get("answer_preserved"))
        expected_preserved = bool(row.get("expected_answer_preserved"))
        counters[failure_type]["answer_preserved"] += int(answer_preserved)
        counters[failure_type]["answer_changed"] += int(not answer_preserved)
        counters[failure_type]["expected_answer_preserved"] += int(expected_preserved)
        counters[failure_type]["expected_answer_changed"] += int(not expected_preserved)
        counters[failure_type]["policy_mismatch"] += int(answer_preserved != expected_preserved)
    return counters


def collect_aru_stats(aru_file: Path) -> Dict[str, Dict[str, Any]]:
    by_type: Dict[str, Dict[str, Any]] = {
        failure_type: {
            "retained_pairs": 0,
            "aru_counts": [],
            "role_counts": Counter(),
        }
        for failure_type in FAILURE_TYPES
    }
    for row in read_jsonl(aru_file):
        failure_type = str(row.get("error_symbol") or "")
        if failure_type not in by_type:
            continue
        arus = row.get("arus") or []
        by_type[failure_type]["retained_pairs"] += 1
        by_type[failure_type]["aru_counts"].append(len(arus))
        for aru in arus:
            role = str(aru.get("type") or "").upper()
            if role:
                by_type[failure_type]["role_counts"][role] += 1

    overall_role_counts = Counter()
    overall_aru_counts: List[int] = []
    overall_pairs = 0
    for stats in by_type.values():
        overall_pairs += stats["retained_pairs"]
        overall_aru_counts.extend(stats["aru_counts"])
        overall_role_counts.update(stats["role_counts"])
    by_type["Overall"] = {
        "retained_pairs": overall_pairs,
        "aru_counts": overall_aru_counts,
        "role_counts": overall_role_counts,
    }
    return by_type


def load_source_metadata(path_text: str) -> Dict[str, Dict[str, Any]]:
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Source metadata file not found: {path}")

    rows: Dict[str, Dict[str, Any]] = {}
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                source_id = first_non_empty(row, ("source_id", "id", "example_id"))
                if source_id:
                    rows[source_id] = dict(row)
    else:
        for row in read_jsonl(path):
            source_id = first_non_empty(row, ("source_id", "id", "example_id"))
            if source_id:
                rows[source_id] = dict(row)
    return rows


def first_non_empty(row: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def build_source_distribution(
    source_type_counts: Mapping[str, Counter[str]],
    source_metadata: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source_id in sorted(source_type_counts, key=lambda item: int(item) if item.isdigit() else item):
        type_counts = source_type_counts[source_id]
        metadata = source_metadata.get(source_id, {})
        rows.append(
            {
                "source_id": source_id,
                "retained_pairs": sum(type_counts.values()),
                "failure_types": ";".join(
                    failure_type for failure_type in FAILURE_TYPES if type_counts.get(failure_type, 0)
                ),
                "F1_pairs": type_counts.get("F1", 0),
                "F2_pairs": type_counts.get("F2", 0),
                "F3_pairs": type_counts.get("F3", 0),
                "F4_pairs": type_counts.get("F4", 0),
                "metadata_available": bool(metadata),
                "specialty": first_non_empty(metadata, ("specialty", "medical_specialty", "department")),
                "disease_group": first_non_empty(metadata, ("disease_group", "disease", "diagnosis")),
                "question_format": first_non_empty(metadata, ("question_format", "format", "task_type")),
                "answer_option": first_non_empty(metadata, ("answer_option", "answer_idx", "label")),
            }
        )
    return rows


def build_source_reuse_summary(source_rows: List[Mapping[str, Any]], total_pairs: int) -> List[Dict[str, Any]]:
    reuse_counter = Counter(int(row["retained_pairs"]) for row in source_rows)
    total_sources = len(source_rows)
    rows = []
    for retained_pairs_per_source in sorted(reuse_counter):
        source_count = reuse_counter[retained_pairs_per_source]
        retained_pair_count = retained_pairs_per_source * source_count
        rows.append(
            {
                "retained_pairs_per_source": retained_pairs_per_source,
                "source_count": source_count,
                "source_pct": pct(source_count, total_sources),
                "retained_pair_count": retained_pair_count,
                "retained_pair_pct": pct(retained_pair_count, total_pairs),
            }
        )
    return rows


def build_taxonomy_rows(
    retained_stats: Mapping[str, RetainedStats],
    aru_stats: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for failure_type in FAILURE_TYPES:
        stats = retained_stats[failure_type]
        mean_arus = safe_mean(aru_stats.get(failure_type, {}).get("aru_counts", []))
        rows.append(
            {
                "failure_type": failure_type,
                "raw_attempts": args.source_positives,
                "after_qc": None,
                "retained": stats.rows,
                "qc_rate_pct": None,
                "retention_among_qc_pct": None,
                "retention_among_raw_pct": pct(stats.rows, args.source_positives),
                "mean_arus": mean_arus,
                "mean_high_risk_arus": None,
                "notes": "Per-type QC/discarded logs unavailable; raw attempts are design-implied.",
            }
        )
    overall_mean_arus = safe_mean(aru_stats.get("Overall", {}).get("aru_counts", []))
    rows.append(
        {
            "failure_type": "Overall",
            "raw_attempts": args.raw_attempts_overall,
            "after_qc": args.qc_candidates_overall,
            "retained": retained_stats["Overall"].rows,
            "qc_rate_pct": pct(args.qc_candidates_overall, args.raw_attempts_overall),
            "retention_among_qc_pct": pct(retained_stats["Overall"].rows, args.qc_candidates_overall),
            "retention_among_raw_pct": pct(retained_stats["Overall"].rows, args.raw_attempts_overall),
            "mean_arus": overall_mean_arus,
            "mean_high_risk_arus": None,
            "notes": "Overall raw/QC counts use construction-flow counts recorded in the work-package plan.",
        }
    )
    return rows


def build_answer_policy_rows(answer_policy: Mapping[str, Counter[Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    overall = Counter()
    for failure_type in FAILURE_TYPES:
        counter = answer_policy.get(failure_type, Counter())
        overall.update(counter)
        retained = int(counter.get("retained_pairs", 0))
        rows.append(
            {
                "type": failure_type,
                "retained_pairs": retained,
                "answer_preserved": int(counter.get("answer_preserved", 0)),
                "answer_changed": int(counter.get("answer_changed", 0)),
                "answer_preserved_pct": pct(counter.get("answer_preserved", 0), retained),
                "expected_answer_preserved": int(counter.get("expected_answer_preserved", 0)),
                "expected_answer_changed": int(counter.get("expected_answer_changed", 0)),
                "policy_mismatch": int(counter.get("policy_mismatch", 0)),
            }
        )
    retained = int(overall.get("retained_pairs", 0))
    rows.append(
        {
            "type": "Overall",
            "retained_pairs": retained,
            "answer_preserved": int(overall.get("answer_preserved", 0)),
            "answer_changed": int(overall.get("answer_changed", 0)),
            "answer_preserved_pct": pct(overall.get("answer_preserved", 0), retained),
            "expected_answer_preserved": int(overall.get("expected_answer_preserved", 0)),
            "expected_answer_changed": int(overall.get("expected_answer_changed", 0)),
            "policy_mismatch": int(overall.get("policy_mismatch", 0)),
        }
    )
    return rows


def build_aru_role_rows(aru_stats: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for failure_type in (*FAILURE_TYPES, "Overall"):
        stats = aru_stats.get(failure_type, {})
        retained = int(stats.get("retained_pairs", 0))
        aru_counts = stats.get("aru_counts", [])
        role_counts = stats.get("role_counts", Counter())
        total_arus = sum(role_counts.values())
        row: Dict[str, Any] = {
            "type": failure_type,
            "retained_pairs": retained,
            "total_arus": total_arus,
            "mean_arus": safe_mean(aru_counts),
        }
        for role in ("O", "W", "D", "C"):
            role_total = int(role_counts.get(role, 0))
            row[f"{role}_total"] = role_total
            row[f"{role}_mean_per_pair"] = role_total / retained if retained else None
            row[f"{role}_share_pct"] = pct(role_total, total_arus)
        rows.append(row)
    return rows


def reservoir_add(
    reservoirs: Dict[str, List[Dict[str, Any]]],
    seen: Counter[str],
    key: str,
    row: Dict[str, Any],
    sample_size: int,
    rng: random.Random,
) -> None:
    seen[key] += 1
    bucket = reservoirs[key]
    if len(bucket) < sample_size:
        bucket.append(row)
        return
    replacement_index = rng.randrange(seen[key])
    if replacement_index < sample_size:
        bucket[replacement_index] = row


def make_constructed_audit_sample(
    negatives_file: Path,
    sample_size_per_type: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    reservoirs: Dict[str, List[Dict[str, Any]]] = {failure_type: [] for failure_type in FAILURE_TYPES}
    seen: Counter[str] = Counter()
    for row in read_jsonl(negatives_file):
        failure_type = str(row.get("error_symbol") or "")
        if failure_type not in reservoirs:
            continue
        audit_row = {
            "sample_source": "constructed_controlled_rewrite",
            "record_id": row.get("id"),
            "source_id": row.get("source_id"),
            "target_error_symbol_for_coordinator": failure_type,
            "case_context": row.get("patient_context") or row.get("prompt"),
            "preferred_response": row.get("chosen") or row.get("correct_reasoning"),
            "candidate_negative_response": row.get("rejected") or row.get("negative_reasoning"),
            "preferred_final_answer": row.get("final_answer") or row.get("correct_answer_text"),
            "candidate_final_answer": row.get("negative_final_answer") or row.get("predicted_answer_text"),
            "answer_preserved": row.get("answer_preserved"),
            "expected_answer_preserved": row.get("expected_answer_preserved"),
            "modified_spans_for_coordinator": row.get("modified_spans"),
            "edit_summary_for_coordinator": row.get("edit_summary"),
            "response_naturalness_1_5": None,
            "clinical_plausibility_1_5": None,
            "dominant_error_category": None,
            "maps_to_F1_F4": None,
            "local_rewrite_artifact": None,
            "suitable_as_dpo_rejected": None,
            "annotator_notes": None,
        }
        reservoir_add(reservoirs, seen, failure_type, audit_row, sample_size_per_type, rng)

    sample_rows: List[Dict[str, Any]] = []
    audit_index = 1
    for failure_type in FAILURE_TYPES:
        bucket = list(reservoirs[failure_type])
        rng.shuffle(bucket)
        for row in bucket:
            row["audit_id"] = f"CONSTRUCTED_{audit_index:04d}"
            sample_rows.append(row)
            audit_index += 1
    return sample_rows


def make_model_sampled_audit_sample(path_text: str, seed: int, total_sample_size: int) -> List[Dict[str, Any]]:
    if not path_text:
        return []
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Model-sampled negatives file not found: {path}")
    rng = random.Random(seed + 17)
    bucket: List[Dict[str, Any]] = []
    seen = 0
    for row in read_jsonl(path):
        seen += 1
        audit_row = {
            "sample_source": "model_sampled_negative",
            "record_id": row.get("id") or row.get("record_id"),
            "source_id": row.get("source_id"),
            "target_error_symbol_for_coordinator": row.get("error_symbol"),
            "case_context": row.get("patient_context") or row.get("prompt") or row.get("case_context"),
            "preferred_response": row.get("chosen") or row.get("preferred_response"),
            "candidate_negative_response": (
                row.get("rejected")
                or row.get("negative_reasoning")
                or row.get("candidate_negative_response")
                or row.get("model_response")
            ),
            "preferred_final_answer": row.get("final_answer") or row.get("correct_answer_text"),
            "candidate_final_answer": row.get("negative_final_answer") or row.get("predicted_answer_text"),
            "answer_preserved": row.get("answer_preserved"),
            "expected_answer_preserved": row.get("expected_answer_preserved"),
            "modified_spans_for_coordinator": None,
            "edit_summary_for_coordinator": None,
            "response_naturalness_1_5": None,
            "clinical_plausibility_1_5": None,
            "dominant_error_category": None,
            "maps_to_F1_F4": None,
            "local_rewrite_artifact": None,
            "suitable_as_dpo_rejected": None,
            "annotator_notes": None,
        }
        if len(bucket) < total_sample_size:
            bucket.append(audit_row)
            continue
        replacement_index = rng.randrange(seen)
        if replacement_index < total_sample_size:
            bucket[replacement_index] = audit_row
    rng.shuffle(bucket)
    for offset, row in enumerate(bucket, start=1):
        row["audit_id"] = f"MODEL_SAMPLED_{offset:04d}"
    return bucket


def markdown_table(headers: List[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def fmt_number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(
    path: Path,
    retained_rows: List[Dict[str, Any]],
    taxonomy_rows: List[Dict[str, Any]],
    answer_rows: List[Dict[str, Any]],
    source_reuse_rows: List[Dict[str, Any]],
    metadata_available: bool,
    constructed_sample_size: int,
    model_sample_size: int,
) -> None:
    overall_retained = next(row for row in retained_rows if row["type"] == "Overall")
    overall_taxonomy = next(row for row in taxonomy_rows if row["failure_type"] == "Overall")
    f4_answer = next(row for row in answer_rows if row["type"] == "F4")

    retained_table = markdown_table(
        [
            "Type",
            "Retained",
            "Unique sources",
            "Mean chosen words",
            "Mean rejected words",
            "Mean mask tokens",
            "High-weight share",
            "Mean margin",
        ],
        [
            [
                row["type"],
                fmt_number(row["retained_pairs"], 0),
                fmt_number(row["unique_sources"], 0),
                fmt_number(row["mean_chosen_words"], 1),
                fmt_number(row["mean_rejected_words"], 1),
                fmt_number(row["mean_mask_tokens"], 1),
                f"{fmt_number(row['high_weight_token_share_pct'], 1)}%",
                fmt_number(row["mean_margin"], 2),
            ]
            for row in retained_rows
        ],
    )
    source_table = markdown_table(
        ["Retained pairs/source", "Source count", "Source %", "Pair count", "Pair %"],
        [
            [
                row["retained_pairs_per_source"],
                fmt_number(row["source_count"], 0),
                f"{fmt_number(row['source_pct'], 1)}%",
                fmt_number(row["retained_pair_count"], 0),
                f"{fmt_number(row['retained_pair_pct'], 1)}%",
            ]
            for row in source_reuse_rows
        ],
    )
    answer_table = markdown_table(
        ["Type", "Retained", "Answer preserved", "Answer changed", "Preserved %"],
        [
            [
                row["type"],
                fmt_number(row["retained_pairs"], 0),
                fmt_number(row["answer_preserved"], 0),
                fmt_number(row["answer_changed"], 0),
                f"{fmt_number(row['answer_preserved_pct'], 1)}%",
            ]
            for row in answer_rows
        ],
    )

    lines = [
        "# Dataset Bias and Coverage Summary",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Measured Retained-Pool Statistics",
        "",
        (
            f"The retained Fa-DPO pool contains {fmt_number(overall_retained['retained_pairs'], 0)} "
            f"preference pairs from {fmt_number(overall_retained['unique_sources'], 0)} unique "
            "MedCaseReasoning source positives. The released token mask marks "
            f"{fmt_number(overall_retained['high_weight_token_share_pct'], 1)}% of rejected-response "
            "tokens as above the low background value 0.1."
        ),
        "",
        retained_table,
        "",
        "## Construction Flow and Taxonomy Boundary",
        "",
        (
            f"Aggregate construction flow: {fmt_number(overall_taxonomy['raw_attempts'], 0)} raw attempts "
            f"-> {fmt_number(overall_taxonomy['after_qc'], 0)} quality-controlled candidates "
            f"-> {fmt_number(overall_taxonomy['retained'], 0)} retained preference pairs. "
            f"This is {fmt_number(overall_taxonomy['retention_among_qc_pct'], 1)}% retention among QC "
            f"and {fmt_number(overall_taxonomy['retention_among_raw_pct'], 1)}% retention among raw attempts."
        ),
        "",
        (
            "Per-type raw attempts are design-implied at one attempted rewrite per source positive "
            "and failure type. Per-type QC counts, discarded-risk distributions, and high-risk ARU "
            "counts are not available in the compact retained release and are left as NA in "
            "`taxonomy_retention_table.csv`."
        ),
        "",
        "## Source Reuse",
        "",
        source_table,
        "",
        "Source specialty, disease-group, answer-option, and question-format distributions were "
        + (
            "merged from the provided source metadata file."
            if metadata_available
            else "not measured because no source metadata export was available in this workspace."
        ),
        "",
        "## Answer-Preservation Policy",
        "",
        answer_table,
        "",
        (
            f"F1--F3 preserve the extracted final answer in the retained file. F4 contains "
            f"{fmt_number(f4_answer['answer_changed'], 0)} answer-changing retained pairs out of "
            f"{fmt_number(f4_answer['retained_pairs'], 0)} F4 pairs; the remaining F4 rows preserve "
            "the answer while weakening or breaking reasoning-answer support."
        ),
        "",
        "## Construction-Artifact Audit Status",
        "",
        (
            f"`construction_artifact_audit_sample.jsonl` contains {constructed_sample_size} constructed "
            "controlled-rewrite negatives for blind artifact/plausibility review."
        ),
        (
            f"A model-sampled comparator file was provided and {model_sample_size} comparator rows were sampled."
            if model_sample_size
            else "No model-sampled negative comparator was available, so the artifact-rate comparison remains planned."
        ),
        "",
        "## Reporting Boundary",
        "",
        "Supported wording: the dataset is controlled, interpretable, and intentionally risk-enriched for F1--F4 process failures.",
        "",
        "Avoid wording: the 34,204 pairs are an unbiased sample of natural clinical reasoning errors, F1--F4 are a complete taxonomy, or local rewrites are equivalent to naturally model-sampled errors.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retained_stats, retained_context = collect_retained_stats(Path(args.train_file), Path(args.eval_file))
    answer_policy = collect_answer_policy(Path(args.negatives_file))
    aru_stats = collect_aru_stats(Path(args.aru_file))
    source_metadata = load_source_metadata(args.source_metadata_file)

    total_retained = retained_context["total_retained_pairs"]
    retained_rows = [
        retained_stats[failure_type].as_row(failure_type, total_retained)
        for failure_type in (*FAILURE_TYPES, "Overall")
    ]
    source_rows = build_source_distribution(retained_context["source_type_counts"], source_metadata)
    source_reuse_rows = build_source_reuse_summary(source_rows, total_retained)
    taxonomy_rows = build_taxonomy_rows(retained_stats, aru_stats, args)
    answer_rows = build_answer_policy_rows(answer_policy)
    aru_role_rows = build_aru_role_rows(aru_stats)

    constructed_sample = make_constructed_audit_sample(
        Path(args.negatives_file),
        sample_size_per_type=args.audit_sample_per_type,
        seed=args.seed,
    )
    model_sample = make_model_sampled_audit_sample(
        args.model_sampled_negatives_file,
        seed=args.seed,
        total_sample_size=len(constructed_sample),
    )
    audit_sample = constructed_sample + model_sample

    write_csv(
        output_dir / "retained_pool_distribution.csv",
        [
            "type",
            "retained_pairs",
            "retained_pair_pct",
            "unique_sources",
            "mean_chosen_words",
            "mean_rejected_words",
            "mean_mask_tokens",
            "mean_high_weight_tokens",
            "high_weight_token_share_pct",
            "mean_margin",
            "train_pairs",
            "eval_pairs",
        ],
        retained_rows,
    )
    write_csv(
        output_dir / "source_distribution.csv",
        [
            "source_id",
            "retained_pairs",
            "failure_types",
            "F1_pairs",
            "F2_pairs",
            "F3_pairs",
            "F4_pairs",
            "metadata_available",
            "specialty",
            "disease_group",
            "question_format",
            "answer_option",
        ],
        source_rows,
    )
    write_csv(
        output_dir / "source_reuse_summary.csv",
        [
            "retained_pairs_per_source",
            "source_count",
            "source_pct",
            "retained_pair_count",
            "retained_pair_pct",
        ],
        source_reuse_rows,
    )
    write_csv(
        output_dir / "taxonomy_retention_table.csv",
        [
            "failure_type",
            "raw_attempts",
            "after_qc",
            "retained",
            "qc_rate_pct",
            "retention_among_qc_pct",
            "retention_among_raw_pct",
            "mean_arus",
            "mean_high_risk_arus",
            "notes",
        ],
        taxonomy_rows,
    )
    write_csv(
        output_dir / "answer_policy_summary.csv",
        [
            "type",
            "retained_pairs",
            "answer_preserved",
            "answer_changed",
            "answer_preserved_pct",
            "expected_answer_preserved",
            "expected_answer_changed",
            "policy_mismatch",
        ],
        answer_rows,
    )
    write_csv(
        output_dir / "aru_role_distribution.csv",
        [
            "type",
            "retained_pairs",
            "total_arus",
            "mean_arus",
            "O_total",
            "O_mean_per_pair",
            "O_share_pct",
            "W_total",
            "W_mean_per_pair",
            "W_share_pct",
            "D_total",
            "D_mean_per_pair",
            "D_share_pct",
            "C_total",
            "C_mean_per_pair",
            "C_share_pct",
        ],
        aru_role_rows,
    )
    write_jsonl(output_dir / "construction_artifact_audit_sample.jsonl", audit_sample)
    write_csv(
        output_dir / "construction_artifact_audit_results.csv",
        [
            "audit_id",
            "sample_source",
            "response_naturalness_1_5",
            "clinical_plausibility_1_5",
            "dominant_error_category",
            "maps_to_F1_F4",
            "local_rewrite_artifact",
            "suitable_as_dpo_rejected",
            "annotator_notes",
        ],
        [],
    )
    write_summary(
        output_dir / "dataset_bias_summary.md",
        retained_rows=retained_rows,
        taxonomy_rows=taxonomy_rows,
        answer_rows=answer_rows,
        source_reuse_rows=source_reuse_rows,
        metadata_available=bool(source_metadata),
        constructed_sample_size=len(constructed_sample),
        model_sample_size=len(model_sample),
    )
    write_json(
        output_dir / "manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "train_file": args.train_file,
                "eval_file": args.eval_file,
                "negatives_file": args.negatives_file,
                "aru_file": args.aru_file,
                "source_metadata_file": args.source_metadata_file or None,
                "model_sampled_negatives_file": args.model_sampled_negatives_file or None,
            },
            "parameters": {
                "source_positives": args.source_positives,
                "raw_attempts_overall": args.raw_attempts_overall,
                "qc_candidates_overall": args.qc_candidates_overall,
                "audit_sample_per_type": args.audit_sample_per_type,
                "seed": args.seed,
            },
            "outputs": [
                "retained_pool_distribution.csv",
                "source_distribution.csv",
                "source_reuse_summary.csv",
                "taxonomy_retention_table.csv",
                "answer_policy_summary.csv",
                "aru_role_distribution.csv",
                "construction_artifact_audit_sample.jsonl",
                "construction_artifact_audit_results.csv",
                "dataset_bias_summary.md",
            ],
        },
    )

    print(f"Wrote dataset-bias outputs to {output_dir}")
    print(f"Retained pairs: {total_retained}")
    print(f"Unique source positives: {retained_context['total_unique_sources']}")
    print(f"Constructed audit sample rows: {len(constructed_sample)}")
    if model_sample:
        print(f"Model-sampled comparator rows: {len(model_sample)}")


if __name__ == "__main__":
    main()
