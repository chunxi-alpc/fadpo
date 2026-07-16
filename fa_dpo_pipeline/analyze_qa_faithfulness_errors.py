#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import httpx

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (
        append_jsonl,
        ensure_parent,
        extract_json,
        first_non_empty,
        iso_utc_now,
        load_jsonl,
        normalize_answer,
        normalize_text,
        read_json,
        write_json,
        write_jsonl,
    )
else:
    from .common import (
        append_jsonl,
        ensure_parent,
        extract_json,
        first_non_empty,
        iso_utc_now,
        load_jsonl,
        normalize_answer,
        normalize_text,
        read_json,
        write_json,
        write_jsonl,
    )

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - dependency availability is environment-specific
    AsyncOpenAI = None


SYSTEM_PROMPT = """You are a strict clinical reasoning auditor.

Your task is to inspect a model-generated medical QA response and determine whether it exhibits one or more faithfulness errors along the evidence -> reasoning -> answer chain.

Be conservative:
- Do not mark F1 just because every detail is not mentioned.
- Do not mark F2 unless cited evidence is actually interpreted incorrectly.
- Do not mark F3 unless the reasoning overstates the strength of available evidence.
- Do not mark F4 unless the final answer is unsupported by, or inconsistent with, the preceding reasoning.
- A response may have the correct final answer and still be unfaithful.
- If there is no clear F1-F4 error, output NONE.
- If multiple error types are comparably strong, output MIXED.
- If the response lacks enough reasoning to judge, or is unusable/off-topic, output INVALID.

Return one JSON object only.
""".strip()


QUESTION_FIELD_CANDIDATES = (
    "patient_context",
    "case_prompt",
    "question",
    "prompt",
    "case",
    "context",
    "input",
    "input_str",
)

EVIDENCE_FIELD_CANDIDATES = (
    "retrieved_evidence",
    "text",
    "article_text",
    "evidence",
    "supporting_evidence",
    "documents",
    "docs",
)

REFERENCE_REASONING_STRICT_FIELDS = (
    "correct_reasoning",
    "diagnostic_reasoning",
    "reference_reasoning",
    "gold_reasoning",
    "ground_truth_reasoning",
    "answer_explanation",
    "explanation",
)

REFERENCE_ANSWER_STRICT_FIELDS = (
    "correct_answer_text",
    "final_diagnosis",
    "reference_answer",
    "gold_answer",
    "ground_truth_answer",
)

GENERIC_REFERENCE_REASONING_FIELDS = (
    "reasoning",
    "diagnostic_reasoning",
    "correct_reasoning",
    "reference_reasoning",
    "gold_reasoning",
)

GENERIC_REFERENCE_ANSWER_FIELDS = (
    "final_answer",
    "answer",
    "final_diagnosis",
    "correct_answer_text",
    "reference_answer",
    "gold_answer",
)

PREDICTION_TEXT_FIELDS = (
    "output",
    "prediction",
    "response",
    "generated_text",
    "model_output",
    "completion",
    "assistant_output",
    "raw_output",
)

PREDICTED_REASONING_FIELDS = (
    "predicted_reasoning",
    "model_reasoning",
    "generated_reasoning",
    "assistant_reasoning",
    "candidate_reasoning",
    "negative_reasoning",
)

PREDICTED_ANSWER_FIELDS = (
    "predicted_answer_text",
    "model_answer",
    "generated_answer",
    "assistant_answer",
    "candidate_answer",
    "negative_final_answer",
)

ID_FIELD_CANDIDATES = (
    "id",
    "example_id",
    "source_id",
    "pmcid",
    "",
    "qid",
    "uid",
)

GROUP_FIELD_CANDIDATES = (
    "source",
    "model_id",
    "benchmark",
    "subset",
    "split",
    "dataset_name",
)

STATUS_DONE = {"judged", "manual_judged", "reused_existing_label", "skipped_no_reasoning"}
ERROR_TYPES = ("F1", "F2", "F3", "F4")
NON_ERROR_TYPES = ("NONE", "MIXED", "INVALID")
ANALYSIS_PROFILES = ("auto", "case_full", "mcq_f4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze model-generated medical QA outputs for faithfulness errors "
            "(F1/F2/F3/F4), then produce per-example annotations and summary "
            "statistics."
        )
    )
    parser.add_argument("--input-file", required=True, help="Input JSON/JSONL file.")
    parser.add_argument(
        "--annotation-mode",
        choices=["llm", "existing"],
        default="llm",
        help="Use an LLM judge or reuse existing labels in the input file.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Annotated JSONL output path. Defaults to <input>.faithfulness_analysis.jsonl",
    )
    parser.add_argument(
        "--summary-file",
        default="",
        help="Summary JSON output path. Defaults to <input>.faithfulness_summary.json",
    )
    parser.add_argument(
        "--report-file",
        default="",
        help="Markdown report output path. Defaults to <input>.faithfulness_report.md",
    )
    parser.add_argument(
        "--group-summary-csv",
        default="",
        help="Flat CSV summary path. Defaults to <input>.faithfulness_group_summary.csv",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL for judge mode.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API key for judge mode.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", ""),
        help="Judge model name for llm mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=900,
        help="Max completion tokens for judge mode.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Concurrent judge requests in llm mode.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds for the OpenAI-compatible judge backend.",
    )
    parser.add_argument(
        "--request-max-retries",
        type=int,
        default=2,
        help="How many times to retry a failed judge request before marking it failed.",
    )
    parser.add_argument(
        "--request-retry-backoff",
        type=float,
        default=2.0,
        help="Base backoff in seconds between judge request retries.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Optional sample cap.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="End index (exclusive). -1 means all rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotated output instead of resuming.",
    )
    parser.add_argument(
        "--min-reasoning-chars",
        type=int,
        default=30,
        help="Rows below this reasoning length are marked as skipped_no_reasoning.",
    )
    parser.add_argument(
        "--group-by",
        default="auto",
        help=(
            "Comma-separated grouping fields for aggregated analysis. Use 'auto' to "
            "pick the first available field among source/model_id/benchmark/subset/"
            "dataset_name, or 'none' to disable group summaries."
        ),
    )
    parser.add_argument(
        "--analysis-profile",
        choices=ANALYSIS_PROFILES,
        default="auto",
        help=(
            "Judging profile. `case_full` evaluates F1-F4; `mcq_f4` focuses on "
            "reasoning-answer mismatch for MCQ-style benchmarks."
        ),
    )
    parser.add_argument("--id-field", default="", help="Override example id field.")
    parser.add_argument("--question-field", default="", help="Override question/context field.")
    parser.add_argument("--evidence-field", default="", help="Override evidence field.")
    parser.add_argument(
        "--reference-reasoning-field",
        default="",
        help="Override reference reasoning field.",
    )
    parser.add_argument(
        "--reference-answer-field",
        default="",
        help="Override reference answer field.",
    )
    parser.add_argument(
        "--prediction-field",
        default="",
        help="Override combined prediction text field.",
    )
    parser.add_argument(
        "--predicted-reasoning-field",
        default="",
        help="Override predicted reasoning field.",
    )
    parser.add_argument(
        "--predicted-answer-field",
        default="",
        help="Override predicted answer field.",
    )
    return parser.parse_args()


def default_output_path(input_file: str, suffix: str) -> str:
    path = Path(input_file)
    stem = path.with_suffix("")
    return str(stem) + suffix


def parse_csv_fields(raw: str) -> List[str]:
    text = normalize_text(raw)
    if not text or text.lower() in {"none", "null"}:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return bool(normalize_text(value))


def parse_bool(value: Any) -> bool | None:
    if value in ("", None):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = normalize_text(value).lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def safe_int(value: Any, default: int) -> int:
    if value in ("", None):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_true(value: Any) -> bool:
    return parse_bool(value) is True


def flatten_json_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [row if isinstance(row, dict) else {"value": row} for row in payload]
    if isinstance(payload, dict):
        for key in ("data", "records", "items", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                rows: List[Dict[str, Any]] = []
                for row in value:
                    rows.append(row if isinstance(row, dict) else {"value": row})
                return rows
        if payload and all(isinstance(v, list) for v in payload.values()):
            rows = []
            for source, items in payload.items():
                for row in items:
                    item = row if isinstance(row, dict) else {"value": row}
                    if "source" not in item:
                        item = dict(item)
                        item["source"] = source
                    rows.append(item)
            return rows
        return [payload]
    raise ValueError("Unsupported JSON structure: expected list or dict.")


def load_records(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(file_path)
    if suffix == ".json":
        return flatten_json_payload(read_json(file_path))
    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported input format: {file_path.suffix}")


def strip_markdown_artifacts(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"</s>\s*$", "", cleaned).strip()
    cleaned = re.sub(r"^\s*```(?:json|text|markdown)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


def format_options(options: Any) -> str:
    if isinstance(options, Mapping):
        lines = [f"{key}. {normalize_text(value)}" for key, value in options.items()]
        return "\n".join(line for line in lines if line.strip())
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        lines = [f"{chr(65 + idx)}. {normalize_text(value)}" for idx, value in enumerate(options)]
        return "\n".join(line for line in lines if line.strip())
    return normalize_text(options)


def get_options_map(options: Any) -> Dict[str, str]:
    if isinstance(options, Mapping):
        return {str(key): normalize_text(value) for key, value in options.items()}
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        return {chr(65 + idx): normalize_text(value) for idx, value in enumerate(options)}
    return {}


def extract_choice_label(text: str, options: Mapping[str, str]) -> str:
    if not text or not options:
        return ""
    valid_keys = "".join(str(key).upper() for key in options.keys())
    stripped = strip_markdown_artifacts(text)
    patterns = [
        rf"(?is)\\boxed\{{([{valid_keys}])\}}",
        rf"(?is)(?:final answer|answer|option|choice|diagnosis)\s*(?:is|:)\s*[{re.escape('([{<')}]?\s*([{valid_keys}])\b",
        rf"(?is)\bthe answer is\s*[{re.escape('([{<')}]?\s*([{valid_keys}])\b",
        rf"(?is)(?<![A-Za-z])([{valid_keys}])(?![A-Za-z])",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, stripped))
        if matches:
            return matches[-1].group(1).upper()

    lowered = stripped.lower()
    option_hits: List[Tuple[int, str]] = []
    for key, value in options.items():
        value_norm = normalize_text(value).lower()
        if value_norm and value_norm in lowered:
            option_hits.append((lowered.rfind(value_norm), str(key).upper()))
    if option_hits:
        option_hits.sort()
        return option_hits[-1][1]
    return ""


def extract_final_answer_text(text: str, options: Mapping[str, str] | None = None) -> str:
    cleaned = strip_markdown_artifacts(text)
    if not cleaned:
        return ""

    if options:
        label = extract_choice_label(cleaned, options)
        if label and label in options:
            return normalize_text(options[label]) or label

    patterns = [
        r"(?is)(?:^|\n)\s*(?:final answer|final diagnosis|diagnosis|answer)\s*[:：]\s*(.+?)\s*$",
        r"(?is)\bthe answer is\s+(.+?)\s*$",
        r"(?is)\btherefore[, ]+(?:the )?(?:final )?(?:answer|diagnosis) is\s+(.+?)\s*$",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, cleaned))
        if matches:
            candidate = normalize_text(matches[-1].group(1))
            if candidate:
                return candidate
    last_line = cleaned.splitlines()[-1].strip()
    return last_line


def split_prediction_text(text: str, options: Mapping[str, str] | None = None) -> Tuple[str, str]:
    cleaned = strip_markdown_artifacts(text)
    if not cleaned:
        return "", ""

    answer_text = extract_final_answer_text(cleaned, options)
    answer_start = None
    answer_patterns = [
        r"(?is)(?:^|\n)\s*(?:final answer|final diagnosis|diagnosis|answer)\s*[:：]\s*",
        r"(?is)\bthe answer is\s+",
        r"(?is)\btherefore[, ]+(?:the )?(?:final )?(?:answer|diagnosis) is\s+",
    ]
    for pattern in answer_patterns:
        matches = list(re.finditer(pattern, cleaned))
        if matches:
            answer_start = matches[-1].start()
            break
    if answer_start is not None:
        reasoning = cleaned[:answer_start].strip()
        if reasoning:
            return reasoning, answer_text
    return cleaned, answer_text


def normalize_error_type(raw: Any) -> str:
    text = normalize_text(raw).lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "f1": "F1",
        "evidence_omission": "F1",
        "omission": "F1",
        "f2": "F2",
        "evidence_misinterpretation": "F2",
        "misinterpretation": "F2",
        "f3": "F3",
        "evidence_overinterpretation": "F3",
        "overinterpretation": "F3",
        "evidence_overreach": "F3",
        "overreach": "F3",
        "f4": "F4",
        "reasoning_answer_mismatch": "F4",
        "reasoninganswermismatch": "F4",
        "derivational_mismatch": "F4",
        "mismatch": "F4",
        "faithful": "NONE",
        "none": "NONE",
        "no_error": "NONE",
        "no_major_error": "NONE",
        "mixed": "MIXED",
        "invalid": "INVALID",
    }
    return mapping.get(text, normalize_text(raw).upper())


def pick_override_or_candidates(override: str, candidates: Sequence[str]) -> List[str]:
    override_fields = parse_csv_fields(override)
    return override_fields if override_fields else list(candidates)


def build_case_context(row: Dict[str, Any], args: argparse.Namespace) -> str:
    question_field_candidates = pick_override_or_candidates(args.question_field, QUESTION_FIELD_CANDIDATES)
    case_text = first_non_empty(row, question_field_candidates)
    options_text = format_options(row.get("options"))
    if options_text:
        if case_text:
            return f"{case_text}\n\nOptions:\n{options_text}"
        return f"Options:\n{options_text}"
    return case_text


def build_evidence_text(row: Dict[str, Any], args: argparse.Namespace) -> str:
    evidence_field_candidates = pick_override_or_candidates(args.evidence_field, EVIDENCE_FIELD_CANDIDATES)
    return first_non_empty(row, evidence_field_candidates)


def detect_reference_fields(row: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, str]:
    strict_reasoning_candidates = pick_override_or_candidates(
        args.reference_reasoning_field,
        REFERENCE_REASONING_STRICT_FIELDS,
    )
    strict_answer_candidates = pick_override_or_candidates(
        args.reference_answer_field,
        REFERENCE_ANSWER_STRICT_FIELDS,
    )
    reference_reasoning = first_non_empty(row, strict_reasoning_candidates)
    reference_answer = first_non_empty(row, strict_answer_candidates)

    has_prediction_markers = any(
        is_non_empty(row.get(field))
        for field in (
            *PREDICTION_TEXT_FIELDS,
            *PREDICTED_REASONING_FIELDS,
            *PREDICTED_ANSWER_FIELDS,
        )
    )

    if not reference_reasoning and has_prediction_markers:
        reference_reasoning = first_non_empty(row, GENERIC_REFERENCE_REASONING_FIELDS)
    if not reference_answer and has_prediction_markers:
        reference_answer = first_non_empty(row, GENERIC_REFERENCE_ANSWER_FIELDS)

    return reference_reasoning, reference_answer


def detect_prediction_fields(
    row: Dict[str, Any],
    reference_reasoning: str,
    reference_answer: str,
    args: argparse.Namespace,
) -> Tuple[str, str, str]:
    options = get_options_map(row.get("options"))
    prediction_text = first_non_empty(
        row,
        pick_override_or_candidates(args.prediction_field, PREDICTION_TEXT_FIELDS),
    )
    predicted_reasoning = first_non_empty(
        row,
        pick_override_or_candidates(args.predicted_reasoning_field, PREDICTED_REASONING_FIELDS),
    )
    predicted_answer = first_non_empty(
        row,
        pick_override_or_candidates(args.predicted_answer_field, PREDICTED_ANSWER_FIELDS),
    )

    if prediction_text:
        parsed_reasoning, parsed_answer = split_prediction_text(prediction_text, options)
        if not predicted_reasoning:
            predicted_reasoning = parsed_reasoning
        if not predicted_answer:
            predicted_answer = parsed_answer

    if not prediction_text and not predicted_reasoning:
        generic_reasoning = normalize_text(row.get("reasoning"))
        if generic_reasoning and generic_reasoning != normalize_text(reference_reasoning):
            predicted_reasoning = generic_reasoning

    if not prediction_text and not predicted_answer:
        for field in ("final_answer", "answer"):
            generic_answer = normalize_text(row.get(field))
            if generic_answer and generic_answer != normalize_text(reference_answer):
                predicted_answer = generic_answer
                break

    return prediction_text, predicted_reasoning, predicted_answer


def detect_gold_choice_label(row: Dict[str, Any], reference_answer: str) -> str:
    for key in ("answer_idx", "label"):
        label = normalize_text(row.get(key)).upper()
        if label:
            return label
    options = get_options_map(row.get("options"))
    if not options or not reference_answer:
        return ""
    normalized_reference = normalize_answer(reference_answer)
    for label, text in options.items():
        if normalize_answer(text) == normalized_reference:
            return str(label).upper()
    return ""


def compute_answer_match(
    row: Dict[str, Any],
    prediction_text: str,
    predicted_answer: str,
    reference_answer: str,
) -> Dict[str, Any]:
    options = get_options_map(row.get("options"))
    gold_label = detect_gold_choice_label(row, reference_answer)
    predicted_label = ""
    if prediction_text:
        predicted_label = extract_choice_label(prediction_text, options)
    if not predicted_label and predicted_answer:
        predicted_label = extract_choice_label(predicted_answer, options)

    if gold_label and predicted_label:
        exact_match = gold_label == predicted_label
        return {
            "available": True,
            "method": "choice_label",
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "exact_match": exact_match,
        }

    if reference_answer and predicted_answer:
        exact_match = normalize_answer(reference_answer) == normalize_answer(predicted_answer)
        return {
            "available": True,
            "method": "normalized_text",
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "exact_match": exact_match,
        }

    return {
        "available": False,
        "method": "none",
        "gold_label": gold_label,
        "predicted_label": predicted_label,
        "exact_match": None,
    }


def resolve_example_id(row: Dict[str, Any], index: int, args: argparse.Namespace) -> str:
    candidates = pick_override_or_candidates(args.id_field, ID_FIELD_CANDIDATES)
    example_id = first_non_empty(row, candidates)
    if example_id:
        return example_id
    return f"row_{index}"


def resolve_group_fields(rows: Sequence[Dict[str, Any]], raw_group_by: str) -> List[str]:
    text = normalize_text(raw_group_by).lower()
    if not text or text == "none":
        return []
    if text != "auto":
        return parse_csv_fields(raw_group_by)
    for field in GROUP_FIELD_CANDIDATES:
        if any(is_non_empty(row.get(field)) for row in rows):
            return [field]
    return []


def resolve_analysis_profile(row: Dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    preferred = normalize_text(row.get("preferred_analysis_profile"))
    if preferred in ANALYSIS_PROFILES and preferred != "auto":
        return preferred
    has_options = isinstance(row.get("options"), Mapping) or (
        isinstance(row.get("options"), Sequence) and not isinstance(row.get("options"), (str, bytes))
    )
    has_reference_reasoning = is_non_empty(
        first_non_empty(row, (*REFERENCE_REASONING_STRICT_FIELDS, *GENERIC_REFERENCE_REASONING_FIELDS))
    )
    if has_options and not has_reference_reasoning:
        return "mcq_f4"
    return "case_full"


def resolve_group_key(row: Dict[str, Any], group_fields: Sequence[str]) -> str:
    if not group_fields:
        return "overall"
    parts = []
    for field in group_fields:
        value = normalize_text(row.get(field)) or "NA"
        parts.append(f"{field}={value}")
    return " | ".join(parts)


def build_user_prompt(normalized: Dict[str, Any], analysis_profile: str) -> str:
    case_context = normalized.get("case_context") or "(none)"
    evidence = normalized.get("evidence_text") or "(none)"
    reference_reasoning = normalized.get("reference_reasoning") or "(not available)"
    reference_answer = normalized.get("reference_answer") or "(not available)"
    raw_response = normalized.get("prediction_text") or "(not available)"
    predicted_reasoning = normalized.get("predicted_reasoning") or "(not available)"
    predicted_answer = normalized.get("predicted_answer") or "(not available)"
    if analysis_profile == "mcq_f4":
        profile_instructions = (
            "[Profile]\n"
            "This is an MCQ-style external benchmark item.\n"
            "Focus primarily on F4 (reasoning-answer mismatch).\n"
            "Do not assign F1/F2/F3 unless the model explicitly introduces and then "
            "clearly omits/misinterprets/overstates concrete evidence inside its own visible reasoning.\n"
            "When evidence grounding is too weak for F1/F2/F3, prefer NONE rather than over-calling them.\n"
        )
    else:
        profile_instructions = (
            "[Profile]\n"
            "This is a case-style item with enough context to evaluate the full F1/F2/F3/F4 taxonomy.\n"
        )
    return f"""
[Case / Question]
{case_context}

[Retrieved Evidence]
{evidence}

[Reference Reasoning]
{reference_reasoning}

[Reference Final Answer]
{reference_answer}

[Model Full Response]
{raw_response}

[Parsed Model Reasoning]
{predicted_reasoning}

[Parsed Model Final Answer]
{predicted_answer}

[Profile]
{profile_instructions}

[Definitions]
- F1 Evidence Omission: clinically important available evidence is omitted or mentioned without being substantively used in the reasoning chain.
- F2 Evidence Misinterpretation: available evidence is used, but its clinical meaning is interpreted incorrectly.
- F3 Evidence Overinterpretation: available evidence is interpreted too strongly and supports claims beyond what it warrants.
- F4 Reasoning-Answer Mismatch: the final answer is not sufficiently supported by the preceding reasoning or is inconsistent with it.

Return one JSON object with exactly these keys:
- "dominant_error_type": one of F1, F2, F3, F4, NONE, MIXED, INVALID
- "has_f1": boolean
- "has_f2": boolean
- "has_f3": boolean
- "has_f4": boolean
- "single_dominant_error": boolean
- "answer_supported_by_reasoning": boolean or null
- "clinical_plausibility_score": integer 1-5
- "confidence_score": integer 1-5
- "short_rationale": short string
""".strip()


def normalize_judge_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    dominant = normalize_error_type(payload.get("dominant_error_type")) or "INVALID"
    if dominant not in {*ERROR_TYPES, *NON_ERROR_TYPES}:
        dominant = "INVALID"
    analysis = {
        "dominant_error_type": dominant,
        "has_f1": as_true(payload.get("has_f1")),
        "has_f2": as_true(payload.get("has_f2")),
        "has_f3": as_true(payload.get("has_f3")),
        "has_f4": as_true(payload.get("has_f4")),
        "single_dominant_error": as_true(payload.get("single_dominant_error")),
        "answer_supported_by_reasoning": parse_bool(payload.get("answer_supported_by_reasoning")),
        "clinical_plausibility_score": max(
            1,
            min(5, safe_int(payload.get("clinical_plausibility_score"), 1)),
        ),
        "confidence_score": max(1, min(5, safe_int(payload.get("confidence_score"), 1))),
        "short_rationale": normalize_text(payload.get("short_rationale")),
    }
    if dominant in ERROR_TYPES:
        for error_type in ERROR_TYPES:
            analysis[f"has_{error_type.lower()}"] = analysis[f"has_{error_type.lower()}"] or (dominant == error_type)
    return analysis


def analysis_from_existing_row(row: Dict[str, Any]) -> Dict[str, Any]:
    existing = row.get("faithfulness_analysis")
    if isinstance(existing, dict) and existing.get("dominant_error_type"):
        copied = dict(existing)
        copied["dominant_error_type"] = normalize_error_type(copied.get("dominant_error_type"))
        return copied

    llm_qc = row.get("llm_qc")
    if isinstance(llm_qc, dict):
        dominant = normalize_error_type(llm_qc.get("realized_error_type"))
        return {
            "dominant_error_type": dominant or "INVALID",
            "has_f1": dominant == "F1",
            "has_f2": dominant == "F2",
            "has_f3": dominant == "F3",
            "has_f4": dominant == "F4",
            "single_dominant_error": as_true(llm_qc.get("single_dominant_error")),
            "answer_supported_by_reasoning": parse_bool(llm_qc.get("answer_preserved_judged")),
            "clinical_plausibility_score": safe_int(
                llm_qc.get("clinical_plausibility_score"),
                3,
            ),
            "confidence_score": 3,
            "short_rationale": normalize_text(llm_qc.get("reason")),
        }

    dominant = normalize_error_type(row.get("dominant_error_type"))
    if dominant in {*ERROR_TYPES, *NON_ERROR_TYPES}:
        return {
            "dominant_error_type": dominant,
            "has_f1": dominant == "F1",
            "has_f2": dominant == "F2",
            "has_f3": dominant == "F3",
            "has_f4": dominant == "F4",
            "single_dominant_error": as_true(row.get("single_dominant_error")),
            "answer_supported_by_reasoning": parse_bool(row.get("answer_supported_by_reasoning")),
            "clinical_plausibility_score": safe_int(row.get("clinical_plausibility_score"), 3),
            "confidence_score": safe_int(row.get("confidence_score"), 3),
            "short_rationale": normalize_text(row.get("short_rationale")),
        }

    target = normalize_error_type(row.get("error_symbol") or row.get("error_type"))
    if target in ERROR_TYPES:
        return {
            "dominant_error_type": target,
            "has_f1": target == "F1",
            "has_f2": target == "F2",
            "has_f3": target == "F3",
            "has_f4": target == "F4",
            "single_dominant_error": as_true(row.get("adjudicated_single_dominant_error")),
            "answer_supported_by_reasoning": None,
            "clinical_plausibility_score": safe_int(
                row.get("adjudicated_clinical_plausibility"),
                3,
            ),
            "confidence_score": 3,
            "short_rationale": normalize_text(row.get("adjudicated_notes") or row.get("rewrite_self_check")),
        }

    raise ValueError("Could not find reusable faithfulness labels in the input row.")


def normalize_row(
    row: Dict[str, Any],
    index: int,
    group_fields: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    example_id = resolve_example_id(row, index, args)
    case_context = build_case_context(row, args)
    evidence_text = build_evidence_text(row, args)
    reference_reasoning, reference_answer = detect_reference_fields(row, args)
    prediction_text, predicted_reasoning, predicted_answer = detect_prediction_fields(
        row=row,
        reference_reasoning=reference_reasoning,
        reference_answer=reference_answer,
        args=args,
    )
    answer_match = compute_answer_match(
        row=row,
        prediction_text=prediction_text,
        predicted_answer=predicted_answer,
        reference_answer=reference_answer,
    )
    group_key = resolve_group_key(row, group_fields)
    normalized = {
        "example_id": example_id,
        "case_context": case_context,
        "evidence_text": evidence_text,
        "reference_reasoning": reference_reasoning,
        "reference_answer": reference_answer,
        "prediction_text": prediction_text,
        "predicted_reasoning": predicted_reasoning,
        "predicted_answer": predicted_answer,
        "group_key": group_key,
        "answer_match": answer_match,
        "reference_available": bool(reference_reasoning or reference_answer or evidence_text or case_context),
        "prediction_available": bool(prediction_text or predicted_reasoning or predicted_answer),
    }
    return normalized


async def judge_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    normalized: Dict[str, Any],
    analysis_profile: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(normalized, analysis_profile)},
    ]
    max_retries = max(int(args.request_max_retries), 0)
    base_backoff = max(float(args.request_retry_backoff), 0.0)
    async with semaphore:
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                break
            except Exception as exc:  # pragma: no cover - backend behavior is environment-specific
                last_exc = exc
                if attempt >= max_retries:
                    example_id = normalize_text(normalized.get("example_id")) or "<unknown>"
                    raise RuntimeError(f"Judge failed for `{example_id}`: {exc}") from exc
                if base_backoff > 0:
                    await asyncio.sleep(base_backoff * (2**attempt))
        else:  # pragma: no cover - loop always breaks or raises
            raise RuntimeError(f"Judge failed without an exception: {last_exc}")
    raw_output = response.choices[0].message.content or ""
    payload = extract_json(raw_output)
    analysis = normalize_judge_payload(payload)
    analysis["judge_model_requested"] = args.model
    analysis["judge_model_snapshot"] = response.model
    analysis["raw_judge_output"] = raw_output
    return analysis


def completed_ids(path: str) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    ids = set()
    for row in load_jsonl(path):
        analysis = row.get("faithfulness_analysis") if isinstance(row, dict) else None
        if not isinstance(analysis, dict):
            continue
        status = normalize_text(analysis.get("status"))
        if status in STATUS_DONE:
            ids.add(normalize_text(analysis.get("example_id") or row.get("id")))
    return ids


def analysis_example_id(row: Mapping[str, Any]) -> str:
    analysis = row.get("faithfulness_analysis") if isinstance(row, dict) else None
    if isinstance(analysis, dict):
        value = normalize_text(analysis.get("example_id"))
        if value:
            return value
    if isinstance(row, dict):
        return normalize_text(row.get("id") or row.get("source_id") or row.get("example_id"))
    return ""


def ordered_upsert_rows(
    existing_rows: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    ordered_ids: List[str] = []
    row_map: Dict[str, Dict[str, Any]] = {}
    passthrough_rows: List[Dict[str, Any]] = []

    for row in list(existing_rows) + list(new_rows):
        row_id = analysis_example_id(row)
        if not row_id:
            passthrough_rows.append(dict(row))
            continue
        if row_id not in row_map:
            ordered_ids.append(row_id)
        row_map[row_id] = dict(row)

    return [row_map[row_id] for row_id in ordered_ids] + passthrough_rows


def build_output_row(
    row: Dict[str, Any],
    normalized: Dict[str, Any],
    analysis_profile: str,
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    output_row = dict(row)
    output_row["faithfulness_analysis"] = {
        "example_id": normalized["example_id"],
        "group_key": normalized["group_key"],
        "analysis_profile": analysis_profile,
        "case_context_available": bool(normalized["case_context"]),
        "evidence_available": bool(normalized["evidence_text"]),
        "reference_reasoning_available": bool(normalized["reference_reasoning"]),
        "reference_answer_available": bool(normalized["reference_answer"]),
        "prediction_available": normalized["prediction_available"],
        "parsed_predicted_answer": normalized["predicted_answer"],
        "answer_match": normalized["answer_match"],
        **analysis,
    }
    return output_row


def rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round(num / den, 6)


def safe_mean(values: Iterable[int | float | None]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 4)


def format_rate(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def summarize_rows(
    rows: Sequence[Dict[str, Any]],
    input_file: str,
    annotation_mode: str,
    group_fields: Sequence[str],
) -> Dict[str, Any]:
    analyses = []
    for row in rows:
        analysis = row.get("faithfulness_analysis")
        if isinstance(analysis, dict):
            analyses.append(analysis)

    total = len(analyses)
    dominant_counter = Counter()
    presence_counter = Counter()
    group_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    status_counter = Counter()
    model_ids = set()
    dataset_names = set()
    profiles = Counter()

    for analysis in analyses:
        dominant_counter[str(analysis.get("dominant_error_type", "INVALID"))] += 1
        for error_type in ERROR_TYPES:
            if analysis.get(f"has_{error_type.lower()}"):
                presence_counter[error_type] += 1
        status_counter[str(analysis.get("status", "unknown"))] += 1
        group_buckets[str(analysis.get("group_key", "overall"))].append(analysis)
        if is_non_empty(analysis.get("analysis_profile")):
            profiles[str(analysis.get("analysis_profile"))] += 1

    for row in rows:
        if is_non_empty(row.get("model_id")):
            model_ids.add(normalize_text(row.get("model_id")))
        if is_non_empty(row.get("dataset_name")):
            dataset_names.add(normalize_text(row.get("dataset_name")))
        elif is_non_empty(row.get("benchmark")):
            dataset_names.add(normalize_text(row.get("benchmark")))

    answer_available = [analysis for analysis in analyses if analysis.get("answer_match", {}).get("available")]
    answer_correct = [
        analysis
        for analysis in answer_available
        if analysis.get("answer_match", {}).get("exact_match") is True
    ]

    any_error_rows = [
        analysis
        for analysis in analyses
        if any(bool(analysis.get(f"has_{error_type.lower()}")) for error_type in ERROR_TYPES)
    ]
    no_error_rows = [
        analysis
        for analysis in analyses
        if not any(bool(analysis.get(f"has_{error_type.lower()}")) for error_type in ERROR_TYPES)
    ]

    def summarize_group(items: Sequence[Dict[str, Any]], group_name: str) -> Dict[str, Any]:
        dominant = Counter(str(item.get("dominant_error_type", "INVALID")) for item in items)
        presence = Counter()
        for item in items:
            for error_type in ERROR_TYPES:
                if item.get(f"has_{error_type.lower()}"):
                    presence[error_type] += 1
        answer_items = [item for item in items if item.get("answer_match", {}).get("available")]
        answer_correct_items = [
            item for item in answer_items if item.get("answer_match", {}).get("exact_match") is True
        ]
        return {
            "group": group_name,
            "count": len(items),
            "dominant_error_counts": {key: dominant.get(key, 0) for key in (*ERROR_TYPES, *NON_ERROR_TYPES)},
            "dominant_error_rates": {
                key: rate(dominant.get(key, 0), len(items))
                for key in (*ERROR_TYPES, *NON_ERROR_TYPES)
            },
            "presence_counts": {error_type: presence.get(error_type, 0) for error_type in ERROR_TYPES},
            "presence_rates": {
                error_type: rate(presence.get(error_type, 0), len(items))
                for error_type in ERROR_TYPES
            },
            "single_dominant_error_rate": rate(
                sum(1 for item in items if item.get("single_dominant_error")),
                len(items),
            ),
            "answer_supported_rate": rate(
                sum(1 for item in items if item.get("answer_supported_by_reasoning") is True),
                sum(1 for item in items if item.get("answer_supported_by_reasoning") is not None),
            ),
            "mean_clinical_plausibility": safe_mean(
                item.get("clinical_plausibility_score") for item in items
            ),
            "mean_confidence": safe_mean(item.get("confidence_score") for item in items),
            "answer_exact_match_rate": rate(len(answer_correct_items), len(answer_items)),
            "answer_exact_match_count": len(answer_correct_items),
            "answer_exact_match_available": len(answer_items),
        }

    group_summaries = [
        summarize_group(items, group_name)
        for group_name, items in sorted(group_buckets.items())
    ]

    accuracy_by_dominant = {}
    for key in (*ERROR_TYPES, *NON_ERROR_TYPES):
        items = [item for item in answer_available if item.get("dominant_error_type") == key]
        correct = [item for item in items if item.get("answer_match", {}).get("exact_match") is True]
        accuracy_by_dominant[key] = {
            "count": len(items),
            "exact_match_rate": rate(len(correct), len(items)),
        }

    summary = {
        "generated_at": iso_utc_now(),
        "input_file": str(input_file),
        "annotation_mode": annotation_mode,
        "group_fields": list(group_fields),
        "analysis_profiles": dict(profiles),
        "available_model_ids": sorted(model_ids),
        "available_datasets": sorted(dataset_names),
        "overall": {
            "count": total,
            "status_counts": dict(status_counter),
            "dominant_error_counts": {
                key: dominant_counter.get(key, 0) for key in (*ERROR_TYPES, *NON_ERROR_TYPES)
            },
            "dominant_error_rates": {
                key: rate(dominant_counter.get(key, 0), total) for key in (*ERROR_TYPES, *NON_ERROR_TYPES)
            },
            "presence_counts": {error_type: presence_counter.get(error_type, 0) for error_type in ERROR_TYPES},
            "presence_rates": {
                error_type: rate(presence_counter.get(error_type, 0), total) for error_type in ERROR_TYPES
            },
            "single_dominant_error_rate": rate(
                sum(1 for analysis in analyses if analysis.get("single_dominant_error")),
                total,
            ),
            "answer_supported_rate": rate(
                sum(1 for analysis in analyses if analysis.get("answer_supported_by_reasoning") is True),
                sum(1 for analysis in analyses if analysis.get("answer_supported_by_reasoning") is not None),
            ),
            "mean_clinical_plausibility": safe_mean(
                analysis.get("clinical_plausibility_score") for analysis in analyses
            ),
            "mean_confidence": safe_mean(analysis.get("confidence_score") for analysis in analyses),
            "answer_exact_match_rate": rate(len(answer_correct), len(answer_available)),
            "answer_exact_match_count": len(answer_correct),
            "answer_exact_match_available": len(answer_available),
            "any_faithfulness_error_rate": rate(len(any_error_rows), total),
            "no_faithfulness_error_rate": rate(len(no_error_rows), total),
        },
        "accuracy_by_dominant_error": accuracy_by_dominant,
        "group_summaries": group_summaries,
    }
    summary["observations"] = derive_observations(summary, answer_available, any_error_rows, no_error_rows)
    return summary


def derive_observations(
    summary: Dict[str, Any],
    answer_available: Sequence[Dict[str, Any]],
    any_error_rows: Sequence[Dict[str, Any]],
    no_error_rows: Sequence[Dict[str, Any]],
) -> List[str]:
    observations: List[str] = []
    overall = summary.get("overall", {})
    dominant_counts = overall.get("dominant_error_counts", {})
    dominant_rates = overall.get("dominant_error_rates", {})

    top_error = None
    top_count = -1
    tied_top_errors: List[str] = []
    for error_type in ERROR_TYPES:
        count = int(dominant_counts.get(error_type, 0) or 0)
        if count > top_count:
            top_error = error_type
            top_count = count
            tied_top_errors = [error_type]
        elif count == top_count and count > 0:
            tied_top_errors.append(error_type)
    if top_error and top_count > 0 and len(tied_top_errors) == 1:
        observations.append(
            f"Most common dominant faithfulness error: {top_error} "
            f"({(dominant_rates.get(top_error) or 0.0) * 100:.1f}%)."
        )
    elif tied_top_errors:
        observations.append(
            "Top dominant faithfulness errors were tied across "
            + ", ".join(tied_top_errors)
            + "."
        )
    elif int(dominant_counts.get("NONE", 0) or 0) > 0:
        observations.append(
            f"No clear F1-F4 error was the dominant label for "
            f"{(dominant_rates.get('NONE') or 0.0) * 100:.1f}% of responses."
        )

    if (dominant_rates.get("F4") or 0.0) >= 0.2:
        observations.append(
            f"F4 is frequent ({(dominant_rates.get('F4') or 0.0) * 100:.1f}%), "
            "suggesting answers often outrun the preceding reasoning."
        )

    if (dominant_rates.get("INVALID") or 0.0) >= 0.1:
        observations.append(
            f"{(dominant_rates.get('INVALID') or 0.0) * 100:.1f}% of rows were "
            "too weak or too underspecified to judge cleanly."
        )

    if answer_available:
        faithful_correct = sum(
            1
            for item in no_error_rows
            if item.get("answer_match", {}).get("available")
            and item.get("answer_match", {}).get("exact_match") is True
        )
        faithful_total = sum(
            1 for item in no_error_rows if item.get("answer_match", {}).get("available")
        )
        unfaithful_correct = sum(
            1
            for item in any_error_rows
            if item.get("answer_match", {}).get("available")
            and item.get("answer_match", {}).get("exact_match") is True
        )
        unfaithful_total = sum(
            1 for item in any_error_rows if item.get("answer_match", {}).get("available")
        )
        if faithful_total > 0 and unfaithful_total > 0:
            faithful_rate = faithful_correct / faithful_total
            unfaithful_rate = unfaithful_correct / unfaithful_total
            observations.append(
                f"Exact-match answer accuracy was {faithful_rate * 100:.1f}% on rows without "
                f"detected F1-F4 errors versus {unfaithful_rate * 100:.1f}% on rows with at "
                "least one detected F1-F4 error."
            )

    group_summaries = summary.get("group_summaries", [])
    if len(group_summaries) >= 2:
        scored = []
        for item in group_summaries:
            burden = sum(item.get("presence_rates", {}).get(error_type, 0.0) or 0.0 for error_type in ERROR_TYPES)
            scored.append((burden, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if scored and abs(scored[0][0] - scored[-1][0]) > 1e-9:
            worst = scored[0][1]
            best = scored[-1][1]
            observations.append(
                f"Highest overall error burden appeared in `{worst['group']}`, while "
                f"`{best['group']}` was the cleanest group under the same judge."
            )

    return observations


def write_group_summary_csv(path: str, summary: Dict[str, Any]) -> None:
    rows = []
    for item in summary.get("group_summaries", []):
        row = {
            "group": item.get("group"),
            "count": item.get("count"),
            "single_dominant_error_rate": item.get("single_dominant_error_rate"),
            "answer_supported_rate": item.get("answer_supported_rate"),
            "mean_clinical_plausibility": item.get("mean_clinical_plausibility"),
            "mean_confidence": item.get("mean_confidence"),
            "answer_exact_match_rate": item.get("answer_exact_match_rate"),
        }
        for key in (*ERROR_TYPES, *NON_ERROR_TYPES):
            row[f"dominant_{key}_count"] = item.get("dominant_error_counts", {}).get(key, 0)
            row[f"dominant_{key}_rate"] = item.get("dominant_error_rates", {}).get(key)
        for key in ERROR_TYPES:
            row[f"presence_{key}_count"] = item.get("presence_counts", {}).get(key, 0)
            row[f"presence_{key}_rate"] = item.get("presence_rates", {}).get(key)
        rows.append(row)

    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            writer = csv.writer(handle)
            writer.writerow(["group", "count"])
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: Dict[str, Any]) -> str:
    overall = summary.get("overall", {})
    lines = [
        "# Faithfulness Error Report",
        "",
        f"- Input file: `{summary.get('input_file', '')}`",
        f"- Annotation mode: `{summary.get('annotation_mode', '')}`",
        f"- Total analyzed rows: {overall.get('count', 0)}",
        "",
        "## Overall",
        "",
    ]

    for key in ERROR_TYPES:
        count = overall.get("presence_counts", {}).get(key, 0)
        rate_value = overall.get("presence_rates", {}).get(key)
        lines.append(f"- {key} present: {count} ({format_rate(rate_value)})")
    for key in (*NON_ERROR_TYPES,):
        count = overall.get("dominant_error_counts", {}).get(key, 0)
        rate_value = overall.get("dominant_error_rates", {}).get(key)
        lines.append(f"- Dominant {key}: {count} ({format_rate(rate_value)})")

    lines.extend(
        [
            f"- Single-dominant-error rate: {format_rate(overall.get('single_dominant_error_rate'))}",
            f"- Answer-supported rate: {format_rate(overall.get('answer_supported_rate'))}",
            f"- Mean clinical plausibility: {overall.get('mean_clinical_plausibility')}",
            f"- Mean judge confidence: {overall.get('mean_confidence')}",
        ]
    )

    if overall.get("answer_exact_match_available", 0):
        lines.append(
            f"- Exact-match answer accuracy: "
            f"{format_rate(overall.get('answer_exact_match_rate'))} "
            f"({overall.get('answer_exact_match_count', 0)}/"
            f"{overall.get('answer_exact_match_available', 0)})"
        )

    observations = summary.get("observations", [])
    if observations:
        lines.extend(["", "## Observations", ""])
        for item in observations:
            lines.append(f"- {item}")

    group_summaries = summary.get("group_summaries", [])
    if group_summaries:
        lines.extend(["", "## Group Breakdown", ""])
        for item in group_summaries:
            lines.append(
                f"- `{item['group']}`: n={item['count']}, "
                f"F1={((item.get('presence_rates', {}).get('F1') or 0.0) * 100):.1f}%, "
                f"F2={((item.get('presence_rates', {}).get('F2') or 0.0) * 100):.1f}%, "
                f"F3={((item.get('presence_rates', {}).get('F3') or 0.0) * 100):.1f}%, "
                f"F4={((item.get('presence_rates', {}).get('F4') or 0.0) * 100):.1f}%."
            )

    return "\n".join(lines) + "\n"


async def main_async(args: argparse.Namespace) -> None:
    if args.annotation_mode == "llm" and AsyncOpenAI is None:
        raise RuntimeError("Missing openai dependency. Install openai>=1.30.0 first.")
    if args.annotation_mode == "llm" and not args.model:
        raise RuntimeError("--model or OPENAI_MODEL is required in llm mode.")

    if not args.output_file:
        args.output_file = default_output_path(args.input_file, ".faithfulness_analysis.jsonl")
    if not args.summary_file:
        args.summary_file = default_output_path(args.input_file, ".faithfulness_summary.json")
    if not args.report_file:
        args.report_file = default_output_path(args.input_file, ".faithfulness_report.md")
    if not args.group_summary_csv:
        args.group_summary_csv = default_output_path(args.input_file, ".faithfulness_group_summary.csv")

    rows = load_records(args.input_file)
    if args.start > 0 or args.end > 0:
        end = None if args.end < 0 else args.end
        rows = rows[args.start:end]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    group_fields = resolve_group_fields(rows, args.group_by)

    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    existing_rows = load_jsonl(args.output_file) if os.path.exists(args.output_file) else []
    already_done = completed_ids(args.output_file) if not args.overwrite else set()
    output_rows: List[Dict[str, Any]] = []
    pending: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    for index, row in enumerate(rows):
        analysis_profile = resolve_analysis_profile(row, args.analysis_profile)
        normalized = normalize_row(row=row, index=index, group_fields=group_fields, args=args)
        if normalized["example_id"] in already_done:
            continue

        if args.annotation_mode == "existing":
            try:
                existing_analysis = analysis_from_existing_row(row)
                existing_analysis["status"] = "reused_existing_label"
            except Exception as exc:
                existing_analysis = {
                    "status": "existing_label_error",
                    "dominant_error_type": "INVALID",
                    "has_f1": False,
                    "has_f2": False,
                    "has_f3": False,
                    "has_f4": False,
                    "single_dominant_error": False,
                    "answer_supported_by_reasoning": None,
                    "clinical_plausibility_score": None,
                    "confidence_score": None,
                    "short_rationale": f"Could not reuse existing label: {exc}",
                }
            output_row = build_output_row(row, normalized, analysis_profile, existing_analysis)
            append_jsonl(args.output_file, output_row)
            output_rows.append(output_row)
            continue

        reasoning_length = len(normalized.get("predicted_reasoning") or "")
        if reasoning_length < args.min_reasoning_chars:
            analysis = {
                "status": "skipped_no_reasoning",
                "dominant_error_type": "INVALID",
                "has_f1": False,
                "has_f2": False,
                "has_f3": False,
                "has_f4": False,
                "single_dominant_error": False,
                "answer_supported_by_reasoning": None,
                "clinical_plausibility_score": None,
                "confidence_score": None,
                "short_rationale": "Predicted reasoning is too short to judge reliably.",
            }
            output_row = build_output_row(row, normalized, analysis_profile, analysis)
            append_jsonl(args.output_file, output_row)
            output_rows.append(output_row)
            continue

        pending.append((row, normalized, analysis_profile))

    if args.annotation_mode == "llm" and pending:
        client_kwargs = {
            "api_key": args.api_key,
            "max_retries": 0,
            "timeout": args.request_timeout,
        }
        if args.base_url:
            client_kwargs["base_url"] = args.base_url
        http_client = httpx.AsyncClient(
            trust_env=False,
            timeout=httpx.Timeout(args.request_timeout),
        )
        client_kwargs["http_client"] = http_client
        client = AsyncOpenAI(**client_kwargs)
        semaphore = asyncio.Semaphore(args.max_concurrency)

        async def run_one(
            row: Dict[str, Any],
            normalized: Dict[str, Any],
            analysis_profile: str,
        ) -> Dict[str, Any]:
            try:
                analysis = await judge_one(client, semaphore, normalized, analysis_profile, args)
                analysis["status"] = "judged"
            except Exception as exc:
                analysis = {
                    "status": "judge_failed",
                    "dominant_error_type": "INVALID",
                    "has_f1": False,
                    "has_f2": False,
                    "has_f3": False,
                    "has_f4": False,
                    "single_dominant_error": False,
                    "answer_supported_by_reasoning": None,
                    "clinical_plausibility_score": None,
                    "confidence_score": None,
                    "short_rationale": f"Judge call failed: {exc}",
                }
            output_row = build_output_row(row, normalized, analysis_profile, analysis)
            append_jsonl(args.output_file, output_row)
            return output_row

        tasks = [run_one(row, normalized, analysis_profile) for row, normalized, analysis_profile in pending]
        try:
            for future in asyncio.as_completed(tasks):
                output_rows.append(await future)
        finally:
            await http_client.aclose()

    all_rows = ordered_upsert_rows(existing_rows, output_rows)
    if all_rows:
        write_jsonl(args.output_file, all_rows)
    summary = summarize_rows(
        rows=all_rows,
        input_file=args.input_file,
        annotation_mode=args.annotation_mode,
        group_fields=group_fields,
    )
    write_json(args.summary_file, summary)
    write_group_summary_csv(args.group_summary_csv, summary)
    report_text = render_report(summary)
    report_path = ensure_parent(args.report_file)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(report_text)

    overall = summary.get("overall", {})
    print(f"Annotated rows: {overall.get('count', 0)}")
    print(f"Output JSONL: {args.output_file}")
    print(f"Summary JSON: {args.summary_file}")
    print(f"Markdown report: {args.report_file}")
    print(f"Group summary CSV: {args.group_summary_csv}")
    print(
        "Dominant counts:",
        json.dumps(overall.get("dominant_error_counts", {}), ensure_ascii=False),
    )


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
