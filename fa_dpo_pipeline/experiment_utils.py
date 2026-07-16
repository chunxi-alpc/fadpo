from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from common import (
    ensure_parent,
    load_jsonl,
    normalize_answer,
    normalize_text,
    read_json,
    safe_float,
    to_bool,
    write_json,
    write_jsonl,
)


ID_KEYS = ("id", "question_id", "sample_id", "example_id", "uid", "source_id")
PROMPT_KEYS = ("patient_context", "case_prompt", "question", "prompt", "input", "context")
REASONING_KEYS = (
    "reasoning",
    "negative_reasoning",
    "predicted_reasoning",
    "diagnostic_reasoning",
    "generated_reasoning",
    "response_reasoning",
    "chosen",
    "rejected",
    "response",
    "output",
)
ANSWER_KEYS = (
    "negative_final_answer",
    "predicted_answer_text",
    "final_answer",
    "answer",
    "prediction",
    "generated_answer",
    "final_diagnosis",
)
GOLD_ANSWER_KEYS = (
    "gold_answer",
    "correct_answer_text",
    "reference_answer",
    "target_answer",
    "final_diagnosis",
    "answer",
)


def first_available(record: Mapping[str, Any], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        if key not in record:
            continue
        value = normalize_text(record.get(key))
        if value:
            return value
    return default


def load_records(path: str | Path) -> List[Dict[str, Any]]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(file_path)
    if suffix == ".json":
        payload = read_json(file_path)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("records", "rows", "data", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
            return [payload]
    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported input format: {file_path}")


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_list_field(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_text(item) for item in value if normalize_text(item)]
    text = normalize_text(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [normalize_text(item) for item in parsed if normalize_text(item)]
        except json.JSONDecodeError:
            pass
    if "||" in text:
        return [part.strip() for part in text.split("||") if part.strip()]
    if "\n" in text:
        return [part.strip() for part in text.splitlines() if part.strip()]
    return [text]


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).strip()


def simple_tokens(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_space(text).lower())


def jaccard_similarity(left: Any, right: Any) -> float:
    left_tokens = set(simple_tokens(left))
    right_tokens = set(simple_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def text_match_score(left: Any, right: Any) -> float:
    left_norm = normalize_answer(left)
    right_norm = normalize_answer(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        return 0.95
    return jaccard_similarity(left_norm, right_norm)


def match_any_text(candidates: Sequence[str], targets: Sequence[str], min_score: float = 0.8) -> bool:
    for candidate in candidates:
        for target in targets:
            if text_match_score(candidate, target) >= min_score:
                return True
    return False


def guess_row_id(record: Mapping[str, Any], fallback_index: int | None = None) -> str:
    value = first_available(record, ID_KEYS)
    if value:
        return value
    return str(fallback_index) if fallback_index is not None else ""


def guess_prompt(record: Mapping[str, Any], override: str | None = None) -> str:
    keys = (override,) if override else PROMPT_KEYS
    return first_available(record, [key for key in keys if key])


def split_response(text: str) -> Dict[str, str]:
    clean = normalize_text(text)
    if not clean:
        return {"reasoning": "", "final_answer": ""}
    marker = re.search(r"\n\s*Final Answer\s*:\s*", clean, flags=re.IGNORECASE)
    if not marker:
        return {"reasoning": clean, "final_answer": ""}
    reasoning = clean[: marker.start()].strip()
    final_answer = clean[marker.end() :].strip()
    return {"reasoning": reasoning, "final_answer": final_answer}


def guess_reasoning(record: Mapping[str, Any], override: str | None = None) -> str:
    keys = [override] if override else []
    keys.extend(REASONING_KEYS)
    reasoning = first_available(record, [key for key in keys if key])
    if reasoning:
        return reasoning
    response = first_available(record, ("response", "output", "completion"))
    return split_response(response).get("reasoning", "")


def guess_answer(record: Mapping[str, Any], override: str | None = None) -> str:
    keys = [override] if override else []
    keys.extend(ANSWER_KEYS)
    answer = first_available(record, [key for key in keys if key])
    if answer:
        return answer
    response = first_available(record, ("response", "output", "completion"))
    return split_response(response).get("final_answer", "")


def guess_gold_answer(record: Mapping[str, Any], override: str | None = None) -> str:
    keys = [override] if override else []
    keys.extend(GOLD_ANSWER_KEYS)
    return first_available(record, [key for key in keys if key])


def canonical_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    id_field: str | None = None,
    prompt_field: str | None = None,
    reasoning_field: str | None = None,
    answer_field: str | None = None,
    gold_answer_field: str | None = None,
) -> List[Dict[str, Any]]:
    canonical: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        canonical.append(
            {
                "id": normalize_text(row.get(id_field)) if id_field else guess_row_id(row, index),
                "model_id": model_id,
                "prompt": guess_prompt(row, prompt_field),
                "reasoning": guess_reasoning(row, reasoning_field),
                "final_answer": guess_answer(row, answer_field),
                "gold_answer": guess_gold_answer(row, gold_answer_field),
                "raw": dict(row),
            }
        )
    return canonical


def summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    if len(values) == 1:
        return {"mean": round(values[0], 6), "std": 0.0, "count": 1}
    return {
        "mean": round(mean(values), 6),
        "std": round(pstdev(values), 6),
        "count": len(values),
    }


def coerce_bool_label(value: Any) -> bool | None:
    text = normalize_text(value).lower()
    if text in {"", "na", "n/a", "none"}:
        return None
    if text in {"entail", "entailed", "supported", "support", "true", "yes", "1"}:
        return True
    if text in {"contradict", "contradiction", "neutral", "false", "no", "0"}:
        return False
    return to_bool(value)


__all__ = [
    "ANSWER_KEYS",
    "GOLD_ANSWER_KEYS",
    "ID_KEYS",
    "PROMPT_KEYS",
    "REASONING_KEYS",
    "canonical_prediction_rows",
    "coerce_bool_label",
    "first_available",
    "guess_answer",
    "guess_gold_answer",
    "guess_prompt",
    "guess_reasoning",
    "guess_row_id",
    "jaccard_similarity",
    "load_records",
    "match_any_text",
    "normalize_answer",
    "normalize_space",
    "parse_list_field",
    "safe_float",
    "simple_tokens",
    "summarize_numeric",
    "text_match_score",
    "to_bool",
    "write_csv",
    "write_json",
    "write_jsonl",
]
