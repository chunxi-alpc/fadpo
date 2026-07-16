#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full
import re
import traceback
from typing import Any, Dict, Iterable, List, Tuple

import nltk
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from common import (
    append_jsonl,
    artifact_path,
    ensure_parent,
    iso_utc_now,
    normalize_text,
    read_json,
    result_path,
    write_run_metadata,
)
from common import load_jsonl as load_rows

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover - dependency availability is environment-specific
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


def ensure_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="Compute ARU-level faithfulness risk and sample-level Fa-DPO margins."
    )
    parser.add_argument(
        "--input-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.strict_clean.arus.with_evidence.jsonl"),
        help="ARU JSONL with attached evidence from stage 05.",
    )
    parser.add_argument(
        "--source-aru-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.strict_clean.arus.jsonl"),
        help="Optional stage-04 ARU JSONL used for preflight consistency checks.",
    )
    parser.add_argument(
        "--stage05-metadata-file",
        default=artifact_path("fa_dpo_pipeline", "run_metadata", "05_attach_aru_evidence.json"),
        help="Optional stage-05 metadata JSON used for preflight comparison.",
    )
    parser.add_argument(
        "--output-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.strict_clean.scores.jsonl"),
        help="Faithfulness score JSONL output path.",
    )
    parser.add_argument(
        "--preflight-report-file",
        default=artifact_path("fa_dpo_pipeline", "reports", "06_score_faithfulness_preflight.md"),
        help="Markdown report written before scoring.",
    )
    parser.add_argument(
        "--preflight-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only run the stage-05 -> stage-06 readiness check and exit.",
    )
    parser.add_argument(
        "--skip-preflight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip evidence-attachment readiness checks before scoring.",
    )
    parser.add_argument(
        "--strict-preflight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Abort if the preflight check reports warnings.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from an existing output JSONL by skipping already-scored row ids.",
    )
    parser.add_argument(
        "--max-missing-wd-rate",
        type=float,
        default=0.2,
        help="Warn when more than this fraction of W/D nodes have no retained evidence.",
    )
    parser.add_argument(
        "--max-all-wd-missing-row-rate",
        type=float,
        default=0.1,
        help="Warn when too many rows with W/D nodes end up with zero retained W/D evidence.",
    )
    parser.add_argument(
        "--min-avg-wd-evidence",
        type=float,
        default=1.0,
        help="Warn when the average retained evidence per W/D node falls below this value.",
    )
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        help="Zero-shot NLI checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="NLI batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="NLI max length.",
    )
    parser.add_argument(
        "--lambda-hal",
        type=float,
        default=0.2,
        help="Observation fallback hallucination penalty.",
    )
    parser.add_argument(
        "--lambda-miss",
        type=float,
        default=0.2,
        help="Missing-record penalty for observation nodes when the case context is neutral.",
    )
    parser.add_argument(
        "--lambda-sup",
        type=float,
        default=0.8,
        help="Unsupported-knowledge penalty.",
    )
    parser.add_argument(
        "--lambda-diff",
        type=float,
        default=0.8,
        help="Unsupported-differentiation penalty for D nodes scored against patient context plus evidence.",
    )
    parser.add_argument(
        "--lambda-log",
        type=float,
        default=0.5,
        help="Logical-neutrality penalty.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.6,
        help="Weight on max-risk in the v2 final score.",
    )
    parser.add_argument(
        "--alpha-cov",
        type=float,
        default=0.4,
        help="Coverage-penalty weight from 6_step4_v2.py.",
    )
    parser.add_argument(
        "--avg-defect-weight",
        type=float,
        default=0.1,
        help="Average-defect weight from 6_step4_v2.py.",
    )
    parser.add_argument(
        "--margin-scale",
        type=float,
        default=3.0,
        help="Scale from sample risk to margin.",
    )
    parser.add_argument(
        "--max-margin",
        type=float,
        default=2.5,
        help="Cap for the sample-level margin.",
    )
    parser.add_argument(
        "--entailment-threshold",
        type=float,
        default=0.5,
        help="Threshold for treating a pair as supported.",
    )
    parser.add_argument(
        "--contradiction-threshold",
        type=float,
        default=0.7,
        help="Threshold for treating a pair as contradictory.",
    )
    parser.add_argument(
        "--default-target-coverage",
        type=float,
        default=0.5,
        help="Coverage target when no prevalence excuse is used.",
    )
    parser.add_argument(
        "--prevalence-target-coverage",
        type=float,
        default=0.8,
        help="Coverage target when prevalence-style reasoning is used.",
    )
    parser.add_argument(
        "--doc-saved-penalty",
        type=float,
        default=0.1,
        help="Deprecated compatibility flag; observation fallback no longer removes the missing-record penalty.",
    )
    parser.add_argument(
        "--device",
        default=default_device,
        help="Torch device. When `--gpus` is empty, use `cuda` to leverage visible GPUs via DataParallel.",
    )
    parser.add_argument(
        "--gpus",
        default="",
        help=(
            "Comma-separated visible GPU ids for one-worker-per-GPU scoring, "
            "for example `0,1,2,3` or `all`. This is usually faster than `--device cuda` "
            "for this stage because NLI calls are small and frequent."
        ),
    )
    parser.add_argument(
        "--metadata-file",
        default=artifact_path("fa_dpo_pipeline", "run_metadata", "06_score_faithfulness.json"),
        help="Where to write run metadata JSON.",
    )
    return parser.parse_args()


def node_claim_text(node: Dict[str, Any]) -> str:
    return normalize_text(
        node.get("nli_claim_text")
        or node.get("claim_text")
        or node.get("atomic_proposition")
        or node.get("text")
    )


def split_sentences(text: str) -> List[str]:
    clean = normalize_text(text).replace("\n", " ")
    sentences = [sentence.strip() for sentence in nltk.sent_tokenize(clean) if len(sentence.strip()) > 5]
    return sentences or ([clean] if clean else [])


PREVALENCE_EXCUSE_KEYWORDS = (
    "rare",
    "unlikely",
    "uncommon",
    "seldom",
    "incidence",
    "prevalence",
    "epidemiology",
    "罕见",
    "少见",
    "发生率",
    "流行病学",
    "可能性低",
    "极少",
)

LOGIC_NEUTRAL_DISCOUNT = 0.2
LOGIC_HISTORY_ENTRY_WINDOW = 8
LOGIC_HISTORY_WINDOW = 6
LOGIC_HISTORY_MAX_OBSERVATIONS = 4
LOGIC_HISTORY_MAX_THEORIES = 2
LOGIC_HISTORY_MAX_THEORY_SCORE = 0.5
LOGIC_HISTORY_MAX_PRIOR_CONCLUSIONS = 1
LOGIC_HISTORY_MAX_CONCLUSION_SCORE = 0.35
LOGIC_HISTORY_RECENT_CONCLUSION_WINDOW = 2
LOGIC_HISTORY_FALLBACK_ITEMS = 2
SOFT_WARRANT_NEUTRAL_WEIGHT = 0.35
PROXY_DIFF_CONFLICT_CAP = 0.55
LOGIC_CARRY_FORWARD_MARKERS = (
    "supported",
    "supporting",
    "favored",
    "favoured",
    "less likely",
    "unlikely",
    "arguing against",
    "against",
    "not definitively",
    "not diagnostic",
    "not consistent",
    "consistent with",
    "inconsistent with",
)
LOGIC_MANAGEMENT_MARKERS = (
    "considered",
    "suspected",
    "possible",
    "favored",
    "favoured",
    "provisional clinical diagnosis",
    "was given",
    "ruled out",
    "excluded",
    "less likely",
    "unlikely",
    "diagnosis was",
)
LOGIC_EXCLUSION_MARKERS = (
    "ruled out",
    "excluded",
    "less likely",
    "unlikely",
    "arguing against",
    "not consistent",
    "negative for",
)
LOGIC_EXCLUSION_SUPPORT_MARKERS = (
    "negative",
    "no evidence",
    "without",
    "normal",
    "nondiagnostic",
    "did not",
    "absence of",
    "lack of",
    "ruled out",
    "excluded",
    "not detected",
)
EXCLUSION_NUMERIC_SUPPORT_MARKERS = (
    "activity",
    "titer",
    "titre",
    "count",
    "level",
    "ratio",
    "adamts13",
    "coombs",
    "rpr",
    "western blot",
    "igg",
    "igm",
)
FRAGMENT_PLACEHOLDER_CLAIMS = {
    "a differential diagnosis",
    "differential diagnosis",
    "differential diagnoses",
    "a radiological differential diagnosis",
    "radiological differential diagnosis",
    "in the differential",
    "the findings",
    "the immunohistochemical profile",
    "the diagnosis was excluded",
}
SHORT_PREPOSITION_PREFIXES = {
    "in",
    "of",
    "because",
    "by",
    "after",
    "given",
    "since",
    "as",
}
CLAIM_COMPLETION_CUES = (
    "considered",
    "suspected",
    "excluded",
    "ruled out",
    "supported",
    "showed",
    "revealed",
    "demonstrated",
    "confirmed",
    "unlikely",
    "less likely",
)
LOW_INFORMATION_HISTORY_TOKENS = {
    "ct",
    "cta",
    "mri",
    "emg",
    "ncs",
    "ecg",
    "echo",
    "angiogram",
    "scan",
    "imaging",
    "biopsy",
    "pathology",
    "ultrasound",
    "xray",
    "pet",
}
CATALOGUE_STYLE_MARKERS = (
    "can mimic",
    "may mimic",
    "other uncommon",
    "other differentials",
    "other differential",
    "differential diagnosis includes",
    "differential diagnoses include",
)

OBSERVATION_ANCHOR_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "being",
    "by",
    "can",
    "clinical",
    "common",
    "compatible",
    "consistent",
    "considered",
    "demonstrated",
    "disease",
    "evidence",
    "exam",
    "examination",
    "feature",
    "features",
    "finding",
    "findings",
    "for",
    "found",
    "from",
    "had",
    "has",
    "have",
    "history",
    "immediately",
    "imaging",
    "in",
    "is",
    "it",
    "its",
    "laboratory",
    "lesion",
    "likely",
    "mass",
    "normal",
    "noted",
    "of",
    "on",
    "or",
    "pathologic",
    "patient",
    "physical",
    "possible",
    "post",
    "prior",
    "radiologic",
    "revealed",
    "scan",
    "seen",
    "showed",
    "sign",
    "signs",
    "study",
    "suggestive",
    "suspected",
    "that",
    "the",
    "there",
    "this",
    "to",
    "was",
    "were",
    "with",
    "without",
    "after",
    "before",
    "during",
}
OBSERVATION_ANCHOR_PATTERN = re.compile(r"[a-z0-9]+")
OBSERVATION_STEMMER = nltk.stem.PorterStemmer()

PREVIEW_LIMIT = 10


def normalize_device(device: Any) -> str:
    text = str(device).strip().lower()
    if not text or text in {"-1", "cpu"}:
        return "cpu"
    if torch is None or not torch.cuda.is_available():
        return "cpu"
    if text == "cuda":
        return "cuda"
    if text.isdigit():
        return f"cuda:{text}"
    if text.startswith("cuda:"):
        return text
    return text


def parse_gpu_list(value: Any) -> List[str]:
    text = normalize_text(value).lower().replace(" ", "")
    if not text:
        return []
    if text in {"all", "auto"}:
        if torch is None or not torch.cuda.is_available():
            return []
        return [str(index) for index in range(torch.cuda.device_count())]

    parsed: List[str] = []
    seen = set()
    for token in text.split(","):
        clean = token.strip().lower()
        if not clean:
            continue
        if clean.startswith("cuda:"):
            clean = clean.split(":", 1)[1]
        if not clean.isdigit():
            raise ValueError(
                f"Unsupported GPU token `{token}` in `--gpus`. Use values like `0,1,2,3` or `all`."
            )
        if clean in seen:
            continue
        seen.add(clean)
        parsed.append(clean)
    return parsed


def validate_gpu_list(gpu_list: List[str]) -> None:
    if not gpu_list:
        return
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("`--gpus` was provided, but CUDA is not available.")

    visible_gpu_count = torch.cuda.device_count()
    invalid = [gpu_id for gpu_id in gpu_list if int(gpu_id) < 0 or int(gpu_id) >= visible_gpu_count]
    if invalid:
        raise ValueError(
            f"`--gpus` contains ids outside the visible CUDA range [0, {visible_gpu_count - 1}]: {invalid}"
        )


def canonical_label(label: Any) -> str:
    text = normalize_text(label).lower().replace("-", "_").replace(" ", "_")
    if "entail" in text:
        return "entailment"
    if "contra" in text:
        return "contradiction"
    if "neutral" in text:
        return "neutral"
    return ""


def build_label_lookup(model: AutoModelForSequenceClassification) -> Dict[str, int]:
    config = model.module.config if isinstance(model, torch.nn.DataParallel) else model.config
    raw_mapping = getattr(config, "id2label", {}) or {}
    label_lookup: Dict[str, int] = {}
    for key, value in raw_mapping.items():
        canonical = canonical_label(value)
        if canonical:
            label_lookup[canonical] = int(key)
    if len(label_lookup) == 3:
        return label_lookup

    if len(raw_mapping) == 3:
        fallback = {0: "contradiction", 1: "neutral", 2: "entailment"}
        return {label: index for index, label in fallback.items()}
    raise ValueError(f"Unsupported NLI label mapping: {raw_mapping}")


def load_nli(args: argparse.Namespace) -> Dict[str, Any]:
    device = normalize_device(args.device)
    model_kwargs: Dict[str, Any] = {}
    if device.startswith("cuda"):
        if device != "cuda":
            torch.cuda.set_device(device)
        model_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.nli_model, **model_kwargs)
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "label_lookup": build_label_lookup(model),
    }


def run_nli_pairs(
    nli_state: Dict[str, Any],
    pairs: Iterable[Tuple[str, str]],
    batch_size: int,
    max_length: int,
) -> List[Dict[str, Any]]:
    clean_pairs: List[Tuple[str, str]] = []
    for premise, hypothesis in pairs:
        clean_premise = normalize_text(premise)
        clean_hypothesis = normalize_text(hypothesis)
        if not clean_premise or not clean_hypothesis:
            continue
        clean_pairs.append((clean_premise, clean_hypothesis))
    if not clean_pairs:
        return []

    tokenizer = nli_state["tokenizer"]
    model = nli_state["model"]
    device = nli_state["device"]
    label_lookup = nli_state["label_lookup"]
    parsed: List[Dict[str, Any]] = []
    for start in range(0, len(clean_pairs), batch_size):
        batch_pairs = clean_pairs[start : start + batch_size]
        premises = [premise for premise, _ in batch_pairs]
        hypotheses = [hypothesis for _, hypothesis in batch_pairs]
        inputs = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu()
        for (premise, _), prob_row in zip(batch_pairs, probs):
            parsed.append(
                {
                    "premise": premise,
                    "entailment": float(prob_row[label_lookup["entailment"]]),
                    "contradiction": float(prob_row[label_lookup["contradiction"]]),
                    "neutral": float(prob_row[label_lookup["neutral"]]),
                }
            )
    return parsed


def run_nli(
    nli_state: Dict[str, Any],
    premises: Iterable[str],
    hypothesis: str,
    batch_size: int,
    max_length: int,
) -> List[Dict[str, Any]]:
    return run_nli_pairs(
        nli_state,
        [(premise, hypothesis) for premise in premises],
        batch_size=batch_size,
        max_length=max_length,
    )


def aggregate_doc_support(doc_probs: List[Dict[str, Any]]) -> Dict[str, float]:
    if not doc_probs:
        return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
    return {
        "entailment": max(prob.get("entailment", 0.0) for prob in doc_probs),
        "contradiction": max(prob.get("contradiction", 0.0) for prob in doc_probs),
        "neutral": max(prob.get("neutral", 0.0) for prob in doc_probs),
    }


def observation_anchor_keys(text: str) -> set[str]:
    anchors: set[str] = set()
    clean = normalize_text(text).lower()
    for token in OBSERVATION_ANCHOR_PATTERN.findall(clean):
        if token in OBSERVATION_ANCHOR_STOPWORDS:
            continue
        has_digit = any(char.isdigit() for char in token)
        if len(token) < 3 and not has_digit:
            continue
        anchor = token if has_digit else OBSERVATION_STEMMER.stem(token)
        if len(anchor) < 3 and not has_digit:
            continue
        anchors.add(anchor)
    return anchors


def observation_conflict_is_anchored(hypothesis: str, premise: str) -> bool:
    overlap = observation_anchor_keys(hypothesis) & observation_anchor_keys(premise)
    if len(overlap) >= 2:
        return True
    if any(any(char.isdigit() for char in token) for token in overlap):
        return True
    return False


def select_observation_context_entry(
    entries: List[Dict[str, Any]],
    hypothesis: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], int, str]:
    if not entries:
        return empty_probs(), -1, "none"

    best_entail_index, best_entail = max(
        enumerate(entries),
        key=lambda item: item[1].get("entailment", 0.0),
    )
    best_contradiction_index, best_contradiction = max(
        enumerate(entries),
        key=lambda item: item[1].get("contradiction", 0.0),
    )

    if best_entail.get("entailment", 0.0) >= args.entailment_threshold:
        return dict(best_entail), best_entail_index, "entail"

    contradiction_margin = 0.2
    neutral_margin = 0.1
    if (
        observation_conflict_is_anchored(hypothesis, best_contradiction.get("premise", ""))
        and best_contradiction.get("contradiction", 0.0) >= args.contradiction_threshold
        and best_contradiction.get("contradiction", 0.0)
        >= best_entail.get("entailment", 0.0) + contradiction_margin
        and best_contradiction.get("contradiction", 0.0)
        >= best_contradiction.get("neutral", 0.0) + neutral_margin
    ):
        return dict(best_contradiction), best_contradiction_index, "contradict"

    return dict(best_entail), best_entail_index, "neutral"


def trace_entry(index: int, node_type: str, score: float, tag: str) -> Dict[str, object]:
    return {"id": index, "t": node_type, "s": round(float(score), 4), "p": tag}


def empty_probs() -> Dict[str, Any]:
    return {"premise": "", "entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}


def concat_premise(parts: Iterable[Any]) -> str:
    seen = set()
    ordered: List[str] = []
    for part in parts:
        text = normalize_text(part)
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return "\n\n".join(ordered)


def preview_text(text: Any, limit: int = 400) -> str:
    clean = normalize_text(text)
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def word_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", normalize_text(text).lower())


def is_fragmentary_claim(text: str) -> bool:
    clean = normalize_text(text).strip()
    lowered = clean.lower().strip(" .;,:!?")
    if not lowered:
        return True
    if lowered in FRAGMENT_PLACEHOLDER_CLAIMS:
        return True
    if re.search(r"\b(?:was|were|is|are)\.?$", lowered):
        return True
    tokens = word_tokens(lowered)
    if not tokens:
        return True
    if len(tokens) <= 4 and tokens[0] in SHORT_PREPOSITION_PREFIXES:
        if not any(cue in lowered for cue in CLAIM_COMPLETION_CUES):
            return True
    return False


def is_low_information_history_entry(text: str, entry_type: str) -> bool:
    tokens = word_tokens(text)
    if not tokens:
        return True
    if entry_type.upper()[:1] != "O":
        return False
    return len(tokens) <= 2 and all(token in LOW_INFORMATION_HISTORY_TOKENS for token in tokens)


def is_catalogue_style_statement(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(marker in lowered for marker in CATALOGUE_STYLE_MARKERS)


def is_management_claim(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(marker in lowered for marker in LOGIC_MANAGEMENT_MARKERS)


def is_considered_like_claim(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(marker in lowered for marker in ("considered", "suspected", "possible", "favored", "favoured"))


def is_exclusion_style_claim(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(marker in lowered for marker in LOGIC_EXCLUSION_MARKERS)


def has_exclusion_support_anchor(text: str) -> bool:
    lowered = normalize_text(text).lower()
    if any(marker in lowered for marker in LOGIC_EXCLUSION_SUPPORT_MARKERS):
        return True
    return bool(
        re.search(r"(?:<|>|<=|>=|\d+%|\d+\.\d+|\d+/\d+)", lowered)
        and any(marker in lowered for marker in EXCLUSION_NUMERIC_SUPPORT_MARKERS)
    )


def diagnostic_conclusion_head(text: str) -> str:
    clean = normalize_text(text).strip()
    if not clean:
        return ""
    patterns = (
        r"^(?:an?\s+|the\s+)?(?P<head>.+?)\s+was\s+(?:considered|suspected|favou?red|ruled out|excluded|deemed unlikely|less likely|unlikely)\b",
        r"^(?:an?\s+)?provisional clinical diagnosis of\s+['\"]?(?P<head>.+?)['\"]?\s+was given\b",
        r"^(?:an?\s+|the\s+)?(?P<head>.+?)\s+(?:remains|was)\s+the most likely diagnosis\b",
    )
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match:
            head = normalize_text(match.group("head")).lower().strip(" '\".,;:")
            head = re.sub(r"^(?:an?|the)\s+", "", head)
            return head
    return ""


def same_management_conclusion(current_text: str, prior_text: str) -> bool:
    current_head = diagnostic_conclusion_head(current_text)
    prior_head = diagnostic_conclusion_head(prior_text)
    return bool(current_head and prior_head and current_head == prior_head)


def lexical_anchor_overlap(a: str, b: str) -> int:
    return len(observation_anchor_keys(a) & observation_anchor_keys(b))


def select_context_premise_for_claim(
    context_sentences: List[str],
    claim_text: str,
    *,
    max_sentences: int = 2,
) -> str:
    scored: List[Tuple[int, int, str]] = []
    for idx, sentence in enumerate(context_sentences):
        overlap = lexical_anchor_overlap(claim_text, sentence)
        if overlap <= 0:
            continue
        scored.append((overlap, idx, sentence))
    if not scored:
        return ""
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = sorted(scored[:max_sentences], key=lambda item: item[1])
    return concat_premise(sentence for _, _, sentence in selected)


def score_warrant_probs(probs: Dict[str, float], node_text: str, args: argparse.Namespace) -> float:
    neutral_weight = args.lambda_sup
    if neutral_dominant(probs) and is_catalogue_style_statement(node_text):
        neutral_weight = min(neutral_weight, SOFT_WARRANT_NEUTRAL_WEIGHT)
    return score_from_probs(probs, neutral_weight)


def is_proxy_differential_claim(text: str) -> bool:
    lowered = normalize_text(text).lower()
    symptom_absence = (
        ("absence of" in lowered or lowered.startswith("no ") or "without " in lowered)
        and "unlikely" in lowered
    )
    treatment_response = any(
        marker in lowered
        for marker in ("did not elicit", "no response", "failed to respond", "not responsive")
    )
    return symptom_absence or treatment_response


def single_premise_entry(
    nli_state: Dict[str, Any],
    premise: str,
    hypothesis: str,
    *,
    batch_size: int,
    max_length: int,
) -> Dict[str, Any]:
    entries = run_nli(
        nli_state,
        [premise],
        hypothesis,
        batch_size=batch_size,
        max_length=max_length,
    )
    return entries[0] if entries else empty_probs()


def neutral_dominant(probs: Dict[str, float]) -> bool:
    neutral = probs.get("neutral", 0.0)
    return neutral >= max(probs.get("entailment", 0.0), probs.get("contradiction", 0.0))


def score_from_probs(probs: Dict[str, float], neutral_weight: float) -> float:
    return max(
        probs.get("contradiction", 0.0),
        neutral_weight * probs.get("neutral", 0.0),
    )


def logic_score_from_probs(probs: Dict[str, float], args: argparse.Namespace) -> float:
    contradiction = probs.get("contradiction", 0.0)
    # Neutral often just means the history is incomplete, not that the conclusion is
    # strongly wrong. Discount it unless contradiction evidence is actually present.
    uncertainty = max(0.0, probs.get("neutral", 0.0) - probs.get("entailment", 0.0))
    return max(contradiction, args.lambda_log * LOGIC_NEUTRAL_DISCOUNT * uncertainty)


def carries_forward_logic_conclusion(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(marker in lowered for marker in LOGIC_CARRY_FORWARD_MARKERS)


def logic_history_fallback_items(
    history_entries: List[Dict[str, Any]],
    current_text: str = "",
) -> List[Tuple[int, str]]:
    fallback_items: List[Tuple[int, str]] = []
    current_is_management = is_management_claim(current_text)
    current_is_exclusion = is_exclusion_style_claim(current_text)
    start_index = max(0, len(history_entries) - LOGIC_HISTORY_ENTRY_WINDOW)
    for original_index in range(len(history_entries) - 1, start_index - 1, -1):
        entry = history_entries[original_index]
        entry_type = str(entry.get("type", "")).upper()[:1]
        text = normalize_text(entry.get("text"))
        decision = normalize_text(entry.get("decision_label")).lower()
        score = float(entry.get("score", 0.0) or 0.0)
        if not text or decision == "contradict":
            continue
        if is_fragmentary_claim(text) or is_low_information_history_entry(text, entry_type):
            continue
        if entry_type in {"W", "D"} and score > LOGIC_HISTORY_MAX_THEORY_SCORE:
            continue
        if entry_type == "C" and score > LOGIC_HISTORY_MAX_CONCLUSION_SCORE:
            continue
        if entry_type not in {"O", "W", "D", "C"}:
            continue
        if entry_type == "C" and same_management_conclusion(current_text, text):
            continue
        if current_is_management and entry_type == "C" and is_management_claim(text):
            continue
        if current_is_management and entry_type == "D" and is_considered_like_claim(text):
            continue
        if current_is_management and entry_type == "W" and is_catalogue_style_statement(text):
            continue
        fallback_items.append((original_index, f"[{entry_type}] {text}"))
        if len(fallback_items) >= LOGIC_HISTORY_FALLBACK_ITEMS:
            break
    if current_is_exclusion and not any(has_exclusion_support_anchor(text) for _, text in fallback_items):
        return []
    fallback_items.sort(key=lambda item: item[0])
    return fallback_items


def logic_history_premise(history_entries: List[Dict[str, Any]], current_text: str = "") -> str:
    selected_items: List[Tuple[int, str]] = []
    recent_observations = 0
    recent_theories = 0
    recent_conclusions = 0
    current_is_management = is_management_claim(current_text)
    current_is_exclusion = is_exclusion_style_claim(current_text)
    start_index = max(0, len(history_entries) - LOGIC_HISTORY_ENTRY_WINDOW)
    recent_conclusion_floor = max(0, len(history_entries) - LOGIC_HISTORY_RECENT_CONCLUSION_WINDOW)
    for original_index in range(len(history_entries) - 1, start_index - 1, -1):
        entry = history_entries[original_index]
        entry_type = str(entry.get("type", "")).upper()[:1]
        text = normalize_text(entry.get("text"))
        decision = normalize_text(entry.get("decision_label")).lower()
        score = float(entry.get("score", 0.0) or 0.0)
        if not text:
            continue
        if decision == "contradict":
            continue
        if is_fragmentary_claim(text) or is_low_information_history_entry(text, entry_type):
            continue
        if entry_type == "O":
            if recent_observations >= LOGIC_HISTORY_MAX_OBSERVATIONS:
                continue
            selected_items.append((original_index, f"[O] {text}"))
            recent_observations += 1
        elif entry_type in {"W", "D"}:
            if score > LOGIC_HISTORY_MAX_THEORY_SCORE:
                continue
            if current_is_management and entry_type == "D" and is_considered_like_claim(text):
                continue
            if current_is_management and entry_type == "W" and is_catalogue_style_statement(text):
                continue
            if recent_theories >= LOGIC_HISTORY_MAX_THEORIES:
                continue
            selected_items.append((original_index, f"[{entry_type}] {text}"))
            recent_theories += 1
        elif entry_type == "C":
            if score > LOGIC_HISTORY_MAX_CONCLUSION_SCORE:
                continue
            if recent_conclusions >= LOGIC_HISTORY_MAX_PRIOR_CONCLUSIONS:
                continue
            if same_management_conclusion(current_text, text):
                continue
            if current_is_management and is_management_claim(text):
                continue
            is_recent_bridge = original_index >= recent_conclusion_floor
            if not is_recent_bridge and not carries_forward_logic_conclusion(text):
                continue
            selected_items.append((original_index, f"[C] {text}"))
            recent_conclusions += 1
        if len(selected_items) >= LOGIC_HISTORY_WINDOW:
            break
    if not selected_items:
        # Keep the recent-history design, but avoid collapsing C to LogicBase
        # when the immediate local chain is short or phrased without markers.
        selected_items = logic_history_fallback_items(history_entries, current_text=current_text)
    if current_is_exclusion and not any(has_exclusion_support_anchor(text) for _, text in selected_items):
        return ""
    selected_items.sort(key=lambda item: item[0])
    return "\n".join(text for _, text in selected_items)


def dominant_probability_key(probs: Dict[str, float]) -> str:
    return max(("entailment", "contradiction", "neutral"), key=lambda key: probs.get(key, 0.0))


def representative_entry(entries: List[Dict[str, Any]], aggregate: Dict[str, float] | None = None) -> Dict[str, Any]:
    if not entries:
        return empty_probs()
    aggregate_probs = aggregate or aggregate_doc_support(entries)
    key = dominant_probability_key(aggregate_probs)
    return max(entries, key=lambda item: item.get(key, 0.0))


def select_retrieval_entry(
    entries: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, float], str]:
    if not entries:
        return empty_probs(), empty_probs(), "none"

    aggregate = aggregate_doc_support(entries)
    best_entail = max(entries, key=lambda item: item.get("entailment", 0.0))
    best_contradiction = max(entries, key=lambda item: item.get("contradiction", 0.0))

    contradiction_margin = 0.2
    if (
        best_contradiction.get("contradiction", 0.0) >= args.contradiction_threshold
        and best_contradiction.get("contradiction", 0.0)
        >= best_entail.get("entailment", 0.0) + contradiction_margin
    ):
        return dict(best_contradiction), aggregate, "contradict"

    return dict(best_entail), aggregate, "entail"


def decision_label(probs: Dict[str, float], args: argparse.Namespace) -> str:
    if probs.get("contradiction", 0.0) >= args.contradiction_threshold:
        return "contradict"
    if probs.get("entailment", 0.0) >= args.entailment_threshold:
        return "entail"
    return "neutral"


def argmax_label(probs: Dict[str, float]) -> str:
    mapping = {
        "entailment": "entail",
        "contradiction": "contradict",
        "neutral": "neutral",
    }
    best_key = max(mapping, key=lambda key: probs.get(key, 0.0))
    return mapping[best_key]


def has_prevalence_excuse(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(keyword in lowered for keyword in PREVALENCE_EXCUSE_KEYWORDS)


def maybe_load_rows(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    return load_rows(str(file_path))


def maybe_load_json(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    payload = read_json(str(file_path))
    return payload if isinstance(payload, dict) else {}


def completed_ids(path: str) -> Tuple[set[str], int]:
    file_path = Path(path)
    if not file_path.exists():
        return set(), 0

    seen: set[str] = set()
    malformed_lines = 0
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            clean = line.strip()
            if not clean:
                continue
            try:
                payload = json.loads(clean)
            except json.JSONDecodeError:
                malformed_lines += 1
                print(
                    "Warning: ignoring malformed JSONL line while resuming "
                    f"{file_path} (line {line_number})."
                )
                continue
            row_id = normalize_text(payload.get("id")) if isinstance(payload, dict) else ""
            if row_id:
                seen.add(row_id)
    return seen, malformed_lines


def evidence_texts(value: Any) -> List[str]:
    if isinstance(value, list):
        texts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                text = normalize_text(item.get("text"))
            else:
                text = normalize_text(item)
            if text:
                texts.append(text)
        return texts
    if isinstance(value, dict):
        text = normalize_text(value.get("text"))
        return [text] if text else []
    text = normalize_text(value)
    return [text] if text else []


def row_fallback_evidence(value: Any) -> List[str]:
    return evidence_texts(value)


def preview_list(items: List[Any], limit: int = PREVIEW_LIMIT) -> List[Any]:
    return items[:limit]


def require_existing_file(path: str, *, arg_name: str, purpose: str) -> None:
    file_path = Path(path)
    if file_path.exists():
        return
    raise FileNotFoundError(
        f"Missing {purpose}: {file_path}\n"
        f"Pass it explicitly with `{arg_name} /path/to/file`, or set the runtime root env vars.\n"
        "Default paths are resolved from the current working directory via:\n"
        "- `FA_DPO_RESULT_ROOT` for result files\n"
        "- `FA_DPO_ARTIFACT_ROOT` for artifact files"
    )


def preflight_summary(
    rows: List[Dict[str, Any]],
    source_rows: List[Dict[str, Any]],
    stage05_metadata: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    blocking: List[str] = []
    warnings: List[str] = []

    source_by_id = {str(row.get("id")): row for row in source_rows if row.get("id") is not None}
    row_by_id = {str(row.get("id")): row for row in rows if row.get("id") is not None}

    missing_from_input = sorted([row_id for row_id in source_by_id if row_id not in row_by_id])
    extra_in_input = sorted([row_id for row_id in row_by_id if row_id not in source_by_id])
    aru_count_mismatch_ids: List[str] = []
    aru_content_mismatch_ids: List[str] = []

    if source_rows:
        for row_id, source_row in source_by_id.items():
            scored_row = row_by_id.get(row_id)
            if scored_row is None:
                continue
            source_arus = source_row.get("arus") or []
            scored_arus = scored_row.get("arus") or []
            if len(source_arus) != len(scored_arus):
                aru_count_mismatch_ids.append(row_id)
                continue
            for source_node, scored_node in zip(source_arus, scored_arus):
                source_type = str(source_node.get("type", "")).upper()[:1]
                scored_type = str(scored_node.get("type", "")).upper()[:1]
                source_text = normalize_text(source_node.get("text"))
                scored_text = normalize_text(scored_node.get("text"))
                if source_type != scored_type or source_text != scored_text:
                    aru_content_mismatch_ids.append(row_id)
                    break

    total_rows = len(rows)
    rows_without_arus: List[str] = []
    rows_missing_patient_context: List[str] = []
    rows_with_wd_nodes = 0
    rows_with_all_wd_missing: List[str] = []
    rows_with_any_wd_missing: List[str] = []
    wd_missing_examples: List[Dict[str, Any]] = []
    invalid_wd_field_examples: List[Dict[str, Any]] = []

    total_nodes = 0
    total_o_nodes = 0
    total_wd_nodes = 0
    total_c_nodes = 0
    wd_nodes_with_evidence = 0
    wd_nodes_missing_evidence = 0
    wd_nodes_missing_field = 0
    retrieval_candidate_field_missing = 0
    total_retained_wd_evidence = 0
    total_candidates = 0

    for row in rows:
        row_id = str(row.get("id"))
        arus = row.get("arus")
        if not isinstance(arus, list) or not arus:
            rows_without_arus.append(row_id)
            arus = arus if isinstance(arus, list) else []
        patient_context = normalize_text(row.get("patient_context"))
        has_o_node = any(str(node.get("type", "")).upper()[:1] == "O" for node in arus)
        if has_o_node and not patient_context:
            rows_missing_patient_context.append(row_id)

        row_wd_total = 0
        row_wd_missing = 0

        for index, node in enumerate(arus):
            node_type = str(node.get("type", "")).upper()[:1]
            total_nodes += 1
            if node_type == "O":
                total_o_nodes += 1
            elif node_type in {"W", "D"}:
                total_wd_nodes += 1
                row_wd_total += 1

                if "retrieved_evidence" not in node:
                    wd_nodes_missing_field += 1
                    row_wd_missing += 1
                    if len(invalid_wd_field_examples) < PREVIEW_LIMIT:
                        invalid_wd_field_examples.append(
                            {"id": row_id, "aru_id": index, "issue": "missing retrieved_evidence"}
                        )
                    continue
                if "retrieval_candidates" not in node:
                    retrieval_candidate_field_missing += 1
                    if len(invalid_wd_field_examples) < PREVIEW_LIMIT:
                        invalid_wd_field_examples.append(
                            {"id": row_id, "aru_id": index, "issue": "missing retrieval_candidates"}
                        )

                retained_texts = evidence_texts(node.get("retrieved_evidence"))
                candidate_texts = evidence_texts(node.get("retrieval_candidates"))
                total_retained_wd_evidence += len(retained_texts)
                total_candidates += len(candidate_texts)
                if retained_texts:
                    wd_nodes_with_evidence += 1
                else:
                    wd_nodes_missing_evidence += 1
                    row_wd_missing += 1
                    if len(wd_missing_examples) < PREVIEW_LIMIT:
                        wd_missing_examples.append(
                            {
                                "id": row_id,
                                "aru_id": index,
                                "type": node_type,
                                "text": normalize_text(node.get("text"))[:200],
                            }
                        )
            elif node_type == "C":
                total_c_nodes += 1

        if row_wd_total > 0:
            rows_with_wd_nodes += 1
            if row_wd_missing > 0:
                rows_with_any_wd_missing.append(row_id)
            if row_wd_missing == row_wd_total:
                rows_with_all_wd_missing.append(row_id)

    wd_missing_rate = wd_nodes_missing_evidence / max(total_wd_nodes, 1)
    row_all_wd_missing_rate = len(rows_with_all_wd_missing) / max(rows_with_wd_nodes, 1)
    avg_wd_evidence = total_retained_wd_evidence / max(total_wd_nodes, 1)
    avg_candidate_count = total_candidates / max(total_wd_nodes, 1)

    if source_rows:
        if missing_from_input:
            blocking.append(
                f"Stage-05 output is missing {len(missing_from_input)} row(s) that exist in stage-04 ARUs."
            )
        if extra_in_input:
            blocking.append(
                f"Stage-05 output has {len(extra_in_input)} unexpected row(s) not found in stage-04 ARUs."
            )
        if aru_count_mismatch_ids:
            blocking.append(f"{len(aru_count_mismatch_ids)} row(s) changed ARU count between stage 04 and 05.")
        if aru_content_mismatch_ids:
            blocking.append(f"{len(aru_content_mismatch_ids)} row(s) changed ARU text/type between stage 04 and 05.")

    if wd_nodes_missing_field:
        blocking.append(f"{wd_nodes_missing_field} W/D node(s) are missing the `retrieved_evidence` field.")
    if retrieval_candidate_field_missing:
        blocking.append(f"{retrieval_candidate_field_missing} W/D node(s) are missing the `retrieval_candidates` field.")

    if rows_missing_patient_context:
        warnings.append(
            f"{len(rows_missing_patient_context)} row(s) contain O nodes but have empty patient context."
        )
    if wd_missing_rate > args.max_missing_wd_rate:
        warnings.append(
            "Missing W/D evidence rate "
            f"{wd_missing_rate:.2%} exceeds threshold {args.max_missing_wd_rate:.2%}."
        )
    if row_all_wd_missing_rate > args.max_all_wd_missing_row_rate:
        warnings.append(
            "Rows with zero retained W/D evidence "
            f"{row_all_wd_missing_rate:.2%} exceeds threshold {args.max_all_wd_missing_row_rate:.2%}."
        )
    if avg_wd_evidence < args.min_avg_wd_evidence and total_wd_nodes > 0:
        warnings.append(
            f"Average retained evidence per W/D node is {avg_wd_evidence:.2f}, below {args.min_avg_wd_evidence:.2f}."
        )

    metadata_stats = stage05_metadata.get("stats") if isinstance(stage05_metadata.get("stats"), dict) else {}
    metadata_comparison: Dict[str, Any] = {}
    if metadata_stats:
        metadata_comparison = {
            "processed_rows": metadata_stats.get("processed_rows"),
            "wd_node_count": metadata_stats.get("wd_node_count"),
            "avg_retained_evidence": metadata_stats.get("avg_retained_evidence"),
            "avg_reranked_candidates": metadata_stats.get("avg_reranked_candidates"),
        }
        processed_rows = metadata_stats.get("processed_rows")
        if processed_rows not in {None, ""} and int(processed_rows) != total_rows:
            warnings.append(
                f"Stage-05 metadata says {int(processed_rows)} rows, but current input has {total_rows} rows."
            )
        meta_wd_nodes = metadata_stats.get("wd_node_count")
        if meta_wd_nodes not in {None, ""} and int(meta_wd_nodes) != total_wd_nodes:
            warnings.append(
                f"Stage-05 metadata says {int(meta_wd_nodes)} W/D nodes, but current input has {total_wd_nodes}."
            )

    status = "ok"
    if blocking:
        status = "failed"
    elif warnings:
        status = "warning"

    return {
        "status": status,
        "blocking_issues": blocking,
        "warnings": warnings,
        "stats": {
            "total_rows": total_rows,
            "total_nodes": total_nodes,
            "total_o_nodes": total_o_nodes,
            "total_wd_nodes": total_wd_nodes,
            "total_c_nodes": total_c_nodes,
            "rows_with_wd_nodes": rows_with_wd_nodes,
            "rows_without_arus": len(rows_without_arus),
            "rows_missing_patient_context": len(rows_missing_patient_context),
            "wd_nodes_with_evidence": wd_nodes_with_evidence,
            "wd_nodes_missing_evidence": wd_nodes_missing_evidence,
            "wd_nodes_missing_field": wd_nodes_missing_field,
            "retrieval_candidate_field_missing": retrieval_candidate_field_missing,
            "wd_missing_rate": round(float(wd_missing_rate), 6),
            "rows_all_wd_missing": len(rows_with_all_wd_missing),
            "rows_any_wd_missing": len(rows_with_any_wd_missing),
            "row_all_wd_missing_rate": round(float(row_all_wd_missing_rate), 6),
            "avg_wd_evidence": round(float(avg_wd_evidence), 6),
            "avg_candidate_count": round(float(avg_candidate_count), 6),
            "source_row_count": len(source_rows),
            "missing_from_input": len(missing_from_input),
            "extra_in_input": len(extra_in_input),
            "aru_count_mismatches": len(aru_count_mismatch_ids),
            "aru_content_mismatches": len(aru_content_mismatch_ids),
        },
        "examples": {
            "missing_from_input_ids": preview_list(missing_from_input),
            "extra_in_input_ids": preview_list(extra_in_input),
            "aru_count_mismatch_ids": preview_list(aru_count_mismatch_ids),
            "aru_content_mismatch_ids": preview_list(aru_content_mismatch_ids),
            "rows_without_arus": preview_list(rows_without_arus),
            "rows_missing_patient_context": preview_list(rows_missing_patient_context),
            "rows_with_all_wd_missing": preview_list(rows_with_all_wd_missing),
            "rows_with_any_wd_missing": preview_list(rows_with_any_wd_missing),
            "wd_missing_examples": wd_missing_examples,
            "invalid_wd_field_examples": invalid_wd_field_examples,
        },
        "stage05_metadata": metadata_comparison,
    }


def render_preflight_report(summary: Dict[str, Any], args: argparse.Namespace) -> str:
    stats = summary.get("stats", {})
    examples = summary.get("examples", {})
    lines = [
        "# Stage 05 -> 06 Preflight Report",
        "",
        f"Status: **{summary.get('status', 'unknown').upper()}**",
        "",
        "## Inputs",
        "",
        f"- `input_file`: `{args.input_file}`",
        f"- `source_aru_file`: `{args.source_aru_file}`",
        f"- `stage05_metadata_file`: `{args.stage05_metadata_file}`",
        "",
        "## Summary",
        "",
        f"- Rows: {stats.get('total_rows', 0)}",
        f"- Total ARUs: {stats.get('total_nodes', 0)}",
        f"- O nodes: {stats.get('total_o_nodes', 0)}",
        f"- W/D nodes: {stats.get('total_wd_nodes', 0)}",
        f"- C nodes: {stats.get('total_c_nodes', 0)}",
        f"- W/D nodes with evidence: {stats.get('wd_nodes_with_evidence', 0)}",
        f"- W/D nodes missing evidence: {stats.get('wd_nodes_missing_evidence', 0)}",
        f"- Missing W/D evidence rate: {stats.get('wd_missing_rate', 0.0):.2%}",
        f"- Rows with all W/D evidence missing: {stats.get('rows_all_wd_missing', 0)}",
        f"- All-W/D-missing row rate: {stats.get('row_all_wd_missing_rate', 0.0):.2%}",
        f"- Average retained evidence per W/D node: {stats.get('avg_wd_evidence', 0.0):.2f}",
        f"- Average reranked candidates per W/D node: {stats.get('avg_candidate_count', 0.0):.2f}",
    ]

    blocking = summary.get("blocking_issues") or []
    warnings = summary.get("warnings") or []
    lines.extend(["", "## Blocking Issues", ""])
    if blocking:
        lines.extend([f"- {item}" for item in blocking])
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- None")

    metadata_summary = summary.get("stage05_metadata") or {}
    if metadata_summary:
        lines.extend(["", "## Stage 05 Metadata Snapshot", ""])
        for key, value in metadata_summary.items():
            lines.append(f"- {key}: {value}")

    lines.extend(["", "## Examples", ""])
    for key in [
        "missing_from_input_ids",
        "extra_in_input_ids",
        "aru_count_mismatch_ids",
        "aru_content_mismatch_ids",
        "rows_without_arus",
        "rows_missing_patient_context",
        "rows_with_all_wd_missing",
    ]:
        values = examples.get(key) or []
        if values:
            lines.append(f"- {key}: {values}")
    if examples.get("wd_missing_examples"):
        lines.append(f"- wd_missing_examples: {examples['wd_missing_examples']}")
    if examples.get("invalid_wd_field_examples"):
        lines.append(f"- invalid_wd_field_examples: {examples['invalid_wd_field_examples']}")
    return "\n".join(lines) + "\n"


def run_preflight(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Path]:
    source_rows = maybe_load_rows(args.source_aru_file)
    stage05_metadata = maybe_load_json(args.stage05_metadata_file)
    summary = preflight_summary(rows=rows, source_rows=source_rows, stage05_metadata=stage05_metadata, args=args)
    report_text = render_preflight_report(summary, args)
    report_path = ensure_parent(args.preflight_report_file)
    report_path.write_text(report_text, encoding="utf-8")
    return summary, report_path


def score_row(row: Dict, nli_pipe, args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, float]]:
    patient_context = normalize_text(row.get("patient_context"))
    context_sentences = split_sentences(patient_context)
    total_ctx_sentences = len(context_sentences)
    fallback_docs = row_fallback_evidence(row.get("retrieved_evidence"))
    prepared_nodes: List[Dict[str, Any]] = []
    precomputed_pairs: List[Tuple[str, str]] = []

    history_entries: List[Dict[str, str]] = []
    node_scores: List[float] = []
    trace: List[Dict[str, object]] = []
    node_details: List[Dict[str, Any]] = []
    covered_ctx_indices = set()
    has_excuse = False
    stats: Dict[str, float] = {
        "node_count": 0,
        "total_nli_pairs": 0,
        "patient_context_pairs": 0,
        "fallback_pairs": 0,
        "retrieval_pairs": 0,
        "history_pairs": 0,
    }

    for index, node in enumerate(row.get("arus", [])):
        node_type = str(node.get("type", "C")).upper()[:1] or "C"
        node_text = node_claim_text(node)
        if not node_text:
            continue
        prepared: Dict[str, Any] = {
            "index": index,
            "node": node,
            "node_type": node_type,
            "node_text": node_text,
        }
        if is_fragmentary_claim(node_text):
            prepared["skip_reason"] = "fragment"
            prepared_nodes.append(prepared)
            continue
        if node_type == "O":
            start = len(precomputed_pairs)
            for premise in context_sentences:
                precomputed_pairs.append((premise, node_text))
            prepared["context_span"] = (start, len(precomputed_pairs))
        elif node_type == "W":
            docs = evidence_texts(node.get("retrieved_evidence")) or fallback_docs
            evidence_premise = concat_premise(docs)
            prepared["warrant_index"] = -1
            if evidence_premise:
                prepared["warrant_index"] = len(precomputed_pairs)
                precomputed_pairs.append((evidence_premise, node_text))
        elif node_type == "D":
            docs = evidence_texts(node.get("retrieved_evidence")) or fallback_docs
            start = len(precomputed_pairs)
            for premise in docs:
                precomputed_pairs.append((premise, node_text))
            prepared["diff_doc_span"] = (start, len(precomputed_pairs))
            prepared["patient_context_index"] = -1
            if docs and patient_context and not is_considered_like_claim(node_text):
                context_premise = select_context_premise_for_claim(context_sentences, node_text)
                prepared["patient_context_premise"] = context_premise
                if context_premise:
                    prepared["patient_context_index"] = len(precomputed_pairs)
                    precomputed_pairs.append((context_premise, node_text))
            elif patient_context:
                prepared["patient_context_premise"] = ""
        prepared_nodes.append(prepared)

    precomputed_results = run_nli_pairs(
        nli_pipe,
        precomputed_pairs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    for prepared in prepared_nodes:
        index = int(prepared["index"])
        node = prepared["node"]
        node_type = prepared["node_type"]
        node_text = prepared["node_text"]

        stats["node_count"] += 1
        representative: Dict[str, Any] = empty_probs()
        aggregate = empty_probs()
        primary_probs: Dict[str, Any] = empty_probs()
        auxiliary_probs: Dict[str, Any] = empty_probs()
        route = "unknown"
        primary_route = "unknown"
        premise_type = "unknown"
        auxiliary_route = ""
        auxiliary_premise_type = ""
        pair_count = 0
        covered_sentence_index = None

        if prepared.get("skip_reason") == "fragment":
            score = 0.0
            tag = "FragmentSkip"
            route = "skipped"
            primary_route = route
            premise_type = "none"
            node_scores.append(score)
            trace.append(trace_entry(index, node_type, score, tag))
            node_details.append(
                {
                    "id": index,
                    "type": node_type,
                    "text": node_text,
                    "span_text": normalize_text(node.get("span_text") or node.get("text")),
                    "nli_claim_text": node_text,
                    "route": route,
                    "primary_route": primary_route,
                    "premise_type": premise_type,
                    "auxiliary_route": "",
                    "auxiliary_premise_type": "",
                    "pair_count": 0,
                    "entailment": 0.0,
                    "contradiction": 0.0,
                    "neutral": 1.0,
                    "aggregate_entailment": 0.0,
                    "aggregate_contradiction": 0.0,
                    "aggregate_neutral": 1.0,
                    "decision_label": "neutral",
                    "argmax_label": "neutral",
                    "supported": False,
                    "contradictory": False,
                    "auxiliary_entailment": 0.0,
                    "auxiliary_contradiction": 0.0,
                    "auxiliary_neutral": 1.0,
                    "auxiliary_decision_label": "",
                    "auxiliary_supported": False,
                    "score": 0.0,
                    "tag": tag,
                    "covered_context_sentence": False,
                    "covered_context_sentence_index": None,
                    "selected_premise": "",
                    "selected_premise_entailment": 0.0,
                    "selected_premise_contradiction": 0.0,
                    "selected_premise_neutral": 1.0,
                    "auxiliary_premise": "",
                }
            )
            continue

        if has_prevalence_excuse(node_text):
            has_excuse = True

        if node_type == "O":
            primary_route = "patient_context"
            context_start, context_end = prepared.get("context_span", (0, 0))
            context_probs = precomputed_results[context_start:context_end]
            stats["patient_context_pairs"] += len(context_probs)
            pair_count += len(context_probs)
            selected_ctx, selected_ctx_index, selection_mode = select_observation_context_entry(
                context_probs,
                node_text,
                args,
            )
            aggregate = dict(selected_ctx)
            primary_probs = dict(selected_ctx)
            representative = dict(selected_ctx)
            route = "patient_context"
            premise_type = "patient_context_sentence"

            if not context_probs:
                score = 0.5
                tag = "NoCtx"
            else:
                if (
                    selection_mode == "entail"
                    and selected_ctx.get("entailment", 0.0) >= args.entailment_threshold
                ):
                    covered_ctx_indices.add(selected_ctx_index)
                    covered_sentence_index = selected_ctx_index

                score = max(
                    selected_ctx.get("contradiction", 0.0),
                    args.lambda_miss * selected_ctx.get("neutral", 0.0),
                )
                if selection_mode == "contradict":
                    tag = f"CtxConflict:{selected_ctx.get('contradiction', 0.0):.2f}"
                elif selection_mode == "entail":
                    tag = f"Supported:{selected_ctx.get('entailment', 0.0):.2f}"
                elif decision_label(selected_ctx, args) == "contradict":
                    score = args.lambda_miss * selected_ctx.get("neutral", 0.0)
                    tag = f"CtxConflictWeakAnchor:{selected_ctx.get('contradiction', 0.0):.2f}"
                else:
                    tag = f"MissingCtx:{selected_ctx.get('neutral', 0.0):.2f}"

                if neutral_dominant(selected_ctx):
                    docs = evidence_texts(node.get("retrieved_evidence")) or fallback_docs
                    doc_premise = concat_premise(docs)
                    if doc_premise:
                        auxiliary_probs = single_premise_entry(
                            nli_pipe,
                            doc_premise,
                            node_text,
                            batch_size=args.batch_size,
                            max_length=args.max_length,
                        )
                        stats["fallback_pairs"] += 1
                        pair_count += 1
                        route = "observation_fallback"
                        auxiliary_route = "observation_fallback"
                        auxiliary_premise_type = "retrieved_evidence"
                        doc_term = args.lambda_hal * auxiliary_probs.get("contradiction", 0.0)
                        if doc_term > score:
                            tag = f"DocPlausibility:{auxiliary_probs.get('contradiction', 0.0):.2f}"
                        score = max(score, doc_term)

        elif node_type == "W":
            route = "retrieved_evidence"
            primary_route = route
            premise_type = "evidence_only"
            warrant_index = int(prepared.get("warrant_index", -1))
            if warrant_index < 0:
                score = args.lambda_sup
                tag = "NoEvid"
            else:
                primary_probs = dict(precomputed_results[warrant_index])
                stats["retrieval_pairs"] += 1
                pair_count = 1
                aggregate = primary_probs
                representative = primary_probs
                score = score_warrant_probs(primary_probs, node_text, args)
                if decision_label(primary_probs, args) == "contradict":
                    tag = f"TheoryConflict:{primary_probs.get('contradiction', 0.0):.2f}"
                elif is_catalogue_style_statement(node_text) and neutral_dominant(primary_probs):
                    tag = f"WarrantSoft:{score:.2f}"
                else:
                    tag = f"Warrant:{score:.2f}"

        elif node_type == "D":
            route = "retrieved_evidence"
            primary_route = route
            premise_type = "evidence_doc"
            doc_start, doc_end = prepared.get("diff_doc_span", (0, 0))
            doc_entries = precomputed_results[doc_start:doc_end]
            patient_context_index = int(prepared.get("patient_context_index", -1))
            if not doc_entries:
                score = args.lambda_diff
                tag = "NoEvid"
            else:
                stats["retrieval_pairs"] += len(doc_entries)
                pair_count = len(doc_entries)
                selected_doc, doc_aggregate, selection_mode = select_retrieval_entry(doc_entries, args)
                primary_probs = selected_doc
                aggregate = doc_aggregate
                representative = selected_doc
                score = score_from_probs(selected_doc, args.lambda_diff)

                if patient_context_index >= 0:
                    auxiliary_probs = dict(precomputed_results[patient_context_index])
                    stats["patient_context_pairs"] += 1
                    pair_count += 1
                    auxiliary_route = "patient_context"
                    auxiliary_premise_type = "patient_context"
                    if auxiliary_probs.get("contradiction", 0.0) > score:
                        score = auxiliary_probs.get("contradiction", 0.0)
                        representative = auxiliary_probs
                        route = "patient_context_veto"

                if selection_mode == "contradict":
                    tag = f"DiffConflict:{selected_doc.get('contradiction', 0.0):.2f}"
                elif auxiliary_probs.get("contradiction", 0.0) >= args.contradiction_threshold:
                    tag = f"CaseConflict:{auxiliary_probs.get('contradiction', 0.0):.2f}"
                else:
                    tag = f"Diff:{score:.2f}"
                if is_proxy_differential_claim(node_text) and auxiliary_probs.get("contradiction", 0.0) < args.contradiction_threshold:
                    if score > PROXY_DIFF_CONFLICT_CAP:
                        score = PROXY_DIFF_CONFLICT_CAP
                        tag = f"DiffProxy:{score:.2f}"

        else:
            route = "reasoning_history"
            primary_route = route
            premise_type = "reasoning_history"
            if not history_entries:
                primary_probs = {"premise": "", "entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
                aggregate = primary_probs
                representative = primary_probs
                score = 0.0
                tag = "LogicBase"
            else:
                history_text = logic_history_premise(history_entries, current_text=node_text)
                if not history_text:
                    primary_probs = {"premise": "", "entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
                    aggregate = primary_probs
                    representative = primary_probs
                    score = 0.0
                    tag = "LogicBase"
                else:
                    primary_probs = single_premise_entry(
                        nli_pipe,
                        history_text,
                        node_text,
                        batch_size=args.batch_size,
                        max_length=args.max_length,
                    )
                    stats["history_pairs"] += 1
                    pair_count = 1
                    aggregate = primary_probs
                    representative = primary_probs
                    score = logic_score_from_probs(primary_probs, args)
                    tag = f"Logic:{score:.2f}"

        node_scores.append(score)
        trace.append(trace_entry(index, node_type, score, tag))
        stats["total_nli_pairs"] += pair_count

        node_probs = {
            "entailment": primary_probs.get("entailment", 0.0),
            "contradiction": primary_probs.get("contradiction", 0.0),
            "neutral": primary_probs.get("neutral", 1.0),
        }
        routed_label = decision_label(node_probs, args)
        auxiliary_label = decision_label(auxiliary_probs, args) if auxiliary_route else ""
        node_details.append(
            {
                "id": index,
                "type": node_type,
                "text": node_text,
                "span_text": normalize_text(node.get("span_text") or node.get("text")),
                "nli_claim_text": node_text,
                "route": route,
                "primary_route": primary_route,
                "premise_type": premise_type,
                "auxiliary_route": auxiliary_route,
                "auxiliary_premise_type": auxiliary_premise_type,
                "pair_count": pair_count,
                "entailment": round(float(node_probs["entailment"]), 4),
                "contradiction": round(float(node_probs["contradiction"]), 4),
                "neutral": round(float(node_probs["neutral"]), 4),
                "aggregate_entailment": round(float(aggregate.get("entailment", 0.0)), 4),
                "aggregate_contradiction": round(float(aggregate.get("contradiction", 0.0)), 4),
                "aggregate_neutral": round(float(aggregate.get("neutral", 1.0)), 4),
                "decision_label": routed_label,
                "argmax_label": argmax_label(node_probs),
                "supported": routed_label == "entail",
                "contradictory": routed_label == "contradict",
                "auxiliary_entailment": round(float(auxiliary_probs.get("entailment", 0.0)), 4),
                "auxiliary_contradiction": round(float(auxiliary_probs.get("contradiction", 0.0)), 4),
                "auxiliary_neutral": round(float(auxiliary_probs.get("neutral", 1.0)), 4),
                "auxiliary_decision_label": auxiliary_label,
                "auxiliary_supported": auxiliary_label == "entail",
                "score": round(float(score), 4),
                "tag": tag,
                "covered_context_sentence": covered_sentence_index is not None,
                "covered_context_sentence_index": covered_sentence_index,
                "selected_premise": preview_text(representative.get("premise", "")),
                "selected_premise_entailment": round(float(representative.get("entailment", 0.0)), 4),
                "selected_premise_contradiction": round(float(representative.get("contradiction", 0.0)), 4),
                "selected_premise_neutral": round(float(representative.get("neutral", 1.0)), 4),
                "auxiliary_premise": preview_text(auxiliary_probs.get("premise", "")),
            }
        )
        history_entries.append(
            {
                "type": node_type,
                "text": node_text,
                "score": score,
                "decision_label": routed_label,
            }
        )

    if node_scores:
        max_risk = max(node_scores)
        avg_defect = sum(node_scores) / len(node_scores)
    else:
        max_risk = 0.0
        avg_defect = 0.0
    coverage_ratio = len(covered_ctx_indices) / total_ctx_sentences if total_ctx_sentences > 0 else 0.0
    target_coverage = args.prevalence_target_coverage if has_excuse else args.default_target_coverage
    coverage_penalty = max(0.0, target_coverage - coverage_ratio)
    if has_excuse and coverage_penalty > 0.0:
        coverage_penalty = min(1.0, coverage_penalty * 1.5)
    sample_score = min(
        1.0,
        (args.gamma * max_risk)
        + (args.alpha_cov * coverage_penalty)
        + (args.avg_defect_weight * avg_defect),
    )
    dpo_margin = min(sample_score * args.margin_scale, args.max_margin)
    stats["covered_sentences"] = float(len(covered_ctx_indices))
    stats["total_sentences"] = float(total_ctx_sentences)
    stats["coverage_ratio"] = float(coverage_ratio)
    stats["penalty_score"] = float(coverage_penalty)
    stats["has_prevalence_excuse"] = 1.0 if has_excuse else 0.0

    return {
        "id": row.get("id"),
        "source_id": row.get("source_id"),
        "error_symbol": row.get("error_symbol"),
        "unfaithfulness_score": round(float(sample_score), 4),
        "dpo_margin": round(float(dpo_margin), 4),
        "verifier": {
            "model": args.nli_model,
            "entailment_threshold": args.entailment_threshold,
            "contradiction_threshold": args.contradiction_threshold,
        },
        "metrics": {
            "max_risk": round(float(max_risk), 4),
            "avg_defect": round(float(avg_defect), 4),
            "coverage_ratio": round(float(coverage_ratio), 4),
            "penalty_score": round(float(coverage_penalty), 4),
            "covered_sentences": int(len(covered_ctx_indices)),
            "total_sentences": int(total_ctx_sentences),
            "has_prevalence_excuse": has_excuse,
            "target_coverage": round(float(target_coverage), 4),
            "trace": trace,
            "node_details": node_details,
        },
    }, stats


def score_worker(
    gpu_id: str,
    args: argparse.Namespace,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    worker_args = argparse.Namespace(**vars(args))
    worker_args.device = f"cuda:{gpu_id}"
    current_row_id = ""

    try:
        if torch is not None:
            torch.cuda.set_device(int(gpu_id))
        print(f"[GPU {gpu_id}] worker starting on {worker_args.device}", flush=True)
        nli_pipe = load_nli(worker_args)
        print(f"[GPU {gpu_id}] model loaded", flush=True)
        while True:
            row = task_queue.get()
            if row is None:
                break
            current_row_id = str(row.get("id"))
            scored_row, row_stats = score_row(row, nli_pipe, worker_args)
            result_queue.put({"kind": "result", "row": scored_row, "stats": row_stats})
            current_row_id = ""
    except Exception:
        result_queue.put(
            {
                "kind": "worker_error",
                "gpu_id": gpu_id,
                "row_id": current_row_id or None,
                "error": traceback.format_exc(),
            }
        )
    finally:
        result_queue.put({"kind": "done", "gpu_id": gpu_id})


def score_rows_multi_gpu(
    pending_rows: List[Dict[str, Any]],
    output_path: Path,
    args: argparse.Namespace,
    aggregate_stats: Dict[str, float],
    gpu_list: List[str],
) -> Dict[str, float]:
    context = mp.get_context("spawn")
    task_queue = context.Queue(maxsize=max(len(gpu_list) * 2, 1))
    result_queue = context.Queue()
    workers: Dict[str, mp.Process] = {}

    for gpu_id in gpu_list:
        process = context.Process(target=score_worker, args=(gpu_id, args, task_queue, result_queue))
        process.start()
        workers[gpu_id] = process

    next_row_index = 0
    sentinels_sent = 0

    def feed_tasks() -> None:
        nonlocal next_row_index, sentinels_sent
        while next_row_index < len(pending_rows):
            try:
                task_queue.put(pending_rows[next_row_index], timeout=0.1)
            except Full:
                return
            next_row_index += 1

        while sentinels_sent < len(gpu_list):
            try:
                task_queue.put(None, timeout=0.1)
            except Full:
                return
            sentinels_sent += 1

    feed_tasks()

    completed_workers: set[str] = set()
    worker_errors: List[Dict[str, Any]] = []

    with tqdm(total=len(pending_rows), desc="Scoring faithfulness") as progress:
        while len(completed_workers) < len(workers):
            feed_tasks()
            try:
                message = result_queue.get(timeout=1)
            except Empty:
                for gpu_id, process in workers.items():
                    if gpu_id in completed_workers:
                        continue
                    if process.is_alive() or process.exitcode is None:
                        continue
                    completed_workers.add(gpu_id)
                    if process.exitcode != 0:
                        worker_errors.append(
                            {
                                "gpu_id": gpu_id,
                                "row_id": None,
                                "error": (
                                    f"Worker exited with code {process.exitcode} "
                                    "without reporting an exception."
                                ),
                            }
                        )
                continue

            kind = message.get("kind")
            if kind == "result":
                append_jsonl(output_path, message["row"])
                aggregate_stats["scored_rows"] += 1
                for key, value in message["stats"].items():
                    aggregate_stats[key] = aggregate_stats.get(key, 0.0) + float(value)
                progress.update(1)
                feed_tasks()
            elif kind == "worker_error":
                worker_errors.append(message)
            elif kind == "done":
                completed_workers.add(str(message.get("gpu_id")))

    for process in workers.values():
        process.join()

    if worker_errors:
        first_error = worker_errors[0]
        row_hint = (
            f" while processing row `{first_error['row_id']}`" if first_error.get("row_id") is not None else ""
        )
        raise RuntimeError(
            f"Multi-GPU scoring failed on GPU {first_error.get('gpu_id')}{row_hint}:\n"
            f"{first_error.get('error', 'unknown worker error')}"
        )

    return aggregate_stats


def main() -> None:
    args = parse_args()
    started_at = iso_utc_now()
    if args.preflight_only and args.skip_preflight:
        raise ValueError("`--preflight-only` cannot be combined with `--skip-preflight`.")
    require_existing_file(args.input_file, arg_name="--input-file", purpose="stage-05 ARU evidence file")
    rows = load_rows(args.input_file)
    preflight_result: Dict[str, Any] = {}
    preflight_report_path: Path | None = None

    if not args.skip_preflight:
        preflight_result, preflight_report_path = run_preflight(args, rows)
        print(
            "Preflight status: "
            f"{preflight_result['status']} "
            f"(report: {preflight_report_path})"
        )
        if preflight_result.get("blocking_issues"):
            metadata_path = write_run_metadata(
                stage_name="06_score_faithfulness",
                args=args,
                inputs={
                    "input_file": args.input_file,
                    "source_aru_file": args.source_aru_file,
                    "stage05_metadata_file": args.stage05_metadata_file,
                },
                outputs={"preflight_report_file": str(preflight_report_path)},
                stats={"preflight": preflight_result},
                metadata_file=args.metadata_file,
                started_at=started_at,
                finished_at=iso_utc_now(),
                status="blocked_preflight",
            )
            raise RuntimeError(
                "Preflight failed. Inspect the report before running stage 06: "
                f"{preflight_report_path}\nRun metadata: {metadata_path}"
            )
        if args.strict_preflight and preflight_result.get("warnings"):
            metadata_path = write_run_metadata(
                stage_name="06_score_faithfulness",
                args=args,
                inputs={
                    "input_file": args.input_file,
                    "source_aru_file": args.source_aru_file,
                    "stage05_metadata_file": args.stage05_metadata_file,
                },
                outputs={"preflight_report_file": str(preflight_report_path)},
                stats={"preflight": preflight_result},
                metadata_file=args.metadata_file,
                started_at=started_at,
                finished_at=iso_utc_now(),
                status="blocked_preflight_warning",
            )
            raise RuntimeError(
                "Preflight raised warnings and `--strict-preflight` is enabled. "
                f"Inspect: {preflight_report_path}\nRun metadata: {metadata_path}"
            )
        if args.preflight_only:
            metadata_path = write_run_metadata(
                stage_name="06_score_faithfulness",
                args=args,
                inputs={
                    "input_file": args.input_file,
                    "source_aru_file": args.source_aru_file,
                    "stage05_metadata_file": args.stage05_metadata_file,
                },
                outputs={"preflight_report_file": str(preflight_report_path)},
                stats={"preflight": preflight_result},
                metadata_file=args.metadata_file,
                started_at=started_at,
                finished_at=iso_utc_now(),
                status="preflight_only",
            )
            print(f"Preflight-only complete: {preflight_report_path}")
            print(f"Run metadata: {metadata_path}")
            return

    output_path = ensure_parent(args.output_file)
    resumed_ids: set[str] = set()
    malformed_resume_lines = 0
    if args.resume:
        resumed_ids, malformed_resume_lines = completed_ids(str(output_path))
    elif output_path.exists():
        output_path.unlink()

    pending_rows = [row for row in rows if str(row.get("id")) not in resumed_ids]
    if args.resume and resumed_ids:
        print(
            f"Resume enabled: {len(resumed_ids)} rows already scored in {output_path}; "
            f"{len(pending_rows)} rows remaining."
        )
    if malformed_resume_lines:
        print(
            "Resume warning: "
            f"ignored {malformed_resume_lines} malformed line(s) from existing output."
        )
    if not pending_rows:
        metadata_path = write_run_metadata(
            stage_name="06_score_faithfulness",
            args=args,
            inputs={"input_file": args.input_file},
            outputs={"output_file": str(output_path)},
            stats={
                "scored_rows": 0,
                "resumed_rows": len(resumed_ids),
                "new_rows_scored": 0,
                "total_output_rows": len(resumed_ids),
                "malformed_resume_lines": malformed_resume_lines,
                "preflight": preflight_result,
            },
            metadata_file=args.metadata_file,
            started_at=started_at,
            finished_at=iso_utc_now(),
            status="resume_complete",
        )
        print(f"All rows already scored: {output_path}")
        if preflight_report_path is not None:
            print(f"Preflight report: {preflight_report_path}")
        print(f"Run metadata: {metadata_path}")
        return

    if torch is None or AutoModelForSequenceClassification is None or AutoTokenizer is None:
        raise RuntimeError("Missing scoring dependencies. Install torch and transformers first.")

    gpu_list = parse_gpu_list(args.gpus)
    if gpu_list:
        validate_gpu_list(gpu_list)
    ensure_punkt()
    aggregate_stats: Dict[str, float] = {
        "scored_rows": 0,
        "node_count": 0,
        "total_nli_pairs": 0,
        "patient_context_pairs": 0,
        "fallback_pairs": 0,
        "retrieval_pairs": 0,
        "history_pairs": 0,
    }

    if len(gpu_list) > 1:
        print(
            "Multi-GPU worker mode enabled on visible CUDA devices: "
            + ", ".join(f"cuda:{gpu_id}" for gpu_id in gpu_list)
        )
        aggregate_stats = score_rows_multi_gpu(
            pending_rows=pending_rows,
            output_path=output_path,
            args=args,
            aggregate_stats=aggregate_stats,
            gpu_list=gpu_list,
        )
    else:
        if len(gpu_list) == 1:
            args.device = f"cuda:{gpu_list[0]}"
            print(f"Single-GPU mode pinned to {args.device} via `--gpus`.")
        nli_pipe = load_nli(args)
        for row in tqdm(pending_rows, desc="Scoring faithfulness"):
            scored_row, row_stats = score_row(row, nli_pipe, args)
            append_jsonl(output_path, scored_row)
            aggregate_stats["scored_rows"] += 1
            for key, value in row_stats.items():
                aggregate_stats[key] = aggregate_stats.get(key, 0.0) + float(value)
    total_output_rows = len(resumed_ids) + int(aggregate_stats["scored_rows"])
    metadata_path = write_run_metadata(
        stage_name="06_score_faithfulness",
        args=args,
        inputs={"input_file": args.input_file},
        outputs={"output_file": str(output_path)},
        stats={
            **aggregate_stats,
            "resumed_rows": len(resumed_ids),
            "new_rows_scored": aggregate_stats["scored_rows"],
            "total_output_rows": total_output_rows,
            "malformed_resume_lines": malformed_resume_lines,
            "avg_nodes_per_sample": round(
                aggregate_stats["node_count"] / max(aggregate_stats["scored_rows"], 1), 4
            ),
            "avg_nli_pairs_per_sample": round(
                aggregate_stats["total_nli_pairs"] / max(aggregate_stats["scored_rows"], 1), 4
            ),
            "avg_coverage_ratio": round(
                aggregate_stats.get("coverage_ratio", 0.0) / max(aggregate_stats["scored_rows"], 1), 4
            ),
            "avg_penalty_score": round(
                aggregate_stats.get("penalty_score", 0.0) / max(aggregate_stats["scored_rows"], 1), 4
            ),
            "prevalence_excuse_rate": round(
                aggregate_stats.get("has_prevalence_excuse", 0.0) / max(aggregate_stats["scored_rows"], 1), 4
            ),
            "preflight": preflight_result,
        },
        metadata_file=args.metadata_file,
        started_at=started_at,
        finished_at=iso_utc_now(),
    )
    print(f"Saved score file: {output_path}")
    if preflight_report_path is not None:
        print(f"Preflight report: {preflight_report_path}")
    print(f"Run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
