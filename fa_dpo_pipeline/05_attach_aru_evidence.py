#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm

from common import artifact_path, iso_utc_now, load_jsonl, normalize_text, result_path, write_jsonl, write_run_metadata

try:
    import faiss
    import numpy as np
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover - dependency availability is environment-specific
    faiss = None
    np = None
    torch = None
    AutoModel = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


WORKFLOW_WRAPPER_PREFIXES = (
    "cardiology evaluation",
    "clinical evaluation",
    "coronary computed tomography angiography",
    "ct angiography",
    "ct pulmonary angiography",
    "on mri",
    "on ct",
    "on hrct",
    "on ultrasound",
    "on pet/ct",
    "on mammography",
    "on x-ray",
    "on radiograph",
    "fast ultrasound",
    "focused assessment with sonography in trauma",
    "histopathology",
    "pathology report",
    "pathology",
    "biopsy",
    "biopsied",
    "imaging",
    "mri",
    "ultrasound",
    "angiography",
    "workup",
    "evaluation",
    "assessment",
    "consultation",
    "histology",
)

PROPOSITION_STARTERS = (
    "was",
    "were",
    "is",
    "are",
    "can be",
    "could be",
    "may be",
    "has",
    "have",
    "had",
    "showed",
    "shows",
    "showing",
    "demonstrated",
    "demonstrates",
    "revealed",
    "reveals",
    "found",
    "finds",
    "excluded",
    "excludes",
    "exclude",
    "ruled out",
    "rule out",
    "argued against",
    "argues against",
    "considered",
    "considers",
    "listed",
    "suspected",
    "suggested",
    "suggests",
    "means",
    "indicated",
    "indicates",
    "cannot be excluded",
)

TARGET_LEADING_PATTERNS = (
    r"^(?:findings?\s+that\s+)?(?:argue|argues|argued)\s+against\s+",
    r"^(?:findings?\s+that\s+)?(?:support|supports|supported)\s+",
    r"^(?:findings?\s+that\s+)?(?:suggest|suggests|suggested)\s+",
    r"^(?:part of|in the differential diagnosis of|in differential diagnosis of|the differential diagnosis of|differential diagnosis of|a cause of|an? causes? of|the cause of|cause of|other causes? of|other etiologies? of|alternative causes? of|alternative etiologies? of|the possibility of|possibility of|evaluation of|assessment of|the clinical presentation of|clinical presentation of)\s+",
    r"^(?:this|that|the)\s+diagnosis(?:\s+was|\s+is)?\s+",
    r"^(?:this|that|the)\s+case\s+",
    r"^.+?\b(?:means|indicates|suggests)\s+",
    r"^of\s+",
)

TARGET_TRAILING_PATTERN = re.compile(
    r"\b(?:cannot be excluded|cannot be ruled out|could not be excluded|may be excluded|was less likely|were less likely|is less likely|are less likely|was(?:\s+(?:highly|very))?\s+unlikely|were(?:\s+(?:highly|very))?\s+unlikely|is(?:\s+(?:highly|very))?\s+unlikely|are(?:\s+(?:highly|very))?\s+unlikely|(?:was|were|is|are)\s+(?:briefly|initially|tentatively|also|ultimately|subsequently)\s+(?:considered|suspected|listed|entertained|favored|favoured)|was considered|were considered|was suspected|were suspected|was evaluated|were evaluated|was entertained|were entertained|was listed|were listed|was diagnosed|were diagnosed|was found|were found|was ruled out|were ruled out|is excluded|are excluded|excluded|ruled out|considered|suspected|evaluated|diagnosed|entertained|listed|found|identified|indicated)\b.*$",
    flags=re.IGNORECASE,
)

GENERIC_TARGET_EXACTS = {
    "assessment",
    "case",
    "clinical presentation",
    "clinical presentations",
    "diagnosis",
    "differential diagnosis",
    "evaluation",
    "infectious causes",
    "metabolic disorders",
    "neoplastic processes",
    "other causes",
    "rheumatologic diseases",
    "workup",
    "patient",
    "part of the preoperative differential diagnosis",
    "the diagnosis",
    "the possibility",
    "this diagnosis",
    "that diagnosis",
}

GENERIC_QUERY_EXACTS = {
    "conjunctival nevus",
    "normal dive profile",
    "tongue lesion",
    "thrombosis cannot be differential diagnosis",
}

GENERIC_QUERY_PREFIXES = (
    "was considered",
    "were considered",
    "was suspected",
    "were suspected",
    "was evaluated",
    "were evaluated",
    "was ruled out",
    "were ruled out",
    "differential diagnosis",
    "clinical features",
    "this diagnosis",
    "that diagnosis",
    "findings that argue against",
    "findings that support",
)


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="Retrieve and attach external evidence for W/D ARUs."
    )
    parser.add_argument(
        "--aru-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.strict_clean.arus.jsonl"),
        help="ARU decomposition JSONL from stage 04.",
    )
    parser.add_argument(
        "--corpus-file",
        default=artifact_path("fa_dpo_pipeline", "miriad_corpus.jsonl"),
        help="MIRIAD corpus JSONL.",
    )
    parser.add_argument(
        "--index-file",
        default=artifact_path("fa_dpo_pipeline", "miriad_faiss.index"),
        help="FAISS index from stage 01.",
    )
    parser.add_argument(
        "--offsets-file",
        default=artifact_path("fa_dpo_pipeline", "miriad_medcpt.offsets.npy"),
        help="Line-offset file from stage 01.",
    )
    parser.add_argument(
        "--output-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.strict_clean.arus.with_evidence.jsonl"),
        help="Output JSONL with attached node-level evidence.",
    )
    parser.add_argument(
        "--query-encoder",
        default="ncbi/MedCPT-Query-Encoder",
        help="Retriever query encoder.",
    )
    parser.add_argument(
        "--reranker-model",
        default="ncbi/MedCPT-Cross-Encoder",
        help="Cross-encoder reranker.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=50,
        help="Initial retrieval depth.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=50,
        help="How many retrieved documents to rerank.",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=5,
        help="How many documents to keep per ARU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Retriever encoding batch size.",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=64,
        help="Cross-encoder rerank batch size.",
    )
    parser.add_argument(
        "--rerank-query-block-size",
        type=int,
        default=2048,
        help="How many queries to group together before one large rerank pass.",
    )
    parser.add_argument(
        "--max-query-variants",
        type=int,
        default=4,
        help="Maximum retrieval query variants per W/D node.",
    )
    parser.add_argument(
        "--max-merged-candidates",
        type=int,
        default=80,
        help="Maximum unique candidates to keep after merging hits from multiple query variants, before reranking.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length for both retriever and reranker.",
    )
    parser.add_argument(
        "--device",
        default=default_device,
        help="Torch device. Use `cuda` to leverage all visible GPUs via DataParallel.",
    )
    parser.add_argument(
        "--faiss-device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="FAISS retrieval backend. `auto` prefers GPU when GPU FAISS is available.",
    )
    parser.add_argument(
        "--faiss-search-batch-size",
        type=int,
        default=16384,
        help="Batch size used for FAISS retrieval search.",
    )
    parser.add_argument(
        "--faiss-gpu-use-float16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store the sharded GPU FAISS index in float16 for lower memory use and potentially faster search.",
    )
    parser.add_argument(
        "--metadata-file",
        default=artifact_path("fa_dpo_pipeline", "run_metadata", "05_attach_aru_evidence.json"),
        help="Where to write run metadata JSON.",
    )
    return parser.parse_args()


def clean_query_text(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r'"[^"]{0,400}"', "", cleaned)
    cleaned = re.sub(r"^(?:according to|per)\s+[^,]{1,120},\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^on\s+(?:mri|ct|hrct|ultrasound|us|pet/ct|pet-ct|mammography|x-ray|radiograph)\s*,\s*", "", cleaned, flags=re.IGNORECASE)
    if re.search(r"\s+[—–]\s+", cleaned):
        head = re.split(r"\s+[—–]\s+", cleaned, maxsplit=1)[0]
        if len(head.split()) >= 4:
            cleaned = head
    cleaned = re.sub(r"\b(?:this|the)\s+patient\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bthis\s+case\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ;,.-")


def node_claim_text(node: Dict[str, Any]) -> str:
    return normalize_text(
        node.get("nli_claim_text")
        or node.get("claim_text")
        or node.get("atomic_proposition")
        or node.get("text")
    )


def strip_workflow_wrapper(text: str) -> str:
    cleaned = clean_query_text(text)
    lowered = cleaned.lower()
    for prefix in WORKFLOW_WRAPPER_PREFIXES:
        if lowered == prefix or lowered.startswith(prefix + " ") or lowered.startswith(prefix + ",") or lowered.startswith(prefix + ":"):
            remainder = cleaned[len(prefix) :].lstrip(" :,-")
            if not remainder:
                return cleaned
            if prefix.startswith("on "):
                return remainder
            remainder_lower = remainder.lower()
            if any(
                remainder_lower.startswith(starter)
                for starter in PROPOSITION_STARTERS
            ):
                return remainder
    return cleaned


def is_workflow_heavy_query(text: str) -> bool:
    cleaned = clean_query_text(text).lower()
    return any(
        cleaned.startswith(prefix + " ") or cleaned == prefix or cleaned.startswith(prefix + ",") or cleaned.startswith(prefix + ":")
        for prefix in WORKFLOW_WRAPPER_PREFIXES
    )


def is_generic_query_variant(text: str) -> bool:
    cleaned = clean_query_text(text)
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if lowered in GENERIC_QUERY_EXACTS:
        return True
    if any(lowered.startswith(prefix + " ") or lowered == prefix for prefix in GENERIC_QUERY_PREFIXES):
        return True
    if lowered.startswith("findings that argue against ") and len(cleaned.split()) <= 5:
        return True
    if lowered.startswith("findings that support ") and len(cleaned.split()) <= 5:
        return True
    if len(cleaned.split()) <= 2 and not re.search(r"[A-Z0-9]", cleaned):
        return True
    return False


def is_bare_differential_variant(text: str) -> bool:
    cleaned = clean_query_text(text)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if "differential diagnosis" not in lowered and "clinical features" not in lowered:
        return False
    tokens = cleaned.split()
    if len(tokens) <= 3:
        return True
    if is_generic_query_variant(cleaned):
        return True
    if lowered.startswith("findings that argue against ") and len(tokens) <= 6:
        return True
    if lowered.startswith("findings that support ") and len(tokens) <= 6:
        return True
    return False


def ordered_unique(items: Iterable[str]) -> List[str]:
    seen = set()
    values: List[str] = []
    for item in items:
        clean = clean_query_text(item)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        values.append(clean)
    return values


def shorten_fragment(text: str, max_tokens: int = 18) -> str:
    tokens = clean_query_text(text).split()
    if not tokens:
        return ""
    return " ".join(tokens[:max_tokens])


def normalize_target_fragment(text: str) -> str:
    cleaned = clean_query_text(text)
    if not cleaned:
        return ""
    for pattern in TARGET_LEADING_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = TARGET_TRAILING_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\b(?:cannot be|cannot)\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = clean_query_text(cleaned)
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = clean_query_text(cleaned)
    lowered = cleaned.lower()
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s+than\s+.+$", "", cleaned, flags=re.IGNORECASE)
    cleaned = clean_query_text(cleaned)
    lowered = cleaned.lower()
    cleaned = re.sub(
        r"^(?:other|alternative|additional)\s+(?:causes|etiologies|diagnoses|explanations)\s+of\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = clean_query_text(cleaned)
    cleaned = TARGET_TRAILING_PATTERN.sub("", cleaned)
    cleaned = clean_query_text(cleaned)
    lowered = cleaned.lower()
    if lowered in GENERIC_TARGET_EXACTS:
        return ""
    if any(lowered.startswith(prefix + " ") or lowered == prefix for prefix in GENERIC_QUERY_PREFIXES):
        return ""
    if lowered.startswith("of "):
        return ""
    if lowered.startswith(("because ", "due to ", "given ", "since ", "as ", "lack ", "lack of ", "absence of ", "without ", "negative ", "no ")):
        return ""
    return cleaned


def normalize_feature_fragment(text: str) -> str:
    cleaned = clean_query_text(text)
    replacements = (
        r"^(?:because|due to|given|since|as)\s+",
        r"^(?:they|it)\s+lacks?\s+",
        r"^(?:they|it)\s+does?\s+not\s+have\s+",
        r"^(?:they|it)\s+do\s+not\s+have\s+",
        r"^(?:they|it)\s+have\s+no\s+",
        r"^(?:they|it|lesions?|masses?|tumou?rs?|neoplasms?)\s+consists?\s+of\s+",
        r"^consists?\s+of\s+",
        r"^(?:there\s+is|there\s+are)\s+no\s+",
        r"^(?:they|it)\s+are\s+negative\s+for\s+",
        r"^(?:they|it)\s+is\s+negative\s+for\s+",
        r"^(?:they|it)\s+(?:often|usually|commonly|frequently|typically|occasionally|sometimes)?\s*(?:can|could|may|might)?\s*present\s+as\s+",
        r"^(?:they|it)\s+(?:often|usually|commonly|frequently|typically|occasionally|sometimes)?\s*(?:can|could|may|might)?\s*present\s+with\s+",
        r"^(?:they|it)\s+(?:often|usually|commonly|frequently|typically|occasionally|sometimes)?\s*(?:can|could|may|might)?\s*present\s+in\s+",
        r"^(?:they|it)\s+(?:can|could|may|might)\s+present\s+as\s+",
        r"^(?:they|it)\s+(?:can|could|may|might)\s+present\s+with\s+",
        r"^(?:they|it)\s+(?:can|could|may|might)\s+present\s+in\s+",
        r"^(?:they|it)\s+present\s+as\s+",
        r"^(?:they|it)\s+present\s+with\s+",
        r"^(?:they|it)\s+present\s+in\s+",
        r"^(?:they|it)\s+(?:can|could|may|might)\s+be\s+associated\s+with\s+",
        r"^(?:they|it)\s+(?:is|are|was|were)\s+associated\s+with\s+",
        r"^(?:they|it)\s+(?:can|could|may|might)\s+be\s+linked\s+to\s+",
        r"^(?:they|it)\s+(?:is|are|was|were)\s+linked\s+to\s+",
        r"^(?:they|it)\s+(?:can|could|may|might)\s+yield\s+",
        r"^(?:they|it)\s+yields?\s+",
        r"^(?:they|it)\s+(?:can|could|may|might)\s+produce\s+",
        r"^(?:they|it)\s+produces?\s+",
        r"^of\s+",
    )
    for pattern in replacements:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(?:was|were|is|are)\s+(?:excluded|ruled out|less likely|unlikely|considered|suspected|listed|entertained)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    if is_workflow_heavy_query(cleaned) and not re.search(
        r"\b(signal|signals|enhancement|enhance|calcif|lesion|lesions|mass|masses|finding|findings|feature|features|appearance|intensity|density|morpholog|edema|nodule|nodules|opacity|opacities)\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return ""
    return clean_query_text(cleaned)


def is_procedure_heavy_feature(text: str) -> bool:
    cleaned = clean_query_text(text).lower()
    if not cleaned:
        return False
    procedure_terms = (
        "biopsy",
        "colonoscopy",
        "endoscopy",
        "hrct",
        "mri",
        "ultrasound",
        "angiography",
        "pet/ct",
        "pet ct",
        "workup",
        "evaluation",
    )
    descriptor_terms = (
        "signal",
        "enhancement",
        "calcif",
        "lesion",
        "mass",
        "nodule",
        "infiltrate",
        "lymphadenopathy",
        "edema",
        "opacity",
        "pattern",
        "thickening",
        "ground-glass",
        "air-fluid",
        "effusion",
        "shunt",
        "dilation",
    )
    procedure_count = sum(term in cleaned for term in procedure_terms)
    if procedure_count == 0:
        return False
    if any(term in cleaned for term in descriptor_terms):
        return False
    return procedure_count >= 2 or cleaned.startswith("negative ")


def extract_contrastive_feature_target(text: str) -> Tuple[str, str]:
    cleaned = clean_query_text(text)
    if not cleaned:
        return "", ""
    candidate = strip_workflow_wrapper(cleaned)
    candidate = re.sub(
        r"^(?:.+?\b)?(?:showed|shows|demonstrated|demonstrates|revealed|reveals|identified|indicated|found|finds)\s+",
        "",
        candidate,
        count=1,
        flags=re.IGNORECASE,
    )
    match = re.search(
        r"^(?P<feature>.+?)\s+(?:rather than|instead of)\s+(?P<target>.+)$",
        candidate,
        flags=re.IGNORECASE,
    )
    if not match:
        return "", ""
    feature = normalize_feature_fragment(match.group("feature"))
    target = clean_query_text(match.group("target"))
    return feature, target


def extract_consistency_feature_target(text: str) -> Tuple[str, str]:
    candidate = strip_workflow_wrapper(clean_query_text(text))
    if not candidate:
        return "", ""
    match = re.search(
        r"^(?P<feature>.+?)\s+(?:is|are|was|were)\s+(?:more|most)\s+(?:consistent with|compatible with)\s+(?P<target>.+?)(?:\s+than\s+.+)?$",
        candidate,
        flags=re.IGNORECASE,
    )
    if not match:
        return "", ""
    feature = normalize_feature_fragment(match.group("feature"))
    target = normalize_target_fragment(match.group("target"))
    return feature, target


def extract_make_unlikely_feature_target(text: str) -> Tuple[str, str]:
    candidate = strip_workflow_wrapper(clean_query_text(text))
    if not candidate:
        return "", ""
    match = re.search(
        r"^(?P<feature>.+?)\s+(?:make|makes|made)\s+(?P<target>.+?)\s+(?:(?:highly|very)\s+)?(?:unlikely|less likely)\b.*$",
        candidate,
        flags=re.IGNORECASE,
    )
    if not match:
        return "", ""
    feature = normalize_feature_fragment(match.group("feature"))
    target = normalize_target_fragment(match.group("target"))
    return feature, target


def extract_support_features(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r'"[^"]{0,400}"', "", cleaned)
    cleaned = cleaned.strip(" ;,.-")
    stripped = strip_workflow_wrapper(cleaned)
    candidates = [cleaned]
    if stripped and stripped not in candidates:
        candidates = [stripped, cleaned]
    cues = [
        " given ",
        " because ",
        " based on ",
        " by ",
        " with ",
        " when ",
        " since ",
        " due to ",
        " after ",
        " from ",
    ]
    terminator = r"(?:\b(?:argues? against|supports?|suggests?|suggested|is consistent with|is compatible with|is diagnostic of|is typical of|is associated with|is linked to|excludes?|rules? out|considered|listed|suspected|found|found to be|showed|shows|revealed|reveals|demonstrated|demonstrates|identified|indicated|indicates|cannot be excluded|cannot be ruled out|may be excluded|less likely|unlikely)\b|$)"
    negative_patterns = [
        rf"\b(?P<lead>no|negative|normal|without|lack of|absence of|did not have)\s+(?P<feature>.+?)(?={terminator})",
    ]
    for candidate in candidates:
        contrastive_feature, _ = extract_contrastive_feature_target(candidate)
        if contrastive_feature:
            return shorten_fragment(contrastive_feature)
        consistency_feature, _ = extract_consistency_feature_target(candidate)
        if consistency_feature:
            return shorten_fragment(consistency_feature)
        make_feature, _ = extract_make_unlikely_feature_target(candidate)
        if make_feature:
            return shorten_fragment(make_feature)
        lowered = candidate.lower()
        for prefix in ("because ", "due to ", "given ", "since ", "as "):
            if lowered.startswith(prefix):
                fragment = normalize_feature_fragment(candidate[len(prefix) :].strip(" ;,.-"))
                if fragment:
                    return shorten_fragment(fragment)
        for cue in cues:
            index = lowered.find(cue)
            if index >= 0:
                fragment = normalize_feature_fragment(candidate[index + len(cue) :].strip(" ;,.-"))
                if fragment:
                    return shorten_fragment(fragment)
        for pattern in negative_patterns:
            match = re.search(pattern, candidate, flags=re.IGNORECASE)
            if match:
                lead = clean_query_text(match.group("lead"))
                feature = normalize_feature_fragment(match.group("feature").strip(" ;,.-"))
                if feature:
                    if is_workflow_heavy_query(feature) and not re.search(
                        r"\b(signal|signals|enhancement|enhance|calcif|lesion|lesions|mass|masses|finding|findings|feature|features|appearance|intensity|density|morpholog|edema|nodule|nodules|opacity|opacities)\b",
                        feature,
                        flags=re.IGNORECASE,
                    ):
                        continue
                    return shorten_fragment(f"{lead} {feature}")
    return ""


def extract_warrant_focus_feature(text: str) -> Tuple[str, str]:
    cleaned = clean_query_text(text)
    stripped = strip_workflow_wrapper(cleaned)
    candidates = [cleaned]
    if stripped and stripped not in candidates:
        candidates = [stripped, cleaned]
    patterns = [
        (
            r"^(?P<feature>.+?)\s+(?:is|are|was|were)\s+(?:associated with|linked to|consistent with|characteristic of|diagnostic of|indicative of|typical of|seen in|seen with|correlates with)\s+(?P<focus>.+)$",
            "feature_first",
        ),
        (
            r"^(?P<feature>.+?)\s+(?:is|are|was|were)\s+(?:rarely\s+|often\s+|commonly\s+|frequently\s+|usually\s+|typically\s+|classically\s+)?seen(?:\s+on\s+[^,.;]+?)?\s+(?:in|with)\s+(?P<focus>.+)$",
            "feature_first",
        ),
        (
            r"^(?P<feature>.+?)\s+(?:can|may|could)\s+occur\s+(?:in|with)\s+(?P<focus>.+)$",
            "feature_first",
        ),
        (
            r"^(?P<focus>.+?)\s+(?:presents?|presented|present)\s+(?:with|as)\s+(?P<feature>.+)$",
            "focus_first",
        ),
        (
            r"^(?P<focus>.+?)\s+(?:can|may|could)\s+(?:cause|mimic|result in)\s+(?P<feature>.+)$",
            "focus_first",
        ),
        (
            r"^(?P<feature>.+?)\s+(?:is|are|was|were)\s+(?:highly\s+)?suspicious for\s+(?P<focus>.+)$",
            "feature_first",
        ),
        (
            r"^(?P<focus>.+?)\s+(?:(?:typically|usually|classically|often)\s+)?(?:shows?|reveals?|demonstrates?|displays?|exhibits?|indicates?)\s+(?P<feature>.+)$",
            "focus_first",
        ),
        (
            r"^(?P<focus>.+?)\s+(?:requires?|is defined by|are defined by|is characterized by|are characterized by|is distinguished by|are distinguished by|is identified by|are identified by|is marked by|are marked by)\s+(?P<feature>.+)$",
            "focus_first",
        ),
        (
            r"^(?P<focus>.+?)\s+(?:almost always|typically|usually)\s+produces?\s+(?P<feature>.+)$",
            "focus_first",
        ),
    ]
    for candidate in candidates:
        for pattern, _mode in patterns:
            match = re.search(pattern, candidate, flags=re.IGNORECASE)
            if not match:
                continue
            focus = normalize_target_fragment(match.group("focus"))
            feature = normalize_feature_fragment(match.group("feature"))
            if pattern[0] == "^" and "suspicious for" in pattern:
                focus = re.sub(r",\s*not pathognomonic for\b.*$", "", focus, flags=re.IGNORECASE).strip(" ;,.-")
                if " with " in feature.lower():
                    feature = clean_query_text(feature.rsplit(" with ", 1)[-1])
            focus = re.sub(
                r"\b(?:rarely|often|commonly|frequently|usually|typically|occasionally|sometimes)\s*$",
                "",
                focus,
                flags=re.IGNORECASE,
            ).strip(" ;,.-")
            if focus:
                return focus, feature
    return "", ""


def split_targets(target_text: str) -> List[str]:
    cleaned = clean_query_text(target_text)
    if not cleaned:
        return []
    parts = re.split(r"\s*,\s*|\s+(?:and|or|such as|including)\s+", cleaned)
    targets = []
    for part in parts:
        value = normalize_target_fragment(part.strip(" ;,.-"))
        if not value or len(value.split()) > 16:
            continue
        targets.append(value)
    return ordered_unique(targets)


def extract_parenthetical_targets(text: str) -> List[str]:
    cleaned = clean_query_text(text)
    if not cleaned or "(" not in cleaned or ")" not in cleaned:
        return []
    targets: List[str] = []
    for match in re.finditer(r"\(([^)]{2,200})\)", cleaned):
        inner = clean_query_text(match.group(1))
        inner = re.sub(r"^(?:e\.g\.|eg|i\.e\.)[,:\s]+", "", inner, flags=re.IGNORECASE)
        inner = inner.replace("/", ",").replace(";", ",")
        inner = re.sub(r"\s+(?:and|or)\s+", ",", inner)
        targets.extend(split_targets(inner))
    return ordered_unique(targets)


def extract_differential_targets(text: str) -> List[str]:
    cleaned = clean_query_text(text)
    stripped = strip_workflow_wrapper(cleaned)
    candidates = [cleaned]
    if stripped and stripped not in candidates:
        candidates = [stripped, cleaned]
    patterns = [
        r"^(?P<feature>.+?)\b(?:argue|argues|argued) against\s+(?P<target>.+?)(?:\b(?:given|because|based on|by|with|when|since|due to|after)\b.*)?$",
        r"^(?P<feature>.+?)\b(?:excludes?|rules? out|supports|supported by|suggests?|suggested by|is consistent with|are consistent with|is compatible with|are compatible with)\s+(?P<target>.+?)(?:\b(?:given|because|based on|by|with|when|since|due to|after)\b.*)?$",
        r"^(?P<target>.+?)\b(?:was|were|is|are)?\s*(?:included(?:\s+in)?|listed(?:\s+in)?|entered(?:\s+into)?|entertained(?:\s+as)?|considered(?:\s+as)?)\s+(?:part of\s+)?(?:the\s+|a\s+)?differential diagnosis\b(?:\s+(?:given|because|based on|by|with|when|since|due to|after|for)\b.*)?$",
        r"^(?P<target>.+?)\b(?:was|were|is|are|remains|became|can be|could be|may be)?\s*(?:(?:briefly|initially|tentatively|also|ultimately|subsequently)\s+)?(?:(?:highly|very)\s+unlikely|considered|listed|suspected|favored|favoured|excluded|ruled out|unlikely|less likely|differentiated from|distinguished from)\b(?:\s+(?:given|because|based on|by|with|when|since|due to|after)\b.*)?$",
        r"^(?:excluded|ruled out|(?:highly|very)\s+unlikely|unlikely|less likely|cannot be excluded|considered|listed|suspected|favored|favoured|demonstrated|demonstrates|showed|shows|revealed|reveals|found|finds|identified|indicated|indicates)\s+(?P<target>.+)$",
        r"^(?P<prefix>.+?)\b(?:means|indicates|suggests)\s+(?P<target>.+?)(?:\s+(?:cannot be excluded|cannot be ruled out|may be excluded|is not excluded|is likely|is unlikely)\b.*)?$",
    ]
    for candidate in candidates:
        parenthetical_targets = extract_parenthetical_targets(candidate)
        _, contrastive_target = extract_contrastive_feature_target(candidate)
        if contrastive_target:
            targets = split_targets(re.sub(r"\([^)]*\)", " ", contrastive_target))
            if parenthetical_targets:
                targets = ordered_unique(targets + parenthetical_targets)
            if targets:
                return targets[:3]
        _, consistency_target = extract_consistency_feature_target(candidate)
        if consistency_target:
            targets = split_targets(re.sub(r"\([^)]*\)", " ", consistency_target))
            if targets:
                return targets[:3]
        _, make_target = extract_make_unlikely_feature_target(candidate)
        if make_target:
            targets = split_targets(re.sub(r"\([^)]*\)", " ", make_target))
            if targets:
                return targets[:3]
        for pattern in patterns:
            match = re.search(pattern, candidate, flags=re.IGNORECASE)
            if not match:
                continue
            raw_target = clean_query_text(match.group("target"))
            targets = split_targets(re.sub(r"\([^)]*\)", " ", raw_target))
            if parenthetical_targets:
                if targets:
                    targets = ordered_unique(targets + parenthetical_targets)
                else:
                    targets = parenthetical_targets
            if targets:
                return targets[:3]
        if parenthetical_targets and re.search(
            r"\b(?:considered|listed|suspected|favored|favoured|excluded|ruled out|unlikely|less likely|highly unlikely|very unlikely)\b",
            candidate,
            flags=re.IGNORECASE,
        ):
            return parenthetical_targets[:3]
    return []


def extract_warrant_focus(text: str) -> str:
    focus, _ = extract_warrant_focus_feature(text)
    if focus:
        return shorten_fragment(focus, max_tokens=14)
    cleaned = clean_query_text(text)
    candidates = [cleaned]
    stripped = strip_workflow_wrapper(cleaned)
    if stripped not in candidates:
        candidates.append(stripped)
    patterns = [
        r"^(?P<focus>.+?)\s+(?:is|are)\s+(?:typically\s+)?(?:linked to|linked with|associated with|consistent with|characteristic of|diagnostic of|indicative of|typical of|caused by|present with|present as|results in|can cause|may cause|seen in|seen with|defined by|characterized by|correlates with)\b",
        r"^(?P<focus>.+?)\s+(?:can|may)\s+(?:cause|present with|mimic)\b",
        r"^(?P<focus>.+?)\s+clinically presents with\b",
        r"^(?P<focus>.+?)\s+(?:rarely|often|commonly|frequently|usually|typically)?\s*(?:present|presents|presented|occur|occurs|occurred)\s+(?:as|with|in)\b",
        r"^(?P<focus>.+?)\s+(?:can|may|could)\s+occur\s+(?:in|with)\b",
        r"^.+?\b(?:gold standard|diagnostic gold standard|investigation of choice|definitive diagnosis)\b.*?\bfor\s+(?P<focus>.+)$",
        r"^(?:biopsy|histopathology|pathology|histology)\b.*?\bfor\s+(?P<focus>.+)$",
    ]
    for candidate in candidates:
        for pattern in patterns:
            match = re.search(pattern, candidate, flags=re.IGNORECASE)
            if match:
                return shorten_fragment(match.group("focus"), max_tokens=14)
    return ""


def infer_d_relation(text: str) -> str:
    cleaned = strip_workflow_wrapper(text).lower()
    if any(
        marker in cleaned
        for marker in (
            "argues against",
            "argue against",
            "argued against",
            "excluded",
            "ruled out",
            "less likely",
            "unlikely",
            "cannot be excluded",
            "cannot be ruled out",
        )
    ):
        return "against"
    if any(
        marker in cleaned
        for marker in (
            "supports",
            "support",
            "supported by",
            "suggests",
            "suggested by",
            "consistent with",
            "compatible with",
            "diagnostic of",
            "diagnostic for",
        )
    ):
        return "support"
    return "consider"


def variant_mentions_target(variant: str, source_query: str, source_text: str) -> bool:
    cleaned_variant = clean_query_text(variant).lower()
    if not cleaned_variant:
        return False
    targets = extract_differential_targets(source_query) or extract_differential_targets(source_text)
    if not targets:
        focus = extract_warrant_focus(source_query) or extract_warrant_focus(source_text)
        if focus:
            targets = [focus]
    return any(clean_query_text(target).lower() in cleaned_variant for target in targets if clean_query_text(target))


def build_query_variants(
    node_text: str,
    node_type: str,
    retrieval_query: str,
    max_variants: int,
) -> List[str]:
    source_query = clean_query_text(retrieval_query)
    source_text = clean_query_text(node_text)
    dewrapped_query = strip_workflow_wrapper(source_query or source_text)
    dewrapped_text = strip_workflow_wrapper(source_text)
    variants: List[str] = []

    target_seeds = [source_text, dewrapped_text, source_query, dewrapped_query]
    feature_seeds = [source_text, dewrapped_text, source_query, dewrapped_query]

    def first_targets() -> List[str]:
        for seed in target_seeds:
            targets = extract_differential_targets(seed)
            if targets:
                return targets
        return []

    def first_focus() -> str:
        for seed in target_seeds:
            focus = extract_warrant_focus(seed)
            if focus:
                return focus
        return ""

    def first_features() -> str:
        for seed in feature_seeds:
            if node_type == "W":
                focus, warrant_feature = extract_warrant_focus_feature(seed)
                if focus and warrant_feature:
                    return warrant_feature
            features = extract_support_features(seed)
            if features:
                return features
        return ""

    targets = first_targets()
    focus = first_focus()
    features = first_features()
    relation = infer_d_relation(f"{source_text} {source_query}".strip())

    if node_type == "D":
        added_target_variant = False
        for target in targets:
            added_target_variant = True
            if features:
                variants.append(f"{target} {features}")
                if relation == "against":
                    variants.append(f"findings that argue against {target} {features}")
                elif relation == "support":
                    variants.append(f"findings supporting {target} {features}")
                else:
                    variants.append(f"{target} differential diagnosis {features}")
            variants.append(f"{target} differential diagnosis")
            variants.append(f"{target} clinical features")
        if len(targets) > 1 and not features:
            combined_targets = " ".join(targets)
            variants.append(f"{combined_targets} differential diagnosis")
            variants.append(f"{combined_targets} clinical features")
        if features and (relation == "against" or not targets):
            variants.append(features)
        if not added_target_variant:
            if dewrapped_query and dewrapped_query != source_query:
                variants.append(dewrapped_query)
            if source_query and not is_workflow_heavy_query(source_query):
                variants.append(source_query)
            if source_text and source_text != source_query:
                variants.append(source_text)
    elif node_type == "W":
        if focus and not is_workflow_heavy_query(focus):
            criteria_like = bool(
                re.search(
                    r"\b(?:requires?|criteria|defined by|characterized by|distinguished by|identified by|marked by)\b",
                    f"{source_text} {source_query}",
                    flags=re.IGNORECASE,
                )
            )
            if is_workflow_heavy_query(source_text) or is_workflow_heavy_query(source_query) or criteria_like:
                variants.append(f"{focus} diagnosis")
                variants.append(f"{focus} diagnostic criteria")
            if features:
                variants.append(f"{focus} {features}")
            variants.append(focus)
            variants.append(f"{focus} clinical features")
        elif targets:
            for target in targets:
                variants.append(f"{target} differential diagnosis")
                if features:
                    variants.append(f"findings that argue against {target} {features}")
        if features:
            variants.append(features)
        if not focus and not targets:
            if dewrapped_query and dewrapped_query != source_query:
                variants.append(dewrapped_query)
            if source_query and not is_workflow_heavy_query(source_query):
                variants.append(source_query)
            if source_text and source_text != source_query:
                variants.append(source_text)
    else:
        if source_query:
            variants.append(source_query)
        if source_text and source_text != source_query:
            variants.append(source_text)

    normalized = []
    for variant in ordered_unique(variants):
        if 2 <= len(variant.split()) <= 40:
            normalized.append(variant)
    if not normalized:
        fallback = dewrapped_query or source_query or source_text
        if fallback:
            normalized = [fallback]
    return normalized[: max(1, max_variants)]


def score_query_variant(
    variant: str,
    node_type: str,
    source_query: str,
    source_text: str,
) -> Tuple[int, int, int, int, int]:
    cleaned = clean_query_text(variant)
    lowered = cleaned.lower()
    workflow_heavy = is_workflow_heavy_query(cleaned)
    raw_source = clean_query_text(source_query)
    raw_text = clean_query_text(source_text)
    is_raw_source = bool(raw_source) and cleaned == raw_source
    is_raw_text = bool(raw_text) and cleaned == raw_text
    token_count = len(cleaned.split())
    mentions_target = variant_mentions_target(cleaned, source_query=source_query, source_text=source_text)
    relation = infer_d_relation(f"{source_text} {source_query}".strip())
    source_targets = extract_differential_targets(source_query) or extract_differential_targets(source_text)
    target_tokens = max((len(target.split()) for target in source_targets), default=0)
    extra_tokens = max(0, token_count - target_tokens) if target_tokens else token_count
    has_differential = "differential diagnosis" in lowered
    has_clinical_features = "clinical features" in lowered
    has_diagnosis = (
        lowered.endswith(" diagnosis")
        or " diagnosis " in lowered
        or "diagnostic criteria" in lowered
        or "diagnostic approach" in lowered
    )
    has_argue_against = "argue against" in lowered or "argues against" in lowered or "argued against" in lowered
    feature_text = extract_support_features(cleaned)
    has_feature = bool(feature_text) and feature_text.lower() not in {"tongue lesion", "conjunctival nevus", "normal dive profile"}
    if node_type == "W":
        _, source_feature_hint = extract_warrant_focus_feature(source_text)
        if source_feature_hint and source_feature_hint.lower() in lowered:
            has_feature = True
            feature_text = source_feature_hint
    elif node_type == "D":
        source_feature_hint, _ = extract_contrastive_feature_target(source_text)
        if not source_feature_hint:
            source_feature_hint = extract_support_features(source_text) or extract_support_features(source_query)
        if source_feature_hint and source_feature_hint.lower() in lowered:
            has_feature = True
            feature_text = source_feature_hint
    procedure_heavy_feature = bool(feature_text) and is_procedure_heavy_feature(feature_text)
    if procedure_heavy_feature:
        has_feature = False
    bare_differential = is_bare_differential_variant(cleaned)
    claimish = any(
        marker in lowered
        for marker in (
            "was considered",
            "were considered",
            "was suspected",
            "were suspected",
            "was evaluated",
            "were evaluated",
            "was listed",
            "were listed",
            "was entertained",
            "were entertained",
            "was excluded",
            "were excluded",
            "was ruled out",
            "were ruled out",
            "cannot be excluded",
            "cannot be ruled out",
            "less likely",
            "unlikely",
        )
    )

    discriminative = 0
    if node_type == "D":
        if relation == "against":
            if has_feature and mentions_target and not workflow_heavy:
                discriminative = 5
            elif mentions_target and (has_argue_against or has_differential or has_clinical_features) and not workflow_heavy:
                discriminative = 4
            elif mentions_target and extra_tokens >= 2 and not workflow_heavy and not claimish:
                discriminative = 3
            elif mentions_target and not workflow_heavy:
                discriminative = 2
        elif relation == "support":
            if has_feature and mentions_target and not workflow_heavy:
                discriminative = 5
            elif mentions_target and extra_tokens >= 2 and not workflow_heavy and not claimish:
                discriminative = 4
            elif mentions_target and (has_differential or has_clinical_features) and not workflow_heavy:
                discriminative = 3
            elif mentions_target and not workflow_heavy and not claimish:
                discriminative = 2
        else:
            if has_feature and mentions_target and not workflow_heavy:
                discriminative = 5
            elif mentions_target and extra_tokens >= 2 and not workflow_heavy and not claimish:
                discriminative = 4
            elif mentions_target and (has_differential or has_clinical_features) and not workflow_heavy:
                discriminative = 3
            elif mentions_target and not workflow_heavy and not claimish:
                discriminative = 2
    elif node_type == "W":
        if has_feature and mentions_target and not workflow_heavy:
            discriminative = 5
        elif has_feature and not workflow_heavy:
            discriminative = 4
        elif has_diagnosis and not workflow_heavy:
            discriminative = 4
        elif mentions_target and not workflow_heavy and extra_tokens >= 1 and not claimish:
            discriminative = 3
        elif not workflow_heavy and not claimish and token_count >= 3:
            discriminative = 2

    if workflow_heavy:
        discriminative -= 3
    if procedure_heavy_feature:
        discriminative -= 2
    if is_generic_query_variant(cleaned):
        discriminative -= 2
    if bare_differential and not has_feature:
        discriminative -= 3
    if claimish and not has_feature:
        discriminative -= 2
    if token_count <= 2:
        discriminative -= 2
    if token_count <= 3 and not has_feature:
        discriminative -= 1
    if has_differential and not has_feature:
        discriminative -= 1
    if has_argue_against:
        discriminative += 1
    if has_clinical_features and not has_feature:
        discriminative -= 1
    length_bonus = 1 if 4 <= token_count <= 18 else 0
    source_penalty = 1 if (is_raw_source or is_raw_text) else 0
    return (discriminative, length_bonus, 1 - int(workflow_heavy), 1 - source_penalty, 1 if has_feature else 0)


def select_preferred_variant(
    variants: List[str],
    node_type: str,
    source_query: str,
    source_text: str,
) -> List[str]:
    scored = sorted(
        enumerate(variants),
        key=lambda item: (
            score_query_variant(item[1], node_type=node_type, source_query=source_query, source_text=source_text),
            -len(clean_query_text(item[1]).split()),
            -item[0],
        ),
        reverse=True,
    )
    return [item[1] for item in scored]


def unique_wd_query_plans(rows: Iterable[Dict], max_query_variants: int) -> List[Dict[str, Any]]:
    plans: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for node in row.get("arus", []):
            node_type = str(node.get("type", "")).upper()[:1]
            text = normalize_text(node.get("text"))
            claim_text = node_claim_text(node)
            if node_type not in {"W", "D"} or not text:
                continue
            if text in plans:
                existing = plans[text]
                if not existing.get("retrieval_query"):
                    existing["retrieval_query"] = clean_query_text(node.get("retrieval_query"))
                    existing["query_variants"] = build_query_variants(
                        node_text=claim_text or text,
                        node_type=node_type,
                        retrieval_query=existing["retrieval_query"],
                        max_variants=max_query_variants,
                    )
                    existing["query_variants"] = select_preferred_variant(
                        existing["query_variants"],
                        node_type=node_type,
                        source_query=existing["retrieval_query"],
                        source_text=claim_text or text,
                    )
                    existing["rerank_query"] = existing["query_variants"][0]
                    existing["nli_claim_text"] = claim_text
                continue
            retrieval_query = clean_query_text(node.get("retrieval_query"))
            query_variants = build_query_variants(
                node_text=claim_text or text,
                node_type=node_type,
                retrieval_query=retrieval_query,
                max_variants=max_query_variants,
            )
            query_variants = select_preferred_variant(
                query_variants,
                node_type=node_type,
                source_query=retrieval_query,
                source_text=claim_text or text,
            )
            plans[text] = {
                "original_text": text,
                "nli_claim_text": claim_text,
                "node_type": node_type,
                "retrieval_query": retrieval_query,
                "query_variants": query_variants,
                "rerank_query": query_variants[0],
                "retrieval_query_source": "llm" if retrieval_query else "heuristic",
            }
    return list(plans.values())


def flatten_variant_queries(plans: List[Dict[str, Any]]) -> List[str]:
    variant_index: Dict[str, int] = {}
    variants: List[str] = []
    for plan in plans:
        positions: List[int] = []
        for variant in plan.get("query_variants", []):
            if variant not in variant_index:
                variant_index[variant] = len(variants)
                variants.append(variant)
            positions.append(variant_index[variant])
        plan["variant_positions"] = positions
    return variants


def encode_queries(
    queries: List[str],
    encoder_name: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = AutoModel.from_pretrained(encoder_name)
    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    vectors: List[np.ndarray] = []
    for start in tqdm(range(0, len(queries), batch_size), desc="Encoding W/D queries"):
        batch = queries[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        vectors.append(outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32))
    return np.vstack(vectors) if vectors else np.zeros((0, 768), dtype=np.float32)


def load_doc_row(corpus_handle, offsets: np.ndarray, doc_id: int, cache: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    if doc_id in cache:
        return cache[doc_id]
    corpus_handle.seek(int(offsets[doc_id]))
    row = json.loads(corpus_handle.readline())
    cache[doc_id] = row
    return row


def search_index_in_batches(
    index: Any,
    query_vectors: np.ndarray,
    top_k: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(query_vectors) == 0:
        return (
            np.zeros((0, top_k), dtype=np.float32),
            np.zeros((0, top_k), dtype=np.int64),
        )

    all_scores: List[np.ndarray] = []
    all_ids: List[np.ndarray] = []
    for start in tqdm(range(0, len(query_vectors), batch_size), desc="Searching FAISS index"):
        batch_vectors = query_vectors[start : start + batch_size]
        batch_scores, batch_ids = index.search(batch_vectors, top_k)
        all_scores.append(batch_scores)
        all_ids.append(batch_ids)
    return np.vstack(all_scores), np.vstack(all_ids)


def faiss_gpu_count() -> int:
    if faiss is None:
        return 0
    if hasattr(faiss, "get_num_gpus"):
        try:
            return int(faiss.get_num_gpus())
        except Exception:
            pass
    if torch is not None and torch.cuda.is_available():
        return int(torch.cuda.device_count())
    return 0


def faiss_gpu_supported() -> bool:
    required = ("StandardGpuResources", "GpuMultipleClonerOptions", "index_cpu_to_gpu_multiple_py")
    return faiss is not None and all(hasattr(faiss, name) for name in required) and faiss_gpu_count() > 0


def resolve_faiss_backend(requested: str, torch_device: str) -> str:
    requested_value = str(requested).strip().lower()
    if requested_value == "cpu":
        return "cpu"
    if requested_value == "gpu":
        if not faiss_gpu_supported():
            raise RuntimeError(
                "Requested `--faiss-device gpu`, but GPU FAISS is unavailable. "
                "Install a GPU-enabled FAISS build and make sure CUDA devices are visible."
            )
        return "gpu"
    if requested_value != "auto":
        raise ValueError(f"Unsupported FAISS backend: {requested}")
    if str(torch_device).startswith("cuda") and faiss_gpu_supported():
        return "gpu"
    return "cpu"


def clone_index_to_gpu(index: Any, use_float16: bool) -> Tuple[Any, Dict[str, Any]]:
    gpu_count = faiss_gpu_count()
    if gpu_count <= 0:
        raise RuntimeError("No visible GPUs available for FAISS search.")
    resources = [faiss.StandardGpuResources() for _ in range(gpu_count)]
    co = faiss.GpuMultipleClonerOptions()
    co.shard = gpu_count > 1
    co.useFloat16 = bool(use_float16)
    try:
        gpu_index = faiss.index_cpu_to_gpu_multiple_py(resources, index, co=co)
    except TypeError:
        gpu_index = faiss.index_cpu_to_gpu_multiple_py(resources, index, co)
    return gpu_index, {
        "gpu_count": gpu_count,
        "shard": bool(co.shard),
        "use_float16": bool(co.useFloat16),
        "resources": resources,
    }


def collect_candidate_docs(
    query_pos: int,
    retrieved_ids: np.ndarray,
    retrieved_scores: np.ndarray,
    rerank_top_k: int,
    offsets: np.ndarray,
    corpus_handle: Any,
    cache: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidate_ids = retrieved_ids[query_pos][:rerank_top_k]
    candidate_scores = retrieved_scores[query_pos][:rerank_top_k]
    docs: List[Dict[str, Any]] = []
    seen_docs = set()
    for rank, (doc_id, retriever_score) in enumerate(zip(candidate_ids, candidate_scores), start=1):
        if doc_id < 0 or doc_id >= len(offsets):
            continue
        row = load_doc_row(corpus_handle, offsets, int(doc_id), cache)
        text = str(row.get("text", ""))
        if not text or text in seen_docs:
            continue
        seen_docs.add(text)
        docs.append(
            {
                "doc_id": int(doc_id),
                "corpus_id": row.get("_id"),
                "title": row.get("title"),
                "text": text,
                "metadata": row.get("metadata") or {},
                "retriever_score": float(retriever_score),
                "retriever_rank": rank,
            }
        )
    return docs


def merge_candidate_docs(
    doc_groups: List[Tuple[str, List[Dict[str, Any]]]],
    max_candidates: int,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for query_variant, docs in doc_groups:
        for doc in docs:
            text = normalize_text(doc.get("text"))
            if not text:
                continue
            current = merged.get(text)
            if current is None:
                row = dict(doc)
                row["matched_queries"] = [query_variant]
                row["matched_query_count"] = 1
                row["best_retriever_score"] = float(doc.get("retriever_score", 0.0))
                row["best_retriever_rank"] = int(doc.get("retriever_rank", 10**9))
                merged[text] = row
                continue
            if query_variant not in current["matched_queries"]:
                current["matched_queries"].append(query_variant)
            current["matched_query_count"] = len(current["matched_queries"])
            doc_score = float(doc.get("retriever_score", 0.0))
            doc_rank = int(doc.get("retriever_rank", 10**9))
            if doc_score > float(current.get("best_retriever_score", 0.0)):
                current["best_retriever_score"] = doc_score
                current["retriever_score"] = doc_score
            if doc_rank < int(current.get("best_retriever_rank", 10**9)):
                current["best_retriever_rank"] = doc_rank
                current["retriever_rank"] = doc_rank
                current["doc_id"] = doc.get("doc_id")
                current["corpus_id"] = doc.get("corpus_id")
                current["title"] = doc.get("title")
                current["metadata"] = doc.get("metadata") or {}

    ranked = list(merged.values())
    ranked.sort(
        key=lambda item: (
            -float(item.get("best_retriever_score", 0.0)),
            int(item.get("best_retriever_rank", 10**9)),
            -int(item.get("matched_query_count", 0)),
        )
    )
    for rank, item in enumerate(ranked, start=1):
        item["merged_retriever_rank"] = rank
    return ranked[: max(1, max_candidates)]


def score_query_doc_pairs(
    pairs: List[List[str]],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    batch_size: int,
    max_length: int,
    device: str,
    progress_desc: str | None = None,
) -> List[float]:
    if not pairs:
        return []

    ordered = sorted(
        enumerate(pairs),
        key=lambda item: len(item[1][0]) + len(item[1][1]),
    )
    ordered_pairs = [pair for _, pair in ordered]
    ordered_indices = [idx for idx, _ in ordered]
    scores: List[float] = [0.0] * len(pairs)

    iterator: Iterable[int] = range(0, len(ordered_pairs), batch_size)
    if progress_desc:
        iterator = tqdm(
            iterator,
            total=(len(ordered_pairs) + batch_size - 1) // batch_size,
            desc=progress_desc,
            leave=False,
        )

    for start in iterator:
        batch_pairs = ordered_pairs[start : start + batch_size]
        inputs = tokenizer(
            batch_pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        if logits.ndim == 1 or logits.shape[-1] == 1:
            batch_scores = logits.view(-1).detach().cpu().numpy().tolist()
        else:
            batch_scores = logits[:, 1].detach().cpu().numpy().tolist()
        for original_idx, score in zip(ordered_indices[start : start + batch_size], batch_scores):
            scores[original_idx] = float(score)
    return scores


def rank_docs_with_scores(
    docs: List[Dict[str, Any]],
    scores: List[float],
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for doc, score in zip(docs, scores):
        row = dict(doc)
        row["reranker_score"] = float(score)
        ranked.append(row)
    ranked.sort(key=lambda item: item.get("reranker_score", 0.0), reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["reranker_rank"] = rank
    return ranked


def main() -> None:
    args = parse_args()
    started_at = iso_utc_now()
    stage_started = time.perf_counter()
    if any(
        module is None
        for module in [faiss, np, torch, AutoModel, AutoModelForSequenceClassification, AutoTokenizer]
    ):
        raise RuntimeError(
            "Missing retrieval dependencies. Install faiss/numpy/torch/transformers first."
        )
    rows = load_jsonl(args.aru_file)
    query_plans = unique_wd_query_plans(rows, max_query_variants=args.max_query_variants)
    if not query_plans:
        write_jsonl(args.output_file, rows)
        print(f"No W/D queries found. Wrote passthrough file: {args.output_file}")
        return
    variant_queries = flatten_variant_queries(query_plans)
    query_vectors = encode_queries(
        variant_queries,
        encoder_name=args.query_encoder,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    if args.device.startswith("cuda") and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"Unique W/D nodes: {len(query_plans):,}; "
        f"retrieval query variants: {len(variant_queries):,}"
    )
    faiss_load_started = time.perf_counter()
    index = faiss.read_index(args.index_file)
    faiss_load_seconds = time.perf_counter() - faiss_load_started
    cpu_index_type = type(index).__name__
    index_docs = int(getattr(index, "ntotal", 0))
    print(f"Loaded FAISS index: {cpu_index_type} with {index_docs:,} documents")
    if "IndexFlat" in cpu_index_type and index_docs:
        estimated_pairs = len(variant_queries) * index_docs
        print(
            "Exact flat retrieval workload: "
            f"{estimated_pairs:,} query-document scores before top-{args.retrieval_top_k} filtering"
        )
    faiss_backend = resolve_faiss_backend(args.faiss_device, args.device)
    faiss_transfer_seconds = 0.0
    faiss_gpu_info: Dict[str, Any] = {}
    search_index = index
    if faiss_backend == "gpu":
        faiss_transfer_started = time.perf_counter()
        search_index, faiss_gpu_info = clone_index_to_gpu(
            index=index,
            use_float16=args.faiss_gpu_use_float16,
        )
        faiss_transfer_seconds = time.perf_counter() - faiss_transfer_started
        print(
            "FAISS backend: GPU "
            f"({faiss_gpu_info['gpu_count']} visible GPU(s), "
            f"shard={faiss_gpu_info['shard']}, float16={faiss_gpu_info['use_float16']})"
        )
    else:
        print("FAISS backend: CPU")
    faiss_search_started = time.perf_counter()
    retrieved_scores, retrieved_ids = search_index_in_batches(
        index=search_index,
        query_vectors=query_vectors,
        top_k=args.retrieval_top_k,
        batch_size=max(1, args.faiss_search_batch_size),
    )
    faiss_search_seconds = time.perf_counter() - faiss_search_started
    del query_vectors
    del search_index
    if faiss_backend == "gpu":
        faiss_gpu_info.pop("resources", None)
    del index
    gc.collect()
    if torch.cuda.is_available() and (args.device.startswith("cuda") or faiss_backend == "gpu"):
        torch.cuda.empty_cache()
    offsets = np.load(args.offsets_file)

    print(f"Loading reranker: {args.reranker_model}")
    reranker_load_started = time.perf_counter()
    rerank_tokenizer = AutoTokenizer.from_pretrained(args.reranker_model)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(args.reranker_model)
    if args.device.startswith("cuda") and torch.cuda.device_count() > 1:
        rerank_model = torch.nn.DataParallel(rerank_model)
    rerank_model.to(args.device)
    rerank_model.eval()
    reranker_load_seconds = time.perf_counter() - reranker_load_started
    print(
        "Planned rerank workload: "
        f"up to {len(query_plans) * args.max_merged_candidates:,} query-document pairs "
        f"(after merging hits from up to {args.max_query_variants} variants per node)"
    )

    evidence_map: Dict[str, List[Dict[str, Any]]] = {}
    doc_cache: Dict[int, Dict[str, Any]] = {}
    total_rerank_pairs = 0
    merged_candidate_counts: List[int] = []
    rerank_started = time.perf_counter()
    with Path(args.corpus_file).open("r", encoding="utf-8") as corpus_handle:
        query_block_size = max(1, args.rerank_query_block_size)
        total_blocks = (len(query_plans) + query_block_size - 1) // query_block_size
        for block_index, block_start in enumerate(
            tqdm(range(0, len(query_plans), query_block_size), total=total_blocks, desc="Retrieving ARU evidence"),
            start=1,
        ):
            block_end = min(block_start + query_block_size, len(query_plans))
            block_plans = query_plans[block_start:block_end]
            block_docs: List[List[Dict[str, Any]]] = []
            pair_ranges: List[Tuple[int, int]] = []
            flat_pairs: List[List[str]] = []

            for plan in block_plans:
                doc_groups: List[Tuple[str, List[Dict[str, Any]]]] = []
                for variant, query_pos in zip(plan.get("query_variants", []), plan.get("variant_positions", [])):
                    docs = collect_candidate_docs(
                        query_pos=query_pos,
                        retrieved_ids=retrieved_ids,
                        retrieved_scores=retrieved_scores,
                        rerank_top_k=args.rerank_top_k,
                        offsets=offsets,
                        corpus_handle=corpus_handle,
                        cache=doc_cache,
                    )
                    doc_groups.append((variant, docs))
                docs = merge_candidate_docs(doc_groups, max_candidates=args.max_merged_candidates)
                merged_candidate_counts.append(len(docs))
                block_docs.append(docs)
                pair_start = len(flat_pairs)
                rerank_query = str(
                    plan.get("rerank_query")
                    or plan.get("retrieval_query")
                    or plan.get("nli_claim_text")
                    or plan.get("original_text")
                    or ""
                )
                flat_pairs.extend([[rerank_query, str(doc.get("text", ""))] for doc in docs])
                pair_ranges.append((pair_start, len(flat_pairs)))

            total_rerank_pairs += len(flat_pairs)
            block_scores = score_query_doc_pairs(
                pairs=flat_pairs,
                tokenizer=rerank_tokenizer,
                model=rerank_model,
                batch_size=args.rerank_batch_size,
                max_length=args.max_length,
                device=args.device,
                progress_desc=f"Reranking pairs block {block_index}/{total_blocks}",
            )
            for plan, docs, (pair_start, pair_end) in zip(block_plans, block_docs, pair_ranges):
                reranked = rank_docs_with_scores(docs, block_scores[pair_start:pair_end])
                evidence_map[str(plan.get("original_text", ""))] = reranked
    rerank_seconds = time.perf_counter() - rerank_started

    merged_rows: List[Dict] = []
    wd_node_count = 0
    retained_counts: List[int] = []
    candidate_counts: List[int] = []
    plan_lookup = {str(plan.get("original_text", "")): plan for plan in query_plans}
    for row in rows:
        row_copy = dict(row)
        merged_nodes: List[Dict] = []
        for node in row.get("arus", []):
            node_copy = dict(node)
            node_type = str(node.get("type", "")).upper()[:1]
            if node_type in {"W", "D"}:
                wd_node_count += 1
                original_text = normalize_text(node.get("text"))
                plan = plan_lookup.get(original_text, {})
                candidates = evidence_map.get(original_text, [])
                node_copy["retrieval_query"] = str(plan.get("rerank_query") or plan.get("retrieval_query") or "")
                node_copy["rerank_query"] = str(plan.get("rerank_query") or "")
                node_copy["retrieval_query_variants"] = list(plan.get("query_variants") or [])
                node_copy["retrieval_query_source"] = str(plan.get("retrieval_query_source") or "")
                node_copy["retrieval_candidates"] = candidates
                node_copy["retrieved_evidence"] = [
                    str(candidate.get("text", "")) for candidate in candidates[: args.final_top_k]
                ]
                retained_counts.append(len(node_copy["retrieved_evidence"]))
                candidate_counts.append(len(candidates))
            else:
                node_copy["retrieval_query"] = ""
                node_copy["rerank_query"] = ""
                node_copy["retrieval_query_variants"] = []
                node_copy["retrieval_query_source"] = ""
                node_copy["retrieval_candidates"] = []
                node_copy["retrieved_evidence"] = []
            merged_nodes.append(node_copy)
        row_copy["arus"] = merged_nodes
        merged_rows.append(row_copy)

    write_jsonl(args.output_file, merged_rows)
    total_seconds = time.perf_counter() - stage_started
    metadata_path = write_run_metadata(
        stage_name="05_attach_aru_evidence",
        args=args,
        inputs={
            "aru_file": args.aru_file,
            "corpus_file": args.corpus_file,
            "index_file": args.index_file,
            "offsets_file": args.offsets_file,
        },
        outputs={"output_file": args.output_file},
        stats={
            "processed_rows": len(rows),
            "wd_node_count": wd_node_count,
            "unique_queries": len(query_plans),
            "unique_query_variants": len(variant_queries),
            "avg_query_variants_per_wd": round(
                sum(len(plan.get("query_variants", [])) for plan in query_plans) / max(len(query_plans), 1), 4
            ),
            "avg_retained_evidence": round(sum(retained_counts) / max(len(retained_counts), 1), 4),
            "avg_reranked_candidates": round(sum(candidate_counts) / max(len(candidate_counts), 1), 4),
            "avg_merged_candidates_pre_rerank": round(
                sum(merged_candidate_counts) / max(len(merged_candidate_counts), 1), 4
            ),
            "total_rerank_pairs": total_rerank_pairs,
            "faiss_backend": faiss_backend,
            "faiss_index_type": cpu_index_type,
            "faiss_index_documents": index_docs,
            "faiss_gpu_count": faiss_gpu_info.get("gpu_count"),
            "faiss_gpu_shard": faiss_gpu_info.get("shard"),
            "faiss_gpu_use_float16": faiss_gpu_info.get("use_float16"),
            "faiss_load_seconds": round(faiss_load_seconds, 4),
            "faiss_transfer_seconds": round(faiss_transfer_seconds, 4),
            "faiss_search_seconds": round(faiss_search_seconds, 4),
            "reranker_load_seconds": round(reranker_load_seconds, 4),
            "rerank_seconds": round(rerank_seconds, 4),
            "total_runtime_seconds": round(total_seconds, 4),
        },
        metadata_file=args.metadata_file,
        started_at=started_at,
        finished_at=iso_utc_now(),
    )
    print(f"Saved ARU evidence file: {args.output_file}")
    print(
        f"Unique W/D queries: {len(query_plans)} "
        f"(variants: {len(variant_queries)})"
    )
    print(
        "Stage timings (s): "
        f"faiss_load={faiss_load_seconds:.2f}, "
        f"faiss_transfer={faiss_transfer_seconds:.2f}, "
        f"faiss_search={faiss_search_seconds:.2f}, "
        f"reranker_load={reranker_load_seconds:.2f}, "
        f"rerank={rerank_seconds:.2f}, "
        f"total={total_seconds:.2f}"
    )
    print(f"Run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
