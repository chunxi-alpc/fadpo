#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from common import (
    artifact_path,
    append_jsonl,
    count_jsonl,
    extract_json,
    iso_utc_now,
    load_jsonl,
    normalize_text,
    result_path,
    write_run_metadata,
)

try:
    from aiolimiter import AsyncLimiter
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - dependency availability is environment-specific
    AsyncLimiter = None
    AsyncOpenAI = None


SYSTEM_PROMPT = """You are a rigorous NLI-oriented clinical reasoning span annotator.

Your task is to decompose the "Reasoning to decompose" into Atomic Reasoning Units
(ARUs) by selecting exact substrings from that reasoning text.

Critical rule: this is an extractive span task for `span_text`.
- Every ARU must be copied exactly from the "Reasoning to decompose" text.
- Do not paraphrase, summarize, rewrite, normalize, or add words in `span_text`.
- Do not create spans from the patient context. Patient context is only for understanding.
- `span_text` must equal reasoning_text[char_start:char_end].
- `char_start` is zero-based and inclusive.
- `char_end` is zero-based and exclusive.
- Never output the whole reasoning text or a whole paragraph as one ARU.
- Do not over-split. A `span_text` must be the shortest exact substring that is still independently checkable by an NLI/verifier model.
- Avoid fragment spans such as "excluded when", "confirmed by", "because of imaging", or "and immunoprofile" unless the diagnosis or clinical object is also included.
- Prefer complete clinical propositions over tiny phrase fragments.

There are two related boundaries:
- `span_text` / `char_start` / `char_end` = exact original-text boundary. It must be a minimal but complete reasoning unit.
- `nli_claim_text` = verifier/NLI proposition. It may rewrite the span into a cleaner self-contained proposition.

The exact span should usually already contain the diagnosis/finding/rule being checked. Use `nli_claim_text` to decontextualize, not to rescue an incomplete fragment.

NLI claim text:
- Also produce `nli_claim_text` for every ARU.
- `nli_claim_text` is the short self-contained proposition that the downstream NLI verifier should check.
- It may rewrite or decontextualize the exact span, but it must preserve the same meaning and must not add unsupported facts.
- Resolve pronouns or vague phrases when the target is clear from nearby reasoning, e.g. "this diagnosis was unlikely" -> "pulmonary embolism was unlikely".
- For D units, include the diagnosis/alternative and the relation: considered, supported, excluded, ruled out, less likely, or more likely.
- For W units, state the general medical rule as a proposition.
- For O units, state the patient-specific finding as a proposition.
- For C units, state the conclusion as a proposition.

Role labels:
- O / Observation: patient-specific facts stated in the model response.
- W / Warrant: general medical knowledge, mechanisms, diagnostic criteria, or principles.
- D / Differentiation: considering, supporting, excluding, comparing, favoring, or lowering the likelihood of an alternative diagnosis.
- C / Claim: standalone intermediate or final clinical conclusion.

Role decision order:
1. If the span considers, excludes, supports, compares, or ranks a diagnosis, label it D.
2. Else if it states a general medical rule, label it W.
3. Else if it states a patient-specific fact from this case, label it O.
4. Else if it states a standalone conclusion, label it C.

Role calibration rules:
- A diagnosis name in a differential/list heading is usually D, not C.
- A sentence that says a diagnosis was considered, excluded, unlikely, favored, argued against, or supported is D.
- A concrete finding, test result, imaging result, pathology result, culture result, serology result, or criterion result is O, even if it helps support or exclude a diagnosis.
- Quoted case-record evidence is usually O.
- Quoted general medical knowledge is W only when it is clearly a general rule outside this specific patient.
- Do not label a span C just because it contains the final-answer string. C requires a standalone conclusion such as "X was diagnosed", "diagnosis of X was confirmed", or "findings confirmed X".

Boundary rules:
- Use the smallest exact span that still preserves the clinical reasoning function.
- For each numbered or sentence-level differential item, usually extract 1-3 ARUs.
- Keep a diagnosis decision together with its rationale when they form one checkable diagnostic judgment.
- Do not split a diagnosis title from the explanation after a dash when the title and explanation together express one differential decision.
- Split quoted evidence from the unquoted model interpretation.
- If the unquoted interpretation contains a diagnosis decision tightly linked to its reason, keep the decision and reason together as one D span.
- If a clause says "X was considered because/due to/given Y but was excluded by Z", extract one complete D span when it forms a single diagnostic judgment, e.g. "X was considered because Y but was excluded by Z".
- Split quoted evidence separately only when the quote states a distinct patient fact, medical rule, or diagnosis that should be checked on its own.
- Split quoted evidence from the model's interpretation of that evidence.
- Do split a finding, a medical rule, and a diagnosis conclusion if they are separable.
- Do not split every noun phrase or lab value into separate ARUs.
- Do not include list numbers such as "1." unless they are unavoidable.
- Preserve response order.

Coverage rules:
- A response with several numbered items should almost never have fewer ARUs than numbered items.
- A long response with multiple diagnoses and quoted evidence usually needs many ARUs.
- If the response mentions 8 alternative diagnoses, extract spans for all 8; do not stop early.

Retrieval query:
- For W and D units, produce a short decontextualized `retrieval_query`.
- For O and C units, use an empty string.

Final answer and masking:
- `is_final_answer_claim` is true only for a C span that explicitly states the final answer/diagnosis.
- Do not mark a bare diagnosis heading as final-answer claim.
- Do not mark "suggested", "fit", "consistent with", "favored", or "supported" spans as final-answer claim unless the span explicitly says diagnosed/confirmed/final diagnosis.
- `mask_eligible` is usually true for W, D, and C; usually false for direct O.

Good examples:
Input reasoning:
1. Emphysematous pyelonephritis was diagnosed — "Given the patient's history of uncontrolled diabetes mellitus, microbiology findings and radiological CT findings, a diagnosis of severe emphysematous pyelonephritis was made."
2. Nephrocolic fistula was considered but deemed unlikely — "Other differentials to consider would include a nephrocolic fistula ... although these were not likely in this case given the bilateral symmetrical gas distribution, which is more characteristic of emphysematous pyelonephritis than fistulous communication."
3. Anaphylaxis — absence of allergic signs argued against this diagnosis: "She had no pruritus."

Good output spans:
- "Emphysematous pyelonephritis was diagnosed" -> C; nli_claim_text: "Emphysematous pyelonephritis was diagnosed."
- "Given the patient's history of uncontrolled diabetes mellitus, microbiology findings and radiological CT findings" -> O; nli_claim_text: "The patient had uncontrolled diabetes, relevant microbiology findings, and relevant CT findings."
- "a diagnosis of severe emphysematous pyelonephritis was made" -> O; nli_claim_text: "The case record stated that severe emphysematous pyelonephritis was diagnosed."
- "Nephrocolic fistula was considered but deemed unlikely" -> D; nli_claim_text: "Nephrocolic fistula was considered but deemed unlikely."
- "the bilateral symmetrical gas distribution, which is more characteristic of emphysematous pyelonephritis than fistulous communication" -> W; nli_claim_text: "Bilateral symmetrical gas distribution is more characteristic of emphysematous pyelonephritis than fistulous communication."
- "Anaphylaxis — absence of allergic signs argued against this diagnosis" -> D; nli_claim_text: "The absence of allergic signs argued against anaphylaxis."
- "She had no pruritus" -> O; nli_claim_text: "The patient had no pruritus."

Additional calibration examples:

Input reasoning:
1. Pneumonia was considered because fever and a right lower-lobe opacity were present, but it was excluded after "blood and sputum cultures were negative."
2. Autoimmune myelitis — "the AQP4-IgG test was negative and the required longitudinally extensive lesion was absent."
3. General diagnostic rule — "A disease may mimic infection when it causes fever and focal inflammatory changes."
4. Diagnosis of reactive arthritis was confirmed after synovial fluid culture was negative and HLA-B27 was positive.

Good output spans:
- "Pneumonia was considered because fever and a right lower-lobe opacity were present, but it was excluded after" -> D; nli_claim_text: "Pneumonia was considered because fever and a right lower-lobe opacity were present, but it was excluded after negative cultures."
- "blood and sputum cultures were negative" -> O; nli_claim_text: "Blood and sputum cultures were negative."
- "Autoimmune myelitis — the AQP4-IgG test was negative and the required longitudinally extensive lesion was absent" -> D; nli_claim_text: "Autoimmune myelitis was not supported because the AQP4-IgG test was negative and the required longitudinally extensive lesion was absent."
- "A disease may mimic infection when it causes fever and focal inflammatory changes" -> W; nli_claim_text: "A disease may mimic infection when it causes fever and focal inflammatory changes."
- "Diagnosis of reactive arthritis was confirmed" -> C; is_final_answer_claim: true; nli_claim_text: "Reactive arthritis was confirmed."
- "synovial fluid culture was negative" -> O; nli_claim_text: "Synovial fluid culture was negative."
- "HLA-B27 was positive" -> O; nli_claim_text: "HLA-B27 was positive."

Return one JSON object only:
{
  "units": [
    {
      "span_text": "exact substring from Reasoning to decompose",
      "nli_claim_text": "self-contained proposition for NLI",
      "char_start": 0,
      "char_end": 10,
      "type": "O|W|D|C",
      "retrieval_query": "",
      "is_final_answer_claim": false,
      "mask_eligible": true
    }
  ]
}
""".strip()

MAX_PATIENT_CONTEXT_CHARS = 2000
MAX_REASONING_FOR_RETRY_ECHO_CHARS = 5000


def clean_query_text(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r'"[^"]{0,400}"', "", cleaned)
    if "—" in cleaned:
        head, tail = cleaned.split("—", 1)
        if len(head.split()) >= 4:
            cleaned = head
    cleaned = re.sub(r"\b(?:this|the)\s+patient\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bthis\s+case\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ;,.-")


ROLE_LABELS = {
    "O": "Observation",
    "W": "Warrant",
    "D": "Differentiation",
    "C": "Claim",
}

ROLE_TO_TYPE = {
    "OBSERVATION": "O",
    "WARRANT": "W",
    "DIFFERENTIATION": "D",
    "CLAIM": "C",
}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = normalize_text(value).lower()
    return text in {"1", "true", "yes", "y", "t"}


def parse_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = normalize_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_role_type(value: Any) -> str:
    cleaned = normalize_text(value).strip()
    if not cleaned:
        return "C"
    upper = cleaned.upper()
    if upper[:1] in {"O", "W", "D", "C"} and upper in {"O", "W", "D", "C"}:
        return upper
    return ROLE_TO_TYPE.get(upper, upper[:1] if upper[:1] in {"O", "W", "D", "C"} else "C")


def trim_span(text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def valid_span_word_boundaries(text: str, start: int, end: int) -> bool:
    if not (0 <= start < end <= len(text)):
        return False
    if start > 0 and text[start - 1].isalnum() and text[start].isalnum():
        return False
    if end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        return False
    return True


def locate_exact_span(reasoning_text: str, span_text: str, cursor: int = 0) -> Optional[Tuple[int, int]]:
    if not span_text:
        return None
    for search_start in (max(cursor, 0), 0):
        start = reasoning_text.find(span_text, search_start)
        while start >= 0:
            end = start + len(span_text)
            trimmed_start, trimmed_end = trim_span(reasoning_text, start, end)
            if trimmed_start < trimmed_end and valid_span_word_boundaries(reasoning_text, trimmed_start, trimmed_end):
                return trimmed_start, trimmed_end
            start = reasoning_text.find(span_text, start + 1)
    return None


def valid_explicit_span(reasoning_text: str, start: Optional[int], end: Optional[int], span_text: str) -> Optional[Tuple[int, int]]:
    if start is None or end is None:
        return None
    if not (0 <= start < end <= len(reasoning_text)):
        return None
    start, end = trim_span(reasoning_text, start, end)
    if start >= end:
        return None
    if not valid_span_word_boundaries(reasoning_text, start, end):
        return None
    actual = reasoning_text[start:end]
    if span_text and actual != span_text:
        return None
    return start, end


def compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text).lower())


def final_answer_is_mentioned(span_text: str, final_answer: str) -> bool:
    text = normalize_text(span_text).lower()
    answer = normalize_text(final_answer).lower()
    if not text or not answer:
        return False
    text_compact = compact_text(text)
    answer_compact = compact_text(answer)
    if text_compact == answer_compact:
        return True
    answer_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", answer)
        if token not in {"the", "a", "an", "of", "and", "or", "with", "without"}
    ]
    text_tokens = re.findall(r"[a-z0-9]+", text)
    if not answer_tokens:
        return False
    if len(answer_tokens) == 1:
        token = answer_tokens[0]
        return len(token) >= 5 and token in text_tokens
    if re.search(r"(?<![a-z0-9])" + re.escape(answer) + r"(?![a-z0-9])", text):
        return True
    matched = sum(1 for token in answer_tokens if token in text_tokens)
    return matched / len(answer_tokens) >= 0.5


def context_indicates_nonfinal_claim(span_text: str, context_text: str) -> bool:
    span = normalize_text(span_text)
    context = normalize_text(context_text)
    if not span or not context:
        return False
    idx = context.find(span)
    before = context[:idx] if idx >= 0 else context[:120]
    after = context[idx + len(span) :] if idx >= 0 else context[-120:]
    near_before_raw = before[-140:]
    numbered_marker = list(re.finditer(r"(?:^|\s)\d+\.\s*", near_before_raw))
    if numbered_marker:
        near_before_raw = near_before_raw[numbered_marker[-1].end() :]
    near_before = near_before_raw.lower()
    near_after = after[:140].lower()
    if re.search(
        r"\b(?:considered|suspected|differential|excluded|ruled out|less likely|unlikely|deemed unlikely|argues? against|did not support|not support)\b",
        near_before,
    ):
        return True
    if re.search(r"\b(?:but|however)\b.{0,80}\b(?:excluded|ruled out|less likely|unlikely|not support)\b", near_after):
        return True
    return False


def infer_final_answer_claim(
    unit_type: str,
    span_text: str,
    row: Dict[str, Any],
    context_text: str = "",
) -> bool:
    if unit_type != "C":
        return False
    final_answer = row_final_answer(row).lower()
    if not final_answer:
        return False
    text = normalize_text(span_text).lower()
    if not final_answer_is_mentioned(text, final_answer):
        return False
    if re.search(
        r"\b(?:suspected|considered|possible|differential|ruled out|excluded|less likely|unlikely|argues? against|not support|did not support|does not specifically confirm|not specifically confirm)\b",
        text,
    ):
        return False
    if context_indicates_nonfinal_claim(span_text, context_text):
        return False
    if re.search(
        r"\b(?:final|correct|confirmed|confirms|diagnosed|diagnosis was made|diagnosis of|answer is|proposed the diagnosis|confirmed the diagnosis)\b",
        text,
    ):
        return True
    return False


def mask_eligible_default(unit_type: str) -> bool:
    return unit_type in {"W", "D", "C"}


def final_claim_candidate_score(unit: Dict[str, Any], reasoning_text: str, row: Dict[str, Any]) -> float:
    if unit.get("type") != "C":
        return -1.0
    span_text = normalize_text(unit.get("span_text") or unit.get("text"))
    start = parse_int(unit.get("char_start"))
    end = parse_int(unit.get("char_end"))
    if start is None or end is None:
        context_text = ""
    else:
        context_text = reasoning_text[max(0, start - 120) : min(len(reasoning_text), end + 220)]
    if not infer_final_answer_claim("C", span_text, row, context_text):
        return -1.0
    final_answer = row_final_answer(row)
    token_count = len(re.findall(r"[A-Za-z0-9]+", span_text))
    score = 1.0
    if final_answer_title_like(span_text, final_answer):
        score += 0.5
    if re.search(
        r"\b(?:final|correct|confirmed|confirms|diagnosed|diagnosis was made|answer is|"
        r"proposed the diagnosis|confirmed the diagnosis)\b",
        span_text,
        flags=re.IGNORECASE,
    ):
        score += 3.0
    if token_count >= 4:
        score += min(2.0, token_count / 8.0)
    if title_context_indicates_differentiation(span_text, context_text):
        score -= 3.0
    return score


def enforce_single_final_answer_claim(
    units: List[Dict[str, Any]],
    reasoning_text: str,
    row: Dict[str, Any],
) -> None:
    scored: List[Tuple[float, int, int]] = []
    for index, unit in enumerate(units):
        unit["is_final_answer_claim"] = False
        score = final_claim_candidate_score(unit, reasoning_text, row)
        if score >= 1.0:
            length = int(unit.get("char_end") or 0) - int(unit.get("char_start") or 0)
            scored.append((score, length, index))
    if not scored:
        return
    _score, _length, index = max(scored, key=lambda item: (item[0], item[1], -item[2]))
    units[index]["is_final_answer_claim"] = True


def truncate_for_prompt(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head
    return (
        text[:keep_head].rstrip()
        + "\n\n[...patient context truncated for span extraction...]\n\n"
        + text[-keep_tail:].lstrip()
    )


COVERAGE_CUE_PATTERN = re.compile(
    r"\b(?:was|were|is|are|be|been)\s+"
    r"(?:considered|suspected|excluded|ruled out|deemed unlikely|unlikely|less likely|supported|favou?red)"
    r"|\b(?:argues?|argued|supports?|supported|suggests?|suggested|rules?|ruled)\s+"
    r"(?:against|out|for)"
    r"|\bdifferential(?: diagnosis| diagnoses)?\b",
    flags=re.IGNORECASE,
)


def estimate_min_units(reasoning_text: str) -> int:
    text = normalize_text(reasoning_text)
    if not text:
        return 0
    numbered_items = len(re.findall(r"(?:^|\n)\s*\d+\.", text))
    sentence_breaks = len(re.findall(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])", text))
    cue_count = len(COVERAGE_CUE_PATTERN.findall(text))
    candidates = [4, len(text) // 140]
    if numbered_items:
        candidates.append(numbered_items * 3)
    if sentence_breaks >= 3:
        candidates.append(sentence_breaks * 2)
    if cue_count:
        candidates.append(cue_count + max(1, len(text) // 350))
    return min(36, max(candidates))


def coverage_issue(reasoning_text: str, units: List[Dict[str, Any]]) -> Tuple[bool, str, int]:
    expected_min = estimate_min_units(reasoning_text)
    if not reasoning_text:
        return False, "", expected_min
    accepted = len(units)
    if accepted == 0:
        return True, "no_accepted_units", expected_min
    longest_ratio = max((unit["char_end"] - unit["char_start"]) / max(len(reasoning_text), 1) for unit in units)
    if len(reasoning_text) >= 900 and longest_ratio >= 0.65 and accepted <= 3:
        return True, "single_or_giant_span", expected_min
    if expected_min >= 8 and accepted < max(4, int(expected_min * 0.75)):
        return True, "too_few_units_for_reasoning_complexity", expected_min
    return False, "", expected_min


WORKFLOW_STATEMENT_PATTERNS = (
    r"\bcardiology evaluation\b",
    r"\bpathology\b",
    r"\bpathology review\b",
    r"\bconsultation\b",
    r"\bworkup\b",
    r"\bevaluation\b",
)

WORKFLOW_RESULT_PATTERNS = (
    r"\bexcluded\b",
    r"\bruled out\b",
    r"\brules out\b",
    r"\bconsidered\b",
    r"\bsuspected\b",
    r"\bdemonstrated\b",
    r"\bshowed\b",
    r"\brevealed\b",
    r"\bconfirmed\b",
    r"\bdiagnostic of\b",
    r"\bdiagnostic for\b",
)

COMBINED_CLAUSE_PATTERNS = (
    re.compile(
        r"^(?P<feature>.+?)\s+(?P<cue>argues? against|argued against|supports|supported by|suggests?|suggested by|excludes?|excluded|rules? out|ruled out|is consistent with|are consistent with|is compatible with|are compatible with)\s+(?P<target>.+)$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<target>.+?)\s+(?P<cue>was considered|were considered|is considered|are considered|was excluded|were excluded|is excluded|are excluded|was ruled out|were ruled out|is ruled out|are ruled out|was suspected|were suspected|is suspected|are suspected|was less likely|were less likely|is less likely|are less likely|was unlikely|were unlikely|is unlikely|are unlikely|was highly unlikely|were highly unlikely|is highly unlikely|are highly unlikely|was very unlikely|were very unlikely|is very unlikely|are very unlikely|was listed as(?: a)? possible etiology of|were listed as(?: possible)? etiologies of|was in the differential diagnosis|were in the differential diagnosis|is in the differential diagnosis|are in the differential diagnosis|is in the differential diagnosis for|are in the differential diagnosis for|means|mean|indicates|indicate|implies|imply|reflects|reflect|was evaluated|were evaluated|is evaluated|are evaluated)\s*(?:because|given|due to|since|as|based on|with|from)?\s*(?P<feature>.+)$",
        flags=re.IGNORECASE,
    ),
)

FRAGMENTARY_OBSERVATION_PATTERNS = (
    r"^of\b",
    r"^part of\b",
    r"^but\b",
    r"^a differential diagnosis\b",
    r"^differential diagnoses\b",
    r"^the findings\b",
    r"^the immunohistochemical profile\b",
    r"^the clinical presentation\b",
    r"^the clinical picture\b",
    r"^the possibility(?: of)?\b",
    r"^in the differential diagnosis(?: of| for)?\b",
    r"^the differential diagnosis(?: of| for)?\b",
    r"^a(?: [a-z-]+){0,4} cause of\b",
    r"^a(?: [a-z-]+){0,4} possible etiology of\b",
    r"\b(?:was|were|is|are)\.?$",
)

EVALUATION_ONLY_PATTERNS = (
    r"\bevaluated\b",
    r"\bassessed\b",
    r"\breviewed\b",
    r"\bworkup\b",
    r"\bworked up\b",
)

GENERIC_DIAGNOSIS_REFERENTS = {
    "diagnosis",
    "the diagnosis",
    "this diagnosis",
    "that diagnosis",
    "other causes",
    "other cause",
    "alternative diagnoses",
    "alternative diagnosis",
}

SIMPLE_DIFFERENTIAL_CLAUSE_PATTERN = re.compile(
    r"^(?P<targets>.+?)\s+(?P<cue>was considered|were considered|is considered|are considered|was suspected|were suspected|is suspected|are suspected|was excluded|were excluded|is excluded|are excluded|was ruled out|were ruled out|is ruled out|are ruled out|was less likely|were less likely|is less likely|are less likely|was unlikely|were unlikely|is unlikely|are unlikely|was supported|were supported|is supported|are supported)\.?\s*$",
    flags=re.IGNORECASE,
)

DIAGNOSIS_LIKE_PATTERNS = (
    r"\b(?:disease|syndrome|infection|exacerbation|carcinoma|cancer|tumou?r|neoplasm|malignancy|adenoma|cyst|abscess|embolism|thrombosis|infarction|lymphoma|leukemia|melanoma|schwannoma|neurilemmoma|varicocele|syphilis|tuberculosis|myelopathy|neuropathy|vasculitis|sclerosis|sclerosus|pneumonia|bronchiolitis|colitis|enteritis|myelitis|encephalitis|cardiomyopathy|cholangitis|cholecystitis|osteomyelitis|hypophysitis|amyloidosis|fibromatosis|fibroma|lipoma|hemangioma|osteoma|myxoma|palsy|pseudotumou?r|appendicitis|diverticulitis|peritonitis|pneumonitis|nephritis|myositis|dermatitis|psoriasis|eczema|histiocytosis|granulomatosis|scleroderma|lupus|myasthenia|fibrosis|malformation|stenosis|atresia|obstruction|torsion|ulcer|pruritus)\b",
)


def is_generic_diagnosis_target(text: str) -> bool:
    cleaned = clean_query_text(text).lower()
    return bool(cleaned) and cleaned in GENERIC_DIAGNOSIS_REFERENTS


def strip_differential_query_tail(text: str) -> str:
    cleaned = clean_query_text(text)
    for pattern in (
        r"\s+\bbut\s+deemed\b.*$",
        r"\s+\bbut\s+(?:(?:was|were|is|are)\s+)?(?:deemed\s+)?(?:unlikely|less likely|ruled out|excluded)\b.*$",
        r"\s+\b(?:was|were|is|are)\s+(?:considered|suspected|evaluated|entertained|listed|deemed unlikely|unlikely|less likely|excluded|ruled out)\b.*$",
        r"\s+\b(?:considered|suspected|evaluated|entertained|listed|excluded|ruled out)\b.*$",
    ):
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" ;,.-")
    return cleaned


def looks_like_diagnosis_target(text: str) -> bool:
    cleaned = clean_query_text(text).lower()
    if not cleaned or is_generic_diagnosis_target(cleaned):
        return False
    if re.search(r"\b(?:connections?|features?|findings?|criteria|changes|signals?|uptake|enhancement|calcifications?|nodules?|lesions?|defects?|masses?|imaging|ultrasound|mri|ct|hrct|angiography)\b", cleaned):
        return False
    return any(re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in DIAGNOSIS_LIKE_PATTERNS)


def unique_texts(items: List[str]) -> List[str]:
    seen = set()
    values: List[str] = []
    for item in items:
        cleaned = clean_query_text(item)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        values.append(cleaned)
    return values


def split_conjoined_targets(text: str) -> List[str]:
    cleaned = clean_query_text(text)
    if not cleaned:
        return []
    parts = re.split(r"\s+(?:or|and)\s+", cleaned)
    if len(parts) <= 1:
        return [cleaned]
    targets: List[str] = []
    for part in parts:
        candidate = extract_target_like_phrase(part) or clean_query_text(part)
        candidate = clean_query_text(candidate)
        if candidate:
            targets.append(candidate)
    targets = unique_texts(targets)
    if len(targets) < 2:
        return [cleaned]
    return targets


def is_generic_differential_statement(text: str) -> bool:
    cleaned = clean_query_text(text).lower()
    if not cleaned:
        return False
    return bool(
        re.fullmatch(
            r"(?:(?:this|the|that)\s+diagnosis|(?:this|the|that)\s+diagnoses|other causes?|alternative diagnoses?|alternative diagnosis)\s+"
            r"(?:was|were|is|are)?\s*"
            r"(?:considered|excluded|ruled out|suspected|evaluated|listed|favored|favoured|less likely|unlikely)\b.*",
            cleaned,
            flags=re.IGNORECASE,
        )
    )


def is_non_diagnosis_d_target(text: str) -> bool:
    cleaned = clean_query_text(text).lower()
    if not cleaned:
        return False
    non_diagnosis_patterns = (
        r"\bconnections? between\b",
        r"\bdistribution of\b",
        r"\bpattern of\b",
        r"\b(?:radiographic|imaging|mri|ct|ultrasound|pet/?ct)\s+features\b",
    )
    diagnosis_markers = (
        r"\b(disease|syndrome|cancer|carcinoma|melanoma|lymphoma|tumou?r|infection|malignancy|metastasis|embolism|myelopathy|tuberculosis|bronchiolitis|cholecystitis|cholangitis|diarrhea|cyst|neoplasm|hypophysitis|lymphangioma|varicocele|arthritis)\b",
    )
    return any(re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in non_diagnosis_patterns) and not any(
        re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in diagnosis_markers
    )


def split_simple_differential_targets(text: str) -> Tuple[List[str], str]:
    cleaned = clean_query_text(text)
    match = SIMPLE_DIFFERENTIAL_CLAUSE_PATTERN.match(cleaned)
    if not match:
        return [], ""
    targets_text = clean_query_text(match.group("targets"))
    cue = clean_query_text(match.group("cue"))
    lowered = targets_text.lower()
    if any(
        marker in lowered
        for marker in (" because ", " given ", " due to ", " with ", " from ", " based on ", " since ", " as ", " such as ", " including ", " between ")
    ):
        return [], ""
    separator = None
    if " or " in lowered:
        separator = r"\s+or\s+"
    elif " and " in lowered:
        separator = r"\s+and\s+"
    if separator is None:
        return [], ""
    raw_parts = [clean_query_text(part) for part in re.split(separator, targets_text) if clean_query_text(part)]
    if not (2 <= len(raw_parts) <= 3):
        return [], ""
    parts: List[str] = []
    for part in raw_parts:
        if is_generic_diagnosis_target(part) or is_non_diagnosis_d_target(part):
            return [], ""
        if len(part.split()) > 8:
            return [], ""
        parts.append(part)
    return parts, cue


def is_workflow_statement(text: str) -> bool:
    cleaned = clean_query_text(text).lower()
    if not cleaned:
        return False
    if not any(re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in WORKFLOW_STATEMENT_PATTERNS):
        return False
    return any(re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in WORKFLOW_RESULT_PATTERNS)


def is_fragmentary_observation(text: str) -> bool:
    cleaned = clean_query_text(text)
    lowered = cleaned.lower()
    if not lowered:
        return False
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in FRAGMENTARY_OBSERVATION_PATTERNS)


def is_evaluation_only_statement(text: str) -> bool:
    cleaned = clean_query_text(text).lower()
    if not cleaned:
        return False
    if not any(re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in EVALUATION_ONLY_PATTERNS):
        return False
    return not any(
        re.search(pattern, cleaned, flags=re.IGNORECASE)
        for pattern in (
            r"\bexcluded\b",
            r"\bruled out\b",
            r"\brules out\b",
            r"\bargues? against\b",
            r"\bsupports?\b",
            r"\bsuggests?\b",
            r"\bconsidered\b",
            r"\bsuspected\b",
        )
    )


def soften_query_phrasing(text: str) -> str:
    softened = text
    replacements = (
        (r"\bdiagnostic of\b", ""),
        (r"\bdiagnostic for\b", ""),
        (r"\bcan present as\b", ""),
        (r"\bcan present with\b", ""),
        (r"\bmay present as\b", ""),
        (r"\bmay present with\b", ""),
        (r"\bcan cause\b", ""),
        (r"\bmay cause\b", ""),
        (r"\bis associated with\b", ""),
        (r"\bare associated with\b", ""),
        (r"\bis linked to\b", ""),
        (r"\bare linked to\b", ""),
        (r"\bcannot be excluded\b", ""),
        (r"\bcannot exclude\b", ""),
        (r"\bcan be excluded\b", ""),
        (r"\bwas less likely\b", ""),
        (r"\bwere less likely\b", ""),
        (r"\bis less likely\b", ""),
        (r"\bare less likely\b", ""),
        (r"\bwas unlikely\b", ""),
        (r"\bwere unlikely\b", ""),
        (r"\bis unlikely\b", ""),
        (r"\bare unlikely\b", ""),
        (r"\bcannot be excluded\b", ""),
        (r"\bcannot exclude\b", ""),
        (r"\bargues? against\b", ""),
        (r"\bsupports?\b", ""),
        (r"\bsupported by\b", ""),
        (r"\bsuggests?\b", ""),
        (r"\bsuggested by\b", ""),
        (r"\bis consistent with\b", ""),
        (r"\bare consistent with\b", ""),
        (r"\bis compatible with\b", ""),
        (r"\bare compatible with\b", ""),
        (r"\bis in the differential diagnosis(?: of| for)?\b", ""),
        (r"\bare in the differential diagnosis(?: of| for)?\b", ""),
        (r"\bwas considered\b", ""),
        (r"\bwere considered\b", ""),
        (r"\bwas excluded\b", ""),
        (r"\bwere excluded\b", ""),
        (r"\bwas ruled out\b", ""),
        (r"\bwere ruled out\b", ""),
        (r"\bwas suspected\b", ""),
        (r"\bwere suspected\b", ""),
        (r"\bwas evaluated\b", ""),
        (r"\bwere evaluated\b", ""),
        (r"\bwas assessed\b", ""),
        (r"\bwere assessed\b", ""),
        (r"\bthis patient\b", ""),
        (r"\bthe patient\b", ""),
        (r"\bin this patient\b", ""),
        (r"\bpatient\b", ""),
        (r"\bthey\b", ""),
        (r"\bit\b", ""),
    )
    for pattern, replacement in replacements:
        softened = re.sub(pattern, replacement, softened, flags=re.IGNORECASE)
    softened = re.sub(r"\s+", " ", softened)
    return softened.strip(" ;,.-")


def extract_target_like_phrase(text: str) -> str:
    cleaned = clean_query_text(text)
    if not cleaned:
        return ""
    for pattern in COMBINED_CLAUSE_PATTERNS:
        match = pattern.match(cleaned)
        if not match:
            continue
        target = clean_query_text(match.groupdict().get("target", ""))
        if target and target.lower() not in GENERIC_DIAGNOSIS_REFERENTS:
            return target
    lowered = cleaned.lower()
    prefixes = (
        "because of ",
        "because ",
        "due to ",
        "given ",
        "since ",
        "as ",
        "findings that argue against ",
        "findings arguing against ",
        "findings that support ",
        "findings supporting ",
        "evidence that argues against ",
        "evidence arguing against ",
        "the possibility of ",
        "the possibility ",
        "the clinical presentation of ",
        "the clinical presentation ",
        "in the differential diagnosis of ",
        "in the differential diagnosis for ",
        "in the differential diagnosis ",
        "the differential diagnosis of ",
        "the differential diagnosis for ",
        "the differential diagnosis ",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            lowered = cleaned.lower()
            break
    split_markers = (
        r"\bwas less likely\b",
        r"\bwere less likely\b",
        r"\bis less likely\b",
        r"\bare less likely\b",
        r"\bless likely\b",
        r"\bwas highly unlikely\b",
        r"\bwere highly unlikely\b",
        r"\bis highly unlikely\b",
        r"\bare highly unlikely\b",
        r"\bhighly unlikely\b",
        r"\bwas very unlikely\b",
        r"\bwere very unlikely\b",
        r"\bis very unlikely\b",
        r"\bare very unlikely\b",
        r"\bvery unlikely\b",
        r"\bwas unlikely\b",
        r"\bwere unlikely\b",
        r"\bis unlikely\b",
        r"\bare unlikely\b",
        r"\bunlikely\b",
        r"\bwas considered\b",
        r"\bwere considered\b",
        r"\bwas excluded\b",
        r"\bwere excluded\b",
        r"\bwas ruled out\b",
        r"\bwere ruled out\b",
        r"\bwas suspected\b",
        r"\bwere suspected\b",
        r"\bwas evaluated\b",
        r"\bwere evaluated\b",
        r"\bmeans\b",
        r"\bmean\b",
        r"\bindicates\b",
        r"\bindicate\b",
        r"\bimplies\b",
        r"\bimply\b",
        r"\breflects\b",
        r"\breflect\b",
        r"\bargues? against\b",
        r"\bsupports?\b",
        r"\bsupported by\b",
        r"\bsuggests?\b",
        r"\bsuggested by\b",
        r"\bis consistent with\b",
        r"\bare consistent with\b",
        r"\bis compatible with\b",
        r"\bare compatible with\b",
        r"\bpossible etiology of\b",
        r"\betiology of\b",
        r"\bcause of\b",
        r"\bas cause of\b",
    )
    for marker in split_markers:
        parts = re.split(marker, cleaned, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1 and parts[0].strip():
            cleaned = parts[0].strip()
            break
    cleaned = cleaned.strip(" ;,.-")
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,.-")
    if " and " in cleaned and len(cleaned.split()) <= 8:
        cleaned = cleaned.split(" and ", 1)[0].strip()
    elif " or " in cleaned and len(cleaned.split()) <= 8:
        cleaned = cleaned.split(" or ", 1)[0].strip()
    elif " in " in cleaned and len(cleaned.split()) <= 8:
        head, tail = cleaned.split(" in ", 1)
        if head.strip() and tail.strip():
            cleaned = head.strip()
    if is_generic_diagnosis_target(cleaned):
        return ""
    return cleaned


def build_retrieval_seed(text: str, unit_type: str) -> str:
    cleaned = clean_query_text(text)
    if not cleaned:
        return ""
    for pattern in COMBINED_CLAUSE_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            feature = clean_query_text(match.groupdict().get("feature", ""))
            target = clean_query_text(match.groupdict().get("target", ""))
            cue = clean_query_text(match.groupdict().get("cue", ""))
            target_only = extract_target_like_phrase(target)
            if target_only and looks_like_diagnosis_target(target_only):
                target = target_only
            stripped_target = strip_differential_query_tail(target)
            if stripped_target and looks_like_diagnosis_target(stripped_target):
                target = stripped_target
            if re.fullmatch(
                r"(?:but\s+)?(?:(?:was|were|is|are)\s+)?(?:deemed\s+)?(?:unlikely|less likely|excluded|ruled out)\b.*",
                feature,
                flags=re.IGNORECASE,
            ):
                feature = ""
            if target and feature:
                if cue and re.search(r"\b(excluded|ruled out|argue|support|suggest|consistent|compatible|diagnostic|differential diagnosis|considered|suspected)\b", cue, flags=re.IGNORECASE):
                    return f"{target} {feature}"
            if target:
                return target
    if unit_type == "D":
        cleaned = re.sub(
            r"^\s*the differential diagnosis (?:may )?(?:also )?include\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        target = extract_target_like_phrase(cleaned)
        if target and looks_like_diagnosis_target(target):
            stripped_target = strip_differential_query_tail(target)
            return stripped_target or target
        stripped_cleaned = strip_differential_query_tail(cleaned)
        if stripped_cleaned and looks_like_diagnosis_target(stripped_cleaned):
            return stripped_cleaned
    return cleaned


def default_retrieval_query(text: str, unit_type: str) -> str:
    if unit_type not in {"W", "D"}:
        return ""
    if is_workflow_statement(text):
        return ""
    cleaned = build_retrieval_seed(text, unit_type)
    cleaned = soften_query_phrasing(cleaned)
    return cleaned


def observation_from_feature(feature: str) -> str:
    cleaned = clean_query_text(feature)
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if lowered.startswith("because "):
        cleaned = cleaned[8:].strip()
    elif lowered.startswith("given "):
        cleaned = cleaned[6:].strip()
    elif lowered.startswith("due to "):
        cleaned = cleaned[7:].strip()
    elif lowered.startswith("since "):
        cleaned = cleaned[6:].strip()
    elif lowered.startswith("as "):
        cleaned = cleaned[3:].strip()
    elif lowered.startswith("based on "):
        cleaned = cleaned[9:].strip()
    elif lowered.startswith("for "):
        cleaned = cleaned[4:].strip()
    elif lowered.startswith("the patient had "):
        cleaned = cleaned[16:].strip()
    elif lowered.startswith("patient had "):
        cleaned = cleaned[12:].strip()
    elif lowered.startswith("had "):
        cleaned = cleaned[4:].strip()
    lowered = cleaned.lower()
    if lowered.startswith("absence of "):
        cleaned = f"No {cleaned[len('absence of '):].strip()}"
    elif lowered.startswith("lack of "):
        cleaned = f"No {cleaned[len('lack of '):].strip()}"
    elif lowered.startswith("no "):
        cleaned = cleaned
    elif lowered.startswith("negative "):
        cleaned = cleaned
    elif lowered.startswith("without "):
        cleaned = cleaned
    elif lowered.startswith("normal "):
        cleaned = cleaned
    elif lowered.startswith("denied "):
        cleaned = cleaned
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,.-")
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned and cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def is_observation_like_feature(feature: str) -> bool:
    cleaned = clean_query_text(feature)
    lowered = cleaned.lower()
    if not lowered:
        return False
    if is_fragmentary_observation(cleaned):
        return False
    if lowered.startswith(("it ", "they ", "this ", "these ", "those ", "such ")):
        return False
    if re.search(
        r"\b(cannot be excluded|cannot exclude|less likely|unlikely|excluded|ruled out|considered|suspected|evaluated)\b",
        lowered,
    ):
        return False
    if re.search(r"\b(can|may|might|typically|usually|often|classically)\b", lowered):
        return False
    general_rule_markers = (
        "present with",
        "present as",
        "associated with",
        "linked to",
        "linked with",
        "characteristic of",
        "diagnostic of",
        "diagnostic for",
        "consistent with",
        "compatible with",
        "caused by",
        "seen in",
        "seen with",
        "mimic",
        "results in",
        "defined by",
        "characterized by",
    )
    if any(marker in lowered for marker in general_rule_markers):
        return False
    return True


def subject_is_plural(subject: str) -> bool:
    lowered = clean_query_text(subject).lower()
    if not lowered:
        return False
    plural_markers = (
        "causes",
        "findings",
        "symptoms",
        "criteria",
        "lesions",
        "nodes",
        "diagnoses",
        "diseases",
        "results",
        "labs",
        "signs",
        "features",
        "abnormalities",
        "tests",
    )
    if any(marker in lowered for marker in plural_markers):
        return True
    if lowered.endswith("s") and not lowered.endswith(("is", "us", "ss")):
        return True
    return False


def claim_from_cue(target: str, cue: str, original_text: str) -> str:
    target_clean = clean_query_text(target)
    cue_clean = clean_query_text(cue).lower()
    if not target_clean:
        return clean_query_text(original_text)
    verb = "were" if subject_is_plural(target_clean) else "was"
    if "argue" in cue_clean:
        return f"{target_clean} {verb} less likely."
    if "exclude" in cue_clean or "rule out" in cue_clean:
        return f"{target_clean} {verb} excluded."
    if "likely" in cue_clean or "unlikely" in cue_clean:
        return f"{target_clean} {verb} less likely."
    if "consider" in cue_clean or "in the differential diagnosis" in cue_clean:
        return f"{target_clean} {verb} considered."
    if "suspect" in cue_clean:
        return f"{target_clean} {verb} suspected."
    if "support" in cue_clean:
        return f"{target_clean} {verb} supported."
    if "suggest" in cue_clean or "consistent with" in cue_clean or "compatible with" in cue_clean:
        return f"{target_clean} {verb} supported by the findings."
    if "mean" in cue_clean or "indicate" in cue_clean or "imply" in cue_clean or "reflect" in cue_clean:
        return f"{target_clean}."
    if "diagnostic" in cue_clean:
        return f"{target_clean} {verb} associated with the findings."
    return clean_query_text(original_text)


def clean_nli_claim_text(text: Any, fallback: str) -> str:
    cleaned = normalize_text(text).strip(" \"'")
    if cleaned.lower() in {"none", "null", "n/a", "na"}:
        cleaned = ""
    if not cleaned:
        cleaned = normalize_text(fallback).strip(" \"'")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > 600:
        fallback_clean = normalize_text(fallback).strip()
        cleaned = fallback_clean if fallback_clean else cleaned[:600].rstrip()
    return cleaned


def default_nli_claim_text(span_text: str, unit_type: str) -> str:
    cleaned = normalize_text(span_text)
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if unit_type == "O" and lowered.startswith(
        ("because ", "given ", "due to ", "since ", "as ", "based on ", "for ")
    ):
        observation_text = observation_from_feature(cleaned)
        if observation_text:
            return observation_text
    if unit_type in {"D", "C"}:
        for pattern in COMBINED_CLAUSE_PATTERNS:
            match = pattern.match(cleaned)
            if not match:
                continue
            target = clean_query_text(match.groupdict().get("target", ""))
            cue = clean_query_text(match.groupdict().get("cue", ""))
            if target and cue:
                proposition = claim_from_cue(target, cue, cleaned)
                if proposition:
                    return proposition
    return cleaned


def nli_claim_text_from_item(item: Dict[str, Any], span_text: str, unit_type: str) -> str:
    candidate = item.get("nli_claim_text")
    if candidate is None:
        candidate = item.get("claim_text")
    if candidate is None:
        candidate = item.get("atomic_proposition")
    if candidate is None:
        candidate = item.get("proposition")
    return clean_nli_claim_text(candidate, default_nli_claim_text(span_text, unit_type))


DIAGNOSIS_DIRECTION_PATTERN = re.compile(
    r"\b(?:"
    r"differential diagnosis|possible diagnos(?:is|es)|alternative diagnos(?:is|es)|"
    r"was considered|were considered|is considered|are considered|"
    r"was suspected|were suspected|is suspected|are suspected|"
    r"suspicion of|suspected by|"
    r"was excluded|were excluded|is excluded|are excluded|"
    r"was ruled out|were ruled out|is ruled out|are ruled out|"
    r"ruled out|excluded|excludes|exclude(?:s|d)?|"
    r"was less likely|were less likely|is less likely|are less likely|"
    r"was unlikely|were unlikely|is unlikely|are unlikely|"
    r"deemed unlikely|made .{0,120} unlikely|made .{0,120} less likely|diagnosis unlikely|"
    r"argues? against|argued against|"
    r"did not support|does not support|not support|"
    r"favou?red|strongly favou?red|leading diagnosis|"
    r"support(?:ed|s)? the diagnosis|supported by|supports?|"
    r"suggest(?:ed|s)? by|"
    r"mimic(?:s|ked)? .{0,120}(?:diagnos|malignancy|infection|metasta|sarcoma|carcinoma|tumou?r|appendicitis|causes? of)|"
    r"exclude .{0,120}(?:malignant|malignancy|sarcoma|carcinoma|neuroblastoma|rhabdomyosarcoma)"
    r")\b",
    flags=re.IGNORECASE,
)

GENERAL_MEDICAL_RULE_PATTERN = re.compile(
    r"\b(?:"
    r"can|may|might|often|typically|usually|classically|"
    r"characteristic of|diagnostic of|diagnostic for|associated with|linked to|linked with|"
    r"defined by|characterized by|presents? with|presents? as|seen in|seen with|"
    r"possible diagnoses include|possible diagnoses, including|"
    r"should be correlated|gold standard|investigation of choice"
    r")\b",
    flags=re.IGNORECASE,
)

CASE_OBSERVATION_PATTERN = re.compile(
    r"\b(?:"
    r"patient (?:had|has|presented|showed|was|were)|"
    r"(?:she|he|they) (?:had|has|presented|showed|was|were|reported|denied)|"
    r"no (?:[a-z0-9-]+ ){0,5}(?:pruritus|pain|fever|weakness|numbness|metastases|proteinuria|lesions?|masses?|allergic signs?)|"
    r"history of|laboratory|lab(?:oratory)? evaluation|"
    r"ct|mri|ultrasound|radiograph|x-?ray|imaging|scan|"
    r"patholog(?:y|ical)|histolog(?:y|ical)|biopsy|culture|"
    r"showed|revealed|demonstrated|found|identified|detected|"
    r"intraoperative|preoperative|postoperative|"
    r"no metastases|no pathology|absence of|lack of|"
    r"[A-Z][A-Za-z0-9-]*\s*\d+(?:\.\d+)?\s*[%/()]"
    r")\b",
    flags=re.IGNORECASE,
)

CLAIM_CONFIRMATION_PATTERN = re.compile(
    r"\b(?:"
    r"final diagnosis|correct diagnosis|confirmed the diagnosis|diagnosis was confirmed|"
    r"was diagnosed|were diagnosed|diagnosis was made|diagnosis of|answer is|"
    r"findings confirmed|which confirmed"
    r")\b",
    flags=re.IGNORECASE,
)


def row_final_answer(row: Dict[str, Any]) -> str:
    return normalize_text(
        row.get("negative_final_answer")
        or row.get("predicted_answer_text")
        or row.get("final_answer")
        or row.get("correct_answer_text")
    )


def final_answer_title_like(span_text: str, final_answer: str) -> bool:
    cleaned = normalize_text(span_text).strip(" \"'—–-:;,.")
    answer = normalize_text(final_answer).strip(" \"'—–-:;,.")
    if not cleaned or not answer:
        return False
    return cleaned.lower() == answer.lower()


def has_diagnosis_direction(text: str) -> bool:
    cleaned = normalize_text(text)
    if not cleaned:
        return False
    if DIAGNOSIS_DIRECTION_PATTERN.search(cleaned):
        return True
    if SIMPLE_DIFFERENTIAL_CLAUSE_PATTERN.match(clean_query_text(cleaned)):
        return True
    target = extract_target_like_phrase(cleaned)
    return bool(target and looks_like_diagnosis_target(target) and re.search(
        r"\b(?:suggests?|supports?|argues?|exclude|excluded|unlikely|suspected|considered|differential)\b",
        cleaned,
        flags=re.IGNORECASE,
    ))


def has_case_observation_signal(text: str) -> bool:
    cleaned = normalize_text(text)
    if not cleaned:
        return False
    return bool(CASE_OBSERVATION_PATTERN.search(cleaned))


def has_general_rule_signal(text: str) -> bool:
    cleaned = normalize_text(text)
    if not cleaned:
        return False
    return bool(GENERAL_MEDICAL_RULE_PATTERN.search(cleaned))


def evidence_based_case_diagnosis(span_text: str) -> bool:
    text = normalize_text(span_text).lower()
    return bool(
        re.search(r"^(?:given|based on|because of|because|with|for)\b", text)
        and re.search(r"\b(?:patient|history|findings|imaging|radiological|microbiology|clinical|laboratory|patholog)\b", text)
        and re.search(r"\b(?:diagnosis of|diagnosis was made|diagnosis was kept|was diagnosed)\b", text)
    )


def context_after_span(span_text: str, context_text: str) -> str:
    span = normalize_text(span_text)
    context = normalize_text(context_text)
    if not span or not context:
        return ""
    idx = context.find(span)
    if idx < 0:
        return context[:180]
    return context[idx + len(span) : idx + len(span) + 180]


def context_before_span(span_text: str, context_text: str) -> str:
    span = normalize_text(span_text)
    context = normalize_text(context_text)
    if not span or not context:
        return ""
    idx = context.find(span)
    if idx < 0:
        return context[-180:]
    return context[max(0, idx - 180) : idx]


def title_context_has_final_confirmation(span_text: str, context_text: str) -> bool:
    before = context_before_span(span_text, context_text).lower()
    after = context_after_span(span_text, context_text).lower()
    final_before = bool(
        re.search(
            r"\b(?:final|correct|confirmed|definitive|established|made)\s+(?:diagnosis|answer)\b|"
            r"\bdiagnosis\s+(?:was|is)\s+(?:confirmed|made|established)\b",
            before,
            flags=re.IGNORECASE,
        )
    )
    final_after = bool(
        re.search(
            r"^[\s:;,\-.—–]*(?:was|is)?\s*(?:the\s+)?(?:final|correct|confirmed|definitive)\s+"
            r"(?:diagnosis|answer)\b|"
            r"\b(?:confirmed|established)\s+(?:the\s+)?diagnosis\b",
            after,
            flags=re.IGNORECASE,
        )
    )
    return final_before or final_after


def title_context_has_heading_rationale(span_text: str, context_text: str) -> bool:
    after = context_after_span(span_text, context_text).lower()
    return bool(
        re.search(
            r"^[\s:;,\-.—–]*(?:because|due to|given|with|without|absence|lack|the|it|this|these|"
            r"supported|suggested|favou?red|consistent|rather than|instead of|argued|excluded|ruled out|"
            r"unlikely|less likely|considered|suspected)\b",
            after,
            flags=re.IGNORECASE,
        )
    )


def title_context_indicates_differentiation(span_text: str, context_text: str) -> bool:
    after = context_after_span(span_text, context_text).lower()
    before = context_before_span(span_text, context_text).lower()
    if not after and not before:
        return False
    if title_context_has_final_confirmation(span_text, context_text):
        return False
    after_indicates_differentiation = bool(
        re.search(
            r"^[\s:;,\-.—–]*(?:"
            r"considered|included|suspected|possible|strongly favou?red|favou?red|leading diagnosis|"
            r"supported|supported by|argued against|excluded|ruled out|less likely|unlikely|due to|because|given|"
            r"with|without|absence|lack|the|it|this|these|consistent|rather than|instead of"
            r")\b",
            after,
            flags=re.IGNORECASE,
        )
    )
    before_indicates_differentiation = bool(
        re.search(
            r"\b(?:in favor of|favou?r(?:ed|ing)|supported(?: the)? diagnosis of|suggested(?: the)? diagnosis of|"
            r"considered|included|suspected|possible|differential diagnosis of|diagnosis includes?)\s*$",
            before,
            flags=re.IGNORECASE,
        )
    )
    return after_indicates_differentiation or before_indicates_differentiation


def dash_span_indicates_differentiation(span_text: str) -> bool:
    parts = re.split(r"\s+[—–-]\s+", normalize_text(span_text), maxsplit=1)
    if len(parts) != 2:
        return False
    head, tail = parts[0].strip(), parts[1].strip()
    if not head or not tail:
        return False
    if not looks_like_diagnosis_target(head):
        return False
    return bool(
        has_diagnosis_direction(tail)
        or has_case_observation_signal(tail)
        or has_general_rule_signal(tail)
        or re.search(r"\b(?:consistent with|rather than|instead of|mimics?|suggested by|favou?red by)\b", tail, flags=re.IGNORECASE)
    )


def explicit_diagnosis_decision_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:considered|suspected|excluded|ruled out|unlikely|less likely|deemed unlikely|made unlikely|"
            r"favou?red|supported|argued against|not satisfied|did not support|differential diagnosis|differentials?)\b",
            normalize_text(text),
            flags=re.IGNORECASE,
        )
    )


def coerce_role_type(
    unit_type: str,
    span_text: str,
    nli_claim_text: str,
    row: Dict[str, Any],
    context_text: str = "",
) -> str:
    span = normalize_text(span_text)
    claim = normalize_text(nli_claim_text)
    direction_text = span if has_case_observation_signal(span) else f"{span} {claim}".strip()
    final_answer = row_final_answer(row)

    if evidence_based_case_diagnosis(span):
        return "O"
    if final_answer_title_like(span, final_answer):
        if title_context_has_final_confirmation(span, context_text):
            return "C"
        return "D"
    if dash_span_indicates_differentiation(span):
        return "D"
    if re.search(r"\bpreoperative diagnosis\b", span, flags=re.IGNORECASE) and not re.search(
        r"\b(?:clarify|exclude|excluded|malignant|malignancy|suspected|considered|differential)\b",
        span,
        flags=re.IGNORECASE,
    ):
        return "O"
    if final_answer and final_answer_is_mentioned(span, final_answer) and not re.search(
        r"\b(?:suspected|considered|possible|differential|ruled out|excluded|less likely|unlikely|argues? against|not support|did not support)\b",
        span,
        flags=re.IGNORECASE,
    ) and re.search(
        r"\b(?:diagnosed|diagnosis was made|confirmed the diagnosis|diagnosis was confirmed|correct diagnosis|final diagnosis)\b",
        span,
        flags=re.IGNORECASE,
    ):
        return "C"
    if CLAIM_CONFIRMATION_PATTERN.search(span) and not has_diagnosis_direction(span):
        if re.search(r"\b(?:final diagnosis|correct diagnosis|answer is)\b", span, flags=re.IGNORECASE):
            return "C"
        if final_answer and final_answer_is_mentioned(span, final_answer) and not context_indicates_nonfinal_claim(
            span, context_text
        ):
            return "C"
        if has_case_observation_signal(span):
            return "O"
        return "D"
    if has_general_rule_signal(span) and not explicit_diagnosis_decision_signal(span):
        return "W"
    if has_diagnosis_direction(direction_text):
        return "D"
    if has_case_observation_signal(span) and not has_general_rule_signal(span):
        return "O"
    if has_general_rule_signal(span):
        return "W"
    return unit_type


ARU_TRIM_LEFT_CHARS = " \t\r\n\"'“”‘’:;,-–—"
ARU_TRIM_RIGHT_CHARS = " \t\r\n\"'“”‘’.,;:-–—"
DASH_SEPARATOR_PATTERN = re.compile(r"\s+[—–]\s+|\s+-\s+")
QUOTE_CONTENT_PATTERNS = (
    re.compile(r'"(?P<inner>[^"]{3,800})"'),
    re.compile(r"“(?P<inner>[^”]{3,800})”"),
)


def trim_aru_span(text: str, start: int, end: int) -> Tuple[int, int]:
    start, end = trim_span(text, start, end)
    if start >= end:
        return start, end
    prefix = re.match(r"\d+\.\s*", text[start:end])
    if prefix:
        start += prefix.end()
    while start < end and text[start] in ARU_TRIM_LEFT_CHARS:
        start += 1
    while end > start and text[end - 1] in ARU_TRIM_RIGHT_CHARS:
        end -= 1
    return trim_span(text, start, end)


def useful_refined_span(
    reasoning_text: str,
    start: int,
    end: int,
    unit_type: str,
    row: Dict[str, Any],
) -> bool:
    if not valid_span_word_boundaries(reasoning_text, start, end):
        return False
    span = normalize_text(reasoning_text[start:end])
    if len(span) < 3 or not re.search(r"[A-Za-z0-9]", span):
        return False
    if is_fragmentary_nli_span(span, unit_type):
        return False
    lowered = span.lower().strip(" .,:;")
    if lowered in {
        "because",
        "due to",
        "given",
        "since",
        "as",
        "with",
        "but",
        "however",
        "therefore",
        "this diagnosis",
        "the diagnosis",
    }:
        return False
    tokens = re.findall(r"[A-Za-z0-9]+", span)
    final_answer = row_final_answer(row)
    if final_answer_title_like(span, final_answer):
        context_text = reasoning_text[max(0, start - 120) : min(len(reasoning_text), end + 180)]
        before = reasoning_text[max(0, start - 24) : start]
        after = reasoning_text[end : min(len(reasoning_text), end + 24)]
        standalone_heading = bool(
            re.search(r"(?:^|\n)\s*(?:\d+\.\s*)?$", before)
            or re.match(r"^[\s:;,\-.—–]+", after)
        )
        if not standalone_heading and not title_context_has_final_confirmation(span, context_text):
            return False
    if len(tokens) == 1 and unit_type not in {"D", "C"}:
        return False
    if len(tokens) == 1:
        return len(tokens[0]) >= 5 or final_answer_title_like(span, row_final_answer(row))
    return True


def is_fragmentary_nli_span(span_text: str, unit_type: str = "") -> bool:
    """Reject connector-only fragments that are not independently checkable."""
    span = normalize_text(span_text)
    if not span:
        return True
    lowered = span.lower().strip(" .,:;")
    tokens = re.findall(r"[A-Za-z0-9]+", span)
    if not tokens:
        return True
    if lowered in {
        "excluded when",
        "excluded because",
        "confirmed by",
        "supported by",
        "and immunoprofile",
        "and imaging showed",
        "and no histologic lichenoid changes",
        "due to rapid enlargement",
        "given reported associations",
    }:
        return True
    if re.match(r"^(?:and|or|but|because|given|due to|as|since|when|after|by)\b", lowered) and len(tokens) <= 8:
        return True
    if re.match(
        r"^(?:it|this|these|they)\s+(?:was|were)\s+(?:excluded|ruled out|considered|deemed unlikely|made unlikely)$",
        lowered,
    ):
        return True
    if re.match(
        r"^(?:was|were|is|are)?\s*(?:excluded|ruled out|confirmed|supported|considered)\s+"
        r"(?:when|because|by|due to|given|after)?$",
        lowered,
    ):
        return True
    if len(tokens) <= 4 and unit_type == "O" and not has_case_observation_signal(span):
        return True
    if len(tokens) <= 2 and unit_type == "W" and not has_general_rule_signal(span):
        return True
    return False


def find_quoted_inner_ranges(
    reasoning_text: str,
    start: int,
    end: int,
) -> List[Tuple[int, int, int, int]]:
    region = reasoning_text[start:end]
    ranges: List[Tuple[int, int, int, int]] = []
    for pattern in QUOTE_CONTENT_PATTERNS:
        for match in pattern.finditer(region):
            inner_start, inner_end = trim_aru_span(
                reasoning_text,
                start + match.start("inner"),
                start + match.end("inner"),
            )
            outer_start, outer_end = start + match.start(), start + match.end()
            if inner_start < inner_end:
                ranges.append((outer_start, outer_end, inner_start, inner_end))
    covered = [(outer_start - start, outer_end - start) for outer_start, outer_end, _inner_start, _inner_end in ranges]
    for match in re.finditer(r'["“]', region):
        if any(left <= match.start() < right for left, right in covered):
            continue
        closing_positions = [
            pos
            for pos in (
                region.find('"', match.end()),
                region.find("”", match.end()),
            )
            if pos >= 0
        ]
        close = min(closing_positions) if closing_positions else len(region)
        inner_start, inner_end = trim_aru_span(reasoning_text, start + match.end(), start + close)
        outer_start = start + match.start()
        outer_end = start + close + (1 if close < len(region) else 0)
        if inner_start < inner_end:
            ranges.append((outer_start, outer_end, inner_start, inner_end))
    ranges.sort(key=lambda item: (item[0], item[1]))
    non_overlapping: List[Tuple[int, int, int, int]] = []
    last_end = -1
    for item in ranges:
        if item[0] < last_end:
            continue
        non_overlapping.append(item)
        last_end = item[1]
    return non_overlapping


def split_dash_heading(
    reasoning_text: str,
    start: int,
    end: int,
    unit_type: str,
    row: Dict[str, Any],
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    region = reasoning_text[start:end]
    for match in DASH_SEPARATOR_PATTERN.finditer(region):
        if match.start() > 140:
            break
        head_start, head_end = trim_aru_span(reasoning_text, start, start + match.start())
        tail_start, tail_end = trim_aru_span(reasoning_text, start + match.end(), end)
        if head_start >= head_end or tail_start >= tail_end:
            continue
        head_text = reasoning_text[head_start:head_end]
        if len(head_text) > 140 or len(re.findall(r"[A-Za-z0-9]+", head_text)) > 14:
            continue
        if normalize_text(head_text).lower().startswith(("because ", "given ", "due to ", "since ", "as ")):
            continue
        if (
            unit_type in {"D", "C"}
            or looks_like_diagnosis_target(head_text)
            or final_answer_is_mentioned(head_text, row_final_answer(row))
            or CLAIM_CONFIRMATION_PATTERN.search(head_text)
        ):
            return (head_start, head_end), (tail_start, tail_end)
    return None


def classify_refined_unit_type(
    span_text: str,
    parent_type: str,
    row: Dict[str, Any],
    context_text: str,
    *,
    is_heading: bool = False,
    is_quoted: bool = False,
) -> str:
    default_claim = default_nli_claim_text(span_text, parent_type)
    coerced = coerce_role_type(parent_type, span_text, default_claim, row, context_text)
    if is_heading:
        if final_answer_title_like(span_text, row_final_answer(row)):
            if title_context_has_final_confirmation(span_text, context_text):
                return "C"
            return "D"
        if coerced in {"C", "D"}:
            return coerced
        if looks_like_diagnosis_target(span_text) or parent_type == "D":
            return "D"
        return coerced
    if is_quoted:
        if has_general_rule_signal(span_text) and not has_case_observation_signal(span_text):
            return "W"
        if has_case_observation_signal(span_text) or CLAIM_CONFIRMATION_PATTERN.search(span_text):
            return "O"
        return coerced
    if parent_type == "D" and not has_case_observation_signal(span_text) and not has_general_rule_signal(span_text):
        return "D"
    return coerced


def add_refined_candidate(
    candidates: List[Dict[str, Any]],
    reasoning_text: str,
    start: int,
    end: int,
    parent_type: str,
    row: Dict[str, Any],
    *,
    is_heading: bool = False,
    is_quoted: bool = False,
) -> None:
    start, end = trim_aru_span(reasoning_text, start, end)
    if start >= end:
        return
    span_text = reasoning_text[start:end]
    context_text = reasoning_text[max(0, start - 160) : min(len(reasoning_text), end + 240)]
    unit_type = classify_refined_unit_type(
        span_text,
        parent_type,
        row,
        context_text,
        is_heading=is_heading,
        is_quoted=is_quoted,
    )
    if not useful_refined_span(reasoning_text, start, end, unit_type, row):
        return
    nli_claim_text = default_nli_claim_text(span_text, unit_type)
    candidates.append(
        {
            "span_text": span_text,
            "nli_claim_text": nli_claim_text,
            "char_start": start,
            "char_end": end,
            "type": unit_type,
        }
    )


def add_region_candidates(
    candidates: List[Dict[str, Any]],
    reasoning_text: str,
    start: int,
    end: int,
    parent_type: str,
    row: Dict[str, Any],
) -> None:
    quoted_ranges = find_quoted_inner_ranges(reasoning_text, start, end)
    cursor = start
    for outer_start, outer_end, inner_start, inner_end in quoted_ranges:
        if cursor < outer_start:
            add_refined_candidate(candidates, reasoning_text, cursor, outer_start, parent_type, row)
        add_refined_candidate(
            candidates,
            reasoning_text,
            inner_start,
            inner_end,
            parent_type,
            row,
            is_quoted=True,
        )
        cursor = outer_end
    if cursor < end:
        add_refined_candidate(candidates, reasoning_text, cursor, end, parent_type, row)


def refine_extractive_unit(
    reasoning_text: str,
    start: int,
    end: int,
    unit_type: str,
    nli_claim_text: str,
    row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    span_text = reasoning_text[start:end]
    parent = {
        "span_text": span_text,
        "nli_claim_text": nli_claim_text or default_nli_claim_text(span_text, unit_type),
        "char_start": start,
        "char_end": end,
        "type": unit_type,
    }
    candidates: List[Dict[str, Any]] = []
    dash_split = split_dash_heading(reasoning_text, start, end, unit_type, row)
    if dash_split is not None:
        (head_start, head_end), (tail_start, tail_end) = dash_split
        head_text = reasoning_text[head_start:head_end]
        add_refined_candidate(
            candidates,
            reasoning_text,
            head_start,
            head_end,
            unit_type,
            row,
            is_heading=True,
        )
        tail_parent_type = "D" if unit_type == "D" or looks_like_diagnosis_target(head_text) else unit_type
        add_region_candidates(candidates, reasoning_text, tail_start, tail_end, tail_parent_type, row)
    elif find_quoted_inner_ranges(reasoning_text, start, end):
        add_region_candidates(candidates, reasoning_text, start, end, unit_type, row)
    else:
        return [parent]

    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, str]] = set()
    for candidate in sorted(candidates, key=lambda item: (item["char_start"], item["char_end"], item["type"])):
        key = (candidate["char_start"], candidate["char_end"], candidate["type"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    if len(deduped) >= 2:
        return deduped
    return [parent]


def duplicate_unit_score(unit: Dict[str, Any], reasoning_text: str, row: Dict[str, Any]) -> float:
    unit_type = normalize_role_type(unit.get("type"))
    start = parse_int(unit.get("char_start"))
    end = parse_int(unit.get("char_end"))
    if start is None or end is None:
        return -100.0
    span_text = reasoning_text[start:end]
    context_text = reasoning_text[max(0, start - 160) : min(len(reasoning_text), end + 240)]
    nli_claim_text = clean_nli_claim_text(unit.get("nli_claim_text"), default_nli_claim_text(span_text, unit_type))
    inferred = coerce_role_type(unit_type, span_text, nli_claim_text, row, context_text)
    score = 0.0
    if inferred == unit_type:
        score += 2.0
    if unit_type == "O" and has_case_observation_signal(span_text):
        score += 5.0
    if unit_type == "W" and has_general_rule_signal(span_text) and not has_case_observation_signal(span_text):
        score += 5.0
    if unit_type == "D" and (has_diagnosis_direction(span_text) or looks_like_diagnosis_target(span_text)):
        score += 4.0
    if unit_type == "C" and infer_final_answer_claim(unit_type, span_text, row, context_text):
        score += 5.0
    if unit.get("retrieval_query"):
        score += 0.1
    score -= (end - start) * 0.0001
    return score


def dedupe_units_by_boundary(
    units: List[Dict[str, Any]],
    reasoning_text: str,
    row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for unit in units:
        start = parse_int(unit.get("char_start"))
        end = parse_int(unit.get("char_end"))
        if start is None or end is None:
            continue
        surface_key = normalize_text(reasoning_text[start:end]).lower().strip(" \"'“”‘’.,;:-–—")
        key = (start, surface_key)
        existing = grouped.get(key)
        if existing is None or duplicate_unit_score(unit, reasoning_text, row) > duplicate_unit_score(
            existing, reasoning_text, row
        ):
            grouped[key] = unit
    return sorted(grouped.values(), key=lambda unit: (unit["char_start"], unit["char_end"], unit["type"]))


NUMBERED_ITEM_PATTERN = re.compile(r"(?m)(?:^|\n)\s*\d+\.\s*")


def numbered_item_ranges(text: str) -> List[Tuple[int, int]]:
    matches = list(NUMBERED_ITEM_PATTERN.finditer(text))
    if not matches:
        return [(0, len(text))]
    ranges: List[Tuple[int, int]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        ranges.append((match.start(), end))
    return ranges


def normalized_surface_key(text: str) -> str:
    return re.sub(r"^[\s\"'“”‘’]+|[\s\"'“”‘’.,;:]+$", "", normalize_text(text)).lower()


def build_consolidated_unit(
    reasoning_text: str,
    start: int,
    end: int,
    unit_type: str,
    row: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    start, end = trim_aru_span(reasoning_text, start, end)
    if start >= end:
        return None
    span_text = reasoning_text[start:end]
    if not useful_refined_span(reasoning_text, start, end, unit_type, row):
        return None
    context_text = reasoning_text[max(0, start - 160) : min(len(reasoning_text), end + 240)]
    corrected_type = coerce_role_type(
        unit_type,
        span_text,
        default_nli_claim_text(span_text, unit_type),
        row,
        context_text,
    )
    if has_diagnosis_direction(span_text) and not (
        corrected_type == "C" and infer_final_answer_claim(corrected_type, span_text, row, context_text)
    ):
        corrected_type = "D"
    nli_claim_text = default_nli_claim_text(span_text, corrected_type)
    retrieval_query = default_retrieval_query(nli_claim_text or span_text, corrected_type)
    payload: Dict[str, Any] = {
        "text": span_text,
        "span_text": span_text,
        "nli_claim_text": nli_claim_text,
        "char_start": start,
        "char_end": end,
        "type": corrected_type,
        "role_label": ROLE_LABELS[corrected_type],
        "is_final_answer_claim": infer_final_answer_claim(corrected_type, span_text, row, context_text),
        "mask_eligible": mask_eligible_default(corrected_type),
    }
    if corrected_type in {"W", "D"} and retrieval_query:
        payload["retrieval_query"] = retrieval_query
    return payload


def dedupe_units_by_surface_and_containment(units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for unit in sorted(units, key=lambda item: (parse_int(item.get("char_start")) or 0, parse_int(item.get("char_end")) or 0)):
        start = parse_int(unit.get("char_start"))
        end = parse_int(unit.get("char_end"))
        if start is None or end is None or start >= end:
            continue
        unit_type = normalize_role_type(unit.get("type") or unit.get("role_label"))
        surface = normalized_surface_key(unit.get("span_text") or unit.get("text") or "")
        drop = False
        for index, existing in enumerate(kept):
            existing_start = parse_int(existing.get("char_start"))
            existing_end = parse_int(existing.get("char_end"))
            if existing_start is None or existing_end is None or existing_start >= existing_end:
                continue
            existing_type = normalize_role_type(existing.get("type") or existing.get("role_label"))
            existing_surface = normalized_surface_key(existing.get("span_text") or existing.get("text") or "")
            if unit_type == existing_type and surface == existing_surface:
                if (end - start) < (existing_end - existing_start):
                    kept[index] = unit
                drop = True
                break
            if (
                unit_type == existing_type
                and existing_start <= start
                and end <= existing_end
                and existing_end > existing_start
                and (end - start) / (existing_end - existing_start) > 0.85
                and surface
                and surface in existing_surface
            ):
                if unit_type in {"D", "W"}:
                    drop = True
                    break
                kept[index] = unit
                drop = True
                break
        if not drop:
            kept.append(unit)
    return kept


ROLE_CALIBRATION_CASE_PATTERN = re.compile(
    r"\b(?:patient|she|he|ct|mri|ultrasound|radiograph|x-?ray|imaging|biopsy|histolog|"
    r"patholog|culture|pcr|laboratory|serum|urine|blood|showed|revealed|demonstrated|"
    r"found|confirmed|negative|positive|absence|lack|no\s+[a-z-]+|"
    r"\d+(?:\.\d+)?\s*(?:%|mg|mmol|pg|μ|mL|/))\b",
    flags=re.IGNORECASE,
)

ROLE_CALIBRATION_GENERAL_PATTERN = re.compile(
    r"\b(?:can|may|might|usually|typically|classically|often|characteristically|pathognomonic|"
    r"diagnostic of|associated with|presents? with|defined by|characterized by|mimic|mimics|"
    r"must be considered|should be considered)\b",
    flags=re.IGNORECASE,
)

ROLE_CALIBRATION_DECISION_PATTERN = re.compile(
    r"\b(?:differential diagnos(?:is|es)|differentials?|working diagnosis|preoperative diagnosis|"
    r"initial(?:ly)? diagnos|initial(?:ly)? suspect|was considered|were considered|is considered|"
    r"are considered|should .*considered|must .*considered|was excluded|were excluded|ruled out|"
    r"excluded|unlikely|less likely|argued against|did not explain|did not support|not conclusive|"
    r"favou?red|suggested|supported|consistent with|rather than|instead of|key differentials?|"
    r"most common of which)\b",
    flags=re.IGNORECASE,
)


def role_calibration_has_case_signal(text: str) -> bool:
    return bool(ROLE_CALIBRATION_CASE_PATTERN.search(normalize_text(text)))


def role_calibration_has_general_signal(text: str) -> bool:
    return bool(ROLE_CALIBRATION_GENERAL_PATTERN.search(normalize_text(text)))


def role_calibration_has_decision_signal(text: str) -> bool:
    return bool(ROLE_CALIBRATION_DECISION_PATTERN.search(normalize_text(text)))


def parent_context_indicates_differentiation(before_text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:considered|excluded|ruled out|unlikely|less likely|differential|diagnosis|"
            r"because|given|due to|supported by|argued against)\s*[\"“:]?\s*$",
            normalize_text(before_text),
            flags=re.IGNORECASE,
        )
    )


def drop_contained_duplicate_units(units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove small nested units when a broader NLI unit already covers them."""
    parsed: List[Tuple[int, int, int, Dict[str, Any]]] = []
    for index, unit in enumerate(units):
        start = parse_int(unit.get("char_start"))
        end = parse_int(unit.get("char_end"))
        if start is not None and end is not None and start < end:
            parsed.append((index, start, end, unit))
    drop: set[int] = set()
    for index, start, end, unit in parsed:
        if bool(unit.get("is_final_answer_claim")):
            continue
        length = end - start
        for other_index, other_start, other_end, other in parsed:
            if index == other_index:
                continue
            parent_length = other_end - other_start
            if not (other_start <= start and end <= other_end and parent_length > length):
                continue
            if parent_length < 45 or length / parent_length > 0.92:
                continue
            parent_role = normalize_role_type(other.get("type") or other.get("role_label"))
            if parent_role in {"D", "O", "W"}:
                drop.add(index)
                break
    return [unit for index, unit in enumerate(units) if index not in drop]


def calibrate_unit_role_for_nli(unit: Dict[str, Any], reasoning_text: str) -> bool:
    span_text = normalize_text(unit.get("span_text") or unit.get("text") or "")
    lowered = span_text.lower()
    role = normalize_role_type(unit.get("type") or unit.get("role_label"))
    start = parse_int(unit.get("char_start")) or 0
    before_text = reasoning_text[max(0, start - 120) : start]
    original_role = role

    if role == "C":
        if re.search(
            r"\b(?:could not be diagnosed|excluded|ruled out|unlikely|less likely|over true|rather than|"
            r"instead of|initially diagnosed|working diagnosis|differential)\b",
            lowered,
            flags=re.IGNORECASE,
        ):
            role = "D"
        elif not bool(unit.get("is_final_answer_claim")):
            if role_calibration_has_decision_signal(span_text) or parent_context_indicates_differentiation(before_text):
                role = "D"
            elif role_calibration_has_general_signal(span_text) and not role_calibration_has_case_signal(span_text):
                role = "W"
            else:
                role = "O"

    if role == "W":
        if re.search(
            r"\b(?:differential diagnos(?:is|es)|differentials?|key differentials?|most common of which|"
            r"must be considered|should be considered)\b",
            lowered,
            flags=re.IGNORECASE,
        ):
            role = "D"
        elif role_calibration_has_case_signal(span_text) and not role_calibration_has_general_signal(span_text):
            role = "O"
    elif role == "D":
        if not role_calibration_has_decision_signal(span_text) and not parent_context_indicates_differentiation(before_text):
            if role_calibration_has_case_signal(span_text):
                role = "O"
            elif role_calibration_has_general_signal(span_text):
                role = "W"
    elif role == "O":
        if re.search(
            r"\b(?:working diagnosis|preoperative diagnosis|initially diagnosed|initial diagnosis|"
            r"not conclusive for|should .*differential|differential diagnoses? .*include|"
            r"consistent with .* rather than|more typical of|not used to refine the differential)\b",
            lowered,
            flags=re.IGNORECASE,
        ):
            role = "D"

    if role == original_role:
        return False
    unit["type"] = role
    unit["role_label"] = ROLE_LABELS[role]
    unit["mask_eligible"] = mask_eligible_default(role)
    if role not in {"W", "D"}:
        unit.pop("retrieval_query", None)
    return True


def calibrate_roles_for_nli(units: List[Dict[str, Any]], reasoning_text: str) -> int:
    changed = 0
    for unit in units:
        if calibrate_unit_role_for_nli(unit, reasoning_text):
            changed += 1
    return changed


def consolidate_nli_oriented_units(
    units: List[Dict[str, Any]],
    reasoning_text: str,
    row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Merge over-split diagnosis reasoning into NLI-checkable spans."""
    sorted_units = sorted(
        [unit for unit in units if not is_fragmentary_nli_span(unit.get("span_text") or unit.get("text") or "", normalize_role_type(unit.get("type") or unit.get("role_label")))],
        key=lambda item: (parse_int(item.get("char_start")) or 0, parse_int(item.get("char_end")) or 0),
    )
    item_ranges = numbered_item_ranges(reasoning_text)
    consolidated: List[Dict[str, Any]] = []
    for item_start, item_end in item_ranges:
        item_units = [
            unit
            for unit in sorted_units
            if (parse_int(unit.get("char_start")) is not None and item_start <= (parse_int(unit.get("char_start")) or -1) < item_end)
        ]
        if not item_units:
            continue
        quote_ranges = find_quoted_inner_ranges(reasoning_text, item_start, item_end)
        segments: List[Tuple[str, int, int]] = []
        cursor = item_start
        for outer_start, outer_end, inner_start, inner_end in quote_ranges:
            if cursor < outer_start:
                segments.append(("plain", cursor, outer_start))
            segments.append(("quote", inner_start, inner_end))
            cursor = outer_end
        if cursor < item_end:
            segments.append(("plain", cursor, item_end))

        for segment_type, segment_start, segment_end in segments:
            segment_units = [
                unit
                for unit in item_units
                if (parse_int(unit.get("char_start")) or -1) < segment_end
                and (parse_int(unit.get("char_end")) or -1) > segment_start
            ]
            if not segment_units:
                continue
            segment_units.sort(key=lambda item: parse_int(item.get("char_start")) or 0)
            if segment_type == "quote":
                consolidated.extend(segment_units)
                continue
            index = 0
            while index < len(segment_units):
                unit = segment_units[index]
                unit_type = normalize_role_type(unit.get("type") or unit.get("role_label"))
                unit_start = parse_int(unit.get("char_start"))
                unit_end = parse_int(unit.get("char_end"))
                if unit_start is None or unit_end is None:
                    index += 1
                    continue
                if unit_type == "D":
                    run_start = unit_start
                    run_end = unit_end
                    next_index = index + 1
                    while next_index < len(segment_units):
                        next_unit = segment_units[next_index]
                        next_start = parse_int(next_unit.get("char_start"))
                        next_end = parse_int(next_unit.get("char_end"))
                        if next_start is None or next_end is None:
                            break
                        gap_text = reasoning_text[run_end:next_start]
                        if len(gap_text) > 80 or re.search(r"\n\s*\d+\.", gap_text):
                            break
                        if re.search(r"[.;]\s*$", gap_text) and not re.search(
                            r"\b(?:but|and|because|given|due to|when|after|by|with|without|in the absence)\b",
                            gap_text,
                            flags=re.IGNORECASE,
                        ):
                            break
                        candidate_text = reasoning_text[run_start:next_end]
                        if normalize_role_type(next_unit.get("type") or next_unit.get("role_label")) in {"O", "D", "C", "W"} and (
                            dash_span_indicates_differentiation(candidate_text)
                            or has_diagnosis_direction(candidate_text)
                            or re.search(
                                r"\b(?:because|given|due to|but|excluded|unlikely|ruled out|supported|absence|lack|without|with no|"
                                r"revealed|showed|demonstrated|corresponded|consistent with|rather than|instead of|compatible with)\b",
                                candidate_text,
                                flags=re.IGNORECASE,
                            )
                        ):
                            run_end = next_end
                            next_index += 1
                            continue
                        break
                    if run_end > unit_end:
                        merged = build_consolidated_unit(reasoning_text, run_start, run_end, "D", row)
                        if merged is not None:
                            consolidated.append(merged)
                            index = next_index
                            continue
                consolidated.append(unit)
                index += 1
    return dedupe_units_by_surface_and_containment(consolidated)


def normalize_unit_record(item: Dict[str, Any]) -> List[Dict[str, str]]:
    text = normalize_text(item.get("text"))
    source_text = text
    unit_type = normalize_text(item.get("type")).upper()[:1]
    if not text:
        return []
    if unit_type not in {"O", "W", "D", "C"}:
        unit_type = "C"

    if is_fragmentary_observation(text):
        if unit_type == "O":
            return []
        return [{"text": text, "type": "C"}]

    if is_workflow_statement(text):
        unit_type = "C"
        unit_payload = {"text": text, "type": unit_type}
        return [unit_payload]
    if is_evaluation_only_statement(text):
        unit_type = "C"
        unit_payload = {"text": text, "type": unit_type}
        return [unit_payload]
    if unit_type == "D" and re.search(r"\blife support\b", text, flags=re.IGNORECASE) and not re.search(
        r"\b(?:diagnosis|differential|considered|excluded|ruled out|suspected|argue|argues|argued|suggest|suggests|consistent|compatible)\b",
        text,
        flags=re.IGNORECASE,
    ):
        unit_type = "O"
        unit_payload = {"text": text, "type": unit_type}
        return [unit_payload]
    if unit_type == "D" and is_generic_differential_statement(text):
        unit_type = "C"
        unit_payload = {"text": text, "type": unit_type}
        return [unit_payload]
    query_seed = normalize_text(item.get("retrieval_query") or item.get("evidence_query") or item.get("query"))
    split_units: List[Dict[str, str]] = []
    matched_feature = ""
    matched_target = ""
    matched_cue = ""
    for pattern in COMBINED_CLAUSE_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        matched_feature = clean_query_text(match.groupdict().get("feature", ""))
        matched_target = clean_query_text(match.groupdict().get("target", ""))
        matched_cue = clean_query_text(match.groupdict().get("cue", ""))
        break

    split_feature_into_observation = bool(matched_feature and is_observation_like_feature(matched_feature))

    if split_feature_into_observation:
        observation_text = observation_from_feature(matched_feature)
        if observation_text:
            split_units.append({"text": observation_text, "type": "O"})
        if not query_seed and matched_target:
            query_seed = f"{matched_target} {clean_query_text(observation_text)}".strip()

    generic_target_block = False
    target_candidates: List[str] = []
    if matched_target and matched_cue:
        if matched_target.lower() in GENERIC_DIAGNOSIS_REFERENTS:
            inferred_target = extract_target_like_phrase(query_seed or source_text)
            if (
                inferred_target
                and not is_generic_diagnosis_target(inferred_target)
                and not is_observation_like_feature(inferred_target)
            ):
                matched_target = inferred_target
            else:
                unit_type = "C"
                generic_target_block = True
                query_seed = ""

        if not generic_target_block:
            target_candidates = split_conjoined_targets(matched_target or query_seed or source_text)
            diagnosis_like_count = sum(1 for target in target_candidates if looks_like_diagnosis_target(target))
            if diagnosis_like_count == 0:
                if "consider" in matched_cue.lower() or "in the differential diagnosis" in matched_cue.lower() or "support" in matched_cue.lower() or "argue" in matched_cue.lower() or "exclude" in matched_cue.lower() or "rule out" in matched_cue.lower() or "suspect" in matched_cue.lower() or "likely" in matched_cue.lower():
                    unit_type = "C"
                    generic_target_block = True
                    query_seed = ""
            else:
                matched_target = target_candidates[0]
                if not query_seed:
                    query_seed = target_candidates[0]

        if split_feature_into_observation:
            text = claim_from_cue(matched_target, matched_cue, text)
        if not generic_target_block:
            if "consider" in matched_cue.lower() or "in the differential diagnosis" in matched_cue.lower():
                unit_type = "D"
            elif "support" in matched_cue.lower() or "argue" in matched_cue.lower() or "exclude" in matched_cue.lower() or "rule out" in matched_cue.lower() or "suspect" in matched_cue.lower() or "likely" in matched_cue.lower():
                unit_type = "D"
            elif any(token in matched_cue.lower() for token in ("mean", "indicate", "imply", "reflect")):
                unit_type = "C"
            else:
                unit_type = "C"

    if unit_type == "D" and len(target_candidates) > 1 and not generic_target_block:
        for target in target_candidates:
            unit_text = claim_from_cue(target, matched_cue, text)
            retrieval_query = default_retrieval_query(target, unit_type)
            unit_payload = {"text": unit_text, "type": unit_type}
            if retrieval_query:
                unit_payload["retrieval_query"] = retrieval_query
            split_units.append(unit_payload)
        return split_units

    if unit_type == "D":
        lowered_text = clean_query_text(text).lower()
        if re.fullmatch(r"(?:this|the)\s+diagnosis\s+was\s+(?:less likely|unlikely)", lowered_text):
            inferred_target = extract_target_like_phrase(query_seed or matched_target or text)
            if inferred_target:
                text = claim_from_cue(inferred_target, "less likely", text)
                query_seed = inferred_target
            else:
                unit_type = "C"
        elif re.fullmatch(r".+\s+was evaluated", lowered_text) or re.fullmatch(r".+\s+were evaluated", lowered_text):
            unit_type = "C"
        elif lowered_text.startswith(("because ", "due to ", "given ", "since ", "as ")):
            unit_type = "C"
            query_seed = ""
        else:
            inferred_target = extract_target_like_phrase(query_seed or matched_target or text)
            if not looks_like_diagnosis_target(inferred_target):
                unit_type = "C"
                query_seed = ""

    retrieval_query = default_retrieval_query(query_seed or text or source_text, unit_type)
    unit_payload = {"text": text, "type": unit_type}
    if retrieval_query:
        unit_payload["retrieval_query"] = retrieval_query
    split_units.append(unit_payload)
    return split_units


def build_user_prompt(row: Dict[str, Any], reasoning_text: str) -> str:
    patient_context = normalize_text(
        row.get("patient_context")
        or row.get("question_or_case_context")
        or row.get("case_context")
        or row.get("context")
    )
    patient_context = truncate_for_prompt(patient_context, MAX_PATIENT_CONTEXT_CHARS)
    final_answer = normalize_text(
        row.get("negative_final_answer")
        or row.get("predicted_answer_text")
        or row.get("final_answer")
        or row.get("correct_answer_text")
    )
    return (
        "Patient context:\n"
        f"{patient_context or '[EMPTY PATIENT CONTEXT]'}\n\n"
        "Final answer:\n"
        f"{final_answer or '[EMPTY FINAL ANSWER]'}\n\n"
        "Reasoning to decompose:\n"
        f"{reasoning_text}\n"
    )


def build_retry_user_prompt(
    row: Dict[str, Any],
    reasoning_text: str,
    units: List[Dict[str, Any]],
    issue: str,
    expected_min_units: int,
) -> str:
    prior_spans = [
        {
            "span_text": unit.get("span_text", ""),
            "char_start": unit.get("char_start"),
            "char_end": unit.get("char_end"),
            "type": unit.get("type"),
        }
        for unit in units[:12]
    ]
    return (
        build_user_prompt(row, reasoning_text)
        + "\nCoverage repair instruction:\n"
        + f"The previous extraction was rejected for `{issue}`. It produced {len(units)} accepted ARUs; "
        + f"this response likely needs at least about {expected_min_units} ARUs.\n"
        + "Redo the extraction from scratch with higher recall.\n"
        + "Important fixes:\n"
        + "- Do not output the whole response or whole numbered item as one ARU.\n"
        + "- For every numbered item or sentence-level differential, extract the diagnosis decision, "
        + "the explicit findings/rationale, and the quoted evidence as separate spans when present.\n"
        + "- Include all alternative diagnoses and all explicit exclusions/support statements.\n"
        + "- Keep every span as an exact substring with correct char_start and char_end.\n"
        + "- For each span, also provide nli_claim_text as the self-contained proposition for NLI.\n"
        + "- Return only the corrected JSON object.\n\n"
        + "Rejected prior sample spans:\n"
        + json.dumps(prior_spans, ensure_ascii=False)
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose accepted negative reasoning traces into ARUs."
    )
    parser.add_argument(
        "--input-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.strict_clean.jsonl"),
        help="Strict-clean negative JSONL from stage 03.",
    )
    parser.add_argument(
        "--output-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.strict_clean.arus.jsonl"),
        help="ARU decomposition JSONL output path.",
    )
    parser.add_argument(
        "--reasoning-field",
        default="negative_reasoning",
        help="Primary field to decompose.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API key for the decomposition model.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "/home/models/Qwen3-Next-80B-A3B-Instruct"),
        help="LLM used for ARU decomposition.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=64,
        help="Maximum concurrent LLM requests.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start row index.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="End row index (exclusive). -1 means all rows.",
    )
    parser.add_argument(
        "--max-reasoning-chars",
        type=int,
        default=0,
        help="If >0, skip rows whose selected reasoning text is longer than this many characters.",
    )
    parser.add_argument(
        "--skipped-file",
        default="",
        help="Optional JSONL file where skipped over-length rows are recorded for a later separate run.",
    )
    parser.add_argument(
        "--sort-by-reasoning-length",
        choices=["none", "asc", "desc"],
        default="none",
        help="Process remaining rows by selected reasoning length. Use asc to finish shorter rows first.",
    )
    parser.add_argument(
        "--metadata-file",
        default=artifact_path("fa_dpo_pipeline", "run_metadata", "04_decompose_negatives_to_arus.json"),
        help="Where to write run metadata JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output file before running. Use this when regenerating old non-extractive ARU outputs.",
    )
    return parser.parse_args()


def completed_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    return {row_identifier(row) for row in load_jsonl(path)}


def row_identifier(row: Dict[str, Any]) -> str:
    return str(row.get("id") or row.get("item_id") or row.get("source_id") or "")


def select_reasoning(row: Dict[str, Any], primary_field: str) -> str:
    for field in [primary_field, "negative_reasoning", "predicted_reasoning", "response_text", "rejected"]:
        value = normalize_text(row.get(field))
        if value:
            return value
    return ""


def record_skipped_rows(rows: List[Dict[str, Any]], args: argparse.Namespace) -> int:
    if not rows or not args.skipped_file:
        return 0
    already_recorded = completed_ids(args.skipped_file)
    written = 0
    for row in rows:
        if row_identifier(row) in already_recorded:
            continue
        reasoning_text = select_reasoning(row, args.reasoning_field)
        skipped_row = dict(row)
        skipped_row["stage04_skip_reason"] = "reasoning_too_long"
        skipped_row["stage04_reasoning_chars"] = len(reasoning_text)
        skipped_row["stage04_max_reasoning_chars"] = args.max_reasoning_chars
        append_jsonl(args.skipped_file, skipped_row)
        already_recorded.add(row_identifier(row))
        written += 1
    return written


def normalize_extractive_units(
    payload: Dict[str, Any],
    reasoning_text: str,
    row: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    raw_units = payload.get("units") or payload.get("arus") or []
    units: List[Dict[str, Any]] = []
    stats = {
        "raw_units": len(raw_units) if isinstance(raw_units, list) else 0,
        "accepted_units": 0,
        "rejected_units": 0,
        "explicit_span_units": 0,
        "exact_located_units": 0,
    }
    if not isinstance(raw_units, list):
        return units, stats

    cursor = 0
    seen_spans: set[Tuple[int, int, str]] = set()
    for item in raw_units:
        if not isinstance(item, dict):
            stats["rejected_units"] += 1
            continue
        span_text = normalize_text(item.get("span_text") or item.get("text"))
        unit_type = normalize_role_type(item.get("type") or item.get("role_label") or item.get("role"))
        explicit = valid_explicit_span(
            reasoning_text,
            parse_int(item.get("char_start")),
            parse_int(item.get("char_end")),
            span_text,
        )
        if explicit is not None:
            start, end = explicit
            stats["explicit_span_units"] += 1
        else:
            located = locate_exact_span(reasoning_text, span_text, cursor)
            if located is None:
                stats["rejected_units"] += 1
                continue
            start, end = located
            stats["exact_located_units"] += 1
        span_text = reasoning_text[start:end]
        nli_claim_text = nli_claim_text_from_item(item, span_text, unit_type)
        local_context = reasoning_text[max(0, start - 120) : min(len(reasoning_text), end + 220)]
        corrected_unit_type = coerce_role_type(unit_type, span_text, nli_claim_text, row, local_context)
        if corrected_unit_type != unit_type:
            stats["role_corrected_units"] = stats.get("role_corrected_units", 0) + 1
            unit_type = corrected_unit_type
            nli_claim_text = nli_claim_text_from_item(item, span_text, unit_type)
        refined_units = refine_extractive_unit(
            reasoning_text,
            start,
            end,
            unit_type,
            nli_claim_text,
            row,
        )
        if len(refined_units) > 1:
            stats["split_parent_units"] = stats.get("split_parent_units", 0) + 1
            stats["split_child_units"] = stats.get("split_child_units", 0) + len(refined_units)
        for refined in refined_units:
            refined_start = parse_int(refined.get("char_start"))
            refined_end = parse_int(refined.get("char_end"))
            if refined_start is None or refined_end is None:
                stats["rejected_units"] += 1
                continue
            refined_type = normalize_role_type(refined.get("type"))
            refined_span_text = reasoning_text[refined_start:refined_end]
            refined_nli_claim_text = clean_nli_claim_text(
                refined.get("nli_claim_text"),
                default_nli_claim_text(refined_span_text, refined_type),
            )
            refined_context = reasoning_text[
                max(0, refined_start - 120) : min(len(reasoning_text), refined_end + 220)
            ]
            corrected_refined_type = coerce_role_type(
                refined_type,
                refined_span_text,
                refined_nli_claim_text,
                row,
                refined_context,
            )
            if corrected_refined_type != refined_type:
                stats["role_corrected_units"] = stats.get("role_corrected_units", 0) + 1
                refined_type = corrected_refined_type
                refined_nli_claim_text = clean_nli_claim_text(
                    refined.get("nli_claim_text"),
                    default_nli_claim_text(refined_span_text, refined_type),
                )
            dedupe_key = (refined_start, refined_end, refined_type)
            if dedupe_key in seen_spans:
                stats["rejected_units"] += 1
                continue
            seen_spans.add(dedupe_key)
            if refined_end >= cursor:
                cursor = refined_end
            parent_same_span = refined_start == start and refined_end == end
            retrieval_query = ""
            if parent_same_span:
                retrieval_query = normalize_text(item.get("retrieval_query") or item.get("evidence_query") or item.get("query"))
            if not retrieval_query:
                retrieval_query = default_retrieval_query(refined_nli_claim_text or refined_span_text, refined_type)
            if refined_type not in {"W", "D"}:
                retrieval_query = ""
            is_final_answer_claim = infer_final_answer_claim(
                refined_type,
                refined_span_text,
                row,
                refined_context,
            )
            mask_eligible = mask_eligible_default(refined_type)
            unit_payload: Dict[str, Any] = {
                "text": refined_span_text,
                "span_text": refined_span_text,
                "nli_claim_text": refined_nli_claim_text,
                "char_start": refined_start,
                "char_end": refined_end,
                "type": refined_type,
                "role_label": ROLE_LABELS[refined_type],
                "is_final_answer_claim": is_final_answer_claim,
                "mask_eligible": mask_eligible,
            }
            if retrieval_query:
                unit_payload["retrieval_query"] = retrieval_query
            units.append(unit_payload)
            stats["accepted_units"] += 1
    before_boundary_dedupe = len(units)
    units = dedupe_units_by_boundary(units, reasoning_text, row)
    if len(units) != before_boundary_dedupe:
        stats["boundary_deduped_units"] = before_boundary_dedupe - len(units)
    before_nli_consolidation = len(units)
    units = consolidate_nli_oriented_units(units, reasoning_text, row)
    if len(units) != before_nli_consolidation:
        stats["nli_oriented_consolidation_delta"] = len(units) - before_nli_consolidation
    enforce_single_final_answer_claim(units, reasoning_text, row)
    before_containment_dedupe = len(units)
    units = drop_contained_duplicate_units(units)
    if len(units) != before_containment_dedupe:
        stats["contained_duplicate_units"] = before_containment_dedupe - len(units)
    role_calibrated_units = calibrate_roles_for_nli(units, reasoning_text)
    if role_calibrated_units:
        stats["nli_role_calibrated_units"] = role_calibrated_units
    return units, stats


def normalize_units(payload: Dict[str, Any], fallback_text: str) -> List[Dict[str, str]]:
    raw_units = payload.get("units") or payload.get("arus") or []
    units: List[Dict[str, str]] = []
    for item in raw_units:
        if not isinstance(item, dict):
            continue
        normalized_items = normalize_unit_record(item)
        if normalized_items:
            units.extend(normalized_items)
    if units:
        return units
    fallback = normalize_text(fallback_text)
    return [{"text": fallback, "type": "C"}] if fallback else []


def reduced_max_tokens_from_context_error(message: str, current_max_tokens: int) -> Optional[int]:
    match = re.search(
        r"maximum context length is (\d+) tokens and your request has (\d+) input tokens",
        message,
    )
    if not match:
        return None
    max_context = int(match.group(1))
    input_tokens = int(match.group(2))
    safe_max_tokens = max_context - input_tokens - 32
    if safe_max_tokens >= current_max_tokens:
        return None
    return max(256, safe_max_tokens)


async def create_chat_completion_with_token_retry(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    messages: List[Dict[str, str]],
) -> Tuple[Any, int]:
    max_tokens = args.max_tokens
    for _ in range(3):
        try:
            response = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=max_tokens,
            )
            return response, max_tokens
        except Exception as exc:
            reduced_max_tokens = reduced_max_tokens_from_context_error(str(exc), max_tokens)
            if reduced_max_tokens is None or reduced_max_tokens >= max_tokens:
                raise
            max_tokens = reduced_max_tokens
    response = await client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=max_tokens,
    )
    return response, max_tokens


async def decompose_one(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    limiter: AsyncLimiter,
    semaphore: asyncio.Semaphore,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    reasoning_text = select_reasoning(row, args.reasoning_field)
    async with semaphore:
        async with limiter:
            response, initial_max_tokens_used = await create_chat_completion_with_token_retry(
                client,
                args,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(row, reasoning_text)},
                ],
            )

    raw_output = response.choices[0].message.content or ""
    payload = extract_json(raw_output)
    units, validation = normalize_extractive_units(payload, reasoning_text, row)
    if initial_max_tokens_used != args.max_tokens:
        validation["initial_max_tokens_reduced_to"] = initial_max_tokens_used
    needs_retry, retry_reason, expected_min_units = coverage_issue(reasoning_text, units)
    retry_raw_output = ""
    if needs_retry:
        validation["coverage_retry_attempted"] = 1
        validation["coverage_retry_reason"] = retry_reason
        validation["expected_min_units"] = expected_min_units
        validation["initial_accepted_units"] = len(units)
        async with semaphore:
            async with limiter:
                retry_response, retry_max_tokens_used = await create_chat_completion_with_token_retry(
                    client,
                    args,
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": build_retry_user_prompt(
                                row,
                                reasoning_text,
                                units,
                                retry_reason,
                                expected_min_units,
                            ),
                        },
                    ],
                )
        if retry_max_tokens_used != args.max_tokens:
            validation["retry_max_tokens_reduced_to"] = retry_max_tokens_used
        retry_raw_output = retry_response.choices[0].message.content or ""
        retry_payload = extract_json(retry_raw_output)
        retry_units, retry_validation = normalize_extractive_units(retry_payload, reasoning_text, row)
        validation["retry_raw_units"] = retry_validation.get("raw_units", 0)
        validation["retry_accepted_units"] = retry_validation.get("accepted_units", 0)
        validation["retry_rejected_units"] = retry_validation.get("rejected_units", 0)
        if len(retry_units) >= len(units):
            units = retry_units
            validation["coverage_retry_used"] = 1
        else:
            validation["coverage_retry_used"] = 0
    else:
        validation["coverage_retry_attempted"] = 0
        validation["expected_min_units"] = expected_min_units
    if not units:
        legacy_units = normalize_units(payload, fallback_text=reasoning_text)
        units = []
        cursor = 0
        for legacy_unit in legacy_units:
            span_text = normalize_text(legacy_unit.get("text"))
            located = locate_exact_span(reasoning_text, span_text, cursor)
            if located is None:
                continue
            start, end = located
            if end >= cursor:
                cursor = end
            unit_type = normalize_role_type(legacy_unit.get("type"))
            span_text = reasoning_text[start:end]
            nli_claim_text = clean_nli_claim_text(
                legacy_unit.get("nli_claim_text")
                or legacy_unit.get("claim_text")
                or legacy_unit.get("atomic_proposition"),
                default_nli_claim_text(span_text, unit_type),
            )
            local_context = reasoning_text[max(0, start - 120) : min(len(reasoning_text), end + 220)]
            corrected_unit_type = coerce_role_type(unit_type, span_text, nli_claim_text, row, local_context)
            if corrected_unit_type != unit_type:
                unit_type = corrected_unit_type
                nli_claim_text = clean_nli_claim_text(
                    legacy_unit.get("nli_claim_text")
                    or legacy_unit.get("claim_text")
                    or legacy_unit.get("atomic_proposition"),
                    default_nli_claim_text(span_text, unit_type),
                )
            unit_payload: Dict[str, Any] = {
                "text": span_text,
                "span_text": span_text,
                "nli_claim_text": nli_claim_text,
                "char_start": start,
                "char_end": end,
                "type": unit_type,
                "role_label": ROLE_LABELS[unit_type],
                "is_final_answer_claim": infer_final_answer_claim(unit_type, span_text, row, local_context),
                "mask_eligible": mask_eligible_default(unit_type),
            }
            retrieval_query = normalize_text(legacy_unit.get("retrieval_query"))
            if not retrieval_query:
                retrieval_query = default_retrieval_query(nli_claim_text or span_text, unit_type)
            if retrieval_query and unit_type in {"W", "D"}:
                unit_payload["retrieval_query"] = retrieval_query
            units.append(unit_payload)
        enforce_single_final_answer_claim(units, reasoning_text, row)
        validation["legacy_fallback_units"] = len(units)
    patient_context = normalize_text(
        row.get("patient_context")
        or row.get("question_or_case_context")
        or row.get("case_context")
        or row.get("context")
    )
    return {
        "id": row.get("id") or row.get("item_id"),
        "source_id": row.get("source_id"),
        "error_symbol": row.get("error_symbol"),
        "patient_context": patient_context,
        "negative_reasoning": reasoning_text,
        "arus": units,
        "raw_llm_output": raw_output,
        "retry_raw_llm_output": retry_raw_output,
        "span_validation": validation,
    }


async def main_async(args: argparse.Namespace) -> None:
    if AsyncLimiter is None or AsyncOpenAI is None:
        raise RuntimeError("Missing async generation dependencies. Install openai and aiolimiter first.")
    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)
    rows = load_jsonl(args.input_file)
    end = len(rows) if args.end < 0 else min(args.end, len(rows))
    rows = rows[args.start:end]
    seen = completed_ids(args.output_file)
    rows = [row for row in rows if row_identifier(row) not in seen and select_reasoning(row, args.reasoning_field)]
    setattr(args, "_completed_rows_before", len(seen))
    setattr(args, "_candidate_rows_before_length_filter", len(rows))

    skipped_rows: List[Dict[str, Any]] = []
    if args.max_reasoning_chars > 0:
        kept_rows: List[Dict[str, Any]] = []
        for row in rows:
            reasoning_text = select_reasoning(row, args.reasoning_field)
            if len(reasoning_text) > args.max_reasoning_chars:
                skipped_rows.append(row)
            else:
                kept_rows.append(row)
        rows = kept_rows
    skipped_written = record_skipped_rows(skipped_rows, args)
    setattr(args, "_skipped_long_rows", len(skipped_rows))
    setattr(args, "_skipped_long_rows_written", skipped_written)

    if args.sort_by_reasoning_length != "none":
        rows.sort(
            key=lambda row: len(select_reasoning(row, args.reasoning_field)),
            reverse=args.sort_by_reasoning_length == "desc",
        )
    setattr(args, "_rows_scheduled", len(rows))

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    limiter = AsyncLimiter(args.max_concurrency, time_period=1)
    semaphore = asyncio.Semaphore(args.max_concurrency)

    tasks = [decompose_one(client, args, limiter, semaphore, row) for row in rows]
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Decomposing ARUs"):
        result = await future
        append_jsonl(args.output_file, result)


def main() -> None:
    args = parse_args()
    started_at = iso_utc_now()
    asyncio.run(main_async(args))
    output_rows = load_jsonl(args.output_file)
    total_arus = sum(len(row.get("arus") or []) for row in output_rows)
    metadata_path = write_run_metadata(
        stage_name="04_decompose_negatives_to_arus",
        args=args,
        inputs={"input_file": args.input_file},
        outputs={"output_file": args.output_file},
        stats={
            "processed_rows": count_jsonl(args.output_file),
            "total_arus": total_arus,
            "avg_arus_per_sample": round(total_arus / max(len(output_rows), 1), 4),
            "completed_rows_before": getattr(args, "_completed_rows_before", 0),
            "candidate_rows_before_length_filter": getattr(args, "_candidate_rows_before_length_filter", 0),
            "rows_scheduled": getattr(args, "_rows_scheduled", 0),
            "skipped_long_rows": getattr(args, "_skipped_long_rows", 0),
            "skipped_long_rows_written": getattr(args, "_skipped_long_rows_written", 0),
            "max_reasoning_chars": args.max_reasoning_chars,
            "sort_by_reasoning_length": args.sort_by_reasoning_length,
        },
        metadata_file=args.metadata_file,
        started_at=started_at,
        finished_at=iso_utc_now(),
    )
    print(f"Run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
