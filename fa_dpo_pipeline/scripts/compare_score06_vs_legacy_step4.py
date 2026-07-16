#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from itertools import islice
import math
import statistics
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fa_dpo_pipeline.common import (
    append_jsonl,
    artifact_path,
    ensure_parent,
    iso_utc_now,
    normalize_text,
    read_jsonl,
    write_run_metadata,
)


def load_module(module_name: str, file_path: Path) -> ModuleType:
    parent = str(file_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run current stage-06 scoring and a legacy 6_step4_v2-compatible scorer "
            "on the same stage-05 ARU-with-evidence file, then write a comparison report."
        )
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Stage-05 ARU JSONL with attached evidence.",
    )
    parser.add_argument(
        "--output-dir",
        default=artifact_path("fa_dpo_pipeline", "score_method_comparison"),
        help="Directory for modern scores, legacy scores, comparison JSONL, and Markdown report.",
    )
    parser.add_argument(
        "--source-aru-file",
        default="",
        help="Optional stage-04 ARU JSONL used by the stage-06 preflight checker.",
    )
    parser.add_argument(
        "--stage05-metadata-file",
        default="",
        help="Optional stage-05 metadata JSON used by the stage-06 preflight checker.",
    )
    parser.add_argument(
        "--preflight-report-file",
        default="",
        help="Optional markdown path for the stage-06 preflight report.",
    )
    parser.add_argument(
        "--preflight-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only run preflight and exit without scoring.",
    )
    parser.add_argument(
        "--skip-preflight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the stage-06 preflight check.",
    )
    parser.add_argument(
        "--strict-preflight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Abort if preflight raises warnings.",
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
        "--nli-model",
        default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        help="Zero-shot NLI checkpoint used by both scorers.",
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
        help="Observation fallback hallucination penalty in stage-06.",
    )
    parser.add_argument(
        "--lambda-miss",
        type=float,
        default=0.2,
        help="Missing-record penalty for observation nodes in stage-06.",
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
        help="Unsupported-differentiation penalty in stage-06.",
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
        help="Weight on max-risk in the final score.",
    )
    parser.add_argument(
        "--alpha-cov",
        type=float,
        default=0.4,
        help="Coverage-penalty weight.",
    )
    parser.add_argument(
        "--avg-defect-weight",
        type=float,
        default=0.1,
        help="Average-defect weight in the final score.",
    )
    parser.add_argument(
        "--margin-scale",
        type=float,
        default=3.0,
        help="Scale from sample risk to DPO margin.",
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
        "--device",
        default="cuda",
        help="Torch device. Use `cuda` to leverage all visible GPUs via DataParallel.",
    )
    parser.add_argument(
        "--max-missing-wd-rate",
        type=float,
        default=0.2,
        help="Preflight warning threshold copied from stage-06.",
    )
    parser.add_argument(
        "--max-all-wd-missing-row-rate",
        type=float,
        default=0.1,
        help="Preflight warning threshold copied from stage-06.",
    )
    parser.add_argument(
        "--min-avg-wd-evidence",
        type=float,
        default=1.0,
        help="Preflight warning threshold copied from stage-06.",
    )
    parser.add_argument(
        "--legacy-history-window",
        type=int,
        default=5,
        help="How many prior node texts the legacy-compatible C-node scorer sees.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.1,
        help="Threshold used when counting large score deltas.",
    )
    parser.add_argument(
        "--high-risk-threshold",
        type=float,
        default=0.5,
        help="Threshold used when comparing high-risk row counts.",
    )
    parser.add_argument(
        "--top-k-report",
        type=int,
        default=20,
        help="How many largest disagreements to include in the Markdown report.",
    )
    parser.add_argument(
        "--metadata-file",
        default=artifact_path(
            "fa_dpo_pipeline",
            "run_metadata",
            "compare_score06_vs_legacy_step4.json",
        ),
        help="Where to write run metadata JSON.",
    )
    return parser.parse_args()

def load_jsonl_window(path: str, start: int, end: int) -> List[Dict[str, Any]]:
    start_index = max(0, int(start))
    end_index = None if int(end) < 0 else max(start_index, int(end))
    return list(islice(read_jsonl(path), start_index, end_index))


def filter_rows_by_ids(rows: List[Dict[str, Any]], allowed_ids: set[str]) -> List[Dict[str, Any]]:
    if not allowed_ids:
        return []
    return [row for row in rows if str(row.get("id")) in allowed_ids]


def pipeline_probs(entry: Dict[str, Any]) -> List[Dict[str, float]]:
    return [
        {"label": "entailment", "score": float(entry.get("entailment", 0.0))},
        {"label": "contradiction", "score": float(entry.get("contradiction", 0.0))},
        {"label": "neutral", "score": float(entry.get("neutral", 1.0))},
    ]


def normalized_aru_nodes(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for node in row.get("arus") or []:
        if not isinstance(node, dict):
            normalized.append({"text": "", "type": "C"})
            continue
        node_type = normalize_text(node.get("type")).upper()[:1] or "C"
        if node_type not in {"O", "W", "D", "C"}:
            node_type = "C"
        normalized.append({**node, "text": normalize_text(node.get("text")), "type": node_type})
    return normalized


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def calculate_legacy_v2_final_score(
    arus: List[Dict[str, Any]],
    nli_results: List[Dict[str, Any]],
    total_ctx_sentences: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    node_scores: List[float] = []
    node_traces: List[Dict[str, Any]] = []
    covered_ctx_indices = set()
    has_prevalence_excuse = False
    excuse_keywords = (
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

    for node_index, node in enumerate(arus):
        node_type = normalize_text(node.get("type")).upper()[:1] or "O"
        node_text = normalize_text(node.get("text")).lower()
        result_pack = nli_results[node_index] if node_index < len(nli_results) else {}
        if any(keyword in node_text for keyword in excuse_keywords):
            has_prevalence_excuse = True

        ctx_probs = []
        for ctx_entry in result_pack.get("context_check") or []:
            ctx_probs.append({str(item["label"]).lower(): float(item["score"]) for item in ctx_entry})

        doc_probs = []
        for doc_entry in result_pack.get("doc_check") or []:
            doc_probs.append({str(item["label"]).lower(): float(item["score"]) for item in doc_entry})

        score = 0.0
        trace = ""

        if node_type == "O":
            if not ctx_probs:
                score = 0.5
                trace = "NoCtx"
            else:
                best_match_idx, best_ctx = max(
                    enumerate(ctx_probs),
                    key=lambda item: item[1].get("entailment", 0.0),
                )
                p_ctx_con = best_ctx.get("contradiction", 0.0)
                p_ctx_ent = best_ctx.get("entailment", 0.0)
                p_ctx_neu = best_ctx.get("neutral", 0.0)

                if p_ctx_con > args.contradiction_threshold:
                    score = 1.0
                    trace = f"CONFLICT:{p_ctx_con:.2f}"
                elif p_ctx_ent > args.entailment_threshold:
                    score = 0.0
                    trace = f"Supported:{p_ctx_ent:.2f}"
                    covered_ctx_indices.add(best_match_idx)
                else:
                    if not doc_probs:
                        score = args.lambda_hal * p_ctx_neu
                        trace = f"Neu:{p_ctx_neu:.2f}"
                    else:
                        best_doc = max(doc_probs, key=lambda item: item.get("entailment", 0.0))
                        doc_ent = best_doc.get("entailment", 0.0)
                        doc_con = best_doc.get("contradiction", 0.0)
                        if doc_con > args.contradiction_threshold:
                            score = doc_con
                            trace = f"DocConflict:{doc_con:.2f}"
                        elif doc_ent > args.entailment_threshold:
                            score = 0.1
                            trace = f"DocSaved:{doc_ent:.2f}"
                        else:
                            score = args.lambda_hal
                            trace = "Unsupported"

        elif node_type in {"W", "D"}:
            if not doc_probs:
                score = args.lambda_sup
                trace = "NoEvid"
            else:
                best_doc = max(doc_probs, key=lambda item: item.get("entailment", 0.0))
                max_con = best_doc.get("contradiction", 0.0)
                max_ent = best_doc.get("entailment", 0.0)
                if max_con > args.contradiction_threshold:
                    score = 1.0
                    trace = f"THEORY_CONFLICT:{max_con:.2f}"
                else:
                    score = max(0.0, args.lambda_sup * (1.0 - max_ent))
                    trace = f"E:{max_ent:.2f}"

        elif node_type == "C":
            if not ctx_probs:
                score = 0.0
            else:
                ctx = ctx_probs[0]
                score = max(ctx.get("contradiction", 0.0), args.lambda_log * ctx.get("neutral", 0.0))
                trace = f"Logic:{score:.2f}"

        node_scores.append(score)
        node_traces.append({"id": node_index, "t": node_type, "s": round(float(score), 4), "p": trace})

    if not node_scores:
        return {"unfaithfulness_score": 0.0, "metrics": {}}

    max_risk = max(node_scores)
    avg_defect = sum(node_scores) / len(node_scores)
    coverage_ratio = len(covered_ctx_indices) / total_ctx_sentences if total_ctx_sentences > 0 else 0.0
    target_coverage = (
        args.prevalence_target_coverage if has_prevalence_excuse else args.default_target_coverage
    )
    coverage_penalty = max(0.0, target_coverage - coverage_ratio)
    if has_prevalence_excuse and coverage_penalty > 0.0:
        coverage_penalty = min(1.0, coverage_penalty * 1.5)

    final_score = min(
        1.0,
        (args.gamma * max_risk)
        + (args.alpha_cov * coverage_penalty)
        + (args.avg_defect_weight * avg_defect),
    )
    return {
        "unfaithfulness_score": round(float(final_score), 4),
        "metrics": {
            "max_risk": round(float(max_risk), 4),
            "avg_defect": round(float(avg_defect), 4),
            "coverage_ratio": round(float(coverage_ratio), 4),
            "penalty_score": round(float(coverage_penalty), 4),
            "covered_sentences": int(len(covered_ctx_indices)),
            "total_sentences": int(total_ctx_sentences),
            "has_prevalence_excuse": bool(has_prevalence_excuse),
            "trace": node_traces,
        },
    }


def summarize_vector(values: Iterable[float]) -> Dict[str, float]:
    vector = [float(value) for value in values]
    if not vector:
        return {"count": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "count": float(len(vector)),
        "mean": float(statistics.mean(vector)),
        "std": float(statistics.pstdev(vector)) if len(vector) > 1 else 0.0,
        "min": float(min(vector)),
        "max": float(max(vector)),
        "median": float(statistics.median(vector)),
    }


def pearson_correlation(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    numerator = sum(x * y for x, y in zip(centered_x, centered_y))
    denom_x = math.sqrt(sum(x * x for x in centered_x))
    denom_y = math.sqrt(sum(y * y for y in centered_y))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return float(numerator / (denom_x * denom_y))


def preview_trace(trace: Iterable[Dict[str, Any]], limit: int = 6) -> List[Dict[str, Any]]:
    items = list(trace)
    return items[:limit]


def ensure_punkt_resources(score06: ModuleType) -> None:
    nltk_module = score06.nltk
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for resource_path, package_name in resources:
        try:
            nltk_module.data.find(resource_path)
        except Exception:
            nltk_module.download(package_name)


def compare_row(
    row: Dict[str, Any],
    modern_row: Dict[str, Any],
    legacy_row: Dict[str, Any],
) -> Dict[str, Any]:
    modern_score = safe_float(modern_row.get("unfaithfulness_score"))
    legacy_score = safe_float(legacy_row.get("unfaithfulness_score"))
    delta = modern_score - legacy_score
    return {
        "id": row.get("id"),
        "source_id": row.get("source_id"),
        "error_symbol": row.get("error_symbol"),
        "modern_unfaithfulness_score": round(modern_score, 4),
        "legacy_unfaithfulness_score": round(legacy_score, 4),
        "score_delta": round(delta, 4),
        "abs_score_delta": round(abs(delta), 4),
        "modern_dpo_margin": round(safe_float(modern_row.get("dpo_margin")), 4),
        "legacy_dpo_margin": round(safe_float(legacy_row.get("dpo_margin")), 4),
        "modern_trace": modern_row.get("metrics", {}).get("trace") or [],
        "legacy_trace": legacy_row.get("metrics", {}).get("trace") or [],
    }


def legacy_score_row(
    row: Dict[str, Any],
    nli_state: Dict[str, Any],
    score06: ModuleType,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    patient_context = normalize_text(row.get("patient_context"))
    context_sentences = score06.split_sentences(patient_context)
    total_ctx_sentences = len(context_sentences)
    nodes = normalized_aru_nodes(row)

    history: List[str] = []
    node_packs: List[Dict[str, Any]] = []
    stats = {
        "node_count": 0.0,
        "patient_context_pairs": 0.0,
        "retrieval_pairs": 0.0,
        "history_pairs": 0.0,
        "total_nli_pairs": 0.0,
    }

    for node in nodes:
        node_text = normalize_text(node.get("text"))
        node_type = normalize_text(node.get("type")).upper()[:1] or "C"
        stats["node_count"] += 1.0
        result_pack = {"context_check": [], "doc_check": []}

        if node_text:
            if node_type == "O":
                ctx_entries = score06.run_nli(
                    nli_state,
                    context_sentences,
                    node_text,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )
                result_pack["context_check"] = [pipeline_probs(entry) for entry in ctx_entries]
                stats["patient_context_pairs"] += float(len(ctx_entries))
                stats["total_nli_pairs"] += float(len(ctx_entries))
            elif node_type == "C":
                history_text = " ".join(history[-args.legacy_history_window :]) or patient_context[-1000:]
                if history_text:
                    ctx_entry = score06.single_premise_entry(
                        nli_state,
                        history_text,
                        node_text,
                        batch_size=args.batch_size,
                        max_length=args.max_length,
                    )
                    result_pack["context_check"] = [pipeline_probs(ctx_entry)]
                    stats["history_pairs"] += 1.0
                    stats["total_nli_pairs"] += 1.0

            if node_type in {"W", "D", "O"}:
                docs = score06.evidence_texts(node.get("retrieved_evidence"))
                doc_entries: List[Dict[str, Any]] = []
                for doc_text in docs:
                    doc_entry = score06.single_premise_entry(
                        nli_state,
                        doc_text,
                        node_text,
                        batch_size=args.batch_size,
                        max_length=args.max_length,
                    )
                    doc_entries.append(doc_entry)
                result_pack["doc_check"] = [pipeline_probs(entry) for entry in doc_entries]
                stats["retrieval_pairs"] += float(len(doc_entries))
                stats["total_nli_pairs"] += float(len(doc_entries))

        node_packs.append(result_pack)
        history.append(node_text)

    legacy_result = calculate_legacy_v2_final_score(
        arus=nodes,
        nli_results=node_packs,
        total_ctx_sentences=total_ctx_sentences,
        args=args,
    )
    sample_score = safe_float(legacy_result.get("unfaithfulness_score"))
    dpo_margin = min(sample_score * args.margin_scale, args.max_margin)
    metrics = legacy_result.get("metrics") if isinstance(legacy_result.get("metrics"), dict) else {}

    return {
        "id": row.get("id"),
        "source_id": row.get("source_id"),
        "error_symbol": row.get("error_symbol"),
        "unfaithfulness_score": round(sample_score, 4),
        "dpo_margin": round(float(dpo_margin), 4),
        "verifier": {
            "model": args.nli_model,
            "entailment_threshold": args.entailment_threshold,
            "contradiction_threshold": args.contradiction_threshold,
            "mode": "legacy_6_step4_v2_compatible",
        },
        "metrics": {
            "max_risk": round(safe_float(metrics.get("max_risk")), 4),
            "avg_defect": round(safe_float(metrics.get("avg_defect")), 4),
            "coverage_ratio": round(safe_float(metrics.get("coverage_ratio")), 4),
            "penalty_score": round(safe_float(metrics.get("penalty_score")), 4),
            "covered_sentences": int(metrics.get("covered_sentences", 0) or 0),
            "total_sentences": int(metrics.get("total_sentences", 0) or 0),
            "has_prevalence_excuse": bool(metrics.get("has_prevalence_excuse", False)),
            "target_coverage": round(
                float(
                    args.prevalence_target_coverage
                    if metrics.get("has_prevalence_excuse")
                    else args.default_target_coverage
                ),
                4,
            ),
            "trace": metrics.get("trace") or [],
        },
    }, stats


def render_report(
    args: argparse.Namespace,
    preflight_summary: Dict[str, Any],
    modern_rows: List[Dict[str, Any]],
    legacy_rows: List[Dict[str, Any]],
    comparison_rows: List[Dict[str, Any]],
) -> str:
    modern_scores = [safe_float(row.get("unfaithfulness_score")) for row in modern_rows]
    legacy_scores = [safe_float(row.get("unfaithfulness_score")) for row in legacy_rows]
    deltas = [safe_float(row.get("score_delta")) for row in comparison_rows]
    abs_deltas = [abs(delta) for delta in deltas]
    modern_high_risk = sum(score >= args.high_risk_threshold for score in modern_scores)
    legacy_high_risk = sum(score >= args.high_risk_threshold for score in legacy_scores)
    large_delta = sum(abs(delta) >= args.delta_threshold for delta in deltas)

    modern_summary = summarize_vector(modern_scores)
    legacy_summary = summarize_vector(legacy_scores)
    delta_summary = summarize_vector(deltas)
    abs_delta_summary = summarize_vector(abs_deltas)
    correlation = pearson_correlation(modern_scores, legacy_scores)

    lines = [
        "# Score06 vs Legacy Step4 Comparison",
        "",
        "## Inputs",
        "",
        f"- input_file: `{args.input_file}`",
        f"- output_dir: `{args.output_dir}`",
        f"- nli_model: `{args.nli_model}`",
        f"- batch_size: {args.batch_size}",
        f"- device: `{args.device}`",
        "",
        "## Notes",
        "",
        "- `modern` uses `fa_dpo_pipeline/06_score_faithfulness.py::score_row` directly.",
        "- `legacy` reuses the same stage-05 node-level evidence but applies `待整理code/6_step4_v2.py::calculate_final_score` semantics.",
        "- The comparison isolates scoring-function differences on the same input rows instead of mixing in retrieval or source-file differences.",
        "",
        "## Preflight",
        "",
        f"- status: `{preflight_summary.get('status', 'skipped')}`",
    ]

    blocking = preflight_summary.get("blocking_issues") or []
    warnings = preflight_summary.get("warnings") or []
    lines.append(f"- blocking_issues: {len(blocking)}")
    lines.append(f"- warnings: {len(warnings)}")
    if blocking:
        lines.extend([f"- blocking: {item}" for item in blocking])
    if warnings:
        lines.extend([f"- warning: {item}" for item in warnings[:10]])

    lines.extend(
        [
            "",
            "## Score Summary",
            "",
            (
                f"- modern: count={int(modern_summary['count'])}, mean={modern_summary['mean']:.4f}, "
                f"std={modern_summary['std']:.4f}, min={modern_summary['min']:.4f}, "
                f"median={modern_summary['median']:.4f}, max={modern_summary['max']:.4f}"
            ),
            (
                f"- legacy: count={int(legacy_summary['count'])}, mean={legacy_summary['mean']:.4f}, "
                f"std={legacy_summary['std']:.4f}, min={legacy_summary['min']:.4f}, "
                f"median={legacy_summary['median']:.4f}, max={legacy_summary['max']:.4f}"
            ),
            (
                f"- delta(modern-legacy): mean={delta_summary['mean']:.4f}, "
                f"std={delta_summary['std']:.4f}, min={delta_summary['min']:.4f}, "
                f"median={delta_summary['median']:.4f}, max={delta_summary['max']:.4f}"
            ),
            (
                f"- abs_delta: mean={abs_delta_summary['mean']:.4f}, "
                f"median={abs_delta_summary['median']:.4f}, max={abs_delta_summary['max']:.4f}"
            ),
            f"- pearson_correlation: {correlation:.4f}",
            f"- high_risk_rows(modern, threshold={args.high_risk_threshold}): {modern_high_risk}",
            f"- high_risk_rows(legacy, threshold={args.high_risk_threshold}): {legacy_high_risk}",
            f"- abs_delta>={args.delta_threshold}: {large_delta}/{len(comparison_rows)}",
            "",
            "## Largest Disagreements",
            "",
        ]
    )

    ranked_rows = sorted(
        comparison_rows,
        key=lambda row: safe_float(row.get("abs_score_delta")),
        reverse=True,
    )
    if not ranked_rows:
        lines.append("- none")
        return "\n".join(lines) + "\n"

    for row in ranked_rows[: max(args.top_k_report, 0)]:
        lines.append(
            "- "
            f"id={row.get('id')} source_id={row.get('source_id')} "
            f"modern={safe_float(row.get('modern_unfaithfulness_score')):.4f} "
            f"legacy={safe_float(row.get('legacy_unfaithfulness_score')):.4f} "
            f"delta={safe_float(row.get('score_delta')):.4f}"
        )
        lines.append(f"  modern_trace={preview_trace(row.get('modern_trace') or [])}")
        lines.append(f"  legacy_trace={preview_trace(row.get('legacy_trace') or [])}")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    started_at = iso_utc_now()

    output_dir = Path(args.output_dir)
    modern_score_file = ensure_parent(output_dir / "modern_score06.jsonl")
    legacy_score_file = ensure_parent(output_dir / "legacy_step4_compatible.jsonl")
    comparison_file = ensure_parent(output_dir / "score_comparison.jsonl")
    report_file = ensure_parent(output_dir / "score_comparison_report.md")

    if not args.preflight_report_file:
        args.preflight_report_file = str(output_dir / "preflight_report.md")

    score06 = load_module("fa_dpo_score06", REPO_ROOT / "fa_dpo_pipeline" / "06_score_faithfulness.py")
    rows = load_jsonl_window(args.input_file, args.start, args.end)

    preflight_result: Dict[str, Any] = {"status": "skipped", "blocking_issues": [], "warnings": []}
    if args.preflight_only and args.skip_preflight:
        raise ValueError("`--preflight-only` cannot be combined with `--skip-preflight`.")
    if not args.skip_preflight:
        allowed_ids = {str(row.get("id")) for row in rows if row.get("id") is not None}
        source_rows = score06.maybe_load_rows(args.source_aru_file) if args.source_aru_file else []
        if allowed_ids and source_rows:
            source_rows = filter_rows_by_ids(source_rows, allowed_ids)
        stage05_metadata = score06.maybe_load_json(args.stage05_metadata_file) if args.stage05_metadata_file else {}
        preflight_result = score06.preflight_summary(
            rows=rows,
            source_rows=source_rows,
            stage05_metadata=stage05_metadata,
            args=args,
        )
        preflight_text = score06.render_preflight_report(preflight_result, args)
        Path(args.preflight_report_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.preflight_report_file).write_text(preflight_text, encoding="utf-8")
        if preflight_result.get("blocking_issues"):
            metadata_path = write_run_metadata(
                stage_name="compare_score06_vs_legacy_step4",
                args=args,
                inputs={"input_file": args.input_file},
                outputs={"preflight_report_file": args.preflight_report_file},
                stats={"preflight": preflight_result},
                metadata_file=args.metadata_file,
                started_at=started_at,
                finished_at=iso_utc_now(),
                status="blocked_preflight",
            )
            raise RuntimeError(
                "Preflight failed. Inspect the report before scoring: "
                f"{args.preflight_report_file}\nRun metadata: {metadata_path}"
            )
        if args.strict_preflight and preflight_result.get("warnings"):
            metadata_path = write_run_metadata(
                stage_name="compare_score06_vs_legacy_step4",
                args=args,
                inputs={"input_file": args.input_file},
                outputs={"preflight_report_file": args.preflight_report_file},
                stats={"preflight": preflight_result},
                metadata_file=args.metadata_file,
                started_at=started_at,
                finished_at=iso_utc_now(),
                status="blocked_preflight_warning",
            )
            raise RuntimeError(
                "Preflight raised warnings and `--strict-preflight` is enabled. "
                f"Inspect: {args.preflight_report_file}\nRun metadata: {metadata_path}"
            )
        if args.preflight_only:
            metadata_path = write_run_metadata(
                stage_name="compare_score06_vs_legacy_step4",
                args=args,
                inputs={"input_file": args.input_file},
                outputs={"preflight_report_file": args.preflight_report_file},
                stats={"preflight": preflight_result},
                metadata_file=args.metadata_file,
                started_at=started_at,
                finished_at=iso_utc_now(),
                status="preflight_only",
            )
            print(f"Preflight-only complete: {args.preflight_report_file}")
            print(f"Run metadata: {metadata_path}")
            return

    if modern_score_file.exists():
        modern_score_file.unlink()
    if legacy_score_file.exists():
        legacy_score_file.unlink()
    if comparison_file.exists():
        comparison_file.unlink()

    if any(
        getattr(score06, name, None) is None
        for name in ("torch", "AutoModelForSequenceClassification", "AutoTokenizer")
    ):
        raise RuntimeError(
            "Missing scoring dependencies in the active Python environment. "
            "Install working torch + transformers first, then rerun the comparison script."
        )

    ensure_punkt_resources(score06)
    nli_state = score06.load_nli(args)

    modern_rows: List[Dict[str, Any]] = []
    legacy_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []

    for row in score06.tqdm(rows, desc="Comparing score methods"):
        modern_row, _ = score06.score_row(row, nli_state, args)
        legacy_row, _ = legacy_score_row(row, nli_state, score06, args)
        comparison_row = compare_row(row, modern_row, legacy_row)

        modern_rows.append(modern_row)
        legacy_rows.append(legacy_row)
        comparison_rows.append(comparison_row)

        append_jsonl(modern_score_file, modern_row)
        append_jsonl(legacy_score_file, legacy_row)
        append_jsonl(comparison_file, comparison_row)

    report_text = render_report(
        args=args,
        preflight_summary=preflight_result,
        modern_rows=modern_rows,
        legacy_rows=legacy_rows,
        comparison_rows=comparison_rows,
    )
    report_file.write_text(report_text, encoding="utf-8")

    modern_scores = [safe_float(row.get("unfaithfulness_score")) for row in modern_rows]
    legacy_scores = [safe_float(row.get("unfaithfulness_score")) for row in legacy_rows]
    deltas = [safe_float(row.get("score_delta")) for row in comparison_rows]
    metadata_path = write_run_metadata(
        stage_name="compare_score06_vs_legacy_step4",
        args=args,
        inputs={"input_file": args.input_file},
        outputs={
            "modern_score_file": modern_score_file,
            "legacy_score_file": legacy_score_file,
            "comparison_file": comparison_file,
            "report_file": report_file,
            "preflight_report_file": args.preflight_report_file if not args.skip_preflight else "",
        },
        stats={
            "rows": len(rows),
            "modern_mean": round(statistics.mean(modern_scores), 6) if modern_scores else 0.0,
            "legacy_mean": round(statistics.mean(legacy_scores), 6) if legacy_scores else 0.0,
            "delta_mean": round(statistics.mean(deltas), 6) if deltas else 0.0,
            "abs_delta_mean": round(statistics.mean([abs(delta) for delta in deltas]), 6) if deltas else 0.0,
            "pearson_correlation": round(pearson_correlation(modern_scores, legacy_scores), 6),
            "delta_threshold": args.delta_threshold,
            "large_delta_rows": sum(abs(delta) >= args.delta_threshold for delta in deltas),
            "high_risk_threshold": args.high_risk_threshold,
            "modern_high_risk_rows": sum(score >= args.high_risk_threshold for score in modern_scores),
            "legacy_high_risk_rows": sum(score >= args.high_risk_threshold for score in legacy_scores),
            "preflight": preflight_result,
        },
        metadata_file=args.metadata_file,
        started_at=started_at,
        finished_at=iso_utc_now(),
    )

    print(f"Modern scores: {modern_score_file}")
    print(f"Legacy scores: {legacy_score_file}")
    print(f"Comparison JSONL: {comparison_file}")
    print(f"Comparison report: {report_file}")
    if not args.skip_preflight:
        print(f"Preflight report: {args.preflight_report_file}")
    print(f"Run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
