#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from common import artifact_path, build_full_response, iso_utc_now, normalize_text, result_path, write_jsonl, write_run_metadata

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - dependency availability is environment-specific
    AutoTokenizer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert scored ARU negatives into Topo-DPO training/eval JSONL."
    )
    parser.add_argument(
        "--negatives-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.strict_clean.jsonl"),
        help="Strict-clean negatives JSONL.",
    )
    parser.add_argument(
        "--aru-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.strict_clean.arus.with_evidence.jsonl"),
        help="ARU JSONL with evidence.",
    )
    parser.add_argument(
        "--score-file",
        default=result_path("fa_dpo_pipeline", "medcase_unfaithful_negatives.strict_clean.scores.jsonl"),
        help="Faithfulness score JSONL.",
    )
    parser.add_argument(
        "--base-model",
        default="TsinghuaC3I/Llama-3.1-8B-UltraMedical",
        help="Tokenizer checkpoint used to build topo masks.",
    )
    parser.add_argument(
        "--output-dir",
        default=artifact_path("fa_dpo_pipeline", "topodpo_data"),
        help="Where to write train/eval JSONL.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.05,
        help="Eval split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Split seed.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Error-region amplification factor.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Background weight for non-error regions.",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.1,
        help="Only ARUs above this score get an amplified mask weight.",
    )
    parser.add_argument(
        "--metadata-file",
        default=artifact_path("fa_dpo_pipeline", "run_metadata", "07_prepare_topodpo_data.json"),
        help="Where to write run metadata JSON.",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_reasoning_from_arus(aru_nodes: List[Dict]) -> str:
    parts = [normalize_text(node.get("text")) for node in aru_nodes]
    return "\n".join(part for part in parts if part)


def sequential_spans(full_text: str, aru_texts: List[str]) -> List[Tuple[int, int]] | None:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for aru in aru_texts:
        fragment = normalize_text(aru)
        if not fragment:
            spans.append((cursor, cursor))
            continue
        position = full_text.find(fragment, cursor)
        if position < 0:
            pattern = re.escape(fragment)
            pattern = pattern.replace(r"\ ", r"\s+")
            match = re.search(pattern, full_text[cursor:], flags=re.DOTALL)
            if match is None:
                return None
            start = cursor + match.start()
            end = cursor + match.end()
        else:
            start = position
            end = position + len(fragment)
        spans.append((start, end))
        cursor = end
    return spans


def build_topo_mask(
    tokenizer: AutoTokenizer,
    full_text: str,
    aru_texts: List[str],
    aru_scores: List[float],
    alpha: float,
    epsilon: float,
    risk_threshold: float,
) -> List[float] | None:
    spans = sequential_spans(full_text, aru_texts)
    if spans is None:
        return None
    weights: List[float] = []
    for score in aru_scores:
        weights.append(1.0 + alpha * score if score > risk_threshold else epsilon)

    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding.offset_mapping
    mask: List[float] = []
    span_index = 0
    for start, end in offsets:
        if start == end:
            mask.append(epsilon)
            continue
        midpoint = (start + end) / 2.0
        while span_index < len(spans) and midpoint >= spans[span_index][1]:
            span_index += 1
        if span_index >= len(spans):
            mask.append(epsilon)
            continue
        span_start, span_end = spans[span_index]
        if span_start <= midpoint < span_end:
            mask.append(weights[span_index])
        else:
            mask.append(epsilon)
    return mask


def load_jsonl_tolerant(path: str) -> tuple[List[Dict], int]:
    rows: List[Dict] = []
    malformed = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                malformed += 1
                if malformed <= 5:
                    print(
                        f"Warning: skipping malformed JSONL line {line_number} in {path}: {exc}",
                        flush=True,
                    )
    if malformed:
        print(f"Skipped {malformed} malformed JSONL lines from {path}", flush=True)
    return rows, malformed


def split_dataset(rows: List[Dict], eval_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    split_index = int(len(shuffled) * (1.0 - eval_ratio))
    return shuffled[:split_index], shuffled[split_index:]


def main() -> None:
    args = parse_args()
    started_at = iso_utc_now()
    if AutoTokenizer is None:
        raise RuntimeError("Missing tokenizer dependencies. Install transformers (and torch) first.")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    negative_rows, malformed_negatives = load_jsonl_tolerant(args.negatives_file)
    aru_rows, malformed_arus = load_jsonl_tolerant(args.aru_file)
    score_rows, malformed_scores = load_jsonl_tolerant(args.score_file)
    negatives = {str(row.get("id")): row for row in negative_rows}
    arus = {str(row.get("id")): row for row in aru_rows}
    scores = {str(row.get("id")): row for row in score_rows}

    core_dataset: List[Dict] = []
    skipped_prompt = 0
    skipped_score = 0
    skipped_aru = 0
    skipped_topo = 0
    skipped_trace_aru_mismatch = 0
    skipped_topo_alignment = 0
    rebuilt_rejected_rows = 0
    fallback_original_rejected_rows = 0
    original_rejected_diff_rows = 0

    for row_id, negative_row in tqdm(negatives.items(), desc="Building preference rows"):
        chosen = build_full_response(
            negative_row.get("reasoning") or negative_row.get("correct_reasoning"),
            negative_row.get("final_answer") or negative_row.get("correct_answer_text"),
        )
        negative_answer = negative_row.get("negative_final_answer") or negative_row.get("predicted_answer_text")
        original_rejected = build_full_response(
            negative_row.get("negative_reasoning") or negative_row.get("predicted_reasoning"),
            negative_answer,
        )
        prompt = normalize_text(negative_row.get("patient_context") or negative_row.get("prompt"))
        if not chosen or not prompt:
            skipped_prompt += 1
            continue

        score_row = scores.get(row_id)
        aru_row = arus.get(row_id)
        if score_row is None:
            skipped_score += 1
            continue
        if aru_row is None:
            skipped_aru += 1
            continue

        trace = sorted(score_row.get("metrics", {}).get("trace", []), key=lambda item: item.get("id", 0))
        aru_nodes = aru_row.get("arus", [])
        if len(trace) != len(aru_nodes) or not aru_nodes:
            skipped_topo += 1
            skipped_trace_aru_mismatch += 1
            continue

        aru_texts = [normalize_text(node.get("text")) for node in aru_nodes]
        aru_scores = [float(item.get("s", 0.0)) for item in trace]
        rebuilt_reasoning = build_reasoning_from_arus(aru_nodes)
        rebuilt_rejected = build_full_response(rebuilt_reasoning, negative_answer)
        if original_rejected and rebuilt_rejected and original_rejected != rebuilt_rejected:
            original_rejected_diff_rows += 1

        rejected = rebuilt_rejected or original_rejected
        topo_mask = build_topo_mask(
            tokenizer=tokenizer,
            full_text=rejected,
            aru_texts=aru_texts,
            aru_scores=aru_scores,
            alpha=args.alpha,
            epsilon=args.epsilon,
            risk_threshold=args.risk_threshold,
        )
        if topo_mask is None:
            if not original_rejected or original_rejected == rejected:
                skipped_topo += 1
                skipped_topo_alignment += 1
                continue
            fallback_topo_mask = build_topo_mask(
                tokenizer=tokenizer,
                full_text=original_rejected,
                aru_texts=aru_texts,
                aru_scores=aru_scores,
                alpha=args.alpha,
                epsilon=args.epsilon,
                risk_threshold=args.risk_threshold,
            )
            if fallback_topo_mask is None:
                skipped_topo += 1
                skipped_topo_alignment += 1
                continue
            rejected = original_rejected
            topo_mask = fallback_topo_mask
            fallback_original_rejected_rows += 1
        else:
            rebuilt_rejected_rows += 1

        base_row = {
            "id": row_id,
            "source_id": negative_row.get("source_id"),
            "error_symbol": negative_row.get("error_symbol"),
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

        core_dataset.append(
            {
                **base_row,
                "margin": float(score_row.get("dpo_margin", 0.0)),
                "topo_mask": topo_mask,
            }
        )

    train_core_rows, eval_core_rows = split_dataset(core_dataset, args.eval_ratio, args.seed)

    train_dpo_rows = [
        {
            "id": row["id"],
            "source_id": row.get("source_id"),
            "error_symbol": row.get("error_symbol"),
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }
        for row in train_core_rows
    ]
    eval_dpo_rows = [
        {
            "id": row["id"],
            "source_id": row.get("source_id"),
            "error_symbol": row.get("error_symbol"),
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }
        for row in eval_core_rows
    ]
    train_margindpo_rows = [
        {
            "id": row["id"],
            "source_id": row.get("source_id"),
            "error_symbol": row.get("error_symbol"),
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "margin": row["margin"],
        }
        for row in train_core_rows
    ]
    eval_margindpo_rows = [
        {
            "id": row["id"],
            "source_id": row.get("source_id"),
            "error_symbol": row.get("error_symbol"),
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "margin": row["margin"],
        }
        for row in eval_core_rows
    ]
    train_fadpo_rows = [
        {
            "id": row["id"],
            "source_id": row.get("source_id"),
            "error_symbol": row.get("error_symbol"),
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "topo_mask": row["topo_mask"],
            "margin": row["margin"],
        }
        for row in train_core_rows
    ]
    eval_fadpo_rows = [
        {
            "id": row["id"],
            "source_id": row.get("source_id"),
            "error_symbol": row.get("error_symbol"),
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "topo_mask": row["topo_mask"],
            "margin": row["margin"],
        }
        for row in eval_core_rows
    ]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train_dpo.jsonl", train_dpo_rows)
    write_jsonl(output_dir / "eval_dpo.jsonl", eval_dpo_rows)
    write_jsonl(output_dir / "train_margindpo.jsonl", train_margindpo_rows)
    write_jsonl(output_dir / "eval_margindpo.jsonl", eval_margindpo_rows)
    write_jsonl(output_dir / "train_fadpo.jsonl", train_fadpo_rows)
    write_jsonl(output_dir / "eval_fadpo.jsonl", eval_fadpo_rows)
    write_jsonl(output_dir / "train_topodpo.jsonl", train_fadpo_rows)
    write_jsonl(output_dir / "eval_topodpo.jsonl", eval_fadpo_rows)

    metadata_path = write_run_metadata(
        stage_name="07_prepare_topodpo_data",
        args=args,
        inputs={
            "negatives_file": args.negatives_file,
            "aru_file": args.aru_file,
            "score_file": args.score_file,
        },
        outputs={
            "output_dir": output_dir,
            "train_dpo_file": output_dir / "train_dpo.jsonl",
            "eval_dpo_file": output_dir / "eval_dpo.jsonl",
            "train_margindpo_file": output_dir / "train_margindpo.jsonl",
            "eval_margindpo_file": output_dir / "eval_margindpo.jsonl",
            "train_fadpo_file": output_dir / "train_fadpo.jsonl",
            "eval_fadpo_file": output_dir / "eval_fadpo.jsonl",
            "train_topodpo_file_compat": output_dir / "train_topodpo.jsonl",
            "eval_topodpo_file_compat": output_dir / "eval_topodpo.jsonl",
        },
        stats={
            "matched_core_total_rows": len(core_dataset),
            "matched_core_train_rows": len(train_core_rows),
            "matched_core_eval_rows": len(eval_core_rows),
            "dpo_total_rows": len(train_dpo_rows) + len(eval_dpo_rows),
            "dpo_train_rows": len(train_dpo_rows),
            "dpo_eval_rows": len(eval_dpo_rows),
            "margindpo_total_rows": len(train_margindpo_rows) + len(eval_margindpo_rows),
            "margindpo_train_rows": len(train_margindpo_rows),
            "margindpo_eval_rows": len(eval_margindpo_rows),
            "fadpo_total_rows": len(core_dataset),
            "fadpo_train_rows": len(train_fadpo_rows),
            "fadpo_eval_rows": len(eval_fadpo_rows),
            "skipped_missing_prompt_rows": skipped_prompt,
            "skipped_missing_score_rows": skipped_score,
            "skipped_missing_aru_rows": skipped_aru,
            "skipped_invalid_topo_rows": skipped_topo,
            "skipped_trace_aru_mismatch_rows": skipped_trace_aru_mismatch,
            "skipped_topo_alignment_rows": skipped_topo_alignment,
            "rebuilt_rejected_rows": rebuilt_rejected_rows,
            "fallback_original_rejected_rows": fallback_original_rejected_rows,
            "original_rejected_diff_rows": original_rejected_diff_rows,
            "input_negative_rows": len(negative_rows),
            "input_aru_rows": len(aru_rows),
            "input_score_rows": len(score_rows),
            "malformed_negative_lines": malformed_negatives,
            "malformed_aru_lines": malformed_arus,
            "malformed_score_lines": malformed_scores,
        },
        metadata_file=args.metadata_file,
        started_at=started_at,
        finished_at=iso_utc_now(),
    )

    print(f"Matched core rows: {len(core_dataset)}")
    print(f"Matched core train rows: {len(train_core_rows)}")
    print(f"Matched core eval rows: {len(eval_core_rows)}")
    print(f"DPO train rows: {len(train_dpo_rows)}")
    print(f"DPO eval rows: {len(eval_dpo_rows)}")
    print(f"marginDPO train rows: {len(train_margindpo_rows)}")
    print(f"marginDPO eval rows: {len(eval_margindpo_rows)}")
    print(f"FaDPO train rows: {len(train_fadpo_rows)}")
    print(f"FaDPO eval rows: {len(eval_fadpo_rows)}")
    print(f"Skipped missing prompt rows: {skipped_prompt}")
    print(f"Skipped missing score rows: {skipped_score}")
    print(f"Skipped missing ARU rows: {skipped_aru}")
    print(f"Skipped invalid topo rows: {skipped_topo}")
    print(f"Skipped trace/ARU mismatch rows: {skipped_trace_aru_mismatch}")
    print(f"Skipped topo alignment rows: {skipped_topo_alignment}")
    print(f"Rebuilt rejected rows: {rebuilt_rejected_rows}")
    print(f"Fallback original rejected rows: {fallback_original_rejected_rows}")
    print(f"Original/rebuilt rejected diff rows: {original_rejected_diff_rows}")
    print(f"Malformed negatives lines: {malformed_negatives}")
    print(f"Malformed ARU lines: {malformed_arus}")
    print(f"Malformed score lines: {malformed_scores}")
    print(f"Output dir: {output_dir}")
    print(f"DPO train file: {output_dir / 'train_dpo.jsonl'}")
    print(f"DPO eval file: {output_dir / 'eval_dpo.jsonl'}")
    print(f"marginDPO train file: {output_dir / 'train_margindpo.jsonl'}")
    print(f"marginDPO eval file: {output_dir / 'eval_margindpo.jsonl'}")
    print(f"FaDPO train file: {output_dir / 'train_fadpo.jsonl'}")
    print(f"FaDPO eval file: {output_dir / 'eval_fadpo.jsonl'}")
    print(f"Compat TopoDPO train file: {output_dir / 'train_topodpo.jsonl'}")
    print(f"Compat TopoDPO eval file: {output_dir / 'eval_topodpo.jsonl'}")
    print(f"Run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
