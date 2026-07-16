#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from datasets import load_dataset
from tqdm import tqdm

from common import artifact_path, ensure_parent, iso_utc_now, normalize_text, write_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a JSONL retrieval corpus from the MIRIAD dataset."
    )
    parser.add_argument(
        "--dataset-name",
        default="miriad/miriad-5.8M",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to export.",
    )
    parser.add_argument(
        "--output-file",
        default=artifact_path("fa_dpo_pipeline", "miriad_corpus.jsonl"),
        help="JSONL corpus output path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Optional debug limit. -1 means all rows.",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=20,
        help="Skip passages shorter than this many characters.",
    )
    parser.add_argument(
        "--sep-token",
        default=" [SEP] ",
        help="Separator inserted between title and passage.",
    )
    parser.add_argument(
        "--metadata-file",
        default=artifact_path("fa_dpo_pipeline", "run_metadata", "00_build_miriad_corpus.json"),
        help="Where to write run metadata JSON.",
    )
    return parser.parse_args()


def build_doc(
    sample: Dict[str, Any],
    sep_token: str,
    min_text_length: int,
    dataset_name: str,
    split: str,
) -> Dict[str, Any] | None:
    title = normalize_text(sample.get("paper_title"))
    passage = normalize_text(sample.get("passage_text"))
    if not passage or len(passage) < min_text_length:
        return None
    text = f"{title}{sep_token}{passage}" if title else passage
    return {
        "_id": normalize_text(sample.get("qa_id")),
        "title": title,
        "text": text,
        "metadata": {
            "dataset_name": dataset_name,
            "dataset_split": split,
            "specialty": sample.get("specialty"),
            "year": sample.get("year"),
            "paper_url": sample.get("paper_url"),
        },
    }


def main() -> None:
    args = parse_args()
    started_at = iso_utc_now()
    output_path = ensure_parent(args.output_file)
    dataset = load_dataset(args.dataset_name, split=args.split)
    total = len(dataset) if args.limit < 0 else min(args.limit, len(dataset))

    kept = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(tqdm(dataset, total=total, desc="Building corpus")):
            if args.limit >= 0 and index >= args.limit:
                break
            doc = build_doc(
                sample,
                sep_token=args.sep_token,
                min_text_length=args.min_text_length,
                dataset_name=args.dataset_name,
                split=args.split,
            )
            if doc is None:
                skipped += 1
                continue
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1

    metadata_path = write_run_metadata(
        stage_name="00_build_miriad_corpus",
        args=args,
        inputs={"dataset_name": args.dataset_name, "split": args.split},
        outputs={"output_file": output_path},
        stats={"kept_docs": kept, "skipped_docs": skipped, "requested_total": total},
        metadata_file=args.metadata_file,
        started_at=started_at,
        finished_at=iso_utc_now(),
    )
    print(f"Saved corpus: {output_path}")
    print(f"Kept docs: {kept}")
    print(f"Skipped docs: {skipped}")
    print(f"Run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
