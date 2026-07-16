#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

from tqdm import tqdm

from common import artifact_path, ensure_parent, iso_utc_now, write_run_metadata

try:
    import faiss
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - dependency availability is environment-specific
    faiss = None
    np = None
    torch = None
    AutoModel = None
    AutoTokenizer = None


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="Build a FAISS index for the exported MIRIAD corpus."
    )
    parser.add_argument(
        "--corpus-file",
        default=artifact_path("fa_dpo_pipeline", "miriad_corpus.jsonl"),
        help="JSONL corpus file from 00_build_miriad_corpus.py.",
    )
    parser.add_argument(
        "--index-file",
        default=artifact_path("fa_dpo_pipeline", "miriad_faiss.index"),
        help="Where to write the FAISS index.",
    )
    parser.add_argument(
        "--offsets-file",
        default=artifact_path("fa_dpo_pipeline", "miriad_medcpt.offsets.npy"),
        help="Where to write corpus line offsets for random-access retrieval.",
    )
    parser.add_argument(
        "--encoder",
        default="ncbi/MedCPT-Article-Encoder",
        help="Document encoder checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max encoder sequence length.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Optional document limit. -1 means all rows.",
    )
    parser.add_argument(
        "--device",
        default=default_device,
        help="Torch device. Use `cuda` to leverage all visible GPUs via DataParallel.",
    )
    parser.add_argument(
        "--metadata-file",
        default=artifact_path("fa_dpo_pipeline", "run_metadata", "01_build_miriad_index.json"),
        help="Where to write run metadata JSON.",
    )
    parser.add_argument(
        "--write-offsets-only",
        action="store_true",
        help="Only write corpus line offsets without encoding documents or building a FAISS index.",
    )
    return parser.parse_args()


def count_lines(path: Path, limit: int) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in tqdm(handle, desc="Counting corpus rows", unit="rows"):
            total += 1
            if 0 <= limit == total:
                break
    return total


def batched_corpus(path: Path, batch_size: int, limit: int) -> Iterable[Tuple[List[int], List[str]]]:
    offsets: List[int] = []
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            row = json.loads(line)
            offsets.append(offset)
            texts.append(str(row.get("text", "")))
            if 0 <= limit == len(texts):
                yield offsets, texts
                return
            if len(texts) >= batch_size:
                yield offsets, texts
                offsets, texts = [], []
    if texts:
        yield offsets, texts


def collect_offsets(path: Path, limit: int) -> List[int]:
    offsets: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        with tqdm(desc="Collecting offsets", unit="rows") as progress:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                offsets.append(offset)
                progress.update(1)
                if 0 <= limit == len(offsets):
                    break
    return offsets


def encode_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_length: int,
) -> np.ndarray:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
    return cls_embeddings


def main() -> None:
    args = parse_args()
    started_at = iso_utc_now()
    if np is None:
        raise RuntimeError("Missing numpy dependency. Install numpy first.")
    corpus_path = Path(args.corpus_file)
    offsets_path = ensure_parent(args.offsets_file)

    if args.write_offsets_only:
        all_offsets = collect_offsets(corpus_path, args.limit)
        np.save(offsets_path, np.asarray(all_offsets, dtype=np.int64))
        metadata_path = write_run_metadata(
            stage_name="01_build_miriad_index",
            args=args,
            inputs={"corpus_file": corpus_path},
            outputs={"offsets_file": offsets_path},
            stats={
                "indexed_documents": 0,
                "offset_documents": len(all_offsets),
                "embedding_dim": None,
                "encoder": None,
                "write_offsets_only": True,
            },
            metadata_file=args.metadata_file,
            started_at=started_at,
            finished_at=iso_utc_now(),
        )
        print(f"Offset documents: {len(all_offsets)}")
        print(f"Offsets file: {offsets_path}")
        print(f"Run metadata: {metadata_path}")
        return

    total_rows = count_lines(corpus_path, args.limit)

    if any(module is None for module in [faiss, torch, AutoModel, AutoTokenizer]):
        raise RuntimeError(
            "Missing index-build dependencies. Install faiss/torch/transformers first."
        )
    index_path = ensure_parent(args.index_file)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = AutoModel.from_pretrained(args.encoder)
    if args.device.startswith("cuda") and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)
    model.eval()

    index: faiss.IndexFlatIP | None = None
    all_offsets: List[int] = []
    processed = 0

    for batch_offsets, batch_texts in tqdm(
        batched_corpus(corpus_path, batch_size=args.batch_size, limit=args.limit),
        total=(total_rows + args.batch_size - 1) // args.batch_size,
        desc="Encoding corpus",
    ):
        embeddings = encode_texts(
            batch_texts,
            tokenizer=tokenizer,
            model=model,
            device=args.device,
            max_length=args.max_length,
        )
        if index is None:
            index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        all_offsets.extend(batch_offsets)
        processed += len(batch_texts)

    if index is None:
        raise RuntimeError("No documents were encoded; index build aborted.")

    faiss.write_index(index, str(index_path))
    np.save(offsets_path, np.asarray(all_offsets, dtype=np.int64))

    metadata_path = write_run_metadata(
        stage_name="01_build_miriad_index",
        args=args,
        inputs={"corpus_file": corpus_path},
        outputs={"index_file": index_path, "offsets_file": offsets_path},
        stats={
            "indexed_documents": processed,
            "offset_documents": len(all_offsets),
            "embedding_dim": index.d,
            "encoder": args.encoder,
            "write_offsets_only": False,
        },
        metadata_file=args.metadata_file,
        started_at=started_at,
        finished_at=iso_utc_now(),
    )
    print(f"Indexed documents: {processed}")
    print(f"FAISS index: {index_path}")
    print(f"Offsets file: {offsets_path}")
    print(f"Run metadata: {metadata_path}")


if __name__ == "__main__":
    main()
