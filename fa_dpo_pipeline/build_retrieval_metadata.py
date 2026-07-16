#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from common import artifact_path, output_path, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize corpus stats and retrieval pipeline configuration for the Fa-DPO verifier."
    )
    parser.add_argument(
        "--corpus-file",
        default=artifact_path("fa_dpo_pipeline", "miriad_corpus.jsonl"),
        help="Retrieval corpus JSONL.",
    )
    parser.add_argument(
        "--index-file",
        default=artifact_path("fa_dpo_pipeline", "miriad_faiss.index"),
        help="FAISS index path.",
    )
    parser.add_argument(
        "--corpus-name",
        default="miriad/miriad-5.8M",
        help="Human-readable corpus source name.",
    )
    parser.add_argument(
        "--corpus-license",
        default="unknown",
        help="Corpus license string for reporting.",
    )
    parser.add_argument(
        "--article-encoder",
        default="ncbi/MedCPT-Article-Encoder",
        help="Document encoder used to build the index.",
    )
    parser.add_argument(
        "--query-encoder",
        default="ncbi/MedCPT-Query-Encoder",
        help="Query encoder used for retrieval.",
    )
    parser.add_argument(
        "--reranker-model",
        default="ncbi/MedCPT-Cross-Encoder",
        help="Cross-encoder reranker name.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=50,
        help="Initial retriever depth.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=50,
        help="How many retrieved docs are reranked.",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=5,
        help="How many docs are kept per query.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token length used by retriever and reranker.",
    )
    parser.add_argument(
        "--index-metadata-file",
        default=artifact_path("fa_dpo_pipeline", "run_metadata", "01_build_miriad_index.json"),
        help="Optional stage-01 metadata JSON.",
    )
    parser.add_argument(
        "--retrieval-metadata-file",
        default=artifact_path("fa_dpo_pipeline", "run_metadata", "05_attach_aru_evidence.json"),
        help="Optional stage-05 metadata JSON.",
    )
    parser.add_argument(
        "--corpus-stats-output",
        default=output_path("retrieval_protocol", "corpus_stats.json"),
        help="Where to write corpus stats JSON.",
    )
    parser.add_argument(
        "--pipeline-config-output",
        default=output_path("retrieval_protocol", "pipeline_config.json"),
        help="Where to write pipeline config JSON.",
    )
    return parser.parse_args()


def maybe_read_json(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return read_json(file_path)


def collect_corpus_stats(path: str, corpus_name: str, corpus_license: str) -> Dict[str, Any]:
    chunk_count = 0
    char_lengths = []
    token_lengths = []
    paper_urls = set()
    dataset_names = Counter()
    specialties = Counter()
    years = Counter()

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get("text", ""))
            metadata = row.get("metadata") or {}
            chunk_count += 1
            char_lengths.append(len(text))
            token_lengths.append(len(text.split()))
            if metadata.get("paper_url"):
                paper_urls.add(str(metadata["paper_url"]))
            if metadata.get("dataset_name"):
                dataset_names[str(metadata["dataset_name"])] += 1
            if metadata.get("specialty"):
                specialties[str(metadata["specialty"])] += 1
            if metadata.get("year"):
                years[str(metadata["year"])] += 1

    return {
        "corpus_name": corpus_name,
        "license": corpus_license,
        "chunk_count": chunk_count,
        "document_count": len(paper_urls) or chunk_count,
        "avg_chunk_chars": round(sum(char_lengths) / max(len(char_lengths), 1), 4),
        "avg_chunk_tokens": round(sum(token_lengths) / max(len(token_lengths), 1), 4),
        "dataset_source_breakdown": dict(dataset_names),
        "top_specialties": dict(specialties.most_common(20)),
        "year_breakdown": dict(years),
    }


def main() -> None:
    args = parse_args()
    corpus_stats = collect_corpus_stats(args.corpus_file, args.corpus_name, args.corpus_license)
    write_json(args.corpus_stats_output, corpus_stats)

    pipeline_config = {
        "corpus_file": args.corpus_file,
        "index_file": args.index_file,
        "article_encoder": args.article_encoder,
        "query_encoder": args.query_encoder,
        "reranker_model": args.reranker_model,
        "retrieval_top_k": args.retrieval_top_k,
        "rerank_top_k": args.rerank_top_k,
        "final_top_k": args.final_top_k,
        "max_length": args.max_length,
        "index_stage_metadata": maybe_read_json(args.index_metadata_file),
        "retrieval_stage_metadata": maybe_read_json(args.retrieval_metadata_file),
    }
    write_json(args.pipeline_config_output, pipeline_config)

    print(f"Corpus stats: {args.corpus_stats_output}")
    print(f"Pipeline config: {args.pipeline_config_output}")


if __name__ == "__main__":
    main()
