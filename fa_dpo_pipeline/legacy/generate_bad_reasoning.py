from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from json import JSONDecodeError, JSONDecoder
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from openai import OpenAI


# ================= Config =================
# Set "api" to use a cloud API, or "local" to use an OpenAI-compatible local server.
LLM_MODE = "local"

# API mode: you can paste your API key directly here.
OPENAI_API_KEY = "-90GoVjbXc2QD2klo9XymWIB8VKh5nKcEoJ1VYD937ixmxi7n"
OPENAI_BASE_URL = "https://api.key77qiqi.cn/v1"
OPENAI_MODEL = "gpt-5.4"

# Local mode: adjust these to match your local OpenAI-compatible service.
LOCAL_API_BASE = "http://localhost:8886/v1"
LOCAL_API_KEY = "EMPTY"
LOCAL_MODEL_NAME = "/home/models/Qwen3-Next-80B-A3B-Instruct"


DEFAULT_MEDCASE_DATASET_NAME = "zou-lab/MedCaseReasoning"
DEFAULT_RUNTIME_ROOT = os.getenv("FA_DPO_RUNTIME_ROOT", "").strip() or os.getcwd()
DEFAULT_DATA_ROOT = os.getenv("FA_DPO_DATA_ROOT", "").strip()
if DEFAULT_DATA_ROOT:
    if os.path.isabs(DEFAULT_DATA_ROOT):
        _default_data_root = DEFAULT_DATA_ROOT
    else:
        _default_data_root = os.path.join(DEFAULT_RUNTIME_ROOT, DEFAULT_DATA_ROOT)
else:
    _default_data_root = os.path.join(DEFAULT_RUNTIME_ROOT, "data")

DEFAULT_MEDCASE_DATASET_DIR = os.path.join(_default_data_root, "hf_datasets", "MedCaseReasoning")
MEDCASE_REMOTE_FILES = {
    "README.md": (
        "https://huggingface.co/datasets/zou-lab/MedCaseReasoning/"
        "resolve/main/README.md?download=true"
    ),
    "data/train-00000-of-00001.parquet": (
        "https://huggingface.co/datasets/zou-lab/MedCaseReasoning/"
        "resolve/main/data/train-00000-of-00001.parquet?download=true"
    ),
    "data/val-00000-of-00001.parquet": (
        "https://huggingface.co/datasets/zou-lab/MedCaseReasoning/"
        "resolve/main/data/val-00000-of-00001.parquet?download=true"
    ),
    "data/test-00000-of-00001.parquet": (
        "https://huggingface.co/datasets/zou-lab/MedCaseReasoning/"
        "resolve/main/data/test-00000-of-00001.parquet?download=true"
    ),
}
MEDCASE_SPLIT_TO_FILE = {
    "train": "data/train-00000-of-00001.parquet",
    "val": "data/val-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
MEDCASE_SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
}


ERROR_SPECS: Dict[str, Dict[str, str]] = {
    "F1": {
        "name": "evidence_omission",
        "interface": "(X,G)->R",
        "definition": (
            "Clinically important available evidence is omitted or mentioned "
            "without being substantively used in the reasoning."
        ),
        "instruction": (
            "Remove, down-weight, or fail to use key positive findings, key "
            "negative findings, contradictory evidence, or essential support "
            "already available in the case context or evidence. Do not invent "
            "new facts; the main failure should be underuse of existing evidence. "
            "Do not merely shorten the trace; make one local evidential step lose "
            "its diagnostic force while keeping the surrounding structure intact."
        ),
    },
    "F2": {
        "name": "evidence_misinterpretation",
        "interface": "(X,G)->R",
        "definition": (
            "Available evidence is used, but its medical meaning is interpreted "
            "incorrectly."
        ),
        "instruction": (
            "Keep the evidence in play, but misread its significance: for "
            "example thresholds, units, reference ranges, temporal order, "
            "demographic adjustment, literature meaning, or pathophysiologic "
            "implication. The evidence must still be used, but misunderstood."
        ),
    },
    "F3": {
        "name": "evidence_overreach",
        "interface": "(X,G)->R",
        "definition": (
            "The reasoning introduces claims that are not supported by the "
            "available patient context or evidence."
        ),
        "instruction": (
            "Insert a small number of unsupported claims, over-extend limited "
            "evidence into stronger conclusions, or convert suggestive evidence "
            "into definitive support. Keep the topic clinically relevant and do "
            "not introduce unrelated disease pathways."
        ),
    },
    "F4": {
        "name": "derivational_mismatch",
        "interface": "R->Y",
        "definition": (
            "The final answer is not sufficiently supported by the preceding "
            "reasoning trace, or is inconsistent with it."
        ),
        "instruction": (
            "Break the inferential link from reasoning to answer. Typical forms "
            "include reasoning that mainly supports another diagnosis, reasoning "
            "that remains tentative while the final answer becomes definitive, "
            "or recommendations that appear without adequate support."
        ),
    },
}


META_PHRASES = [
    "flawed reasoning",
    "logical flaw",
    "negative sample",
    "simulating",
    "here is the rewritten reasoning",
    "the intended error type",
    "answer-preserved",
]


GENERATOR_SYSTEM_PROMPT = """
You are a clinical reasoning negative-sample constructor.

Your task is NOT to produce nonsense or simply make the final answer wrong.
Your task is to transform a faithful positive reasoning trace into a locally
unfaithful but still clinically plausible negative reasoning trace.

Global rules:
1. Introduce only ONE dominant unfaithfulness error.
2. Keep the case topic, terminology style, and most of the reasoning unchanged.
3. Prefer answer-preserved negatives unless the requested setup says otherwise.
4. Make the error local rather than rewriting the whole trace.
5. Do not add unrelated diseases, tests, or treatments.
6. Return JSON only.
""".strip()


JUDGE_SYSTEM_PROMPT = """
You are a clinical reasoning quality reviewer.

Your task is to compare a candidate negative reasoning trace against the
original positive trace and decide:
- what unfaithfulness type the candidate mainly realizes,
- whether it matches the requested target type,
- whether it contains one dominant error or multiple mixed errors,
- whether it remains clinically plausible,
- whether it should be accepted as a controlled negative sample.

You must judge from the perspective of clinical faithfulness, not merely answer
correctness. Return JSON only.
""".strip()


@dataclass
class Sample:
    source_id: str
    dataset_name: str
    split: str
    patient_context: str
    retrieved_evidence: str
    reasoning: str
    final_answer: str
    raw_record: Dict[str, Any]


@dataclass
class LLMRuntimeConfig:
    llm_mode: str
    model: str
    base_url: str
    api_key: str


@dataclass
class PlanResult:
    feasible: bool
    preserve_answer: bool
    target_span: str
    rationale: str
    confidence: float


@dataclass
class RewriteResult:
    negative_reasoning: str
    negative_final_answer: str
    modified_spans: List[str]
    self_check: str
    edit_summary: List[str]
    target_span: str = ""
    replacement_span: str = ""


@dataclass
class JudgeResult:
    realized_error_type: str
    target_type_matched: bool
    single_dominant_error: bool
    answer_preserved_judged: Optional[bool]
    clinical_plausibility_score: int
    faithfulness_drop_score: int
    off_topic: bool
    decision: str
    reason: str


TERMINAL_STATUSES = {
    "accepted_without_judge",
    "accepted_by_judge",
    "needs_human_review",
    "rejected_by_judge",
    "rejected_by_heuristic_qc",
    "plan_infeasible",
}

RETRYABLE_ERROR_PATTERNS = (
    "429",
    "too many requests",
    "too_many_requests",
    "rate limit",
    "timed out",
    "timeout",
    "connection error",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "proxyerror",
    "apiconnectionerror",
)


def parse_args() -> argparse.Namespace:
    """
    作用：解析命令行参数，生成脚本运行所需的全部配置。
    输入：无，参数来自命令行。
    输出：`argparse.Namespace`，包含数据源、模型、QC、输出路径等配置项。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate type-controlled unfaithful reasoning negatives from "
            "MedCaseReasoning or a local file."
        )
    )
    parser.add_argument(
        "--source",
        choices=["medcase", "local"],
        default="medcase",
        help="Positive sample source. Defaults to Hugging Face MedCaseReasoning.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_MEDCASE_DATASET_NAME,
        help="Dataset name when --source medcase is used.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split for --source medcase.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_MEDCASE_DATASET_DIR,
        help=(
            "Local directory used to cache/download MedCaseReasoning parquet "
            "files before loading."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_HUB_TOKEN", "")),
        help="Optional Hugging Face token used when downloading dataset files.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download MedCaseReasoning files even if they already exist.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download MedCaseReasoning to --dataset-dir and exit.",
    )
    parser.add_argument(
        "--input-file",
        default="",
        help="Local JSONL/JSON file when --source local is used.",
    )
    parser.add_argument(
        "--output-file",
        default="medcase_unfaithful_negatives.jsonl",
        help="Where to write the generated negatives.",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["api", "local"],
        default=LLM_MODE,
        help="Choose 'api' for cloud API or 'local' for an OpenAI-compatible local model server.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Override model name from the code-top config section.",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Override base URL from the code-top config section.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Override API key from the code-top config section.",
    )
    parser.add_argument(
        "--error-types",
        default="F1,F2,F3,F4",
        help="Comma-separated target error types.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["all", "sampled"],
        default="all",
        help="Generate one negative per requested type, or sample one type per case.",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=["staged", "rewrite_only"],
        default="staged",
        help=(
            "Use the default plan->rewrite pipeline, or skip the explicit plan stage "
            "for a faster/cheaper single-call rewrite."
        ),
    )
    parser.add_argument(
        "--rewrite-strategy",
        choices=["freeform", "span_patch"],
        default="span_patch",
        help=(
            "Let the model rewrite the whole trace, or force a targeted span replacement "
            "that gets applied back onto the original reasoning."
        ),
    )
    parser.add_argument(
        "--f4-answer-change-rate",
        type=float,
        default=0.2,
        help=(
            "For F4 only, probability of changing the final answer instead of "
            "keeping it unchanged."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of positive samples to process. 0 means all.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start offset into the positive split.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle positive samples before slicing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1800,
        help="Max output tokens for generation.",
    )
    parser.add_argument(
        "--plan-max-tokens",
        type=int,
        default=220,
        help="Max output tokens for the plan stage.",
    )
    parser.add_argument(
        "--rewrite-max-tokens",
        type=int,
        default=900,
        help="Max output tokens for the rewrite stage. 0 means use --max-tokens.",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=0,
        help="Max output tokens for the optional judge stage. 0 means use 600.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Retry count for generation calls.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=2.0,
        help="Optional sleep between API calls.",
    )
    parser.add_argument(
        "--evidence-field",
        default="",
        help=(
            "Optional field to treat as retrieved evidence. For MedCaseReasoning, "
            "use 'text' if you want the article text as G. Empty means no G."
        ),
    )
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=5000,
        help="Truncate retrieved evidence to this many characters.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.55,
        help="Minimum reasoning similarity between positive and negative.",
    )
    parser.add_argument(
        "--max-similarity",
        type=float,
        default=0.98,
        help="Maximum reasoning similarity between positive and negative.",
    )
    parser.add_argument(
        "--min-length-ratio",
        type=float,
        default=0.6,
        help="Minimum negative/positive token length ratio.",
    )
    parser.add_argument(
        "--max-length-ratio",
        type=float,
        default=1.5,
        help="Maximum negative/positive token length ratio.",
    )
    parser.add_argument(
        "--max-novelty-ratio",
        type=float,
        default=0.4,
        help="Upper bound on novel token ratio for localized rewrite screening.",
    )
    parser.add_argument(
        "--enable-llm-qc",
        action="store_true",
        help="Run a judge-model review pass after rewrite generation.",
    )
    parser.add_argument(
        "--qc-model",
        default="",
        help="Optional judge model; defaults to --model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file instead of resuming.",
    )
    return parser.parse_args()


def pick_first_non_empty_string(*values: str) -> str:
    """
    作用：从多个字符串候选里取第一个非空值。
    输入：`*values`，若干字符串候选。
    输出：`str`，第一个去空白后仍非空的字符串；若都为空则返回空串。
    """
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return ""


def safe_int(value: Any, default: int) -> int:
    """
    作用：安全地把任意值转换成整数。
    输入：`value` 为待转换对象，`default` 为失败时的回退值。
    输出：`int`，成功则为转换结果，失败则为 `default`。
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float) -> float:
    """
    作用：安全地把任意值转换成浮点数。
    输入：`value` 为待转换对象，`default` 为失败时的回退值。
    输出：`float`，成功则为转换结果，失败则为 `default`。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_bool(value: Any, default: bool) -> bool:
    """
    作用：把常见的布尔表达形式统一转换成布尔值。
    输入：`value` 为待解析对象，`default` 为无法识别时的默认值。
    输出：`bool`，解析后的真假值。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return default


def resolve_llm_runtime_config(args: argparse.Namespace) -> LLMRuntimeConfig:
    """
    作用：综合命令行、代码顶部常量和环境变量，确定最终使用的模型配置。
    输入：`args`，命令行参数对象。
    输出：`LLMRuntimeConfig`，包含 llm_mode、model、base_url、api_key。
    """
    llm_mode = pick_first_non_empty_string(args.llm_mode, LLM_MODE).lower()
    if llm_mode not in {"api", "local"}:
        raise ValueError("`LLM_MODE` must be either 'api' or 'local'.")

    if llm_mode == "local":
        model = pick_first_non_empty_string(args.model, LOCAL_MODEL_NAME)
        base_url = pick_first_non_empty_string(args.base_url, LOCAL_API_BASE)
        api_key = pick_first_non_empty_string(args.api_key, LOCAL_API_KEY, "EMPTY")
        if not model:
            raise ValueError("Local mode requires `LOCAL_MODEL_NAME` or `--model`.")
        if not base_url:
            raise ValueError("Local mode requires `LOCAL_API_BASE` or `--base-url`.")
        return LLMRuntimeConfig(
            llm_mode=llm_mode,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

    model = pick_first_non_empty_string(
        args.model,
        OPENAI_MODEL,
        os.getenv("OPENAI_MODEL", ""),
        "gpt-4o-mini",
    )
    base_url = pick_first_non_empty_string(
        args.base_url,
        OPENAI_BASE_URL,
        os.getenv("OPENAI_BASE_URL", ""),
    )
    api_key = pick_first_non_empty_string(
        args.api_key,
        OPENAI_API_KEY,
        os.getenv("OPENAI_API_KEY", ""),
    )
    if not api_key:
        raise ValueError(
            "API mode requires OPENAI_API_KEY. Set it in the code-top config, "
            "pass `--api-key`, or export `OPENAI_API_KEY`."
        )
    return LLMRuntimeConfig(
        llm_mode=llm_mode,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


def apply_llm_runtime_config(args: argparse.Namespace) -> LLMRuntimeConfig:
    """
    作用：把解析出的模型配置回写到 `args`，方便后续流程统一读取。
    输入：`args`，命令行参数对象。
    输出：`LLMRuntimeConfig`，即最终生效的模型配置。
    """
    runtime_config = resolve_llm_runtime_config(args)
    args.llm_mode = runtime_config.llm_mode
    args.model = runtime_config.model
    args.base_url = runtime_config.base_url
    args.api_key = runtime_config.api_key
    return runtime_config


def build_client(args: argparse.Namespace) -> OpenAI:
    """
    作用：根据当前参数创建 OpenAI-compatible 客户端。
    输入：`args`，其中至少需要 `api_key`，可选 `base_url`。
    输出：`OpenAI` 客户端实例。
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Please install 'openai' to generate negatives: pip install openai"
        ) from exc

    kwargs: Dict[str, Any] = {"api_key": args.api_key.strip()}
    if args.base_url.strip():
        kwargs["base_url"] = args.base_url.strip()
    return OpenAI(**kwargs)


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """
    作用：逐行读取 JSONL 文件，并把每一行解析成字典。
    输入：`path`，JSONL 文件路径。
    输出：`Iterable[Dict[str, Any]]`，按行产出的记录字典；坏行会被跳过。
    """
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def resolve_medcase_split(split: str) -> str:
    """
    作用：把用户输入的 split 别名归一化为 MedCaseReasoning 真正的 split 名称。
    输入：`split`，如 `train`、`validation`、`dev`。
    输出：`str`，规范化后的 split 名称，如 `train`、`val`、`test`。
    """
    normalized = split.strip().lower()
    if normalized not in MEDCASE_SPLIT_ALIASES:
        supported = ", ".join(sorted(MEDCASE_SPLIT_ALIASES))
        raise ValueError(f"Unsupported MedCaseReasoning split '{split}'. Use one of: {supported}")
    return MEDCASE_SPLIT_ALIASES[normalized]


def download_file(url: str, output_path: str, hf_token: str = "") -> None:
    """
    作用：使用标准库把远程文件下载到本地，并采用 `.part` 临时文件避免半写入。
    输入：`url` 为下载地址，`output_path` 为本地目标路径，`hf_token` 为可选鉴权令牌。
    输出：无，成功时在磁盘写入目标文件，失败时抛异常。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".part"
    headers = {"User-Agent": "medcase-reasoning-downloader/1.0"}
    if hf_token.strip():
        headers["Authorization"] = f"Bearer {hf_token.strip()}"

    request = Request(url, headers=headers)
    try:
        with urlopen(request) as response, open(tmp_path, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        os.replace(tmp_path, output_path)
    except (HTTPError, URLError, OSError) as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Failed to download {url} -> {output_path}: {exc}") from exc


def ensure_medcase_dataset_downloaded(args: argparse.Namespace) -> Dict[str, str]:
    """
    作用：确保 MedCaseReasoning 所需文件都已下载到本地目录。
    输入：`args`，主要使用 `dataset_dir`、`hf_token`、`force_download`。
    输出：`Dict[str, str]`，键为相对文件名，值为对应本地绝对路径。
    """
    dataset_dir = os.path.abspath(os.path.expanduser(args.dataset_dir))
    local_files: Dict[str, str] = {}

    for relative_path, remote_url in MEDCASE_REMOTE_FILES.items():
        output_path = os.path.join(dataset_dir, *relative_path.split("/"))
        needs_download = (
            args.force_download
            or not os.path.exists(output_path)
            or os.path.getsize(output_path) <= 0
        )
        if needs_download:
            print(f"Downloading {relative_path} -> {output_path}")
            download_file(remote_url, output_path, hf_token=args.hf_token)
        local_files[relative_path] = output_path

    return local_files


def load_local_parquet_records(parquet_path: str) -> List[Dict[str, Any]]:
    """
    作用：从本地 parquet 文件读取全部记录，并统一转成字典列表。
    输入：`parquet_path`，parquet 文件路径。
    输出：`List[Dict[str, Any]]`，每个元素是一条样本记录。
    """
    loader_errors: List[str] = []

    try:
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files={"train": parquet_path}, split="train")
        return [dict(row) for row in dataset]
    except Exception as exc:
        loader_errors.append(f"datasets: {exc}")

    try:
        import pandas as pd

        return pd.read_parquet(parquet_path).to_dict(orient="records")
    except Exception as exc:
        loader_errors.append(f"pandas: {exc}")

    try:
        import pyarrow.parquet as pq

        return pq.read_table(parquet_path).to_pylist()
    except Exception as exc:
        loader_errors.append(f"pyarrow: {exc}")

    joined_errors = " | ".join(loader_errors) if loader_errors else "no loader attempted"
    raise ImportError(
        "Loading local parquet requires one of: 'datasets', 'pandas' with a parquet "
        f"engine, or 'pyarrow'. Details: {joined_errors}"
    )


def load_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    作用：根据数据源配置加载原始记录，支持本地 JSON/JSONL 和 MedCaseReasoning。
    输入：`args`，包含 `source`、`input_file`、`dataset_name`、`split` 等参数。
    输出：`List[Dict[str, Any]]`，尚未标准化的原始样本列表。
    """
    if args.source == "local":
        if not args.input_file:
            raise ValueError("--input-file is required when --source local.")
        path = args.input_file
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        if path.lower().endswith(".jsonl"):
            return list(read_jsonl(path))
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], list):
                    return data["data"]
                raise ValueError("JSON object input must contain a list under 'data'.")
        raise ValueError("Only .jsonl and .json inputs are supported for --source local.")

    # For the default MedCaseReasoning dataset, download official parquet files locally first.
    if args.dataset_name == DEFAULT_MEDCASE_DATASET_NAME:
        split_name = resolve_medcase_split(args.split)
        local_files = ensure_medcase_dataset_downloaded(args)
        parquet_path = local_files[MEDCASE_SPLIT_TO_FILE[split_name]]
        return load_local_parquet_records(parquet_path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Please install 'datasets' to load non-default datasets: pip install datasets"
        ) from exc

    dataset = load_dataset(args.dataset_name, split=args.split)
    return [dict(row) for row in dataset]


def normalize_text(value: Any) -> str:
    """
    作用：把不同类型的值统一整理成便于后续处理的字符串。
    输入：`value`，可以是 `str`、`list`、`dict` 或其他可转字符串对象。
    输出：`str`，清洗后的文本；空值返回空串。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            normalized_item = normalize_text(item)
            if normalized_item:
                parts.append(normalized_item)
        return "\n\n".join(parts)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def truncate_text(text: str, max_chars: int) -> str:
    """
    作用：按最大字符数截断文本，超长时在末尾补 `...`。
    输入：`text` 为待截断文本，`max_chars` 为最大字符数。
    输出：`str`，截断后的文本。
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def first_non_empty(record: Dict[str, Any], keys: Sequence[str]) -> str:
    """
    作用：按字段优先级从记录中取出第一个非空文本字段。
    输入：`record` 为原始记录，`keys` 为按优先级排列的字段名列表。
    输出：`str`，第一个有效字段的标准化文本；若没有则返回空串。
    """
    for key in keys:
        if key in record:
            value = normalize_text(record.get(key))
            if value:
                return value
    return ""


def normalize_record(
    record: Dict[str, Any],
    index: int,
    dataset_name: str,
    split: str,
    evidence_field: str,
    max_evidence_chars: int,
) -> Optional[Sample]:
    """
    作用：把不同来源的原始记录映射成统一的 `Sample` 结构。
    输入：原始 `record`，样本索引 `index`，数据集名 `dataset_name`，split 名，
    证据字段名 `evidence_field`，以及证据最大长度 `max_evidence_chars`。
    输出：`Sample`；若缺少必要字段则返回 `None`。
    """
    patient_context = first_non_empty(record, ["case_prompt", "question", "prompt", "input"])
    reasoning = first_non_empty(
        record,
        [
            "diagnostic_reasoning",
            "reasoning",
            "correct_reasoning",
            "chosen",
            "gt_reasoning",
        ],
    )
    final_answer = first_non_empty(
        record,
        [
            "final_diagnosis",
            "final_answer",
            "answer",
            "correct_answer_text",
            "diagnosis",
        ],
    )
    if not patient_context or not reasoning or not final_answer:
        return None

    evidence = ""
    if evidence_field.strip():
        evidence = normalize_text(record.get(evidence_field))
        evidence = truncate_text(evidence, max_evidence_chars)

    source_id = first_non_empty(record, ["id", "source_id", "id_in_dataset"])
    if not source_id:
        source_id = str(index)

    return Sample(
        source_id=str(source_id),
        dataset_name=dataset_name,
        split=split,
        patient_context=patient_context,
        retrieved_evidence=evidence,
        reasoning=reasoning,
        final_answer=final_answer,
        raw_record=record,
    )


def build_samples(args: argparse.Namespace) -> List[Sample]:
    """
    作用：加载原始记录、执行切片与打乱，并转换为统一的 `Sample` 列表。
    输入：`args`，包含加载、shuffle、start-index、max-samples 等配置。
    输出：`List[Sample]`，可直接进入生成流程的标准化样本列表。
    """
    raw_records = load_records(args)
    if args.shuffle:
        random.shuffle(raw_records)

    if args.start_index:
        raw_records = raw_records[args.start_index :]
    if args.max_samples > 0:
        raw_records = raw_records[: args.max_samples]

    dataset_name = args.dataset_name if args.source == "medcase" else os.path.basename(args.input_file)
    samples: List[Sample] = []
    for idx, record in enumerate(raw_records, start=args.start_index):
        sample = normalize_record(
            record=record,
            index=idx,
            dataset_name=dataset_name,
            split=args.split,
            evidence_field=args.evidence_field,
            max_evidence_chars=args.max_evidence_chars,
        )
        if sample is not None:
            samples.append(sample)
    return samples


def parse_error_types(raw_value: str) -> List[str]:
    """
    作用：把逗号分隔的错误类型字符串解析成错误类型列表。
    输入：`raw_value`，例如 `F1,F2,F4`。
    输出：`List[str]`，经过校验的错误类型列表。
    """
    result: List[str] = []
    for item in raw_value.split(","):
        item = item.strip().upper()
        if not item:
            continue
        if item not in ERROR_SPECS:
            raise ValueError(f"Unsupported error type: {item}")
        result.append(item)
    if not result:
        raise ValueError("At least one error type must be provided.")
    return result


def parse_plan_result(payload: Dict[str, Any]) -> PlanResult:
    """
    作用：把模型返回的 plan 阶段 JSON 解析并校验为 `PlanResult`。
    输入：`payload`，模型输出解析后的字典。
    输出：`PlanResult`，包含是否可构造、是否保留答案、目标改写片段等信息。
    """
    feasible = coerce_bool(payload.get("feasible"), False)
    preserve_answer = coerce_bool(payload.get("preserve_answer"), True)
    target_span = normalize_text(payload.get("target_span"))
    rationale = normalize_text(payload.get("rationale"))
    confidence = safe_float(payload.get("confidence"), 0.0)

    if feasible and not target_span:
        raise ValueError("Plan output is missing target_span for a feasible case.")

    return PlanResult(
        feasible=feasible,
        preserve_answer=preserve_answer,
        target_span=target_span,
        rationale=rationale,
        confidence=confidence,
    )


def locality_guidance(error_type: str) -> str:
    """
    作用：为不同错误类型补充额外的“局部改写”约束，减少整段重写和空改写。
    输入：`error_type` 为目标错误类型。
    输出：`str`，可直接插入提示词的附加指导语。
    """
    if error_type == "F1":
        return (
            "For F1, keep the enumerated structure, bullet count, and almost all wording "
            "unchanged. Select one contiguous evidential clause or one existing bullet, "
            "and make that evidence omitted, down-weighted, or no longer substantively "
            "used. Do not compress multiple bullets into prose, do not choose distant spans, "
            "and do not leave the target span effectively unchanged."
        )
    return (
        "Preserve the original sentence order, numbering or bullets, and formatting as much "
        "as possible. Prefer editing one sentence or clause over regenerating multiple bullets, "
        "and do not merge list items into prose."
    )


def json_output_guidance() -> str:
    """
    作用：统一补充 JSON 输出格式要求，减少解析失败。
    输入：无。
    输出：`str`，可直接插入提示词的 JSON 规范文本。
    """
    return (
        "Output a single valid JSON object only. Use double-quoted JSON strings, "
        "escape any inner quotes inside string values, and do not include markdown "
        "fences, comments, or trailing commas."
    )


def parse_rewrite_result(payload: Dict[str, Any]) -> RewriteResult:
    """
    作用：把 rewrite 阶段的模型输出解析为统一结构。
    输入：`payload`，模型输出解析后的字典。
    输出：`RewriteResult`，包含负推理、负答案、改写片段和自检说明。
    """
    negative_reasoning = normalize_text(payload.get("negative_reasoning"))
    negative_final_answer = normalize_text(
        payload.get("negative_final_answer") or payload.get("negative_answer")
    )
    target_span = normalize_text(payload.get("target_span"))
    replacement_span = normalize_text(payload.get("replacement_span"))
    modified_spans = payload.get("modified_spans") or []
    self_check = normalize_text(payload.get("self_check"))
    edit_summary = payload.get("edit_summary") or []

    if not negative_reasoning and not replacement_span:
        raise ValueError("Missing negative_reasoning or replacement_span in model output.")
    if not negative_final_answer:
        raise ValueError("Missing negative_final_answer in model output.")
    if not isinstance(modified_spans, list):
        modified_spans = [modified_spans]
    if not isinstance(edit_summary, list):
        edit_summary = [edit_summary]

    normalized_modified_spans = [
        normalize_text(item) for item in modified_spans if normalize_text(item)
    ]
    normalized_edit_summary = [
        normalize_text(item) for item in edit_summary if normalize_text(item)
    ]

    return RewriteResult(
        negative_reasoning=negative_reasoning,
        negative_final_answer=negative_final_answer,
        modified_spans=normalized_modified_spans,
        self_check=self_check,
        edit_summary=normalized_edit_summary,
        target_span=target_span,
        replacement_span=replacement_span,
    )


def parse_judge_result(payload: Dict[str, Any]) -> JudgeResult:
    """
    作用：把 judge 阶段的模型输出解析为结构化评审结果。
    输入：`payload`，judge 模型输出解析后的字典。
    输出：`JudgeResult`，包含错误类型匹配、可信度评分和 accept/review/reject 决策。
    """
    answer_preserved_raw = payload.get("answer_preserved_judged")
    answer_preserved_judged: Optional[bool]
    if answer_preserved_raw is None:
        answer_preserved_judged = None
    else:
        answer_preserved_judged = coerce_bool(answer_preserved_raw, False)

    decision = normalize_text(payload.get("decision")).lower() or "reject"
    if decision not in {"accept", "review", "reject"}:
        decision = "reject"

    return JudgeResult(
        realized_error_type=normalize_text(payload.get("realized_error_type")) or "Invalid",
        target_type_matched=coerce_bool(payload.get("target_type_matched"), False),
        single_dominant_error=coerce_bool(payload.get("single_dominant_error"), False),
        answer_preserved_judged=answer_preserved_judged,
        clinical_plausibility_score=max(
            1,
            min(5, safe_int(payload.get("clinical_plausibility_score"), 1)),
        ),
        faithfulness_drop_score=max(
            1,
            min(5, safe_int(payload.get("faithfulness_drop_score"), 1)),
        ),
        off_topic=coerce_bool(payload.get("off_topic"), False),
        decision=decision,
        reason=normalize_text(payload.get("reason")),
    )


def answer_should_be_preserved(error_type: str, args: argparse.Namespace) -> bool:
    """
    作用：确定本次目标错误类型默认是否应保留最终答案。
    输入：`error_type` 为错误类型，`args` 提供 F4 的答案修改概率配置。
    输出：`bool`，表示该轮尝试默认是否期望保留答案。
    """
    if error_type != "F4":
        return True
    return random.random() >= args.f4_answer_change_rate


def build_plan_prompt(
    sample: Sample,
    error_type: str,
    preferred_preserve_answer: bool,
) -> str:
    """
    作用：构造 plan 阶段提示词，让模型先判断该错误类型是否可局部构造。
    输入：`sample` 为正样本，`error_type` 为目标错误类型，
    `preferred_preserve_answer` 为本轮默认答案策略。
    输出：`str`，发给生成模型的 plan 阶段提示词。
    """
    spec = ERROR_SPECS[error_type]
    evidence_block = sample.retrieved_evidence if sample.retrieved_evidence else "(none)"
    preserve_text = "YES" if preferred_preserve_answer else "NO"
    return f"""
You are given one positive clinical reasoning sample.

[Patient Context X]
{sample.patient_context}

[Retrieved Evidence G]
{evidence_block}

[Reference Reasoning R+]
{sample.reasoning}

[Reference Final Answer Y]
{sample.final_answer}

[Target Error Type]
{error_type} | {spec["name"]} | interface {spec["interface"]}

[Core Definition]
{spec["definition"]}

[Locality Guidance]
{locality_guidance(error_type)}

[Preferred Answer Handling For This Attempt]
Preserve the final answer exactly if feasible: {preserve_text}

[Task]
1. Decide whether it is FEASIBLE to construct a negative sample of type {error_type}.
2. If feasible, decide whether the final answer should be preserved.
3. Identify one local span in R+ that should be minimally rewritten.
4. Briefly explain why.

[Important]
1. Do not construct the negative yet.
2. Be conservative: if the sample is not suitable for this target type, set feasible=false.
3. Keep the plan local rather than rewriting the whole trace.
4. Choose one contiguous span copied verbatim from R+; do not select multiple distant spans.
5. If R+ is enumerated, prefer one clause or one existing bullet rather than a broad multi-bullet block.
6. Return JSON only.

[JSON Formatting Rules]
{json_output_guidance()}

[JSON Schema]
{{
  "feasible": true,
  "preserve_answer": true,
  "target_span": "exact or near-exact span to modify",
  "rationale": "short explanation",
  "confidence": 0.0
}}
""".strip()


def build_rewrite_prompt(
    sample: Sample,
    error_type: str,
    plan: PlanResult,
) -> str:
    """
    作用：根据 plan 结果构造 rewrite 阶段提示词，引导模型做最小局部改写。
    输入：`sample` 为正样本，`error_type` 为目标错误类型，`plan` 为已批准的改写计划。
    输出：`str`，发给生成模型的 rewrite 阶段提示词。
    """
    spec = ERROR_SPECS[error_type]
    evidence_block = sample.retrieved_evidence if sample.retrieved_evidence else "(none)"
    preserve_text = "YES" if plan.preserve_answer else "NO"
    return f"""
[Patient Context X]
{sample.patient_context}

[Retrieved Evidence G]
{evidence_block}

[Reference Reasoning R+]
{sample.reasoning}

[Reference Final Answer Y]
{sample.final_answer}

[Target Error Type]
{error_type} | {spec["name"]} | interface {spec["interface"]}

[Approved Rewrite Plan]
- feasible: {str(plan.feasible).lower()}
- preserve_answer: {str(plan.preserve_answer).lower()}
- target_span: {plan.target_span}
- rationale: {plan.rationale}
- confidence: {plan.confidence:.2f}

[Targeted Local Rewrite Instruction]
{spec["instruction"]}

[Locality Guidance]
{locality_guidance(error_type)}

[Answer Preservation]
Preserve the final answer exactly if possible: {preserve_text}

[Output Requirements]
Return one JSON object with exactly these keys:
- "negative_reasoning": the locally rewritten reasoning trace R-
- "negative_final_answer": the final answer paired with R-
- "modified_spans": list of 1-3 short snippets that were actually changed
- "self_check": one short sentence describing the intended local change
- "edit_summary": short list of 1-3 strings describing what was changed

[Critical Constraints]
1. Rewrite only a limited portion of R+ instead of replacing the whole reasoning.
2. Keep the same case topic, broad structure, and style.
3. Keep the text clinically plausible and relevant to this case.
4. Inject one dominant {error_type} error and avoid contamination with other error types.
5. Do not add meta-commentary, headings about flaws, or dataset language.
6. If preserving the answer, keep the answer string identical to Y.
7. Keep the reasoning close in length to the original trace.

[JSON Formatting Rules]
{json_output_guidance()}
""".strip()


def build_direct_rewrite_prompt(
    sample: Sample,
    error_type: str,
    preserve_answer: bool,
) -> str:
    """
    作用：构造单阶段 rewrite 提示词，让模型自己选择局部改写点并直接产出候选。
    输入：`sample` 为正样本，`error_type` 为目标错误类型，`preserve_answer` 为答案策略。
    输出：`str`，发给生成模型的单阶段 rewrite 提示词。
    """
    spec = ERROR_SPECS[error_type]
    evidence_block = sample.retrieved_evidence if sample.retrieved_evidence else "(none)"
    preserve_text = "YES" if preserve_answer else "NO"
    return f"""
[Patient Context X]
{sample.patient_context}

[Retrieved Evidence G]
{evidence_block}

[Reference Reasoning R+]
{sample.reasoning}

[Reference Final Answer Y]
{sample.final_answer}

[Target Error Type]
{error_type} | {spec["name"]} | interface {spec["interface"]}

[Targeted Local Rewrite Instruction]
{spec["instruction"]}

[Locality Guidance]
{locality_guidance(error_type)}

[Answer Preservation]
Preserve the final answer exactly if possible: {preserve_text}

[Task]
1. Choose one local span in R+ that can be minimally rewritten.
2. Rewrite only that limited portion to introduce one dominant {error_type} error.
3. Keep the rest of the reasoning as close as possible to the original.

[Output Requirements]
Return one JSON object with exactly these keys:
- "negative_reasoning": the locally rewritten reasoning trace R-
- "negative_final_answer": the final answer paired with R-
- "modified_spans": list of 1-3 short snippets that were actually changed
- "self_check": one short sentence describing the intended local change
- "edit_summary": short list of 1-3 strings describing what was changed

[Critical Constraints]
1. Rewrite only a limited portion of R+ instead of replacing the whole reasoning.
2. Keep the same case topic, broad structure, and style.
3. Keep the text clinically plausible and relevant to this case.
4. Inject one dominant {error_type} error and avoid contamination with other error types.
5. Do not add meta-commentary, headings about flaws, or dataset language.
6. If preserving the answer, keep the answer string identical to Y.
7. Keep the reasoning close in length to the original trace.

[JSON Formatting Rules]
{json_output_guidance()}
""".strip()


def build_patch_rewrite_prompt(
    sample: Sample,
    error_type: str,
    plan: PlanResult,
) -> str:
    """
    作用：构造严格局部编辑提示词，让模型只返回一个待替换片段及其替换内容。
    输入：`sample` 为正样本，`error_type` 为目标错误类型，`plan` 为已批准计划。
    输出：`str`，要求模型产出 target_span + replacement_span 的 patch 风格提示词。
    """
    spec = ERROR_SPECS[error_type]
    evidence_block = sample.retrieved_evidence if sample.retrieved_evidence else "(none)"
    preserve_text = "YES" if plan.preserve_answer else "NO"
    return f"""
[Patient Context X]
{sample.patient_context}

[Retrieved Evidence G]
{evidence_block}

[Reference Reasoning R+]
{sample.reasoning}

[Reference Final Answer Y]
{sample.final_answer}

[Target Error Type]
{error_type} | {spec["name"]} | interface {spec["interface"]}

[Approved Rewrite Plan]
- feasible: {str(plan.feasible).lower()}
- preserve_answer: {str(plan.preserve_answer).lower()}
- target_span: {plan.target_span}
- rationale: {plan.rationale}
- confidence: {plan.confidence:.2f}

[Targeted Local Rewrite Instruction]
{spec["instruction"]}

[Locality Guidance]
{locality_guidance(error_type)}

[Answer Preservation]
Preserve the final answer exactly if possible: {preserve_text}

[Task]
Return a localized edit patch instead of rewriting the whole reasoning.

[Output Requirements]
Return one JSON object with exactly these keys:
- "target_span": the exact span from R+ that should be replaced
- "replacement_span": the minimally edited replacement text
- "negative_final_answer": the final answer paired with the patched reasoning
- "modified_spans": list of 1-3 short snippets that were actually changed
- "self_check": one short sentence describing the intended local change
- "edit_summary": short list of 1-3 strings describing what was changed

[Critical Constraints]
1. Copy "target_span" exactly from R+ instead of paraphrasing it.
2. Only change that local span; the rest of R+ should remain untouched.
3. Make the edit substantive enough that the patched trace is not nearly identical.
4. Keep the same case topic, broad structure, numbering, and style.
5. Inject one dominant {error_type} error and avoid contamination with other error types.
6. Do not add meta-commentary, headings about flaws, or dataset language.
7. If preserving the answer, keep the answer string identical to Y.

[JSON Formatting Rules]
{json_output_guidance()}
""".strip()


def build_direct_patch_rewrite_prompt(
    sample: Sample,
    error_type: str,
    preserve_answer: bool,
) -> str:
    """
    作用：构造单阶段 patch 提示词，让模型自己选目标片段并返回 replacement patch。
    输入：`sample` 为正样本，`error_type` 为目标错误类型，`preserve_answer` 为答案策略。
    输出：`str`，发给模型的单阶段 patch 提示词。
    """
    spec = ERROR_SPECS[error_type]
    evidence_block = sample.retrieved_evidence if sample.retrieved_evidence else "(none)"
    preserve_text = "YES" if preserve_answer else "NO"
    return f"""
[Patient Context X]
{sample.patient_context}

[Retrieved Evidence G]
{evidence_block}

[Reference Reasoning R+]
{sample.reasoning}

[Reference Final Answer Y]
{sample.final_answer}

[Target Error Type]
{error_type} | {spec["name"]} | interface {spec["interface"]}

[Targeted Local Rewrite Instruction]
{spec["instruction"]}

[Locality Guidance]
{locality_guidance(error_type)}

[Answer Preservation]
Preserve the final answer exactly if possible: {preserve_text}

[Task]
1. Choose one exact local span in R+ that can be minimally rewritten.
2. Return a patch that replaces only that span with a controlled unfaithful variant.
3. Keep the rest of R+ unchanged.

[Output Requirements]
Return one JSON object with exactly these keys:
- "target_span": the exact span from R+ that should be replaced
- "replacement_span": the minimally edited replacement text
- "negative_final_answer": the final answer paired with the patched reasoning
- "modified_spans": list of 1-3 short snippets that were actually changed
- "self_check": one short sentence describing the intended local change
- "edit_summary": short list of 1-3 strings describing what was changed

[Critical Constraints]
1. Copy "target_span" exactly from R+ instead of paraphrasing it.
2. Only change that local span; the rest of R+ should remain untouched.
3. Make the edit substantive enough that the patched trace is not nearly identical.
4. Keep the same case topic, broad structure, numbering, and style.
5. Inject one dominant {error_type} error and avoid contamination with other error types.
6. Do not add meta-commentary, headings about flaws, or dataset language.
7. If preserving the answer, keep the answer string identical to Y.

[JSON Formatting Rules]
{json_output_guidance()}
""".strip()


def strip_code_fences(text: str) -> str:
    """
    作用：去掉模型输出外层可能包裹的 Markdown 代码块标记。
    输入：`text`，原始模型输出文本。
    输出：`str`，去除 ``` 或 ```json 包裹后的文本。
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def extract_json(text: str) -> Dict[str, Any]:
    """
    作用：尽量从模型文本中提取首个 JSON 对象。
    输入：`text`，模型原始输出。
    输出：`Dict[str, Any]`，解析得到的 JSON 字典；失败时抛异常。
    """
    clean = strip_code_fences(text)
    decoder = JSONDecoder()
    try:
        return json.loads(clean)
    except JSONDecodeError:
        for match in re.finditer(r"\{", clean):
            start = match.start()
            try:
                payload, _ = decoder.raw_decode(clean[start:])
            except JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise


def chat_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> Dict[str, Any]:
    """
    作用：调用 Chat Completions 接口并把结果解析成 JSON。
    输入：客户端、模型名、system/user prompt、温度、最大输出 token、重试次数。
    输出：`Dict[str, Any]`，解析后的模型 JSON 结果。
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            return extract_json(content)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"Generation failed after {max_retries} attempts: {last_error}")


def tokenize(text: str) -> List[str]:
    """
    作用：把文本拆成仅由字母数字下划线组成的小写 token 列表。
    输入：`text`，任意文本。
    输出：`List[str]`，分词结果。
    """
    return re.findall(r"\w+", text.lower())


def normalize_compact(text: str) -> str:
    """
    作用：把文本压成单空格的小写紧凑形式，便于相等比较。
    输入：`text`，原始文本。
    输出：`str`，归一化后的紧凑文本。
    """
    return re.sub(r"\s+", " ", text).strip().lower()


def normalized_answer(text: str) -> str:
    """
    作用：对答案字符串做宽松归一化，便于比较答案是否一致。
    输入：`text`，答案文本。
    输出：`str`，统一大小写、空白和尾部句号后的答案文本。
    """
    return re.sub(r"\s+", " ", text.lower()).strip().rstrip(".")


def canonicalize_similarity_text(text: str) -> str:
    """
    作用：对文本做相似度比较前的轻量标准化，减少引号、空白和列表编号噪声。
    输入：`text`，待比较文本。
    输出：`str`，用于相似度计算的标准化文本。
    """
    text = normalize_text(text).lower()
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    text = re.sub(r"(?<!\w)(\d{1,2})\s*[\.)](?=\s+[a-z])", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sequence_similarity(a: str, b: str) -> float:
    """
    作用：计算两段文本的序列相似度。
    输入：`a`、`b` 两段文本。
    输出：`float`，范围通常在 0 到 1 之间，越大表示越相似。
    """
    return SequenceMatcher(
        None,
        canonicalize_similarity_text(a),
        canonicalize_similarity_text(b),
    ).ratio()


def token_length_ratio(source: str, target: str) -> float:
    """
    作用：计算目标文本相对源文本的 token 长度比例。
    输入：`source` 为源文本，`target` 为目标文本。
    输出：`float`，即 `len(target_tokens) / len(source_tokens)`。
    """
    source_tokens = tokenize(source)
    target_tokens = tokenize(target)
    if not source_tokens:
        return 0.0
    return len(target_tokens) / max(len(source_tokens), 1)


def novelty_ratio(source: str, target: str) -> float:
    """
    作用：估算目标文本中“源文本未出现的新 token”比例。
    输入：`source` 为源文本，`target` 为目标文本。
    输出：`float`，新 token 占目标 token 的比例。
    """
    source_tokens = set(tokenize(source))
    target_tokens = tokenize(target)
    if not target_tokens:
        return 1.0
    novel = [token for token in target_tokens if token not in source_tokens]
    return len(novel) / len(target_tokens)


def topic_overlap(case_text: str, reasoning_text: str) -> float:
    """
    作用：粗略估算候选推理与病例上下文的主题重叠度。
    输入：`case_text` 为病例上下文，`reasoning_text` 为候选推理。
    输出：`float`，推理 token 中有多少比例与病例 token 重合。
    """
    case_tokens = set(tokenize(case_text))
    reasoning_tokens = set(tokenize(reasoning_text))
    if not case_tokens or not reasoning_tokens:
        return 0.0
    return len(case_tokens & reasoning_tokens) / len(reasoning_tokens)


def contains_meta_language(text: str) -> bool:
    """
    作用：检查文本里是否包含“我是负样本/这是改写”等元话语。
    输入：`text`，待检测文本。
    输出：`bool`，存在元话语则为真。
    """
    lower = text.lower()
    return any(phrase in lower for phrase in META_PHRASES)


def span_match_variants(text: str) -> List[str]:
    """
    作用：生成目标片段的少量格式变体，方便在原文中做精确替换。
    输入：`text`，目标片段。
    输出：`List[str]`，按优先级排列的候选字符串。
    """
    variants: List[str] = []
    normalized = normalize_text(text)
    for candidate in [
        text,
        normalized,
        normalized.replace('"', "“"),
        normalized.replace('"', "”"),
        normalized.replace('"', "“").replace('"', "”"),
        normalized.replace("“", '"').replace("”", '"'),
    ]:
        candidate = candidate.strip()
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def align_replacement_prefix(matched_target_span: str, replacement_span: str) -> str:
    """
    作用：在 fuzzy 命中整条编号句时，把编号前缀补回 replacement，避免破坏列表结构。
    输入：命中的原始片段与替换片段。
    输出：`str`，必要时保留了编号前缀的替换文本。
    """
    replacement_span = normalize_text(replacement_span)
    prefix_match = re.match(r"^(\d+\.\s+)", matched_target_span)
    if prefix_match and not re.match(r"^\d+\.\s+", replacement_span):
        return prefix_match.group(1) + replacement_span.lstrip()
    return replacement_span


def token_windows(text: str) -> List[Tuple[int, int]]:
    """
    作用：提取文本中每个非空白 token 的原始字符区间，供 fuzzy span 定位使用。
    输入：`text`，原始文本。
    输出：`List[Tuple[int, int]]`，每个元素为 `(start, end)` 字符区间。
    """
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def fuzzy_patch_target(source: str, target_span: str) -> Tuple[str, float]:
    """
    作用：当 target_span 不是原文精确子串时，按 token 窗口搜索最相近的连续片段。
    输入：`source` 为原始推理，`target_span` 为模型给出的目标片段。
    输出：二元组 `(best_candidate, score)`；若未达到阈值则返回空串。
    """
    target_norm = canonicalize_similarity_text(target_span)
    if not target_norm:
        return "", 0.0

    windows = token_windows(source)
    if not windows:
        return "", 0.0

    target_token_count = max(1, len(re.findall(r"\S+", target_span)))
    tolerance = max(3, min(12, target_token_count // 3 + 2))
    min_len = max(1, target_token_count - tolerance)
    max_len = min(len(windows), target_token_count + tolerance)

    best_candidate = ""
    best_score = 0.0
    for start_idx in range(len(windows)):
        for window_len in range(min_len, max_len + 1):
            end_idx = start_idx + window_len - 1
            if end_idx >= len(windows):
                break
            candidate = source[windows[start_idx][0] : windows[end_idx][1]]
            score = SequenceMatcher(
                None,
                canonicalize_similarity_text(candidate),
                target_norm,
            ).ratio()
            if score > best_score:
                best_candidate = candidate
                best_score = score

    if best_score >= 0.78:
        return best_candidate, best_score
    return "", best_score


def apply_local_patch(source: str, target_span: str, replacement_span: str) -> Tuple[str, str]:
    """
    作用：把一个局部 replacement patch 应用回原始推理文本。
    输入：`source` 为原始推理，`target_span` 为待替换片段，`replacement_span` 为替换文本。
    输出：二元组 `(patched_text, matched_target_span)`。
    """
    replacement_span = normalize_text(replacement_span)
    if not replacement_span:
        raise ValueError("replacement_span is empty.")

    for candidate in span_match_variants(target_span):
        if candidate and candidate in source:
            aligned_replacement = align_replacement_prefix(candidate, replacement_span)
            return source.replace(candidate, aligned_replacement, 1), candidate

    fuzzy_candidate, fuzzy_score = fuzzy_patch_target(source, target_span)
    if fuzzy_candidate:
        aligned_replacement = align_replacement_prefix(fuzzy_candidate, replacement_span)
        return source.replace(fuzzy_candidate, aligned_replacement, 1), fuzzy_candidate

    raise ValueError(
        "Could not locate target_span in the source reasoning "
        f"(best fuzzy score={fuzzy_score:.3f})."
    )


def materialize_rewrite_result(
    sample: Sample,
    plan: PlanResult,
    rewrite: RewriteResult,
    args: argparse.Namespace,
) -> RewriteResult:
    """
    作用：根据 rewrite 策略把模型输出整理成最终候选推理。
    输入：原始样本、plan、初始 rewrite 结果及运行参数。
    输出：`RewriteResult`，其中 `negative_reasoning` 已就绪。
    """
    if args.rewrite_strategy != "span_patch":
        return rewrite

    target_span = rewrite.target_span or plan.target_span
    replacement_span = rewrite.replacement_span
    if not target_span or not replacement_span:
        if rewrite.negative_reasoning:
            return rewrite
        raise ValueError("span_patch strategy requires target_span and replacement_span.")

    if normalize_compact(target_span) == normalize_compact(replacement_span):
        raise ValueError("replacement_span is effectively identical to target_span.")

    patched_reasoning, matched_target_span = apply_local_patch(
        source=sample.reasoning,
        target_span=target_span,
        replacement_span=replacement_span,
    )
    modified_spans = rewrite.modified_spans or [matched_target_span, replacement_span]
    return RewriteResult(
        negative_reasoning=patched_reasoning,
        negative_final_answer=rewrite.negative_final_answer,
        modified_spans=modified_spans,
        self_check=rewrite.self_check,
        edit_summary=rewrite.edit_summary,
        target_span=matched_target_span,
        replacement_span=replacement_span,
    )


def heuristic_qc(
    sample: Sample,
    error_type: str,
    preserve_answer: bool,
    candidate_reasoning: str,
    candidate_answer: str,
    args: argparse.Namespace,
) -> Tuple[bool, Dict[str, Any]]:
    """
    作用：用启发式规则快速筛掉偏移过大、答案策略错误或明显跑题的候选。
    输入：正样本 `sample`、目标错误类型 `error_type`、是否保留答案 `preserve_answer`、
    候选推理 `candidate_reasoning`、候选答案 `candidate_answer`、以及阈值配置 `args`。
    输出：二元组 `(passed, report)`；`passed` 表示是否通过，`report` 为详细检查项。
    """
    similarity = sequence_similarity(sample.reasoning, candidate_reasoning)
    length_ratio = token_length_ratio(sample.reasoning, candidate_reasoning)
    novel_ratio = novelty_ratio(sample.reasoning, candidate_reasoning)
    overlap = topic_overlap(sample.patient_context, candidate_reasoning)
    same_answer = normalized_answer(sample.final_answer) == normalized_answer(candidate_answer)

    qc = {
        "similarity": round(similarity, 4),
        "length_ratio": round(length_ratio, 4),
        "novelty_ratio": round(novel_ratio, 4),
        "topic_overlap": round(overlap, 4),
        "same_answer": same_answer,
        "checks": {},
    }

    checks = qc["checks"]
    checks["non_empty"] = bool(candidate_reasoning and candidate_answer)
    checks["not_identical"] = normalize_compact(sample.reasoning) != normalize_compact(candidate_reasoning)
    checks["similarity_ok"] = args.min_similarity <= similarity <= args.max_similarity
    checks["length_ok"] = args.min_length_ratio <= length_ratio <= args.max_length_ratio
    checks["novelty_ok"] = novel_ratio <= args.max_novelty_ratio
    checks["topic_ok"] = overlap >= 0.03
    checks["no_meta_language"] = not contains_meta_language(candidate_reasoning + "\n" + candidate_answer)

    # F1/F2/F3 must preserve the final answer; F4 follows preserve_answer.
    if error_type in {"F1", "F2", "F3"}:
        checks["answer_policy_ok"] = same_answer
    elif preserve_answer:
        checks["answer_policy_ok"] = same_answer
    else:
        checks["answer_policy_ok"] = not same_answer

    qc["failed_checks"] = [name for name, value in checks.items() if not value]
    passed = all(bool(value) for value in checks.values())
    return passed, qc


def judge_passes(judge_result: JudgeResult, error_type: str, preserve_answer: bool) -> bool:
    """
    作用：根据 judge 结果判断候选是否可视为最终通过。
    输入：`judge_result` 为 judge 阶段结果，`error_type` 为目标类型，
    `preserve_answer` 为本轮答案策略。
    输出：`bool`，只有 judge 决策为 accept 且满足关键约束时才返回真。
    """
    if judge_result.decision != "accept":
        return False
    if not judge_result.target_type_matched:
        return False
    if judge_result.realized_error_type.upper() != error_type:
        return False
    if not judge_result.single_dominant_error or judge_result.off_topic:
        return False
    if preserve_answer and judge_result.answer_preserved_judged is not True:
        return False
    return True


def build_judge_prompt(
    sample: Sample,
    error_type: str,
    plan: PlanResult,
    rewrite: RewriteResult,
) -> str:
    """
    作用：构造 judge 阶段提示词，让独立模型评审候选负样本质量。
    输入：正样本 `sample`、目标类型 `error_type`、plan 结果 `plan`、rewrite 结果 `rewrite`。
    输出：`str`，发给 judge 模型的提示词。
    """
    evidence_block = sample.retrieved_evidence if sample.retrieved_evidence else "(none)"
    return f"""
You need to review whether the candidate negative sample is a good controlled negative.

[Patient Context X]
{sample.patient_context}

[Retrieved Evidence G]
{evidence_block}

[Reference Reasoning R+]
{sample.reasoning}

[Reference Final Answer Y]
{sample.final_answer}

[Candidate Negative Reasoning R-]
{rewrite.negative_reasoning}

[Candidate Negative Final Answer]
{rewrite.negative_final_answer}

[Target Unfaithfulness Type Requested During Generation]
{error_type}

[Generation Plan]
- preserve_answer: {str(plan.preserve_answer).lower()}
- target_span: {plan.target_span}
- rationale: {plan.rationale}

[Definitions]
- F1 Evidence Omission: important available evidence omitted or not substantively used.
- F2 Evidence Misinterpretation: evidence used but medically interpreted incorrectly.
- F3 Evidence Overreach: unsupported claim added beyond X or G.
- F4 Derivational Mismatch: final answer not sufficiently supported by preceding reasoning.
- Faithful: candidate does not clearly introduce an unfaithfulness error.
- Mixed: multiple error types are equally prominent.
- Invalid: off-topic, implausible, or unusable as a training negative.

[Review Criteria]
1. What is the realized dominant error type?
2. Does it match the target type?
3. Is there a single dominant error rather than mixed contamination?
4. If preserve_answer=true was intended, is the answer effectively preserved?
5. Is the candidate clinically plausible on the surface?
6. Does it show a meaningful faithfulness drop compared with the positive trace?
7. Should it be accepted, reviewed by a human, or rejected?

Return one JSON object with exactly these keys:
- "realized_error_type": one of F1, F2, F3, F4, Faithful, Mixed, Invalid
- "target_type_matched": boolean
- "single_dominant_error": boolean
- "answer_preserved_judged": boolean or null
- "clinical_plausibility_score": integer 1-5
- "faithfulness_drop_score": integer 1-5
- "off_topic": boolean
- "decision": one of accept, review, reject
- "reason": short string
""".strip()


def judge_candidate(
    client: OpenAI,
    sample: Sample,
    error_type: str,
    plan: PlanResult,
    rewrite: RewriteResult,
    args: argparse.Namespace,
) -> JudgeResult:
    """
    作用：调用 judge 模型审查候选负样本，并解析成结构化结果。
    输入：客户端、正样本、目标错误类型、plan 结果、rewrite 结果和运行参数。
    输出：`JudgeResult`，包含 judge 的类型判断、质量评分和 accept/review/reject 决策。
    """
    qc_model = args.qc_model or args.model
    judge_payload = chat_json(
        client=client,
        model=qc_model,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=build_judge_prompt(
            sample=sample,
            error_type=error_type,
            plan=plan,
            rewrite=rewrite,
        ),
        temperature=0.0,
        max_tokens=args.judge_max_tokens if args.judge_max_tokens > 0 else 600,
        max_retries=args.max_retries,
    )
    return parse_judge_result(judge_payload)


def append_jsonl_record(path: str, record: Dict[str, Any]) -> None:
    """
    作用：向指定 JSONL 文件追加一条记录。
    输入：`path` 为目标文件路径，`record` 为待写入字典。
    输出：无；成功时在磁盘上追加一行 JSON。
    """
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_retryable_runtime_error(exc: Exception) -> bool:
    """
    作用：判断异常是否更像瞬时网络/API 问题，从而允许后续 resume 自动重跑。
    输入：`exc`，当前捕获到的异常对象。
    输出：`bool`，若是限流、超时、连接异常等临时问题则返回真。
    """
    message = str(exc).strip().lower()
    return any(pattern in message for pattern in RETRYABLE_ERROR_PATTERNS)


def output_sidecar_paths(output_file: str) -> Dict[str, str]:
    """
    作用：根据主输出文件路径生成各类 sidecar 日志文件路径。
    输入：`output_file`，主输出 JSONL 文件路径。
    输出：`Dict[str, str]`，包含 attempts/review/rejected/infeasible/failures/progress 路径。
    """
    root, ext = os.path.splitext(output_file)
    suffix = ext or ".jsonl"
    return {
        "attempts": f"{root}.attempts{suffix}",
        "review": f"{root}.review{suffix}",
        "rejected": f"{root}.rejected{suffix}",
        "infeasible": f"{root}.infeasible{suffix}",
        "failures": f"{root}.failures{suffix}",
        "progress": f"{root}.progress.json",
    }


def existing_keys(*output_files: str) -> set:
    """
    作用：读取已有结果文件中的终态 `(source_id, error_symbol)` 键，用于断点续跑去重。
    输入：`*output_files`，一个或多个 JSONL 输出文件路径。
    输出：`set`，其中每个元素都是一个二元组键。
    """
    keys = set()
    for output_file in output_files:
        if not output_file or not os.path.exists(output_file):
            continue
        for row in read_jsonl(output_file):
            if not isinstance(row, dict):
                continue
            status = normalize_text(row.get("status"))
            if status and status not in TERMINAL_STATUSES:
                continue
            key = (str(row.get("source_id", "")), str(row.get("error_symbol", "")))
            if key != ("", ""):
                keys.add(key)
    return keys


def sync_attempt_ledger(output_file: str, sidecar_paths: Dict[str, str]) -> int:
    """
    作用：启动恢复时，把已经存在于主文件或终态 sidecar 中、但缺失于 attempts 的记录补回账本。
    输入：`output_file` 为主输出文件，`sidecar_paths` 为 sidecar 路径映射。
    输出：`int`，本次补写到 attempts 账本的记录条数。
    """
    attempts_path = sidecar_paths["attempts"]
    known_ids = set()
    if os.path.exists(attempts_path):
        for row in read_jsonl(attempts_path):
            if not isinstance(row, dict):
                continue
            record_id = normalize_text(row.get("id"))
            if record_id:
                known_ids.add(record_id)

    recovered = 0
    ledger_sources = [
        output_file,
        sidecar_paths["review"],
        sidecar_paths["rejected"],
        sidecar_paths["infeasible"],
    ]
    for source_path in ledger_sources:
        if not source_path or not os.path.exists(source_path):
            continue
        for row in read_jsonl(source_path):
            if not isinstance(row, dict):
                continue
            status = normalize_text(row.get("status"))
            if not status or status not in TERMINAL_STATUSES:
                continue
            record_id = normalize_text(row.get("id"))
            if not record_id or record_id in known_ids:
                continue
            append_jsonl_record(attempts_path, row)
            known_ids.add(record_id)
            recovered += 1
    return recovered


def write_progress_snapshot(
    path: str,
    *,
    output_file: str,
    total_planned: int,
    seen: set,
    written: int,
    reviewed: int,
    rejected: int,
    infeasible: int,
    failed: int,
    skipped: int,
    phase: str,
    current_sample_index: int = 0,
    current_sample_id: str = "",
    current_error_type: str = "",
) -> None:
    """
    作用：把当前运行进度写到进度文件，便于中断后快速确认恢复位置。
    输入：进度文件路径、输出文件路径、总任务量、已完成键集合、以及各类计数和当前处理位置。
    输出：无；会覆盖写入一个 JSON 进度快照。
    """
    payload = {
        "output_file": os.path.abspath(output_file),
        "phase": phase,
        "total_planned": total_planned,
        "completed_terminal": len(seen),
        "written_this_run": written,
        "reviewed_this_run": reviewed,
        "rejected_this_run": rejected,
        "infeasible_this_run": infeasible,
        "failed_this_run": failed,
        "skipped_this_run": skipped,
        "current_sample_index": current_sample_index,
        "current_sample_id": current_sample_id,
        "current_error_type": current_error_type,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def make_attempt_record(
    sample: Sample,
    error_type: str,
    status: str,
    expected_answer_preserved: Optional[bool],
    plan: Optional[PlanResult] = None,
    rewrite: Optional[RewriteResult] = None,
    heuristic_report: Optional[Dict[str, Any]] = None,
    llm_qc_report: Optional[JudgeResult] = None,
    error_message: str = "",
) -> Dict[str, Any]:
    """
    作用：构造任意执行状态下的通用 attempt 记录，便于写入 sidecar 日志。
    输入：样本、错误类型、状态、期望答案策略、以及各阶段可选产物和错误信息。
    输出：`Dict[str, Any]`，包含当前 attempt 的上下文与状态信息。
    """
    spec = ERROR_SPECS[error_type]
    negative_reasoning = rewrite.negative_reasoning if rewrite is not None else ""
    negative_answer = rewrite.negative_final_answer if rewrite is not None else ""
    answer_preserved_flag = (
        normalized_answer(sample.final_answer) == normalized_answer(negative_answer)
        if negative_answer
        else False
    )
    return {
        "id": f"{sample.source_id}::{error_type}",
        "dataset_name": sample.dataset_name,
        "split": sample.split,
        "source_id": sample.source_id,
        "error_symbol": error_type,
        "error_type": spec["name"],
        "interface": spec["interface"],
        "status": status,
        "patient_context": sample.patient_context,
        "retrieved_evidence": sample.retrieved_evidence,
        "prompt": sample.patient_context,
        "reasoning": sample.reasoning,
        "chosen": sample.reasoning,
        "correct_reasoning": sample.reasoning,
        "final_answer": sample.final_answer,
        "correct_answer_text": sample.final_answer,
        "negative_reasoning": negative_reasoning,
        "rejected": negative_reasoning,
        "predicted_reasoning": negative_reasoning,
        "negative_final_answer": negative_answer,
        "predicted_answer_text": negative_answer,
        "answer_preserved": answer_preserved_flag,
        "expected_answer_preserved": expected_answer_preserved,
        "neg_type": spec["name"],
        "type": spec["name"],
        "generation_plan": asdict(plan) if plan is not None else None,
        "modified_spans": rewrite.modified_spans if rewrite is not None else [],
        "rewrite_target_span": rewrite.target_span if rewrite is not None else "",
        "rewrite_replacement_span": rewrite.replacement_span if rewrite is not None else "",
        "rewrite_self_check": rewrite.self_check if rewrite is not None else "",
        "edit_summary": rewrite.edit_summary if rewrite is not None else [],
        "heuristic_qc": heuristic_report,
        "llm_qc": asdict(llm_qc_report) if llm_qc_report is not None else None,
        "error_message": error_message,
    }


def make_output_record(
    sample: Sample,
    error_type: str,
    status: str,
    plan: PlanResult,
    rewrite: RewriteResult,
    heuristic_report: Dict[str, Any],
    llm_qc_report: Optional[JudgeResult],
) -> Dict[str, Any]:
    """
    作用：把正样本、plan、rewrite、heuristic QC、judge QC 组装成最终输出记录。
    输入：正样本 `sample`、目标类型 `error_type`、当前状态 `status`、plan 结果、
    rewrite 结果、heuristic 报告和可选 judge 报告。
    输出：`Dict[str, Any]`，可直接写入 JSONL 的完整样本记录。
    """
    return make_attempt_record(
        sample=sample,
        error_type=error_type,
        status=status,
        expected_answer_preserved=plan.preserve_answer,
        plan=plan,
        rewrite=rewrite,
        heuristic_report=heuristic_report,
        llm_qc_report=llm_qc_report,
    )


def requested_error_types_for_sample(
    all_error_types: Sequence[str],
    mode: str,
) -> List[str]:
    """
    作用：确定当前样本本轮要尝试哪些错误类型。
    输入：`all_error_types` 为候选类型列表，`mode` 为 `all` 或 `sampled`。
    输出：`List[str]`，当前样本需要尝试的错误类型列表。
    """
    if mode == "all":
        return list(all_error_types)
    return [random.choice(list(all_error_types))]


def validate_generation_payload(payload: Dict[str, Any]) -> RewriteResult:
    """
    作用：对 rewrite 阶段输出做最低限度校验，并转成 `RewriteResult`。
    输入：`payload`，rewrite 模型输出解析后的字典。
    输出：`RewriteResult`，若关键字段缺失则抛异常。
    """
    return parse_rewrite_result(payload)


def main() -> None:
    """
    作用：主入口，串联参数解析、数据加载、plan/rewrite/judge、QC 和结果写出。
    输入：无，配置来自命令行参数。
    输出：无；主要副作用是打印日志并把结果写入输出 JSONL 文件。
    """
    args = parse_args()
    random.seed(args.seed)

    if args.download_only:
        if args.source != "medcase":
            raise ValueError("--download-only can only be used with --source medcase.")
        if args.dataset_name != DEFAULT_MEDCASE_DATASET_NAME:
            raise ValueError(
                "--download-only currently supports only the default "
                f"dataset: {DEFAULT_MEDCASE_DATASET_NAME}"
            )
        local_files = ensure_medcase_dataset_downloaded(args)
        print(f"Dataset downloaded to: {os.path.abspath(os.path.expanduser(args.dataset_dir))}")
        for split_name, relative_path in MEDCASE_SPLIT_TO_FILE.items():
            print(f"{split_name}: {local_files[relative_path]}")
        return

    llm_runtime = apply_llm_runtime_config(args)
    error_types = parse_error_types(args.error_types)
    samples = build_samples(args)
    if not samples:
        raise ValueError("No valid positive samples found after normalization.")

    print(f"Loaded {len(samples)} positive samples.")
    print(f"Target error types: {', '.join(error_types)}")
    print(f"Generation mode: {args.generation_mode}")
    print(f"Pipeline mode: {args.pipeline_mode}")
    print(f"Rewrite strategy: {args.rewrite_strategy}")
    print(f"LLM mode: {llm_runtime.llm_mode}")
    print(f"LLM model: {llm_runtime.model}")
    if llm_runtime.base_url:
        print(f"LLM base URL: {llm_runtime.base_url}")
    print(f"Output file: {args.output_file}")

    client = build_client(args)
    mode = "w" if args.overwrite else "a"
    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sidecar_paths = output_sidecar_paths(args.output_file)
    total_planned = len(samples) * (len(error_types) if args.generation_mode == "all" else 1)
    if args.overwrite:
        for sidecar_path in sidecar_paths.values():
            with open(sidecar_path, "w", encoding="utf-8"):
                pass
    recovered_attempts = 0
    if not args.overwrite:
        recovered_attempts = sync_attempt_ledger(args.output_file, sidecar_paths)
    seen = set() if args.overwrite else existing_keys(
        args.output_file,
        sidecar_paths["attempts"],
        sidecar_paths["review"],
        sidecar_paths["rejected"],
        sidecar_paths["infeasible"],
    )
    if args.overwrite:
        print("Resume mode: disabled because --overwrite was set.")
    else:
        print(
            "Resume mode: enabled. "
            f"Recovered {recovered_attempts} missing attempt records; "
            f"found {len(seen)} finished (source_id, error_type) pairs."
        )

    written = 0
    skipped = 0
    reviewed = 0
    rejected = 0
    infeasible = 0
    failed = 0
    write_progress_snapshot(
        sidecar_paths["progress"],
        output_file=args.output_file,
        total_planned=total_planned,
        seen=seen,
        written=written,
        reviewed=reviewed,
        rejected=rejected,
        infeasible=infeasible,
        failed=failed,
        skipped=skipped,
        phase="starting",
    )

    with open(args.output_file, mode, encoding="utf-8") as handle:
        # Generate negatives sample by sample and support resume when output already exists.
        try:
            for sample_index, sample in enumerate(samples, start=1):
                sample_error_types = requested_error_types_for_sample(
                    all_error_types=error_types,
                    mode=args.generation_mode,
                )

                for error_type in sample_error_types:
                    resume_key = (sample.source_id, error_type)
                    write_progress_snapshot(
                        sidecar_paths["progress"],
                        output_file=args.output_file,
                        total_planned=total_planned,
                        seen=seen,
                        written=written,
                        reviewed=reviewed,
                        rejected=rejected,
                        infeasible=infeasible,
                        failed=failed,
                        skipped=skipped,
                        phase="running",
                        current_sample_index=sample_index,
                        current_sample_id=sample.source_id,
                        current_error_type=error_type,
                    )
                    if resume_key in seen:
                        skipped += 1
                        continue

                    try:
                        preferred_preserve_answer = answer_should_be_preserved(error_type, args)
                        if args.pipeline_mode == "staged":
                            plan_payload = chat_json(
                                client=client,
                                model=args.model,
                                system_prompt=GENERATOR_SYSTEM_PROMPT,
                                user_prompt=build_plan_prompt(
                                    sample=sample,
                                    error_type=error_type,
                                    preferred_preserve_answer=preferred_preserve_answer,
                                ),
                                temperature=0.2,
                                max_tokens=args.plan_max_tokens,
                                max_retries=args.max_retries,
                            )
                            plan = parse_plan_result(plan_payload)
                            if error_type in {"F1", "F2", "F3"}:
                                plan = PlanResult(
                                    feasible=plan.feasible,
                                    preserve_answer=True,
                                    target_span=plan.target_span,
                                    rationale=plan.rationale,
                                    confidence=plan.confidence,
                                )

                            if not plan.feasible:
                                attempt_record = make_attempt_record(
                                    sample=sample,
                                    error_type=error_type,
                                    status="plan_infeasible",
                                    expected_answer_preserved=plan.preserve_answer,
                                    plan=plan,
                                )
                                append_jsonl_record(sidecar_paths["attempts"], attempt_record)
                                append_jsonl_record(sidecar_paths["infeasible"], attempt_record)
                                seen.add(resume_key)
                                infeasible += 1
                                write_progress_snapshot(
                                    sidecar_paths["progress"],
                                    output_file=args.output_file,
                                    total_planned=total_planned,
                                    seen=seen,
                                    written=written,
                                    reviewed=reviewed,
                                    rejected=rejected,
                                    infeasible=infeasible,
                                    failed=failed,
                                    skipped=skipped,
                                    phase="running",
                                    current_sample_index=sample_index,
                                    current_sample_id=sample.source_id,
                                    current_error_type=error_type,
                                )
                                print(
                                    f"[plan-infeasible] sample={sample_index} "
                                    f"source_id={sample.source_id} type={error_type} "
                                    f"plan={json.dumps(asdict(plan), ensure_ascii=False)}"
                                )
                                continue
                            if args.rewrite_strategy == "span_patch":
                                rewrite_prompt = build_patch_rewrite_prompt(
                                    sample=sample,
                                    error_type=error_type,
                                    plan=plan,
                                )
                            else:
                                rewrite_prompt = build_rewrite_prompt(
                                    sample=sample,
                                    error_type=error_type,
                                    plan=plan,
                                )
                        else:
                            if error_type in {"F1", "F2", "F3"}:
                                preferred_preserve_answer = True
                            plan = PlanResult(
                                feasible=True,
                                preserve_answer=preferred_preserve_answer,
                                target_span="",
                                rationale=(
                                    "rewrite_only pipeline: let the model choose one local span "
                                    "implicitly during rewrite"
                                ),
                                confidence=0.0,
                            )
                            if args.rewrite_strategy == "span_patch":
                                rewrite_prompt = build_direct_patch_rewrite_prompt(
                                    sample=sample,
                                    error_type=error_type,
                                    preserve_answer=plan.preserve_answer,
                                )
                            else:
                                rewrite_prompt = build_direct_rewrite_prompt(
                                    sample=sample,
                                    error_type=error_type,
                                    preserve_answer=plan.preserve_answer,
                                )

                        generation = chat_json(
                            client=client,
                            model=args.model,
                            system_prompt=GENERATOR_SYSTEM_PROMPT,
                            user_prompt=rewrite_prompt,
                            temperature=args.temperature,
                            max_tokens=(
                                args.rewrite_max_tokens
                                if args.rewrite_max_tokens > 0
                                else args.max_tokens
                            ),
                            max_retries=args.max_retries,
                        )
                        rewrite = validate_generation_payload(generation)
                        rewrite = materialize_rewrite_result(
                            sample=sample,
                            plan=plan,
                            rewrite=rewrite,
                            args=args,
                        )

                        heuristics_passed, heuristic_report = heuristic_qc(
                            sample=sample,
                            error_type=error_type,
                            preserve_answer=plan.preserve_answer,
                            candidate_reasoning=rewrite.negative_reasoning,
                            candidate_answer=rewrite.negative_final_answer,
                            args=args,
                        )
                        if not heuristics_passed:
                            attempt_record = make_attempt_record(
                                sample=sample,
                                error_type=error_type,
                                status="rejected_by_heuristic_qc",
                                expected_answer_preserved=plan.preserve_answer,
                                plan=plan,
                                rewrite=rewrite,
                                heuristic_report=heuristic_report,
                            )
                            append_jsonl_record(sidecar_paths["attempts"], attempt_record)
                            append_jsonl_record(sidecar_paths["rejected"], attempt_record)
                            seen.add(resume_key)
                            rejected += 1
                            write_progress_snapshot(
                                sidecar_paths["progress"],
                                output_file=args.output_file,
                                total_planned=total_planned,
                                seen=seen,
                                written=written,
                                reviewed=reviewed,
                                rejected=rejected,
                                infeasible=infeasible,
                                failed=failed,
                                skipped=skipped,
                                phase="running",
                                current_sample_index=sample_index,
                                current_sample_id=sample.source_id,
                                current_error_type=error_type,
                            )
                            print(
                                f"[heuristic-qc-fail] sample={sample_index} "
                                f"source_id={sample.source_id} type={error_type} "
                                f"report={json.dumps(heuristic_report, ensure_ascii=False)}"
                            )
                            continue

                        llm_qc_report: Optional[JudgeResult] = None
                        if args.enable_llm_qc:
                            llm_qc_report = judge_candidate(
                                client=client,
                                sample=sample,
                                error_type=error_type,
                                plan=plan,
                                rewrite=rewrite,
                                args=args,
                            )
                            if llm_qc_report.decision == "review":
                                record = make_output_record(
                                    sample=sample,
                                    error_type=error_type,
                                    status="needs_human_review",
                                    plan=plan,
                                    rewrite=rewrite,
                                    heuristic_report=heuristic_report,
                                    llm_qc_report=llm_qc_report,
                                )
                                append_jsonl_record(sidecar_paths["attempts"], record)
                                append_jsonl_record(sidecar_paths["review"], record)
                                seen.add(resume_key)
                                reviewed += 1
                                write_progress_snapshot(
                                    sidecar_paths["progress"],
                                    output_file=args.output_file,
                                    total_planned=total_planned,
                                    seen=seen,
                                    written=written,
                                    reviewed=reviewed,
                                    rejected=rejected,
                                    infeasible=infeasible,
                                    failed=failed,
                                    skipped=skipped,
                                    phase="running",
                                    current_sample_index=sample_index,
                                    current_sample_id=sample.source_id,
                                    current_error_type=error_type,
                                )
                                print(
                                    f"[judge-review] sample={sample_index} "
                                    f"source_id={sample.source_id} type={error_type} "
                                    f"qc={json.dumps(asdict(llm_qc_report), ensure_ascii=False)}"
                                )
                                continue

                            if not judge_passes(
                                judge_result=llm_qc_report,
                                error_type=error_type,
                                preserve_answer=plan.preserve_answer,
                            ):
                                record = make_output_record(
                                    sample=sample,
                                    error_type=error_type,
                                    status="rejected_by_judge",
                                    plan=plan,
                                    rewrite=rewrite,
                                    heuristic_report=heuristic_report,
                                    llm_qc_report=llm_qc_report,
                                )
                                append_jsonl_record(sidecar_paths["attempts"], record)
                                append_jsonl_record(sidecar_paths["rejected"], record)
                                seen.add(resume_key)
                                rejected += 1
                                write_progress_snapshot(
                                    sidecar_paths["progress"],
                                    output_file=args.output_file,
                                    total_planned=total_planned,
                                    seen=seen,
                                    written=written,
                                    reviewed=reviewed,
                                    rejected=rejected,
                                    infeasible=infeasible,
                                    failed=failed,
                                    skipped=skipped,
                                    phase="running",
                                    current_sample_index=sample_index,
                                    current_sample_id=sample.source_id,
                                    current_error_type=error_type,
                                )
                                print(
                                    f"[judge-reject] sample={sample_index} "
                                    f"source_id={sample.source_id} type={error_type} "
                                    f"qc={json.dumps(asdict(llm_qc_report), ensure_ascii=False)}"
                                )
                                continue

                        accepted_status = (
                            "accepted_by_judge" if args.enable_llm_qc else "accepted_without_judge"
                        )
                        record = make_output_record(
                            sample=sample,
                            error_type=error_type,
                            status=accepted_status,
                            plan=plan,
                            rewrite=rewrite,
                            heuristic_report=heuristic_report,
                            llm_qc_report=llm_qc_report,
                        )
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        append_jsonl_record(sidecar_paths["attempts"], record)
                        seen.add(resume_key)
                        written += 1
                        write_progress_snapshot(
                            sidecar_paths["progress"],
                            output_file=args.output_file,
                            total_planned=total_planned,
                            seen=seen,
                            written=written,
                            reviewed=reviewed,
                            rejected=rejected,
                            infeasible=infeasible,
                            failed=failed,
                            skipped=skipped,
                            phase="running",
                            current_sample_index=sample_index,
                            current_sample_id=sample.source_id,
                            current_error_type=error_type,
                        )
                        print(
                            f"[ok] sample={sample_index}/{len(samples)} "
                            f"source_id={sample.source_id} type={error_type} "
                            f"written={written}"
                        )
                    except Exception as exc:
                        failed += 1
                        retryable_error = is_retryable_runtime_error(exc)
                        plan_for_error = locals().get("plan")
                        rewrite_for_error = locals().get("rewrite")
                        attempt_record = make_attempt_record(
                            sample=sample,
                            error_type=error_type,
                            status=(
                                "generation_retryable_error"
                                if retryable_error
                                else "generation_failed"
                            ),
                            expected_answer_preserved=None,
                            plan=plan_for_error if isinstance(plan_for_error, PlanResult) else None,
                            rewrite=(
                                rewrite_for_error
                                if isinstance(rewrite_for_error, RewriteResult)
                                else None
                            ),
                            error_message=str(exc),
                        )
                        append_jsonl_record(sidecar_paths["attempts"], attempt_record)
                        if not retryable_error:
                            append_jsonl_record(sidecar_paths["failures"], attempt_record)
                        write_progress_snapshot(
                            sidecar_paths["progress"],
                            output_file=args.output_file,
                            total_planned=total_planned,
                            seen=seen,
                            written=written,
                            reviewed=reviewed,
                            rejected=rejected,
                            infeasible=infeasible,
                            failed=failed,
                            skipped=skipped,
                            phase="running",
                            current_sample_index=sample_index,
                            current_sample_id=sample.source_id,
                            current_error_type=error_type,
                        )
                        print(
                            (
                                f"[generation-retryable] sample={sample_index} "
                                f"source_id={sample.source_id} type={error_type} error={exc}"
                            )
                            if retryable_error
                            else (
                                f"[generation-fail] sample={sample_index} "
                                f"source_id={sample.source_id} type={error_type} error={exc}"
                            )
                        )

                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
        except KeyboardInterrupt:
            write_progress_snapshot(
                sidecar_paths["progress"],
                output_file=args.output_file,
                total_planned=total_planned,
                seen=seen,
                written=written,
                reviewed=reviewed,
                rejected=rejected,
                infeasible=infeasible,
                failed=failed,
                skipped=skipped,
                phase="interrupted",
            )
            print(
                "Interrupted. Re-run the same command without --overwrite to resume "
                "from existing output and sidecar files."
            )
            return

    write_progress_snapshot(
        sidecar_paths["progress"],
        output_file=args.output_file,
        total_planned=total_planned,
        seen=seen,
        written=written,
        reviewed=reviewed,
        rejected=rejected,
        infeasible=infeasible,
        failed=failed,
        skipped=skipped,
        phase="completed",
    )
    print(
        f"Done. written={written} reviewed={reviewed} rejected={rejected} "
        f"skipped={skipped} infeasible={infeasible} failed={failed} "
        f"output={args.output_file}"
    )


if __name__ == "__main__":
    main()
