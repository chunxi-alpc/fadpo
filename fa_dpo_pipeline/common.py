from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def runtime_root() -> Path:
    configured = os.getenv("FA_DPO_RUNTIME_ROOT", "").strip()
    base = Path(configured).expanduser() if configured else Path.cwd()
    return base.resolve()


def _runtime_subdir(env_name: str, default_name: str) -> Path:
    configured = os.getenv(env_name, "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else (runtime_root() / path)
    return runtime_root() / default_name


def data_root() -> Path:
    return _runtime_subdir("FA_DPO_DATA_ROOT", "data")


def result_root() -> Path:
    return _runtime_subdir("FA_DPO_RESULT_ROOT", "result")


def artifact_root() -> Path:
    return _runtime_subdir("FA_DPO_ARTIFACT_ROOT", "artifacts")


def output_root() -> Path:
    return _runtime_subdir("FA_DPO_OUTPUT_ROOT", "outputs")


def checkpoint_root() -> Path:
    return _runtime_subdir("FA_DPO_CHECKPOINT_ROOT", "checkpoints")


def package_path(*parts: str) -> str:
    return str(PACKAGE_ROOT.joinpath(*parts))


def data_path(*parts: str) -> str:
    return str(data_root().joinpath(*parts))


def result_path(*parts: str) -> str:
    return str(result_root().joinpath(*parts))


def artifact_path(*parts: str) -> str:
    return str(artifact_root().joinpath(*parts))


def output_path(*parts: str) -> str:
    return str(output_root().joinpath(*parts))


def checkpoint_path(*parts: str) -> str:
    return str(checkpoint_root().joinpath(*parts))


def ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [normalize_text(item) for item in value]
        return "\n\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def first_non_empty(record: Dict[str, Any], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        if key not in record:
            continue
        value = normalize_text(record.get(key))
        if value:
            return value
    return default


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    output_path = ensure_parent(path)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[7:]
    elif stripped.startswith("```"):
        stripped = stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def extract_json(text: str) -> Dict[str, Any]:
    clean = strip_code_fences(text)
    decoder = JSONDecoder()
    try:
        payload = json.loads(clean)
        if isinstance(payload, dict):
            return payload
    except JSONDecodeError:
        pass

    for match in re.finditer(r"\{", clean):
        start = match.start()
        try:
            payload, _ = decoder.raw_decode(clean[start:])
        except JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Could not extract a JSON object from model output.")


def build_full_response(reasoning: str, answer: str) -> str:
    reasoning_text = normalize_text(reasoning)
    answer_text = normalize_text(answer)
    if reasoning_text and answer_text:
        return f"{reasoning_text}\n\nFinal Answer: {answer_text}"
    return reasoning_text or answer_text


def count_jsonl(path: str | Path) -> int:
    file_path = Path(path)
    if not file_path.exists():
        return 0
    with file_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = normalize_text(value).lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    return False


def normalize_answer(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_metadata_path(stage_name: str) -> Path:
    return ensure_parent(artifact_root() / "fa_dpo_pipeline" / "run_metadata" / f"{stage_name}.json")


def _json_ready(value: Any) -> Any:
    if isinstance(value, argparse.Namespace):
        return {key: _json_ready(val) for key, val in vars(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


def _elapsed_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    if not started_at or not finished_at:
        return None
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(finished_at)
    except ValueError:
        return None
    return round((end - start).total_seconds(), 4)


def collect_hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
    }
    try:
        cpu_count = subprocess.run(
            ["bash", "-lc", "nproc"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if cpu_count:
            info["cpu_count"] = int(cpu_count)
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        gpus: List[Dict[str, Any]] = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                gpus.append(
                    {
                        "name": parts[0],
                        "memory_total": parts[1],
                        "driver_version": parts[2],
                    }
                )
        if gpus:
            info["gpus"] = gpus
            info["gpu_count"] = len(gpus)
    except Exception:
        pass
    return info


def write_run_metadata(
    *,
    stage_name: str,
    args: argparse.Namespace | Mapping[str, Any] | None,
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    stats: Mapping[str, Any] | None = None,
    metadata_file: str | Path | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    status: str = "completed",
) -> Path:
    payload = {
        "stage_name": stage_name,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": _elapsed_seconds(started_at, finished_at),
        "args": _json_ready(args),
        "inputs": _json_ready(inputs or {}),
        "outputs": _json_ready(outputs or {}),
        "stats": _json_ready(stats or {}),
        "hardware": collect_hardware_info(),
    }
    target_path = ensure_parent(metadata_file) if metadata_file else default_run_metadata_path(stage_name)
    write_json(target_path, payload)
    return target_path


def run_python_script(script_relpath: str, passthrough_args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    script_path = REPO_ROOT / script_relpath
    command = [sys.executable, str(script_path), *passthrough_args]
    return subprocess.run(command, check=True)
