#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/generate_bad_reasoning.py"

PYTHON_BIN="${PYTHON_BIN:-python}"
MATRIX_FILE="${MATRIX_FILE:-$ROOT_DIR/comparison/model_matrix.example.tsv}"

SOURCE="${SOURCE:-medcase}"
DATASET_NAME="${DATASET_NAME:-zou-lab/MedCaseReasoning}"
SPLIT="${SPLIT:-test}"
INPUT_FILE="${INPUT_FILE:-}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/hf_datasets/MedCaseReasoning}"
EVIDENCE_FIELD="${EVIDENCE_FIELD:-text}"

ERROR_TYPES="${ERROR_TYPES:-F1,F2,F3,F4}"
GENERATION_MODE="${GENERATION_MODE:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
START_INDEX="${START_INDEX:-0}"
TEMPERATURE="${TEMPERATURE:-0.3}"
MAX_TOKENS="${MAX_TOKENS:-1800}"
MAX_RETRIES="${MAX_RETRIES:-3}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.0}"
SEEDS="${SEEDS:-42 43 44}"

# For cross-model comparison, keep this OFF by default.
# The current generator uses the same endpoint/client for generation and QC.
ENABLE_LLM_QC="${ENABLE_LLM_QC:-0}"
QC_MODEL="${QC_MODEL:-}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/comparison_outputs/$RUN_TAG}"

if [[ ! -f "$MATRIX_FILE" ]]; then
  echo "Matrix file not found: $MATRIX_FILE" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

echo "Run tag: $RUN_TAG"
echo "Output root: $OUT_ROOT"
echo "Matrix file: $MATRIX_FILE"

while IFS=$'\t' read -r enabled model_id llm_mode model_name base_url api_key_env notes; do
  if [[ "$enabled" == "enabled" ]]; then
    continue
  fi

  if [[ "$enabled" != "1" ]]; then
    continue
  fi

  api_key=""
  if [[ -n "${api_key_env:-}" && "${api_key_env:-}" != "-" ]]; then
    api_key="${!api_key_env:-}"
  fi

  if [[ "$llm_mode" == "local" && -z "$api_key" ]]; then
    api_key="EMPTY"
  fi

  echo
  echo "=== Model: $model_id ==="
  echo "Mode: $llm_mode"
  echo "Model route: $model_name"
  echo "Base URL: $base_url"
  echo "Notes: $notes"

  for seed in $SEEDS; do
    model_out_dir="$OUT_ROOT/$model_id/seed_${seed}"
    output_file="$model_out_dir/accepted.jsonl"
    mkdir -p "$model_out_dir"

    cmd=(
      "$PYTHON_BIN" "$SCRIPT_PATH"
      --source "$SOURCE"
      --dataset-name "$DATASET_NAME"
      --split "$SPLIT"
      --dataset-dir "$DATASET_DIR"
      --output-file "$output_file"
      --llm-mode "$llm_mode"
      --model "$model_name"
      --base-url "$base_url"
      --api-key "$api_key"
      --error-types "$ERROR_TYPES"
      --generation-mode "$GENERATION_MODE"
      --max-samples "$MAX_SAMPLES"
      --start-index "$START_INDEX"
      --seed "$seed"
      --temperature "$TEMPERATURE"
      --max-tokens "$MAX_TOKENS"
      --max-retries "$MAX_RETRIES"
      --sleep-seconds "$SLEEP_SECONDS"
      --evidence-field "$EVIDENCE_FIELD"
      --overwrite
    )

    if [[ "$SOURCE" == "local" ]]; then
      if [[ -z "$INPUT_FILE" ]]; then
        echo "INPUT_FILE must be set when SOURCE=local" >&2
        exit 1
      fi
      cmd+=(--input-file "$INPUT_FILE")
    fi

    if [[ "$ENABLE_LLM_QC" == "1" ]]; then
      cmd+=(--enable-llm-qc)
      if [[ -n "$QC_MODEL" ]]; then
        cmd+=(--qc-model "$QC_MODEL")
      fi
    fi

    echo
    echo "[run] model=$model_id seed=$seed"
    printf ' %q' "${cmd[@]}"
    echo
    "${cmd[@]}"
  done
done < "$MATRIX_FILE"

echo
echo "All runs finished. Outputs are under: $OUT_ROOT"
