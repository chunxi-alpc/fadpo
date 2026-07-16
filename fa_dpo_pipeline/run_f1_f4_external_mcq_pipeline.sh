#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-outputs/f1_f4_natural_error}"
INPUT_FILE="${INPUT_FILE:-}"
MODEL="${MODEL:-}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
DATASET_NAME="${DATASET_NAME:-}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-8}"

if [[ -z "${INPUT_FILE}" ]]; then
  echo "Set INPUT_FILE=/path/to/benchmark.jsonl or .json"
  exit 1
fi

if [[ -z "${MODEL}" ]]; then
  echo "Set MODEL=/path/or/name/of/model"
  exit 1
fi

mkdir -p "${ROOT_DIR}"

DATASET_STEM="$(basename "${INPUT_FILE}")"
DATASET_STEM="${DATASET_STEM%.*}"
if [[ -n "${DATASET_NAME}" ]]; then
  DATASET_STEM="${DATASET_NAME}"
fi

MODEL_STEM="$(basename "${MODEL}")"
PREP_FILE="${PREP_FILE:-${ROOT_DIR}/${DATASET_STEM}.normalized.jsonl}"
GEN_FILE="${GEN_FILE:-${ROOT_DIR}/${DATASET_STEM}.${MODEL_STEM}.jsonl}"

prep_cmd=(
  python fa_dpo_pipeline/prepare_benchmark_reasoning_eval.py
  --input-file "${INPUT_FILE}"
  --output-file "${PREP_FILE}"
  --max-samples "${MAX_SAMPLES}"
)
if [[ -n "${DATASET_NAME}" ]]; then
  prep_cmd+=(--dataset-name "${DATASET_NAME}")
fi
"${prep_cmd[@]}"

gen_cmd=(
  python fa_dpo_pipeline/generate_mcq_reasoning_eval.py
  --input-file "${PREP_FILE}"
  --output-file "${GEN_FILE}"
  --model "${MODEL}"
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --max-samples "${MAX_SAMPLES}"
)
"${gen_cmd[@]}"

analyze_cmd=(
  python fa_dpo_pipeline/analyze_qa_faithfulness_errors.py
  --input-file "${GEN_FILE}"
  --annotation-mode llm
  --analysis-profile mcq_f4
  --model "${JUDGE_MODEL}"
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --max-concurrency "${JUDGE_CONCURRENCY}"
  --group-by model_id,dataset_name,subset
  --max-samples "${MAX_SAMPLES}"
)
"${analyze_cmd[@]}"
