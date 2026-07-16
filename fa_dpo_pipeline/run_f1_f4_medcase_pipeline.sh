#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
SPLITS="${SPLITS:-test}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-8}"
STUDY_DIR="${STUDY_DIR:-outputs/f1_f4_natural_error}"
PREP_FILE="${PREP_FILE:-result/f1_f4_natural_error/medcase_${SPLITS//,/_}.jsonl}"

if [[ -z "$MODEL" ]]; then
  echo "MODEL is required. Example:"
  echo "  MODEL=/home/models/Qwen3-Next-80B-A3B-Instruct bash fa_dpo_pipeline/run_f1_f4_medcase_pipeline.sh"
  exit 1
fi

MODEL_TAG="$(basename "$MODEL" | tr ' /' '__')"
GEN_FILE="${STUDY_DIR}/medcase_${MODEL_TAG}.jsonl"
ANALYSIS_FILE="${STUDY_DIR}/medcase_${MODEL_TAG}.faithfulness_analysis.jsonl"
SUMMARY_FILE="${STUDY_DIR}/medcase_${MODEL_TAG}.faithfulness_summary.json"
REPORT_FILE="${STUDY_DIR}/medcase_${MODEL_TAG}.faithfulness_report.md"
GROUP_FILE="${STUDY_DIR}/medcase_${MODEL_TAG}.faithfulness_group_summary.csv"

python fa_dpo_pipeline/prepare_medcase_reasoning_eval.py \
  --input-csv hf_datasets/MedCaseReasoning/medcasereasoning_core.csv \
  --splits "$SPLITS" \
  --max-samples "$MAX_SAMPLES" \
  --output-file "$PREP_FILE"

python fa_dpo_pipeline/generate_case_reasoning_eval.py \
  --input-file "$PREP_FILE" \
  --output-file "$GEN_FILE" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --max-samples "$MAX_SAMPLES"

python fa_dpo_pipeline/analyze_qa_faithfulness_errors.py \
  --input-file "$GEN_FILE" \
  --annotation-mode llm \
  --model "$JUDGE_MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --max-concurrency "$JUDGE_CONCURRENCY" \
  --group-by model_id,split \
  --output-file "$ANALYSIS_FILE" \
  --summary-file "$SUMMARY_FILE" \
  --report-file "$REPORT_FILE" \
  --group-summary-csv "$GROUP_FILE"

echo "Generation file: $GEN_FILE"
echo "Summary file: $SUMMARY_FILE"
echo "Report file: $REPORT_FILE"
