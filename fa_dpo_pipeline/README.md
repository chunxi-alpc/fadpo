# Fa-DPO 流水线

这个目录现在已经整理成“单文件夹可上传”的版本。

你的使用方式可以非常简单：

- 只上传 `fa_dpo_pipeline/` 这个文件夹到服务器
- 在服务器上准备一个执行根目录
- 把现成数据放到执行根目录下的 `data/`
- 从执行根目录运行 `python fa_dpo_pipeline/<script>.py`

也就是说，代码和数据分开：

- 代码在 `fa_dpo_pipeline/`
- 数据在执行根目录的 `data/`
- 新生成的结果默认写到执行根目录下的：
  - `result/`
  - `artifacts/`
  - `outputs/`
  - `checkpoints/`

## 推荐目录结构

假设你的执行根目录是 `/workspace/med_run`，推荐长这样：

```text
/workspace/med_run/
  data/
    result/
      qwen-next-stage2/
        medcase_unfaithful_negatives.jsonl
    hf_datasets/
      MedCaseReasoning/
        medcasereasoning_core.csv
  fa_dpo_pipeline/
    ...
  result/
  artifacts/
  outputs/
  checkpoints/
```

默认情况下，所有脚本都会把当前工作目录当作执行根目录。

如果你不想依赖当前目录，也可以手动指定：

- `FA_DPO_RUNTIME_ROOT`
- `FA_DPO_DATA_ROOT`
- `FA_DPO_RESULT_ROOT`
- `FA_DPO_ARTIFACT_ROOT`
- `FA_DPO_OUTPUT_ROOT`
- `FA_DPO_CHECKPOINT_ROOT`

## 目录说明

- `00_build_miriad_corpus.py`
  - 导出 MIRIAD 语料到执行根目录下的 `artifacts/`
- `01_build_miriad_index.py`
  - 构建 MedCPT FAISS 索引
- `02_generate_negatives.py`
  - 受控负样本生成入口
  - 实际调用 `legacy/generate_bad_reasoning.py`
- `03_build_qc_artifacts.py`
  - strict-clean 和 review sample 构建入口
  - 实际调用 `legacy/build_negative_qc_artifacts.py`
- `04_decompose_negatives_to_arus.py`
  - ARU 分解
  - 为 `W/D` 节点同时生成 `retrieval_query`
- `05_attach_aru_evidence.py`
  - 给 `W/D` 节点挂检索证据
  - 优先使用 `retrieval_query`，并对每个节点生成少量 query variants 后再 merge + rerank
- `06_score_faithfulness.py`
  - 计算 ARU 级风险和样本级 margin
- `07_prepare_topodpo_data.py`
  - 构造 Topo-DPO / Fa-DPO 训练集
- `08_train_topodpo.py`
  - 训练入口
  - 实际调用 `trainers/topodpo.py`

补充目录：

- `review/`
  - 医生复审相关工具
- `legacy/`
  - 为了保持 02/03 可用而收进来的旧脚本
- `trainers/`
  - 训练脚本
- `docs/experiment_plan.md`
  - 实验计划
- `docs/aru_retrieval_failure_taxonomy.md`
  - ARU 分解和检索失败模式总结
- `manual_review_templates/`
  - 人工标注模板

## 默认输入输出规则

### 主流水线

- 输入：
  - 优先用执行过程中刚生成的 `result/` 和 `artifacts/`
- 输出：
  - `result/fa_dpo_pipeline/`
  - `artifacts/fa_dpo_pipeline/`
  - `checkpoints/fa_dpo_pipeline/`

### 分析实验

- 输出统一写到：
  - `outputs/`

### 外部已有数据

如果是你已经在服务器上准备好的数据，默认从执行根目录下的 `data/` 读取，例如：

- `data/result/qwen-next-stage2/medcase_unfaithful_negatives.jsonl`
- `data/hf_datasets/MedCaseReasoning/medcasereasoning_core.csv`

## 安装依赖

```bash
pip install -r fa_dpo_pipeline/requirements.txt
```

说明：

- 如果你用 GPU 版 FAISS，请把 `faiss-cpu` 换成环境对应的 `faiss-gpu` 或 conda 包
- `02_generate_negatives.py` 和 `04_decompose_negatives_to_arus.py` 需要可用的 OpenAI-compatible LLM 服务

## 两种常见用法

如果你要直接按服务器流程跑 `qwen-next-stage2`，优先看：

- `docs/qwen_next_stage2_server_run.md`

### 用法 1：从头跑主流水线

```bash
python fa_dpo_pipeline/00_build_miriad_corpus.py
python fa_dpo_pipeline/01_build_miriad_index.py

python fa_dpo_pipeline/02_generate_negatives.py
python fa_dpo_pipeline/03_build_qc_artifacts.py

python fa_dpo_pipeline/04_decompose_negatives_to_arus.py
python fa_dpo_pipeline/05_attach_aru_evidence.py
python fa_dpo_pipeline/06_score_faithfulness.py
python fa_dpo_pipeline/07_prepare_topodpo_data.py \
  --base-model TsinghuaC3I/Llama-3.1-8B-UltraMedical

python fa_dpo_pipeline/08_train_topodpo.py \
  --base-model TsinghuaC3I/Llama-3.1-8B-UltraMedical
```

### 用法 2：直接从 `qwen-next-stage2` 开始

如果服务器上已经有：

- `data/result/qwen-next-stage2/medcase_unfaithful_negatives.jsonl`

那就不需要再跑 `02/03`，可以直接从 `04` 开始：

```bash
python fa_dpo_pipeline/00_build_miriad_corpus.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python fa_dpo_pipeline/01_build_miriad_index.py \
  --batch-size 1024 \
  --device cuda

# 这一步建议在单独的 tmux/screen 窗口里启动
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /home/models/Qwen3-Next-80B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90

python fa_dpo_pipeline/04_decompose_negatives_to_arus.py \
  --input-file data/result/qwen-next-stage2/medcase_unfaithful_negatives.jsonl \
  --output-file result/qwen-next-stage2/medcase_unfaithful_negatives.arus.jsonl \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --model /home/models/Qwen3-Next-80B-A3B-Instruct \
  --max-concurrency 64

# 04 跑完后停掉 vLLM；如果是后台起的，也可以直接用这句
pkill -f "vllm serve /home/models/Qwen3-Next-80B-A3B-Instruct"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python fa_dpo_pipeline/05_attach_aru_evidence.py \
  --aru-file result/qwen-next-stage2/medcase_unfaithful_negatives.arus.jsonl \
  --output-file result/qwen-next-stage2/medcase_unfaithful_negatives.arus.with_evidence.jsonl \
  --batch-size 1024 \
  --rerank-batch-size 128 \
  --device cuda

# 说明：
# - 05 会优先读取 04 产出的 `retrieval_query`
# - 如果 04 没产出，05 会自动做 heuristic query rewrite
# - 如果环境里装的是 GPU 版 FAISS，05 会自动把检索阶段切到 GPU
# - 如果希望进一步压低显存占用、追求更快检索，可额外加 --faiss-gpu-use-float16

python fa_dpo_pipeline/sample_retrieval_review.py \
  --aru-file result/qwen-next-stage2/medcase_unfaithful_negatives.arus.with_evidence.jsonl \
  --total-samples 120 \
  --output-jsonl outputs/retrieval_review/qwen_next_stage2.sample.jsonl \
  --output-csv outputs/retrieval_review/qwen_next_stage2.sample.csv \
  --output-summary outputs/retrieval_review/qwen_next_stage2.summary.json

# 说明：
# - 这一步是可选的，但很适合在跑 06 前先抽查 W/D 检索质量
# - sample 输出里现在会保留 `retrieval_query` 和 `retrieval_query_variants`

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python fa_dpo_pipeline/06_score_faithfulness.py \
  --input-file result/qwen-next-stage2/medcase_unfaithful_negatives.arus.with_evidence.jsonl \
  --output-file result/qwen-next-stage2/medcase_unfaithful_negatives.scores.jsonl \
  --batch-size 256 \
  --device cuda

python fa_dpo_pipeline/07_prepare_topodpo_data.py \
  --negatives-file data/result/qwen-next-stage2/medcase_unfaithful_negatives.jsonl \
  --aru-file result/qwen-next-stage2/medcase_unfaithful_negatives.arus.with_evidence.jsonl \
  --score-file result/qwen-next-stage2/medcase_unfaithful_negatives.scores.jsonl \
  --output-dir artifacts/qwen-next-stage2/topodpo_data \
  --base-model TsinghuaC3I/Llama-3.1-8B-UltraMedical
```

训练时，推荐直接多卡启动底层训练脚本：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  fa_dpo_pipeline/trainers/topodpo.py \
  --base_model TsinghuaC3I/Llama-3.1-8B-UltraMedical \
  --train_dataset artifacts/qwen-next-stage2/topodpo_data/train_topodpo.jsonl \
  --eval_dataset artifacts/qwen-next-stage2/topodpo_data/eval_topodpo.jsonl \
  --output_dir checkpoints/qwen-next-stage2/topodpo \
  --batch_size 1 \
  --grad_accum 4 \
  --learning_rate 5e-7
```

## 主流水线产物

默认会生成：

- `artifacts/fa_dpo_pipeline/miriad_corpus.jsonl`
- `artifacts/fa_dpo_pipeline/miriad_faiss.index`
- `artifacts/fa_dpo_pipeline/miriad_medcpt.offsets.npy`
- `result/fa_dpo_pipeline/medcase_unfaithful_negatives.jsonl`
- `result/fa_dpo_pipeline/medcase_unfaithful_negatives.strict_clean.jsonl`
- `result/fa_dpo_pipeline/medcase_unfaithful_negatives.strict_clean.arus.jsonl`
- `result/fa_dpo_pipeline/medcase_unfaithful_negatives.strict_clean.arus.with_evidence.jsonl`
- `result/fa_dpo_pipeline/medcase_unfaithful_negatives.strict_clean.scores.jsonl`
- `artifacts/fa_dpo_pipeline/topodpo_data/train_topodpo.jsonl`
- `artifacts/fa_dpo_pipeline/topodpo_data/eval_topodpo.jsonl`
- `checkpoints/fa_dpo_pipeline/topodpo/`

运行元数据会统一写到：

- `artifacts/fa_dpo_pipeline/run_metadata/`

这些元数据会记录：

- 输入输出路径
- 启停时间
- 耗时
- 样本计数
- 基本硬件信息

## 分析实验入口

下面这些分析脚本现在都已经收进这个单文件夹里了：

- `python fa_dpo_pipeline/analyze_qa_faithfulness_errors.py`
- `python fa_dpo_pipeline/prepare_medcase_reasoning_eval.py`
- `python fa_dpo_pipeline/prepare_benchmark_reasoning_eval.py`
- `python fa_dpo_pipeline/generate_case_reasoning_eval.py`
- `python fa_dpo_pipeline/generate_mcq_reasoning_eval.py`
- `bash fa_dpo_pipeline/run_f1_f4_medcase_pipeline.sh`
- `bash fa_dpo_pipeline/run_f1_f4_external_mcq_pipeline.sh`
- `python fa_dpo_pipeline/aggregate_f1_f4_study_results.py`
- `python fa_dpo_pipeline/f1_f4_study/run_study.py`
- `python fa_dpo_pipeline/f1_f4_study/analyze_stage_results.py`
- `python fa_dpo_pipeline/sample_nli_validation.py`
- `python fa_dpo_pipeline/eval_nli_validation.py`
- `python fa_dpo_pipeline/build_retrieval_metadata.py`
- `python fa_dpo_pipeline/eval_retrieval_quality.py`
- `python fa_dpo_pipeline/build_same_answer_subset.py`
- `python fa_dpo_pipeline/eval_process_metrics.py`
- `python fa_dpo_pipeline/run_pairwise_judge.py`
- `python fa_dpo_pipeline/bootstrap_judge_ci.py`
- `python fa_dpo_pipeline/check_overlap_exact.py`
- `python fa_dpo_pipeline/check_overlap_neardup.py`
- `python fa_dpo_pipeline/prepare_near_duplicate_review_sheet.py`
- `python fa_dpo_pipeline/prepare_runtime_review_sheet.py`
- `python fa_dpo_pipeline/aggregate_costs.py`
- `python fa_dpo_pipeline/aggregate_multiseed_results.py`

这些脚本的输出默认都会写到执行根目录下的 `outputs/`。

## 医生复审工具

现在对应入口是：

- `python fa_dpo_pipeline/review/build_simulated_clinical_review.py`
- `python fa_dpo_pipeline/review/finalize_clinical_review.py`
- `python fa_dpo_pipeline/review/merge_negative_pool.py`

如果你服务器上的第二阶段原始数据在：

- `data/result/qwen-next-stage2/`

那么这几个脚本默认会从那里读输入，再把新结果写到执行根目录的 `result/`。

## 人工模板

模板文件已经收进单文件夹里：

- `fa_dpo_pipeline/manual_review_templates/nli_annotation_example.csv`
- `fa_dpo_pipeline/manual_review_templates/near_duplicate_review_example.csv`
- `fa_dpo_pipeline/manual_review_templates/runtime_review_example.csv`

## 实验计划

实验计划文档现在在：

- `fa_dpo_pipeline/docs/experiment_plan.md`

## 当前已经落地的分析脚本

已经能直接跑的：

- `P0.1`
- `P0.2`
- `P0.3`
- `P0.4`
- `P0.5`
- `P0.6`
- `P0.7`

还没补脚本的：

- `P1.1`
- `P1.2`
- `P1.3`
- `P2.1`
- `P2.2`
- `P2.3`

## 一句话总结

现在这套结构的目标就是：

- 你只上传 `fa_dpo_pipeline/`
- 服务器已有数据就放在执行根目录的 `data/`
- 从执行根目录直接跑
- 中间结果、分析结果、训练结果自动分流到各自目录
