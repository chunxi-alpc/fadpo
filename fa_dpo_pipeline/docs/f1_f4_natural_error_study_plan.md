# F1-F4 Natural Error Study Plan

这份文档专门回答一个问题：

> 大模型在医学推理生成中，是否会自然地产生 F1/F2/F3/F4 这四类 faithfulness 错误？

这里的重点不是“我们能不能在一个数据集上构造 F1-F4 负样本”，而是“在真实模型输出里，这些错误是否稳定存在、是否跨模型出现、是否跨数据集出现”。

## 1. 最有说服力的组合

最推荐的组合是：

### 1.1 数据集组合

- 主数据集：`MedCaseReasoning test`
  - 作用：主分析集
  - 原因：天然有 `case_prompt / diagnostic_reasoning / final_diagnosis`
  - 优势：最适合分析 `(X,G) -> c -> a` 链条，尤其是 F1/F2/F3

- 外部验证集 A：`MedBullets OP4 + OP5`
  - 作用：高质量医学考试题验证
  - 优势：题目自然、医学知识密度高、社区熟悉
  - 局限：原始文件是 MCQ，不天然提供 gold reasoning；F1/F2/F3 需要额外补证据或弱化解释

- 外部验证集 B：`MedQA`
  - 作用：更通用的大规模医疗 MCQ 验证
  - 优势：广泛使用，便于横向比较
  - 局限：和 MedBullets 类似，更适合验证 F4 与“答对但理由不对”

- 外部验证集 C：`Med-HALT reasoning`
  - 作用：对抗式 / 异常推理压力测试
  - 优势：能验证模型在困难、反常、诱导性题目中的推理稳定性
  - 局限：更适合作为 robustness 附加验证，不适合作为 F1/F2/F3 主场

### 1.2 模型组合

- 模型组 1：前沿闭源或网关强模型
  - 例如你当前能访问的 `gpt-*` 路由
  - 作用：代表“最强上限”

- 模型组 2：强开源通用模型
  - 例如 `Qwen3-Next` 或你本地已有的强通用推理模型
  - 作用：代表当前开源主力

- 模型组 3：医学领域模型
  - 例如仓库现有的 `MedReason-8B` 或你自己的医疗 DPO 模型
  - 作用：验证“医学专门模型也是否存在同类问题”

- 可选模型组 4：更小模型
  - 作用：形成规模梯度，分析错误随能力变化的趋势

### 1.3 为什么这个组合最有说服力

- 只用 `MedCaseReasoning`：
  - 能做现象发现
  - 但容易被质疑为 dataset-specific

- 只用 MCQ benchmark：
  - 容易跑
  - 但 F1/F2/F3 很难定义得稳

- `MedCaseReasoning + MedBullets/MedQA + Med-HALT + 多模型`：
  - 同时覆盖病例型、考试型、对抗型
  - 同时覆盖闭源强模型、开源强模型、医学模型
  - 最容易把结论提升到“跨数据集、跨模型都存在的系统性问题”

## 2. 我们真正要回答的研究问题

### RQ1

在病例型医学推理生成中，F1/F2/F3/F4 是否自然出现？

- 主回答数据：`MedCaseReasoning test`

### RQ2

这些错误是否只出现在某一个模型，还是多个模型都会出现？

- 主回答设置：多模型对比

### RQ3

这些错误是否只在某一个数据集里出现，还是换到别的数据源也能观察到？

- 主回答设置：多数据集外部验证

### RQ4

这些错误是否仅仅等价于“答错了”，还是存在“答对但推理不忠实”？

- 主回答指标：按 answer correctness 分层

## 3. 评测设计原则

### 3.1 主分析与外部验证分工

- 主分析：
  - `MedCaseReasoning test`
  - 用于最严格地定义 F1/F2/F3/F4

- 外部验证：
  - `MedBullets OP4/OP5`
  - `MedQA`
  - `Med-HALT reasoning`
  - 用于证明现象泛化，不是单一数据集产物

### 3.2 F1-F4 的评测边界

- F1/F2/F3 需要可比较稳定的“可用证据集”
- F4 只需要 reasoning 和 answer 的一致性关系

所以：

- 在 `MedCaseReasoning` 上：
  - F1/F2/F3/F4 都做

- 在 MCQ 数据集上：
  - 主做 F4
  - F1/F2/F3 作为弱验证，前提是补统一检索证据或参考解释

### 3.3 不把“隐藏思维”说得过头

论文表述建议坚持：

- 我们分析的是 `visible reasoning traces`
- 我们发现的是“模型生成出的推理轨迹中的 faithfulness error”
- 不把它直接上升为“模型内部真实思维机制”

这样更稳，也更难被审稿人抓语言漏洞。

## 4. 分阶段实施

## Phase 0: 数据统一

### 目标

把主数据集和外部数据集统一成一套分析输入格式。

### 统一格式

每条样本尽量包含：

- `id`
- `dataset_name`
- `split`
- `patient_context` 或 `question`
- `reference_reasoning` 或 `reasoning`
- `reference_answer` 或 `answer`
- `output`
- `predicted_reasoning`
- `predicted_answer_text`
- `model_id`
- `source`

### 已实现

- [prepare_medcase_reasoning_eval.py](/root/med/fa_dpo_pipeline/prepare_medcase_reasoning_eval.py)
  - 把 `medcasereasoning_core.csv` 转成统一 JSONL
- [prepare_benchmark_reasoning_eval.py](/root/med/fa_dpo_pipeline/prepare_benchmark_reasoning_eval.py)
  - 把 `MedBullets / MedQA / Med-HALT` 这类外部 MCQ 基准统一成同一套 JSONL

### 建议命令

```bash
python fa_dpo_pipeline/prepare_medcase_reasoning_eval.py \
  --input-csv hf_datasets/MedCaseReasoning/medcasereasoning_core.csv \
  --splits test \
  --output-file result/f1_f4_natural_error/medcase_test.jsonl
```

## Phase 1: 主数据集生成

### 目标

先在 `MedCaseReasoning test` 上得到多个模型的可见推理输出。

### 已实现

- [generate_case_reasoning_eval.py](/root/med/fa_dpo_pipeline/generate_case_reasoning_eval.py)
  - 对病例型数据生成 reasoning + final answer

### 建议命令

```bash
python fa_dpo_pipeline/generate_case_reasoning_eval.py \
  --input-file result/f1_f4_natural_error/medcase_test.jsonl \
  --output-file outputs/f1_f4_natural_error/medcase_test_qwen_next.jsonl \
  --model /home/models/Qwen3-Next-80B-A3B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --max-concurrency 16
```

对多个模型分别跑：

- 闭源强模型
- 开源强模型
- 医学模型

## Phase 2: F1-F4 自动判定

### 目标

对模型真实输出进行 F1/F2/F3/F4 归因统计。

### 已实现

- [analyze_qa_faithfulness_errors.py](/root/med/fa_dpo_pipeline/analyze_qa_faithfulness_errors.py)
  - 支持 `json/jsonl/csv`
  - 支持 `llm judge` 模式
  - 支持总体统计、分组统计、报告导出

### 主分析命令

```bash
python fa_dpo_pipeline/analyze_qa_faithfulness_errors.py \
  --input-file outputs/f1_f4_natural_error/medcase_test_qwen_next.jsonl \
  --annotation-mode llm \
  --model /home/models/Qwen3-Next-80B-A3B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --group-by model_id,split
```

### 主表建议

- Overall F1/F2/F3/F4 presence rate
- Dominant-error rate
- Single-dominant-error rate
- Answer exact match
- Correct-answer subset 上的 F1-F4 rate

## Phase 3: 外部数据集验证

### 目标

证明现象不是 `MedCaseReasoning` 特供。

### 3.1 MedBullets / MedQA

现在更推荐直接走统一外部流水线：

- [prepare_benchmark_reasoning_eval.py](/root/med/fa_dpo_pipeline/prepare_benchmark_reasoning_eval.py)
- [generate_mcq_reasoning_eval.py](/root/med/fa_dpo_pipeline/generate_mcq_reasoning_eval.py)
- [run_f1_f4_external_mcq_pipeline.sh](/root/med/fa_dpo_pipeline/run_f1_f4_external_mcq_pipeline.sh)

这样会直接产出与主数据集兼容的：

- `output`
- `predicted_reasoning`
- `predicted_answer_text`
- `model_id`
- `dataset_name / subset / source`

但这里建议：

- 主分析 F4
- F1/F2/F3 只在补证据后作为弱结论

### 建议命令

```bash
INPUT_FILE=test/eval_data/medbullets_op4.jsonl \
DATASET_NAME=MedBullets-OP4 \
MODEL=/home/models/Qwen3-Next-80B-A3B-Instruct \
BASE_URL=http://localhost:8000/v1 \
API_KEY=EMPTY \
bash fa_dpo_pipeline/run_f1_f4_external_mcq_pipeline.sh
```

如果你想单独拆开三步跑：

```bash
python fa_dpo_pipeline/prepare_benchmark_reasoning_eval.py \
  --input-file test/eval_data/medqa_test.jsonl \
  --dataset-name MedQA \
  --output-file outputs/f1_f4_natural_error/medqa.normalized.jsonl

python fa_dpo_pipeline/generate_mcq_reasoning_eval.py \
  --input-file outputs/f1_f4_natural_error/medqa.normalized.jsonl \
  --output-file outputs/f1_f4_natural_error/medqa.qwen_next.jsonl \
  --model /home/models/Qwen3-Next-80B-A3B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY

python fa_dpo_pipeline/analyze_qa_faithfulness_errors.py \
  --input-file outputs/f1_f4_natural_error/medqa.qwen_next.jsonl \
  --annotation-mode llm \
  --analysis-profile mcq_f4 \
  --model /home/models/Qwen3-Next-80B-A3B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --group-by model_id,dataset_name,subset
```

### 3.2 Med-HALT reasoning

用途：

- 验证在反常推理或诱导性问题里，F4 是否明显增加
- 验证模型是否更容易出现 unsupported answer

### 外部验证输出

每个模型、每个外部数据集至少导出：

- answer accuracy
- dominant F4 rate
- any-error rate
- wrong-answer subset 的 F4 rate

### 多数据集总汇总命令

```bash
python fa_dpo_pipeline/aggregate_f1_f4_study_results.py \
  --summary-glob 'outputs/f1_f4_natural_error/*.faithfulness_summary.json' \
  --output-prefix outputs/f1_f4_natural_error/study_aggregate
```

这一步会统一导出：

- `study_aggregate.json`
- `study_aggregate.csv`
- `study_aggregate.groups.csv`
- `study_aggregate.md`

## Phase 4: 人工审查

### 目标

避免审稿人质疑自动 judge 自说自话。

### 建议抽样

- 主数据集：每模型抽 `50-80` 条
- 外部数据集：每数据集每模型抽 `20-30` 条
- 分层覆盖：
  - F1/F2/F3/F4
  - answer correct / answer wrong
  - judge high confidence / low confidence

### 人工要回答的问题

- 目标错误类型是否成立
- 是否是单一主导错误
- 是否只是“写法不同但合理”
- F4 是否真的是 reasoning-answer mismatch

## Phase 5: 论文结果组织

### 主文最应该放的三张表

#### Table 1

`MedCaseReasoning test` 上的多模型 F1/F2/F3/F4 总体分布

#### Table 2

Correct-answer subset 上的 F1/F2/F3/F4

核心回答：

- 模型即使答对，仍然会出现不忠实推理

#### Table 3

外部数据集上的主结论

- MedBullets / MedQA / Med-HALT
- 至少报 `answer accuracy + dominant F4 + any-error rate`

### 图建议

- 堆叠柱状图：不同模型的 dominant F1/F2/F3/F4 分布
- 分层图：answer correct vs answer wrong 的错误率
- 热图：模型 × 数据集 的 F4 rate

## 5. 最小可交付包

如果资源有限，先完成这套最小组合：

### 数据

- `MedCaseReasoning test`
- `MedBullets OP4/OP5`
- `MedQA`

### 模型

- 一个闭源强模型
- 一个开源强模型
- 一个医学模型

### 指标

- answer accuracy
- dominant F1/F2/F3/F4
- any-error rate
- correct-answer subset 的 F1/F2/F3/F4

### 审查

- 至少 100 条人工抽样

这已经足够支撑一条很强的论文主线：

> F1-F4 不是合成数据里才有的标签，而是多个模型在多个医学推理场景里都会自然生成的可见推理错误。

## 6. 当前建议的逐步实施顺序

1. 先跑 `MedCaseReasoning test`
2. 先只比较 3 个模型
3. 先完成主表和 100 条人工抽样
4. 再扩到 `MedBullets / MedQA`
5. 最后再加 `Med-HALT reasoning` 做 robustness

## 7. 当前仓库里已经能直接用的入口

- 主数据集准备：
  - [prepare_medcase_reasoning_eval.py](/root/med/fa_dpo_pipeline/prepare_medcase_reasoning_eval.py)

- 病例型推理生成：
  - [generate_case_reasoning_eval.py](/root/med/fa_dpo_pipeline/generate_case_reasoning_eval.py)

- 外部 MCQ 数据统一：
  - [prepare_benchmark_reasoning_eval.py](/root/med/fa_dpo_pipeline/prepare_benchmark_reasoning_eval.py)

- 外部 MCQ 推理生成：
  - [generate_mcq_reasoning_eval.py](/root/med/fa_dpo_pipeline/generate_mcq_reasoning_eval.py)

- F1-F4 自动分析：
  - [analyze_qa_faithfulness_errors.py](/root/med/fa_dpo_pipeline/analyze_qa_faithfulness_errors.py)

- 外部 MCQ 一键流水线：
  - [run_f1_f4_external_mcq_pipeline.sh](/root/med/fa_dpo_pipeline/run_f1_f4_external_mcq_pipeline.sh)

- 多数据集总汇总：
  - [aggregate_f1_f4_study_results.py](/root/med/fa_dpo_pipeline/aggregate_f1_f4_study_results.py)

## 8. 一句话决策

如果你要一个最有说服力、同时又能在当前仓库里逐步落地的组合：

- 主场：`MedCaseReasoning test`
- 外部验证：`MedBullets + MedQA + Med-HALT`
- 模型：`闭源强模型 + 开源强模型 + 医学模型`
- 主结论：`总体错误率 + 答对但推理不忠实`
