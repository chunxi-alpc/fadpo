# Fa-DPO Experiment Implementation Plan

这份文档面向“拿到实验环境后直接写代码”的执行场景。目标不是重复论文表述，而是把还需要补的实验整理成可实现、可交付、可回填论文的任务清单。

## 0. 使用说明

- 单文件夹运行约定：
  - 现在默认只上传 `fa_dpo_pipeline/`
  - 所有命令都从执行根目录运行
  - 现成数据默认放在执行根目录的 `data/`
  - 这里提到的脚本都位于 `fa_dpo_pipeline/`
  - 文中出现的 `outputs/...`、`result/...`、`artifacts/...` 都是相对于执行根目录
- `脚本位置`
  - 主流水线与分析脚本：`fa_dpo_pipeline/`
  - review 工具：`fa_dpo_pipeline/review/`
  - 训练脚本：`fa_dpo_pipeline/trainers/`

- `状态`
  - `待跑`：还没有真实结果，需要在实验环境中实现并运行。
  - `需人工`：必须有人参与标注、核查或填写元数据。
  - `可自动`：主要靠脚本即可完成。
  - `可选增强`：不是最小可提交包，但能明显增强说服力。
- `论文回填位置`
  - 主文显著性占位：`tab:significance_placeholder`
  - 主文 process-faithfulness 占位：`tab:process_metrics_placeholder`
  - Retrieval appendix：`app:retrieval_protocol`
  - NLI verifier validation：`app:nli_validation`
  - Natural error coverage：`app:natural_error_coverage`
  - Overlap / fairness：`app:data_fairness`
  - Cost / scalability：`app:cost_scalability`

## 1. 当前建议的执行顺序

1. `P0.1` NLI verifier 人工校验
2. `P0.2` Retrieval pipeline 真实配置与 `hit@K`
3. `P0.3` Same-answer process-faithfulness metrics
4. `P0.4` GPT blind judge 的 CI 与 protocol 落实
5. `P0.5` Overlap / contamination / fairness
6. `P0.6` Preprocessing cost / scalability
7. `P0.7` Same-backbone 多 seed 显著性
8. `P1.1` Hyperparameter sensitivity
9. `P1.2` Retrieval robustness / noisy evidence
10. `P1.3` Natural error coverage
11. `P2.1` OOD / unseen evaluation
12. `P2.2` Multi-backbone validation

## 2. 最小可提交实验包

这部分是最应该优先完成的。默认都要做。

### P0.1 NLI verifier 人工校验

- `目标`
  - 直接验证 DeBERTa-v3 verifier 在医学 `ARU-evidence` 对上的三分类可靠性。
- `状态`
  - `待跑` + `需人工`
- `为什么优先`
  - 现有 human alignment 只能说明 chain-level proxy 有用，不能说明 verifier 本身判得准。
- `输入`
  - 已经生成好的 ARU
  - ARU 对应 routed evidence
  - verifier 输出标签和概率
- `建议样本量`
  - 最小 `100`
  - 推荐 `150-200`
  - 尽量覆盖 `O / W / D / C` 四类 ARU
- `需要人工做的事`
  - 人工标注每条 `ARU-evidence` 为 `entail / neutral / contradict`
  - 至少抽一小部分双标，便于看一致性
- `你需要写的代码`
  - `sample_nli_validation.py`
    - 从已验证样本中抽取 `ARU-evidence` 对
    - 按 ARU role 分层采样
    - 导出标注文件
  - `eval_nli_validation.py`
    - 读取人工标注
    - 计算 `accuracy / macro-F1 / class-wise F1`
    - 导出 confusion matrix
    - 抽取典型错误案例
- `建议输出文件`
  - `outputs/nli_validation/sample.jsonl`
  - `outputs/nli_validation/annotation.csv`
  - `outputs/nli_validation/metrics.json`
  - `outputs/nli_validation/confusion_matrix.csv`
  - `outputs/nli_validation/error_cases.md`
- `论文回填`
  - `tab:nli_validation_template`
  - `tab:nli_confusion_template`
  - `app:nli_validation`

### P0.2 Retrieval pipeline 真实配置与 hit@K

- `目标`
  - 把 verifier-side retrieval pipeline 从“写作模板”变成真实可复现协议。
- `状态`
  - `待跑`
- `输入`
  - verifier corpus
  - retriever / reranker / index 配置
  - 一批 verifier query 或 ARU-evidence target
- `你需要写的代码`
  - `build_retrieval_metadata.py`
    - 统计 corpus source、document count、chunk count、avg chunk length
    - 导出 retriever / embedding / reranker / top-K 配置
  - `eval_retrieval_quality.py`
    - 计算 `Hit@5 / Hit@20`
    - 计算 support coverage
    - 分路由统计 `W/D` 和 `Observation fallback`
- `建议输出文件`
  - `outputs/retrieval_protocol/corpus_stats.json`
  - `outputs/retrieval_protocol/pipeline_config.json`
  - `outputs/retrieval_protocol/retrieval_quality.csv`
- `论文回填`
  - `tab:retrieval_protocol_template`
  - `tab:retrieval_quality_placeholder`
  - `app:retrieval_protocol`

### P0.3 Same-answer process-faithfulness metrics

- `目标`
  - 用真实 verifier 导出的 step-level 指标替换主文 placeholder。
- `状态`
  - `待跑`
- `输入`
  - Fa-DPO 与 Standard DPO 在评测集上的推理输出
  - 正确答案
  - verifier 的 ARU 和 NLI 结果
- `核心 subset`
  - 两个模型都答对 final answer
  - 只在这个 same-answer hard subset 上比较 process quality
- `你需要写的代码`
  - `build_same_answer_subset.py`
    - 取两个模型输出的交集
    - 只保留 final answer 都正确的样本
    - 可加复杂度过滤
  - `eval_process_metrics.py`
    - 计算 `support ratio`
    - 计算 `contradiction rate`
    - 计算 `unsupported warrant ratio`
    - 计算 `claim inconsistency rate`
- `建议输出文件`
  - `outputs/process_metrics/same_answer_subset.jsonl`
  - `outputs/process_metrics/metrics.csv`
  - `outputs/process_metrics/per_example_metrics.jsonl`
- `论文回填`
  - `tab:process_metrics_placeholder`
  - `app:process_metrics`

### P0.4 GPT blind judge 的 CI 与 protocol 落实

- `目标`
  - 让 Figure 4 不只报 point estimate，而是有可复查的 blind setup 和置信区间。
- `状态`
  - `待跑`
- `输入`
  - 现有 pairwise judged outputs 或待重新 judge 的样本对
- `你需要写的代码`
  - `run_pairwise_judge.py`
    - 随机交换 A/B 顺序
    - 记录 judge model snapshot
    - 输出 win/tie/loss
  - `bootstrap_judge_ci.py`
    - 对 question-level outcomes 做 bootstrap
    - 输出每个 criterion 的 95% CI
- `建议输出文件`
  - `outputs/gpt_judge/judgments.jsonl`
  - `outputs/gpt_judge/win_rates.csv`
  - `outputs/gpt_judge/bootstrap_ci.json`
  - `outputs/gpt_judge/prompt.txt`
- `论文回填`
  - Figure 4 对应分析段
  - `app:gpt4_protocol`

### P0.5 Overlap / contamination / fairness

- `目标`
  - 说明训练偏好数据与评测 benchmark 是否存在 overlap，保证横向比较口径稳。
- `状态`
  - `待跑`
- `输入`
  - MedCaseReasoning train
  - MedQA / MedBullets / MedXpertQA evaluation data
- `你需要写的代码`
  - `check_overlap_exact.py`
    - question stem + answer exact match / normalized match
  - `check_overlap_neardup.py`
    - lexical 或 embedding 检索近重复
    - 导出人工复核候选
- `建议输出文件`
  - `outputs/data_fairness/exact_overlap.csv`
  - `outputs/data_fairness/near_duplicate_candidates.jsonl`
  - `outputs/data_fairness/near_duplicate_summary.csv`
- `论文回填`
  - `tab:overlap_template`
  - `app:data_fairness`

### P0.6 Preprocessing cost / scalability

- `目标`
  - 把离线构造成本写清楚，而不只写 DPO 训练 GPU。
- `状态`
  - `待跑`
- `输入`
  - negative generation logs
  - ARU parsing logs
  - retrieval logs
  - NLI verification logs
  - DPO training logs
- `你需要写的代码`
  - `aggregate_costs.py`
    - 统计每阶段 wall-clock
    - 统计每条样本平均 ARU 数
    - 统计 retrieval / NLI 调用次数
    - 汇总硬件信息
- `建议输出文件`
  - `outputs/cost_scalability/cost_summary.csv`
  - `outputs/cost_scalability/hardware.json`
  - `outputs/cost_scalability/per_stage_stats.json`
- `论文回填`
  - `tab:cost_template`
  - `app:cost_scalability`

### P0.7 Same-backbone 多 seed 显著性

- `目标`
  - 用真实多 seed 结果替换主文显著性占位表。
- `状态`
  - `待跑`
- `输入`
  - Standard DPO 多 seed 结果
  - Fa-DPO 多 seed 结果
- `建议设置`
  - 最小 `3 seeds`
  - 推荐 `5 seeds`
- `你需要写的代码`
  - `aggregate_multiseed_results.py`
    - 汇总每个 setting 的 mean/std
    - 做 paired t-test
- `建议输出文件`
  - `outputs/significance/multiseed_results.csv`
  - `outputs/significance/ttest_summary.csv`
- `论文回填`
  - `tab:significance_placeholder`

## 3. 强烈建议补的分析实验

### P1.1 Hyperparameter sensitivity

- `目标`
  - 证明 Fa-DPO 不是靠偶然超参命中。
- `状态`
  - `待跑`
- `优先扫的参数`
  - `alpha`
  - `lambda`
  - `top-K`
  - 其次 `beta`
  - 其次 `mu`
- `建议设计`
  - one-factor-at-a-time
  - 同时报 `accuracy` + 至少一个 process metric
- `你需要写的代码`
  - `run_sensitivity_sweep.py`
  - `plot_sensitivity.py`
- `建议输出文件`
  - `outputs/sensitivity/results.csv`
  - `outputs/sensitivity/plots.pdf`

### P1.2 Retrieval robustness / noisy evidence

- `目标`
  - 回应 reviewer 对 retrieval miss / noisy evidence / conflicting evidence 的担心。
- `状态`
  - `待跑`
- `建议条件`
  - 降低 `top-K`
  - 注入噪声 snippets
  - 使用更弱 retriever
  - 删除关键 support evidence
- `你需要写的代码`
  - `run_retrieval_stress_test.py`
  - `plot_retrieval_robustness.py`
- `建议输出文件`
  - `outputs/retrieval_robustness/results.csv`
  - `outputs/retrieval_robustness/plots.pdf`

### P1.3 Natural error coverage

- `目标`
  - 量化 F1--F4 对真实错误分布的覆盖率。
- `状态`
  - `待跑` + `需人工`
- `输入`
  - held-out Standard DPO 或 backbone baseline 的真实错误推理
- `需要人工做的事`
  - 给 sampled natural errors 标 dominant type
  - 标是否能映射到 `F1/F2/F3/F4/Other`
- `你需要写的代码`
  - `sample_natural_errors.py`
  - `eval_natural_error_coverage.py`
- `建议输出文件`
  - `outputs/natural_error/sample.jsonl`
  - `outputs/natural_error/annotation.csv`
  - `outputs/natural_error/coverage_summary.csv`
- `论文回填`
  - `tab:natural_error_template`
  - `app:natural_error_coverage`

## 4. 可选增强实验

### P2.1 OOD / unseen evaluation

- `目标`
  - 回应“当前更像 in-distribution”的质疑。
- `状态`
  - `可选增强`
- `优先方案`
  - 新 benchmark
  - 不同子领域
  - 不同来源的 reasoning-heavy medical QA
- `建议输出文件`
  - `outputs/ood_eval/results.csv`

### P2.2 Multi-backbone validation

- `目标`
  - 回应方法是否 backbone-specific。
- `状态`
  - `可选增强`
- `最低可接受形式`
  - 选 1 个新 backbone
  - 同时跑 `Standard DPO vs Fa-DPO`
- `建议输出文件`
  - `outputs/multibackbone/results.csv`

### P2.3 Clinician pairwise process eval

- `目标`
  - 不只依赖 GPT judge，再补一个 clinician 小样本 pairwise。
- `状态`
  - `可选增强` + `需人工`
- `建议输出文件`
  - `outputs/clinician_pairwise/annotation.csv`
  - `outputs/clinician_pairwise/summary.csv`

## 5. 人工参与任务总表

### 必须人工

1. `NLI verifier` 三分类标注
2. natural error coverage 的类型标注
3. overlap near-duplicate 候选的少量人工复核
4. retrieval corpus 的真实来源、规模、license、index 配置确认
5. cost/scalability 中硬件与运行时信息的最终确认

### 如果有资源，建议人工

1. clinician pairwise process evaluation
2. human alignment 的 agreement 补算与说明

## 6. 建议的输出目录结构

```text
outputs/
  nli_validation/
  retrieval_protocol/
  process_metrics/
  gpt_judge/
  data_fairness/
  cost_scalability/
  significance/
  sensitivity/
  retrieval_robustness/
  natural_error/
  ood_eval/
  multibackbone/
  clinician_pairwise/
```

## 7. 最低优先级可跳过项

如果时间非常紧，优先保证：

1. `P0.1` NLI verifier 人工校验
2. `P0.2` Retrieval 真实配置与 `hit@K`
3. `P0.3` Same-answer process metrics
4. `P0.4` GPT judge 的 CI
5. `P0.5` Overlap / fairness
6. `P0.6` Cost / scalability

可以延后：

1. `P1.1` 超参数敏感性
2. `P1.2` Retrieval robustness
3. `P1.3` Natural error coverage
4. `P2.1` OOD
5. `P2.2` Multi-backbone
6. `P2.3` Clinician pairwise

## 8. 完成定义

一个实验任务只有在满足下面三点时，才算真正完成：

1. 有真实输出文件，而不是只在论文里留 protocol 或 placeholder。
2. 有可复查的脚本入口，能从原始输入重跑到最终表格。
3. 已经把结果回填到对应论文位置，而不是只存在实验目录里。
