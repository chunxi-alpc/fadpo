# R4.3 数据集偏差与覆盖分析计划

## 1. 审稿问题

R4.3 关注 34,204 个 preference pairs 来自四类 controlled failure types，而不是自然医学 LLM 错误分布的无偏采样。修订稿需要把数据构造偏差、可观测覆盖范围和剩余限制讲清楚。

本目录只负责 constructed preference dataset 本身的数据卡式分析。不重复以下工作：

- `experiments/verifier_validation/medhalt_mapped_unmapped/`：外部 Med-HALT mapped/unmapped 成对比较；
- `experiments/verifier_validation/natural_output_shift/`：自然输出上的 verifier distribution shift；
- `experiments/clinician_audits/preference_label_reliability/`：positive label 和 pairwise preference label 医生审计；
- `experiments/ablation/answer_shortcut_counterfactual/`：answer-changing shortcut 诊断。

## 2. 需要报告的偏差来源

| 偏差来源 | 管线来源 | 主要风险 | 必要分析或限制 |
|---|---|---|---|
| Source-dataset bias | 所有样本来自 MedCaseReasoning training cases | 专科、疾病、病例风格和题型可能不代表所有临床 QA | 有 metadata 时报告分布；没有则明确缺失 |
| Taxonomy bias | negatives 只覆盖 F1--F4 | F1--F4 之外自然错误不足 | 交叉引用 Med-HALT mapped/unmapped，避免完整 taxonomy claim |
| Construction bias | negatives 是局部 controlled rewrites | 可能产生 artifact，区别于 model-sampled errors | 使用临床 plausibility audit；计划 model-sampled negative check |
| Answer-preservation bias | F1--F3 保留 final answer | 过度代表 correct-answer wrong-reasoning 样本 | 报告 F1--F3 policy 和 F4 answer-changing count；交叉引用 shortcut 诊断 |
| Risk-filtering bias | DPO set 保留 high-risk candidates | retained pool 被高风险局部错误富集 | 报告 raw/QC/retained counts 和 token-mask density |
| Verifier-induced bias | parser/NLI 生成 risk labels 和 masks | label noise 影响 retention 和 token weighting | 交叉引用 parser/NLI validation 和 high-risk FP audit |
| Positive-sample bias | 默认 preferred responses 可靠 | imperfect positives 会污染 DPO labels | 交叉引用 preferred-response 和 pair-level audit |

## 3. 当前本地可报告统计

本地 `data/train_fadpo.jsonl` 和 `data/eval_fadpo.jsonl` 已包含 34,204 个 retained preference pairs，可直接计算 retained-pool distribution、unique source count、chosen/rejected length、token-mask density 和 mean margin。

当前可用汇总：

| Type | Retained pairs | Unique sources | Mean chosen words | Mean rejected words | Mean mask tokens | Mean high-weight tokens | High-weight token share | Mean margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | 6,715 | 6,715 | 155.3 | 191.9 | 330.1 | 80.9 | 24.5% | 1.56 |
| F2 | 9,607 | 9,607 | 156.1 | 198.3 | 340.8 | 87.6 | 25.7% | 1.61 |
| F3 | 9,456 | 9,456 | 157.2 | 195.7 | 337.2 | 88.3 | 26.2% | 1.60 |
| F4 | 8,426 | 8,426 | 155.9 | 196.9 | 338.3 | 84.7 | 25.0% | 1.58 |
| Overall | 34,204 | 12,411 | 156.2 | 196.0 | 337.1 | 85.8 | 25.4% | 1.59 |

High-weight token 指 released `topo_mask` 大于低背景值 0.1 的 rejected tokens。它是 token-mask 诊断，不等同于 raw ARU count。长度统计按当前 released `train_fadpo.jsonl` 和 `eval_fadpo.jsonl` 中 `chosen` / `rejected` 字段的空白分词计算。

## 4. 仍需完整日志的统计

以下字段不能从当前 retained JSONL 可靠恢复，必须等 construction logs 或 metadata export：

- per-type terminal construction logs beyond the design-implied 13,092 attempts per type；
- per-type QC candidate count；
- retained vs discarded risk-score distribution；
- high-risk ARU count；
- specialty、disease group、answer-option、question-format metadata；
- retained vs discarded length/risk comparison；
- natural model-sampled negative comparison。

如果日志不可用，不要把这些项写成 measured results。可以报告 design-implied per-type raw attempts、aggregate raw/QC/retained counts、retained-pool statistics 和 retained ARU counts，并把完整 data-card export 作为后续 artifact。

## 5. 完整分析协议

### 5.1 Source distribution

输入：

- MedCaseReasoning source metadata；
- retained-pair `source_id`；
- split assignment；
- specialty / disease / question format metadata（如果有）。

指标：

- retained source positives by specialty/disease；
- retained pairs by specialty/disease；
- retained-pair count per source positive；
- answer-option distribution；
- preferred/rejected response length distribution；
- source cases retained vs not retained。

不要用未经验证的 free-text classifier 推断 specialty，除非明确标为 heuristic。

### 5.2 Taxonomy 与 retention

主表：

| Failure type | Raw attempts | After QC | Retained | QC rate | Retention among QC | Retention among raw | Mean ARUs | Mean high-risk ARUs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | 13,092 | TBD | 6,715 | TBD | TBD | 51.3% | 20.46 | TBD |
| F2 | 13,092 | TBD | 9,607 | TBD | TBD | 73.4% | 20.98 | TBD |
| F3 | 13,092 | TBD | 9,456 | TBD | TBD | 72.2% | 20.70 | TBD |
| F4 | 13,092 | TBD | 8,426 | TBD | TBD | 64.4% | 20.87 | TBD |
| Overall | 52,368 | 47,393 | 34,204 | 90.5% | 72.2% | 65.3% | 20.77 | TBD |

Retained counts、design-implied per-type raw attempts、overall raw/QC/retained counts、以及 retained ARU counts 当前可直接写；per-type QC、discarded-pool risk distribution 和 high-risk ARU counts 等需要完整 construction 或 score logs 后再填。

### 5.3 Construction-artifact check

目标：检查 local rewritten negatives 是否含有 unnatural artifacts，使 DPO 区分任务比自然错误更容易。

推荐样本：

- 100--200 retained constructed negatives，F1--F4 均衡；
- 100--200 base model 或 Standard DPO-Retained-34K 的 model-sampled negatives。

盲审字段：

- response naturalness；
- clinical plausibility；
- dominant error category；
- 是否映射到 F1--F4；
- 是否像 local rewrite artifact；
- 是否适合作为 DPO rejected response。

### 5.4 外部与标签可靠性

不要在本目录重复下列实验，只交叉引用：

- preferred-response 和 pair-level validity：`experiments/clinician_audits/preference_label_reliability/`；
- Med-HALT mapped/unmapped：`experiments/verifier_validation/medhalt_mapped_unmapped/`；
- answer-changing shortcut：`experiments/ablation/answer_shortcut_counterfactual/`；
- parser/NLI validation：`experiments/verifier_validation/aru_parser_validation/` 和 `experiments/verifier_validation/nli_false_positive/`。

## 6. 论文回填

正文：

- 在 training preference dataset 后加入 `Dataset Bias and Coverage Analysis`；
- 加 bias-source table；
- 加 retained-pool distribution table；
- 明确哪些 per-type QC/ARU/specialty 统计还需要完整日志。

局限性：

- Fa-DPO 针对 controlled evidence-use 和 reasoning-answer consistency failures；
- 不声称覆盖全部自然临床推理错误；
- 交叉引用 Med-HALT coverage、preference-label audit、parser/NLI validation 和 natural-output stress test。

## 7. 输出文件

建议保存：

- `outputs/retained_pool_distribution.csv`：当前 retained JSONL 可直接计算的 F1--F4 分布、长度、mask density 和 margin。
- `outputs/source_distribution.csv`：如 metadata 可用，保存 specialty / disease / question format 分布。
- `outputs/source_reuse_summary.csv`：当前 source positive 被 1--4 个 retained failure types 复用的分布。
- `outputs/taxonomy_retention_table.csv`：raw、QC、retained、QC rate、retention rate、ARU count 等；缺失项保留为 `NA`，不要用猜测值。
- `outputs/answer_policy_summary.csv`：F1--F4 answer-preservation / answer-changing 构造检查。
- `outputs/aru_role_distribution.csv`：retained ARU role counts；不等同于 high-risk ARU count。
- `outputs/construction_artifact_audit_sample.jsonl`：constructed negatives 与 model-sampled negatives 的盲审样本。
- `outputs/construction_artifact_audit_results.csv`：自然度、临床合理性、artifact rate、F1--F4 mapped rate。
- `outputs/dataset_bias_summary.md`：最终写入正文和回复信的结论边界。

如果只能生成 `retained_pool_distribution.csv`，则正文只能报告 retained-pool statistics，不能声称完成完整 data-card。

## 8. 允许与禁止的表述

允许：

- 数据集是 controlled、interpretable，适合 risk-aware preference optimization；
- retained pool 按设计富集 high-risk process failures；
- Med-HALT 显示 F1--F4 外部可见但不完整。

避免：

- 数据集是自然临床推理错误的无偏样本；
- F1--F4 是完整 taxonomy；
- local rewrites 等同 model-sampled errors；
- preferred responses 在审计前都完全 faithful；
- matched Med-HALT 完成前声称 Fa-DPO 改善 unmapped external errors。
