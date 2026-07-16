# R1.14 Component、Progressive 与 Interaction Ablation 计划

## 1. 审稿人问题

当前 Sec. 6.3 的 ablation 是 leave-one-out 设计。它能说明从完整 Fa-DPO pipeline 中拿掉某个组件会降低性能，但不能回答三个问题：

1. 每个组件单独加入时是否有独立正贡献；
2. 各组件的收益是可加的，还是存在耦合；
3. 类似 -2.7 pp 和 -2.5 pp 这种相近 drop 是否统计上可区分。

因此修订稿应保留原 leave-one-out 表作为 necessity diagnostic，同时补 progressive 和 interaction ablations，并用 paired uncertainty estimates 解释结果。

## 2. 当前仓库状态

相关目录：

- `analyses/statistical_ci/`：负责主表和 process/accuracy 表的 paired bootstrap 与 paired tests。
- `experiments/ablation/normalization_stability/`：负责 token-weight normalization 和 sparse high-risk mask 稳定性。
- `experiments/verifier_validation/nli_false_positive/`：负责 verifier false-positive 和医生侧验证。

本目录专门负责 R1.14 要求的 component-addition 和 interaction ablation。

## 3. 主要实验范围

Primary backbone：

- II-Medical-8B，与当前主 ablation 表保持一致。

Primary training budget：

- 与主 Fa-DPO ablation 相同的 retained train/dev split 和 seed；
- 1 epoch；
- 相同 LoRA、learning rate schedule、effective batch size、max sequence length、decoding config、answer extractor、verifier-side corpus、retriever、reranker、NLI verifier 和 evaluation scripts。

Primary evaluation：

- MedBullets-op4 accuracy；
- MedBullets-op5 accuracy；
- MedXpertQA accuracy；
- MedQA accuracy；
- 四个 setting 的 macro-average accuracy；
- MedCaseReasoning same-answer process metrics：
  - Support Rate；
  - Contradiction Rate；
  - Unsupported Medical Reasoning Rate；
  - Claim Inconsistency Rate。

Optional robustness：

- 如果算力允许，在一个 general-instruction backbone 上重复最小 component-addition matrix，优先 Qwen3-8B 或 Llama-3.1-8B-Instruct。

## 4. 实验 A：Negative Filtering x Objective Module

### 4.1 目的

直接回答 reviewer 要求的 “only negative filtering”、“only token weighting”、“only margin” 和组合 objective term。

### 4.2 运行矩阵

采用 2 x 4 设计：

| Negative pool | Standard DPO loss | + token weighting only | + margin only | + token weighting + margin |
|---|---:|---:|---:|---:|
| All QC candidates, 47,393 | A1 | A2 | A3 | A4 |
| High-risk retained, 34,204 | B1 | B2 | B3 | B4 / Full Fa-DPO |

定义：

- `All QC candidates, 47,393`：automatic construction 和 QC 后的全部 candidate preference pairs。
- `High-risk retained, 34,204`：经过 ARU-risk filtering 后的 retained pool。
- `Standard DPO loss`：vanilla sequence-level DPO，不使用 token-level rejected-side weighting，也不用 risk-adaptive margin。
- `+ token weighting only`：使用 rejected-side token reweighting mask，但 sample-level margin 设为 0。
- `+ margin only`：使用 risk-adaptive margin，但 rejected tokens 使用 uniform weighting。
- `+ token weighting + margin`：同时使用两个 objective terms；在 retained pool 上就是 Full Fa-DPO。

### 4.3 最小算力版本

如果 8-run matrix 做不完，至少跑五个 variant：

1. All QC candidates + Standard DPO loss。
2. High-risk retained + Standard DPO loss。
3. High-risk retained + token weighting only。
4. High-risk retained + margin only。
5. High-risk retained + token weighting + margin，即 Full Fa-DPO。

这个最小版本仍然能测试 negative filtering、token weighting、margin 和 combined objective。

### 4.4 估计量

Negative filtering effect：

- `B1 - A1`，比较 retained high-risk negatives 上的 Standard DPO 与 all QC candidates 上的 Standard DPO。

Token-weighting effect：

- `A2 - A1` 和 `B2 - B1`。

Margin effect：

- `A3 - A1` 和 `B3 - B1`。

Combined objective effect：

- `A4 - A1` 和 `B4 - B1`。

Token weighting x margin interaction：

- `(TW + Margin - Standard) - [(TW only - Standard) + (Margin only - Standard)]`。
- retained pool 上为 `(B4 - B1) - [(B2 - B1) + (B3 - B1)]`。

所有 accuracy 和 process metrics delta 均用 percentage points 报告。

## 5. 实验 B：Granularity x Routing Interaction

### 5.1 目的

回应 ARU-level granularity 和 node-type routing 可能存在交互的问题。

严格的 2 x 2 factorial design 不成立，因为 role-specific routing 需要 local reasoning unit 才能 route。whole-response block 没有可分配 role 的局部单元。因此用 sentence-level units 作为中间粒度。

### 5.2 运行矩阵

| Reasoning unit | Routing strategy | Run id | Attribution role |
|---|---|---|---|
| Whole-response block | Uniform routing | C1 | 无局部粒度、无 role routing |
| Sentence-level units | Uniform routing | C2 | 粗粒度局部单元，无 role routing |
| Sentence-level units | Predicted role routing | C3 | 粗粒度局部单元，有 role routing |
| ARU-level units | Uniform routing | C4 | 细粒度局部单元，无 role routing |
| ARU-level units | Role-specific routing | C5 / Full verification design | 细粒度局部单元，有 role routing |

定义：

- `Whole-response block`：把整条 reasoning chain 作为一个 block 验证，并生成 response-level risk signal。
- `Sentence-level units`：按句子切分 reasoning trace，需要时给句子分配粗粒度 role，再映射风险到 span。
- `ARU-level units`：使用论文中的 ARU parser 和 role-conditioned units。
- `Uniform routing`：所有 local units 使用同一套 verification premise strategy。
- `Predicted role routing`：sentence-level units 分配粗 role，并按 role route。
- `Role-specific routing`：完整 ARU role-conditioned routing design。

不要构造 “whole-response block + role-specific routing”，因为 whole response 不是 role-bearing local unit。

### 5.3 估计量

Granularity effect under uniform routing：

- `C4 - C2`，同为 uniform routing 时比较 ARU-level 与 sentence-level。

Routing effect at sentence level：

- `C3 - C2`。

Routing effect at ARU level：

- `C5 - C4`。

ARU granularity plus routing effect：

- `C5 - C1`。

Granularity x routing interaction：

- `(C5 - C4) - (C3 - C2)`，测试 role routing 在 ARU-level 上是否比 sentence-level 更有贡献。

## 6. Leave-One-Out 表的重新解释

保留现有 leave-one-out 表，但其作用限定为：

- 说明 full pipeline 中各组件不是冗余的；
- 不声称每个组件有独立贡献；
- 不仅凭 point-estimate drop 给组件重要性排序。

解释规则：

- 如果 -2.7 pp 和 -2.5 pp 的 drop-difference bootstrap CI 跨 0，则只能写成 magnitude comparable，不能说前者显著更重要。

## 7. 统计分析

### 7.1 相对 Full Fa-DPO 的 paired bootstrap

对每个 variant 和 metric，计算 question-level paired delta：

`delta_i = metric_variant(i) - metric_full(i)`。

设置：

- bootstrap unit：question id；
- replicates：10,000；
- seed：42；
- Full Fa-DPO 和 variant outcome 在 sampled question 内保持配对；
- 报告 mean delta 和 95% percentile CI；
- CI 不跨 0 时，才说该 variant 与 Full Fa-DPO 在该 metric 上 reliably different。

process metrics 使用 same-answer subsets 时，按 case id 做 paired cluster bootstrap，并保留该 case 下所有 ARU。不能把 ARU 当独立样本重采样。

### 7.2 比较两个 drops

不要只比较 point estimate。对每个 question：

`drop_A(i) = metric_full(i) - metric_variant_A(i)`

`drop_B(i) = metric_full(i) - metric_variant_B(i)`

`D(i) = drop_A(i) - drop_B(i)`。

对 `D(i)` 的均值做 bootstrap。如果 95% CI 跨 0，则两个 drops 不能说 statistically distinguishable。

### 7.3 多重比较

如果报告 p-value，在每个 table family 内做 Holm-Bonferroni correction：

- leave-one-out deltas against Full Fa-DPO；
- component-addition matrix deltas；
- granularity/routing interaction matrix deltas。

如果只报告 CI，说明 CI 是 descriptive paired uncertainty estimates，并谨慎解释排序。

## 8. 所需输入文件

每个 run 和 benchmark 需要 question-level prediction logs：

- run id 和 model checkpoint id；
- benchmark name；
- question id；
- gold answer；
- extracted answer；
- correctness flag；
- raw model response；
- decoding configuration hash。

process metrics 还需要：

- case id；
- response id；
- ARU parse；
- ARU role；
- verification premise id 和 text hash；
- verifier label 和 probabilities；
- per-response process metrics。

训练复现需要：

- base model id；
- preference-pair pool id；
- train/dev split id 和 seed；
- loss variant flags；
- token-weighting config；
- margin config；
- routing/granularity config；
- optimizer 和 LoRA config；
- final checkpoint id。

## 9. 报告表格

用 `result_tables_template.tex` 填真实结果。不要把 placeholder 放进最终 manuscript。

建议主文表：

1. 现有 leave-one-out 表，补 delta 和 paired CI。
2. Negative filtering x objective module 表。
3. Granularity x routing interaction 表。

建议 appendix 表：

1. 每个 run 的 per-benchmark accuracy。
2. 每个 run 的 same-answer process metrics。
3. reviewer 关心的 close drop difference CI，例如 -2.7 pp vs -2.5 pp。

## 10. 运行后输出文件

建议保存：

- `outputs/run_registry.csv`：每个 variant 的 checkpoint、训练配置、seed 和数据池。
- `outputs/benchmark_predictions.jsonl`：question-level gold、extracted answer、correctness 和 raw response。
- `outputs/process_metrics.jsonl`：case-level ARU/process metrics 和 verifier outputs。
- `outputs/ablation_summary.csv`：每个 variant 的 Avg. Accuracy、process metrics 和 delta。
- `outputs/paired_bootstrap_ci.csv`：variant-vs-full 的 paired bootstrap CI。
- `outputs/drop_difference_ci.csv`：close-drop comparison，例如 -2.7 pp vs -2.5 pp。
- `outputs/holm_adjusted_pvalues.csv`：如报告 p-value，则保存 Holm correction 后结果。

这些文件存在并检查通过后，再填写 `result_tables_template.tex` 和回复信。

## 11. 回复信解释规则

真实结果出来后：

- 说明原 leave-one-out 表保留，但重新解释为 necessity diagnostic；
- 说明哪些 component-addition effects 有 paired CI 支持；
- 说明 token weighting 和 margin 是 additive 还是 interactive；
- 说明 ARU granularity 和 role-specific routing 是否独立有效或具有协同；
- 不要在 drop-difference CI 没有排除 0 时声称一个 drop 大于另一个。
