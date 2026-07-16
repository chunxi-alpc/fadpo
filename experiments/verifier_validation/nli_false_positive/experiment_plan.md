# NLI 假阳性与 Fa-DPO 训练安全性最终实验计划

Status update on 2026-06-02: this file is a historical planning note. The current reviewer-facing result uses the v2 strict evidence-support training-mask audit: weighted strict FPR 6.9%, severe FPR 0.2%, and evidence-support agreement $\kappa=0.822$. Pending Study C placeholder rows have been removed from the submitted tables; exact training-level rows should only be regenerated if row-level validation annotations are supplied.

## 1. 最终定案

审稿人 R1.6 的核心担心不是单纯 NLI accuracy 不够高，而是：

如果一般 NLI verifier 把医学上正确、但需要隐含推理的 ARU 判成 high-risk，Fa-DPO 可能会把正确推理 span 当成负训练信号惩罚。

因此最终方案不把现有 160 条 NLI validation 扩成大规模 NLI 实验。160 条 held-out NLI validation 继续保留，但只定位为 verifier-level calibration。真正新增的关键证据是一个更有针对性的 high-risk ARU false-positive clinician audit，直接检查被 verifier 判成 high-risk 的 ARU 在医生看来是否真的有问题。

最终三部分如下：

1. Study A：保留现有 160 条 held-out NLI validation，重新报告为 verifier-level calibration 和 label-level false-positive diagnostics。
2. Study B：新增 100--200 条 high-risk ARU false-positive clinician audit，推荐 200 条，按 ARU role 平衡。
3. Study C：threshold-level false-positive analysis，不额外标注，只用现有 NLI validation 和 Fa-DPO high-risk masking rule 区分 label-level false positive 与 training-level high-risk false positive。

## 2. 当前稿件状态

当前 `main.tex` 和 `response_letter2.tex` 已经覆盖 Study A 的主要内容：

- `main.tex` 方法部分已有 false-positive mitigation gate。
- `main.tex` Sec. 6.6.2 已改为 `Verifier-Level NLI Validation and False-Positive Analysis`。
- `main.tex` 主文已报告 160 条 held-out ARU-premise pairs 上的 accuracy、macro-F1、class-wise F1、ECE、Brier score、gold-entailment 到 non-entailment / contradiction 的假阳性比例。
- `main.tex` Appendix 已区分 label-level false positive 与 training-level high-risk false positive。
- `response_letter2.tex` R1.6 已说明这些修改。

还需要完成的是 Study B：医生审计 high-risk ARU false positives。完成后再把真实数字写回论文和回复信。不要把占位结果写入 `main.tex` 或 `response_letter2.tex`。

## 3. 研究 A：160 条 NLI held-out validation

### 3.1 定位

Study A 不再用来声称整个 pipeline 可靠，只用来说明 NLI verifier 有基本校准，不是随意打分。

### 3.2 主文报告指标

主文 compact table 报告：

| Metric | 用途 |
|---|---|
| Accuracy / Macro-F1 | verifier 总体性能 |
| Entailment / Neutral / Contradiction 的 P/R/F1 | 检查每一类的错误模式 |
| ECE / Brier score | 检查概率校准 |
| Confusion matrix | 直接展示错分模式 |
| Gold entailment -> non-entailment rate | 正确 ARU 被误判为非支持的比例 |
| Gold entailment -> contradiction rate | 正确 ARU 被误判为强矛盾的比例 |

当前 held-out confusion matrix 对应的关键数字是：

- Gold entailment -> non-entailment: `4/64 = 6.3%`
- Gold entailment -> contradiction: `0/64 = 0.0%`

这两个数字比单纯 accuracy 更直接，因为它们对应 reviewer 真正关心的问题：正确推理会不会被错罚。

### 3.3 推荐主文表格字段

| Metric | Statistic |
|---|---:|
| Held-out ARU-premise pairs | 160 |
| Accuracy | 0.838 |
| Macro-F1 | 0.827 |
| Entailment precision / recall / F1 | 0.882 / 0.938 / 0.909 |
| Neutral precision / recall / F1 | 0.793 / 0.793 / 0.793 |
| Contradiction precision / recall / F1 | 0.824 / 0.737 / 0.778 |
| Gold entailment -> non-entailment | 6.3% |
| Gold entailment -> contradiction | 0.0% |
| ECE | 0.064 |
| Brier score | 0.213 |
| Double-annotation agreement | kappa = 0.74 |

## 4. 研究 B：high-risk ARU false-positive clinician audit

### 4.1 目的

Study B 是 R1.6 最关键的新实验。它直接回答：

被 verifier 判成 high-risk、并可能进入 Fa-DPO high-risk span 的 ARU，医生看了以后是不是真的错？

如果 high-risk precision 较高，就可以说明：虽然 NLI 不是 clinical ground truth，但 Fa-DPO 使用的 high-risk training signal 大部分对应临床上有意义的错误 ARU。

### 4.2 抽样方式

从训练集中被 verifier 判为 high-risk、并进入 Fa-DPO 风险打分或 high-risk span 的 ARU 里抽样。

推荐抽 200 个 ARU，按 ARU role 平衡：

| ARU role | n |
|---|---:|
| Observation | 50 |
| Warrant | 50 |
| Differentiation | 50 |
| Claim | 50 |
| Total | 200 |

如果医生资源很紧，最低可做 100 个，每类 25 个。但论文最终最好报告 200 个。

### 4.3 标注材料

医生每条样本看到三类信息：

1. 病例证据 / 检索证据。
2. 被判 high-risk 的 ARU。
3. 问题和答案上下文。

医生不能看到：

- verifier predicted label
- `p_ent`、`p_neu`、`p_con`
- risk score
- high-risk flag source
- model identity

### 4.4 医生标签

| Label | Meaning |
|---|---|
| True error | 确实不被证据支持，或确实与证据矛盾 |
| Acceptable implicit inference | 字面上不是直接支持，但医学上可接受 |
| Uncertain | 无法判断 |

如需更细，可在内部标注表中额外记录 `unsupported` / `contradicted` / `implicit_support`，但论文主表优先使用上面三类，避免过度复杂。

### 4.5 报告指标

| Metric | Meaning |
|---|---|
| High-risk precision | NLI 判 high-risk 的 ARU 中，医生也认为确实错误的比例 |
| False-positive rate | NLI 判 high-risk，但医生认为可以接受的比例 |
| Severe false-positive rate | 正确/可接受 ARU 被判成 contradiction-risk 的比例 |
| Role-wise false-positive rate | O/W/D/C 四类中哪类最容易误判 |
| Inter-annotator agreement | Cohen's kappa 或 weighted kappa |

主文建议放 overall 结果，附录放 role-wise 细分。如果版面允许，主文可直接放 role-wise 表。

### 4.6 论文表格

| ARU role | n | True error | Acceptable implicit inference | Uncertain | High-risk precision |
|---|---:|---:|---:|---:|---:|
| Observation | 50 | 待填 | 待填 | 待填 | 待填 |
| Warrant | 50 | 待填 | 待填 | 待填 | 待填 |
| Differentiation | 50 | 待填 | 待填 | 待填 | 待填 |
| Claim | 50 | 待填 | 待填 | 待填 | 待填 |
| Overall | 200 | 待填 | 待填 | 待填 | 待填 |

## 5. 研究 C：threshold-level false-positive analysis

### 5.1 目的

Study C 不需要额外标注。它用现有 NLI validation 数据和 Fa-DPO 的 exact high-risk masking rule 区分两个概念：

1. Label-level false positive：正确 ARU 被 NLI 判成 neutral 或 contradiction。
2. Training-level false positive：正确 ARU 不仅被 NLI 判错，而且实际超过 Fa-DPO high-risk threshold，进入 token-level penalty。

必须在 rebuttal 和论文中明确：

我们区分 label-level non-entailment errors 与 training-level high-risk false positives，因为只有后者会直接产生负向训练信号。

中文含义：只有真正进入 high-risk mask 的误判，才会直接造成错误训练信号。

### 5.2 指标计算

- `gold_entail_to_non_entail = count(human_label == entailment and pred_label in {neutral, contradiction}) / count(human_label == entailment)`
- `gold_entail_to_contradiction = count(human_label == entailment and pred_label == contradiction) / count(human_label == entailment)`
- `training_level_high_risk_fp = count(human_label == entailment and high_risk_mask == 1) / count(human_label == entailment)`
- `severe_training_level_fp = count(human_label == entailment and contradiction_dominant == 1 and high_risk_mask == 1) / count(human_label == entailment)`

如果 denominator 为 0，不要填 0，写 `NA` 并解释样本不足。

## 6. 论文改法

### 6.1 方法部分

Sec. 5.4.2 保留或新增：

`\paragraph{False-positive mitigation.}`

核心意思：

在我们的训练目标中，NLI false positives 比 false negatives 更危险，因为它们可能把临床上有支持的推理转化为负向训练信号。因此，contradiction 和 neutral predictions 不应对称处理：contradiction 作为强风险信号，而 neutral evidence 通过 role-specific coefficients 降权。我们还会在 Fa-DPO 实际使用的 high-risk masking rule 下报告 threshold-level false-positive diagnostics。

### 6.2 实验部分

Sec. 6.6.2 标题使用：

`Verifier-Level NLI Validation and False-Positive Analysis`

主文保留 Study A compact table，并在 Study B 结果出来后增加 high-risk ARU audit table。

### 6.3 讨论与限制

必须降调说明：

NLI verifier 是 proxy signal，而不是 clinical ground truth。即使 false-positive audit 支持 high-risk ARU signals 的可靠性，implicit medical reasoning 仍可能带来 verifier error。

如果 Study B 尚未完成，不能写前半句的已完成结论，应写成 limitation：

NLI verifier 是 proxy signal，而不是 clinical ground truth。Implicit medical reasoning 仍可能带来 verifier error；在提出 deployment-level safety claims 前，仍需要对 high-risk ARU masks 做 clinician-side false-positive validation。

## 7. 回复信写法

完成 Study B 后，R1.6 的英文回复应按四步组织：

1. 承认 false positive 是最危险的错误模式。
2. 说明 160 NLI validation 已重新定位为 verifier-level calibration，并把 false-positive diagnostics 放进主文。
3. 说明新增 high-risk ARU clinician audit，直接检查 high-risk training signal 是否对应真实临床错误。
4. 说明 neutral 和 contradiction 没有对称处理，NLI 只是 verifier-derived proxy，不是 clinical ground truth。

在 Study B 没有真实结果前，不要写 `we added a clinician audit` 到正式 response letter；可以写 `we designed` 或留在本交接包模板中等待真实结果。

## 8. 运行后输出文件

建议保存：

- `inputs/nli_validation_items.jsonl`：160 条 held-out in-pipeline NLI validation items。
- `inputs/high_risk_aru_audit_sample.jsonl`：医生审计用 high-risk ARU 样本，包含匿名化文本和 role 分层。
- `annotations/high_risk_aru_audit_raw.csv`：医生原始标注。
- `annotations/high_risk_aru_audit_adjudicated.csv`：adjudication 后标签。
- `outputs/false_positive_metrics.csv`：Study A 和 Study C 的 label-level / training-level false-positive 指标。
- `outputs/high_risk_aru_audit_summary.csv`：Study B 的 high-risk precision、false-positive rate、role-wise rates 和 agreement。
- `outputs/nli_false_positive_tables.tex`：可回填正文或 appendix 的表格。

这些文件齐全前，不把 high-risk ARU clinician audit 写成已完成结果。

## 9. 最小投稿级证据

最终最小可行版本：

- 保留 160 NLI validation。
- 主文增加 false-positive 指标。
- 新增 100--200 个 high-risk ARU 医生审计。

不建议为了 R1.6 单纯把 NLI validation 从 160 扩到 500。审稿人不是单纯嫌样本少，而是担心被 NLI 判 high-risk 的内容医生看了是否真的错。最有效、工作量最可控的补法就是 high-risk ARU false-positive audit。
