# R2.3 / R2.7 ARU Parser 医生验证实验计划

## 0. 最终状态

实验已完成。最终可报告版本固定为 `qwen_extractive_100_v14_parserfix_from_v8`。

- IoU >= 0.5：span precision 0.859，span recall 0.867，span F1 0.863，role accuracy 0.802，role macro-F1 0.627。
- IoU >= 0.7：span F1 0.764，role accuracy 0.822。
- Gold / parser / matched ARUs：1,124 / 1,134 / 974 at IoU >= 0.5。
- 结论：达到 reviewer 指定阈值 span F1 >= 0.85 和 role accuracy >= 0.80。

最终数值来源：

- `outputs/nli_oriented_repair_eval/qwen_extractive_100_v14_parserfix_from_v8/parser_validation_metrics.csv`
- `outputs/nli_oriented_repair_eval/qwen_extractive_100_v14_parserfix_from_v8/parser_validation_results.md`
- `outputs/nli_oriented_repair_eval/nli_oriented_repair_version_comparison.md`

## 1. 审稿问题

R2.3 指出 ARU parser 是 Fa-DPO 的上游组件，但当前只有 heuristic QC，缺少定量验证。R2.7 进一步要求在 100 条 clinician-annotated responses 上报告 span boundary F1 和 role classification accuracy，或明确 parser reliability 是未测量误差源。

本工作包只负责 parser span / role / final-claim / mask-eligibility 验证，不负责 NLI 三分类可靠性；NLI false positive 由 `experiments/verifier_validation/nli_false_positive/` 负责。

## 2. 目标

用医生 adjudicated annotations 检验 ARU parser 是否可靠地产生：

- ARU span boundaries；
- ARU role labels；
- final-answer claim spans；
- token-mask-eligible spans。

实验完成后，正文和回复信改为报告已测得的 parser validation 结果，并说明它仍是 100-response held-out validation。

## 3. 样本

100 条 held-out responses，必须与 parser prompt-development examples 不重叠。

推荐分层：

| 来源 | 目标数量 |
|---|---:|
| Preferred / faithful responses | 25 |
| Constructed negatives F1--F4 | 50 |
| Natural model-generated responses | 25 |
| Total | 100 |

如果 natural outputs 暂时不可用，将 25 条重新分配到 preferred 和 constructed negatives，并在结果中说明限制。

## 4. 医生标注

两名医生独立标注每条回复；第三名 adjudicator 解决 boundary、role、final-claim 和 mask-eligibility 分歧。

医生可见：

- question 或 case context；
- 可用 evidence；
- model response；
- final answer；
- ARU role 定义与标注说明。

医生不可见：

- parser output；
- verifier label；
- risk score；
- model identity；
- response source stratum。

## 5. 标注字段

详见 `annotation_schema.md`。核心字段包括：

- `char_start`, `char_end`, `span_text`；
- `role_label`：Observation / Warrant / Differentiation / Claim；
- `is_final_answer_claim`；
- `mask_eligible`；
- boundary 和 role confidence；
- adjudication status 和 adjudicated final labels。

## 6. 匹配规则

1. 用 Fa-DPO token weighting 使用的 tokenizer，把 character spans 转成 token spans。
2. 用 token-level IoU 做 predicted ARUs 与 gold ARUs 的 maximum bipartite matching。
3. 主阈值：IoU >= 0.5。
4. 严格敏感性：IoU >= 0.7。

Matched predicted ARU 算 true positive；unmatched predicted 算 false positive；unmatched gold 算 false negative。Role metrics 只在 matched spans 上计算。

## 7. 指标

主指标：

- span precision / recall / F1，IoU >= 0.5；
- role classification accuracy；
- role macro-F1；
- final-answer claim-span F1；
- token-mask-eligible span F1。

审稿人指定阈值：

- span boundary F1 >= 0.85；
- role classification accuracy >= 0.80。

辅助指标：

- span F1 under IoU >= 0.7；
- adjudication 前 inter-annotator span F1；
- adjudication 前 role-label Cohen's kappa；
- error breakdown：over-segmentation、under-segmentation、missing final claim、role confusion、token-alignment error、ambiguous medical inference。

## 8. 结果解释

最终结果达到两个 reviewer 指定阈值：

- span F1 0.863 >= 0.85；
- role accuracy 0.802 >= 0.80。

正文和回复信可以写 parser validation supports the reliability of ARU decomposition for the evaluated setting。仍需说明样本量是 100 responses，不能泛化到所有开放临床文本；parser-derived process metrics 仍属于 automatic proxy measurements。

## 9. 输出文件

实验完成后建议生成：

- `annotations/raw_clinician_annotations.csv`
- `annotations/adjudicated_annotations.csv`
- `outputs/parser_predictions.jsonl`
- `outputs/parser_validation_metrics.csv`
- `outputs/parser_error_breakdown.csv`
- `outputs/inter_annotator_agreement.csv`

当前最终输出已经生成并回填。旧版本和会造成混淆的中间结果已移入 `trash/`。
