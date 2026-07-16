# ARU 解析器验证工作包

本目录对应 R2.3 和 R2.7，用于执行并记录 100-response held-out ARU parser 医生验证。它只负责 parser span boundary、role label、final-answer claim span 和 token-mask eligibility；NLI verifier 可靠性由 `experiments/verifier_validation/nli_false_positive/` 负责。

## 最终口径

最终可报告版本是 `qwen_extractive_100_v14_parserfix_from_v8`。

- 主阈值 IoU >= 0.5：span precision 0.859，span recall 0.867，span F1 0.863，role accuracy 0.802，role macro-F1 0.627。
- 严格敏感性 IoU >= 0.7：span F1 0.764，role accuracy 0.822。
- 数据规模：100 responses，1,124 adjudicated gold ARUs，1,134 parser ARUs，974 matched ARUs at IoU >= 0.5。
- 结论：通过 reviewer 指定的 span F1 >= 0.85 和 role accuracy >= 0.80 阈值。

## 文件

| 文件 | 作用 |
|---|---|
| `experiment_plan.md` | 中文实验计划：样本、标注、matching、指标和回填规则。 |
| `annotation_schema.md` | 医生 boundary / role 标注字段。 |
| `result_tables_template.tex` | 已填入最终数值的正文/回复信表格片段。 |
| `response_update_template.md` | 已填入最终数值的回复信更新文本。 |

## 最终结果来源

- `outputs/nli_oriented_repair_eval/qwen_extractive_100_v14_parserfix_from_v8/parser_validation_metrics.csv`
- `outputs/nli_oriented_repair_eval/qwen_extractive_100_v14_parserfix_from_v8/parser_validation_results.md`
- `outputs/nli_oriented_repair_eval/qwen_extractive_100_v14_parserfix_from_v8/parser_error_breakdown.csv`
- `outputs/final_parser_validation_summary.md`
- `outputs/final_parser_validation_metrics.csv`
