# NLI 假阳性实验交接包

这个文件夹用于完成 Reviewer R1.6 的补充实验与论文/回复信更新。最终方案已经收敛为：

**保留 160 条 NLI held-out validation 作为 verifier-level calibration；保留 high-risk ARU 医生审计作为 high-risk precision evidence；新增 strict evidence-support training-mask 审计，用来估计本应被 evidence 支持的 ARU 被 Fa-DPO 训练 mask 错罚的比例。**

注意：临床上合理但没有被当前 `question_context` 或 `verification_premise` 支持的 Warrant，不能算作 strict gold-entailment false positive，应单独报告为 clinically acceptable but evidence-unsupported。

## 当前论文和回复信已经写了什么

已经写入：

- `response_letter2.tex` 中已有 R1.6 “NLI Verifier Reliability”的正式回复，说明验证重点从总体 accuracy 转向 false-positive analysis，并报告当前 held-out NLI split 的关键数字。
- `main.tex` 方法部分已经加入 false-positive mitigation gate：strong token-level mask 只有在超过风险阈值、不是 entailment-confident、且不是 close-label ambiguity 时才生效。
- `main.tex` Sec. 6.6.2 已写入 `Verifier-Level NLI Validation and False-Positive Analysis`，报告 160 条 held-out ARU-premise pairs 上的 accuracy、macro-F1、class-wise F1、ECE、Brier score、gold-entailment 到 non-entailment / contradiction 的假阳性比例。
- `main.tex` Appendix 已区分 label-level false positive 和 training-level high-risk false positive。

已经完成：

- high-risk ARU clinician audit。这个实验直接回答“被 verifier 判成 high-risk 的 ARU，医生看了是否真的错”。
- 最终采用 `outputs_score06_full200_20260530_round2_final/` 和同步后的 `outputs/high_risk_aru_audit_*` 结果：200 条 role-balanced high-risk ARU，high-risk precision 76.5\%，clinician false-positive rate 16.0\%，severe false-positive rate 3.5\%，Cohen's $\kappa=0.891$。
- 解释口径固定为：NLI verifier 是 in-pipeline proxy signal，不是 clinical ground truth；医生审计用于验证 high-risk training emphasis 的临床合理性。

## 文件说明

- `experiment_plan.md`：历史实验计划，保留执行脉络；当前 reviewer-facing 结果以本 README、`main.tex`、`response_letter2.tex` 和 `outputs/*_summary.csv` 为准。
- `annotation_templates.md`：给医生或医学背景标注者使用的标注字段、标签含义和质控规则。执行时可按最终三标签体系简化。
- `result_tables_template.tex`：当前 reviewer-facing 表格片段，已移除 placeholder 和 pending Study C 行。
- `response_update_template.md`：医生审计完成后，如何把结果更新到回复信。不要在没有真实结果时直接粘贴进正式回复信。
- `scripts/build_high_risk_aru_audit_sample.py`：从 stage-06 score 和 ARU-evidence 文件中抽取 verifier-flagged high-risk ARU，导出 full/internal 与 blind 标注表。
- `scripts/analyze_false_positive_metrics.py`：汇总 Study A label-level false-positive 指标；只有在提供 row-level validation annotation 时才额外输出 training-level 指标。
- `scripts/analyze_high_risk_aru_audit.py`：医生 adjudication 回来后计算 Study B high-risk precision、false-positive rate、severe false-positive rate 和 role-wise 分解。
- `scripts/build_training_mask_fp_audit_sample.py`：构建 role- and mask-balanced training-mask 审计样本。
- `scripts/analyze_training_mask_fp_evidence_support_audit.py`：分析 v2 strict evidence-support 标注，计算 reviewer-facing training-mask false-positive 指标。

## 当前已生成的本地产物

Study A NLI validation sample-preparation package:

- `inputs/nli_validation_items.jsonl`：200 条行级 NLI validation 样本，含 verifier prediction、概率、risk score 和 split；内部分析使用。
- `inputs/nli_validation_internal.csv`：内部核对表，含 verifier prediction/probability，不能给 blinded 标注者。
- `inputs/nli_validation_annotation_template.csv`：给标注者使用的 blind CSV，只填写 `human_label` 和 `notes`。
- `inputs/nli_validation_manifest.jsonl`：compact manifest。
- `inputs/nli_validation_sampling_quota.json`：route / label / subgroup 抽样配额记录。
- `inputs/nli_validation_sampling_summary.md`：抽样统计；当前 calibration/evaluation split 为 40/160。

最终 high-risk ARU 审计包采用 v2 round2 final，其他中间版本已经移入 `trash/nli_false_positive_nonfinal_20260531_123653/`。

- `inputs_score06_full200_20260530_round2_final/high_risk_aru_audit_sample.jsonl`：200 条内部审计样本，Observation/Warrant/Differentiation/Claim 各 50 条。
- `inputs_score06_full200_20260530_round2_final/high_risk_aru_audit_internal.csv`：含 verifier label/probability/risk 的内部表，不能给医生。
- `inputs_score06_full200_20260530_round2_final/high_risk_aru_audit_blind.csv`：给医生标注的 blind CSV。
- `annotations_score06_full200_20260530_round2_final/high_risk_aru_audit_raw_round2.csv`：两名医生的原始独立标注。
- `annotations_score06_full200_20260530_round2_final/high_risk_aru_audit_adjudicated_round2.csv`：仲裁后的最终标签。
- `inputs_score06_full200_20260530_round2_final/high_risk_aru_audit_sampling_summary.md`：抽样统计。
- `outputs/false_positive_metrics.csv`：Study A 已报告的 label-level 指标，以及 v2 strict evidence-support training-mask audit 指标。
- `outputs/nli_false_positive_tables.tex`：已生成 Study A 表、confusion matrix 和 v2 strict evidence-support training-mask audit 表；不再包含 pending Study C 行。
- `outputs/high_risk_aru_audit_summary.csv`：Study B 最终规范结果，已同步 v2 round2 final。旧版 `high_risk_aru_audit_report.md`、`high_risk_aru_audit_tables.tex` 和 `response_update_filled.md` 曾残留错误口径，已移入 `trash/nli_high_risk_conflicting_outputs_20260531/`，不要用于论文或回复信。
- `annotations_training_mask_fp_sample60_20260602/training_mask_fp_audit_raw_template_v2_evidence_support.csv`：strict evidence-support 复核模板。
- `annotations_training_mask_fp_sample60_20260602/training_mask_fp_audit_adjudicated_template_v2_evidence_support.csv`：strict evidence-support 仲裁模板。
- `annotations_training_mask_fp_sample60_20260602/training_mask_fp_audit_raw_v2_evidence_support.csv`：strict evidence-support 双医生原始标注。
- `annotations_training_mask_fp_sample60_20260602/training_mask_fp_audit_adjudicated_v2_evidence_support.csv`：strict evidence-support 仲裁结果。
- `outputs/training_mask_fp_evidence_support_summary.csv`：strict evidence-support training-mask FPR 汇总。当前 reviewer-facing 主结果为 6.9\% strict FPR、0.2\% severe FPR、$\kappa=0.822$。
- `outputs/historical_clinical_acceptability_audit_20260602/`：旧 single-axis clinical-acceptability audit 结果，仅作历史记录，不作为 strict evidence-support FPR 引用。

## High-risk clinician audit 执行状态

已经完成的部分：

- 从训练 pipeline 的 stage-06 verifier scores 中抽取 verifier-flagged high-risk ARU。
- 按 Observation / Warrant / Differentiation / Claim 分层抽样。
- 生成内部表 `inputs/high_risk_aru_audit_internal.csv`，保留 verifier label、概率、risk score、flag source，只供内部核对。
- 生成盲化表 `inputs/high_risk_aru_audit_blind.csv`，已去掉 verifier 分数和模型预测标签。
- 生成双标注者 long-form 原始模板 `annotations/high_risk_aru_audit_raw_template.csv`，默认每条 ARU 两行，`annotator_id` 为 `clinician_1` / `clinician_2`。
- 生成仲裁模板 `annotations/high_risk_aru_audit_adjudicated_template.csv`，用于回填最终 `true_error` / `acceptable_implicit_inference` / `uncertain` 标签。
- 分析脚本已经支持 high-risk precision、false-positive rate、severe false-positive rate、role-wise FPR、exact agreement、Cohen's kappa 和 linear weighted kappa。

已完成医生标注、冲突仲裁和分析脚本汇总。正式论文和回复信应使用 `outputs/high_risk_aru_audit_summary.csv` 中的 v2 round2 final 数值。

## 运行命令

从仓库根目录运行：

```bash
python experiments/verifier_validation/nli_false_positive/scripts/build_study_a_nli_sample.py
python experiments/verifier_validation/nli_false_positive/scripts/build_high_risk_aru_audit_sample.py
python experiments/verifier_validation/nli_false_positive/scripts/analyze_false_positive_metrics.py
python experiments/verifier_validation/nli_false_positive/scripts/analyze_high_risk_aru_audit.py
```

如果换成最终冻结的完整训练 score 文件，应显式传入：

```bash
python experiments/verifier_validation/nli_false_positive/scripts/build_high_risk_aru_audit_sample.py \
  --score-file /path/to/faithfulness_scores.jsonl \
  --evidence-file /path/to/arus.with_evidence.jsonl \
  --negatives-file /path/to/negatives.jsonl \
  --sample-size 200 \
  --per-role 50
```

当前默认 `--gate-mode risk_threshold` 与仓库中 released stage-07 mask builder 的 `score > 0.1` 行为一致。若使用论文 Eq. `high_risk_gate` 的 entailment-confidence 和 top-label-gap 条件，在冻结 `tau_ent` 与 `tau_conf` 后改用：

```bash
python experiments/verifier_validation/nli_false_positive/scripts/build_high_risk_aru_audit_sample.py \
  --gate-mode paper_gate \
  --entailment-confidence-threshold 0.5 \
  --top-label-gap-threshold 0.1
```

## 最小可完成版本

1. 保留现有 160 条 NLI held-out validation，并在主文报告 false-positive 指标。
2. 抽样 100--200 个 verifier-flagged high-risk ARU，推荐 200 个，Observation / Warrant / Differentiation / Claim 各 50 个。
3. 医生 blind annotation：`True error`、`Acceptable implicit inference`、`Uncertain`。
4. 报告 high-risk precision、false-positive rate、severe false-positive rate、role-wise false-positive rate 和 Cohen's kappa / weighted kappa。
5. 更新 `main.tex`、`response_letter2.tex` 和最终表格时只使用真实审计结果。

## 不建议做的事

- 不建议为了 R1.6 单纯把 NLI validation 从 160 扩到 500。Reviewer 的关键担心是 high-risk training signal 是否会错罚正确医学推理，而不是普通 NLI held-out 样本数。
- 不要直接把 `PLACEHOLDER` 写进 `main.tex` 或 `response_letter2.tex`。
- 不要让医生看到 verifier 预测分数后再标注。医生标注必须 blind。
- 不要声称 NLI verifier 是 clinical ground truth。只能写成 verifier-derived proxy signal。

## 最终交付物

完成实验后，请至少交付：

- `inputs/high_risk_aru_audit_sample.jsonl`：Study B 要标注的 high-risk ARU 样本。
- `annotations/high_risk_aru_audit_raw.csv`：医生原始标注。
- `annotations/high_risk_aru_audit_adjudicated.csv`：仲裁后的最终标注。
- `outputs/high_risk_aru_audit_summary.csv`：医生审计指标汇总。
- `outputs/false_positive_metrics.csv`：label-level false-positive 指标，以及 clinician-audited strict evidence-support mask-level false-positive 指标。
- `outputs/nli_false_positive_tables.tex`：Study A 的 LaTeX 表格。
- `outputs/high_risk_aru_audit_summary.csv`：Study B 医生审计指标汇总；正式论文和回复信以此文件以及 `main.tex` 中的 76.5\% / 16.0\% / 3.5\% 口径为准。
