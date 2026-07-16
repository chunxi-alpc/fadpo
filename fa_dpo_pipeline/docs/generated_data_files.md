# Fa-DPO Pipeline 生成数据文件与字段说明

本文档根据 `data/fa_dpo_pipeline` 下的 README、主流程脚本和现有样例数据整理。这里的“表头”对 CSV 指真实 header；对 JSON/JSONL 指顶层字段及关键嵌套字段。默认运行时，代码目录在 `fa_dpo_pipeline/`，生成数据通常写到执行根目录下的 `result/`、`artifacts/`、`outputs/`、`checkpoints/`，而不是直接写回 `data/fa_dpo_pipeline/`。

## 1. 默认输出目录

| 目录 | 含义 |
|---|---|
| `result/fa_dpo_pipeline/` | 主流水线的中间结果，如负样本、ARU、证据、faithfulness 分数。 |
| `artifacts/fa_dpo_pipeline/` | 检索语料、索引、训练集、运行元数据等可复用产物。 |
| `outputs/` | 分析实验、复核抽样、评估汇总等辅助输出。 |
| `checkpoints/fa_dpo_pipeline/` | 训练得到的模型 checkpoint。 |

如果设置了 `FA_DPO_RUNTIME_ROOT`、`FA_DPO_RESULT_ROOT`、`FA_DPO_ARTIFACT_ROOT`、`FA_DPO_OUTPUT_ROOT` 或 `FA_DPO_CHECKPOINT_ROOT`，默认路径会相应改变。

## 2. 主流水线产物总表

| 阶段 | 文件名/路径 | 类型 | 中间/最终 | 说明 |
|---|---|---|---|---|
| 00 | `artifacts/fa_dpo_pipeline/miriad_corpus.jsonl` | JSONL | 中间 | 从 MIRIAD 导出的检索语料。 |
| 01 | `artifacts/fa_dpo_pipeline/miriad_faiss.index` | FAISS binary | 中间 | MedCPT 文档向量索引，无表头。 |
| 01 | `artifacts/fa_dpo_pipeline/miriad_medcpt.offsets.npy` | NumPy array | 中间 | 语料 JSONL 行偏移数组，无表头。 |
| 02 | `result/fa_dpo_pipeline/medcase_unfaithful_negatives.jsonl` | JSONL | 中间 | 通过 QC 的受控负样本。 |
| 02 | `result/fa_dpo_pipeline/medcase_unfaithful_negatives.attempts.jsonl` | JSONL | 中间日志 | 所有终态/尝试记录。 |
| 02 | `result/fa_dpo_pipeline/medcase_unfaithful_negatives.review.jsonl` | JSONL | 中间日志 | 需要人工复核的候选负样本。 |
| 02 | `result/fa_dpo_pipeline/medcase_unfaithful_negatives.rejected.jsonl` | JSONL | 中间日志 | 被启发式 QC 或 judge 拒绝的候选。 |
| 02 | `result/fa_dpo_pipeline/medcase_unfaithful_negatives.infeasible.jsonl` | JSONL | 中间日志 | 计划阶段判定不可构造的样本。 |
| 02 | `result/fa_dpo_pipeline/medcase_unfaithful_negatives.failures.jsonl` | JSONL | 中间日志 | 非临时性生成失败记录。 |
| 02 | `result/fa_dpo_pipeline/medcase_unfaithful_negatives.progress.json` | JSON | 中间日志 | 断点续跑进度快照。 |
| 03 | `result/fa_dpo_pipeline/medcase_unfaithful_negatives.strict_clean.jsonl` | JSONL | 中间 | 严格清洗后的负样本。 |
| 03 | `result/fa_dpo_pipeline/medcase_unfaithful_negatives.review_sample.jsonl` | JSONL | 中间复核 | 分层抽样的复核样本，保留完整上下文。 |
| 03 | `result/fa_dpo_pipeline/medcase_unfaithful_negatives.review_sample.csv` | CSV | 中间复核 | 给标注者使用的扁平复核表。 |
| 03 | `result/fa_dpo_pipeline/medcase_unfaithful_negatives.review_summary.json` | JSON | 中间汇总 | strict-clean 与复核抽样统计。 |
| 04 | `result/fa_dpo_pipeline/medcase_unfaithful_negatives.strict_clean.arus.jsonl` | JSONL | 中间 | 负推理分解后的 ARU 数据。 |
| 05 | `result/fa_dpo_pipeline/medcase_unfaithful_negatives.strict_clean.arus.with_evidence.jsonl` | JSONL | 中间 | 给 W/D ARU 绑定检索证据后的数据。 |
| 06 | `result/fa_dpo_pipeline/medcase_unfaithful_negatives.strict_clean.scores.jsonl` | JSONL | 中间/训练输入 | ARU 级风险与样本级 margin。 |
| 06 | `artifacts/fa_dpo_pipeline/reports/06_score_faithfulness_preflight.md` | Markdown | 中间报告 | Stage 05 到 Stage 06 的一致性和证据覆盖检查。 |
| 07 | `artifacts/fa_dpo_pipeline/topodpo_data/train_dpo.jsonl` | JSONL | 最终训练数据 | 标准 DPO 训练 split。 |
| 07 | `artifacts/fa_dpo_pipeline/topodpo_data/eval_dpo.jsonl` | JSONL | 最终训练数据 | 标准 DPO eval split。 |
| 07 | `artifacts/fa_dpo_pipeline/topodpo_data/train_margindpo.jsonl` | JSONL | 最终训练数据 | 带 margin 的 DPO 训练 split。 |
| 07 | `artifacts/fa_dpo_pipeline/topodpo_data/eval_margindpo.jsonl` | JSONL | 最终训练数据 | 带 margin 的 DPO eval split。 |
| 07 | `artifacts/fa_dpo_pipeline/topodpo_data/train_fadpo.jsonl` | JSONL | 最终训练数据 | 带 `topo_mask` 和 `margin` 的 Fa-DPO 训练 split。 |
| 07 | `artifacts/fa_dpo_pipeline/topodpo_data/eval_fadpo.jsonl` | JSONL | 最终训练数据 | 带 `topo_mask` 和 `margin` 的 Fa-DPO eval split。 |
| 07 | `artifacts/fa_dpo_pipeline/topodpo_data/train_topodpo.jsonl` | JSONL | 最终训练数据 | `train_fadpo.jsonl` 的兼容命名。 |
| 07 | `artifacts/fa_dpo_pipeline/topodpo_data/eval_topodpo.jsonl` | JSONL | 最终训练数据 | `eval_fadpo.jsonl` 的兼容命名。 |
| 08 | `checkpoints/fa_dpo_pipeline/topodpo/` | checkpoint dir | 最终模型 | 训练输出目录，不是表格数据。 |
| all | `artifacts/fa_dpo_pipeline/run_metadata/*.json` | JSON | 元数据 | 每个阶段的运行参数、输入输出、统计和硬件信息。 |

在 `qwen-next-stage2` 等自定义实验里，路径前缀可能变为 `result/qwen-next-stage2/` 或 `artifacts/qwen-next-stage2/`，但文件后缀和字段结构相同。

## 3. `miriad_corpus.jsonl`

| 字段 | 含义 |
|---|---|
| `_id` | 原始 MIRIAD 样本的 `qa_id`，用于检索语料文档标识。 |
| `title` | 论文标题，来自 `paper_title`。 |
| `text` | 检索正文，通常为 `title + [SEP] + passage_text`。 |
| `metadata` | 文档元信息对象。 |

`metadata` 字段：

| 字段 | 含义 |
|---|---|
| `dataset_name` | Hugging Face 数据集名，默认 `miriad/miriad-5.8M`。 |
| `dataset_split` | 数据 split，默认 `train`。 |
| `specialty` | 医学专科。 |
| `year` | 论文年份。 |
| `paper_url` | 原始论文链接。 |

## 4. 负样本 JSONL：accepted、attempts、review、rejected、infeasible、failures、strict_clean

这些文件大多共享同一套字段；区别主要在 `status` 和写入位置。

| 字段 | 含义 |
|---|---|
| `id` | 样本唯一 ID，通常为 `source_id::F1/F2/F3/F4`。 |
| `dataset_name` | 原始数据集名。 |
| `split` | 原始数据 split。 |
| `source_id` | 原始样本 ID。 |
| `error_symbol` | 目标错误类型符号：`F1`、`F2`、`F3`、`F4`。 |
| `error_type` | 错误类型文本名，如 `evidence_omission`。 |
| `interface` | 错误发生接口，如 `(X,G)->R` 或 `R->Y`。 |
| `status` | 当前记录终态，如 `accepted_without_judge`、`accepted_by_judge`、`needs_human_review`、`rejected_by_heuristic_qc`、`rejected_by_judge`、`plan_infeasible`、`generation_failed`。 |
| `patient_context` | 病例上下文 X。 |
| `retrieved_evidence` | 正样本侧可选外部证据 G；若未启用证据字段则为空。 |
| `prompt` | 训练/评估 prompt，当前等同于 `patient_context`。 |
| `reasoning` | 原始正确推理 R+。 |
| `chosen` | 兼容 DPO 命名的正推理，当前等同于 `reasoning`。 |
| `correct_reasoning` | 正确推理别名。 |
| `final_answer` | 原始正确答案 Y。 |
| `correct_answer_text` | 正确答案别名。 |
| `negative_reasoning` | 生成的负推理 R-。 |
| `rejected` | 兼容 DPO 命名的负推理，当前等同于 `negative_reasoning`。 |
| `predicted_reasoning` | 负推理别名。 |
| `negative_final_answer` | 与负推理配对的最终答案。 |
| `predicted_answer_text` | 负答案别名。 |
| `answer_preserved` | 实际负答案是否与原答案一致。 |
| `expected_answer_preserved` | 生成计划期望是否保留答案。 |
| `neg_type` | 错误类型文本名别名。 |
| `type` | 错误类型文本名别名。 |
| `generation_plan` | 计划阶段输出。 |
| `modified_spans` | 被改写的关键片段列表。 |
| `rewrite_target_span` | 计划或模型定位的原始待替换片段。 |
| `rewrite_replacement_span` | 替换后的片段。 |
| `rewrite_self_check` | 生成模型对局部改写的自检说明。 |
| `edit_summary` | 简短改写摘要列表。 |
| `heuristic_qc` | 启发式质量检查结果。 |
| `llm_qc` | 可选 LLM judge 质量检查结果。 |
| `error_message` | 失败记录中的错误信息；成功记录通常为空。 |

`generation_plan` 字段：

| 字段 | 含义 |
|---|---|
| `feasible` | 是否可构造该类型负样本。 |
| `preserve_answer` | 是否计划保留最终答案。 |
| `target_span` | 计划修改的原推理片段。 |
| `rationale` | 计划说明。 |
| `confidence` | 计划置信度。 |

`heuristic_qc` 字段：

| 字段 | 含义 |
|---|---|
| `similarity` | 原推理与负推理的序列相似度。 |
| `length_ratio` | 负推理 token 数 / 原推理 token 数。 |
| `novelty_ratio` | 负推理中新 token 的比例。 |
| `topic_overlap` | 负推理与病例上下文的主题重叠度。 |
| `same_answer` | 负答案是否与原答案归一化后相同。 |
| `checks` | 各启发式检查项的布尔结果。 |
| `failed_checks` | 未通过的检查项列表。 |

`llm_qc` 字段：

| 字段 | 含义 |
|---|---|
| `realized_error_type` | judge 识别出的实际错误类型。 |
| `target_type_matched` | 是否匹配目标错误类型。 |
| `single_dominant_error` | 是否只有一个主导错误。 |
| `answer_preserved_judged` | judge 判断答案是否被保留。 |
| `clinical_plausibility_score` | 表面临床合理性评分，1-5。 |
| `faithfulness_drop_score` | faithfulness 下降程度评分，1-5。 |
| `off_topic` | 是否跑题。 |
| `decision` | `accept`、`review` 或 `reject`。 |
| `reason` | judge 的简短理由。 |

## 5. `medcase_unfaithful_negatives.progress.json`

| 字段 | 含义 |
|---|---|
| `output_file` | 主输出文件绝对路径。 |
| `phase` | 当前阶段：`starting`、`running`、`interrupted`、`completed`。 |
| `total_planned` | 计划处理的样本-错误类型总数。 |
| `completed_terminal` | 已到终态的样本-错误类型数量。 |
| `written_this_run` | 本次运行写入 accepted 的数量。 |
| `reviewed_this_run` | 本次运行写入 review 的数量。 |
| `rejected_this_run` | 本次运行拒绝数量。 |
| `infeasible_this_run` | 本次运行判定不可构造数量。 |
| `failed_this_run` | 本次运行失败数量。 |
| `skipped_this_run` | 本次运行因断点续跑跳过数量。 |
| `current_sample_index` | 当前处理样本序号。 |
| `current_sample_id` | 当前处理原始样本 ID。 |
| `current_error_type` | 当前处理错误类型。 |
| `updated_at` | 进度更新时间。 |

## 6. Stage 03 复核和 summary 文件

### `review_sample.jsonl`

在负样本通用字段基础上额外包含：

| 字段 | 含义 |
|---|---|
| `_risk_tags` | 内部风险标签，如跨标签复用、过高相似度等。 |
| `_risk_score` | 内部风险分数。 |
| `_sampling_bucket` | 复核抽样桶。 |
| `review_assignment` | `single_annotate` 或 `double_annotate`。 |
| `review_round` | 复核轮次，默认 `round1`。 |
| `reviewer1_target_type_matched` | 标注者 1 判断是否匹配目标类型。 |
| `reviewer1_single_dominant_error` | 标注者 1 判断是否单一主导错误。 |
| `reviewer1_answer_preserved` | 标注者 1 判断答案是否保留。 |
| `reviewer1_clinical_plausibility` | 标注者 1 临床合理性评分。 |
| `reviewer1_faithfulness_drop` | 标注者 1 faithfulness 下降评分。 |
| `reviewer1_notes` | 标注者 1 备注。 |
| `reviewer2_*` | 双标时第二位标注者的同类字段。 |

### `review_sample.csv`

| 表头 | 含义 |
|---|---|
| `id` | 样本唯一 ID。 |
| `source_id` | 原始样本 ID。 |
| `error_symbol` | 目标错误类型。 |
| `error_type` | 错误类型文本名。 |
| `review_assignment` | 单标或双标任务。 |
| `sampling_bucket` | 抽样桶。 |
| `risk_tags` | JSON 字符串形式风险标签。 |
| `risk_score` | 风险分数。 |
| `expected_answer_preserved` | 计划是否保留答案。 |
| `answer_preserved` | 实际是否保留答案。 |
| `heuristic_similarity` | 原/负推理相似度。 |
| `heuristic_length_ratio` | 长度比例。 |
| `heuristic_novelty_ratio` | 新 token 比例。 |
| `heuristic_topic_overlap` | 主题重叠度。 |
| `patient_context` | 病例上下文。 |
| `final_answer` | 原正确答案。 |
| `negative_final_answer` | 负样本答案。 |
| `reasoning` | 原正确推理。 |
| `negative_reasoning` | 负推理。 |
| `modified_spans` | JSON 字符串形式修改片段。 |
| `edit_summary` | JSON 字符串形式修改摘要。 |
| `reviewer1_*`、`reviewer2_*` | 人工标注填写列。 |

### `review_summary.json`

| 字段 | 含义 |
|---|---|
| `accepted_total` | 输入 accepted 总数。 |
| `strict_clean_total` | strict-clean 保留数量。 |
| `strict_clean_removed` | strict-clean 移除数量。 |
| `strict_clean_removed_pct` | 移除比例。 |
| `accepted_unique_sources` | accepted 中唯一原始样本数。 |
| `strict_clean_unique_sources` | strict-clean 中唯一原始样本数。 |
| `accepted_by_error` | accepted 按错误类型计数。 |
| `strict_clean_by_error` | strict-clean 按错误类型计数。 |
| `risk_tag_counts` | 各风险标签计数。 |
| `strict_drop_buckets` | 被 strict-clean 删除的桶计数。 |
| `review_sample_total` | 复核样本总数。 |
| `review_sample_by_error` | 复核样本按错误类型计数。 |
| `review_sample_by_assignment` | 单标/双标计数。 |
| `review_sample_by_bucket` | 复核样本按桶计数。 |

## 7. ARU 分解文件：`*.arus.jsonl`

| 字段 | 含义 |
|---|---|
| `id` | 样本唯一 ID。 |
| `source_id` | 原始样本 ID。 |
| `error_symbol` | 错误类型。 |
| `patient_context` | 病例上下文。 |
| `negative_reasoning` | 被分解的负推理。 |
| `arus` | ARU 节点列表。 |
| `raw_llm_output` | 分解模型的原始输出。 |
| `sanitization` | 可选，后处理脚本写入的清洗统计。 |

`arus[]` 字段：

| 字段 | 含义 |
|---|---|
| `text` | 原子推理单元文本；当前与 `span_text` 保持一致，保留用于兼容旧代码。 |
| `span_text` | 原文抽取 span，必须等于 `negative_reasoning[char_start:char_end]`。这是人工评测和 span boundary F1 使用的边界。 |
| `nli_claim_text` | 给 NLI/verifier 使用的完整命题，可在不改变含义的前提下改写 `span_text`。不要用它计算原文 span boundary。 |
| `char_start` | `span_text` 在 `negative_reasoning` 中的起始字符位置，0-based，包含。 |
| `char_end` | `span_text` 在 `negative_reasoning` 中的结束字符位置，0-based，不包含。 |
| `type` | ARU 类型：`O` observation、`W` warrant、`D` differentiation、`C` claim。 |
| `role_label` | `type` 的可读标签：`Observation`、`Warrant`、`Differentiation`、`Claim`。 |
| `is_final_answer_claim` | 是否为明确表达最终答案/最终诊断的 span。 |
| `mask_eligible` | 如果该 ARU 被判定不忠实，是否应该进入 mask/惩罚候选；默认 W/D/C 为 true，直接 O 为 false。 |
| `retrieval_query` | W/D 节点用于医学文献检索的去情境化 query；O/C 通常为空。 |
| `query_quality` | 可选，query 后处理质量标签。 |
| `query_needs_review` | 可选，query 是否需要人工复查。 |

## 8. 证据绑定文件：`*.arus.with_evidence.jsonl`

顶层字段继承 `*.arus.jsonl`。`arus[]` 在 Stage 05 后增加：

| 字段 | 含义 |
|---|---|
| `retrieval_query` | 最终用于检索/重排的主 query。 |
| `rerank_query` | 给 cross-encoder reranker 使用的 query。 |
| `retrieval_query_variants` | 为同一 W/D 节点生成的 query 变体列表。 |
| `retrieval_query_source` | query 来源，通常为 `llm` 或 `heuristic`。 |
| `retrieval_candidates` | 检索并重排后的候选文档列表。 |
| `retrieved_evidence` | 保留给 verifier 的证据文本列表，默认取 top-k 文档正文。 |

`retrieval_candidates[]` 字段：

| 字段 | 含义 |
|---|---|
| `doc_id` | FAISS 内部文档序号。 |
| `corpus_id` | 语料中的 `_id`。 |
| `title` | 文档标题。 |
| `text` | 文档正文。 |
| `metadata` | 语料元信息。 |
| `retriever_score` | 初始向量检索分数。 |
| `retriever_rank` | 在某个 query variant 下的初始检索排名。 |
| `matched_queries` | 命中过该文档的 query variants。 |
| `matched_query_count` | 命中该文档的 query variant 数量。 |
| `best_retriever_score` | 多 variant 合并后的最佳检索分数。 |
| `best_retriever_rank` | 多 variant 合并后的最佳检索排名。 |
| `merged_retriever_rank` | 合并候选后的排序。 |
| `reranker_score` | cross-encoder 重排分数。 |
| `reranker_rank` | cross-encoder 重排排名。 |

## 9. Faithfulness 分数文件：`*.scores.jsonl`

| 字段 | 含义 |
|---|---|
| `id` | 样本唯一 ID。 |
| `source_id` | 原始样本 ID。 |
| `error_symbol` | 错误类型。 |
| `unfaithfulness_score` | 样本级不忠实风险分数，范围裁剪到 0-1。 |
| `dpo_margin` | 从风险分数映射得到的 DPO margin。 |
| `verifier` | NLI verifier 配置。 |
| `metrics` | 节点级和样本级详细指标。 |

`verifier` 字段：

| 字段 | 含义 |
|---|---|
| `model` | 使用的 NLI 模型。 |
| `entailment_threshold` | 支持判定阈值。 |
| `contradiction_threshold` | 冲突判定阈值。 |

`metrics` 字段：

| 字段 | 含义 |
|---|---|
| `max_risk` | 所有 ARU 节点中的最大风险。 |
| `avg_defect` | ARU 节点风险平均值。 |
| `coverage_ratio` | 病例上下文句子被观察节点覆盖的比例。 |
| `penalty_score` | 覆盖不足惩罚。 |
| `covered_sentences` | 已覆盖上下文句子数。 |
| `total_sentences` | 上下文句子总数。 |
| `has_prevalence_excuse` | 是否出现流行病学/罕见性解释。 |
| `target_coverage` | 本样本采用的目标覆盖率。 |
| `trace` | 精简节点打分轨迹。 |
| `node_details` | 节点级完整 verifier 明细。 |

`trace[]` 字段：

| 字段 | 含义 |
|---|---|
| `id` | ARU 节点序号。 |
| `t` | ARU 类型。 |
| `s` | 节点风险分数。 |
| `p` | 打分路径/原因标签。 |

`node_details[]` 关键字段：

| 字段 | 含义 |
|---|---|
| `id`、`type`、`text` | ARU 序号、类型和文本。 |
| `route` | 实际使用的 verifier 路由，如 `patient_context`、`retrieved_evidence`、`reasoning_history`、`patient_context_veto`。 |
| `primary_route` | 主路由。 |
| `premise_type` | 主 premise 类型。 |
| `auxiliary_route` | 可选辅助路由。 |
| `auxiliary_premise_type` | 辅助 premise 类型。 |
| `pair_count` | 本节点运行的 NLI pair 数。 |
| `entailment`、`neutral`、`contradiction` | 主证据 NLI 概率。 |
| `aggregate_entailment`、`aggregate_neutral`、`aggregate_contradiction` | 多证据聚合概率。 |
| `decision_label` | 阈值化后的判定标签。 |
| `argmax_label` | 概率最大标签。 |
| `supported` | 是否判定为支持。 |
| `contradictory` | 是否判定为冲突。 |
| `auxiliary_entailment`、`auxiliary_neutral`、`auxiliary_contradiction` | 辅助证据 NLI 概率。 |
| `auxiliary_decision_label` | 辅助证据判定标签。 |
| `auxiliary_supported` | 辅助证据是否支持。 |
| `score` | 节点风险分数。 |
| `tag` | 节点打分原因标签。 |
| `covered_context_sentence` | 是否覆盖病例上下文句子。 |
| `covered_context_sentence_index` | 被覆盖的上下文句子序号。 |
| `selected_premise` | 选中的主 premise 预览。 |
| `selected_premise_entailment`、`selected_premise_neutral`、`selected_premise_contradiction` | 选中 premise 的 NLI 概率。 |
| `auxiliary_premise` | 辅助 premise 预览。 |

## 10. Topo-DPO / Fa-DPO 训练数据

### `train_dpo.jsonl` / `eval_dpo.jsonl`

| 字段 | 含义 |
|---|---|
| `id` | 样本唯一 ID。 |
| `source_id` | 原始样本 ID。 |
| `error_symbol` | 错误类型。 |
| `prompt` | 训练 prompt，即病例上下文。 |
| `chosen` | 偏好样本中的正确响应，包含推理和 `Final Answer`。 |
| `rejected` | 偏好样本中的负响应，通常由 ARU 重建推理加负答案组成。 |

### `train_margindpo.jsonl` / `eval_margindpo.jsonl`

比 DPO 文件多：

| 字段 | 含义 |
|---|---|
| `margin` | 样本级 DPO margin，来自 `*.scores.jsonl` 的 `dpo_margin`。 |

### `train_fadpo.jsonl`、`eval_fadpo.jsonl`、`train_topodpo.jsonl`、`eval_topodpo.jsonl`

比 marginDPO 文件多：

| 字段 | 含义 |
|---|---|
| `topo_mask` | 与 `rejected` token 对齐的权重列表；高风险 ARU token 权重更高，低风险/背景 token 使用较低权重。 |

`train_topodpo.jsonl` 与 `train_fadpo.jsonl` 内容相同；`eval_topodpo.jsonl` 与 `eval_fadpo.jsonl` 内容相同，是兼容旧训练入口的命名。

## 11. 运行元数据：`run_metadata/*.json`

每个主阶段会写一个元数据 JSON，例如 `00_build_miriad_corpus.json`、`05_attach_aru_evidence.json`、`06_score_faithfulness.json`。

| 字段 | 含义 |
|---|---|
| `stage_name` | 阶段名。 |
| `status` | 运行状态，通常为 `completed`，预检异常时可能为 `blocked`、`failed_preflight` 等。 |
| `started_at` | UTC 开始时间。 |
| `finished_at` | UTC 结束时间。 |
| `elapsed_seconds` | 耗时秒数。 |
| `args` | 命令行参数快照。 |
| `inputs` | 本阶段输入路径或输入标识。 |
| `outputs` | 本阶段输出路径。 |
| `stats` | 阶段统计，如行数、ARU 数、索引规模、NLI pair 数等。 |
| `hardware` | 主机、Python、CPU/GPU 信息。 |

## 12. Score06 人工复核包

脚本 `scripts/build_score06_manual_review_pack.py` 会从完整 Stage 06 分数、Stage 05 证据和原负样本中抽取高风险、中间段、低风险样本。现有样例位于 `data/fa_dpo_pipeline/score06_sample60_3/`。

| 文件名 | 类型 | 含义 |
|---|---|---|
| `sample_manifest.jsonl` | JSONL | 抽样清单。 |
| `sample_manifest.csv` | CSV | 与 JSONL 同内容的表格版清单。 |
| `sampled_ids.txt` | text | 按清单顺序排列的样本 ID。 |
| `faithfulness_scores.sample60.jsonl` | JSONL | 被抽中的 Stage 06 分数子集，字段同 `*.scores.jsonl`。 |
| `arus_with_evidence.sample60.jsonl` | JSONL | 被抽中的 Stage 05 证据子集，字段同 `*.arus.with_evidence.jsonl`。 |
| `negatives.sample60.jsonl` | JSONL | 被抽中的负样本子集，字段同负样本 JSONL。 |
| `arus.sample60.jsonl` | JSONL | 可选，被抽中的 Stage 04 ARU 子集，字段同 `*.arus.jsonl`。 |
| `review_pack_summary.md` | Markdown | 复核包生成摘要、桶计数和缺失 ID 检查。 |

`sample_manifest.csv/jsonl` 表头：

| 表头 | 含义 |
|---|---|
| `review_bucket` | 抽样桶：`high_risk`、`mid_band`、`low_risk`。 |
| `bucket_rank` | 桶内排序序号。 |
| `id` | 样本唯一 ID。 |
| `source_id` | 原始样本 ID。 |
| `error_symbol` | 错误类型。 |
| `unfaithfulness_score` | 样本风险分数。 |
| `dpo_margin` | DPO margin。 |
| `max_risk` | 最大 ARU 风险。 |
| `avg_defect` | 平均 ARU 风险。 |
| `coverage_ratio` | 上下文覆盖率。 |
| `penalty_score` | 覆盖惩罚分。 |

## 13. 检索复核与检索质量辅助产物

### `outputs/retrieval_review/sample.jsonl`

| 字段 | 含义 |
|---|---|
| `sample_id` | 复核样本 ID，格式为 `row_id::aruN`。 |
| `id` | 原样本 ID。 |
| `source_id` | 原始样本 ID。 |
| `error_symbol` | 错误类型。 |
| `row_index` | 输入 JSONL 行序号。 |
| `aru_id` | ARU 节点序号。 |
| `aru_role` | ARU 类型，通常抽样 W/D。 |
| `bucket` | 复核桶，如 `W_with_evidence`、`D_no_evidence`。 |
| `patient_context` | 截断后的病例上下文。 |
| `aru_text` | 截断后的 ARU 文本。 |
| `retrieval_query` | 截断后的主检索 query。 |
| `retrieval_query_source` | query 来源。 |
| `retrieval_query_variants` | 截断后的 query variants。 |
| `retrieved_evidence_count` | 原始 retained evidence 数量。 |
| `retrieval_candidate_count` | 原始候选文档数量。 |
| `retrieved_evidence` | 截断后的 retained evidence 列表。 |
| `retrieval_candidates` | 截断后的候选文档列表。 |

### `outputs/retrieval_review/sample.csv`

| 表头 | 含义 |
|---|---|
| `sample_id`、`id`、`source_id`、`error_symbol` | 样本标识信息。 |
| `aru_id`、`aru_role`、`bucket` | ARU 和抽样桶信息。 |
| `retrieved_evidence_count`、`retrieval_candidate_count` | 证据和候选数。 |
| `aru_text`、`retrieval_query`、`retrieval_query_source`、`retrieval_query_variants` | ARU 与检索 query 信息。 |
| `patient_context` | 病例上下文预览。 |
| `retrieved_evidence_preview` | retained evidence 预览，多个证据用 `||` 拼接。 |
| `retrieval_candidates_preview` | 候选文档预览。 |
| `review_label`、`review_notes` | 人工复核填写列。 |

### `outputs/retrieval_review/summary.json`

| 字段 | 含义 |
|---|---|
| `aru_file` | 输入 Stage 05 文件。 |
| `roles` | 抽样的 ARU 类型。 |
| `total_candidates` | 可抽样节点总数。 |
| `requested_samples` | 请求抽样数。 |
| `sampled_rows` | 实际抽样数。 |
| `bucket_counts` | 各桶候选数。 |
| `bucket_quotas` | 各桶目标抽样数。 |
| `sampled_bucket_counts` | 各桶实际抽样数。 |
| `seed` | 抽样随机种子。 |
| `max_evidence_per_node`、`max_candidates_per_node`、`max_context_chars`、`max_text_chars` | 截断和保留参数。 |
| `invalid_json_lines`、`invalid_json_line_numbers` | 输入坏行统计。 |

### `outputs/retrieval_protocol/retrieval_quality.csv`

| 表头 | 含义 |
|---|---|
| `evaluation_mode` | `gold` 或 `silver`。 |
| `route` | verifier 路由或 `overall`。 |
| `n` | 参与汇总的节点数。 |
| `hit_at_5` | gold 模式下 top-5 命中率。 |
| `hit_at_20` | gold 模式下 top-20 命中率。 |
| `support_coverage` | silver 模式下被 verifier 判为支持的比例。 |
| `avg_candidate_count` | silver 模式下平均候选数。 |

## 14. NLI verifier 人工验证产物

### `outputs/nli_validation/sample.jsonl`

| 字段 | 含义 |
|---|---|
| `sample_id` | `row_id::aruN`。 |
| `id` | 原样本 ID。 |
| `source_id` | 原始样本 ID。 |
| `error_symbol` | 错误类型。 |
| `aru_id` | ARU 序号。 |
| `aru_role` | ARU 类型。 |
| `route_bucket` | 抽样路由桶：observation、warrant、differentiation、claim。 |
| `aru_text` | ARU 文本。 |
| `verifier_route` | Stage 06 实际路由。 |
| `evidence_text` | 被 verifier 使用或 fallback 的证据文本。 |
| `patient_context` | 病例上下文。 |
| `predicted_label` | verifier 预测标签。 |
| `argmax_label` | 概率最大标签。 |
| `entailment`、`neutral`、`contradiction` | NLI 概率。 |
| `predicted_confidence` | 最大概率。 |
| `confidence_band`、`predicted_confidence_band` | 置信度分桶。 |
| `difficulty_band` | 标注难度分桶。 |
| `is_terminal_claim` | 是否终止 claim。 |
| `score` | Stage 06 节点风险分数。 |
| `calibration_split` | `calibration` 或 `evaluation`。 |

### `outputs/nli_validation/annotation.csv`

| 表头 | 含义 |
|---|---|
| `sample_id`、`id`、`source_id`、`error_symbol` | 样本标识。 |
| `aru_id`、`aru_role`、`route_bucket`、`verifier_route` | ARU 与路由信息。 |
| `aru_text`、`evidence_text` | 待标注的 hypothesis 和 premise。 |
| `predicted_label`、`predicted_confidence`、`confidence_band`、`predicted_confidence_band`、`difficulty_band` | verifier 预测和抽样分层信息。 |
| `entailment`、`neutral`、`contradiction` | NLI 概率。 |
| `is_terminal_claim` | 是否终止 claim。 |
| `calibration_split` | 校准/评估 split。 |
| `human_label` | 人工标注标签填写列。 |
| `notes` | 人工备注。 |

### `eval_nli_validation.py` 输出

| 文件 | 主要字段/表头 | 含义 |
|---|---|---|
| `metrics.json` | `n`、`accuracy`、`macro_f1`、`class_wise_f1`、`ece`、`brier` 等 | NLI 标注验证总体指标。 |
| `calibration_metrics.json` | `n`、`n_with_probs`、`ece_bins`、`ece`、`brier`、`avg_predicted_confidence` | 置信度校准指标。 |
| `confusion_matrix.csv` | `gold_label`、`entail`、`neutral`、`contradict` | 总体混淆矩阵。 |
| `route_breakdown.csv` | `route`、`n`、`accuracy`、`macro_f1`、`ece`、`brier`、`entail_f1`、`neutral_f1`、`contradict_f1` | 按路由分组指标。 |
| `route_confusion_matrix.csv` | `route`、`gold_label`、`predicted_label`、`count` | 长表格式路由混淆矩阵。 |
| `reliability_curve.csv` | `bin_index`、`lower_bound`、`upper_bound`、`count`、`avg_confidence`、`accuracy`、`gap` | 可靠性曲线分桶。 |
| `error_cases.md` | Markdown | 人工标签与预测不一致的案例摘要。 |

## 15. 过程质量、成本、显著性和 pairwise judge 辅助产物

### `outputs/process_metrics/same_answer_subset.jsonl`

| 字段 | 含义 |
|---|---|
| `id` | 两个系统共有样本 ID。 |
| `prompt` | 病例上下文。 |
| `gold_answer` | 参考答案。 |
| `left_model_id`、`right_model_id` | 两个系统名称。 |
| `left_reasoning`、`right_reasoning` | 两个系统推理。 |
| `left_final_answer`、`right_final_answer` | 两个系统最终答案。 |
| `left_correct`、`right_correct` | 是否都答对。 |
| `left_reasoning_chars`、`right_reasoning_chars` | 推理字符数。 |

### `outputs/process_metrics/per_example_metrics.jsonl`

| 字段 | 含义 |
|---|---|
| `id` | 样本 ID。 |
| `model_id` | 系统名称。 |
| `total_nodes` | verifier 节点数。 |
| `support_ratio` | 被支持节点比例。 |
| `contradiction_rate` | 冲突节点比例。 |
| `unsupported_warrant_ratio` | W 节点中未被支持比例。 |
| `claim_inconsistency_rate` | C 节点中冲突比例。 |

### `outputs/process_metrics/metrics.csv`

| 表头 | 含义 |
|---|---|
| `model_id` | 系统名称。 |
| `n` | 样本数。 |
| `support_ratio_mean`、`support_ratio_std` | 支持比例均值/标准差。 |
| `contradiction_rate_mean`、`contradiction_rate_std` | 冲突率均值/标准差。 |
| `unsupported_warrant_ratio_mean`、`unsupported_warrant_ratio_std` | 未支持 warrant 比例均值/标准差。 |
| `claim_inconsistency_rate_mean`、`claim_inconsistency_rate_std` | claim 冲突率均值/标准差。 |

### `outputs/cost_scalability/cost_summary.csv`

| 表头 | 含义 |
|---|---|
| `stage_name` | 阶段名。 |
| `status` | 运行状态。 |
| `elapsed_seconds`、`elapsed_hours` | 耗时。 |
| `sample_count` | 推断出的样本/文档/行数。 |
| `throughput_per_hour` | 每小时处理量。 |
| `total_nli_pairs` | Stage 06 NLI pair 数。 |
| `unique_queries` | Stage 05 唯一 query 数。 |
| `avg_arus_per_sample` | 平均 ARU 数。 |

`hardware.json` 汇总各阶段硬件信息；`per_stage_stats.json` 保存完整元数据列表。

### `outputs/cost_scalability/runtime_review.csv`

| 表头 | 含义 |
|---|---|
| `review_id` | 复核行 ID。 |
| `stage_name`、`status`、`started_at`、`finished_at`、`elapsed_seconds` | 阶段运行信息。 |
| `sample_count` | 推断出的样本/文档/行数。 |
| `gpu_summary` | GPU 摘要。 |
| `input_paths`、`output_paths` | 输入输出路径摘要。 |
| `confirm_hardware`、`confirm_paths`、`confirm_runtime`、`notes` | 人工复核填写列。 |

### `outputs/significance/multiseed_results.csv`

| 表头 | 含义 |
|---|---|
| 自定义 group 列 | 如 `setting`。 |
| 模型列 | 默认 `model`。 |
| `metric` | 被汇总的数值指标名。 |
| `n_seeds` | seed 数量。 |
| `mean`、`std` | 多 seed 均值和总体标准差。 |

### `outputs/significance/ttest_summary.csv`

| 表头 | 含义 |
|---|---|
| 自定义 group 列 | 如 `setting`。 |
| `metric` | 指标名。 |
| `n_pairs` | 成对 seed 数。 |
| `baseline_model`、`target_model` | 对比模型。 |
| `baseline_mean`、`target_mean` | 两组均值。 |
| `mean_delta` | target - baseline。 |
| `t_stat`、`p_value` | 配对 t 检验统计量和 p 值。 |

### `outputs/gpt_judge/judgments.jsonl`

| 字段 | 含义 |
|---|---|
| `id` | 样本 ID。 |
| `timestamp` | 判断时间。 |
| `left_model_id`、`right_model_id` | 两个系统名称。 |
| `prompt` | 病例上下文。 |
| `gold_answer` | 参考答案。 |
| `presented_order` | 盲评呈现顺序。 |
| `judge_model_requested` | 请求的 judge 模型名。 |
| `judge_model_snapshot` | API 返回的模型快照名。 |
| `criteria` | 分标准判断结果。 |
| `summary` | 一句总结。 |
| `raw_output` | judge 原始输出。 |

`criteria` 中每个标准包含 `winner`、`presented_winner`、`reason`。标准包括 `hallucination_freedom`、`logical_coherence`、`overall_preference`。

### `outputs/gpt_judge/win_rates.csv`

| 表头 | 含义 |
|---|---|
| `criterion` | 评审标准。 |
| `left_model_id`、`right_model_id` | 两个系统名称。 |
| `n` | 判断数。 |
| `left_win_rate`、`tie_rate`、`right_win_rate` | 左胜、平、右胜比例。 |

## 16. 数据公平性/近重复复核表

### `outputs/data_fairness/near_duplicate_review.csv`

| 表头 | 含义 |
|---|---|
| `review_id` | 复核行 ID。 |
| `eval_dataset` | 评估集名称。 |
| `eval_id` | 评估样本 ID。 |
| `train_id` | 训练样本 ID。 |
| `question_similarity` | 问题相似度。 |
| `answer_similarity` | 答案相似度。 |
| `eval_prompt`、`train_prompt` | 评估/训练问题文本。 |
| `eval_answer`、`train_answer` | 评估/训练答案。 |
| `review_decision` | 人工复核结论。 |
| `confidence` | 人工置信度。 |
| `notes` | 备注。 |

## 17. 当前目录中已有的样例数据文件

当前 `data/fa_dpo_pipeline/score06_sample60_3/` 下已有一个 60 条复核包：

| 文件 | 对应 schema |
|---|---|
| `negatives.sample60.jsonl` | 负样本 JSONL schema。 |
| `arus.sample60.jsonl` | `*.arus.jsonl` schema；含可选 `sanitization`、`query_quality`、`query_needs_review`。 |
| `arus_with_evidence.sample60.jsonl` | `*.arus.with_evidence.jsonl` schema；含可选后处理字段。 |
| `faithfulness_scores.sample60.jsonl` | `*.scores.jsonl` schema。 |
| `sample_manifest.jsonl` / `sample_manifest.csv` | Score06 人工复核包 manifest schema。 |
| `sampled_ids.txt` | 抽样 ID 列表。 |
| `review_pack_summary.md` | 复核包摘要。 |

当前目录根部还有 `data/fa_dpo_pipeline/medcase_unfaithful_negatives.arus.with_evidence.jsonl`，它是一个本地 ARU-with-evidence 样例/旧版产物。顶层字段与 Stage 05 文件一致，但部分节点可能只含 `retrieval_candidates` 和 `retrieved_evidence`，不一定含新版 `retrieval_query_variants`、`rerank_query` 等字段。
