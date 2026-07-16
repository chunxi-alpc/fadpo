# ARU 忠实性变化研究

这个目录用于回答一个更直接的机制问题：

> 在控制 final answer 一致之后，Fa-DPO 是否让不同类型的 ARU 变得更忠实？

核心比较对象是“方法实施前后”的 ARU 不忠实比例变化，默认对应：

- `before`: Standard DPO
- `after`: Fa-DPO

## 实验设计

### 1. 只在 same-answer hard subset 上比较

先用 `build_same_answer_subset.py` 构造同题、同 gold、且两个模型都答对 final answer 的样本子集。

这样可以把分析重点放在 reasoning process，而不是 final-answer accuracy。

### 2. 统计单位用 question，而不是单个 ARU

两个模型对同一题生成的 ARU 数量和切分边界不一定完全一致，因此不能把 ARU 直接当作独立样本做显著性检验。

这里采用更稳妥的设计：

- 描述性统计：汇总 paired question 集合上的 pooled ARU 不忠实比例
- 推断统计：对每道题分别计算各 ARU 角色的不忠实比例，再做 paired test

### 3. 不忠实 ARU 的定义

对 `06_score_faithfulness.py` 输出的 `metrics.node_details`：

- 若 `decision_label == entail`，则记为 faithful
- 否则记为 unfaithful

也就是说：

- `neutral`
- `contradict`
- 以及其他非 `entail` 标签

都计入不忠实 ARU。

### 4. 角色粒度

脚本同时统计：

- `ALL`
- `O`
- `W`
- `D`
- `C`

对于某个具体角色，只保留“before 和 after 都至少包含一个该角色 ARU”的题目进入该角色的 paired analysis。

### 5. 统计分析

脚本默认输出两类统计：

- 两侧 paired sign test
  - 比较每题 role-wise unfaithful rate 的方向性变化
- question-level bootstrap 95% CI
  - 报告 mean paired difference (`after - before`) 的区间

这套口径避免了把同一题里的多个 ARU 当作独立观测所带来的 pseudo-replication。

## 运行方式

### 第一步：构造 same-answer subset

```bash
python fa_dpo_pipeline/build_same_answer_subset.py \
  --left-file /path/to/fa_dpo_predictions.jsonl \
  --right-file /path/to/standard_dpo_predictions.jsonl \
  --gold-file /path/to/gold_or_eval_data.jsonl \
  --left-model-id fa_dpo \
  --right-model-id standard_dpo \
  --output-file outputs/process_metrics/same_answer_subset.jsonl
```

### 第二步：运行 ARU shift analysis

`before` 和 `after` 输入应为 verifier score 文件，格式与 `06_score_faithfulness.py` 的输出一致，即每条样本包含：

- `id`
- `metrics.node_details`

运行示例：

```bash
python fa_dpo_pipeline/aru_faithfulness_shift_study/analyze_aru_shift.py \
  --subset-file outputs/process_metrics/same_answer_subset.jsonl \
  --before-score-file /path/to/standard_dpo_scores.jsonl \
  --after-score-file /path/to/fa_dpo_scores.jsonl \
  --before-label "Standard DPO" \
  --after-label "Fa-DPO" \
  --output-dir outputs/aru_faithfulness_shift_study
```

## 输出文件

- `per_example_rates.jsonl`
  - 每题、每个角色的 before/after 不忠实比例
- `role_summary.csv`
  - 各角色的汇总结果与显著性统计
- `summary.json`
  - 机器可读总结果
- `report.md`
  - 便于直接阅读的实验说明与结果摘要
- `paper_table.tex`
  - 可回填到论文的 LaTeX 表格片段

## 对论文的建议回填口径

主文建议新增一个 ARU-level analysis table，列：

- role
- before unfaithful rate
- after unfaithful rate
- delta (pp)
- 95% CI of mean paired delta
- paired sign test p-value

caption 里需要明确两点：

- rate 是 pooled ARU proportion
- 显著性检验使用的是 question-level paired rates

这样既保留了“ARU 比例变化”的直观性，又保证统计检验口径正确。
