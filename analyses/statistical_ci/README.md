# R1.13 / R2.10 统计置信区间分析

本目录负责 Reviewer R1.13 和 Reviewer R2.10 对应的统计分析回复工作包。

## 当前状态

- 正文已经澄清：报告的 1.2--1.6 个百分点 downstream gain 是六个 backbone 上观察到的结果范围，不是置信区间。
- 计划执行的统计分析定义在 `experiment_plan.md`。
- `result_tables_template.tex` 提供 Table 2 / Table 3 风格更新所需的表格模板，只能在真实测量后填入。
- `response_update_template.md` 提供 paired logs 处理完成后可使用的最终回复文本。
- `scripts/compute_paired_ci.py` 会在 paired logs 导出后计算 case-level process paired bootstrap CI、benchmark-level exact McNemar test、benchmark-stratified macro-average accuracy CI，以及六个 backbone 的 exact Wilcoxon signed-rank 汇总。

不要把占位置信区间或 p 值写入 `main.tex` 或 `response_letter2.tex`。表格只能使用实验计划中定义的 question-level 和 case-level paired logs 计算得到的真实结果填充。

paired logs 准备好后的最小运行命令：

```bash
python analyses/statistical_ci/scripts/compute_paired_ci.py \
  --process-file analyses/statistical_ci/inputs/process_metrics.jsonl \
  --accuracy-file analyses/statistical_ci/inputs/accuracy_correctness.jsonl \
  --output-dir analyses/statistical_ci/outputs
```
