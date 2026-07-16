# R1.14 组件与交互消融工作包

本目录负责 R1.14 要求的新增消融实验：

- 针对 negative filtering、token weighting 和 risk-adaptive margin 的 progressive / component-addition ablations；
- 针对 reasoning-unit granularity 和 role-specific routing 的 interaction ablations；
- 消融结果的 paired bootstrap confidence intervals 和 pairwise drop-comparison tests。

相邻问题已经由其他目录覆盖：

- `analyses/statistical_ci/` 负责 Table 2 / Table 3 的统计置信区间和 paired tests；
- `experiments/ablation/normalization_stability/` 负责 token-weight normalization stability；
- `experiments/verifier_validation/nli_false_positive/` 负责 verifier false positives。

此前没有专门目录负责 R1.14 提出的 component-addition 与 interaction ablation，因此本目录作为该问题的独立工作包。

## 文件

- `experiment_plan.md`：详细 run matrix、估计量、控制变量和统计分析计划。
- `result_tables_template.tex`：可直接用于论文的表格模板，只能填入真实测量值。
- `response_update_template.md`：实验完成后用于回复信的最终表述。
