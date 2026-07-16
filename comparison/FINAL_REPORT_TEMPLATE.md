# 最终对比报告模板

## 1. 结论摘要

- 本次比较的模型：
- 最优主生成器：
- 最优低成本候选生成器：
- 是否推荐混合工作流：
- 最关键发现：

## 2. 实验设置

- 数据集：
- split：
- 样本数：
- 错误类型：
- seeds：
- prompt 版本：
- 是否开启生成期 LLM judge：
- 统一评测方式：

## 3. 模型清单

| model_id | provider | route_name | endpoint_type | notes |
|---|---|---|---|---|

## 4. 总体结果

| model_id | attempts | feasible_rate | valid_rate | accepted_rate | review_rate | rejected_rate | failure_rate |
|---|---:|---:|---:|---:|---:|---:|---:|

## 5. 质量结果

| model_id | target_match_rate | single_dominant_error_rate | answer_policy_match_rate | localized_rewrite_score | plausibility_score |
|---|---:|---:|---:|---:|---:|

## 6. 成本与效率

| model_id | avg_latency_per_attempt | latency_per_accepted_sample | avg_cost_per_attempt | cost_per_accepted_sample |
|---|---:|---:|---:|---:|

## 7. 分错误类型分析

### F1

### F2

### F3

### F4

## 8. 人工复核发现

- 哪个模型最容易过度重写：
- 哪个模型最容易多错误污染：
- 哪个模型最自然但不够受控：
- 哪个模型最受控但略显机械：

## 9. 失败案例分析

- JSON/结构化失败：
- 明显跑题：
- 答案策略错误：
- 计划阶段过度保守：

## 10. 最终建议

### 方案 A：高质量优先

### 方案 B：成本优先

### 方案 C：混合流水线

## 11. 后续动作

1. 
2. 
3. 
