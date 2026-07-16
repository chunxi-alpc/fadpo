# 指标与分析规则

这个文件定义的是“最后怎么判谁更好”。建议不要只看一个总分，而是至少分成 5 组指标。

## 1. 运行稳定性指标

这些指标回答的是：模型能不能稳定把任务做完。

### `plan_feasible_rate`

定义：

`plan_infeasible` 之外的比例。

解释：

- 太低：模型经常觉得任务做不出来
- 太高：不一定是好事，也可能是模型过度自信

### `runtime_failure_rate`

定义：

`failures.jsonl / attempts.jsonl`

解释：

- 看 JSON 崩溃
- 看 API 调用失败
- 看超时重试后仍失败

### `valid_generation_rate`

定义：

能进入 heuristic QC 的比例。

解释：

这比单纯“接口有没有返回文本”更有意义，因为它要求返回结构可解析、关键字段不为空。

## 2. 任务控制能力指标

这些指标回答的是：模型能不能真的按你指定的错误类型去做，而不是乱做。

### `target_error_match_rate`

定义：

在统一 judge 或人工复核中，被判定为目标错误类型的比例。

### `single_dominant_error_rate`

定义：

样本被认为“只有一个主导错误”的比例。

解释：

这个指标很重要，因为你要的是“受控负样本”，不是模糊坏样本。

### `answer_policy_match_rate`

定义：

在应该保留答案时实际保留，在应该不保留时实际改变的比例。

解释：

这个指标特别适合看 `F4` 和其他类型的边界控制能力。

## 3. 局部改写能力指标

这些指标回答的是：模型是不是在做“局部、受控”的改写。

### `localized_rewrite_score`

建议由以下 4 项合成：

- similarity 是否在合理范围
- novelty ratio 是否不过高
- length ratio 是否不过度偏离
- modified spans 是否集中在局部

### `over-rewrite_rate`

定义：

被人工标注为“基本重写了整段 reasoning”的比例。

解释：

很多模型在表面上完成任务，但其实把整段推理重写了。这会严重破坏训练样本的“局部控制性”。

## 4. 临床与可用性指标

这些指标回答的是：样本是否像真的医学推理，而不是胡写。

### `clinical_plausibility_score`

建议人工或统一 judge 评 1 到 5 分：

- 1：明显不合理
- 2：较多明显问题
- 3：基本可读但有明显瑕疵
- 4：整体合理
- 5：非常自然

### `off_topic_rate`

定义：

偏离病例主题或引入无关路径的比例。

### `usable_accept_rate`

定义：

最终可直接入库的样本比例。

这通常比单独的 accepted rate 更关键，因为它代表真实生产价值。

## 5. 效率与成本指标

这些指标回答的是：值不值得长期用。

### `latency_per_attempt`

定义：

每次 attempt 的平均耗时。

### `latency_per_accepted_sample`

定义：

总耗时 / accepted 样本数

解释：

这个指标比“单次响应快不快”更有意义，因为它把失败、重试、拒绝都算进来了。

### `cost_per_attempt`

定义：

单次尝试的平均 token 成本或账单成本。

### `cost_per_accepted_sample`

定义：

总成本 / accepted 样本数

这个指标通常最适合拿来做生产决策。

## 推荐的汇总表

最后建议每个模型至少出一张汇总表：

| model_id | attempts | feasible_rate | valid_rate | accepted_rate | review_rate | target_match_rate | single_error_rate | locality_score | plausibility_score | cost_per_accept | latency_per_accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## 推荐的可视化

至少做下面 5 张图：

1. 每模型 `accepted / review / rejected / infeasible / failure` 堆叠柱状图
2. 每错误类型的 accepted rate 分组柱状图
3. `cost_per_accepted_sample` vs `accepted_rate` 散点图
4. `latency_per_accepted_sample` vs `plausibility_score` 散点图
5. 人工复核的 pairwise win-rate 热力图

## 人工复核的最小维度

人工复核建议至少打下面 6 项：

1. 是否命中目标错误类型
2. 是否只有单一主导错误
3. 是否符合答案保留策略
4. 是否是局部改写
5. 是否临床表面合理
6. 是否总体可入库

对应模板见：
[HUMAN_REVIEW_TEMPLATE.csv](/mnt/d/med03/comparison/HUMAN_REVIEW_TEMPLATE.csv)

## 最终决策规则建议

### 如果目标是“最高质量主生成器”

优先级建议：

1. `usable_accept_rate`
2. `target_error_match_rate`
3. `single_dominant_error_rate`
4. `clinical_plausibility_score`
5. `cost_per_accepted_sample`

### 如果目标是“低成本大规模候选池”

优先级建议：

1. `cost_per_accepted_sample`
2. `latency_per_accepted_sample`
3. `valid_generation_rate`
4. `localized_rewrite_score`
5. 再用强模型做二次筛选

### 如果目标是“构建混合工作流”

推荐你最后输出的不是“唯一冠军”，而是“角色分工”：

- 高质量主生成器
- 低成本候选扩展器
- 强 judge / 强复核模型

这通常比单一模型全包更稳。
