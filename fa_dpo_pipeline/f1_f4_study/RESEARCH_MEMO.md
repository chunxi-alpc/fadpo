# F1-F4 Study Research Memo

Last updated: 2026-04-09 (after 3-model pilot across 4 datasets: `gpt_5_4`, `gemini_3_1_pro`, `claude_sonnet_4_6_thinking`)

这份 memo 用来记录当前任务里最重要、后续论文写作必须保持一致的结论与讨论。

## 1. 已经确认的事实

### 1.1 实验框架已经可用

- `f1_f4_study` 现在支持 `smoke -> pilot -> final` 的分阶段增量实验。
- 同一数据集使用固定采样池，后续阶段只扩展样本前缀，不更换已抽中的样本。
- `manual backend` 已经真实跑通，可以不依赖 OpenAI-compatible HTTP 服务完成 generation、judge 和阶段分析。
- 真实 API 链路也已经跑通，当前 `gpt_5_4`、`gemini_3_1_pro`、`claude_sonnet_4_6_thinking` 的 pilot generation 与 LLM judge 都已真实完成。

### 1.2 批跑稳定性问题已经修好

- `manual_judged` 已纳入 completed 状态。
- manual import 现在按样本 `example_id` 覆盖写入并自动去重，不会重复堆积旧结果。
- generation / judge 脚本现在统一设置 `trust_env=False`，不会再被系统代理中的 SOCKS 配置卡住。
- generation / judge 现在加入显式 request timeout 与 retry，批跑不再因为少数慢请求无限挂住。
- `run_study.py` 现在支持按模型覆盖 generation token 上限。
- 在当前网关条件下，`claude-sonnet-4-6-thinking` 的 case generation 需使用 `max_tokens=800` 才能稳定跑完 `MedCaseReasoning`。
- 当前 `smoke` 和 `pilot` 结果都已按干净口径重算。

## 2. 当前 smoke 结果

来源：

- [stage_report.md](/root/med/outputs/f1_f4_study/medical_f1_f4_natural_error_study/stage_reports/smoke/stage_report.md)

当前 `gpt_5_4` smoke 结果：

- 总样本数：40
- 总体答案准确率：85.0%
- 任意 F1-F4 错误率：2.5%
- `F4` presence：0.0%
- `MedCaseReasoning`: 10 条，准确率 100.0%，未观察到 F1-F4
- `MedBullets-OP4`: 10 条，准确率 90.0%，观察到 1 条 `F2`
- `MedQA`: 10 条，准确率 100.0%，未观察到 F1-F4
- `Med-HALT reasoning`: 10 条，准确率 50.0%，未观察到 F1-F4

这一步的作用已经完成：它证明了流程打通，而且至少 `F2` 能自然出现。

## 3. 当前 pilot 结果

来源：

- [stage_report.md](/root/med/outputs/f1_f4_study/medical_f1_f4_natural_error_study/stage_reports/pilot/stage_report.md)

当前 3 模型 pilot 总结果：

- 总样本数：600
- 总体答案准确率：58.6%
- 任意 F1-F4 错误率：33.7%
- `F4` presence：14.3%

分模型结果：

- `gpt_5_4`
  - 总样本数：200
  - 总体答案准确率：65.0%
  - 任意错误率：25.5%
  - `F1/F2/F3/F4` present：`7.0% / 3.0% / 5.5% / 15.5%`
- `gemini_3_1_pro`
  - 总样本数：200
  - 总体答案准确率：66.0%
  - 任意错误率：32.5%
  - `F1/F2/F3/F4` present：`0.5% / 4.5% / 14.0% / 15.0%`
- `claude_sonnet_4_6_thinking`
  - 总样本数：200
  - 总体答案准确率：44.7%
  - 任意错误率：43.0%
  - `F1/F2/F3/F4` present：`1.0% / 8.0% / 24.0% / 12.5%`

分数据集聚合结果：

- `MedCaseReasoning`，240 条
  - 准确率：33.3%
  - 任意错误率：47.9%
  - `F1/F2/F3/F4`：`7.1% / 2.9% / 34.6% / 10.0%`
- `MedBullets-OP4`，120 条
  - 准确率：82.5%
  - 任意错误率：25.0%
  - `F2/F4`：`14.2% / 11.7%`
- `MedQA`，120 条
  - 准确率：95.0%
  - 任意错误率：6.7%
  - `F2/F4`：`4.2% / 1.7%`
- `Med-HALT reasoning`，120 条
  - 准确率：48.7%
  - 任意错误率：40.8%
  - `F4`：`38.3%`

当前 `gpt_5_4` pilot 总结果：

- 总样本数：200
- 总体答案准确率：65.0%
- 任意 F1-F4 错误率：25.5%
- `F1` present：7.0%
- `F2` present：3.0%
- `F3` present：5.5%
- `F4` present：15.5%

分数据集结果：

- `MedCaseReasoning`，80 条
  - 准确率：46.2%
  - 任意错误率：37.5%
  - `F1`：17.5%
  - `F2`：1.2%
  - `F3`：13.8%
  - `F4`：18.8%
- `MedBullets-OP4`，40 条
  - 准确率：85.0%
  - 任意错误率：17.5%
  - `F2`：7.5%
  - `F4`：10.0%
- `MedQA`，40 条
  - 准确率：97.5%
  - 任意错误率：2.5%
  - `F2`：2.5%
- `Med-HALT reasoning`，40 条
  - 准确率：50.0%
  - 任意错误率：32.5%
  - `F2`：2.5%
  - `F4`：30.0%

## 4. 当前最重要的结论

### 4.1 四类错误现在已有初步多模型证据

- 在 `3` 个模型、`4` 个数据集的 pilot 中，`F1/F2/F3/F4` 四类都已被观察到自然出现。
- 因此，这四类错误不再只是单模型现象，而已有初步 cross-model 证据支持。

### 4.2 `F4` 仍是当前最稳定的跨模型、跨数据集错误

- 三个模型的总体 `F4` presence 分别为 `15.5% / 15.0% / 12.5%`，量级接近。
- 数据集聚合上，`Med-HALT reasoning` 的 `F4` 达到 `38.3%`，说明它仍是最强的 `F4` stress test。
- `F2` 也具有稳定的跨数据集可见性，但整体频率低于 `F4`。

### 4.3 `F1/F3` 仍明显依赖病例型数据

- 聚合后 `MedCaseReasoning` 的 `F1` 为 `7.1%`、`F3` 为 `34.6%`，显著高于三个 MCQ 数据源。
- 这说明 `F1/F3` 仍不是“随便一个医学 benchmark 都能稳定暴露”的错误，而更依赖病例级证据环境。

### 4.4 模型间分布差异已经可见

- `gpt_5_4` 当前总体 any-error 最低，仍是三者中最稳的一档。
- `gemini_3_1_pro` 同时暴露了较明显的 `F3` 与 `F4`。
- `claude_sonnet_4_6_thinking` 当前 `F3` 最高，主要由 `MedCaseReasoning` 拉高。

### 4.3 当前最合理的论文表述

在目前这批证据下，最稳的表述应为：

- `F1-F4` 是一组 clinically meaningful、且在真实医学推理输出中可自然出现的失败模式。
- 我们已在 `3` 个模型、`4` 个数据集的 pilot 上观察到这四类错误的自然出现。
- 这些错误类型的自然频率并不均衡，其中 `F4` 当前最稳定，`F1/F3` 更依赖病例型数据。
- 这说明 faithfulness 风险不是单一模型或单一数据集 artifact，而具有初步跨模型、跨数据集可见性。

## 5. 当前仍然不能过度声称的地方

后续写论文或汇报时，当前阶段不要写成：

- “四类错误在多个模型中都高频出现”
- “F1-F4 已被系统性验证为普遍存在于所有医学 LLM”
- “当前结果已经证明模型内部推理普遍存在这四类失败”
- “三个模型之间的原始 accuracy 可以被直接当作公平 leaderboard 比较”

当前更准确的说法是：

- “我们已经在 3 个模型、4 个数据集的 pilot 上观察到自然出现的 F1-F4。”
- “答案正确并不自动意味着推理 faithful。”
- “不同错误类型的频率不均衡，且与数据形态强相关。”
- “当前 cross-model 分布差异是有信息量的，但仍应视为 pilot-level evidence，而不是最终定论。”

补充 caveat：

- `claude_sonnet_4_6_thinking` 的 `MedCaseReasoning` 运行在当前网关条件下使用了更低的 `generation_case_max_tokens=800`。
- 因此它在 `medcase` 上的 accuracy 与 error mix 可用于自然错误观察，但不宜被过度解读为干净的 capability comparison。

## 6. 为什么当前最稳定的是 F4，而 F1/F3 更依赖病例数据

- `MCQ` 数据上，`F2` 和 `F4` 更容易从可见文本中识别。
- `F1/F3` 更依赖病例级证据集合和更完整的 reasoning chain。
- 因此如果没有病例型主数据集，`F1/F3` 很可能会被低估。
- 当前 judge 较保守，因此它更可能漏报，而不是夸大。

## 7. 当前对数据集角色的判断

- `MedCaseReasoning`
  - 是主数据集
  - 主要用于观察 `F1/F2/F3`
  - 目前也是 `F1/F3` 的主要来源
- `MedBullets` / `MedQA`
  - 更适合做外部验证
  - 更适合看“答对但理由不忠实”与部分 `F4`
- `Med-HALT reasoning`
  - 更像压力测试集
  - 对放大 `F4` 很有帮助
  - 不适合作为主结论的唯一依据

## 8. 下一阶段最关键的问题

截至当前，下面三件事已经得到肯定回答：

1. 至少两类自然 faithfulness error 会稳定出现。
2. 这些错误并非只出现在单一数据集。
3. `F1/F2/F3/F4` 的自然出现并非单模型现象。

下一步更关键的问题变成：

1. 在更大样本的 `final` 阶段，`F4` 是否仍是最稳定的跨模型主导错误。
2. `F1/F3` 是否仍主要局限在病例型数据。
3. `claude_sonnet_4_6_thinking` 在更稳定 backend 或更充足 token 预算下，`medcase` 的高 `F3` 现象是否仍然成立。
4. 更大规模自然错误样本中，有多少比例仍可被 `F1-F4` 覆盖，而不是落入 `Other`。

## 9. 当前任务的核心 takeaways

- 这条研究线值得继续，不是空方向。
- 现在已经不只是 `F2`；`F1/F2/F3/F4` 四类都在 `3` 模型真实 pilot 中出现了。
- 当前最稳、最跨模型、最跨数据集的自然失败模式仍是 `F4`。
- `F1/F3` 的发现依赖病例型主数据集，这说明主数据集选择是有必要的，而不是 dataset artifact。
- 下一步应从“这四类会不会出现”转到“它们在不同模型上的稳定性、分布差异，以及 natural coverage 上限是什么”。
