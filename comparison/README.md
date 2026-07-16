# 跨模型生成对比工作流

这个目录是给你做“同一任务、不同模型”横向比较用的实验工作包。目标不是只看哪家文笔更像，而是把下面几件事分开比较：

1. 谁更容易生成出可用样本
2. 谁更容易精确命中目标错误类型
3. 谁更能做到“局部改写”而不是整段重写
4. 谁更能保持临床表面合理性
5. 谁的成本、速度、失败率更可控

## 先说一个很重要的模型命名校准

截至 **2026-03-18**，我查到的官方页面里：

- OpenAI 官方模型文档当前推荐的最新通用 GPT 主线是 `GPT-5.1`，并明确把 `GPT-5` 标成“previous model”；我没有在官方 OpenAI 模型页看到 `GPT-5.4` 这个官方 API 模型名。
  来源：
  [OpenAI Models](https://platform.openai.com/docs/models/gpt-4-and-gpt)
  [OpenAI GPT-5](https://platform.openai.com/docs/models/gpt-5)
- Anthropic 官方文档/定价页当前列出了 `Claude Opus 4.1`、`Claude Opus 4`、`Claude Sonnet 4` 等型号；如果你说“Claude 最强模型”，按官方列表更接近 `Claude Opus 4.1`。
  来源：
  [Anthropic Models](https://docs.anthropic.com/de/docs/about-claude/models)
  [Anthropic Pricing](https://docs.anthropic.com/de/docs/about-claude/pricing)

这意味着：

- 如果你后面要比较 `gpt-5.4`，建议把它当成“你所在网关/平台暴露出来的路由名”单独记录，而不是默认当成 OpenAI 官方模型名。
- 如果你要比较 Claude，请在实验记录里写清楚“直连 Anthropic API”还是“通过 OpenAI-compatible 网关接入的 Claude 路由”。

## 你应该怎么比

最稳的做法是把实验拆成 6 个阶段：

### 阶段 0：接口与可比性审计

先确认每个候选模型都能满足同一实验协议：

- 都能稳定返回 JSON
- 都能处理你当前 prompt 长度
- 都能接受同样的系统提示和用户提示结构
- 都能通过相同的温度、seed、max_tokens 配置运行
- 都能输出到同一数据结构

如果某个模型只能通过自定义路由名访问，就把“真实 provider + 网关名 + 路由名”全部记进 manifest，不要只写表面模型名。

### 阶段 1：冻结实验协议

这一阶段不要急着跑全量，先把“实验协议”冻结：

- 固定数据集和 split
- 固定样本子集
- 固定错误类型集合
- 固定 prompt 版本
- 固定温度、max_tokens、重试次数
- 固定输出目录结构

一旦开始主实验，就不要再边跑边改 prompt。否则你比较的就不是模型，而是“模型 + prompt 版本”。

### 阶段 2：小样本 pilot

建议先做一个小样本探路：

- 20 到 50 个正样本
- 每个样本跑 `F1/F2/F3/F4`
- 每个模型先跑 1 个 seed

pilot 的目标不是出结论，而是回答：

- 哪些模型会频繁 JSON 失败
- 哪些模型几乎总是整段重写
- 哪些模型在某些错误类型上明显不稳定
- 当前 heuristic 阈值是不是过严或过松

如果 pilot 都跑不稳，不要直接上主实验。

### 阶段 3：主实验批量生成

主实验建议这样组织：

- 同一批输入样本
- 同一 prompt 版本
- 同一超参数
- 每个模型跑多个 seed

建议起步配置：

- 样本数：100 到 300
- 错误类型：`F1,F2,F3,F4`
- seed：至少 3 个，例如 `42, 43, 44`
- generation mode：优先用 `all`

这样你最后能同时看到：

- 平均水平
- 方差
- 某模型是不是“偶尔非常好，但大多数时候不稳”

### 阶段 4：统一评测

这一阶段非常关键：

- **不要把生成模型自己的 judge 结果直接当最终排名**
- 最好把生成和评测拆开

推荐顺序：

1. 生成阶段先关闭 `--enable-llm-qc`
2. 保留 heuristic QC 和 sidecar 日志
3. 后处理时再做统一盲评

统一评测至少包含两层：

- 自动指标层
- 人工复核层

自动指标看规模，人工复核看真实性。

### 阶段 5：分层人工复核

人工复核不要只抽“最好看的样本”，要分层抽样：

- 每个模型都抽
- 每个错误类型都抽
- 每个状态都抽：accepted / review / rejected / infeasible

建议最少抽：

- 每模型每错误类型 20 条
- 若模型很多，可以先每模型总计 80 到 120 条

### 阶段 6：最终分析与选型

最终不要只报一个总分，而要给出：

- 质量
- 稳定性
- 成本
- 延迟
- 最适合的使用位置

例如：

- `GPT-5.1`：总体最稳，适合做高质量主生成器
- `Claude Opus 4.1`：局部改写和文风保真好，适合高质量补充生成
- `Qwen3-Next`：成本低、速度快，适合大规模候选池生成

## 你现在这套仓库最适合的实验策略

结合当前 [generate_bad_reasoning.py](/mnt/d/med03/generate_bad_reasoning.py) 的能力，我建议你这样用：

### 生成阶段

先统一生成，不开启 LLM judge：

```bash
python generate_bad_reasoning.py \
  --source medcase \
  --split test \
  --error-types F1,F2,F3,F4 \
  --generation-mode all \
  --max-samples 100 \
  --seed 42 \
  --temperature 0.3 \
  --output-file outputs/model_x/seed_42/accepted.jsonl
```

这样每个模型都会产出：

- 主输出 `accepted.jsonl`
- `*.attempts.jsonl`
- `*.rejected.jsonl`
- `*.infeasible.jsonl`
- `*.failures.jsonl`

### 统一评测阶段

推荐把主比较指标放在以下几项：

- `plan_feasible_rate`
- `valid_generation_rate`
- `accepted_rate`
- `rejected_rate`
- `answer_policy_match_rate`
- `target_error_match_rate`
- `single_dominant_error_rate`
- `localized_rewrite_score`
- `clinical_plausibility_score`
- `cost_per_accepted_sample`
- `latency_per_successful_sample`

详细定义见：
[METRICS_AND_ANALYSIS.md](/mnt/d/med03/comparison/METRICS_AND_ANALYSIS.md)

## 目录说明

- [EXPERIMENT_MANIFEST.example.json](/mnt/d/med03/comparison/EXPERIMENT_MANIFEST.example.json)
  实验配置总表模板
- [model_matrix.example.tsv](/mnt/d/med03/comparison/model_matrix.example.tsv)
  模型矩阵模板
- [run_generation_sweep.sh](/mnt/d/med03/comparison/run_generation_sweep.sh)
  批量运行脚本模板
- [METRICS_AND_ANALYSIS.md](/mnt/d/med03/comparison/METRICS_AND_ANALYSIS.md)
  指标定义和分析规则
- [HUMAN_REVIEW_TEMPLATE.csv](/mnt/d/med03/comparison/HUMAN_REVIEW_TEMPLATE.csv)
  人工复核表头模板
- [FINAL_REPORT_TEMPLATE.md](/mnt/d/med03/comparison/FINAL_REPORT_TEMPLATE.md)
  最终报告模板

## 一个务实的推荐顺序

如果你想最快得到靠谱结论，我建议：

1. 先把 `GPT-5.1`、你的 `gpt-5.4` 网关路由、`Claude Opus 4.1` 路由、`Qwen3-Next` 本地路由全部登记进 model matrix
2. 先跑 30 条 pilot
3. 看哪个模型需要单独调温度或 max_tokens
4. 确认协议冻结后再跑 100 到 300 条主实验
5. 自动统计 + 人工抽样复核
6. 最后做“质量/成本/速度”三维决策

如果后面你希望，我可以继续把“统一评测脚本”和“结果聚合脚本”也直接补到仓库里。
