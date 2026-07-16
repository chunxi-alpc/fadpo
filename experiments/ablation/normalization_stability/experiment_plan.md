# Token-Weight Normalization Stability 诊断实验计划

## 1. 实验目的

这个补充实验检验 Eq. (11) 的 token-weight normalization 是否通过稳定的局部 credit assignment 帮助 Fa-DPO，而不是通过无控制地放大 rejected-side loss。四个变体都保留 retained preference pairs 和 risk-adaptive margin；因此 `Uniform weighting (+ margin)` 是 margin-only / no token-risk-localization control，不是 Standard DPO-Retained-34K。

它回答三个问题：

1. token-level risk weighting 是否优于 uniform rejected-token weighting；
2. normalization 是否比直接使用 raw token weights 更稳定；
3. sparse high-risk masks 是否会导致梯度不稳定或极端 effective token weight。

这是一个小型诊断实验，不改变主 Fa-DPO 公式，也不需要重跑全部主实验。

## 2. 所需输入

### 2.1 DPO 训练数据

使用主实验中的 retained Fa-DPO preference pairs。

当前包已经提供：

- `data/train_fadpo.jsonl`
- `data/eval_fadpo.jsonl`

最低字段要求：

| Field | 含义 |
|---|---|
| `id` | preference pair ID |
| `prompt` | 医学问题或病例输入 |
| `chosen` | preferred response |
| `rejected` | retained rejected response |
| `topo_mask` | token-aligned raw token weights |
| `margin` | risk-adaptive margin |

本实验直接使用预计算 `topo_mask`，因此不需要重新提供 ARU spans 或 ARU risk scores。

### 2.2 模型输入

需要确认：

| Item | 说明 |
|---|---|
| Base model | II-Medical-8B 本地路径或 HuggingFace ID |
| Reference model | 通常与 base model 相同并冻结 |
| Tokenizer | 通常从 base model 加载 |
| SFT / starting checkpoint | 如果主实验从本地 adapted checkpoint 开始，需要使用同一 checkpoint |

`configs/normalization_stability.yaml` 中的默认 model name 只是占位。实际运行前应检查服务器上的 `/home/models/` 或项目指定模型目录。

### 2.3 训练配置

与主 ablation 保持一致：

| Setting | 默认值 |
|---|---|
| Epochs | 1 |
| Learning rate | `5e-7` |
| Effective batch size | 32 附近，按服务器显存调整 |
| Precision | bf16 |
| Max sequence length | 4096 |
| Max prompt length | 2048 |
| DPO beta | 0.3 |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Gradient clipping | 与主训练一致 |

### 2.4 评估输入

训练后每个 variant 都应评估：

| Evaluation | 需要内容 |
|---|---|
| MedBullets-op4 | dev/test file + answer extraction rule |
| MedBullets-op5 | dev/test file + answer extraction rule |
| MedXpertQA | dev/test file + answer extraction rule |
| MedQA | dev/test file + answer extraction rule |
| MedCaseReasoning process metrics | same-answer subset + ARU/NLI process metric script |

如果暂时没有完整 evaluation runner，至少应生成每个 variant 的 `eval_results.json`，能填多少真实字段填多少；未知字段不要编造。

## 3. 比较的 variants

设 \(\rho_t\) 是 rejected response 上第 \(t\) 个 token 的 policy-over-reference log-ratio。下表只定义 rejected-side token aggregation；完整 DPO logit 仍保留相同的 chosen-side term 和相同的 risk-adaptive margin。

| Variant | 定义 | 目的 |
|---|---|---|
| Uniform weighting (+ margin) | \(\Delta_{\text{uni}}=\sum_t\rho_t\) | 移除 token-risk localization，但保留 risk-adaptive margin |
| No normalization | \(\Delta_{\text{unnorm}}=\sum_t m_t\rho_t\) | 检查 raw loss-scale effect |
| Original normalized weighting | \(\Delta_m=\sum_t \omega_t\rho_t\)，其中 \(\omega_t=T_lm_t/\sum_jm_j\) | 主 Fa-DPO 设计 |
| Capped normalization | 先 normalize，再 cap \(\omega_t\)，最后重新归一化到 \(T_l\) | sparse-mask robustness check |

推荐 cap：

```text
c_norm = 10
```

可选再跑：

```text
c_norm = 20
```

capped variant 只是诊断，不是主方法。

## 4. 实验流程

### 步骤 1：准备数据

1. 加载 retained preference pairs。
2. Tokenize `prompt`、`chosen`、`rejected`。
3. 读取 `topo_mask` 并对齐 rejected-response tokens。
4. 对不同 variant 生成 effective token weights。

当前数据已经包含 token-aligned `topo_mask`，不需要从 ARU span 重新构造 mask。

### 步骤 2：训练四个 variants

四个 variant 使用完全相同设置：

1. `uniform`
2. `no_normalization`
3. `original_normalized`
4. `capped_normalization`

保持不变：

- base model；
- reference model；
- train/eval split；
- LoRA settings；
- learning rate；
- batch size；
- max length；
- number of epochs；
- random seed；
- checkpoint selection rule。

唯一变化是 rejected-side token weighting rule。

### 步骤 3：记录 stability signals

每个 training step 保存：

| Logged field | 含义 |
|---|---|
| `step` | 训练 step |
| `loss` | training loss |
| `grad_norm` | backward 后的 gradient norm |
| `max_effective_weight` | batch 内最大 effective token weight |
| `mean_effective_weight` | batch 内平均 effective token weight |
| `high_risk_token_ratio` | high-risk token 占 rejected tokens 的比例 |
| `high_risk_effective_weight_median` | high-risk token effective weight median |
| `high_risk_effective_weight_p95` | high-risk token effective weight p95 |
| `high_risk_effective_weight_max` | high-risk token effective weight max |
| `high_risk_effective_weight_count` | 用于分布统计的 high-risk token 数 |
| `nan_or_inf` | 是否出现 NaN/Inf |

日志路径：

```text
runs/{variant}/train_metrics.jsonl
```

### 步骤 4：评估

每个 trained variant 评估：

| Metric | 来源 |
|---|---|
| MedBullets-op4 accuracy | downstream dev/test |
| MedBullets-op5 accuracy | downstream dev/test |
| MedXpertQA accuracy | downstream dev/test |
| MedQA accuracy | downstream dev/test |
| Macro-average accuracy | 四个 accuracy 平均 |
| Support Rate | MedCaseReasoning same-answer subset |
| Unsupported Medical Reasoning Rate | MedCaseReasoning same-answer subset |
| Loss trend | training log |
| Grad norm mean | training log |
| Grad norm max | training log |
| NaN/divergence | training log |
| High-risk effective-weight median/p95/max | training log |
| Gradient-norm instability events | summarizer 中定义的 NaN/Inf 或 grad-norm spike |

## 5. 输出文件结构

推荐结构：

```text
data/
  train_fadpo.jsonl
  eval_fadpo.jsonl

experiments/ablation/normalization_stability/
  configs/
    normalization_stability.yaml
  runs/
    uniform/
      train_metrics.jsonl
      eval_results.json
    no_normalization/
      train_metrics.jsonl
      eval_results.json
    original_normalized/
      train_metrics.jsonl
      eval_results.json
    capped_normalization/
      train_metrics.jsonl
      eval_results.json
  analysis/
    normalization_stability_results.csv
    normalization_stability_results.tex
    normalization_weight_diagnostics.csv
    normalization_weight_diagnostics.tex
    loss_curve.pdf
    grad_norm_curve.pdf
```

## 6. 论文结果表

主表字段：

| Variant | Avg. Dev Acc. | Support Rate | Unsupported Med. Reasoning | Loss Trend | Grad Norm Mean | Grad Norm Max | NaN/Div. |
|---|---:|---:|---:|---|---:|---:|---|
| Uniform weighting (+ margin) | 待定 | 待定 | 待定 | 待定 | 待定 | 待定 | 待定 |
| No normalization | 待定 | 待定 | 待定 | 待定 | 待定 | 待定 | 待定 |
| Original normalized weighting | 待定 | 待定 | 待定 | 待定 | 待定 | 待定 | 待定 |
| Capped normalization | 待定 | 待定 | 待定 | 待定 | 待定 | 待定 | 待定 |

补充 weight-distribution 表：

| Variant | High-Risk Weight Median | High-Risk Weight P95 | High-Risk Weight Max | Instability Events |
|---|---:|---:|---:|---:|
| Uniform weighting (+ margin) | 待定 | 待定 | 待定 | 待定 |
| No normalization | 待定 | 待定 | 待定 | 待定 |
| Original normalized weighting | 待定 | 待定 | 待定 | 待定 |
| Capped normalization | 待定 | 待定 | 待定 | 待定 |

没有真实数值前不能写 result claims。

## 7. 解释规则

- 如果 original normalized weighting 优于 uniform weighting，说明 token-level local credit assignment 有用。
- 如果 original normalized weighting 比 no normalization 更稳定，说明 normalization 不是单纯 loss-scale amplifier。
- 如果 no normalization 的 grad norm 更大或 loss behavior 更差，支持 Eq. (11) 的 scale-control 作用。
- 如果 capped normalization 接近 original normalized weighting，说明 sparse high-risk masks 不是主要不稳定来源。
- 如果 high-risk effective-weight p95/max 保持在合理范围且 instability events 很少，支持 revised manuscript 中的 boundedness 论证。
- 如果 capped normalization 明显优于 original normalized weighting，需要讨论主方法是否仍保持 Eq. (11)，或把 capped 作为 robustness check 报告。
- 如果结果混合，直接写 trade-off，避免过度解释。

## 8. 执行前最低检查

运行前确认：

1. `data/train_fadpo.jsonl` 和 `data/eval_fadpo.jsonl` 存在。
2. 字段包含 `id`、`prompt`、`chosen`、`rejected`、`topo_mask`、`margin`。
3. base model path 或 HuggingFace ID 正确。
4. reference model 规则明确。
5. 训练脚本可正常加载 TRL / PEFT / Accelerate。
6. 下游 evaluation script 或 `eval_results.json` schema 明确。
7. 所有 variant 只改变 token weighting rule，不改变其他训练条件。
