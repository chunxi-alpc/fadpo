# 归一化稳定性实验

本工作包用于运行 Fa-DPO token-weight normalization 的补充诊断。它在同一批 retained preference pairs 上训练四个变体，并比较训练稳定性、下游指标和 process metrics。四个变体都保留 risk-adaptive margin；这里的 `uniform` 不是 Standard DPO-Retained-34K，而是只移除 token-risk localization 的 retained-pool margin-only control。

## 实验检验的问题

该实验检验 Eq. (11) 是否通过稳定局部 token-level credit assignment 发挥作用，而不是仅仅放大 rejected-side loss。

## 变体

| 变体 | 含义 |
|---|---|
| `uniform` | 忽略 `topo_mask`；每个 rejected token 的权重都设为 1，同时保留 risk-adaptive margin。 |
| `no_normalization` | 直接使用原始 `topo_mask` 权重。 |
| `original_normalized` | 对 token weights 归一化，使 rejected-side total weight 等于 active rejected-token count。 |
| `capped_normalization` | 先应用原始归一化，再将每个 token weight 截断到 `capped_normalization_cap`，最后重新归一化。 |

## 需要上传的文件

实验目录只应包含代码和配置；共享的 training/evaluation JSONL 文件只在仓库级 `data/` 目录保留一份，避免重复拷贝约 200MB 数据。

```text
experiments/ablation/normalization_stability/
  README.md
  experiment_plan.md
  requirements.txt
  configs/
    normalization_stability.yaml
  scripts/
    train_normalization_variants.py
    summarize_normalization_results.py

data/
  train_fadpo.jsonl
  eval_fadpo.jsonl
```

两个 JSONL 文件必须包含以下字段：

```text
id
prompt
chosen
rejected
topo_mask
margin
```

本工作包使用预先计算好的 `topo_mask`，因此该诊断不需要 ARU span 文件或 ARU score 文件。

## 服务器要求

安装或启用包含以下依赖的环境：

```text
torch
transformers
trl
peft
accelerate
datasets
pyyaml
```

同一依赖列表也写在 `requirements.txt` 中。尽量使用项目或服务器上已经可用的 CUDA-compatible PyTorch，不要在工作正常的训练服务器上盲目重装 PyTorch。

服务器还必须具备主 Fa-DPO 运行所用的 base model path 或 HuggingFace ID。在我们的服务器上，候选模型位于 `/home/models/`；请先检查该目录，并使用匹配的 II-Medical-8B/base checkpoint 路径。

## 运行前检查

从包含 `experiments/ablation/normalization_stability/` 的目录运行以下命令检查数据：

```bash
wc -l data/train_fadpo.jsonl
wc -l data/eval_fadpo.jsonl
head -n 1 data/train_fadpo.jsonl | jq 'keys'
```

当前工作包的预期行数：

```text
train_fadpo.jsonl: 32493
eval_fadpo.jsonl: 1711
```

可以编辑 `configs/normalization_stability.yaml`，也可以在运行时传入 `--base-model`。默认模型名只是占位符，除非它正好匹配服务器设置。建议先检查 `/home/models/`，例如运行 `ls /home/models`。

## 训练命令

推荐每个 job 只运行一个变体，这样失败和重启更容易管理。

```bash
cd /path/to/handoff/root

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  experiments/ablation/normalization_stability/scripts/train_normalization_variants.py \
  --config experiments/ablation/normalization_stability/configs/normalization_stability.yaml \
  --base-model /path/to/II-Medical-8B \
  --variant uniform
```

再分别替换为以下变体重复运行：

```text
no_normalization
original_normalized
capped_normalization
```

如果服务器稳定且显存充足，也可以用一个命令按顺序启动全部变体：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  experiments/ablation/normalization_stability/scripts/train_normalization_variants.py \
  --config experiments/ablation/normalization_stability/configs/normalization_stability.yaml \
  --base-model /path/to/II-Medical-8B \
  --variant all
```

## 训练输出

每个变体都会写出：

```text
experiments/ablation/normalization_stability/runs/{variant}/
  train_metrics.jsonl
  run_config.json
  trainer_eval_metrics.json
  adapter_config.json
  adapter_model.safetensors
```

关键稳定性日志是：

```text
runs/{variant}/train_metrics.jsonl
```

该文件应包含：

```text
step
loss
grad_norm
max_effective_weight
mean_effective_weight
high_risk_token_ratio
high_risk_effective_weight_median
high_risk_effective_weight_p95
high_risk_effective_weight_max
high_risk_effective_weight_count
nan_or_inf
```

## 下游评估

本工作包不包含下游 generation/evaluation runner。每个变体训练完成后，请使用项目评估流水线，在以下任务上评估保存的 adapter：

```text
MedBullets-op4
MedBullets-op5
MedXpertQA
MedQA
MedCaseReasoning process metrics
```

为了让 summarizer 汇总结果，请为每个变体写出以下文件：

```text
experiments/ablation/normalization_stability/runs/{variant}/eval_results.json
```

使用以下 schema：

```json
{
  "medbullets_op4_accuracy": 0.0,
  "medbullets_op5_accuracy": 0.0,
  "medxpertqa_accuracy": 0.0,
  "medqa_accuracy": 0.0,
  "support_rate": 0.0,
  "unsupported_medical_reasoning_rate": 0.0
}
```

未知值应直接省略字段，不要编造数字。

如果缺少 `eval_results.json`，summarizer 仍会报告训练稳定性字段，但 downstream accuracy 和 process metrics 会保持为 `TBD`。

## 汇总结果

训练日志和 `eval_results.json` 文件准备好后运行：

```bash
python experiments/ablation/normalization_stability/scripts/summarize_normalization_results.py
```

输出文件：

```text
experiments/ablation/normalization_stability/analysis/normalization_stability_results.csv
experiments/ablation/normalization_stability/analysis/normalization_stability_results.tex
experiments/ablation/normalization_stability/analysis/normalization_weight_diagnostics.csv
experiments/ablation/normalization_stability/analysis/normalization_weight_diagnostics.tex
experiments/ablation/normalization_stability/analysis/loss_curve.pdf
experiments/ablation/normalization_stability/analysis/grad_norm_curve.pdf
```

## 最终交付物

运行完成后请返回完整目录，尤其是以下文件：

```text
runs/*/train_metrics.jsonl
runs/*/trainer_eval_metrics.json
runs/*/eval_results.json
analysis/normalization_stability_results.csv
analysis/normalization_stability_results.tex
analysis/normalization_weight_diagnostics.csv
analysis/normalization_weight_diagnostics.tex
analysis/loss_curve.pdf
analysis/grad_norm_curve.pdf
```

在 `normalization_stability_results.csv` 的四行结果都填入真实数值之前，不要写入论文结论。

## 故障排查

- 如果模型加载失败是因为 `flash_attention_2` 不可用，请在 `configs/normalization_stability.yaml` 中设置 `attn_implementation: null` 或 `attn_implementation: sdpa`。
- 如果运行时显存不足，请一次只跑一个变体，降低 `--num_processes`，或减小 `per_device_train_batch_size` 并用 `gradient_accumulation_steps` 补偿。
- 如果没有安装 `jq`，可以跳过 `head ... | jq 'keys'` 检查，或用 Python 查看第一行 JSONL。
- 如果要从头重跑某个变体，请先删除 `runs/{variant}/` 下对应的旧目录。
