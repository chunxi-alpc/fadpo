# R4.1 计算成本与部署开销实验计划

## 1. 审稿问题

R4.1 担心 Fa-DPO 的 ARU decomposition、role-specific routing、NLI verification、risk scoring、token weighting 和 adaptive margin 会带来显著计算开销，尤其是部署时是否每个用户请求都要运行完整 verifier pipeline。

本计划只测计算成本，不负责证明 accuracy 或 clinical safety。

## 2. 要验证的边界

1. 昂贵的 ARU/NLI verifier pipeline 是一次性的 offline preference-data construction 成本。
2. DPO 训练阶段相对 Standard DPO 只多出 mask loading、token weighting 和 scalar margin。
3. Fa-DPO 部署推理时是普通 fine-tuned LLM，不需要在线 ARU parsing、NLI verification 或 risk scoring。
4. Verifier reranking 是更强但更贵的在线替代方案，因为每题需要 K 次生成和 verifier scoring。

正确表述：

> Fa-DPO has offline construction overhead and lightweight training-time arithmetic, but no verifier-related inference latency at deployment.

不要写：

> Fa-DPO has no computational overhead.

## 3. 所需输入

必须准备：

- fixed held-out prompt set：用于所有 inference latency 测量；
- Standard DPO-Retained-34K checkpoint；
- Fa-DPO-Retained-34K checkpoint；
- Base model checkpoint；
- verifier reranking 所需 parser、retriever、NLI verifier 配置；
- full candidate pool 或 fixed-size candidate subset；
- 训练日志或可复现实验脚本，用于 Standard DPO 与 Fa-DPO training-time 对比；
- 统一 decoding configuration、max input/output tokens 和 batch settings；
- hardware profile 记录模板。

如果缺少 full candidate pool，可以先报告 fixed subset 的 offline construction cost，但必须说明不能外推到完整构造成本。

## 4. 比较方法

保持 base model、tokenizer、prompt、decoding、max length、retrieval setting 和 evaluation split 一致。

| Method | Training data/objective | Inference mode | 部署时需要 verifier? |
|---|---|---|---|
| Base model | 无 DPO | 单次生成 | 否 |
| Standard DPO-Retained-34K | 34,204 retained pairs, vanilla DPO | 单次生成 | 否 |
| Fa-DPO-Retained-34K | 同一 retained pairs + token weighting + adaptive margin | 单次生成 | 否 |
| Standard DPO-Retained-34K + verifier reranking | 不新增训练 | K 次生成 + ARU/NLI scoring | 是 |

主 reranking 设置为 `K=4`；可选 `K=8`。

## 5. Offline construction 成本

对 full candidate pool 和一个固定大小 subset 记录：

| 指标 | 含义 |
|---|---|
| Candidate responses | 处理的 candidate negative 数 |
| Retained pairs | risk filtering 后保留数量 |
| Mean response tokens | 平均 token 长度 |
| Mean ARUs per response | 每条回复平均 ARU 数 |
| Evidence items per ARU | 每个 ARU 平均 routed evidence 数 |
| ARU parser calls | parser 调用次数 |
| NLI calls | premise-ARU NLI forward 次数 |
| Wall-clock time | 端到端构造时间 |
| GPU hours | parser/verifier GPU 时间 |
| CPU hours | routing/filtering/mask construction CPU 时间 |
| Peak memory | 可记录时写 GPU/CPU 峰值内存 |

## 6. Training-time 成本

Standard DPO-Retained-34K 和 Fa-DPO-Retained-34K 必须使用相同硬件、batch size、gradient accumulation、LoRA、optimizer、learning-rate schedule、max length 和 epoch。

记录：

- training wall-clock time；
- steps/s；
- tokens/s；
- GPU hours；
- peak GPU memory；
- 可选 data-loading time；
- 可选 loss-computation time。

相对训练开销：

\[
\text{overhead} =
\frac{T_{\text{Fa-DPO}}-T_{\text{Std-DPO}}}{T_{\text{Std-DPO}}}.
\]

## 7. Inference-time 成本

使用固定 held-out prompt set。每个方法使用相同 decoding 和硬件，正式计时前 warm up。

记录：

| 指标 | 含义 |
|---|---|
| Mean latency/question | 每题端到端耗时 |
| P50/P90/P95 latency | 延迟分位数 |
| Generated tokens/question | 平均输出 token 数 |
| Generations/question | Base/Standard/Fa-DPO 为 1，reranking 为 K |
| ARU parser calls/question | Fa-DPO deployment 应为 0，reranking 为 K 或更多 |
| NLI calls/question | Fa-DPO deployment 应为 0，reranking 与候选 ARU 数相关 |
| Verifier wall-clock time | 只对 reranking 在线 verifier 记录 |
| Total GPU hours | generation + verification GPU 时间 |

## 8. 硬件与软件环境

必须记录：

- GPU 类型和数量；
- CPU 型号和核心数；
- RAM；
- CUDA、driver、PyTorch、Transformers 版本；
- inference engine；
- precision；
- training/inference batch size；
- max input/output tokens；
- retrieval 是否属于基础 MQA 推理流程。

## 9. 结果表

### Table A. Offline construction cost

| Pool | Responses | Mean ARUs | NLI calls | Wall-clock time | GPU hours | Retained pairs |
|---|---:|---:|---:|---:|---:|---:|
| Full candidate pool | TBD | TBD | TBD | TBD | TBD | TBD |
| Fixed subset | TBD | TBD | TBD | TBD | TBD | TBD |

### Table B. Training overhead

| Method | Wall-clock time | Steps/s | Tokens/s | GPU hours | Peak GPU memory | Relative training time |
|---|---:|---:|---:|---:|---:|---:|
| Standard DPO-Retained-34K | TBD | TBD | TBD | TBD | TBD | 1.00x |
| Fa-DPO-Retained-34K | TBD | TBD | TBD | TBD | TBD | TBD |

### Table C. Inference-time deployment cost

| Method | Generations/question | ARU/NLI at deployment? | Mean latency | P95 latency | Generated tokens/question | Relative latency |
|---|---:|---|---:|---:|---:|---:|
| Standard DPO-Retained-34K | 1 | No | TBD | TBD | TBD | 1.00x |
| Fa-DPO-Retained-34K | 1 | No | TBD | TBD | TBD | TBD |
| Verifier reranking, K=4 | 4 | Yes | TBD | TBD | TBD | TBD |
| Verifier reranking, K=8 | 8 | Yes | TBD | TBD | TBD | TBD |

## 10. 解释规则

- 如果 Fa-DPO inference latency 与 Standard DPO 有差异，先检查输出长度、batching variance 和 decoding randomness；部署架构相同，不应归因于在线 verifier。
- 如果 verifier reranking 质量接近 Fa-DPO，应写成 quality-cost tradeoff：reranking 每题在线 verifier，Fa-DPO 把 verifier-derived risk amortize 到训练中。
- 没有固定硬件日志前，不在正文或回复信写具体成本数值。

## 11. 输出文件

建议保存：

- `offline_construction_timing.csv`
- `training_overhead.csv`
- `inference_latency.csv`
- `hardware_profile.md`
- `raw_logs/`

结果可用后再更新 Discussion cost subsection、appendix cost table 和回复信。
