# ARU Decomposition And Retrieval Failure Taxonomy

这份文档总结了当前 `04_decompose_negatives_to_arus.py -> 05_attach_aru_evidence.py` 链路里最常见的失败模式，目标是把“样本里看起来不太对”的直觉，整理成后续 prompt、后处理和检索策略都能复用的迭代清单。

## 1. 总体判断

从已经抽查的 `medcase_unfaithful_negatives.arus.with_evidence.jsonl` 局部样本看，问题主要分成两个层面：

- `04` 把 reasoning 切成 ARU 时，`D` 节点容易不够原子、太病例化、或把多个鉴别方向塞进一个单元。
- `05` 检索时，如果 query 仍然沿用病例化整句，retriever 很容易只命中“主题相关”的文献，而不是支持该 ARU 判断的证据。

因此当前推荐的修复策略是“双端一起收紧”：

- 在 `04` 中让 `W/D` 同时产出更适合检索的 `retrieval_query`
- 在 `05` 中对旧 ARU 继续做 heuristic rewrite 和 query-variant 检索，作为兼容兜底

## 2. 失败模式

### F1. 一个 D 单元里有多个鉴别目标

- 症状
  - 同一句里同时出现两个或更多诊断方向，例如 “A and B were considered ...”
- 为什么会出问题
  - 一个 query 会把检索信号平均到多个疾病方向上，top evidence 往往只覆盖其中一半
- 常见后果
  - reranker 会偏向关键词更强的那一半
  - 另一半虽然被 reasoning 提到，但几乎没有有效支持证据
- 修复策略
  - `04` prompt 强制 “one D per alternative diagnosis”
  - `05` 对这类句子拆出多个 target-specific variant query
- 代表样本
  - `6823::F3 / aru0`

### F2. D 单元写成病例判决句，而不是可检索的医学命题

- 症状
  - 句子像 “X was excluded in this patient because ...”
  - 或 “but the patient had no ...”
- 为什么会出问题
  - 文献通常支持的是 “哪些发现支持/反对 X”，而不是“这个病人在此刻已被排除”
- 常见后果
  - retriever 只返回疾病概述，而不是支持排除逻辑的文献
- 修复策略
  - `04` 里保留原始 D 文本，但额外产出 decontextualized `retrieval_query`
  - `05` 从句子中抽 negative feature，改写成 `target + supporting/arguing-against finding`
- 代表样本
  - `6798::F2 / aru6`
  - `6859::F2 / aru7`
  - `6816::F3 / aru18`

### F3. 一个 D 单元同时混入诊断、依据和结论强度

- 症状
  - 单句里同时包含 “疾病名 + 依据 + ruled out / excluded / unlikely”
- 为什么会出问题
  - 这会把 “临床判断强度” 和 “可检索证据” 混在一起
  - 文献可能支持 “某 finding argues against X”，但不一定支持 “X is ruled out”
- 常见后果
  - evidence 语义上部分相关，但无法真正支撑结论强度
- 修复策略
  - `04` 让 D 更偏向 “某发现支持/反驳某鉴别对象”
  - `C` 节点单独承接最终结论强度
- 代表样本
  - `6870::F3 / aru7`

### F4. W 节点残留病例叙述，导致 query 不像知识命题

- 症状
  - W 里混有 chart narration、时序描述、患者特异的修饰语
- 为什么会出问题
  - 这类文本不稳定，也不利于命中“疾病-机制/表现”类知识文献
- 常见后果
  - 检到的是病例报告或宽泛主题文献，而不是教科书式支持
- 修复策略
  - `04` 明确要求 W 是 general medical rule
  - `05` 尝试抽取 warrant focus，生成更短的知识型 query
- 代表样本
  - 与下列正例形成对照时最明显：
  - `6810::F2 / aru9`
  - `6820::F1 / aru15`
  - `6812::F4 / aru2`

### F5. O / W / D 边界判断偏移

- 症状
  - 直接来自病例的事实被写成 W
  - 本应是 general rule 的内容被写成 O
- 为什么会出问题
  - `06` 对 O 会优先用 `patient_context` 验证，而 W/D 主要依赖外部证据
  - 一旦分类错了，后面的验证路由也会错
- 常见后果
  - 本来能被病例原文直接支持的节点，被迫去外部检索
  - 或者应该被外部知识审查的节点，被错误地当成病例事实
- 修复策略
  - `04` 在 user prompt 中显式传入 `patient_context`
  - prompt 内强调 O/W/D 的判别边界

### F6. Query 过长、带整句叙述，导致检索噪声上升

- 症状
  - query 几乎等于完整 ARU 句子
  - 包含长引号、破折号后的扩展描述、代词、病例特定限定
- 为什么会出问题
  - 长 query 对稀疏/密集检索都会引入不必要的匹配噪声
- 常见后果
  - 命中“同主题长文”，但不是最支持该命题的段落
- 修复策略
  - `04` 单独生成简短 `retrieval_query`
  - `05` 做 query cleaning，并限制 variant 的长度与数量

## 3. 正例特征

目前检索效果相对稳定的样本，通常具有这些共同点：

- `W` 是标准医学知识命题，而不是病例叙述
- query 只聚焦一个疾病、机制或关键特征
- 句子不包含过多修饰条件
- 证据需求更像教科书事实，而不是个案裁决

代表性正例：

- `6810::F2 / aru9`
- `6820::F1 / aru15`
- `6812::F4 / aru2`

## 4. 当前已落地的修复

### 4.1 Stage 04

- prompt 强制 `W/D` 原子化
- prompt 强制 `D` 一次只谈一个鉴别对象
- prompt 显式要求输出 `retrieval_query`
- prompt 通过 `patient_context` 区分 O/W/D/C
- 后处理会保留 `retrieval_query`，并在缺失时做保底清洗

### 4.2 Stage 05

- 优先使用 LLM 产出的 `retrieval_query`
- 对旧 ARU 使用 heuristic rewrite
- 为同一个 `W/D` 节点构造多个 query variant
- 多个 variant 的候选先合并，再统一 rerank
- 保留 `retrieval_query_variants` 和 `retrieval_query_source` 以便排查

## 5. 下一轮最值得看的指标

重跑 `04 -> 05` 之后，建议优先比较这些信号：

- 平均每个 `D` 的 `retrieval_query_variants` 数量
- `D` 节点 top-1 evidence 的人工 `good / partial / bad` 比例
- “多目标 D” 在新 ARU 中的占比是否明显下降
- 旧样本里那些典型 bad cases 是否能检到更贴近排除逻辑的证据

## 6. 建议的复查顺序

1. 先重跑 `04_decompose_negatives_to_arus.py`
2. 再重跑 `05_attach_aru_evidence.py`
3. 用 `sample_retrieval_review.py` 抽 `W/D` 样本
4. 人工或 LLM 审查 `retrieval_query -> candidates -> retained_evidence` 是否一致
