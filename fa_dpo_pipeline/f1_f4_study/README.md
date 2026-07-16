# F1-F4 可恢复实验

这个文件夹提供一套可直接落地的“多模型、多数据集、分阶段、可恢复”的 F1-F4 自然错误实验代码。

目标：

- 先跑通 `smoke`
- 再扩到 `pilot`
- 最后扩到 `final`
- 扩样本时不重复跑旧样本
- 生成、judge、统计分析全部接好

## 设计原则

### 1. 固定采样池，按前缀扩张

每个数据集先生成一个固定随机种子的 `sample_pool.json`。

之后：

- `smoke` 用前缀子集
- `pilot` 用更长的前缀
- `final` 再继续扩

所以 `smoke -> pilot -> final` 是严格增量的，不会换样本。

### 2. 主输出都是 master 文件

每个 `模型 × 数据集` 都维护一套 master 文件：

- `generations.master.jsonl`
- `faithfulness.master.jsonl`
- `faithfulness.master.summary.json`
- `faithfulness.master.report.md`

再次运行时：

- 生成脚本会按 `id` 跳过已经生成过的样本
- judge 脚本会按 `example_id` 跳过已经判过的样本

### 3. 阶段报告按当前 stage 样本子集重算

虽然 master 文件是累积的，但 `stage_reports/<stage>/` 会只取该 stage 对应的样本集合来重算统计，所以：

- `smoke` 报告只看 smoke 样本
- `pilot` 报告只看 pilot 样本
- `final` 报告只看 final 样本

## 文件说明

- [study_config.example.json](/root/med/fa_dpo_pipeline/f1_f4_study/study_config.example.json)
  - 配置模板
- [run_study.py](/root/med/fa_dpo_pipeline/f1_f4_study/run_study.py)
  - 总控脚本
- [analyze_stage_results.py](/root/med/fa_dpo_pipeline/f1_f4_study/analyze_stage_results.py)
  - 生成阶段级统计表和 Markdown 报告
- [study_lib.py](/root/med/fa_dpo_pipeline/f1_f4_study/study_lib.py)
  - 路径、采样池、配置解析等公共逻辑
- [mock_openai_server.py](/root/med/fa_dpo_pipeline/f1_f4_study/mock_openai_server.py)
  - 本地 OpenAI-compatible mock responder，用于验证整条流水线能否跑通
- [study_config.mock.json](/root/med/fa_dpo_pipeline/f1_f4_study/study_config.mock.json)
  - 极小样本 mock smoke 配置
- [run_manual_study.py](/root/med/fa_dpo_pipeline/f1_f4_study/run_manual_study.py)
  - 非 HTTP 的手工/assistant backend，总控通过 job 文件导出和结果回灌来跑实验
- [RESEARCH_MEMO.md](/root/med/fa_dpo_pipeline/f1_f4_study/RESEARCH_MEMO.md)
  - 当前阶段的重要结论、风险判断和后续实验口径

## 先准备配置

复制模板：

```bash
cp fa_dpo_pipeline/f1_f4_study/study_config.example.json \
  fa_dpo_pipeline/f1_f4_study/study_config.json
```

然后修改：

- `models`
- `judge_model`
- 闭源模型统一的 `base_url`
- 闭源模型统一的 `api_key`

如果本地部署模型，可以保留：

- `base_url=http://localhost:8000/v1`
- `api_key=EMPTY`

当前模板已经按你的网关统一成：

- 闭源 `base_url=https://api.key77qiqi.cn/v1`
- 闭源 `api_key=YOUR_CLOSED_SOURCE_API_KEY`

所以通常只需要把这个占位 key 统一替换掉即可。

## 推荐运行顺序

### 1. 先跑 smoke

```bash
python fa_dpo_pipeline/f1_f4_study/run_study.py \
  --config fa_dpo_pipeline/f1_f4_study/study_config.json \
  --stage smoke
```

### 2. 再扩到 pilot

```bash
python fa_dpo_pipeline/f1_f4_study/run_study.py \
  --config fa_dpo_pipeline/f1_f4_study/study_config.json \
  --stage pilot
```

### 3. 最后扩到 final

```bash
python fa_dpo_pipeline/f1_f4_study/run_study.py \
  --config fa_dpo_pipeline/f1_f4_study/study_config.json \
  --stage final
```

这三次运行会自动复用旧结果，只新增当前 stage 新扩进来的样本。

## Assistant 手工后端

如果你想让 assistant 本身参与回答，而不是走 OpenAI-compatible HTTP 服务，可以使用 `manual backend`。

它的工作方式是：

1. 脚本导出 generation jobs
2. assistant 按 job 文件写 `*.results.jsonl`
3. 脚本把 generation 结果导入 master 文件
4. 脚本导出 judge jobs
5. assistant 再写 judge 的 `*.results.jsonl`
6. 脚本把 judge 结果导入，并重算阶段报告

### 第一步：导出 generation jobs

```bash
python fa_dpo_pipeline/f1_f4_study/run_manual_study.py \
  --config fa_dpo_pipeline/f1_f4_study/study_config.json \
  --stage smoke \
  --phases prepare,sample,export_generate \
  --models gpt_5_4
```

job 文件会出现在：

```text
outputs/f1_f4_study/<study_name>/manual_jobs/<stage>/generation/<model_tag>/
```

### 第二步：assistant 写 generation results

assistant 需要为每个 `*.jobs.jsonl` 生成对应的：

- `*.results.jsonl`

每行一个 JSON，至少包含：

- `example_id`
- `output`

### 第三步：导入 generation 结果并导出 judge jobs

```bash
python fa_dpo_pipeline/f1_f4_study/run_manual_study.py \
  --config fa_dpo_pipeline/f1_f4_study/study_config.json \
  --stage smoke \
  --phases import_generate,export_judge \
  --models gpt_5_4
```

### 第四步：assistant 写 judge results

assistant 为每个 judge job 文件写对应的 `*.results.jsonl`，每行至少包含：

- `example_id`
- `dominant_error_type`
- `has_f1`
- `has_f2`
- `has_f3`
- `has_f4`
- `single_dominant_error`
- `answer_supported_by_reasoning`
- `clinical_plausibility_score`
- `confidence_score`
- `short_rationale`

### 第五步：导入 judge 结果并重算阶段报告

```bash
python fa_dpo_pipeline/f1_f4_study/run_manual_study.py \
  --config fa_dpo_pipeline/f1_f4_study/study_config.json \
  --stage smoke \
  --phases import_judge,analyze \
  --models gpt_5_4
```

这样就能在不依赖 HTTP 模型端点的情况下，让 assistant 自己参与整套实验。

## 本地 mock 测试

如果你想先不接真实大模型，而是只验证流程是否打通，可以先启动本地 mock responder：

```bash
python fa_dpo_pipeline/f1_f4_study/mock_openai_server.py \
  --host 127.0.0.1 \
  --port 18080 \
  --model-id mock-responder
```

然后另开一个终端跑：

```bash
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
python fa_dpo_pipeline/f1_f4_study/run_study.py \
  --config fa_dpo_pipeline/f1_f4_study/study_config.mock.json \
  --stage smoke
```

再扩到 `pilot` 验证“只新增，不重跑旧样本”：

```bash
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
python fa_dpo_pipeline/f1_f4_study/run_study.py \
  --config fa_dpo_pipeline/f1_f4_study/study_config.mock.json \
  --stage pilot
```

说明：

- 这个 mock 测试验证的是“代码链路是否打通”
- 它不代表真实模型质量
- 当前环境如果设置了代理变量，访问本地 `127.0.0.1` 端点时可能触发 `httpx` 的 SOCKS 依赖问题，所以示例命令里显式清掉了代理环境变量

## 只跑部分阶段

例如只做准备和采样：

```bash
python fa_dpo_pipeline/f1_f4_study/run_study.py \
  --config fa_dpo_pipeline/f1_f4_study/study_config.json \
  --stage smoke \
  --phases prepare,sample
```

例如只对已经生成好的结果重做阶段分析：

```bash
python fa_dpo_pipeline/f1_f4_study/run_study.py \
  --config fa_dpo_pipeline/f1_f4_study/study_config.json \
  --stage pilot \
  --phases analyze
```

## 只跑部分模型或数据集

```bash
python fa_dpo_pipeline/f1_f4_study/run_study.py \
  --config fa_dpo_pipeline/f1_f4_study/study_config.json \
  --stage smoke \
  --models gpt_5_4,qwen3_next_80b \
  --datasets medcase,medqa
```

模型支持用：

- `tag`
- 或 `id`

数据集筛选用：

- `key`

## 结果目录

默认写到：

```text
outputs/f1_f4_study/<study_name>/
```

核心结构：

```text
datasets/
  medcase/
    prepared.full.jsonl
    sample_pool.json
    stages/
      smoke.jsonl
      pilot.jsonl
      final.jsonl

runs/
  gpt_5_4/
    medcase/
      generations.master.jsonl
      faithfulness.master.jsonl
      faithfulness.master.summary.json

stage_reports/
  smoke/
    run_level_summary.csv
    model_level_summary.csv
    dataset_level_summary.csv
    stage_report.md
```

## 最后的统计分析产物

每个 stage 都会导出：

- `run_level_summary.csv`
- `model_level_summary.csv`
- `dataset_level_summary.csv`
- `accuracy_pivot.csv`
- `any_error_pivot.csv`
- `presence_f4_pivot.csv`
- `dominant_f4_pivot.csv`
- `overall.summary.json`
- `overall.report.md`
- `stage_report.md`

## 建议的实际用法

最稳的工作流是：

1. 先只跑 `smoke`
2. 检查输出格式和 judge 稳定性
3. 再跑 `pilot`
4. 趋势稳定后再跑 `final`

这套代码已经按这个节奏设计好了，不需要你手动管理“哪些样本已经跑过”。  
