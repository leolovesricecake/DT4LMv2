# DT4LM 改进实验完整指南

本文说明如何从数据预处理开始，依次完成模型训练、随机样本清单生成、SemDT
阈值标定、单项实验运行、自动评估和人工评估。除特别说明外，所有命令均从
`DT4LM/` 目录执行：

```bash
cd DT4LM
```

## 1. 安装环境

```bash
conda env create -f DT4LM.yaml
conda activate DT4LM
# Transformers 4.57 会安全地拒绝 torch<2.6 加载 pickle .bin 权重。
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

确认本地入口：

```bash
python -m textattack --help
python -m textattack semdt-calibrate --help
python -c "import torch; assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 6)"
```

OpenAI-compatible judge 需要 `openai`；本地 HF judge 需要足够显存。BLEU 和
METEOR 依赖 NLTK，BERTScore 依赖 `bert-score` 及显式配置的本地模型。

## 2. 准备数据

原仓库的数据预处理 notebook 已转换为统一脚本：

```bash
python datasets/preprocess_dataset.py sst2
python datasets/preprocess_dataset.py rte
python datasets/preprocess_dataset.py mrpc
python datasets/preprocess_dataset.py mr
```

默认输出：

```text
outputs/datasets/sst2
outputs/datasets/rte
outputs/datasets/mrpc
outputs/datasets/mr
```

GLUE 的公开 test split 没有有效标签，因此脚本沿用原 notebook 的思路，将有标签
数据以固定种子分层划分为 train、validation 和 test。四个数据集的冻结字段
协议为：

| 数据集 | 文本字段 | 任务 |
| --- | --- | --- |
| SST-2 | `sentence` | 二分类情感分析 |
| RTE | `premise`, `hypothesis` | 蕴含判断 |
| MRPC | `sentence1`, `sentence2` | 复述/语义等价判断 |
| MR | `text` | 二分类电影评论情感分析 |

manifest 准备和正式运行都会在加载受测模型前核对这些列。配置声明与本地数据
不一致时会立即失败，不能依靠 TextAttack 自动猜测字段。

显式指定输出路径：

```bash
python datasets/preprocess_dataset.py sst2 \
  --output-dir /data/dt4lm/sst2 \
  --seed 42 \
  --test-size 0.1 \
  --validation-size 0.1
```

输出目录已存在时脚本会拒绝覆盖。

## 3. 准备新旧模型

格式：`bash experiments/finetune/<file> <DATASET> <DEVICE>`

- `DATASET`：`sst2`、`rte`、`mrpc` 或 `mr`
- `DEVICE`：显卡号，`-1` 表示 CPU

```bash
bash experiments/finetune/train_albertbasev1.sh mrpc 1
bash experiments/finetune/train_albertbasev2.sh mrpc 1

bash experiments/finetune/train_debertav1base.sh mr 1
bash experiments/finetune/train_debertav3base.sh mr 1

bash experiments/finetune/train_gpt1.sh sst2 1
bash experiments/finetune/train_gpt2.sh sst2 1
```

checkpoint：

```text
outputs/finetuned/<model>_<dataset>/best_model
```

已有 checkpoint 时可跳过训练，直接修改目标实验 YAML 中：

```yaml
models:
  id: albertbasev1-v2
  old:
    name_or_path: outputs/albertbasev1_sst2/best_model
    revision: null
  new:
    name_or_path: outputs/albertbasev2_sst2/best_model
    revision: null
```

更换任一 checkpoint 或 revision 时必须更换 `models.id`，避免 run 与标定产物
混入旧模型对。

## 4. 选择一份完整实验配置

一个 YAML 表示一次完整实验，不再组合“数据集配置”和“方法配置”：

```text
experiments/improvements/configs/
  sst2/
    albertbasev1-v2-base.yaml
    albertbasev1-v2-static.yaml
    albertbasev1-v2-lexidt.yaml
    albertbasev1-v2-semdt-manual.yaml
    albertbasev1-v2-semdt-openai.yaml
    albertbasev1-v2-semdt-hf.yaml
    albertbasev1-v2-combined-openai.yaml
  rte/
    ...
  mrpc/
    ...
  mr/
    ...
```

每份配置都显式包含：

- 实验 ID、方法和随机种子；
- 数据路径、测试 split、抽样规模和 manifest 路径；
- `models.id` 及新旧 checkpoint；
- recipe、差分目标、语义约束和查询预算；
- NLI 模型及阈值来源；
- 可选标定后端及全部标定超参数；
- Success@B 预算和各质量指标配置。

### 4.1 测试样本数

```yaml
dataset:
  evaluation:
    split: test
    sample_size: 1000
    sample_seed: 765
```

`sample_size` 缺省、为 `null`、0 或负数时使用完整 split；正数时随机无放回
抽取至多该数量。首轮四个数据集都配置为 1000，数据不足时自动全量保留。
抽样不查询模型，也不筛选新旧模型共同预测正确的样本。

### 4.2 标定源样本数

SemDT 标定配置中：

```yaml
calibration:
  source_sampling:
    split: train
    sample_size: 500
    sample_seed: 765
```

该值语义与测试抽样一致。

### 4.3 BERTScore 本地模型

正式运行前必须把每份配置中的路径改为真实本地目录：

```yaml
evaluation:
  quality:
    bertscore:
      enabled: true
      model_name_or_path: /absolute/path/to/roberta-large
      num_layers: 17
      allow_remote_download: false
      device: cuda
      batch_size: 32
      idf: false
      rescale_with_baseline: false
      baseline_path: null
```

`allow_remote_download: false` 时不会隐式下载。暂时不计算 BERTScore 时，应在
该实验 YAML 中设置 `enabled: false`，而不是依赖命令行临时开关。

本地目录还必须满足以下二者之一：

- 包含 `model.safetensors` 或分片 `*.safetensors`，Transformers 会优先使用；
- 只有 `pytorch_model.bin` 时，运行环境必须使用 `torch>=2.6`。

`torch<2.6` 加载 `.bin` 会因 CVE-2025-32434 被新版 Transformers 拒绝。不要
降级 Transformers 或修改其安全检查；使用上面的官方 PyTorch 2.6 CUDA 11.8
安装命令，或把配置指向具有 safetensors 权重的等价 checkpoint。评估器会在
加载 BERTScore 模型前检查这一点，并把明确的修复说明写入 `quality.json`。
依据可查阅 [CVE 公告](https://github.com/advisories/GHSA-53q9-r3pm-6pq6)、
[Transformers 安全说明](https://github.com/huggingface/transformers/security) 和
[PyTorch 历史版本安装命令](https://docs.pytorch.org/get-started/previous-versions/)。

## 5. 生成随机样本 manifest

Base、Static 和 LexiDT 配置只生成 test manifest：

```bash
bash experiments/improvements/prepare_manifests.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-base.yaml
```

启用标定的配置还会生成独立的 train source manifest：

```bash
bash experiments/improvements/prepare_manifests.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-semdt-openai.yaml
```

产物位于：

```text
outputs/dt4lm-improvements/sample_sets/sst2/test-n1000-seed765.json
outputs/dt4lm-improvements/sample_sets/sst2/train-n500-seed765.json
```

sample manifest 与模型无关，只记录数据指纹、split、总体数量、请求/实际数量、
seed、抽样算法、固定索引顺序和 hash。同一数据集所有方法复用同一 test
manifest，从而支持逐样本成对比较。已有 manifest 内容不一致时程序拒绝覆盖；
应换新路径，不能手工修改冻结文件。

## 6. 配置 Judge

复制模板，并使用修正后的 `.secret.yaml` 后缀：

```bash
cp configs/semantic_judge.example.yaml configs/openai.secret.yaml
cp configs/semantic_judge.example.yaml configs/hf.secret.yaml
```

OpenAI-compatible 示例：

```yaml
backend: openai
openai:
  model: your-model
  base_url: https://your-endpoint/v1
  api_key: your-key
  timeout: 60
  max_retries: 3
  max_new_tokens: 32
```

本地 HF 示例：

```yaml
backend: hf
hf:
  model: /models/Qwen2.5-7B-Instruct
  revision: null
  device: cuda
  dtype: float16
  batch_size: 4
  max_retries: 3
  max_new_tokens: 32
```

两种后端天然独立，每次实验只使用一个。`*.secret.yaml` 和旧拼写
`*.secert.yaml` 均被 Git 忽略；新配置统一使用正确拼写。

## 7. 标定 SemDT 阈值

OpenAI 后端：

```bash
bash experiments/improvements/calibrate_semdt.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-semdt-openai.yaml
```

本地 HF 后端：

```bash
bash experiments/improvements/calibrate_semdt.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-semdt-hf.yaml
```

配置会显式控制候选源样本数、1000 条分层候选、800/200 冻结划分、0.01
网格步长、0.95 precision 下限、验证集最少 100 个语义保持正例和最多 2000
条补充标注。补充样本只用于冻结阈值审计，不参与重新调参。

## 8. 一次运行一个实验

命令只接收一份完整配置：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-base.yaml
```

其他方法独立运行，例如：

```bash
CUDA_VISIBLE_DEVICES=3 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-lexidt.yaml

CUDA_VISIBLE_DEVICES=1 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-semdt-openai.yaml
```

## 9. 检查实验产物

```text
outputs/dt4lm-improvements/runs/<dataset>/<models.id>/<experiment.id>/
  config.resolved.yaml
  provenance.json
  status.json
  sample_manifest.json
  results.jsonl
  attack_summary.json
  metrics/
    core.json
    success_queries.json
    quality.json
  successful_examples/
  failed_examples/
  skipped_examples/
```

`results.jsonl` 每个 manifest 样本恰好一行，`result_status` 只能是
`successful`、`failed` 或 `skipped`。`attack_summary.json` 和
`metrics/core.json` 都显式记录三类数量及 `total`。

只有原始输入已经满足“旧模型正确、新模型错误”时才会 skipped；其余三种
初始预测状态均进入搜索。每条记录保留：

- `initial_state` 和 `skip_reason`；
- 初始与最终的新旧模型输出；
- `model_pair_queries`、初始/搜索查询分解；
- 成功样本的 `queries_to_success`；
- 修改率、耗时、显存与可选 NLI 诊断。

`core.json` 包含 PaperGSR、完整样本生成率、Success@100/500/1000、SQ-AUC、
AMR、QPS、状态分布及 `resources`。`success_queries.json` 以等长的
`data.dataset_index` 和 `data.queries_to_success` 两列保存曲线原始数据，避免
`List[Dict]` 重复键名。本轮不绘制 success-query curve；以后可直接读取该文件。

四项质量指标集中在 `quality.json.metrics`，每项独立保存 `status`、配置、值或
错误。v2 的 `resources.json` 和四个单独质量文件不再生成。

QPS 口径为全部 manifest 样本产生的模型对查询总数除以成功生成数，失败和
skipped 的查询也进入分子；成功数为 0 时为 `null`。

## 10. 断点恢复、指标重算与结果整理

### 10.1 重新执行同一实验

再次执行同一份完整配置时，运行器会验证已有 `results.jsonl` 的样本数量、
manifest 顺序与三类状态。验证通过后跳过攻击，仅补算缺失、旧 schema 或失败的
指标；已有 v3 完成指标也会作为 checkpoint 复用。攻击相关配置改变时不会错误
复用旧结果，而会要求使用新的 `experiment.id`。

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-base.yaml
```

### 10.2 批量重算已拉取实验

`status.json` 分别记录 attack、core 和 quality 状态。BERTScore 或 NLTK 指标
失败不会删除攻击结果。升级 PyTorch 或准备 safetensors 模型后，可以删除旧
metrics 并批量重算；脚本会直接使用每个 run 内冻结的配置和 manifest：

```bash
RUN_ROOT=output/dt4lm-improvements/run
find "$RUN_ROOT" -type d -name metrics -prune -exec rm -rf {} +
CUDA_VISIBLE_DEVICES=1 python statistics/recompute_metrics.py --i "$RUN_ROOT"
```

不删除 metrics 也可以执行同一命令：完整 v3 质量子指标会复用，失败项会重试。
脚本逐一处理全部 run，最后集中报告失败项。只重算某一阶段可加
`--stage core` 或 `--stage quality`。

单个 run 也可以独立重跑质量评估：

```bash
python statistics/evaluate_improvements.py \
  --stage quality \
  --config <run-dir>/config.resolved.yaml \
  --results <run-dir>/results.jsonl \
  --manifest <run-dir>/sample_manifest.json \
  --output-dir <run-dir>/metrics \
  --status-file <run-dir>/status.json
```

### 10.3 汇总论文表格

指标全部完成后执行：

```bash
python statistics/aggregate_improvements.py
```

默认读取 `output/dt4lm-improvements/run`，并生成其中的 `summary.csv`。自定义
输入目录和文件名：

```bash
python statistics/aggregate_improvements.py \
  --i outputs/dt4lm-improvements/runs \
  --o first-round.csv
```

CSV 每个 run 一行，包含 dataset、model pair、method、seed，N/A/S/F/K、论文
核心指标、四项质量指标、资源与 NLI 诊断、模型/config/manifest 身份、运行状态
及关键依赖版本。整理器不计算相对 Base 指标；人工评估与标定报告仍使用各自的
专用分析脚本。

## 11. 轨迹审计

正式 SemDT run 完成后，可用相同完整配置审计其实际搜索轨迹：

```bash
bash experiments/improvements/calibrate_semdt.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-semdt-openai.yaml \
  outputs/dt4lm-improvements/runs/sst2/albertbasev1-v2/sst2-albertbasev1-v2-semdt-openai
```

该步骤只评估冻结阈值，不重新调参。

## 12. 人工评估

以 Base 与 SemDT 主结果为输入，按公共成功、Base 独有成功和 SemDT 独有成功
三层抽取 100 个原始样本：

```bash
python statistics/sample_human_evaluation.py \
  --base-results <base-run>/results.jsonl \
  --semdt-results <semdt-run>/results.jsonl \
  --manifest <base-run>/sample_manifest.json \
  --output outputs/dt4lm-improvements/human/sst2/reviews.jsonl \
  --key-output outputs/dt4lm-improvements/human/sst2/method_key.json \
  --sample-size 100 \
  --seed 765
```

只向评审者分发 `reviews.jsonl`。完成双人标注与冲突复核后：

```bash
python statistics/analyze_human_evaluation.py \
  --reviews outputs/dt4lm-improvements/human/sst2/reviews.jsonl \
  --key outputs/dt4lm-improvements/human/sst2/method_key.json \
  --base-core <base-run>/metrics/core.json \
  --semdt-core <semdt-run>/metrics/core.json \
  --output outputs/dt4lm-improvements/human/sst2/analysis.json \
  --bootstrap-samples 10000 \
  --seed 765
```

结果按各层实际占比估计语义保持率、ValidGSR 和 ValidPaperGSR，并报告分层
bootstrap 95% 置信区间、评审一致率和 Cohen's kappa。

## 13. MRPC/MR 首次运行

MRPC：

```bash
python datasets/preprocess_dataset.py mrpc
bash experiments/finetune/train_albertbasev1.sh mrpc 1
bash experiments/finetune/train_albertbasev2.sh mrpc 1
bash experiments/improvements/prepare_manifests.sh \
  experiments/improvements/configs/mrpc/albertbasev1-v2-base.yaml
CUDA_VISIBLE_DEVICES=1 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/mrpc/albertbasev1-v2-base.yaml
```

MR：

```bash
python datasets/preprocess_dataset.py mr
bash experiments/finetune/train_albertbasev1.sh mr 1
bash experiments/finetune/train_albertbasev2.sh mr 1
bash experiments/improvements/prepare_manifests.sh \
  experiments/improvements/configs/mr/albertbasev1-v2-base.yaml
CUDA_VISIBLE_DEVICES=1 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/mr/albertbasev1-v2-base.yaml
```

已有正确数据与 checkpoint 时，可跳过对应步骤。SemDT/OpenAI/HF 配置按第
6-8 节先生成 train manifest 并完成阈值标定。

## 14. 完成检查

- 28 份配置均能独立通过 schema 校验；
- SST-2/RTE/MRPC/MR 各方法分别共享同一 test manifest；
- manifest 不含模型预测或共同正确样本筛选；
- 每个 run 满足 `total = successful + failed + skipped`；
- `results.jsonl` 的索引和顺序与 manifest 完全一致；
- `queries_to_success` 只出现在成功记录中且不超过查询预算；
- BERTScore 使用真实本地路径，禁止下载时不会访问远端；
- `.bin` BERTScore checkpoint 使用 `torch>=2.6`，或直接使用 safetensors；
- 质量指标失败后攻击与核心产物仍可用；
- v3 metrics 仅有 `core.json`、`success_queries.json` 和 `quality.json`；
- `statistics/aggregate_improvements.py` 能生成每个 run 一行的汇总 CSV；
- 单 run 产物中没有 Base 路径、相对差值或扩展决策；
- Git 状态中不存在 secret 配置或 API key。
