# DT4LM 改进实验完整指南

本文给出从环境准备、模型训练、manifest 冻结、SemDT 阈值标定、单项实验运行，
到自动统计和人工评估的完整流程。所有命令均从仓库中的 `DT4LM/` 目录执行：

```bash
cd DT4LM
```

## 1. 先理解三类配置

本实现将容易混淆的配置拆成三类。

1. 数据集配置：
   `experiments/improvements/configs/sst2.yaml` 和 `rte.yaml`。其中包含模型、
   数据集、采样规模、查询预算、NLI 推理参数和阈值搜索超参数。
2. 实验配置：`experiments/improvements/configs/experiments/*.yaml`。每个文件
   只定义一个实验的基础 recipe、差分目标、语义约束和阈值来源。
3. Judge 私密配置：`configs/*.secert.yaml`。一次配置只选择一个 OpenAI
   Responses 或本地 HF 后端；这类文件被 git 忽略。

`run_first_round.sh` 一次只运行“一个数据集配置 + 一个实验配置”，不会再自动
遍历实验矩阵。OpenAI 与 HF 标定同样分开执行，一个后端失败不会带坏另一个。

## 2. 安装环境

推荐用仓库环境文件创建独立 Conda 环境：

```bash
conda env create -f DT4LM.yaml
conda activate DT4LM
pip install -e .
```

确认本地代码入口和关键依赖可用：

```bash
python -m textattack --help
python -m textattack semdt-calibrate --help
python -c "import torch, transformers, datasets, yaml; print(torch.__version__)"
```

OpenAI-compatible 标注还需要 `openai==2.45.0`；自动质量指标需要 NLTK 和
`bert-score`。本地 HF judge 的显存需求取决于所选模型和 dtype。

## 3. 准备数据与新旧模型

SST-2 与 RTE 都使用 ALBERT base v1 作为旧模型、ALBERT base v2 作为新模型。
原仓库四个监督数据 notebook 已转换为
`datasets/preprocess_dataset.py`。该脚本保留 notebook 的标签过滤、固定种子
和划分方式，但默认使用本地 `save_to_disk`，不要求创建或上传个人
Hugging Face 仓库。

### 3.1 预处理 SST-2 和 RTE

从 `DT4LM/` 目录直接执行：

```bash
python datasets/preprocess_dataset.py sst2
python datasets/preprocess_dataset.py rte
```

默认输出为：

```text
outputs/datasets/sst2
outputs/datasets/rte
```

SST-2 和 RTE 的 Hub test split 没有有效标签，因此脚本按照原 notebook
合并有标签的 train/validation 数据，再以种子 42 分层划分为 80% train、
10% validation、10% test。RTE 还会把 `sentence1/sentence2` 重命名为
`premise/hypothesis`。输出目录中的 `preprocessing_metadata.json` 记录数据
来源、种子、比例、转换规则和各 split 实际数量。

可以显式设置输出路径、数据 revision 和划分参数：

```bash
python datasets/preprocess_dataset.py sst2 \
  --output-dir /data/dt4lm/sst2 \
  --revision <dataset-revision> \
  --seed 42 \
  --test-size 0.1 \
  --validation-size 0.1
```

输出目录已存在时脚本会拒绝覆盖，应指定新的 `--output-dir`。只有确实需要共享数据时才使用可选的 `--push-to-hub <namespace/repository>`；本地训练不需要Hub 账号或 token。

其余原始 notebook 使用同一入口：

```bash
python datasets/preprocess_dataset.py mrpc
python datasets/preprocess_dataset.py mr
```

默认分别输出 `outputs/datasets/mrpc` 和 `outputs/datasets/mr`。MRPC 沿用原 notebook 的有效标签与划分判断，MR 则对完整 train/validation/test 分别过滤并以种子 42 打乱。

### 3.2 训练新旧模型

四个首轮训练脚本已经把上述默认数据目录作为默认输入，不再需要填写数据路径：

```bash
bash experiments/finetune/train_albertbasev1_sst2.sh
bash experiments/finetune/train_albertbasev2_sst2.sh
bash experiments/finetune/train_albertbasev1_rte.sh
bash experiments/finetune/train_albertbasev2_rte.sh
```

四个脚本默认分别写入：

```text
outputs/albertbasev1_sst2/best_model
outputs/albertbasev2_sst2/best_model
outputs/albertbasev1_rte/best_model
outputs/albertbasev2_rte/best_model
```

若预处理时使用了自定义目录，可把数据路径和模型输出目录作为前两个参数：

```bash
bash experiments/finetune/train_albertbasev1_sst2.sh \
  /data/dt4lm/sst2 \
  /data/dt4lm/models/albertbasev1_sst2
```

此时还要同步修改数据集 YAML 中的 `dataset.path`、
`dataset.textattack_spec` 和 `models.old/new`。默认配置已经指向
`outputs/datasets/sst2`、`outputs/datasets/rte` 及四个默认模型目录。

若已有 checkpoint，可跳过训练，只需把数据集 YAML 中 `models.old` 和
`models.new` 改成实际路径，并为这个有序模型对设置稳定的
`models.id`：

```yaml
models:
  id: albertbasev1-albertbasev2
  old: outputs/albertbasev1_sst2/best_model
  new: outputs/albertbasev2_sst2/best_model
  old_revision: null
  new_revision: null
```

`models.id` 只允许字母、数字、`.`、`_` 和 `-`，用于隔离 manifest、
标定与正式运行产物。更换任一 checkpoint 或 revision 时必须使用新
ID，不要复用原模型对的目录。两个模型必须使用相同标签空间和
标签映射。

### 3.3 对抗训练 notebook

原 `datasets/adversarial-training/sample.ipynb` 的两个阶段已转换为
`datasets/prepare_adversarial_training.py`。先从原训练集抽取默认 10%：

```bash
python datasets/prepare_adversarial_training.py sample \
  --source outputs/datasets/sst2
```

生成差分输入后，将其与原训练集混合：

```bash
python datasets/prepare_adversarial_training.py combine \
  --adversarial-source <generated-differential-dataset> \
  --original-source outputs/datasets/sst2
```

两个阶段均支持 `--output-dir`；默认分别写到
`outputs/datasets/adversarial-training-sample` 和
`outputs/datasets/adversarial-training-combined`。该流程属于后续对抗训练，完成首轮 SemDT/LexiDT 实验不需要执行。

## 4. 设置采样与校准超参数

数据集 YAML 中以下配置决定 manifest 规模：

```yaml
sampling:
  test:
    strategy: all
  calibration_originals:
    strategy: random_up_to
    size: 500
```

支持三种通用策略：

- `all`：使用全部新旧模型均预测正确的样本，不设置 `size`；
- `random_exact`：固定抽取 `size` 条，不足时明确失败；
- `random_up_to`：最多抽取 `size` 条，不足时使用全部合格样本。

当前 SST-2 和 RTE 配置均对预处理后的 test split 使用 `all`；两个数据集的
训练标定原始样本上限均为 500。若需限制正式测试规模，可把 `test` 改为
`random_exact` 并设置 `size`。这里的数量是 YAML 超参数，不是代码常量。

阈值标定由以下字段控制：

```yaml
calibration:
  candidate_collection:
    base_recipe: kuleshov_var
    differential_objective: dynamic
    semantic_constraint: original
  candidate_sample_size: 1000
  search_sample_size: 800
  minimum_validation_positives: 100
  maximum_total_labels: 2000
  trajectory_sample_size: 100
  annotation_batch_size: 32
  threshold_search:
    method: grid
    step: 0.01
    min_precision: 0.95
    bootstrap_samples: 10000
```

正式设置先分层抽取 1000 个候选，冻结为 800 条搜索集和 200 条验证集；网格
搜索以 0.01 为步长，在 precision 不低于 0.95 时最大化 recall。若验证集
语义保持正例少于 100 条，则追加独立审计样本，但总标注量不超过 2000；
补充样本不会参与调阈值。

## 5. 冻结 train/test manifest

分别生成两个数据集的固定样本清单：

```bash
bash experiments/improvements/prepare_manifests.sh \
  experiments/improvements/configs/sst2.yaml

bash experiments/improvements/prepare_manifests.sh \
  experiments/improvements/configs/rte.yaml
```

检查以下产物：

```text
outputs/dt4lm-improvements/manifests/<dataset>/<models.id>/train_manifest.json
outputs/dt4lm-improvements/manifests/<dataset>/<models.id>/test_manifest.json
outputs/dt4lm-improvements/manifests/<dataset>/<models.id>/manifest_metadata.json
```

配置加载时会强制检查三个路径的目录末尾为
`<dataset>/<models.id>`。

`manifest_metadata.json` 会记录 `models.id`、生成配置哈希、完整生成参数和
实际数量；manifest 自身还记录新旧模型 ID/revision 与数据指纹。
已有产物只有在内容完全一致时才会被复用；任一身份或生成参数变化都会
拒绝覆盖，此时应修改 `models.id` 及对应路径。同一数据集和模型对的
所有方法必须读取同一 test manifest；GSR、Success@B 和人工评估均以
其中 `sample_count` 为分母。

## 6. 配置一个 Judge 后端

模板位于 `configs/semantic_judge.example.yaml`。OpenAI-compatible 配置示例：

```yaml
backend: openai
openai:
  model: deepseek-v4-pro
  base_url: https://api.deepseek.com
  api_key: <your-key>
  timeout: 60
  max_retries: 3
  max_new_tokens: 32
```

本地 HF 配置示例：

```yaml
backend: hf
hf:
  model: Qwen/Qwen2.5-7B-Instruct
  revision: null
  device: cuda
  dtype: float16
  batch_size: 4
  max_retries: 3
  max_new_tokens: 32
```

建议分别使用 `configs/openai.secert.yaml` 和 `configs/hf.secert.yaml`。
`.gitignore` 已忽略 `**.secert.yaml`；不要把 API key 写入数据集配置、实验
配置或命令行。每次标定只读取一个 judge 配置。

## 7. 独立标定 SemDT 阈值

先运行需要作为主结果的 OpenAI-compatible 后端：

```bash
bash experiments/improvements/calibrate_semdt.sh \
  experiments/improvements/configs/sst2.yaml \
  configs/openai.secert.yaml
```

本地 HF 标定是另一条独立命令：

```bash
bash experiments/improvements/calibrate_semdt.sh \
  experiments/improvements/configs/sst2.yaml \
  configs/hf.secert.yaml
```

对 RTE 将第一个参数替换为 `rte.yaml`。首次标定会依次完成 Base 候选收集、
离线双向 NLI 打分和固定划分；后续后端复用同一候选与划分，但标签和阈值写入
各自目录。首次创建共享候选时不要并行启动两个后端。

关键产物：

```text
outputs/dt4lm-improvements/calibration/<dataset>/<models.id>/split/split_manifest.json
outputs/dt4lm-improvements/calibration/<dataset>/<models.id>/<backend>/threshold.json
outputs/dt4lm-improvements/calibration/<dataset>/<models.id>/<backend>/validation_report.json
outputs/dt4lm-improvements/calibration/<dataset>/<models.id>/<backend>/supplemental_report.json
```

标注文件采用追加式写入，可在 API 中断后用同一命令恢复。已完成的候选 ID
不会重复标注。若搜索集不存在满足 precision 下限的阈值，标定会明确失败，
不会回退到人工阈值。

## 8. 一次运行一个实验

通用命令为：

```bash
bash experiments/improvements/run_first_round.sh \
  <dataset-config.yaml> \
  <experiment-config.yaml>
```

建议先运行 Base，以便后续 summary 自动计算相对基线指标：

```bash
bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2.yaml \
  experiments/improvements/configs/experiments/base.yaml
```

然后按需单独运行任一实验，例如：

```bash
# Static
bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2.yaml \
  experiments/improvements/configs/experiments/static.yaml

# LexiDT
bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2.yaml \
  experiments/improvements/configs/experiments/lexidt.yaml

# SemDT 人工阈值
bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2.yaml \
  experiments/improvements/configs/experiments/semdt-manual.yaml

# SemDT OpenAI 标定阈值
bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2.yaml \
  experiments/improvements/configs/experiments/semdt-openai.yaml

# SemDT 本地 HF 标定阈值
bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2.yaml \
  experiments/improvements/configs/experiments/semdt-hf.yaml

# 可选 Combined 兼容性检查
bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2.yaml \
  experiments/improvements/configs/experiments/combined.yaml
```

RTE 同理，只替换第一个配置参数。若当前只想验证攻击链而不下载 BERTScore
模型，可在命令末尾加 `--skip-bertscore`。

每个实验配置明确指定：

```yaml
base_recipe: kuleshov_var
differential_objective: dynamic  # dynamic/static/lexi
semantic_constraint: nli         # original/nli
semantic_threshold:
  source: calibrated             # none/manual/calibrated
  backend: openai
  file: null
```

`file: null` 时自动读取
`calibration.output_root/<backend>/threshold.json`；也可以显式填写阈值文件。
一个 run 目录已有 `results.jsonl` 时，运行器会拒绝覆盖。需要重跑时应使用新的
`experiment.name` 或新的数据集 `output_root`，并保留旧产物供比较。

## 9. 轨迹分布偏移审计

SemDT 正式运行完成后，用相同 judge 后端追加一次冻结阈值审计：

```bash
bash experiments/improvements/calibrate_semdt.sh \
  experiments/improvements/configs/sst2.yaml \
  configs/openai.secert.yaml \
  outputs/dt4lm-improvements/runs/sst2/albertbasev1-albertbasev2/semdt-openai
```

结果写入：

```text
outputs/dt4lm-improvements/calibration/sst2/albertbasev1-albertbasev2/
  openai/trajectory_audits/semdt-openai/report.json
```

该步骤按 YAML 中 `trajectory_sample_size` 分层抽样，只评估冻结阈值在实际
SemDT 搜索轨迹上的 precision、recall、接受率和 NLI 分数偏移，不重新调参。

若两个后端都已标注同一验证集，可单独生成描述性一致率：

```bash
python -m textattack semdt-calibrate \
  --stage judge-agreement \
  --left-labels outputs/dt4lm-improvements/calibration/sst2/albertbasev1-albertbasev2/openai/validation_labels.jsonl \
  --right-labels outputs/dt4lm-improvements/calibration/sst2/albertbasev1-albertbasev2/hf/validation_labels.jsonl \
  --output outputs/dt4lm-improvements/calibration/sst2/albertbasev1-albertbasev2/judge_agreement.json
```

一致率只用于报告，不能混合两个后端的标签或阈值。

## 10. 查看自动评估

每次正式运行结束后会自动生成：

```text
outputs/dt4lm-improvements/runs/<dataset>/<models.id>/<experiment>/
  config.yaml
  environment.json
  sample_manifest.json
  results.jsonl
  attack_summary.json
  summary.json
  successful_examples/
  failed_examples/
  nli_candidates.jsonl
  nli_profile.json
```

最后两个文件只在 NLI 实验中存在。`summary.json` 包含：

- EligibilityRate 和 manifest 实际样本数；
- Perturbation-induced GSR；
- Success@100/500/1000；
- 成功样本上的 AMR；
- BLEU、METEOR、ROUGE-L 和 BERTScore；
- 模型对查询总数与论文口径 QPS；
- 端到端时间、每成功样本时间、峰值显存和 NLI profile；
- 相对 Base 的百分点差、相对差和等效判定。

QPS 使用已经确认的论文口径：

\[
\mathrm{QPS} =
\frac{\text{全部合格样本产生的模型对查询总数}}
{\text{成功生成数}}.
\]

失败和预算耗尽样本的模型查询也进入分子；成功数为 0 时报告 `null`，同时保留
查询总数。NLI 推理不计入模型对查询，但在资源指标中单独报告。

## 11. 生成人工评估任务

每个数据集以 Base 和首轮主结果 `SemDT-openai` 为输入，按公共成功、
Base 独有成功和 SemDT 独有成功三层的实际占比分层抽取 100 个原始样本：

```bash
mkdir -p outputs/dt4lm-improvements/human/sst2/albertbasev1-albertbasev2
python statistics/sample_human_evaluation.py \
  --base-results outputs/dt4lm-improvements/runs/sst2/albertbasev1-albertbasev2/base/results.jsonl \
  --semdt-results outputs/dt4lm-improvements/runs/sst2/albertbasev1-albertbasev2/semdt-openai/results.jsonl \
  --manifest outputs/dt4lm-improvements/runs/sst2/albertbasev1-albertbasev2/base/sample_manifest.json \
  --output outputs/dt4lm-improvements/human/sst2/albertbasev1-albertbasev2/reviews.jsonl \
  --key-output outputs/dt4lm-improvements/human/sst2/albertbasev1-albertbasev2/method_key.json \
  --sample-size 100 \
  --seed 765
```

评审期间只分发 `reviews.jsonl`，不要分发 `method_key.json`。两位评审者分别
填写 `reviewer_1_a/b` 和 `reviewer_2_a/b` 布尔值；意见不一致时，经复核填写
`final_a/b`。若两人一致，分析器会自动采用一致结论。

完成标注后运行：

```bash
python statistics/analyze_human_evaluation.py \
  --reviews outputs/dt4lm-improvements/human/sst2/albertbasev1-albertbasev2/reviews.jsonl \
  --key outputs/dt4lm-improvements/human/sst2/albertbasev1-albertbasev2/method_key.json \
  --output outputs/dt4lm-improvements/human/sst2/albertbasev1-albertbasev2/analysis.json \
  --bootstrap-samples 10000 \
  --seed 765 \
  --base-summary outputs/dt4lm-improvements/runs/sst2/albertbasev1-albertbasev2/base/summary.json \
  --semdt-summary outputs/dt4lm-improvements/runs/sst2/albertbasev1-albertbasev2/semdt-openai/summary.json
```

RTE 使用独立目录重复相同步骤。结果按各层实际总体占比估计语义保持率和
ValidGSR，并以原始样本为单位执行分层 bootstrap，给出 95% 置信区间、
评审一致率和 Cohen's kappa。

## 12. 完成检查表

正式汇总前逐项确认：

- 新旧模型 checkpoint、类别数和标签映射一致；
- `models.id` 与实际有序模型对一致，三类产物均在对应命名空间；
- manifest 元数据中的生成参数、配置哈希和实际数量正确；
- 所有方法使用同一 test manifest；
- 每个实验目录只对应一个实验 YAML；
- OpenAI 与 HF 标签和阈值目录彼此独立；
- 阈值报告记录 `grid`、`0.01`、`min_precision: 0.95`；
- 补充审计未参与阈值搜索；
- SemDT 主结果完成 100 条实际轨迹审计；
- QPS 分母是成功生成数，分子含全部合格样本查询；
- 人工评估按三层实际占比加权，并报告 bootstrap 95% 置信区间；
- GSR 使用 1 个百分点、AMR/QPS 使用 5% 相对变化进行等效判定；
- `git status` 中不存在任何 `*.secert.yaml` 或 API key。

遇到问题时优先检查当前 run 的 `config.yaml`、`environment.json`、
`attack_summary.json`，以及 calibration 目录中的 split manifest、
annotation JSONL 和 validation report。不要通过直接修改冻结产物来修复配置
错误；建立新的输出路径并重新生成对应阶段，才能保持实验可追溯。
