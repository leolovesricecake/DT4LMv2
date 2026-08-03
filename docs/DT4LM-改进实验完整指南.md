# FF-PBS 实验完整指南

本指南对应 `docs/plan-260803-ffpbs.md`。当前主方法是 FF-PBS，E1/E2 的
系统级基线为 DT4LM-Kuleshov、DT4LM-LEAP 和 DT4LM-FastGA；E3-E5
的机制消融统一固定 Kuleshov 的变换与约束。

## 1. 运行位置

所有命令默认从 `DT4LM/` 目录执行：

```bash
cd DT4LM
```

每个 YAML 都是一份完整实验配置。配置之间不继承，每次只运行一份，
因此可以先用一个数据集和模型对试跑。

## 2. 准备数据与模型

使用预处理脚本生成本地 Hugging Face Dataset：

```bash
python datasets/preprocess_dataset.py sst2
python datasets/preprocess_dataset.py rte
python datasets/preprocess_dataset.py mrpc
python datasets/preprocess_dataset.py mr
```

默认输出为 `outputs/datasets/<dataset>/`。可显式设置路径：

```bash
python datasets/preprocess_dataset.py sst2 \
  --output outputs/datasets/sst2
```

新旧模型必须已在同一任务上完成微调，并在配置中写明：

```yaml
models:
  id: albertbasev1-v2
  old:
    name_or_path: outputs/finetuned/albertbasev1_sst2/best_model
    revision: null
  new:
    name_or_path: outputs/finetuned/albertbasev2_sst2/best_model
    revision: null
```

`models.id` 是 model pair 的稳定身份，决定运行产物的命名空间。manifest
只绑定 dataset/split/抽样结果，不绑定 model pair，因此同数据集的所有
模型对故意共用同一 manifest。

## 3. 活动方法矩阵

### 3.1 E1/E2 系统级比较

| 方法 | Recipe | 原生搜索 | 作用 |
| --- | --- | --- | --- |
| DT4LM-Kuleshov | `kuleshov_var` | 宽度 1 贪心 | 与 FF-PBS 最严格的直接基线 |
| DT4LM-LEAP | `leap` | LEAP 粒子群 | 现有通用多候选搜索 |
| DT4LM-FastGA | `faster-alzantot` | Alzantot GA | 现有通用种群演化搜索 |
| FF-PBS | `kuleshov_var` | 异步有界 frontier | 主方法，`K=5` |

三个 DT4LM 基线都使用 dynamic 差分目标，但各自保留原 Recipe 的文本变换、
约束和搜索状态机。LEAP/FastGA 与 FF-PBS 的比较是系统级比较，不能用于
单独归因某一搜索组件。

### 3.2 E3-E5 严格控制实验

| 方法 | ranking | K | 不可行状态 |
| --- | --- | ---: | --- |
| Dynamic-Beam | `dynamic` | 5 | 由 dynamic 标量统一排序 |
| FF-Pareto-Greedy | `feasibility_pareto` | 1 | `fill` |
| Hard-PBS | `feasibility_pareto` | 5 | `discard` |
| FF-MNew | `feasibility_mnew` | 5 | `fill` |
| FF-PBS | `feasibility_pareto` | 5 | `fill` |

`ff-pbs-k3` 和 `ff-pbs-k10` 用于宽度敏感性分析。此类方法都使用相同
Kuleshov 变换、约束、模型 batch size 和查询预算，才可以用于机制归因。

## 4. 生成和检查配置

```bash
python experiments/improvements/generate_ffpbs_configs.py
```

生成器会为每个 dataset/model-pair 生成 10 份完整配置：三个 DT4LM Recipe、
五个主实验/消融及两个额外宽度。当前 4 个数据集、2 个 model pair 共
80 份活动配置。旧 `<model-pair>-base.yaml` 会一次性迁移为
`<model-pair>-dt4lm-kuleshov.yaml`。

每份配置都显式写入：

- `attack.recipe` 与 `attack.recipe_parameters`；
- `attack.search`；
- `attack.query_budget: 1000`；
- `attack.model_batch_size: 32`；
- `Success@100,200,...,1000`。

### 4.1 Kuleshov GPT-2 路径

正式配置默认使用：

```yaml
fluency_model_name_or_path: /mnt/huawei/nsq/models/openai-community/gpt2
```

这是 Kuleshov 的流畅性约束模型，与正在测试的 ALBERT/GPT 新旧模型无关。运行前检查：

```bash
test -d /mnt/huawei/nsq/models/openai-community/gpt2
```

若实际路径不同，修改所有待运行 YAML 中的该字段。使用本地路径可避免
在实验启动时访问 `gpt2` 远程仓库。

### 4.2 LEAP WordNet

LEAP 使用 WordNet 同义词替换。离线运行前先下载 NLTK 资源：

```bash
python -m nltk.downloader wordnet omw-1.4
```

### 4.3 FastGA Learning-to-Write 模型

FastGA 的 `language_model_path: null` 表示使用 TextAttack 标准缓存；首次缺失时会
尝试下载。可先单独准备并查看返回路径：

```bash
python - <<'PY'
from textattack.shared.utils import download_from_s3

print(download_from_s3(
    "constraints/grammaticality/language-models/learning-to-write"
))
PY
```

严格离线环境中，将已准备的模型目录写到 FastGA YAML 的
`attack.recipe_parameters.language_model_path`。配置了路径但目录不存在时，
运行器会在加载新旧模型前报错。

## 5. 准备固定 manifest

同数据集的任意活动配置都可用来准备 manifest：

```bash
bash experiments/improvements/prepare_manifests.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-dt4lm-kuleshov.yaml
```

`dataset.evaluation.sample_size` 的规则是：

- 缺省、`null` 或非正数：使用全部 test 样本；
- 正整数：按 `sample_seed` 随机抽取至多该数量；
- 测试集小于指定数量：使用全部样本。

## 6. 逐个运行实验

先在一个设置上依次运行 E1/E2 的四个方法：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-dt4lm-kuleshov.yaml

CUDA_VISIBLE_DEVICES=1 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-dt4lm-leap.yaml

CUDA_VISIBLE_DEVICES=1 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-dt4lm-fastga.yaml

CUDA_VISIBLE_DEVICES=1 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-ff-pbs.yaml
```

机制消融可独立运行：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-dynamic-beam.yaml
CUDA_VISIBLE_DEVICES=1 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-ff-pareto-greedy.yaml
CUDA_VISIBLE_DEVICES=1 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-hard-pbs.yaml
CUDA_VISIBLE_DEVICES=1 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-ff-mnew.yaml
```

方法名和产物协议已变更。正式重跑前，将历史
`outputs/dt4lm-improvements/runs/` 整体移出或删除，避免断点逻辑拒绝在
同一 experiment id 下混用不同攻击配置。

## 7. 产物协议

```text
outputs/dt4lm-improvements/runs/<dataset>/<model_pair>/<experiment_id>/
  config.resolved.yaml
  provenance.json
  status.json
  sample_manifest.json
  results.jsonl
  attack_summary.json
  metrics/
    core.json
    query_data.json
    quality.json
```

`results.jsonl` 每个 manifest 样本恰好一行，`result_status` 只能是
`successful`、`failed` 或 `skipped`。每个非 skipped 样本记录：

- `model_pair_queries` 和 `queries_to_success`；
- wall-clock 与 peak VRAM；
- `recipe_diagnostics`：所有 Recipe 共用的候选规模统计；
- `search_diagnostics`：FF-PBS 异步 frontier 的专用机制统计。

`query_data.json` 以等长列式数据保存每个样本的状态、查询数、成功查询点
和预算惩罚查询数，可在后续直接绘制 Success-Query curve。

## 8. 自动计算的指标

`metrics/core.json` 包含：

- GSR、SGR、successful/failed/skipped；
- QPS、总模型对查询数、Success@B、Success-Query AUC；
- BPQC、normalized BPQC、AMR；
- 端到端时间、每成功耗时、峰值显存；
- BLEU、METEOR、ROUGE-L 和 BERTScore 单独保存在 `quality.json`。

所有 Recipe 共用的候选规模指标包括：

- `transformation_call_total/mean`；
- `generated_candidate_total/mean`；
- `constraint_filter_call_total/mean`；
- `constraint_filter_input_total/mean`；
- `constraint_passed_candidate_total/mean`；
- `candidate_constraint_pass_rate`；
- `generated_candidates_per_model_pair_query`。

`generated_candidate_*` 计数的是 transformation 产生、约束过滤前的原始候选；
GA crossover 等直接送入模型的状态体现在模型对查询数中。因此候选规模是
辅助的系统成本指标，不取代查询数。

FF-PBS 专用机制指标还包括 non-top1 路径率、暂时不可行路径率、恢复深度、
frontier 大小/多样性、不可行补位率、Hard-PBS 丢弃率和 Pareto 排序时间。

## 9. 断点恢复与指标重算

重复执行同一配置时：

- 完整 `results.jsonl` 存在且与 manifest/攻击身份一致：跳过攻击；
- core 完整：跳过 core；
- quality 不完整或失败：只重跑 quality。

攻击已完成但 metrics 失败时，可删除该 run 的 `metrics/` 后直接重新执行
原配置，也可批量重算：

```bash
python statistics/recompute_metrics.py \
  --i outputs/dt4lm-improvements/runs \
  --stage all
```

`--stage core` 和 `--stage quality` 可分别重算。

## 10. 生成论文总表

```bash
python statistics/aggregate_improvements.py \
  --i outputs/dt4lm-improvements/runs \
  --o summary.csv
```

输出位于 `outputs/dt4lm-improvements/runs/summary.csv`。除全部论文指标外，总表还包含
dataset、model pair、method、seed、recipe、序列化 recipe 参数、实际搜索算法、
manifest 指纹和运行环境时间戳。

## 11. 配对比较

DT4LM-Kuleshov 与 FF-PBS：

```bash
python statistics/compare_search_methods.py \
  --baseline outputs/dt4lm-improvements/runs/sst2/albertbasev1-v2/sst2-albertbasev1-v2-dt4lm-kuleshov \
  --candidate outputs/dt4lm-improvements/runs/sst2/albertbasev1-v2/sst2-albertbasev1-v2-ff-pbs \
  --o outputs/dt4lm-improvements/runs/sst2/albertbasev1-v2/ffpbs-vs-kuleshov.json
```

Hard-PBS 与 FF-PBS 仍可用同一脚本进行机制对照。配对输出包含成功列联表、
McNemar、bootstrap、Wilcoxon、GSR/BPQC 差异及 FF-PBS 独有成功的路径机制统计。

## 12. 人工评价

人工评价只比较 DT4LM-Kuleshov 与 FF-PBS，不扩展到 LEAP/FastGA。在一个
单句任务和一个句子对任务上分别生成盲评样本：

```bash
python statistics/sample_human_evaluation.py \
  --kuleshov-results <kuleshov-run>/results.jsonl \
  --ffpbs-results <ffpbs-run>/results.jsonl \
  --manifest <kuleshov-run>/sample_manifest.json \
  --output human/reviews.jsonl \
  --key-output human/key.json \
  --method-sample-size 100 \
  --unique-sample-size 50
```

每个 review 由两名评审者独立填写标签保持和语义保持判断，不一致时再填写
`final_label_preserved` 或 `final_semantic_preserved`。完成后执行：

```bash
python statistics/analyze_human_evaluation.py \
  --reviews human/reviews.jsonl \
  --key human/key.json \
  --kuleshov-core <kuleshov-run>/metrics/core.json \
  --ffpbs-core <ffpbs-run>/metrics/core.json \
  --output human/analysis.json
```

结果包含两种方法的 LPR、SPR、HVR、ValidGSR，FF-PBS 独有成功的 IVR，
两个判断维度的 Cohen's kappa 与 bootstrap 95% 置信区间。
