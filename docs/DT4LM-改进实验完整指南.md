# FF-PBS 实验完整指南

本指南对应 `plan-260802-ffpbs.md`。当前论文主方法是 FF-PBS，活动实验不再包含
SemDT、LexiDT、Static、AE-PBS 或自适应 epsilon 方法。

## 1. 运行位置

所有命令默认从 `DT4LM/` 目录执行：

```bash
cd DT4LM
```

实验使用一份完整 YAML 配置，不做配置继承，也不用一条命令运行整个矩阵。

## 2. 准备数据与模型

首先使用 `datasets/preprocess_dataset.py` 生成本地 Hugging Face Dataset。例如：

```bash
python datasets/preprocess_dataset.py sst2
python datasets/preprocess_dataset.py rte
python datasets/preprocess_dataset.py mrpc
python datasets/preprocess_dataset.py mr
```

默认输出在 `outputs/datasets/<dataset>/`。可以通过 `--output` 显式指定路径：

```bash
python datasets/preprocess_dataset.py sst2 \
  --output outputs/datasets/sst2
```

新旧模型必须已经针对同一任务完成微调，并在配置中指定：

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

`models.id` 是稳定的 model-pair 身份，决定实验输出命名空间。

## 3. 活动方法矩阵

每个 dataset/model-pair 有六份主实验/消融配置：

| 方法 | ranking | K | 不可行状态 |
| --- | --- | ---: | --- |
| Base | 原 DT4LM dynamic greedy | 1 | 由原目标间接处理 |
| Dynamic-Beam | `dynamic` | 5 | 不特殊处理 |
| FF-Pareto-Greedy | `feasibility_pareto` | 1 | `fill` |
| Hard-PBS | `feasibility_pareto` | 5 | `discard` |
| FF-MNew | `feasibility_mnew` | 5 | `fill` |
| FF-PBS | `feasibility_pareto` | 5 | `fill` |

`fill` 表示先保留旧模型预测正确的状态，只在 frontier 仍有空位时使用最小违反状态
补位。`discard` 表示查询后永久删除旧模型预测错误的 post-root 状态。

FF-MNew 与 FF-PBS 使用相同的 `fill` 政策，但可行候选只按 `m_new` 降序排序，
不使用修改率作为第二目标。

## 4. 生成配置

每个 model pair 只需先写好 `<model-pair>-base.yaml`，然后执行：

```bash
python experiments/improvements/generate_ffpbs_configs.py
```

生成器会：

1. 扫描 `experiments/improvements/configs/*/*-base.yaml`；
2. 为每个 Base 生成五个核心搜索对照和两个额外宽度配置；
3. 统一设置 `Success@100,200,...,1000`；
4. 检查文件名与 `models.id` 是否一致。

另外生成 `ff-pbs-k3` 和 `ff-pbs-k10`，与可复用为 `K=1` 的
FF-Pareto-Greedy 及主方法 `K=5` 共同覆盖参数敏感性实验。

当前四个数据集、ALBERT/GPT 两类 model pair 共有 64 份活动配置。新增 DeBERTa
model pair 时，添加对应 Base 配置后重新运行生成器即可。

## 5. 准备固定 manifest

同一 dataset 的所有 model pair 和方法使用同一份 test manifest：

```bash
bash experiments/improvements/prepare_manifests.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-base.yaml
```

`dataset.evaluation.sample_size` 的规则是：

- 缺省、`null` 或非正数：使用全部 test 样本；
- 正整数：按 `sample_seed` 随机抽取至多该数量；
- 测试集不足 1000 条时：使用全部样本。

manifest 只绑定 dataset/split/抽样结果，不绑定 model pair。

## 6. 运行单个实验

每次只运行一份完整配置：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-base.yaml

CUDA_VISIBLE_DEVICES=1 bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/sst2/albertbasev1-v2-ff-pbs.yaml
```

组件消融可独立运行：

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

本轮 schema 与历史结果不兼容。重跑前应先将旧的 `outputs/dt4lm-improvements/runs/` 整体移出
或删除，避免 Base/Dynamic-Beam 的旧目录被断点逻辑识别。

## 7. 产物协议

完整 run 目录为：

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

`results.jsonl` 使用 schema v4，每个 manifest 样本恰好一行。`result_status` 只能是：

- `successful`；
- `failed`；
- `skipped`。

`query_data.json` 使用等长列式数据，完整保存：

- `dataset_index`；
- `result_status`；
- `model_pair_queries`；
- `queries_to_success`；
- `budget_penalized_queries`。

该文件可以直接支持 Success-Query 曲线、QPS 和 BPQC 重算，当前阶段不强制画图。

## 8. 自动计算的指标

`metrics/core.json` 包含：

### 有效性与效率

- `paper_gsr = successful / attackable`；
- `sample_generation_rate = successful / total`；
- `model_pair_qps = 全部样本模型对查询总数 / successful`；
- `success_at_100` 至 `success_at_1000`；
- `success_query_auc`；
- `bpqc` 和 `normalized_bpqc`；
- `amr`。

### FF-PBS 机制

- `non_top1_path_rate`；
- `post_root_old_prediction_error_path_rate`；
- `recover_first_infeasible_depth_mean/median`；
- `recover_first_recovery_depth_mean/median`；
- `recover_depth_span_mean/median`；
- `infeasible_fill_event_rate`；
- `infeasible_retained_state_rate`；
- `hard_discard_rate`；
- `frontier_size_mean`、`rank1_size_mean`；
- `frontier_modified_set_diversity_mean`；
- `frontier_depth_diversity_mean`；
- `success_path_depth_mean/median`。

每个成功样本还会在 `results.jsonl` 的 `search_diagnostics.successful_path` 中保存紧凑列式
根到终点路径，供恢复深度分析和论文路径案例使用。

修改位置和深度多样性均是对每次 frontier 的 `unique_count / frontier_size` 先归一化，再在
所有 frontier 更新上取平均。

### 资源

`core.json.resources` 包含：

- `end_to_end_seconds`；
- `peak_vram_bytes`；
- `frontier_sort_seconds`；
- `frontier_sort_time_ratio`。

## 9. 重算指标

攻击已完成但 metrics 失败时，不需重跑攻击：

```bash
python statistics/recompute_metrics.py \
  --i outputs/dt4lm-improvements/runs \
  --stage all
```

`core` 和 `quality` 可以分开重算：

```bash
python statistics/recompute_metrics.py --stage core
python statistics/recompute_metrics.py --stage quality
```

## 10. 生成总表

```bash
python statistics/aggregate_improvements.py \
  --i outputs/dt4lm-improvements/runs \
  --o summary.csv
```

输出文件位于：

```text
outputs/dt4lm-improvements/runs/summary.csv
```

汇总器只接受 schema-v4 结果，不会尝试重命名或兼容历史方法。

## 11. 配对比较

Base 与 FF-PBS：

```bash
python statistics/compare_search_methods.py \
  --baseline outputs/dt4lm-improvements/runs/sst2/albertbasev1-v2/sst2-albertbasev1-v2-base \
  --candidate outputs/dt4lm-improvements/runs/sst2/albertbasev1-v2/sst2-albertbasev1-v2-ff-pbs \
  --o outputs/dt4lm-improvements/runs/sst2/albertbasev1-v2/ffpbs-vs-base.json
```

Hard-PBS 与 FF-PBS：

```bash
python statistics/compare_search_methods.py \
  --baseline outputs/dt4lm-improvements/runs/sst2/albertbasev1-v2/sst2-albertbasev1-v2-hard-pbs \
  --candidate outputs/dt4lm-improvements/runs/sst2/albertbasev1-v2/sst2-albertbasev1-v2-ff-pbs \
  --o outputs/dt4lm-improvements/runs/sst2/albertbasev1-v2/ffpbs-vs-hard.json
```

输出包含 McNemar、配对 bootstrap、Wilcoxon、BPQC 差、百分点 GSR 差以及 FF-PBS 独有成功的：

- `non_top1_rate`；
- `unique_post_root_old_prediction_error_path_rate`。

## 12. 人工评价

在一个单句任务和一个句子对任务上分别执行。生成盲评样本：

```bash
python statistics/sample_human_evaluation.py \
  --base-results <base-run>/results.jsonl \
  --ffpbs-results <ffpbs-run>/results.jsonl \
  --manifest <base-run>/sample_manifest.json \
  --output human/reviews.jsonl \
  --key-output human/key.json \
  --method-sample-size 100 \
  --unique-sample-size 50
```

每个 review 由两名评审者分别填写：

- `reviewer_1_label_preserved` 和 `reviewer_2_label_preserved`；
- `reviewer_1_semantic_preserved` 和 `reviewer_2_semantic_preserved`。

两名评审独立标注，不一致时在 `final_label_preserved` 或
`final_semantic_preserved` 填写裁决值；一致时可保持 `null`。完成后分析：

```bash
python statistics/analyze_human_evaluation.py \
  --reviews human/reviews.jsonl \
  --key human/key.json \
  --base-core <base-run>/metrics/core.json \
  --ffpbs-core <ffpbs-run>/metrics/core.json \
  --output human/analysis.json
```

结果包含 Base/FF-PBS 的 LPR、SPR、HVR、ValidGSR，FF-PBS 独有成功的 IVR，以及两个
判断维度各自的 Cohen's kappa 和 bootstrap 95% 置信区间。
