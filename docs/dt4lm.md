# DT4LM 优化计划

## 原仓库分析

**总体判断**
它本质上是一个改造版 TextAttack：保留 TextAttack 的“`Attack = goal function + transformation + constraints + search method`”架构，在其上新增 `pair` recipe 和 `DifferentialClassification`，用于寻找“新模型预测错误、旧模型预测正确”的回归型差分输入。

**目录结构**
- `datasets/`：数据预处理 notebook，覆盖 SST-2、MR、RTE、MRPC；`adversarial-training/sample.ipynb` 用于抽样并混合差分输入做对抗训练。
- `experiments/`：实验脚本。
  - `finetune/`：训练 old/new model pair，并评估 clean accuracy。
  - `difftest/`：直接运行 DT4LM，例如 `textattack attack --recipe pair --base-recipe leap --model new --second-model old ...`。
  - `baseline/`：先对新模型生成普通 adversarial examples，再用旧模型复评筛差分输入。
- `statistics/`：修改率、质量评估、统计检验 notebook。
- `additional_results/`：论文未放入正文的补充 RQ1/RQ4 结果。
- `docs/`：Sphinx 文档配置。
- `textattack/`：核心代码包，是整个框架主体。

**核心代码依赖**
- CLI 入口在 DT4LM/textattack/commands/attack_command.py：解析参数后依次创建 dataset、model wrapper、attack，再交给 `Attacker.attack_dataset()`。
- 参数注册与工厂在 DT4LM/textattack/attack_args.py：维护 recipe、transformation、constraint、goal function 的名字到类映射；`pair` 映射到 `PAIR2024`。
- 模型加载在 DT4LM/textattack/model_args.py：支持 HuggingFace 名称、本地 checkpoint、TextAttack 内置模型；`--second-model` 是 DT4LM 模型对攻击的关键参数。
- 数据加载在 DT4LM/textattack/dataset_args.py：从 HuggingFace 或文件构造 `textattack.datasets.Dataset`。
- `Attack` 聚合器在 DT4LM/textattack/attack.py：保存目标函数、变换、约束和搜索算法，并把 `get_transformations`、`get_goal_results`、`filter_transformations` 注入搜索方法。
- DT4LM 的模型对 recipe 在 DT4LM/textattack/attack_recipes/pair_2024.py：先按 `--base-recipe` 构造普通攻击，再加载第二个模型，并把目标函数替换成 `DifferentialClassification`。
- 差分目标函数在 DT4LM/textattack/goal_functions/classification/differential_classification.py：成功条件是 `model1` 预测错误且 `model2` 预测正确；打分函数同时降低 `model1` 真标签概率、提高 `model2` 真标签概率。
- 双模型查询逻辑在 DT4LM/textattack/goal_functions/goal_function.py：若存在 `model2`，同一批候选文本会分别查询两个模型，再计算差分状态和分数。
- LEAP 搜索在 DT4LM/textattack/search_methods/leap.py：基于 population search、Levy 分布、局部/全局 elite 和扰动迭代找高分候选。
- 结果循环与保存逻辑在 DT4LM/textattack/attacker.py：逐样本攻击、记录成功/失败/跳过；成功样本会被保存为 HuggingFace `Dataset`，路径由模型目录和攻击方法组成。

**DT4LM 工作流程**
1. 数据准备：运行 `datasets/` 下 notebook，把 SST-2、MR、RTE、MRPC 等任务整理成 HuggingFace 数据集格式。
2. 模型更新对准备：用 `experiments/finetune/` 中脚本训练旧模型和新模型。README 强调同一 model pair 要保持相同超参，保证比较公平。
3. 直接差分测试：运行 `experiments/difftest/`。例如 `pair_leap_albertbasev2_sst2.sh` 会使用 `--model` 作为被攻击的新模型，`--second-model` 作为旧模型。
4. 构造攻击对象：`PAIR2024` 根据 `--base-recipe` 选 LEAP、FastGA、Kuleshov 等基础 recipe，继承其 transformation、constraints、search method，然后替换为差分目标函数。
5. 候选生成：以 LEAP 为例，使用 WordNet 同义词替换，约束为最大修改率 `0.16` 和停用词不可改，搜索算法用种群迭代生成候选文本。
6. 差分判定：对每个候选文本同时查询两个模型。成功条件是：新模型 `model1` 对真标签预测错误，旧模型 `model2` 对真标签预测正确。
7. 结果保存：成功的 perturbed input 和对应 original input 会被保存到模型输出目录附近的 `successful_examples/...`，字段会按任务类型组织为 `text/label`、`premise/hypothesis/label` 或 `question1/question2/label`。
8. baseline 流程：先用普通攻击 recipe 只攻击新模型，得到让新模型失败的 adversarial examples；再通过 DT4LM/textattack/commands/eval_model_command.py 让旧模型复评，只保留旧模型预测正确的样本作为差分输入。

**实现上的小观察**
`DifferentialClassification` 明确只支持分类任务；`lambda1/lambda2` 参数目前主要参与保存路径命名，实际打分中使用的是由两个模型输出动态计算的局部权重。另一个细节是，模型对模式下日志结果主要保存 `model1` 的 raw/display output，`model2` 主要参与成功判定和打分。  

## 目前存在问题
1. 原文有约 23% 的标签失真，语义保持能力不足。
2. 不适用于多分类。两个 lambda 都以 0.5 为边界，只在二分类成立。
3. 新旧模型的概率不一定可直接比较。比如：旧模型通常输出0.60—0.80；新模型通常输出0.90—0.99；此时简单相减会系统性偏向某个模型，目标值部分反映的是校准差异，而不是真正的决策边界距离。
4. 目标在直觉上效率不好。目前搜索时优化的是旧、新模型概率的线性组合，但这两者并不完全等价。例如两组候选：（0.99，0.60）、（0.70，0.51）。前者可能分更高，但后者在新模型上可能更容易跨过决策边界。
5. 查询成本高

