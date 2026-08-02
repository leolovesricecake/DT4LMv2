# AE-PBS 实施与实验执行方案

> 本文是 `plan-260802-ae.md` 的代码落地方案。实施时以本文对接口、边界条件、产物协议和验收标准的定义为准，研究动机与数学方法仍以原计划为准。

> 结果后修订：首轮所谓 Strict-PBS 实际采用“可行状态优先、空位由不可行状态
> 填充”的政策，现更名为 Feasibility-First PBS。真正的 Strict-PBS 在 root
> 首次扩展后永久丢弃旧模型预测错误的候选。两者通过
> `epsilon.infeasible_state_policy` 显式区分。

## 1. 方向理解与结论

AE-PBS 不再尝试设计更复杂的单一标量分数，而是改变差分样本的搜索方式：

1. 将旧模型保持正确视为随查询预算逐步收紧的 epsilon 约束。
2. 在当前 epsilon 可行域内，将最大化新模型错误 margin 和偏好低修改率作为两个 Pareto 排序方向。
3. 使用有界异步 frontier 保留多条不同深度的候选路径，每次仅扩展一个状态。
4. 搜索早期可以保留轻度破坏旧模型正确性的路径，但最终成功判定仍严格为“旧模型正确、新模型错误”。

该方法能够作为新模块兼容接入。实现应只在配置显式选择 AE-PBS 时替换搜索器，不改变以下现有能力：

- Kuleshov 词替换和原有约束链。
- 新旧模型加载与 logits/probabilities 输出协议。
- 数据集、model pair、manifest 和样本顺序协议。
- `dynamic`/`static`/`lexi` 目标和当前贪心搜索。
- Original/SemDT 语义约束。
- 成功、失败、跳过的三态统计及现有质量评估。
- 成功样本作为后续对抗训练数据的使用方式。

### 1.1 评审意见处理记录

`reply-260802-ae.md` 中的意见按以下方式处理：

**直接采纳：**

- 将“模型查询去重”与“搜索状态去重”分离，相同文本可复用模型输出，但路径状态不一定合并。
- 明确 AE-PBS 只在搜索中偏好低修改成本，不声称找到全部可达成功输入中的全局最小修改。
- 增加 epsilon 相对原始 old margin 的归一化诊断，并保存首个扩展的 old margin 分布摘要。
- 补齐 Paper GSR、sample generation rate、Success@B、sample-Success@B、QPS 和 AUC 的明确分母。
- 在共同成功样本查询比较之外，增加全部 attackable 配对样本上的预算惩罚查询成本。
- 区分模型微调 seed、manifest 抽样 seed、攻击 seed 和 tie-break seed；确定性攻击不为了凑重复数而更换无效 seed。

**调整后采纳：**

- 采纳“首轮无违反时不应立即永久退化”的判断，但不在预热期改用 Dynamic-Beam。实施为有界延迟初始化：在最多两次扩展中，首次观察到负 old margin 就立即确定 `epsilon_0`；尚未观察到违反时按 Feasibility-First PBS 排序。这样能发现第二层才出现的违反，又不引入额外的 dynamic 排序混杂。
- 主实验只保存首个扩展 margin 的 count/min/Q1/median/Q3/max 摘要，不将整个候选 margin 列表写入每样本核心结果；需要原始分布时使用小样本 trace。

**不直接采纳：**

- 不采用 `alpha * |m_old(x)|` 作为 epsilon 固定下限。它会新增一个需要调参的尺度超参数，且与“从实际候选违反分布估计”的方法动机不如有界延迟初始化一致。

## 2. 落地前必须固定的口径

### 2.1 成功判定与 margin 边界

- `DifferentialClassification` 仍是成功判定的唯一权威：`old_pred == y and new_pred != y`。
- margin 只服务于搜索排序和 epsilon 可行性，不代替预测标签判定。
- 这能正确处理 margin 等于 0 时的 argmax 并列与 first-max 边界。
- 现有 `ClassificationModelOutput.margin()` 已实现“logits 优先，只有概率时使用 `log(p + 1e-12)`”，直接复用，不再实现第二套 margin 计算。

### 2.2 查询预算

- `q` 使用 `goal_function.num_queries`，表示实际传入 `get_results()` 的模型对输入数，不是已物化搜索 state 的数量。query cache hit 不增加 `q`。
- 原始样本的初始查询计入 `q` 和总预算 `Q`。在 `Q=1000` 时，搜索候选最多再消耗 999 次模型对查询。
- 候选的预算截断必须继续由 `GoalFunction.get_results()` 完成；新搜索器不绕过该接口、不直接调用模型。
- epsilon 进度使用 `clip(q / Q, 0, 1)`，严禁使用扩展轮数或生成候选数代替。

### 2.3 初始 epsilon

- 原始状态无条件完成第一次扩展，第一轮之前不需要 epsilon。
- 自适应模式使用有界延迟初始化，主配置 `initialization_max_expansions=2`。它是最大初始化窗口，不是强制预热轮数。
- 每次初始化窗口内的扩展结束后，累积“通过文本约束、已取得模型输出”的唯一 query key 的 old margin。
- 首次观察到任何 `old_margin < 0` 时，立即使用当前累积违反量 `-old_margin` 的 eta 分位数确定 `epsilon_0`，不必等满两次扩展。
- 尚未初始化时，frontier 暂按 `epsilon=0` 的 Feasibility-First PBS 规则更新；不改用 Dynamic-Beam，不无约束保留状态。
- 达到最大初始化扩展数仍无违反时，才固定 `epsilon_0=0`，并记录 `epsilon_zero_initialization=true`。
- 分位数使用确定性线性插值：`h=(n-1)*eta`，在 `floor(h)` 与 `ceil(h)` 之间插值。这个口径避免 Python/NumPy 版本默认分位数方法不同。
- `epsilon_0` 确定后，用当前全局 `q/Q` 计算已衰减的 epsilon，不从初始化完成时重置查询进度。
- 初始化窗口不额外生成或查询候选，所有数据均来自正常搜索扩展。

### 2.4 不可行候选的次序

原计划对“均不可行”的扩展次序没有完全展开，实现固定为：

1. epsilon 可行候选总是优先于不可行候选。
2. 可行候选按 `(new_error_margin, -modification_cost)` 做非支配分层和拥挤距离选择。
3. frontier 仍有空位时，不可行候选按 `violation` 升序填充。
4. `violation` 相同时，依次比较新模型 margin 降序、修改率升序、生成顺序升序。

每次 frontier 重排都根据当前 `q` 重新计算 epsilon 和可行性。早期可行的状态会在 epsilon 收紧后变为不可行，不保留历史“可行”标记。

### 2.5 Pareto 和拥挤距离

- 对可行候选执行标准非支配分层：新模型错误 margin 最大化，修改率最小化。
- 拥挤距离按 NSGA-II 的目标范围归一化方式计算。某个目标的范围为 0 时，该维贡献为 0；边界点距离为正无穷。
- 最后的确定性 tie-break 为：Pareto rank 升序、拥挤距离降序、新 margin 降序、修改率升序、生成顺序升序。
- 旧模型 margin 不加入 Pareto 目标。

### 2.6 查询去重与状态去重

`AttackedText.__eq__()` 还会受 `attack_attrs` 数量影响，不适合直接作为 AE-PBS 的查询或状态标识。实现必须分开两种 key：

```python
query_key = tuple(attacked_text.text_input.items())

state_key = (
    query_key,
    frozenset(attacked_text.attack_attrs["modified_indices"]),
    tuple(attacked_text.attack_attrs["original_index_map"]),
)
```

其中：

- `query_key` 只描述输入模型的有序字段，用于复用新旧模型输出。
- `state_key` 同时包含影响 RepeatModification 和索引可达性的路径状态，只有 `state_key` 相同时才可合并搜索状态。
- 对当前首版限定的 `kuleshov_var + WordSwapEmbedding`，`modified_indices + original_index_map` 覆盖已知的后续可达空间差异。若后续支持插入、删除或其他路径相关约束，必须扩展一个显式 `state_signature` 接口，不能沿用当前签名假设。

查询前只将未出现过的 `query_key` 传入 `get_goal_results()`。已有 `query_key` 通过搜索器的不可变 query-evaluation cache 复用模型输出，并为当前 `AttackedText` 创建路径专属的 result/state；它不增加 `goal_function.num_queries`。这是必要的，因为现有 `get_results()` 按传入列表长度计查询，即使底层模型 cache 命中，重复候选仍会消耗查询预算。

对相同 `state_key` 的重复路径，保留生成顺序最早的确定性代表，并以该实际保留路径计算 escape 诊断。不对不同路径的 `path_has_negative_old_margin` 做 OR 合并，否则可能将本可以经过全可行路径达到的成功误报为 escape 成功。

### 2.7 K=1 的含义

`K=1` 是 epsilon-Pareto 搜索的单路径版本，不等于现有 DT4LM 贪心基线。两者的排序政策不同，因此论文中不应使用“退化为原 DT4LM”的表述。

### 2.8 消融实验的归因边界

- `DT4LM-Dynamic -> Dynamic-Beam` 隔离“异步多路径”收益。
- `Dynamic-Beam -> Feasibility-First PBS` 同时改变排序表示和可行优先政策，只能归因为这一整体政策的收益。
- `Feasibility-First PBS -> AE-PBS` 在同一 beam 大小下隔离自适应 epsilon 的净作用。
- `Strict-PBS -> Feasibility-First PBS` 隔离保留不可行状态的作用。
- `Epsilon-Greedy -> AE-PBS` 在同一 epsilon-Pareto 政策下隔离 `K=1 -> K=5` 的多路径收益。
- 若后续需要严格区分“约束”与“Pareto”，再添加 `Strict-Scalar-Beam` 作为补充消融，不将它纳入首轮必跑矩阵。

### 2.9 路径诊断分母

原计划中非 top-1 路径贡献率的分子是“AE-PBS 独有成功”，分母却是“AE-PBS 全部成功”，解释不直观。实施后同时报告：

- `non_top1_path_rate` = 首层根路径的 dynamic rank 大于 1 的 AE-PBS 成功数 / AE-PBS 成功数。
- `ae_unique_non_top1_rate` = 首层根路径的 dynamic rank 大于 1 的 AE-PBS 独有成功数 / AE-PBS 独有成功数。
- `escape_path_rate` 保留历史口径，按包含 root 的成功路径是否出现 `old_margin < 0` 统计；同时记录 `post_root_escape_path_rate` 和 `post_root_old_prediction_error_path_rate`，作为判断搜索中间状态是否违反约束的主口径。
- Strict-PBS 另外记录 `discarded_infeasible_state_count/rate`，用于核对旧模型预测错误的后继状态确实被永久丢弃。

### 2.10 修改成本的优化含义

AE-PBS 将修改率作为搜索过程中的 Pareto 目标，并在同一批次的多个成功候选中选择修改率最低者。但它在首次出现成功批次时立即终止，因此：

- 它“偏好较低修改成本的搜索路径和当批成功输入”。
- 它不保证输出全部可达成功输入中的全局最小修改率。
- 论文、配置说明和结果解读统一使用“low-modification preference”，不使用“minimum-modification guarantee”。

## 3. 现有代码影响分析

### 3.1 新增模块

| 文件 | 职责 |
| --- | --- |
| `textattack/search_methods/differential_frontier.py` | 纯函数实现 epsilon 调度、约束违反量、Pareto 分层、拥挤距离、frontier 填充和确定性排序。 |
| `textattack/search_methods/async_differential_beam_search.py` | 实现搜索状态、异步扩展循环、query/state 双 key、模型输出复用、成功候选选择和诊断聚合。 |
| `tests/test_ae_pbs.py` | 纯算法、伪模型集成、查询预算和回归测试。 |
| `statistics/compare_search_methods.py` | 第二阶段新增，按 manifest index 配对 Base 与 AE-PBS，输出配对列联表、McNemar、bootstrap 和 Wilcoxon 结果。 |

### 3.2 小范围修改

| 文件 | 修改 | 兼容策略 |
| --- | --- | --- |
| `textattack/search_methods/__init__.py` | 导出新搜索器。 | 只增加 import，不替换旧类。 |
| `textattack/attack_args.py` | 增加 PAIR 专用搜索参数。 | 默认为 `legacy_greedy`，现有 CLI 行为不变。 |
| `textattack/attack_recipes/pair_2024.py` | 根据显式配置创建旧贪心搜索或新异步 frontier 搜索。 | 缺省新参数时仍走当前 `ComparatorGreedySearch`。首版继续限定 `kuleshov_var`。 |
| `textattack/loggers/jsonl_logger.py` | 为每样本增加可选 `search_diagnostics`字段。 | 旧搜索记录该字段为 `null` 或缺省，读取器同时接受旧新 schema。 |
| `improvement_config.py` | 校验可选的 `attack.search` 映射及条件字段。 | 不强制修改 schema v2 的旧配置；新 AE-PBS 配置必须写完整搜索块。 |
| `experiments/improvements/run_improvements.py` | 将 `attack.search` 转换为 CLI 参数，并为可选 trace 设置输出路径。 | 未配置 `attack.search` 时不附加任何新参数，保持旧 run hash 和断点恢复行为。 |
| `statistics/evaluate_improvements.py` | 聚合诊断指标，继续从 `queries_to_success` 产出作图原始数据。 | 旧结果没有诊断字段时输出 `null`，不拒绝重算旧 metrics。 |
| `statistics/aggregate_improvements.py` | 增加搜索身份列和诊断聚合列。 | 旧实验规范化为 `legacy_greedy`，新旧结果可进入同一 CSV。 |
| `experiments/improvements/configs/<dataset>/` | 添加 Dynamic-Beam、Feasibility-First PBS、Strict-PBS、Epsilon-Greedy 和 AE-PBS 完整配置。 | 不改写已有 Base/LexiDT/SemDT 配置和产物。 |

### 3.3 明确不修改的位置

- `textattack/transformations/` 和 Kuleshov 的变换配置。
- `textattack/constraints/` 中的 RepeatModification、Stopword、MaxWordsPerturbed、ThoughtVector、GPT-2 和 NLI 约束。
- `textattack/models/classification_output.py`：已经提供所需的 margin 语义。
- 数据预处理、manifest 生成、模型训练与 model pair 身份校验。
- SemDT 标定、标注后端和语义阈值产物。
- BERTScore 等质量评估的模型加载逻辑。

## 4. 配置与 CLI 协议

### 4.1 新配置结构

每个实验继续使用一份完整 YAML，不引入配置继承。AE-PBS 主配置的 `attack` 部分为：

```yaml
attack:
  recipe: kuleshov_var
  differential_objective: dynamic
  semantic_constraint: original
  query_budget: 1000
  search:
    method: async_frontier
    ranking: epsilon_pareto
    beam_size: 5
    epsilon:
      mode: adaptive
      initial_quantile: 0.75
      initialization_max_expansions: 2
      decay: quadratic
    diagnostics:
      trace_enabled: false
```

`differential_objective: dynamic` 在 AE-PBS 中仅保留 GoalFunctionResult 的兼容分数，以及计算“原 dynamic top-1”路径诊断；AE-PBS 的实际候选选择只由 `attack.search` 决定。结果中必须同时记录 legacy objective 和 search policy，避免将 AE-PBS 误读为 dynamic 搜索。

旧配置中没有 `attack.search` 时，规范化视图为：

```yaml
search:
  method: legacy_greedy
```

该默认值只在运行时解释，不回写旧配置文件，避免破坏已有产物的配置哈希。

### 4.2 方法矩阵的配置映射

| 方法 | `differential_objective` | `method` | `ranking` | `beam_size` | epsilon |
| --- | --- | --- | --- | ---: | --- |
| DT4LM-Dynamic | `dynamic` | `legacy_greedy` | - | 1 | - |
| LexiDT | `lexi` | `legacy_greedy` | - | 1 | - |
| Dynamic-Beam | `dynamic` | `async_frontier` | `dynamic` | 5 | `disabled` |
| Feasibility-First PBS | `dynamic` | `async_frontier` | `epsilon_pareto` | 5 | `strict` + `feasibility_first` |
| Strict-PBS | `dynamic` | `async_frontier` | `epsilon_pareto` | 5 | `strict` + `discard` |
| Epsilon-Greedy | `dynamic` | `async_frontier` | `epsilon_pareto` | 1 | `adaptive` |
| AE-PBS | `dynamic` | `async_frontier` | `epsilon_pareto` | 5 | `adaptive` |

Dynamic-Beam 不应用 epsilon 筛选，否则无法与 DT4LM-Dynamic 隔离多路径搜索收益。它使用与 AE-PBS 相同的状态生命周期、frontier 容量、去重、单点扩展和成功候选选择，仅将 frontier 排序改为 dynamic score 降序。

### 4.3 CLI 参数

PAIR recipe 新增以下参数：

- `--differential-search {legacy_greedy,async_frontier}`
- `--differential-frontier-ranking {dynamic,epsilon_pareto}`
- `--differential-beam-size INT`
- `--epsilon-mode {disabled,strict,adaptive}`
- `--epsilon-initial-quantile FLOAT`
- `--epsilon-initialization-max-expansions INT`
- `--epsilon-decay {linear,quadratic}`
- `--infeasible-state-policy {feasibility_first,discard}`
- `--search-trace-output PATH`，只在 `trace_enabled: true` 时由 runner 传入。

参数校验使用条件规则：

- `beam_size` 必须为正整数。
- `dynamic` ranking 必须搭配 `epsilon.mode=disabled`。
- `epsilon_pareto` 必须搭配 `strict` 或 `adaptive`。
- `adaptive` 要求 `0 <= initial_quantile <= 1`、`initialization_max_expansions >= 1`，且 decay 为 `linear` 或 `quadratic`。
- `strict` 不读取 quantile、initialization window 和 decay；配置中出现无效字段时直接报错，不静默忽略。
- `discard` 只允许与 strict epsilon-Pareto 组合；adaptive epsilon 使用 `feasibility_first`。
- `async_frontier` 首版必须使用 `differential_objective: dynamic`，以保证相同 query key 复用的兼容分数不受路径修改率影响；Static/LexiDT 仍使用旧贪心搜索。
- `async_frontier` 首版只支持 `PAIR + kuleshov_var + DifferentialClassification`，其他 recipe 显式报出兼容性错误。

## 5. 内部数据结构

搜索状态保存 `GoalFunctionResult` 以避免文本、预测和分数脱节，并缓存搜索所需的标量：

```python
@dataclass
class DifferentialSearchState:
    state_id: int
    query_key: tuple
    state_key: tuple
    result: GoalFunctionResult
    old_prediction: int
    new_prediction: int
    old_correct_margin: float
    new_error_margin: float
    modification_cost: float
    modified_indices: frozenset[int]
    parent_id: int | None
    root_child_id: int | None
    depth: int
    generation_order: int
    expanded: bool
    path_has_negative_old_margin: bool
    path_has_old_prediction_error: bool
    root_dynamic_rank: int | None
```

实际实现还需定义三个轻量结构：

- `DifferentialQueryEvaluation`：按 `query_key` 不可变保存新旧模型输出、预测、margin、goal status 和 dynamic score。同文本的新路径复用它，但重新计算路径修改率并绑定当前 `AttackedText`。
- `FrontierRank`：保存当前 epsilon 下的可行性、违反量、Pareto rank 和拥挤距离，每次重排重新生成。
- `SearchDiagnosticsAccumulator`：在搜索中在线累加 frontier 宽度、rank-1 大小、深度多样性、state 去重数和 query cache 复用数，不把全部候选列表留在内存中。

所有新模块、公开类和函数都必须包含说明职责与边界的 docstring；epsilon 初始化、约束支配、拥挤距离和查询截断等非直观代码块必须有简洁注释。不添加只复述赋值语句的空注释。

## 6. 搜索算法的确定性执行流程

### 6.1 初始化

1. 从 `initial_result` 构造 root state，保存原始文本作为全程约束比较基准。
2. 若原始样本已是差分成功，现有 `init_attack_example()` 会将它标记为 skipped，不进入新搜索器。
3. 其他初始状态均进入搜索，不再要求新旧模型在原始样本上都正确。
4. `frontier=[root]`，`epsilon_0=None`，query-evaluation cache 包含 root 的模型输出，已见 state key 集合包含 root。
5. 紧凑记录原始样本 old margin，用于后续计算 `epsilon_0 / (abs(root_old_margin) + 1e-12)`。

### 6.2 单次异步扩展

1. 按当前 policy 选择一个未扩展状态，从 frontier 移除并标记已扩展。root 在首轮直接选中。
2. 调用 `get_transformations(parent_text, original_text=root_text)`，由现有 Attack 统一应用 pre-transformation 和其他约束。
3. 为每个候选计算 query key 和 state key。丢弃已见的相同 state key，但不丢弃 query key 相同、state key 不同的路径。
4. 将 query key 分为 cache hit 和 miss。只将 miss 候选传入 `get_goal_results()`，cache hit 从 `DifferentialQueryEvaluation` 物化路径专属 result。合并时恢复原候选生成顺序，不让 cache 命中与否改变 tie-break。
5. 从现有 `new_model_output`/`old_model_output` 取出预测和 role-specific margin，修改成本继续调用 `attacked_text.modification_rate(root_text)`。相同 query key 的不同 state 可有不同修改成本与路径诊断。
6. 在自适应初始化窗口内，用本轮新观察到的唯一 query evaluation 更新违反集。首次出现负 old margin 时立即计算 `epsilon_0`；到达最大扩展数仍无违反时固定为 0。
7. 先在当前生成的全部路径 state 中检查严格成功，包括 query cache hit。如有多个，按修改率升序、新 margin 降序、旧 margin 降序、生成顺序升序选择，然后立即终止。
8. 如未成功，以更新后的 `q` 计算 epsilon；尚未确定 `epsilon_0` 时使用 0。将剩余 frontier 与新 states 合并，按 policy 裁剪到 `K`。
9. `search_over=True` 时不再扩展，即使该批次未装满 frontier。cache hit 不影响 `search_over`，因为它不消耗新模型对查询。

### 6.3 终止与失败返回

搜索在下列任一条件下终止：

- 找到严格差分成功。
- 查询预算耗尽。
- frontier 为空。
- 所有可达状态均已扩展，无新候选。

失败时不随意返回“最后一个”状态：

- epsilon-Pareto 模式在终止时的 epsilon 下，按与扩展选择相同的确定性次序返回最优的已评估非成功状态。
- Dynamic-Beam 返回 dynamic score 最高的已评估非成功状态。
- 没有任何变换候选时返回 root。

返回前将紧凑的 `search_diagnostics` 附加到最终结果，`SearchMethod.__call__()` 继续统一回填最终查询数。

## 7. 诊断与产物协议

### 7.1 `results.jsonl`

每个 successful/failed 样本增加一个紧凑的 `search_diagnostics` 对象；skipped 样本为 `null`。必须记录：

- `method`、`ranking`、`beam_size`、`epsilon_mode`、`initial_quantile`、`initialization_max_expansions`、`decay`。
- `root_old_margin`、`epsilon_0`、`epsilon_to_root_margin_ratio`、`epsilon_initialization_expansion`、`epsilon_zero_initialization`、`final_epsilon`、`expansion_count`、`max_depth`。
- 第一次扩展 old margin 的 count/min/Q1/median/Q3/max 和负 margin count；主结果不保存整个 margin 列表。
- 生成、约束后、state-key 去重后、query-cache hit/miss、实际查询的候选数，以及预算截断数。
- frontier 宽度、rank-1 大小、不同 modified-index 集合数、不同深度数的 mean/max。
- 成功路径深度、首层 dynamic rank、是否经过负 old margin、是否经过旧模型错误预测。
- 终止原因：`success`/`budget_exhausted`/`frontier_empty`/`no_transformations`。

JSONL 结果 schema 升级时，读取端必须同时支持旧 schema 2 和新 schema 3；旧记录的搜索方式由 resolved config 解释为 `legacy_greedy`。

### 7.2 可选搜索 trace

默认主实验不写全量候选 trace，避免明显增加 I/O 和产物大小。机制分析或小样本 pilot 可显式开启 `search_trace.jsonl`，每次扩展只记录：

- dataset index、expansion index、parent/root-child ID 和深度。
- 当前 q/Q、epsilon、frontier/rank-1 大小。
- 本轮生成、去重、查询和成功数。
- 被选中父状态的 margin、cost、violation 和 rank。

默认 trace 不重复保存候选全文和全部模型向量。如需审查具体文本，只在专门小样本配置中开启增强 trace。

### 7.3 metrics 和 summary

`core.json` 保留现有全部指标，并增加 AE-PBS 诊断聚合：

- `search_expansions_mean`、`search_max_depth_mean`。
- `frontier_size_mean`、`frontier_size_max`、`rank1_size_mean`。
- `frontier_modified_set_diversity_mean`、`frontier_depth_diversity_mean`。
- `duplicate_state_rate`、`query_cache_hit_rate`、`budget_truncation_rate`。
- `non_top1_path_rate`、root-inclusive 的历史 escape 口径、排除 root 的 post-root escape 口径。
- `discarded_infeasible_state_count`、`discarded_infeasible_state_rate`。
- `epsilon_zero_initialization_rate`、`epsilon_to_root_margin_ratio_median`、`epsilon_initialization_expansion_mean`。
- `budget_penalized_queries_mean`、`budget_penalized_queries_median`，其中 failed 按 Q 计、successful 按 `queries_to_success` 计、skipped 不进入。

`success_queries.json` 继续保存 columnar 原始数据，无需在本阶段绘制 success-query curve。所有主实验的 `success_budgets` 设为 `[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]`，以支持后续作图。

`summary.csv` 新增：

- `search_method`、`frontier_ranking`、`beam_size`、`epsilon_mode`、`epsilon_initial_quantile`、`epsilon_initialization_max_expansions`、`epsilon_decay`。
- `old_model_training_seed`、`new_model_training_seed`、`manifest_seed`、`attack_seed`，不再将四种随机性简化为一个含义不明的 seed。
- 上述运行级诊断聚合列。
- 动态展开所有已配置的 `success_at_B` 列，不再只依赖 100/500/1000 的固定列表。

QPS 继续沿用已确认的论文口径：所有 successful/failed/skipped 样本产生的模型对查询总数，除以 successful 数量。

## 8. 实施阶段

### 阶段 0：冻结基线

1. 运行现有 pure tests 和改进实验相关测试，记录通过数。
2. 用固定伪模型记录 Dynamic 和 LexiDT 的候选选择结果。
3. 保存一个小样本现有 CLI 命令和结果，用于新模块接入后做行为回归。

验收：不修改代码时基线测试全部通过，固定输入产物可重复。

### 阶段 1：实现纯算法内核

1. 实现 epsilon 有界延迟初始化、线性/二次衰减和违反量。
2. 实现约束支配、非支配分层、拥挤距离和稳定选择。
3. 实现 dynamic frontier 和 epsilon-Pareto frontier 两个 policy，共用状态与扩展器。

验收：纯函数测试不加载 TextAttack 模型，每个排序结果在重复运行时完全一致。

### 阶段 2：接入 TextAttack 搜索生命周期

1. 实现 `AsyncDifferentialBeamSearch.perform_search()`。
2. 接入 `Attack.get_transformations()` 和 `GoalFunction.get_results()`，确保约束和预算管理仍由现有组件负责；在搜索层另外维护 query-evaluation cache，不将相同文本再次传入 `get_results()`。
3. 在 PAIR recipe 中实现 opt-in 选择，并对错误 goal/recipe 组合提供可操作的错误信息。
4. 完成 query/state 双 key、路径专属 result 物化、成功批次选择和确定性失败返回。

验收：伪变换与伪模型能构造“旧对/新对 -> 旧错/新对 -> 旧对/新错”路径，AE-PBS 找到成功且严格策略无法保留该路径。

### 阶段 3：配置、runner 和产物

1. 增加配置校验与 CLI 参数，让 runner 完全由 YAML 生成搜索命令。
2. 为 JSONL logger 增加搜索诊断，为 trace 实现原子写入和正常 close/flush。
3. 扩展 metrics 和 summary，确保旧结果重算仍可用。
4. 断点恢复仍以完整 `results.jsonl` 为 attack 阶段边界；删除 `metrics/` 后可独立重算指标，不重跑搜索。

验收：新配置的 resolved config、命令、JSONL、core、success queries 和 summary 中的搜索身份完全一致。

### 阶段 4：小样本与回归验证

1. 在 SST-2、RTE、MRPC 和 MR 上各运行 5-10 条样本 smoke test，覆盖单句与句子对、长短文本、失败和跳过。
2. 对同一配置重复运行，校验 dataset index、结果状态、查询数、成功文本和诊断字段确定性。
3. 重跑旧 Base/LexiDT 小样本命令，与阶段 0 产物逐字段比较。
4. 检查查询数永不超过 `Q`，并确认每个结果都是 successful/failed/skipped 三态之一。

验收：四个数据集均能完成小样本运行，旧命令无行为回归。

### 阶段 5：pilot、冻结参数与正式实验

1. 只在与正式 test manifest 不相交的 validation/pilot manifest 上比较 `K in {1,3,5,10}`、`eta in {0.50,0.75,0.90}`、`initialization_max_expansions in {1,2}` 与线性/二次衰减。
2. 冻结主配置后再运行正式 test manifest，不根据 test 结果回选参数。
3. 首轮主比较使用已有 SST-2 与 RTE 模型对和同一 manifest；代码和配置协议同时支持 MRPC/MR，首轮有正向结果后扩展到它们。
4. 快速首轮使用固定 attack seed 765，并对同一 checkpoint/manifest 重复运行以确认攻击是否完全确定。
5. 若攻击确定，最终实验不使用多个无效 attack seed 重复相同结果。不确定性通过固定 manifest 上的配对 bootstrap 衡量；如资源允许检查训练稳健性，使用至少 3 组独立微调 seed 的 old/new checkpoint 对，每组内方法严格配对。
6. 配置与 summary 分别记录 `models.old.training_seed`、`models.new.training_seed`、`dataset.evaluation.sample_seed` 和 `experiment.seed`（attack seed）。如未引入随机 tie-break，不虚构 tie-break seed；RTE 使用全部样本时明确标注 sample seed 不影响样本集。

验收：正式 test 配置和参数冻结记录的时间早于 test 结果，全部方法按 dataset/model pair/seed 完全配对。

## 9. 测试清单

### 9.1 纯函数

- logits 与 probabilities 两种输出的 old/new margin。
- 二分类、多分类、并列最大值和接近 0 概率。
- 空违反集、单元素和多元素分位数，以及第一次扩展无违反、第二次才出现违反的延迟初始化。
- epsilon 在 `q=0`、`q=Q`、`q>Q` 与严格模式的边界。
- 可行支配、不可行 violation 排序、多个 Pareto front 与不完整末层。
- 目标范围为 0、front 仅 1/2 个元素时的拥挤距离。
- 所有值相同时按 generation order 稳定输出。

### 9.2 搜索状态机

- 每轮只扩展一个 state，扩展过的 state 不再进入 frontier。
- frontier 不超过 K，且可同时保留不同深度。
- epsilon 衰减后状态从可行变为不可行并正确重排。
- 同 query key 不同 state key 只查询一次但保留多个搜索状态；相同 state key 才合并，句子对 canonical key 不串列。
- query cache hit 物化的 result 绑定当前 `AttackedText`、修改率和路径属性，且不改变候选生成顺序。
- 同批多个成功候选按规定的成本/margin/顺序选择。
- 预算在候选批中截断时，只为返回结果建立 state，不越界访问未查询候选。
- frontier 空、没有 transformation、预算耗尽三种失败路径。
- K=1、strict epsilon、adaptive epsilon 和 dynamic ranking 的独立行为。

### 9.3 集成与回归

- AE-PBS 只接受 DifferentialClassification，与普通单模型 goal 组合时在查询前报错。
- Original 和 NLI 约束仍通过 `get_transformations()` 被调用。
- MRPC 的双字段修改率和 modified indices 正确。
- successful/failed/skipped 三态与 initial state 计数不变。
- `model_pair_queries <= query_budget`，`queries_to_success` 仅在 successful 样本中为正整数。
- 新 metrics 可从新 JSONL 生成，也可从没有 `search_diagnostics` 的旧 JSONL 重算。
- 旧 YAML 产生的 attack command 不增加新参数，Base/Static/LexiDT/SemDT 搜索类型与修改前一致。
- 同一 checkpoint、manifest 和配置重复运行的确定性检查，并验证 attack seed 在当前算法中是否实际影响产物。

## 10. 实验执行协议

### 10.1 数据与公平性

- 不引入“新旧模型均预测正确”筛选。
- 每个 dataset/model pair/old-new-training-seed-pair/manifest-seed 的所有方法使用同一 test manifest 和完全相同的样本顺序。
- `dataset.evaluation.sample_size` 依现有协议配置：缺省、`null` 或非正数使用全部样本；正数时按 manifest seed 随机抽取至多该数量。
- 正式主实验延续原论文口径：测试集随机最多 1000 条，数量不足的数据集使用全部测试样本。
- 所有对比使用相同 Kuleshov transformation、原有约束、`Q=1000` 和模型批大小。
- 公平性的核心约束是相同模型对查询预算，同时单独报告 wall-clock 和峰值显存。

### 10.2 主指标

设 `S`、`F`、`K` 分别为 successful、failed、skipped 数，`A=S+F` 为 attackable 数，`N=S+F+K` 为 manifest 总数，`q_i` 为样本的模型对查询数，`q_i^success` 为成功样本首次成功的累计查询数。所有比率在分母为 0 时记为 `null`，不记为 0。

- 原始数量：`S`、`F`、`K`、`A`、`N`。
- Paper GSR：`S / A`。
- Sample generation rate：`S / N`。
- Preexisting differential rate：`K / N`。
- AMR：`sum(modification_rate_i for successful i) / S`。
- QPS：`sum(q_i for all N samples) / S`，包含 successful、failed 和 skipped 的查询。
- Success@B：`#{successful i: q_i^success <= B} / A`。
- Sample-Success@B：`#{successful i: q_i^success <= B} / N`。
- Success-query AUC：在 `B=1..Q` 上对 Success@B 取平均，等价于 `sum(Q-q_i^success+1 for successful i) / (A*Q)`。
- 成功样本查询数的 median/Q1/Q3，分母/样本集仅为 `S`。
- 预算惩罚查询成本：对每个 attackable 样本定义 `q_i*=q_i^success`（successful）或 `q_i*=Q`（failed），报告 `A` 个 `q_i*` 的 mean/median。skipped 不进入该指标。
- Success@B 和 Sample-Success@B 的 `B=100,200,...,1000`。
- BLEU、METEOR、ROUGE-L、BERTScore、wall-clock time 和 peak VRAM。
- 第 7 节定义的搜索机制诊断。

### 10.3 配对统计

- 通过 dataset index 对齐同一 manifest 的方法。任何 index 缺失、重复或 manifest hash 不一致都直接报错。
- 成功率使用 McNemar 检验和样本级配对 bootstrap 95% 置信区间。
- 原始查询数与修改率在两种方法共同成功样本上使用 Wilcoxon signed-rank、配对中位数差与 bootstrap 95% 置信区间，作为“成功条件下”的效率/质量分析。
- 在全部 attackable 配对样本上比较 `q_i*`，报告配对 mean/median 差、Wilcoxon 和 bootstrap 95% 置信区间。该分析与 QPS、Success@B 和 AUC 共同解释，不单独把共同成功子集外推到全部样本。
- 同时输出 Base 独有成功、AE-PBS 独有成功、共同成功、共同失败和跳过数。
- skipped 不进入 GSR 的 McNemar 成功/失败配对，但必须在三态列联表中单独呈现。

### 10.4 进入扩展实验的标准

建议将原计划的判定进一步操作化：

1. 在相同预算下，AE-PBS 的 GSR 比 DT4LM-Dynamic 高至少 3 个百分点，且 AMR 相对增幅不超过 10%。
2. GSR 绝对差不超过 1 个百分点时，Success@500 相对增幅至少 10%，或 QPS 相对下降至少 10%。
3. 在至少两个数据集上均有 AE-PBS 独有成功，且这些独有成功中至少 20% 来自非 dynamic top-1 根路径或曾经过负 old margin。

若 Dynamic-Beam 与 AE-PBS 表现接近，结论应是收益主要来自多路径搜索。若所有 beam 方法都不优于贪心搜索，应停止继续增加多目标复杂度，转向候选变换空间、约束过滤率和查询分配机制的分析。

## 11. 主要风险与控制

| 风险 | 影响 | 控制方案 |
| --- | --- | --- |
| 首次扩展候选过多 | 在建立 frontier 前已消耗大量预算 | 严格经过 `get_results()` 截断，记录截断率；首版不改 Kuleshov 候选生成，避免混入第二个变量。 |
| 重复路径浪费查询 | QPS 上升且结果受路径顺序影响 | 使用 query key 复用模型输出，使用包含修改历史的 state key 决定是否合并搜索状态。 |
| 初始化窗口内没有负 old margin | AE-PBS 在该样本上退化为 Feasibility-First PBS | 主配置允许最多两次扩展延迟初始化，报告 `epsilon_zero_initialization_rate` 并在 pilot 比较 1/2 次窗口。 |
| epsilon 过大 | frontier 充满无法恢复的旧模型错误路径 | 按样本候选违反分布自适应，随查询二次收紧，报告 `epsilon_0`、归一化 epsilon 与 escape 成功率。 |
| 低修改率边界点长期占据 frontier | 浅层状态饥饿或无效循环 | 状态扩展后永久移出，按 state key 去重，记录深度与 modified-set 多样性。 |
| 不同模型 logit 尺度不同 | 固定 epsilon 不可迁移 | epsilon 仅用同一旧模型、同一样本初始化窗口内的违反分布初始化；跨样本分析使用归一化 epsilon 诊断，不直接解释 margin 绝对值。 |
| 诊断日志影响性能 | wall-clock 比较被 I/O 污染 | 主实验只写在线聚合摘要，全量 trace 只在小样本机制实验开启。 |
| 新配置隐式改变旧运行 | 历史结果不可重现 | 新搜索严格 opt-in，旧 YAML 不回写、不变更命令、不修改旧搜索类。 |

## 12. 最终验收条件

只有同时满足以下条件，AE-PBS 首版才视为实施完成：

1. 纯算法、状态机、产物与旧功能回归测试全部通过。
2. 旧 Base、Static、LexiDT 和 SemDT 配置不需修改便能按原路径运行。
3. SST-2、RTE、MRPC 和 MR 的小样本 smoke test 均能完成，句子对任务的去重和修改率正确。
4. 任何样本的模型对查询数不超过配置预算，相同 query key 不重复计费，相同文本但 state key 不同的路径仍可同时进入 frontier。
5. 新方法的 resolved config、`results.jsonl`、metrics 和 `summary.csv` 中搜索身份一致，不能只显示 `objective=dynamic`。
6. 删除 `metrics/` 后可直接重算评估，已完成的 attack 结果不会重跑。
7. 可从保存的 columnar success-query 数据生成任意 `B<=Q` 的 Success@B，不需重跑攻击。
8. 配对分析脚本能对 Base 与 AE-PBS 输出三态列联表、成功差、独有成功、共同成功查询/修改统计、全 attackable 样本预算惩罚查询统计和路径机制指标。
9. 归一化 epsilon、首次扩展 margin 摘要、epsilon 零初始化率和初始化扩展位置均能从产物中审核。
10. 同一 checkpoint/manifest 的重复运行能明确判定 attack seed 是否实际影响结果；确定性攻击不重复报告伪独立 seed。

按此边界实现后，AE-PBS 是一个完全显式启用的新搜索策略，而不是对现有 DT4LM 默认行为的全局改写。这使它可以与 Base、LexiDT 和 SemDT 在同一框架、同一 manifest 和同一产物协议下公平对比，也能在结果不理想时无成本回退到现有方法。
