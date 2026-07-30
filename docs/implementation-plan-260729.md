# DT4LM 改进执行方案

本文档将 `plan-260729.md` 中的 SemDT 与 LexiDT 思路落实为可执行的工程、
实验和验收方案。文中决策均已确认，不再保留待定项。

## 0. 评审处理结论

对 `reply-260729.md` 的意见处理如下：

- 直接采纳：RTE 使用全部合格测试样本；所有指标使用 manifest 的实际样本数
  \(N_D\)；Base、Static、LexiDT 共用比较器驱动的搜索实现；LexiScore 增加
  显式预测状态；模型 wrapper 显式声明输出类型；补报 EligibilityRate、
  端到端成本和 NLI 截断信息；补全 NLI 缓存键；人工评估的小层至少抽 5 条
  或全量纳入。
- 调整后采纳：初始 1000 条候选预先固定为 800 条搜索集和 200 条验证集。
  若验证集语义保持正例少于 100 条，新增至多 1000 条独立补充审计样本，使
  Base 候选池总标注量最多为 2000 条；不得把补充样本与原 1000 条重新划分，
  也不得用其重新选择阈值。阈值冻结后，再从 SemDT 的实际搜索轨迹额外抽取
  100 条独立审计，只检查分布偏移。
- 不采纳：将首版缩减为只实现 OpenAI judge、HF 只留接口。该建议与已确认的
  “同时实现 Responses API 与本地 HF 标注器，且每次实验只使用一个后端”
  冲突。两个后端继续独立产出标签、阈值和复现实验结果。

这些修改不改变已确认的 Static 定义、QPS 论文口径、双后端独立性和首轮
ALBERT 模型范围。

## 1. 目标与首轮范围

首轮只研究两个相互独立的改进：

- SemDT：保留 DT4LM 动态目标和 Kuleshov 搜索，只在原有约束之后增加双向
  NLI 语义过滤。
- LexiDT：保留 Kuleshov 的变换及全部原有约束，只把动态标量目标替换为阈值
  字典序目标，并使用能够直接比较字典序分数的贪心搜索。
- Combined：同时启用 NLI 与字典序目标，只用于兼容性检查，不作为两个独立
  研究问题的主要证据。

首轮实验矩阵如下：

| 实验组 | 语义约束 | 差分目标 | 搜索 |
| --- | --- | --- | --- |
| Base | Kuleshov 原约束 | DT4LM dynamic | ComparatorGreedySearch |
| Static | Kuleshov 原约束 | DT4LM static | ComparatorGreedySearch |
| SemDT | Kuleshov 原约束 + NLI | DT4LM dynamic | ComparatorGreedySearch |
| LexiDT | Kuleshov 原约束 | LexiDT | ComparatorGreedySearch |
| Combined | Kuleshov 原约束 + NLI | LexiDT | ComparatorGreedySearch |

Static 是 LexiDT 的辅助对照组，不构成第三个改进方法。

SemDT 还有三个阈值运行变体：

- `SemDT-manual`：使用 `0.90/0.05`，作为阈值敏感性对照；
- `SemDT-openai`：使用 OpenAI 独立标注和标定的阈值，作为首轮主结果；
- `SemDT-hf`：使用本地 HF 模型独立标注和标定的阈值，作为复现结果。

三者共享 SemDT 实现，但属于独立运行，标签和阈值不得混合。Combined 只使用
OpenAI 标定阈值完成兼容性检查，避免把可选实验扩展成另一组后端比较。

统一范围：

- 数据集：SST-2、RTE。
- 模型对：ALBERT base v1 为旧模型，ALBERT base v2 为新模型。
- 基础 recipe：`kuleshov_var`。
- 当前正式配置中 SST-2 和 RTE 都使用预处理后 test split 的全部合格样本；
  采样策略与数量均为 YAML 超参数，代码不按数据集名称推断。
- 数据集 \(D\) 的实际实验规模 \(N_D\) 以冻结的 test manifest 为准。
- 每个原始样本最多 1000 次模型对查询。
- 随机种子：`765`。
- 当前只支持分类任务，但模型输出接口需要兼容 Encoder-only、
  Decoder-only 和 Encoder-Decoder 的序列分类头。

LEAP 和 FastGA 不在首轮范围。只有独立方法达到成功标准后，才扩展 recipe
和模型架构。首轮只实现并验证 ALBERT v1/v2；DeBERTa、Qwen/LLaMA 和 T5
只受统一输出协议约束，不在首轮编写或验证专用适配器。

## 2. 已固定的定义

### 2.1 模型角色和成功条件

沿用当前 `pair` recipe 的模型顺序：

- `--model` 是新模型；
- `--second-model` 是旧模型。

标签为 \(y\) 的候选 \(x'\) 成功，当且仅当：

\[
\operatorname{pred}_{old}(x')=y
\quad\land\quad
\operatorname{pred}_{new}(x')\ne y.
\]

实验样本必须先满足：

\[
\operatorname{pred}_{old}(x)=y
\quad\land\quad
\operatorname{pred}_{new}(x)=y.
\]

因此，原始输入上已经存在的负向翻转不进入生成实验。

### 2.2 修改成本

只使用变换过程记录的词级替换位置：

\[
c(x,x')=
\frac{|\operatorname{modified\_indices}(x')|}
     {N_{\mathrm{words}}(x)}.
\]

句子对的分母是两个原始句子的总词数，分子是两个字段中修改位置数之和。
首轮 recipe 只有替换，没有插入和删除，因此不使用子词数或
Levenshtein 距离。

### 2.3 查询口径

一次候选评估会同时查询新旧模型，记为一次“模型对查询”：

- 初始输入查询计入预算；
- 同一批候选的批处理不改变逻辑查询数；
- NLI 推理不计入受测模型的查询预算；
- 同一数据集的所有方法使用同一 manifest 和相同的 \(N_D\)。

定义：

\[
\mathrm{PerturbationInducedGSR}_D=
\frac{N_{\mathrm{success},D}}{N_D},
\]

\[
\mathrm{Success@B}_D=
\frac{\#\{i:\mathrm{success}_i\land q_i\le B\}}{N_D},
\quad B\in\{100,500,1000\},
\]

\[
\mathrm{QPS}_D=
\frac{\sum_{i=1}^{N_D}q_i}{N_{\mathrm{success},D}}.
\]

QPS 的分子包含成功和失败样本产生的全部模型对查询。若成功数为 0，则 QPS
报告为 `N/A`，同时保留查询总数，不用 0 或无穷大代替。

另外报告：

\[
\mathrm{EligibilityRate}_D=
\frac{\#\{\operatorname{pred}_{old}(x)=y\land
\operatorname{pred}_{new}(x)=y\}}
{\#\{\text{test split 中的样本}\}}.
\]

正文可将 `Perturbation-induced GSR` 简写为 GSR，但表头和机器可读字段必须
保留完整口径。由于 eligible 过滤比原论文的候选范围更严格，该 GSR 不与原
论文表格中的绝对值直接比较；本文内部方法仍可在相同 manifest 上公平比较。

NLI 另外报告：

- NLI 候选数；
- 实际构造的方向句对数；
- NLI batch 数；
- 总耗时和每候选耗时；
- 峰值显存；
- 缓存命中率；
- 端到端总耗时和每个成功输入的端到端耗时。

## 3. 总体架构

继续使用 TextAttack 的四组件结构，不复制整套 recipe：

```text
PAIR2024
  |
  +-- base recipe: Kuleshov2017Var
  |     +-- WordSwapEmbedding(max_candidates=15)
  |     +-- RepeatModification
  |     +-- StopwordModification
  |     +-- MaxWordsPerturbed(max_percent=0.5)
  |     +-- ThoughtVector(threshold=0.2)
  |     +-- GPT2(max_log_prob_diff=2.0)
  |     +-- ComparatorGreedySearch
  |
  +-- objective
  |     +-- dynamic
  |     +-- static
  |     +-- lexi
  |
  +-- optional final constraint
        +-- BidirectionalNLI
```

统一从 `pair` recipe 进入，通过两个正交配置选择实验组：

```text
--differential-objective dynamic|static|lexi
--semantic-constraint original|nli
```

默认值保持 `dynamic` 和 `original`，保证现有 DT4LM 脚本行为不变。首轮脚本
显式指定 `--base-recipe kuleshov_var`，不依赖当前默认的 `leap`。

当 objective 为 `lexi` 时，recipe 只替换 goal objective 和 comparator；
当 semantic constraint 为 `nli` 时，只在现有后置约束列表末尾追加 NLI。
所有 objective 都使用同一个 `ComparatorGreedySearch` 状态机：

```text
ComparatorGreedySearch
  +-- ScalarComparator          # dynamic/static
  +-- LexicographicComparator   # lexi
```

dynamic 模式必须用 golden test 证明候选选择、成功结果、预算截断和查询数与
当前 `GreedySearch` 一致。这样实验组之间只改变比较规则，不引入另一套搜索
控制流。

## 4. 模型输出与目标函数

### 4.1 统一分类输出

当前 `ClassificationGoalFunction` 会立即把 logits 转成概率，导致 LexiDT
无法优先使用 logits。新增一个仅对差分目标启用的结构：

```python
@dataclass(frozen=True)
class ClassificationModelOutput:
    scores: torch.Tensor
    score_type: Literal["logits", "probabilities"]
```

处理规则：

1. Hugging Face 序列分类 wrapper 直接读取模型返回对象的 `.logits`，并将
   `score_type` 声明为 `logits`；
2. 自定义 wrapper 必须在构造时显式配置 `score_type`，禁止根据数值范围、
   行和或其他启发式规则猜测；
3. 结构体按 `score_type` 提供经过校验的 `probabilities` 属性；只有 logits
   来源提供原始 `logits` 属性；
4. 显示、dynamic 和 static 目标使用概率；
5. LexiDT 有 logits 时使用 logits margin，否则使用 log-probability margin；
6. 结果日志继续把新模型概率作为兼容的 `raw_output`，另外保存新旧模型的
   logits、概率、预测标签和 margin。

对未来模型的约束是：ALBERT、DeBERTa、Qwen/LLaMA 和 T5 均以序列分类头
输出 `[batch_size, num_labels]` 分数。首版不支持只生成标签文本的分类方式。

模型对加载逻辑要从 `PAIR2024` 中直接调用
`AutoModelForSequenceClassification` 的方式抽离，复用统一 model wrapper
工厂，并在运行前检查：

- 两个模型类别数一致；
- 数据集标签到模型标签的映射一致；
- 输出是二维分类分数；
- ground-truth label 在合法范围内。

### 4.2 Dynamic 目标

保留当前 DT4LM 动态分段目标作为 Base 和 SemDT 的行为基线。重构时先用
固定输入输出建立回归测试，确保计算结果、成功状态和查询数与当前实现一致。

### 4.3 Static 目标

真实标签的 softmax 概率记为 \(P(y\mid x')\)，定义：

\[
\operatorname{score}_{static}(x')
=
P_{\mathrm{old}}(y\mid x')
-
P_{\mathrm{new}}(y\mid x').
\]

成功条件仍使用 argmax，不用 score 代替成功判定。该目标没有超参数。
保留现有 `lambda1/lambda2` CLI 参数以兼容旧命令，但它们不参与 static、
dynamic 或 LexiDT 的新实现；启动时不把它们写入目标配置。

### 4.4 LexiDT margin

旧模型正确 margin：

\[
m_{\mathrm{old}}(x')
=
z_{\mathrm{old},y}
-
\max_{k\ne y}z_{\mathrm{old},k}.
\]

新模型错误 margin 使用相反方向：

\[
m_{\mathrm{new}}(x')
=
\max_{k\ne y}z_{\mathrm{new},k}
-
z_{\mathrm{new},y}.
\]

如果只有概率，则分别使用：

\[
m_{\mathrm{old}}
=
\log(p_{\mathrm{old},y}+\epsilon)
-
\log(\max_{k\ne y}p_{\mathrm{old},k}+\epsilon),
\]

\[
m_{\mathrm{new}}
=
\log(\max_{k\ne y}p_{\mathrm{new},k}+\epsilon)
-
\log(p_{\mathrm{new},y}+\epsilon),
\]

其中 \(\epsilon=10^{-12}\)。

首轮阈值为：

\[
\kappa_{\mathrm{old}}=\kappa_{\mathrm{new}}=0.
\]

截断分量和最终分数：

\[
g_{\mathrm{old}}=\min(m_{\mathrm{old}},0),
\qquad
g_{\mathrm{new}}=\min(m_{\mathrm{new}},0),
\]

\[
I_{\mathrm{old}}
=
\mathbf{1}[\operatorname{pred}_{old}(x')=y],
\qquad
I_{\mathrm{new}}
=
\mathbf{1}[\operatorname{pred}_{new}(x')\ne y],
\]

\[
G_{\mathrm{LexiDT}}(x')
=
\left(
I_{\mathrm{old}},
g_{\mathrm{old}},
I_{\mathrm{new}},
g_{\mathrm{new}},
-c(x,x')
\right).
\]

预测状态必须由实际 argmax 结果计算，不能从 margin 正负推断。这样即使
margin 恰好为 0 或 argmax 存在并列，排序仍与真正的成功条件一致。新增
不可变 `LexiScore`，禁止把五元组压缩成带权标量。

### 4.5 比较器驱动的统一贪心搜索

当前 `GreedySearch` 继承自 `BeamSearch`，只会把 `result.score` 转成 NumPy
标量数组，因此不能直接承载 LexiDT。新增 `ComparatorGreedySearch`，并让
Base、Static、SemDT、LexiDT 和 Combined 共用以下控制流：

1. 从当前候选生成所有单词替换；
2. 经过 Kuleshov 原有约束，并在启用时继续经过末尾的 NLI 约束；
3. 按剩余查询预算截断候选；
4. 若启用候选观察器，记录截断后实际即将查询的候选；
5. 批量查询新旧模型；
6. 计算每个候选的 objective score；
7. 若本轮存在成功候选，按统一成功项选择规则返回；
8. 否则由 comparator 取最大值进入下一轮；
9. 查询预算耗尽或无候选时，返回全程最优结果。

比较策略分别为：

- `ScalarComparator`：供 dynamic 和 static 使用，复现原 `GreedySearch` 的
  分数排序、稳定 tie-break 和成功项选择；
- `LexicographicComparator`：按 `LexiScore.as_tuple()` 比较；本轮存在多个
  成功项时，在成功集合中选择修改成本最低者。

Base golden test 必须覆盖候选选择、成功结果、查询数、预算边界、同分候选和
无候选返回，确认统一搜索重构不改变现有 dynamic 行为。

完全相同的分数沿用当前 `Attack.filter_transformations` 排序后的稳定顺序，
不增加随机 tie-breaker。

搜索实现不得使用当前 `BeamSearch` 中会吞掉所有异常的裸 `except`。预期的
搜索终止显式处理，其他异常向上抛出并写入运行日志。

## 5. SemDT NLI 约束

### 5.1 约束接口

新增 `BidirectionalNLI(Constraint)`，默认配置：

```text
model_name_or_path: FacebookAI/roberta-large-mnli
entailment_threshold: 0.90
contradiction_threshold: 0.05
compare_against_original: true
```

模型加载要求：

- 使用 `AutoModelForSequenceClassification` 和 `AutoTokenizer`；
- 从 `model.config.id2label` 查找 entailment 和 contradiction；
- 标签名称大小写归一化，但不硬编码类别编号；
- 找不到必要标签时立即失败；
- `model.eval()`、冻结参数并使用 `torch.inference_mode()`；
- device、dtype、batch size 和 max length 均可配置；
- dtype 支持 `float32`、`float16`、`bfloat16`，不支持时明确降级并记录。

NLI 模型不参与新旧模型预测，也不计入模型对查询预算。

### 5.2 单句和句子对

单个句子生成两个方向：

```text
(original, candidate)
(candidate, original)
```

定义：

\[
S_E(x,x')=\min(p^E_{\rightarrow},p^E_{\leftarrow}),
\qquad
S_C(x,x')=\max(p^C_{\rightarrow},p^C_{\leftarrow}).
\]

句子对任务逐字段比较原始值与候选值，只为实际变化的字段生成 NLI 请求。
若两个字段都变化，则对所有变化字段继续取最小 entailment 和最大
contradiction。不得把 RTE 的 premise 和 hypothesis 拼接后送入 NLI。

为避免读取 `AttackedText._text_input` 私有成员，增加返回只读副本的公共字段
访问接口。NLI 分数以普通 Python 数值写入候选的 `attack_attrs`，供日志和
诊断使用，但不向下一轮候选继承。

接受条件使用闭区间：

\[
S_E\ge\tau_E
\quad\land\quad
S_C\le\tau_C.
\]

### 5.3 批处理、缓存和约束顺序

NLI 必须实现 `_check_constraint_many`，将同轮全部候选、两个方向和所有变化
字段合并成批次。缓存键使用：

```text
(
  model_id,
  model_revision,
  tokenizer_id,
  tokenizer_revision,
  max_length,
  truncation_strategy,
  original_field_text,
  candidate_field_text
)
```

缓存保存双向概率结果，不能只缓存最终布尔值，以便更换阈值时复用。逐次运行
还要记录发生 tokenizer 截断的方向句对数、候选数和比例；截断策略或长度改变
后不得命中旧缓存。

在 Kuleshov 中，NLI 的执行位置为：

```text
MaxWordsPerturbed -> ThoughtVector -> GPT2 -> BidirectionalNLI
```

`RepeatModification` 和 `StopwordModification` 仍是 pre-transformation
constraints。这样 NLI 只处理已经通过全部原约束的候选。

NLI、两个分类模型和 Kuleshov 的 GPT-2 约束可能同时占用显存。首轮默认每张
GPU 一个攻击 worker；实际 batch size 通过小规模 profile 决定，不把某台
机器的值写死为框架默认值。

## 6. 自动阈值标定

### 6.1 流水线边界

阈值标定是离线、可恢复的流水线：

```text
collect -> nli-score -> freeze-split -> annotate -> tune/validate -> audit
```

每个阶段读取上一阶段的不可变 JSONL/Parquet 产物，并写出新的带版本产物。
支持从中间阶段恢复，禁止每次调阈值时重新运行 DT4LM 或重新调用 LLM。

### 6.2 候选收集

每个数据集从训练集选择新旧模型都预测正确的若干原始样本；正式配置的
`sampling.calibration_originals` 使用 `random_up_to`、上限为 500，快速
测试可在独立配置中修改该超参数。实现不得把 500 写入采样代码。使用 Base
配置运行原始 DT4LM。候选观察器必须位于目标模型查询入口，执行顺序固定为：

```text
通过全部原约束
-> 按剩余模型对查询预算截断候选
-> 记录实际即将查询的候选
-> 查询新旧模型
```

观察器不得记录因预算不足被截掉、实际从未进入目标模型的候选。

每条记录至少包含：

- dataset、split、数据集 revision/fingerprint 和原始索引；
- ground-truth label 及标签名称；
- 原始和候选的结构化字段；
- 修改字段、修改位置、修改成本；
- 搜索轮次和候选生成顺序；
- 新旧模型标识；
- recipe、随机种子和查询预算。

以“数据集样本索引 + 结构化候选内容”去重。收集阶段固定串行运行，避免多个
worker 对同一输出文件并发写入。

### 6.3 NLI 打分与分层抽样

离线为所有去重候选计算 \(S_E\)、\(S_C\) 及逐字段双向概率。

候选按 \(S_E\) 和 \(S_C\) 的二维十分位区间分层。初始从非空层中以固定种子
抽取 1000 条，记录每层总体数、抽样数和 inclusion weight。抽样应覆盖边界
区域；计算 precision/recall 时使用分层权重恢复候选总体分布。

在调用 judge 前，按数据集标签和分层单元一次性、确定性地冻结：

- 800 条阈值搜索集；
- 200 条验证集。

两个集合的候选 ID 写入 manifest，后续不得重新划分。只在 800 条搜索集上
选择阈值，200 条验证集只用于一次独立评估。

若验证集中的语义保持正例少于 100 条，不重新混合原验证数据，而从尚未标注
候选中按同一分层规则追加至多 1000 条，形成独立的 `supplemental_audit`
manifest，使该数据集的 Base 候选池总标注量最多为 2000 条。补充集只用于
缩窄独立审计的置信区间和报告稳健性，不参与阈值搜索；原 800/200 划分及
已冻结阈值保持不变。若 800 条搜索集上不存在满足 precision 下限的阈值，
则本次标定失败，不得用补充审计集事后补救；需要重新标定时必须建立新的、
预先固定的数据版本。

阈值冻结后，另从对应后端的 SemDT 正式搜索轨迹中抽取 100 条实际进入 NLI
约束的候选，按接受/拒绝结果和分数区间分层，形成
`trajectory_shift_audit`。用同一 judge 后端独立标注，比较 Base 标定候选
与 SemDT 实际候选的分数分布、加权 precision、recall 和接受率；该审计只
检查搜索轨迹导致的分布偏移，不更新阈值。为支持该步骤，NLI 约束将每个实际检查候选的 ID、
结构化文本、分数、接受状态、搜索轮次和顺序写入追加式审计流；审计流不包含
API key，也不改变候选排序或查询计数。

### 6.4 统一标注接口

定义 `SemanticJudge` 协议，至少提供：

```python
class SemanticJudge(Protocol):
    def annotate(self, examples: Sequence[JudgeExample]) -> list[JudgeResult]:
        ...
```

实现两个独立后端：

- `OpenAIResponsesJudge`
- `HuggingFaceCausalLMJudge`

每次标定实验只实例化一个后端，不混合两个后端的标签。两个后端对同一候选集
分别运行时，各自产生独立的标注文件和阈值文件；额外报告相同候选 ID 上的
一致率。

共享 prompt 必须包含：

- 当前任务定义；
- 标签及其自然语言含义；
- 原始完整输入；
- 候选完整输入；
- “是否保持原输入语义并继续对应原标签”的唯一判断问题。

输出 schema 只包含一个布尔标签，不要求自由文本理由。API 拒答、超时或最终
解析失败的样本保留失败记录，但从阈值搜索中排除，并报告失败率。

### 6.5 OpenAI Responses 后端

首轮默认模型快照：

```text
base_url: https://api.deepseek.com
model: deepseek-v4-pro
```

使用 Responses API 和结构化输出，`temperature=0`，最多重试 3 次。记录：

- 完整模型名和快照；
- base URL；
- 调用日期；
- prompt 文本和 hash；
- SDK 版本；
- request ID、token 用量和延迟；
- 解析、拒答、超时和重试状态。

### 6.6 本地 Hugging Face 后端

首轮默认模型：

```text
Qwen/Qwen2.5-7B-Instruct
```

使用 `AutoModelForCausalLM`、对应 tokenizer 和模型 chat template。生成设置：

- `do_sample=False`，语义上对应 temperature 0；
- 模型参数冻结；
- FP16 或 BF16 按设备能力选择；
- batch size、device、dtype、revision 和 max new tokens 可配置；
- 严格解析同一个布尔 schema；
- 失败时最多进行 3 次格式修复重试，仍失败则丢弃并记录。

OpenAI 和本地 HF 后端必须使用同一 prompt 版本、候选顺序和数据切分。

### 6.7 配置与密钥

新增跟踪的模板：

```text
DT4LM/configs/semantic_judge.example.yaml
```

实际运行使用：

```text
DT4LM/configs/<judge_config_name>.secert.yaml
```

并在仓库根 `.gitignore` 中精确忽略 `**.secert.yaml` 文件。实际文件包含当前选择的 backend、
模型、revision、base URL、API key、超时、重试、HF device/dtype/batch size
等参数。

安全要求：

- 日志和实验产物绝不写入 API key；
- 配置对象的 `repr` 对 key 做脱敏；
- 启动时检查 key 非空，但错误信息不回显值；
- 跟踪的 example 文件只能包含占位符；
- 测试使用假 key 和 mock client，不访问真实 API。

### 6.8 阈值网格搜索

分别对 \(\tau_E\) 和 \(\tau_C\) 在 `[0,1]` 上以 `0.01` 为步长搜索。正类是
“LLM 判断语义和标签均保持”。

选择规则：

1. 计算分层加权 precision 和 recall；
2. 过滤掉 precision 低于配置项 `min_precision` 的组合；
3. `min_precision` 首轮为 `0.95`；
4. 在剩余组合中最大化 recall；
5. recall 相同时依次选择 precision 更高、加权接受数量更多的组合；
6. 若没有组合满足 precision 下限，则标定失败，不静默回退到人工阈值。

验证集只用于报告选定阈值的 precision、recall、混淆矩阵和 bootstrap 95%
置信区间，不再次选择阈值。

每个数据集、每个 judge 后端单独产生阈值文件。SemDT 运行时可以选择：

- `manual`：默认 `0.90/0.05`；
- `calibrated`：读取指定数据集和 judge 后端的阈值产物。

## 7. 数据、模型与实验配置

### 7.1 固定样本清单

为 SST-2 和 RTE 分别生成测试 manifest：

```text
dataset_id
dataset_revision_or_fingerprint
split
old_model_id
new_model_id
seed
test_split_size
eligible_count
sample_count
eligible_indices
selected_indices
```

流程是先扫描预处理后的 test split，保留新旧模型均预测正确的样本，再执行
数据集 YAML 中的 `sampling.test` 策略。当前 SST-2 和 RTE 配置均为 `all`，
按原始索引纳入全部合格样本。任一数据集没有合格样本时失败。五种配置只读取
冻结 manifest，不能各自 shuffle；指标分母读取其中的 `sample_count` 作为
\(N_D\)，不得从配置常量推断。

训练集标定样本使用独立 manifest，不能与测试 manifest 混用。

### 7.2 模型微调

`datasets/preprocess_dataset.py` 将原 SST-2、RTE、MRPC 和 MR notebook
转换为固定种子、本地 `save_to_disk` 的命令行流程。SST-2/RTE 默认输出分别
为 `outputs/datasets/sst2` 和 `outputs/datasets/rte`；训练、manifest 和攻击
均读取同一份本地 DatasetDict，避免训练与评估划分不一致。

SST-2 沿用对称 ALBERT v1/v2 脚本，RTE 新增对称脚本：

- old：`albert/albert-base-v1`
- new：`albert/albert-base-v2`
- 相同预处理、max length、batch size、epoch、learning rate 和 seed；
- 只允许架构版本不同；
- checkpoint 选择规则相同；
- 保存 clean validation/test accuracy 和配置 hash。

初始超参数可沿用 SST-2 的配置，若因 RTE 数据规模调整，只能用 validation
split 决定，并同时应用于 v1/v2，不能根据差分测试结果调参。

### 7.3 运行产物

所有运行写入已被 git 忽略的：

```text
DT4LM/outputs/dt4lm-improvements/<run_id>/
```

目录至少包含：

```text
config.yaml
environment.json
sample_manifest.json
results.jsonl
summary.json
successful_examples/
failed_examples/
nli_profile.json
```

`environment.json` 保存 git commit、dirty 状态、Python/PyTorch/Transformers
版本、CUDA、GPU 和模型 revision。默认只保存本地，不自动 push 到
Hugging Face Hub。

每条结果需要同时保存：

- 原始输入、候选输入和标签；
- 成功/失败状态；
- 新旧预测、概率和可用 logits；
- dynamic/static score 或完整 LexiScore；
- 修改位置和修改成本；
- 模型对查询数；
- NLI 分数及其开销；
- dataset index 和运行配置 hash。

## 8. 指标与统计

### 8.1 自动指标

统一报告：

- EligibilityRate 和 manifest 实际样本数 \(N_D\)；
- Perturbation-induced GSR；
- AMR，只在成功差分输入上计算；
- Model-pair QPS；
- Success@100、Success@500、Success@1000；
- BERTScore；
- BLEU；
- METEOR；
- ROUGE；
- NLI 候选数和方向句对数；
- 端到端 wall-clock time、每个成功输入的总运行时间及相对 Base 的倍数；
- 峰值显存和 NLI tokenizer 截断统计。

质量指标沿用 DT4LM 的既有实现和聚合方式。所有方法读取相同成功结果 schema，
避免为某个实验组使用单独 notebook 逻辑。Model-pair QPS 保持论文兼容口径，
不包含 NLI；资源指标并列呈现 SemDT 的真实额外成本。

### 8.2 SemDT 人工评估

每个数据集目标人工评估 100 个原始样本，SemDT 使用首轮主结果
`SemDT-openai`。按该数据集完整 manifest 的 \(N_D\) 条样本上的实际数量
划分：

- Base 和 SemDT 都成功；
- 仅 Base 成功；
- 仅 SemDT 成功。

100 个人工样本只从三个成功层的并集中抽取；若并集少于 100 条则全部纳入并
报告实际评估数。初始分配比例使用三层样本数除以并集样本数。每个需要单独
估计的非空层至少抽取 5 条；层总体少于 5 条时全量纳入。预留这些下限后，
剩余额度按实际层占比和最大余数法分配，使用固定种子抽样。三个层的含义不同，
不为凑样本而合并；估计时使用实际 inclusion weight。共同成功层同时展示
两个候选，其余层只展示实际成功的方法，不伪造失败方法的文本。

评审时：

- 隐藏方法名称并随机化候选左右顺序；
- 判断候选是否保持原语义并继续对应原标签；
- 至少两名评审者独立判断；
- 报告一致率和 Cohen's kappa；
- 分歧经第三方或共同复核得到最终标签。

对方法 \(M\)，先按该方法全部成功样本的实际层占比估计语义保持率：

\[
\widehat{p}_{M}
=
\frac{
\sum_h N_{M,h}\widehat{p}_{M,h}
}{
N_{\mathrm{success},M}
}.
\]

ValidGSR 直接按各层加权估计：

\[
\widehat{\mathrm{ValidGSR}}_M
=
\frac{1}{N_D}
\sum_h
N_{M,h}\widehat{p}_{M,h},
\]

其中 \(N_{M,h}\) 是方法 \(M\) 在层 \(h\) 的成功总数，
\(\widehat{p}_{M,h}\) 是该层人工样本的语义保持率。

以原始样本为重采样单位，在层内执行 10000 次 bootstrap，保持共同成功层中
Base/SemDT 判断的配对关系，给出语义保持率和 ValidGSR 的 95% 置信区间。

### 8.3 等效判定

- “GSR 基本相同”：绝对差不超过 1 个百分点；
- “AMR/QPS 没有明显恶化”：相对增幅不超过 5%；
- 所有比例同时报告百分点差和相对差，避免混淆。

SemDT 进入扩展实验需要：

1. 人工语义保持率至少提高 5 个百分点；
2. ValidGSR 不低于 Base；
3. QPS 相对增幅不超过 30%；
4. 完整报告 NLI 时间、显存、方向句对数、端到端每成功耗时和相对 Base
   倍数，并在扩展决策中说明资源可行性。

首轮不为端到端时间预设缺少硬件依据的通过阈值，但不得只依据 QPS 宣称
SemDT 成本可接受。

LexiDT 进入扩展实验需要满足任一项：

1. GSR 至少提高 2 个百分点，且 AMR、QPS 均未恶化超过 5%；
2. GSR 与 Base 相差不超过 1 个百分点，QPS 至少下降 10%；
3. GSR 与 Base 相差不超过 1 个百分点，AMR 至少下降 10%。

Combined 只验证两项改动能共同运行，并报告全部指标，不用它替代独立方法的
成功标准。

## 9. 文件级实施清单

以下是预期落点；实现时可按现有命名风格微调文件名，但不能改变模块边界。

### 9.1 核心代码

修改：

- `DT4LM/textattack/attack_args.py`
  - 注册 objective、NLI、阈值文件和设备相关参数。
- `DT4LM/textattack/model_args.py`
  - 统一新旧模型 wrapper 的加载和分类输出校验。
- `DT4LM/textattack/dataset_args.py`
  - 支持按固定 manifest 选择样本。
- `DT4LM/textattack/attack_recipes/pair_2024.py`
  - 按两个正交开关组装 objective、search 和 NLI constraint。
- `DT4LM/textattack/goal_functions/goal_function.py`
  - 整理双模型批处理、缓存、查询计数和候选观察接口。
- `DT4LM/textattack/goal_functions/classification/classification_goal_function.py`
  - 增加差分模式下保留 logits/probabilities 的处理。
- `DT4LM/textattack/goal_functions/classification/differential_classification.py`
  - 共享成功条件并委托给 objective strategy。
- `DT4LM/textattack/goal_function_results/`
  - 保存双模型输出、objective 分量和修改成本。
- `DT4LM/textattack/shared/attacked_text.py`
  - 增加安全的结构化字段读取和修改成本 helper。
- `DT4LM/textattack/attacker.py`
  - 写出统一结果 schema，默认关闭外部 push。
- `DT4LM/textattack/loggers/csv_logger.py`
  - 展开 LexiScore、新旧模型和 NLI 字段。

新增：

- `DT4LM/textattack/goal_functions/classification/differential_objectives.py`
  - dynamic、static、lexicographic objective。
- `DT4LM/textattack/search_methods/comparator_greedy_search.py`
  - 所有 objective 共用的贪心状态机。
- `DT4LM/textattack/search_methods/differential_comparators.py`
  - `ScalarComparator` 和原生五元组比较的
    `LexicographicComparator`。
- `DT4LM/textattack/constraints/semantics/bidirectional_nli.py`
  - 批量双向 NLI 约束。
- `DT4LM/textattack/models/classification_output.py`
  - logits/probabilities 统一结构和 margin 计算。
- `DT4LM/textattack/metrics/attack_metrics/differential_metrics.py`
  - EligibilityRate、Perturbation-induced GSR、QPS、Success@B 和 AMR。

同步更新各包 `__init__.py` 导出。

### 9.2 标定与配置

新增：

- `DT4LM/textattack/semantic_validation/schemas.py`
- `DT4LM/textattack/semantic_validation/judges/base.py`
- `DT4LM/textattack/semantic_validation/judges/openai_responses.py`
- `DT4LM/textattack/semantic_validation/judges/huggingface_causal.py`
- `DT4LM/textattack/semantic_validation/candidate_collection.py`
- `DT4LM/textattack/semantic_validation/threshold_search.py`
- `DT4LM/textattack/semantic_validation/distribution_audit.py`
- `DT4LM/textattack/commands/semdt_calibration_command.py`
- `DT4LM/configs/semantic_judge.example.yaml`

修改：

- `DT4LM/textattack/commands/textattack_cli.py`
  - 注册可分阶段恢复的 `semdt-calibrate` 命令。
- `.gitignore`
  - 忽略 `DT4LM/configs/semantic_judge.local.yaml`。
- `DT4LM/requirements.txt`、`DT4LM/DT4LM.yaml`
  - 增加并锁定经测试的 OpenAI SDK 等依赖。

### 9.3 实验和统计

新增：

- `DT4LM/datasets/preprocess_dataset.py`
- `DT4LM/datasets/prepare_adversarial_training.py`
- `DT4LM/experiments/finetune/train_albertbasev1_rte.sh`
- `DT4LM/experiments/finetune/train_albertbasev2_rte.sh`
- `DT4LM/experiments/improvements/configs/sst2.yaml`
- `DT4LM/experiments/improvements/configs/rte.yaml`
- `DT4LM/experiments/improvements/configs/experiments/*.yaml`
- `DT4LM/experiments/improvements/prepare_manifests.sh`
- `DT4LM/experiments/improvements/calibrate_semdt.sh`
- `DT4LM/experiments/improvements/run_first_round.sh`
- `DT4LM/statistics/evaluate_improvements.py`
- `DT4LM/statistics/sample_human_evaluation.py`
- `DT4LM/statistics/analyze_human_evaluation.py`

manifest 脚本接收一个数据集配置；标定脚本接收一个数据集配置和一个私密
judge 配置；实验脚本接收一个数据集配置和一个单实验配置。脚本不内嵌个人
用户名、API key、实验矩阵或绝对模型路径。

`HuggingFaceDataset` 和 manifest 生成器同时支持 Hub ID 与
`DatasetDict.save_to_disk` 目录。预处理脚本必须允许显式 `--output-dir`，
并在目录已存在时拒绝覆盖。

## 10. 测试方案

### 10.1 单元测试

模型输出：

- HF wrapper 从 `.logits` 显式声明 logits；
- 自定义 wrapper 缺少或错误声明 `score_type` 时失败；
- 数值恰好像概率的 logits 不会被启发式改判；
- 概率 fallback 使用 log margin；
- 二分类和多分类 margin 方向正确；
- 新旧模型类别数或标签映射不一致时失败。

目标函数：

- dynamic 与当前实现的固定样例完全一致；
- static 等于旧模型真标签概率减新模型真标签概率；
- LexiScore 依次比较旧模型实际正确状态、旧 margin、新模型实际错误状态、
  新 margin 和修改成本；
- margin 为 0 或 argmax 并列时，显式预测状态仍决定正确顺序；
- 成功条件只接受“旧正确、新错误”；
- 修改成本使用原始词数和 `modified_indices`。

搜索：

- dynamic 的候选选择、成功结果和查询数与原搜索 golden trace 一致；
- Base、Static 和 LexiDT 确实经过同一 `ComparatorGreedySearch`；
- LexiDT 本轮多个成功项时选择成本最低者；
- 五元组相同时保持候选稳定顺序；
- 查询预算在批次边界正确截断；
- 无候选、预算耗尽和异常分别得到正确状态。

NLI：

- 任意 `id2label` 编号都能正确识别标签；
- 单句双向 min/max 聚合正确；
- RTE 只检查变化字段；
- 两字段变化时跨字段 min/max 正确；
- 阈值边界是 `>=` 和 `<=`；
- batch 与逐条计算一致；
- 缓存不会因阈值变化失效；
- tokenizer revision、max length 或截断策略变化会使旧缓存失效；
- 截断方向句对数和候选数被正确累计。

标注和调参：

- API 结构化成功、拒答、超时、重试和脱敏；
- HF chat template、确定性生成和解析失败；
- 800/200 manifest 在标注前冻结且不可重新划分；
- 验证正例少于 100 时只追加独立审计集，Base 候选池总标注量不超过 2000；
- 补充审计和轨迹审计都不能改变已选阈值；
- 分层权重、固定 80%/20% 划分和网格 tie-break 正确；
- precision 下限无可行解时明确失败。

指标：

- QPS 使用全部 \(N_D\) 条的查询总数除以成功数；
- GSR、Success@B 和 ValidGSR 的分母读取 manifest 的 \(N_D\)；
- EligibilityRate 使用 test split 总数而不是 \(N_D\)；
- 成功数为 0 时 QPS 为 `N/A`；
- 分层 ValidGSR 和 bootstrap 保持公共层配对；
- 人工抽样对可单独估计的非空层至少取 5 条，小于 5 条时全量纳入。

### 10.2 集成与回归测试

使用小型 fake model、fake tokenizer、fake NLI 和固定 transformation：

- 跑通 Base、Static、SemDT、LexiDT、Combined 五种配置；
- 验证 SemDT 的 NLI 在原约束之后、模型查询之前执行；
- 验证候选观察器在预算截断之后、目标模型查询之前执行；
- 验证 LexiDT 不经过 NumPy 标量排序；
- 验证 dynamic/static/lexi 只切换 comparator，不切换搜索状态机；
- 验证模型和 NLI batch 不改变逻辑查询数；
- 验证 manifest 在五种配置中产生完全相同的数据顺序；
- 验证 SST-2 manifest 固定 500 条、RTE manifest 纳入全部 eligible 样本；
- 验证完整结果 schema 可被统计脚本读取。

网络相关测试全部 mock。另设手工 smoke test，分别加载
`FacebookAI/roberta-large-mnli`、OpenAI Responses 和
`Qwen/Qwen2.5-7B-Instruct`，但不作为默认 CI。

现有 `pair` 命令在不提供新参数时必须通过回归测试，避免破坏原 DT4LM 使用
方式。

## 11. 实施顺序与阶段出口

### 阶段 0：冻结基线

1. 为当前 dynamic 目标和 `GreedySearch` 搜索轨迹建立 golden tests。
2. 固定 Kuleshov 参数、模型角色、候选选择和当前查询计数。
3. 创建统一运行配置和本地输出 schema。

出口：现有 Base 能在小型 fixture 上重复得到相同结果。

### 阶段 1：输出与指标基础设施

1. 实现 `ClassificationModelOutput`。
2. 整理模型对加载、缓存和结果记录。
3. 实现修改成本、EligibilityRate、Perturbation-induced GSR、AMR、QPS
   和 Success@B。
4. 实现测试/train manifest。

出口：Base 结果不变，且所有新指标可由结构化结果重算。

### 阶段 2：Static 与 LexiDT

1. 把 dynamic/static/lexi 抽成 objective strategy。
2. 实现 `LexiScore`、两个 comparator 和统一
   `ComparatorGreedySearch`。
3. 加入 CLI 参数和 LexiDT 实验配置。
4. 完成单元、集成和 deterministic tie 测试。

出口：Base、static、LexiDT 在 fake model 上满足手算结果；LexiDT 可在 SST-2
少量真实样本上完成 smoke test。

### 阶段 3：SemDT 在线约束

1. 实现双向 NLI scorer、batch 和 cache。
2. 实现单句/句子对字段聚合。
3. 将 NLI 追加到 Kuleshov 原约束末尾。
4. 增加时间、显存和 NLI 次数统计。

出口：SemDT 和 Combined 在 SST-2/RTE 小样本上运行，且约束顺序和阈值边界
通过测试。

### 阶段 4：离线标定

1. 实现候选收集和 NLI 离线打分。
2. 实现 OpenAI Responses judge。
3. 实现本地 HF judge。
4. 实现固定 800/200 分层切分、独立补充审计、网格搜索和验证报告。
5. 加入本地 secret config 及脱敏测试。

出口：两个后端分别完成一次小规模端到端标定；产物可恢复、可审计，仓库中
没有被跟踪的密钥。

### 阶段 5：首轮实验

1. 完成 SST-2/RTE ALBERT v1/v2 微调和 clean accuracy 检查。
2. 生成并冻结 train/test manifests。
3. 分别用 OpenAI 和 HF 后端标定阈值。
4. 按相同 manifest 运行 Base、Static、LexiDT。
5. 独立运行 SemDT-manual、SemDT-openai 和 SemDT-hf。
6. 阈值冻结并完成对应 SemDT 运行后，分别执行两个 judge 后端的 100 条
   实际轨迹分布审计。
7. 主方法结果完成后，使用 OpenAI 阈值运行 Combined。
8. 生成自动指标、资源开销、截断统计和 judge 一致性报告。

出口：五种逻辑配置及 SemDT 的三个阈值变体均有完整配置、环境、逐样本记录
和汇总，能够从原始产物重新计算全部指标。

### 阶段 6：人工评估与结论

1. 按三层实际占比抽样，每个数据集目标 100 个原始样本；并集不足时全量纳入。
2. 完成盲评、分歧处理和一致性统计。
3. 计算加权语义保持率、ValidGSR 和 bootstrap 区间。
4. 按预定义阈值判断 SemDT、LexiDT 是否进入扩展实验。

出口：两个研究问题都有独立结论，不使用 Combined 结果替代消融证据。

## 12. 完成标准

工程完成需要同时满足：

- 五种配置通过单元和集成测试；
- 默认 `pair` 命令保持向后兼容；
- 所有实验由配置和 manifest 可重复运行；
- 结果可在不重新查询模型或 LLM 的情况下重算指标和阈值；
- API key 未进入 git、日志、异常或实验产物；
- 新旧模型、NLI 和 judge 的完整 revision 均被记录；
- 代码中不再依赖手工替换目标函数来运行 static 或 LexiDT；
- 没有吞掉搜索异常或静默回退阈值的路径；
- README 补充安装、配置、标定和五种配置的实验命令。

研究结论完成需要：

- SST-2 有固定 500 条合格测试样本，RTE 使用冻结 manifest 中的全部合格
  测试样本；
- 每个方法使用相同 1000 查询预算；
- SemDT 同时具有 manual、OpenAI 和本地 HF 三个独立阈值运行结果；
- OpenAI 和 HF 运行分别完成冻结阈值后的实际轨迹分布审计；
- LexiDT 与 dynamic/static 使用相同候选生成、约束和搜索状态机；
- 自动指标、资源指标和人工评估均按本文固定口径报告；
- 是否扩展到 LEAP、FastGA 和其他模型架构由预注册成功标准决定。
