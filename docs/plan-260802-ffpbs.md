# 论文方案：Feasibility-First Pareto Search for Differential Testing of Language Model Updates

## 1. 论文定位

主方法名为：FF-PBS: Feasibility-First Pareto Beam Search

论文围绕一个核心判断展开：差分输入的最终结果必须满足旧模型正确，但搜索路径上的每个中间状态不必始终满足旧模型正确。

因此，FF-PBS并不是放松最终差分条件，而是避免把**终点约束**错误地强化为**路径约束**。

论文的研究对象保持为语言模型更新的差分测试。给定旧模型 (M_o)、更新后的新模型 (M_n)、输入 (x) 及真实标签 (y)，目标是寻找语义保持的修改输入 (x')，使：

[
M_o(x')=y,
\qquad
M_n(x')\neq y.
]

FF-PBS保持这一成功条件、文本变换和原有语义约束不变，只替换DT4LM中的：

1. 动态标量目标；
2. 单路径贪心搜索。

---

# 2. 核心思想

## 2.1 DT4LM存在的两个问题

### 问题一：将不同性质的要求压缩为单一标量

DT4LM通过动态目标组合旧模型和新模型的输出，再按该分数选择候选。这样会把以下不同性质的要求混合起来：

* 旧模型最终必须正确：约束；
* 新模型应尽可能接近错误边界：搜索目标；
* 修改幅度应尽量小：质量目标。

单一标量需要人为决定它们如何折中，并且只能为所有候选给出一个全序。

### 问题二：单路径贪心会永久丢弃可恢复路径

DT4LM每轮只保留一个候选。一次局部选择后，其他修改路径永久消失。

更重要的是，有效差分输入可能通过如下路径到达：

[
\text{旧对、新对}
\rightarrow
\text{旧错、新对}
\rightarrow
\text{旧对、新错}.
]

中间状态暂时使旧模型出错，但后续修改又恢复了旧模型，并最终使新模型出错。

如果把旧模型正确性强制施加到所有中间状态，第二个状态就会被永久删除，最终成功状态也因此不可达。

---

## 2.2 终点可行性与路径可行性

定义最终成功条件：

[
\operatorname{Success}(x')
=
\mathbf 1[M_o(x')=y]
\cdot
\mathbf 1[M_n(x')\neq y].
]

这是不可放松的终点条件。

定义当前状态的可行性：

[
\phi(x')
=
\mathbf 1[M_o(x')=y].
]

FF-PBS优先保留满足 (\phi(x')=1) 的候选，但不要求整个搜索路径满足：

[
\phi(x_t)=1,\qquad \forall t.
]

核心区别是：

[
\text{最终解必须可行}
\quad\not\Rightarrow\quad
\text{每个中间状态必须可行}.
]

这构成全文最重要的观点。

---

## 2.3 约束化双目标建模

FF-PBS将差分输入搜索建模为：

[
\begin{aligned}
\max_{x'} \quad
&m_{\mathrm{new}}(x'),\
\min_{x'} \quad
&c(x,x'),\
\text{s.t.}\quad
&M_o(x')=y.
\end{aligned}
]

其中，新模型错误margin为：

[
m_{\mathrm{new}}(x')
=
\max_{k\neq y}z_{\mathrm{new},k}(x')
-z_{\mathrm{new},y}(x').
]

当：

[
m_{\mathrm{new}}(x')>0
]

时，新模型已经预测错误。

修改成本沿用DT4LM的词级修改率：

[
c(x,x')
=
\frac{|\operatorname{modified_indices}(x')|}
{N_{\mathrm{words}}(x)}.
]

因此：

* 旧模型正确性是最终约束；
* 新模型错误margin是测试目标；
* 修改率是输入质量目标。

---

## 2.4 Feasibility-first候选保留

设当前所有候选为 (C)，划分为：

[
C_F={x'\in C:M_o(x')=y},
]

[
C_I={x'\in C:M_o(x')\neq y}.
]

FF-PBS首先从 (C_F) 中选择候选。

对可行候选，根据以下两个方向进行Pareto非支配排序：

[
\left(m_{\mathrm{new}}(x'),-c(x,x')\right).
]

候选 (a) 支配候选 (b)，当且仅当：

[
m_{\mathrm{new}}(a)\ge m_{\mathrm{new}}(b),
]

[
c(x,a)\le c(x,b),
]

且至少一个不等式严格成立。

如果可行候选不足以填满大小为 (K) 的frontier，才从不可行候选中补位。

旧模型正确margin定义为：

[
m_{\mathrm{old}}(x')
=
z_{\mathrm{old},y}(x')
-
\max_{k\neq y}z_{\mathrm{old},k}(x').
]

不可行候选的违反程度为：

[
v(x')=\max(0,-m_{\mathrm{old}}(x')).
]

不可行候选按照：

1. 违反程度更小；
2. 新模型错误margin更大；
3. 修改成本更低；

的顺序补充。

因此，选择关系是：

[
\text{可行候选}
\succ
\text{不可行候选}.
]

不可行候选不会挤掉任何本可以保留的可行候选，只在frontier有剩余空间时保留潜在恢复路径。

---

## 2.5 异步有界frontier

FF-PBS维护大小不超过 (K) 的frontier：

[
B_t={x_t^{(1)},\ldots,x_t^{(K)}}.
]

每次只扩展一个状态：

1. 从frontier中选择优先级最高的状态；
2. 生成其所有合法词级修改；
3. 应用DT4LM原有约束；
4. 查询新旧模型；
5. 检查是否出现严格成功输入；
6. 将未扩展状态和新状态合并；
7. 按feasibility-first Pareto规则裁剪到 (K)。

异步frontier允许同时保留不同修改深度的状态。因此，修改成本不再只是同一深度候选之间几乎恒定的量，而会真实参与浅层与深层路径的比较。

---

# 3. FF-PBS对DT4LM工作流的修改

## 3.1 原始DT4LM

```text
原始输入
  ↓
生成词级候选
  ↓
应用原有修改与语义约束
  ↓
查询新旧模型
  ↓
计算动态标量目标
  ↓
选择分数最高的一个候选
  ↓
继续修改或终止
```

## 3.2 FF-PBS

```text
原始输入
  ↓
初始化有界frontier
  ↓
选择一个待扩展状态
  ↓
生成词级候选
  ↓
应用原有修改与语义约束
  ↓
查询新旧模型
  ↓
计算旧模型可行性、新模型margin和修改率
  ↓
检查严格差分成功条件
  ↓
可行候选进行Pareto排序
  ↓
不可行候选按违反程度补位
  ↓
更新frontier并继续搜索
```

## 3.3 保持不变的部分

| 模块     | 处理            |
| ------ | ------------- |
| 差分输入定义 | 完全沿用DT4LM     |
| 词级变换   | 沿用对应Recipe    |
| 停用词约束  | 沿用            |
| 重复修改约束 | 沿用            |
| 最大修改比例 | 沿用            |
| 原有语义约束 | 沿用            |
| 流畅性约束  | 沿用            |
| 查询预算   | 对所有方法保持一致     |
| 成功样本评估 | 沿用并补充ValidGSR |
| 后续模型修复 | 可继续使用生成的差分输入  |

## 3.4 实质修改

| 模块        | DT4LM    | FF-PBS         |
| --------- | -------- | -------------- |
| 目标表示      | 动态标量     | 最终约束加Pareto双目标 |
| 搜索状态数     | 1        | 最多 (K) 个       |
| 中间旧模型错误状态 | 由标量间接处理  | 可行优先，必要时补位保留   |
| 候选深度      | 单一路径     | 可同时包含不同深度      |
| 修改率作用     | 主要用于结果评价 | 直接参与frontier选择 |

---

# 4. 论文核心贡献

建议将贡献压缩为三项。

## 贡献一：揭示终点可行性与路径可行性的错位

我们指出，在语言模型更新的差分测试中：

* 旧模型正确是最终差分输入必须满足的条件；
* 但强制每个中间修改状态都保持旧模型正确，会删除后续可能恢复的路径。

这一问题在离散、非单调的文本修改空间中尤其明显。

不要声称这是新的通用约束优化理论，而应强调这是对**模型更新差分测试搜索结构的具体诊断**。

## 贡献二：提出FF-PBS

提出面向语言模型更新测试的feasibility-first Pareto搜索：

* 将旧模型正确性作为最终约束；
* 在新模型错误margin和修改成本之间维护Pareto候选；
* 优先保留可行状态；
* 仅在frontier有空位时，以最小违反原则保留不可行状态；
* 使用异步有界frontier探索多条不同深度的修改路径。

## 贡献三：通过机制实验验证可恢复路径的作用

在多个数据集、模型更新和查询预算下验证：

* FF-PBS比原始动态贪心发现更多差分输入；
* 单纯增加beam宽度不足以解释提升；
* 硬删除不可行状态会丢失有效路径；
* FF-PBS的额外成功与非贪心路径及可恢复的不可行中间状态相关；
* 新增差分输入仍具有可接受的语义保持性和修改质量。

---

# 5. 建议的论文主线

## 引言中的逻辑

1. 模型更新可能引入只在特定输入变体上出现的回归；
2. DT4LM通过搜索“旧模型正确、新模型错误”的输入检测这些回归；
3. 现有方法依赖动态标量和单路径贪心；
4. 差分成功条件是终点条件，文本修改过程却是非单调的；
5. 某条有效路径可能暂时离开可行区域后重新返回；
6. 硬过滤会遗漏这些路径，完全无约束搜索又会浪费预算；
7. FF-PBS通过“可行优先、不可行补位”的Pareto frontier解决这一问题；
8. 实验验证其覆盖率、效率、路径机制和输入质量。

## 一句话主线

> **FF-PBS严格保证最终差分输入有效，但不对中间搜索路径施加不必要的硬可行性，从而发现单路径或硬剪枝搜索遗漏的模型更新回归。**

---

# 6. 论文结构

## 1 Introduction

重点介绍：

* 语言模型更新回归测试；
* DT4LM的作用及局限；
* 终点约束与路径约束的区别；
* FF-PBS直觉；
* 贡献。

## 2 Background and Related Work

### 2.1 Testing Language Model Updates

介绍差分测试、回归输入和DT4LM。

### 2.2 Adversarial Text Search

介绍词级变换、贪心搜索、beam和进化搜索。

### 2.3 Constrained Multi-objective Search

只介绍必要背景：

* Pareto dominance；
* feasibility-first约束处理；
* 有界frontier。

避免把论文写成通用多目标优化论文。

## 3 Problem Formulation

定义：

* 新旧模型；
* 差分输入；
* 修改成本；
* 查询预算；
* 旧模型可行性；
* 新模型错误margin；
* 受约束双目标问题。

## 4 Motivation: Terminal versus Path Feasibility

建议包括：

* 一张三步路径示意图；
* 一个真实或构造案例；
* Hard filtering遗漏路径的形式化说明；
* 初步统计：不同任务中旧模型错误中间状态出现频率不同。

可以给出一个简单命题：

> 若从原始输入到某个成功输入的所有可达路径均包含至少一个旧模型错误的中间状态，则任何永久删除该类状态的搜索都无法找到该成功输入；FF-PBS在frontier容量允许时仍可能保留该路径。

## 5 Method

### 5.1 Overview

给出工作流图。

### 5.2 State Representation and Model Margins

定义状态、可行性、margin和修改成本。

### 5.3 Pareto Ranking of Feasible States

定义双目标和非支配排序。

### 5.4 Feasibility-First Frontier Completion

定义可行状态优先和不可行状态补位。

### 5.5 Asynchronous Frontier Search

给出算法和伪代码。

### 5.6 Complexity

设每次扩展后参与选择的状态数为 (n)，简单非支配排序为：

[
O(n^2).
]

由于frontier大小 (K) 很小，主要成本仍然来自模型查询，而不是Pareto排序。

## 6 Experimental Setup

统一介绍数据、模型、基线、预算、指标和统计检验。

## 7 Results

按下文实验顺序组织。

## 8 Discussion

讨论：

* 何时feasibility-first收益最大；
* GSR与查询成本的权衡；
* 语义质量；
* 对不同模型架构的适用性；
* 失败情形。

## 9 Threats to Validity

包括：

* 二分类任务占比较高；
* 词级替换空间限制；
* 旧模型被当作行为参考；
* 自动语义指标局限；
* 查询预算与frontier宽度设置。

## 10 Conclusion

---

# 7. 实验设置

## 7.1 数据集

### 核心可比实验

沿用DT4LM的四个数据集：

* SST-2；
* MR；
* RTE；
* MRPC。

可以再增加一个多分类数据集，例如AG News。

---

## 7.2 模型更新对

* ALBERT v1 → ALBERT v2；
* DeBERTa → DeBERTa-v3；
* GPT-1 → GPT-2；

所有模型使用统一序列分类输出接口，并在相同数据划分和训练配置下完成任务适配。

---

## 7.3 搜索配置

主实验统一使用：

* 相同文本变换；
* 相同原有约束；
* 相同候选数量；
* 相同模型批大小；
* 每个样本相同模型对查询预算；
* 相同测试manifest及样本顺序。

主配置：

[
K=5.
]

查询预算：

[
Q=1000.
]

数据集测试样本：

* 最多随机选择1000条；
* 测试集不足时使用全部；
* 原始差分输入沿用DT4LM协议记为skipped；

---

## 7.4 对比方法

### 外部或原始基线

1. **DT4LM-Dynamic（Base）**
   原论文动态目标加单路径贪心，主要基线。

2. **原DT4LM其他Recipe**
   在可行的情况下报告Kuleshov、LEAP和FastGA配置。

### 内部对照

1. **Dynamic-Beam**
   使用与FF-PBS相同的异步frontier，但仍按DT4LM动态分数排序，用于隔离多路径搜索本身的影响。

2. **Hard-PBS**
   使用Pareto frontier，但永久删除所有旧模型错误状态，用于验证路径硬约束的影响。

3. **FF-Pareto-Greedy**
   使用与FF-PBS相同的feasibility-first策略，但：

   [
   K=1.
   ]

   用于隔离多路径frontier的贡献。

4. **FF-PBS**
   完整方法，(K=5)。

---

## 7.5 指标

设：

- \(S\)：成功生成差分输入的样本数；
- \(F\)：搜索失败的样本数；
- \(K_s\)：原始输入已经满足差分条件、因而被跳过的样本数；
- \(A=S+F\)：实际进入搜索的 attackable 样本数；
- \(N=S+F+K_s\)：测试 manifest 中的样本总数；
- \(Q\)：每个样本的最大模型对查询预算；
- \(q_i\)：样本 \(i\) 实际消耗的模型对查询数；
- \(q_i^{\mathrm{succ}}\)：成功样本首次达到差分条件时的累计查询数。

当指标分母为 0 时，结果记为 `N/A`，不记为 0。

### 主要有效性

Paper GSR 定义为：

\[
\mathrm{GSR}
=
\frac{S}{A}.
\]

它衡量在实际进入搜索的样本中，方法成功生成差分输入的比例。

Sample generation rate 定义为：

\[
\mathrm{SGR}
=
\frac{S}{N}.
\]

它将原始差分样本也计入总体规模，用于反映整个测试集上的差分输入生成覆盖率。

### 查询效率

QPS 沿用 DT4LM 的定义：

\[
\mathrm{QPS}
=
\frac{\sum_{i=1}^{N} q_i}{S}.
\]

QPS 的分子包含 successful、failed 和 skipped 样本产生的全部模型对查询；若 \(S=0\)，则记为 `N/A`。

对于查询预算 \(B\le Q\)，定义：

\[
\mathrm{Success@B}
=
\frac{
\sum_{i=1}^{A}
\mathbf 1[
i\text{ 成功}
\land
q_i^{\mathrm{succ}}\le B
]
}{
A
}.
\]

正文报告：

\[
B\in\{100,200,\ldots,1000\}.
\]

Success-query AUC 定义为 Success@B 曲线在完整查询预算上的离散平均：

\[
\mathrm{AUC}_{\mathrm{query}}
=
\frac{1}{Q}
\sum_{B=1}^{Q}\mathrm{Success@B}.
\]

等价地：

\[
\mathrm{AUC}_{\mathrm{query}}
=
\frac{
\sum_{i:\,\mathrm{success}}
\left(Q-q_i^{\mathrm{succ}}+1\right)
}{
A Q
}.
\]

该指标同时奖励较高的最终成功率和较早的成功时间。

为在全部 attackable 样本上联合比较成功率和查询效率，定义预算惩罚查询成本。对每个 attackable 样本：

\[
\widetilde q_i
=
\begin{cases}
q_i^{\mathrm{succ}},
& i\text{ 成功},\\
Q,
& i\text{ 失败}.
\end{cases}
\]

预算惩罚查询成本为：

\[
\mathrm{BPQC}
=
\frac{1}{A}
\sum_{i=1}^{A}\widetilde q_i.
\]

同时可报告归一化形式：

\[
\mathrm{nBPQC}
=
\frac{\mathrm{BPQC}}{Q}.
\]

其中：

- 较低的 BPQC 表示方法能够在更少查询内覆盖更多样本；
- 即使某个失败样本因无候选而提前终止，仍按 \(Q\) 计，以避免“更早放弃”被错误解释为查询效率提升；
- skipped 样本不进入 BPQC。

Success-query AUC 和 BPQC 表达的是近似相同的成功率—效率权衡。正文以 Success-query 曲线和 BPQC 为主，AUC 可作为汇总值或放入附录，不应将三者解释为彼此独立的证据。

### 修改质量

修改率沿用 DT4LM 的词级定义。成功样本的平均修改率为：

\[
\mathrm{AMR}
=
\frac{1}{S}
\sum_{i:\,\mathrm{success}}c(x_i,x'_i).
\]

同时报告：

- BERTScore；
- BLEU；
- METEOR；
- ROUGE-L。

自动相似度指标仅作为辅助，不能代替人工语义有效性评价。

### 机制指标

设 FF-PBS 在某个成功样本上的搜索路径为：

\[
x_0\rightarrow x_1\rightarrow\cdots\rightarrow x_L,
\]

其中 \(x_0\) 是原始输入，\(x_L\) 是最终成功差分输入。

#### 非贪心路径比例

令 \(r_i\) 为成功路径第一步候选在 DT4LM 动态目标下的稳定排名，则：

\[
R_{\mathrm{non\text{-}top1}}
=
\frac{
\#\{i:i\text{ 成功且 }r_i>1\}
}{
S_{\mathrm{FF}}
}.
\]

它衡量 FF-PBS 的成功中，有多少不是沿原动态贪心的首选分支得到的。

对于 FF-PBS 成功而 Base 失败的独有成功集合 \(U_{\mathrm{FF}}\)，定义：

\[
R_{\mathrm{unique\text{-}non\text{-}top1}}
=
\frac{
\#\{i\in U_{\mathrm{FF}}:r_i>1\}
}{
|U_{\mathrm{FF}}|
}.
\]

该指标更直接衡量新增成功是否来自原贪心会丢弃的路径。

#### 可恢复不可行路径比例

只检查 root 之后、最终成功状态之前的中间状态：

\[
R_{\mathrm{recover}}
=
\frac{
\#\left\{
i:
\exists j\in\{1,\ldots,L_i-1\},
M_o(x_{i,j})\neq y_i
\right\}
}{
S_{\mathrm{FF}}
}.
\]

对于 FF-PBS 独有成功，定义：

\[
R_{\mathrm{unique\text{-}recover}}
=
\frac{
\#\left\{
i\in U_{\mathrm{FF}}:
\exists j\in\{1,\ldots,L_i-1\},
M_o(x_{i,j})\neq y_i
\right\}
}{
|U_{\mathrm{FF}}|
}.
\]

这两个指标分别对应：

- `post_root_old_prediction_error_path_rate`；
- `unique_post_root_old_prediction_error_path_rate`。

root 不计入该指标，以避免原始输入本身的预测状态干扰路径恢复分析。

#### 不可行状态补位频率

设 FF-PBS 共执行 \(T\) 次非空 frontier 更新，第 \(t\) 次更新后的 frontier 为 \(B_t\)，其中旧模型预测错误的状态集合为：

\[
I_t=\{s\in B_t:M_o(s)\neq y\}.
\]

不可行补位事件率定义为：

\[
R_{\mathrm{fill}}
=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbf 1[|I_t|>0].
\]

对应 `infeasible_fill_event_rate`。它表示多少次 frontier 更新需要使用不可行状态补足剩余容量。

不可行状态保留率定义为：

\[
R_{\mathrm{infeasible\text{-}retained}}
=
\frac{
\sum_{t=1}^{T}|I_t|
}{
\sum_{t=1}^{T}|B_t|
}.
\]

对应 `infeasible_retained_state_rate`。它表示所有被保留的 frontier 状态槽位中，有多少由不可行状态占据。

#### 硬剪枝损失

Hard-PBS 相对 FF-PBS 的硬剪枝损失定义为：

\[
L_{\mathrm{hard}}
=
\mathrm{GSR}_{\mathrm{FF\text{-}PBS}}
-
\mathrm{GSR}_{\mathrm{Hard\text{-}PBS}}.
\]

使用百分点报告，而不是相对百分比。

Hard-PBS 的不可行状态丢弃率定义为：

\[
R_{\mathrm{hard\text{-}discard}}
=
\frac{
\#\{\text{Hard-PBS评估后因旧模型错误而删除的post-root状态}\}
}{
\#\{\text{Hard-PBS评估的全部post-root状态}\}
}.
\]

该指标用于分析硬剪枝出现得越频繁时，Hard-PBS 的性能损失是否越大。

#### Frontier 宽度与多样性

平均 frontier 宽度定义为：

\[
W_{\mathrm{frontier}}
=
\frac{1}{T}\sum_{t=1}^{T}|B_t|.
\]

修改位置多样性采用归一化定义：

\[
D_{\mathrm{modified}}
=
\frac{1}{T}
\sum_{t=1}^{T}
\frac{
\left|
\{\operatorname{modified\_indices}(s):s\in B_t\}
\right|
}{
|B_t|
}.
\]

深度多样性定义为：

\[
D_{\mathrm{depth}}
=
\frac{1}{T}
\sum_{t=1}^{T}
\frac{
\left|
\{\operatorname{depth}(s):s\in B_t\}
\right|
}{
|B_t|
}.
\]

二者均位于 \((0,1]\)：

- 越接近 1，表示 frontier 中的状态越多样；
- 越接近 \(1/|B_t|\)，表示 frontier 中状态高度重复。

另报告平均 Pareto 第一前沿大小：

\[
W_{\mathrm{rank1}}
=
\frac{1}{T}
\sum_{t=1}^{T}|P_1(B_t)|.
\]

### 资源开销

报告：

- wall-clock time；
- 峰值显存；
- 模型对总查询数；
- frontier 排序总耗时；
- frontier 排序耗时占总运行时间的比例。

---

## 7.6 统计分析

所有方法使用相同样本，因此采用配对统计。

### 成功率

* McNemar检验；
* 配对bootstrap 95%置信区间；
* Base独有成功、FF-PBS独有成功、共同成功和共同失败。

### 查询与修改率

在共同成功样本上：

* Wilcoxon signed-rank test；
* 配对中位数差；
* bootstrap 95%置信区间。

在所有attackable样本上：

* 比较budget-penalized query cost；
* 报告配对均值和中位数差。

最终多数据集、多方法检验可以使用Holm校正。

---

# 8. 实验设计

以下每个实验均按照“做什么—展示什么—分析什么”组织。

---

## E1：总体差分测试效果

### 目的

验证FF-PBS在固定查询预算下，是否比DT4LM发现更多模型更新回归。

### 做什么

在所有数据集和模型更新对上比较：

* DT4LM-Dynamic；
* FF-PBS。

保持：

* 测试样本；
* 变换和约束；
* 查询预算；
* 模型checkpoint；

完全一致。

### 展示什么

主表每个数据集—模型对报告：GSR↑、QPS↓、AMR↓。

### 分析什么

重点回答：

1. FF-PBS是否稳定提高GSR；
2. 提升是否跨任务和模型架构存在；
3. 提升是否以明显增加AMR或查询量为代价；
4. FF-PBS是否只在低基线成功率任务上有效；
5. 不同模型更新对的收益差异。

### 预期结论形式

> FF-PBS在大多数模型更新设置下发现更多差分输入，说明单路径动态目标遗漏了可达回归；收益在搜索空间更困难或中间不可行状态更多的设置中更加明显。

---

## E2：查询预算下的发现过程

### 目的

分析FF-PBS的收益出现在哪个查询阶段，以及是否只是通过消耗更多预算获得更高最终GSR。

### 做什么

对每个方法记录首次成功查询数，计算：

[
\mathrm{Success@B},
\qquad B=1,\ldots,Q.
]

### 展示什么

为每个数据集绘制Success–Query曲线：

* 横轴：模型对查询预算；
* 纵轴：Success@B；
* 方法：DT4LM-Dynamic、FF-PBS。

再给出：

* Success-query AUC；
* Success@100、500、1000；
* budget-penalized query cost。

### 分析什么

判断FF-PBS属于哪种模式：

* 低预算下也更快成功；
* 低预算相近、中高预算发现更多困难样本；
* 仅靠耗尽预算增加最终成功数。

根据初步结果，FF-PBS的主要优势可能出现在中高预算区域，因此应重点分析新增成功样本的查询分布。

---

## E3：组件消融

### 目的

区分FF-PBS的提升来自：

1. 更换目标表示；
2. 多路径frontier；
3. feasibility-first不可行路径保留；
4. Pareto低修改偏好。

### 做什么

比较：

| 方法               | 排序        | Frontier | 不可行状态     | 可行候选排序 |
| ---------------- | --------- | -------: | --------- | --------- |
| Dynamic-DT4LM     | 动态标量   |        1 | 由原目标决定     | - |
| Dynamic-Beam     | 动态标量   |        5 | 不特殊处理     | - |
| FF-Pareto-Greedy | FF Pareto |        1 | 最小违反候选    | - |
| Hard-PBS         | Pareto    |        5 | 永久丢弃      | 完整 |
| FF-MNew         | FF单目标   |        5 | 可行优先、补位保留 | 只按 m_{new} 排序 |
| FF-PBS           | FF Pareto |        5 | 可行优先、补位保留 | 完整 |

### 展示什么

用一个消融表报告：

* GSR；
* QPS；
* AMR；

### 分析什么

对比关系：

* Dynamic vs Dynamic-Beam：单纯增加多路径是否足够；
* FF-Pareto-Greedy vs FF-PBS：多路径贡献；
* Hard-PBS vs FF-PBS：保留可恢复不可行路径的贡献；
* FF-MNew vs FF-PBS：修改成本作为第二目标的贡献；
* Dynamic-Beam vs FF-PBS：多目标约束化选择的整体贡献。

不能把Dynamic-Beam与FF-PBS的全部差异简单归因于Pareto，需要结合其他消融共同解释。

---

## E4：可恢复路径机制分析

### 目的

直接验证论文核心观点：

> 硬删除旧模型错误的中间状态会丢失后续能够恢复并成功的路径。

### 做什么

重点比较Hard-PBS与FF-PBS。

对于FF-PBS的每个成功输入，恢复完整父节点路径，统计：

[
R_{\mathrm{recover}}
=
\frac{
#{\text{成功路径中存在post-root旧模型错误状态}}
}{
#{\text{FF-PBS成功}}
}.
]

对FF-PBS独有成功定义：

[
R_{\mathrm{unique\text{-}recover}}
=
\frac{
#{\text{FF独有成功且经过旧模型错误状态}}
}{
#{\text{FF独有成功}}
}.
]

定义Hard pruning loss：

[
L_{\mathrm{hard}}
=
\mathrm{GSR}_{\mathrm{FF}}
-
\mathrm{GSR}_{\mathrm{Hard}}.
]

并统计每个设置中Hard-PBS删除的不可行状态比例。

### 展示什么

建议三部分：

1. **机制表**

| Dataset/Pair | Hard discard rate | FF recover rate | Unique recover rate | GSR gap |
| ------------ | ----------------: | --------------: | ------------------: | ------: |

2. **散点图**

* 横轴：Hard-PBS删除状态比例；
* 纵轴：FF-PBS相对Hard-PBS的GSR提升；
* 报告Spearman相关性。

3. **路径案例图**

展示一个真实输入的三至四步修改：

```text
原始状态：旧对 / 新对
    ↓
中间状态：旧错 / 新对
    ↓
恢复状态：旧对 / 新对
    ↓
成功状态：旧对 / 新错
```

### 分析什么

回答：

* 不可行状态出现得越频繁，硬剪枝损失是否越大；
* FF-PBS独有成功中有多少真正依赖可恢复路径；
* 不可行状态是否只是被保留但没有实际贡献；
* 路径恢复发生在什么修改深度。

这是论文最重要的机制实验。

---

## E5：非贪心路径与frontier多样性

### 目的

验证多路径frontier是否真的探索了原动态top-1之外的区域，而不是保存多个近似重复候选。

### 做什么

统计：

[
R_{\mathrm{non\text{-}top1}}
=
\frac{
#{\text{成功路径首步不是动态目标top-1}}
}{
#{\text{FF-PBS成功}}
}.
]

对于FF-PBS独有成功，再计算：

[
R_{\mathrm{unique\text{-}non\text{-}top1}}.
]

同时记录frontier中的：

* 不同修改位置集合数；
* 不同深度数；
* Pareto第一前沿大小；
* frontier实际宽度。

### 展示什么

* 非top-1路径比例柱状图；
* frontier深度多样性分布；
* FF-PBS独有成功的首步dynamic rank直方图。

### 分析什么

说明：

* FF-PBS的成功是否确实来自原贪心会删除的路径；
* beam中的候选是否具有真实路径多样性；
* 搜索收益主要来自更宽探索还是不可行路径恢复。

---

## E6：输入质量与语义有效性

### 目的

确认 FF-PBS 提高 GSR 并不是通过生成标签已经改变或任务相关语义已经破坏的输入获得的，并验证 FF-PBS 新增发现的差分输入具有实际测试价值。

### 自动评价

对所有成功差分输入报告：

- AMR；
- BERTScore；
- BLEU；
- METEOR；
- ROUGE-L。

自动指标只用于描述表面修改幅度和文本相似度，人工评价作为语义有效性的主要依据。

### 人工评价范围

为控制人工成本，只在两个代表性设置上进行人工评价：

1. 一个单句分类任务，例如 SST-2；
2. 一个句子对任务，例如 RTE 或 MRPC。

使用论文主模型更新对和冻结后的主参数配置。

对于每个设置：

- 从 DT4LM-Dynamic 的成功输入中均匀随机抽取 100 个；
- 从 FF-PBS 的成功输入中均匀随机抽取 100 个；
- 成功数不足 100 时使用全部成功输入；
- 对 FF-PBS 独有成功输入额外抽取至多 50 个，专门评价新增发现的有效性；
- 额外抽取的 FF-PBS 独有样本只用于“新增输入有效率”，不用于估计 FF-PBS 的整体有效率。

所有样本隐藏方法名称并随机排序。相同原始输入由两种方法都成功时，两个修改结果分别评价。

### 人工判断问题

每个生成输入只回答以下两个二值问题。

#### Q1：标签是否保持

> 在不参考新旧模型预测结果的情况下，修改后的完整输入是否仍应具有与原始输入相同的真实任务标签？

记为：

\[
L_i\in\{0,1\}.
\]

评价时应基于任务定义和原始标签，而不是根据文本表面相似度判断。

对于句子对任务，应评价修改后完整句子对的关系标签，而不是只判断被修改的单句。

#### Q2：任务相关语义是否保持

> 与当前分类任务和原始标签判断相关的含义是否被完整保留？

记为：

\[
M_i\in\{0,1\}.
\]

允许：

- 同义改写；
- 不影响任务判断的措辞变化；
- 不相关细节的轻微变化。

不允许改变：

- 否定关系；
- 情感极性或关键强度；
- 实体、数量和事实；
- 因果、时间或比较关系；
- RTE/MRPC 中决定句间关系的关键信息。

### 人工评估指标

标签保持率定义为：

\[
\mathrm{LPR}
=
\frac{1}{n}\sum_{i=1}^{n}L_i.
\]

任务相关语义保持率定义为：

\[
\mathrm{SPR}
=
\frac{1}{n}\sum_{i=1}^{n}M_i.
\]

一个差分输入只有在两个判断均为“是”时，才视为人工有效：

\[
V_i=L_i\land M_i.
\]

联合人工有效率定义为：

\[
\mathrm{HVR}
=
\frac{1}{n}\sum_{i=1}^{n}V_i.
\]

对每种方法，利用均匀抽取的成功输入估计：

\[
\widehat{\mathrm{ValidGSR}}
=
\mathrm{GSR}\times\mathrm{HVR}.
\]

ValidGSR 表示一个 attackable 原始输入最终产生人工有效差分输入的估计概率。

对于额外抽取的 FF-PBS 独有成功样本，单独定义新增输入有效率：

\[
\mathrm{IVR}_{\mathrm{FF}}
=
\frac{
\#\{\text{FF-PBS独有成功且 }L_i=1,M_i=1\}
}{
\#\{\text{被人工评价的FF-PBS独有成功}\}
}.
\]

该指标直接回答 FF-PBS 相比 Base 额外发现的输入中，有多少是真实有效的回归测试用例。

### 标注协议

- 至少两名评审者独立评价；
- 评审者不知道输入来自哪种方法；
- 不向评审者展示新旧模型的预测结果、margin 或搜索路径；
- 分歧样本由第三名评审者或共同复核得到最终标签；
- 对 Q1 和 Q2 分别报告 Cohen's \(\kappa\)；
- 对 LPR、SPR、HVR、ValidGSR 和 IVR 报告 bootstrap 95% 置信区间。

### 展示什么

主表报告上述指标。
另单独报告 FF-PBS 独有成功的：

- 样本数量；
- IVR；
- bootstrap 95% 置信区间。

### 分析什么

重点回答：

1. FF-PBS 的标签保持率和语义保持率是否与 Base 相当；
2. GSR 提升后，ValidGSR 是否也相应提高；
3. FF-PBS 独有成功输入是否具有足够高的 IVR；
4. FF-PBS 是否通过略高修改率换取了更多有效回归，而不是更多语义无效输入。

如果 FF-PBS 提高 GSR，但 ValidGSR 没有提高，则不能声称方法发现了更多有效回归；如果 ValidGSR 和 IVR 均保持较高水平，则说明新增覆盖具有实际测试价值。

---

## E7：参数敏感性与搜索成本

### 目的

验证方法不依赖狭窄参数选择，并分析frontier宽度与效果、成本的关系。

### 做什么

比较：

[
K\in{1,3,5,10}.
]

### 展示什么

两张图：

1. (K) 与GSR的关系；
2. (K) 与模型查询的关系。

### 分析什么

判断：

* (K=5)是否已经接近饱和；
* 继续增大frontier是否只增加成本；
* 方法是否对参数稳健。

---

# 9. 正文实验组织建议

正文无需放入所有实验的全部结果，建议如下。

## 正文

* E1：总体有效性；
* E2：查询预算曲线；
* E3：组件消融；
* E4：可恢复路径机制；
* E6：语义有效性；

## 附录

* E5：frontier多样性；
* E7：参数敏感性；
* 全部逐数据集自动质量指标；
* 完整统计检验结果。

---

# 10. 其余建议展示

## Figure 1：动机案例

展示硬过滤遗漏可恢复路径。

## Figure 2：FF-PBS工作流

展示变换、约束、模型查询、Pareto排序、可行优先和frontier更新。

## Algorithm 1：FF-PBS伪代码

---

# 12. 最终论文叙事

论文最有力的故事是：

> **模型更新差分测试定义的是一个最终状态约束，但现有搜索容易将它错误地处理为路径约束。由于文本修改路径具有非单调性，有效回归输入可能需要暂时离开旧模型正确区域。FF-PBS始终优先探索可行状态，同时在frontier有剩余容量时保留最接近可行域的恢复路径，并通过Pareto搜索平衡新模型错误margin和修改质量。这样既不改变最终差分输入的严格有效性，又能发现硬剪枝和单路径搜索遗漏的回归。**

这一主线具有：

* 明确的问题诊断；
* 与问题结构匹配的方法设计；
* 可以通过路径级数据直接验证的机制；
* 已有初步性能证据；
* 比复杂调度方案更低的参数和解释成本。

真正决定论文能否成立的关键，是后续能否在更多数据集和模型更新对上建立以下完整证据链：

[
\text{存在不可行中间状态}
\rightarrow
\text{硬剪枝丢失可恢复路径}
\rightarrow
\text{FF-PBS保留该路径}
\rightarrow
\text{发现更多语义有效的回归输入}.
]
