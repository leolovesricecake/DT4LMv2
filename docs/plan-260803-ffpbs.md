# 论文方案：Feasibility-First Pareto Search for Differential Testing of Language Model Updates

## 1. 论文定位

主方法名为：**FF-PBS: Feasibility-First Pareto Beam Search**。

论文围绕一个核心判断展开：差分输入的最终结果必须满足旧模型正确，但搜索路径上的每个中间状态不必始终满足旧模型正确。

因此，FF-PBS并不是放松最终差分条件，而是区分：

- **终点可行性**：最终生成的差分输入必须满足旧模型正确、新模型错误；
- **路径可行性**：中间修改状态可以暂时不满足旧模型正确，只要该路径仍有机会恢复并到达有效终点。

给定旧模型 $M_o$、更新后的新模型 $M_n$、输入 $x$ 及真实标签 $y$，目标是寻找语义保持的修改输入 $x'$，使：

\[
M_o(x')=y,
\qquad
M_n(x')\neq y.
\]

DT4LM是一个模块化框架：它通过动态标量目标为候选赋分，并可与Kuleshov贪心搜索、LEAP粒子群搜索和FastGA遗传搜索等不同Recipe结合。因此，本文不将DT4LM整体描述为单路径搜索，也不把“增加多个候选”本身作为创新。

FF-PBS研究的是DT4LM尚未显式处理的**搜索层问题结构**。在严格控制实验中，它保持Kuleshov Recipe的文本变换、约束和差分成功条件不变，将“动态标量指导的通用候选选择”替换为：

1. 旧模型正确性的最终可行约束；
2. 新模型错误margin与修改成本的Pareto排序；
3. 可行优先、不可行补位的路径保留机制；
4. 可同时维护不同深度状态的异步有界frontier。

实验同时加入DT4LM-Kuleshov、DT4LM-LEAP和DT4LM-FastGA，以回答：已有通用多候选搜索是否已经足够，以及FF-PBS的收益是否来自差分测试特有的候选选择与路径恢复机制。

# 2. 核心思想

## 2.1 DT4LM留出的搜索层问题

DT4LM提出动态标量目标，并证明该目标可以与贪心、遗传算法和粒子群优化等多种通用搜索策略结合。其优势是模块化，但搜索器仍主要沿用一般对抗文本生成方法，没有显式编码模型更新差分测试的约束结构。

### 问题一：标量分数隐藏了约束与目标的不同角色

DT4LM的动态目标根据旧模型和新模型的预测状态，为每个候选产生一个标量分数。该分数有利于直接接入不同搜索器，但会将候选压缩为单一全序，无法显式表达以下角色差异：

- 旧模型最终正确：差分输入的有效性约束；
- 新模型趋向错误：回归发现目标；
- 修改幅度较小：测试输入的质量目标。

其中，DT4LM的动态标量主要组合新旧模型行为，修改质量则主要由文本约束和结果指标间接控制，并未作为独立搜索目标与新模型错误margin共同维护。因而，多个在“攻击进展—修改成本”上互不支配的候选可能被强制排成一个顺序并过早淘汰。

### 问题二：通用搜索方法未显式利用差分测试特有结构

DT4LM支持多种Recipe：Kuleshov实例采用宽度为1的贪心搜索，LEAP和FastGA则维护多个候选。

但问题是：这些搜索方法均由动态标量指导，没有显式建模**终点可行性与路径可行性的区别**。文本修改空间是离散且非单调的，有效差分输入可能通过如下路径到达：

\[
\text{旧对、新对}
\rightarrow
\text{旧错、新对}
\rightarrow
\text{旧对、新错}.
\]

中间状态暂时使旧模型出错，但后续修改可以恢复旧模型，并最终使新模型出错。现有通用搜索没有专门保证：

- 当前可行状态得到优先利用；
- 可行状态不足时，接近可行域的潜在恢复路径仍被保留；
- 新模型错误margin与修改成本之间的不同折中能够同时存在；
- 不同修改深度的状态能够在同一frontier中竞争。

因此，本文的动机不是“用beam替代DT4LM的贪心搜索”，而是设计一个面向差分测试问题结构的专用候选选择与路径维护方法。

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

## 3.1 DT4LM的通用工作流

```text
原始输入
  ↓
按照Recipe生成候选
  ↓
应用对应的修改与语义约束
  ↓
查询新旧模型
  ↓
计算DT4LM动态标量目标
  ↓
由Recipe对应的通用搜索器更新候选
  ├─ Kuleshov：宽度为1的贪心搜索
  ├─ LEAP：粒子群搜索
  └─ FastGA：遗传搜索
  ↓
继续搜索或终止
```

DT4LM的动态目标是各Recipe共享的回归感知信号；候选如何维护和演化则由具体Recipe决定。

## 3.2 FF-PBS工作流

```text
原始输入
  ↓
初始化异步有界frontier
  ↓
选择一个待扩展状态
  ↓
生成词级候选并应用原有约束
  ↓
查询新旧模型
  ↓
计算旧模型可行性、新模型错误margin和修改率
  ↓
检查严格差分成功条件
  ↓
可行候选执行Pareto非支配排序
  ↓
frontier有空位时按最小违反原则保留不可行候选
  ↓
更新frontier并继续搜索
```

## 3.3 保持不变的部分

| 模块 | 处理 |
| --- | --- |
| 差分输入定义 | 完全沿用DT4LM |
| 词级变换 | 严格控制实验中沿用Kuleshov配置 |
| 停用词约束 | 沿用 |
| 重复修改约束 | 沿用 |
| 最大修改比例 | 沿用 |
| 原有语义约束 | 沿用 |
| 流畅性约束 | 沿用 |
| 查询预算 | 对所有对比方法保持一致 |
| 成功样本评估 | 沿用并补充ValidGSR |
| 后续模型修复 | 可继续使用生成的差分输入 |

## 3.4 实质修改

| 模块 | DT4LM | FF-PBS |
| --- | --- | --- |
| 候选评价 | 动态标量全序 | 最终可行约束加Pareto偏序 |
| 搜索器 | Recipe相关的通用贪心、GA或PSO | 面向差分测试结构的异步frontier |
| 搜索状态数 | 由Recipe决定 | 最多 $K$ 个 |
| 中间旧模型错误状态 | 由动态标量间接评价 | 可行优先，必要时按违反程度补位 |
| 候选深度 | 由Recipe及其迭代方式决定 | 可同时保留不同修改深度 |
| 修改率作用 | 主要由约束控制并用于结果评价 | 作为独立Pareto目标参与候选选择 |

因此，FF-PBS的区别不在于“首次使用多候选搜索”，而在于使用差分测试特有的约束、目标和路径恢复结构来决定哪些候选应被保留和扩展。

# 4. 论文核心贡献

建议将贡献压缩为三项。

## 贡献一：识别差分测试中的结构化搜索缺口

我们指出，DT4LM的动态标量能够与多种通用搜索方法组合，但标量分数无法显式区分：

- 旧模型正确这一最终有效性约束；
- 新模型错误margin这一回归发现目标；
- 修改成本这一输入质量目标。

同时，现有通用贪心、GA和PSO搜索并未专门建模终点可行性与路径可行性的区别，因而不能显式维护可恢复的暂时不可行路径。该问题在离散、非单调的文本修改空间中尤其重要。

## 贡献二：提出FF-PBS

提出面向语言模型更新差分测试的feasibility-first Pareto搜索：

- 将旧模型正确性建模为最终可行约束；
- 在新模型错误margin和修改成本之间维护Pareto候选；
- 优先保留可行状态；
- 仅在frontier有空位时，以最小违反原则保留不可行状态；
- 使用异步有界frontier探索多条不同深度的修改路径。

FF-PBS不放松最终差分成功条件，也不依赖任务特定的加权系数。

## 贡献三：通过系统级与机制级实验验证方法

在多个数据集、模型更新和查询预算下：

- 与DT4LM-Kuleshov、DT4LM-LEAP和DT4LM-FastGA比较，验证FF-PBS相对现有贪心和通用种群搜索的GSR—查询—修改质量权衡；
- 通过Dynamic-Beam验证提升并非仅来自保留更多候选；
- 通过Hard-PBS和路径追踪验证硬删除不可行状态会丢失可恢复路径；
- 通过FF-Pareto-Greedy和FF-MNew分析多路径frontier及修改成本目标的贡献；
- 通过人工评价确认新增差分输入仍具有可接受的标签与任务相关语义保持性。

# 5. 建议的论文主线

## 引言中的逻辑

1. 模型更新可能引入只在特定输入变体上出现的回归；
2. DT4LM通过动态标量目标搜索“旧模型正确、新模型错误”的输入，并可与Kuleshov、LEAP和FastGA等不同Recipe结合；
3. 这种模块化设计证明了动态目标的通用性，但候选仍由标量分数和通用搜索策略指导；
4. 标量分数不能显式区分最终约束、回归目标和修改质量，也无法同时维护多个互不支配的候选折中；
5. 差分成功条件是终点条件，而离散文本修改路径可能暂时离开可行区域后重新返回；
6. 通用搜索没有专门编码“可行优先但保留潜在恢复路径”的差分测试结构，单纯扩大种群也未必解决这一问题；
7. FF-PBS通过“最终约束＋Pareto双目标＋可行优先、不可行补位”的异步frontier实现结构化搜索；
8. 实验与DT4LM三种Recipe及一组严格消融比较其覆盖率、效率、路径机制和输入质量。

## 一句话主线

> **FF-PBS将模型更新差分测试从动态标量指导的通用搜索，重构为显式区分最终可行性、路径恢复性和修改质量的Pareto frontier搜索，从而发现现有贪心和通用种群Recipe遗漏的有效回归。**

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

## 7.3 搜索配置与公平比较协议

实验分为两类，避免将系统级Recipe差异与FF-PBS组件贡献混为一谈。

### 7.3.1 严格控制与机制实验

E3—E5固定使用Kuleshov Recipe的文本变换和原有约束，只替换候选评价与搜索策略。所有方法使用：

- 相同测试manifest及样本顺序；
- 相同新旧模型checkpoint；
- 相同文本变换和约束；
- 相同候选生成顺序；
- 相同模型批大小；
- 相同模型对查询预算。

FF-PBS主配置为：

\[
K=5,
\qquad
Q=1000.
\]

该协议用于回答Dynamic-Beam、Hard-PBS、FF-Pareto-Greedy、FF-MNew和FF-PBS之间的组件归因问题。

### 7.3.2 DT4LM Recipe系统级比较

E1和E2增加以下DT4LM完整Recipe：

- DT4LM-Kuleshov；
- DT4LM-LEAP；
- DT4LM-FastGA。

每个Recipe保留其原有的文本变换、约束和搜索机制，但统一使用：

- 相同数据集、manifest和样本顺序；
- 相同新旧模型checkpoint；
- 相同模型对查询预算 $Q=1000$；
- 相同的模型对查询计数口径；
- 相同硬件与模型批大小；
- 相同成功、失败和skipped统计协议。

FF-PBS使用Kuleshov的文本变换和约束作为主实例化。DT4LM-Kuleshov与FF-PBS构成最严格的直接对照；LEAP和FastGA用于回答通用多候选或种群搜索是否已经足够。

由于LEAP和FastGA还改变了变换与约束组件，它们与FF-PBS的比较属于**系统级比较**，不能单独用于归因某一个搜索组件。除GSR、QPS和AMR外，还应报告总查询数、wall-clock、峰值显存和候选生成规模，以呈现完整成本。

数据集测试样本遵循统一协议：

- 最多随机选择1000条；
- 测试集不足时使用全部；
- 原始差分输入沿用DT4LM协议记为skipped。

## 7.4 对比方法

### 系统级DT4LM基线

1. **DT4LM-Kuleshov**  
   DT4LM动态目标与Kuleshov宽度为1的贪心搜索。它与FF-PBS共享文本变换和原有约束，是最直接的主要基线。

2. **DT4LM-LEAP**  
   DT4LM动态目标与LEAP粒子群搜索。用于验证通用多候选搜索能否达到与FF-PBS相同的覆盖率—查询成本权衡。

3. **DT4LM-FastGA**  
   DT4LM动态目标与FastGA遗传搜索。用于验证更广泛的种群演化是否足以替代问题特定的frontier选择。

4. **FF-PBS**  
   完整方法，使用Kuleshov文本变换和约束，异步frontier大小为 $K=5$。

### 严格控制的内部对照

1. **Dynamic-Beam**  
   使用与FF-PBS相同的异步frontier和 $K=5$，但仍按DT4LM动态标量排序，用于隔离“仅增加多路径”的收益。

2. **Hard-PBS**  
   使用相同Pareto frontier，但永久删除所有post-root旧模型预测错误状态，用于验证可恢复不可行路径的作用。该方法是本文的诊断消融，不是DT4LM原有行为。

3. **FF-Pareto-Greedy**  
   使用与FF-PBS相同的feasibility-first Pareto规则，但令 $K=1$，用于隔离多路径frontier的贡献。

4. **FF-MNew**  
   使用相同feasibility-first与 $K=5$，但可行候选只按 $m_{\mathrm{new}}$ 排序，用于验证修改成本作为第二目标的贡献。

5. **FF-PBS**  
   完整方法。

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

从系统层面验证FF-PBS是否比DT4LM已有的贪心、粒子群和遗传Recipe发现更多模型更新回归，并取得更好的覆盖率—查询—修改质量权衡。

### 做什么

在所有数据集和模型更新对上比较：

- DT4LM-Kuleshov；
- DT4LM-LEAP；
- DT4LM-FastGA；
- FF-PBS。

所有方法使用相同manifest、模型checkpoint和模型对查询预算。各DT4LM Recipe保留原有组件，FF-PBS使用Kuleshov变换与约束。

### 展示什么

主表按数据集—模型对报告：

- GSR $\uparrow$；
- QPS $\downarrow$；
- BPQC $\downarrow$；
- Success-query AUC $\uparrow$；
- AMR $\downarrow$；
- wall-clock和峰值显存。

同时报告FF-PBS相对每个Recipe的配对独有成功数和平均排名。

### 分析什么

重点回答：

1. FF-PBS是否不仅优于宽度为1的Kuleshov贪心，而且优于LEAP和FastGA等已有多候选搜索；
2. LEAP或FastGA能否通过更高查询成本获得相同覆盖率；
3. FF-PBS是否在GSR、BPQC和AMR之间形成更优的Pareto权衡；
4. 提升是否跨任务和模型架构存在；
5. 哪类任务更受益于问题特定的可行性与路径恢复机制。

### 结论边界

若FF-PBS只优于DT4LM-Kuleshov、但不优于LEAP和FastGA，则论文只能证明其改善了高效贪心实例；若FF-PBS在相同预算下相对三种Recipe均取得更好的综合权衡，才能有力支持“通用搜索未充分利用差分测试特有结构”的主张。

## E2：查询预算下的发现过程

### 目的

比较FF-PBS与DT4LM不同Recipe在查询预算增长过程中的回归发现能力，判断性能差异来自更早成功、更多中后期覆盖，还是单纯消耗更多查询。

### 做什么

对以下方法记录每个样本的首次成功查询数：

- DT4LM-Kuleshov；
- DT4LM-LEAP；
- DT4LM-FastGA；
- FF-PBS。

计算：

\[
\mathrm{Success@B},
\qquad B=1,\ldots,Q.
\]

### 展示什么

为代表性数据集绘制四种方法的Success–Query曲线；为避免主文过于拥挤，其余设置放入附录。另报告：

- Success@100、500、1000；
- Success-query AUC；
- BPQC；
- 总模型对查询数；
- 成功样本首次查询数的中位数和四分位数。

### 分析什么

判断：

- FF-PBS是否在低预算下就优于通用Recipe；
- FF-PBS是否主要在中高预算阶段发现额外困难样本；
- LEAP和FastGA的高覆盖是否依赖更快消耗查询预算；
- 不同方法最终GSR相近时，谁具有更低的预算惩罚成本；
- FF-PBS的优势是否来自更合理的候选维护，而非仅增加模型调用。

## E3：组件消融

### 目的

在完全固定Kuleshov文本变换、约束、候选顺序和查询预算的条件下，区分FF-PBS的提升来自：

1. 多路径frontier；
2. feasibility-first不可行路径保留；
3. 修改成本作为第二Pareto目标；
4. 异步frontier与完整结构的组合。

### 做什么

比较：

| 方法 | 候选排序 | Frontier | 不可行状态 | 可行候选目标 |
| --- | --- | ---: | --- | --- |
| DT4LM-Kuleshov | 动态标量 | 1 | 由原目标间接评价 | 动态标量 |
| Dynamic-Beam | 动态标量 | 5 | 不特殊处理 | 动态标量 |
| FF-Pareto-Greedy | feasibility-first | 1 | 最小违反候选 | $m_{\mathrm{new}}$与修改成本 |
| Hard-PBS | Pareto | 5 | 永久丢弃 | $m_{\mathrm{new}}$与修改成本 |
| FF-MNew | feasibility-first | 5 | 可行优先、补位保留 | 仅$m_{\mathrm{new}}$ |
| FF-PBS | feasibility-first Pareto | 5 | 可行优先、补位保留 | $m_{\mathrm{new}}$与修改成本 |

### 展示什么

消融表报告：

- GSR；
- QPS；
- BPQC；
- Success-query AUC；
- AMR；
- 独有成功数。

### 分析什么

对比关系：

- DT4LM-Kuleshov vs Dynamic-Beam：单纯增加多路径是否足够；
- FF-Pareto-Greedy vs FF-PBS：有界多路径frontier的贡献；
- Hard-PBS vs FF-PBS：保留可恢复不可行路径的贡献；
- FF-MNew vs FF-PBS：修改成本作为第二目标的贡献；
- Dynamic-Beam vs FF-PBS：结构化候选选择的整体贡献。

不能把Dynamic-Beam与FF-PBS的全部差异简单归因于Pareto；必须结合其余消融形成完整归因。LEAP和FastGA不进入本实验，因为它们同时改变多个Recipe组件，无法用于严格组件归因。

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

这是论文最重要的机制实验。需要明确：Hard-PBS是本文构造的诊断消融，用于验证路径硬约束的代价，并不代表DT4LM原有Recipe会直接丢弃所有不可行状态。

---

## E5：非贪心路径与frontier多样性

### 目的

在固定Kuleshov配置下，验证FF-PBS是否真的探索了DT4LM动态标量top-1之外的区域，而不是保存多个近似重复候选。该分析针对Kuleshov直接对照，不用于描述LEAP或FastGA的种群轨迹。

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

- 从 DT4LM-Kuleshov 的成功输入中均匀随机抽取 100 个；
- 从 FF-PBS 的成功输入中均匀随机抽取 100 个；
- 成功数不足 100 时使用全部成功输入；
- 对 FF-PBS 独有成功输入额外抽取至多 50 个，专门评价新增发现的有效性；
- 额外抽取的 FF-PBS 独有样本只用于“新增输入有效率”，不用于估计 FF-PBS 的整体有效率。

所有样本隐藏方法名称并随机排序。相同原始输入由两种方法都成功时，两个修改结果分别评价。人工评价聚焦FF-PBS与其最严格的直接基线DT4LM-Kuleshov；DT4LM-LEAP和DT4LM-FastGA在全部设置上报告自动质量指标，但不额外扩大人工标注规模。

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

# 11. 最终论文叙事

论文最有力的故事是：

> **DT4LM通过动态标量目标为不同通用搜索Recipe提供回归感知信号，但标量全序无法显式区分最终可行约束、新模型错误目标和修改质量，也没有专门建模离散修改路径中的暂时不可行与后续恢复。FF-PBS以差分测试的问题结构重新设计候选选择：可行状态优先，在新模型错误margin和修改成本上维护Pareto候选，并仅在frontier有空位时保留最接近可行域的恢复路径。这样既严格保持最终差分条件，又能获得比Kuleshov贪心以及LEAP、FastGA等通用Recipe更好的回归发现与成本权衡。**

这一主线具有：

- 对DT4LM能力边界的准确描述：它支持多种搜索，而非仅支持单路径贪心；
- 明确的问题诊断：动态标量与通用搜索未显式表达差分测试的约束、目标和路径结构；
- 与问题结构匹配的方法设计；
- 可以通过路径级数据直接验证的机制；
- 同时覆盖系统级Recipe比较与严格组件消融的实验证据。

真正决定论文能否成立的关键，是建立以下两条证据链：

\[
\text{DT4LM通用Recipe}
\rightarrow
\text{在相同预算下仍存在覆盖率或成本缺口}
\rightarrow
\text{FF-PBS取得更优综合权衡},
\]

以及：

\[
\text{存在可恢复的暂时不可行路径}
\rightarrow
\text{硬剪枝会丢失该路径}
\rightarrow
\text{FF-PBS保留并恢复该路径}
\rightarrow
\text{发现更多语义有效的回归输入}.
\]

若FF-PBS仅优于DT4LM-Kuleshov而无法优于LEAP和FastGA，论文仍可定位为对高效贪心Recipe的改进，但“通用搜索未利用差分测试结构”的主张需要收窄；若FF-PBS在相同查询预算下相对三种Recipe均表现出更好的GSR—BPQC—AMR权衡，则可以形成更完整、更有说服力的方法贡献。
