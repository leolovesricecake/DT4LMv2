## 总体判断

这份执行计划整体质量较高，工程边界、查询预算、Pareto排序、消融设计、产物协议和测试方案都比较完整，**可以作为实现基础，但不建议原样开工**。其中有三项会影响研究问题或算法正确性，必须先修正。

# 一、必须修正

## 不能只根据文本内容做全局搜索状态去重

计划使用：

```python
tuple(attacked_text.text_input.items())
```

同时排除frontier中、已查询和已扩展的相同文本。

这里混淆了两个不同概念：

### 模型查询去重

两个状态的当前文本相同，则新旧模型输出相同，可以复用模型查询结果。

其键可以是：

```python
query_key = canonical_text_fields
```

### 搜索状态去重

即使当前文本相同，不同路径仍可能具有不同的：

* `modified_indices`；
* 已修改位置历史；
* transformation属性；
* 后续允许修改的位置。

这些信息会影响`RepeatModification`等路径相关约束，因此不能仅按文本合并搜索状态。

建议分开定义：

```python
query_key = canonical_text_fields

state_key = (
    canonical_text_fields,
    frozenset(modified_indices),
)
```

若进一步确认TextAttack中还有影响后续变换的路径属性，也应纳入`state_key`。

结论是：

> 相同文本不重复查询模型，但不一定删除其中一个搜索状态。

否则AE-PBS可能错误地丢弃一条后续可达空间不同的路径。

---

## 首轮没有旧模型错误候选时，AE-PBS会永久退化为Strict-PBS

当前定义为：

[
\epsilon_0
=
Q_{0.75}{-m_{\mathrm{old}}(x'):m_{\mathrm{old}}(x')<0}
]

若第一次扩展没有旧模型错误候选，则：

[
\epsilon_0=0.
]

此后：

[
\epsilon(q)=0
]

AE-PBS便永远不允许暂时违反旧模型约束，即使第二层或第三层才出现有价值的旧模型错误路径。

这与方法“通过临时不可行路径逃离局部最优”的核心动机不完全一致。

建议在验证集上比较两种初始化：

### 方案A：短暂预热，推荐

在前 (W) 个实际查询候选内收集违反量，例如：

[
W=100
]

或前两次扩展，然后固定：

[
\epsilon_0=Q_{0.75}(V_W^+).
]

预热期间只使用Dynamic-Beam或不应用旧模型约束，预热结束后再启动AE-PBS排序。

### 方案B：样本局部下限

[
\epsilon_0
=
\max\left(
Q_{0.75}(V^+),
\alpha |m_{\mathrm{old}}(x)|
\right),
]

其中 (\alpha) 在验证集上确定。

方案A更符合“由实际候选分布估计”，但会稍微增加实现复杂度。至少应把“首轮无违反即退化”的比例作为诊断指标，而不能默认其无影响。

---

# 二、需要明确的方法论问题

## 当前算法不是严格意义上的“最小化修改率”

计划一旦发现成功候选，就在当前批次中选择修改率最低者并立即终止。

因此修改成本只用于：

* 指导成功前的搜索；
* 比较同一批次内的成功输入。

它并不保证：

[
\min_{x'\in\text{全部可达成功输入}}c(x,x').
]

可能后续再搜索50次，就能发现修改更少的成功输入，但算法已经停止。

为了与DT4LM保持相同的首次成功口径，保留设计，但论文和计划中的表述应改为：

> AE-PBS在搜索过程中偏好较低的修改成本，并在同批成功候选中选择成本最低者。

不要声称它直接求解“最小修改差分输入”。

---

## 5. `epsilon_0`最好使用归一化诊断

虽然每个样本独立估计 (\epsilon_0)，不同模型和样本的logit尺度仍可能差异很大。

建议至少额外保存：

[
\frac{\epsilon_0}
{|m_{\mathrm{old}}(x)|+\delta}
]

以及：

* 原始输入旧模型margin；
* 第一轮候选margin分布；
* (\epsilon_0)分布；
* `epsilon_zero_initialization_rate`。

不一定需要立即把方法改成归一化margin，但需要检查AE-PBS效果是否主要由少数超大margin样本驱动。

---

# 三、实验与指标需要调整

## 明确所有指标的分母（重要！）

当前计划同时使用：

* Paper GSR；
* Sample generation rate；
* Success@B；
* sample-Success@B。

但部分公式没有完整写出，请明确！

---

## QPS与配对效率分析不能只看共同成功样本

只在两种方法共同成功的样本上比较查询数，会忽略AE-PBS新增成功样本，也可能产生选择偏差。

建议保留：

* 共同成功样本上的Wilcoxon查询比较；
* QPS；
* Success@B曲线；
* success-query AUC。

并增加一个受限查询成本：

[
q_i^{*}
=
\begin{cases}
q_i,&\text{成功},\
Q,&\text{失败}.
\end{cases}
]

在全部配对样本上比较 (q_i^*)，作为补充的预算受限效率分析。这样成功率和查询效率不会完全割裂。

---

## 三个“攻击随机种子”可能没有意义

Kuleshov候选生成、Pareto排序和tie-breaking都被设计为确定性的。如果模型checkpoint和manifest相同，仅改变“攻击seed”可能得到完全相同的结果。

需要区分：

* 模型微调随机种子；
* 测试样本抽样种子；
* 随机变换或候选顺序种子；
* 算法tie-breaking种子。

如果攻击过程完全确定，就没有必要重复三个攻击seed。最终实验更有意义的是：

* 三个独立微调seed；或
* 固定模型后，通过配对bootstrap衡量测试样本不确定性。

RTE若使用全部合格测试样本，改变manifest seed也不会产生作用。

---

# 四、计划中可以保留的部分

以下设计较好，无需修改：

* 使用`GoalFunction.get_results()`统一管理查询预算；
* margin不替代真实预测标签进行成功判定；
* Dynamic-Beam与AE-PBS共用相同异步frontier，实现公平消融；
* 明确认可`K=1`不等于原DT4LM；
* 承认Dynamic-Beam到Strict-PBS不能单独归因为Pareto收益；
* Strict-PBS到AE-PBS隔离epsilon放宽收益；
* 保存`root_dynamic_rank`、负margin路径和旧模型实际错误路径；
* 主实验不写完整trace，只保存在线诊断；
* 新搜索严格opt-in，不修改旧配置和旧搜索行为；
* 查询、日志、配置、结果重算及回归测试方案完整。
