# Qlib LLM 集成调研与 nj-quant 下阶段路线图

## 调研摘要: Microsoft Qlib + RD-Agent

### Qlib 是什么

微软开源的 AI 导向量化投资平台，覆盖数据层、模型层、回测层、实验追踪的完整链条。

### RD-Agent 是什么

独立于 Qlib 的 LLM 驱动自主 R&D 智能体框架（`github.com/microsoft/RD-Agent`）。把 Qlib 作为"实验基础设施"来调用——LLM 大脑指挥量化工具箱。

### 五单元闭环架构

1. **Specification Unit** — 定义任务规格（数据接口、输出格式、执行环境）
2. **Synthesis Unit** — LLM 生成假设，用"知识森林"从历史实验中学习（成功→增加复杂度，失败→结构调整）
3. **Implementation Unit** — Co-STEER 代码生成（DAG 任务调度 + 知识库检索，Pass@5 > 90%）
4. **Validation Unit** — 因子去重（IC≥0.99 则丢弃）+ Qlib 回测验证
5. **Analysis Unit** — Multi-Armed Bandit 调度器 + SOTA 追踪（决定下一轮优化方向）

### 三个关键自我优化机制

1. **Co-STEER 知识库**: `K(t+1) = K(t) ∪ {(task, code, feedback)}` — 实验经验持续积累
2. **Multi-Armed Bandit 调度**: 8 维状态向量 + Thompson Sampling，动态选择优化因子还是优化模型
3. **因子-模型联合优化**: 因子优化和模型优化交替进行，彼此受益

### 实证结果

年化收益 14.21% vs 基线 5.70%，因子减少 70%，单次成本 <$10（LLM API 费用）

### nj-quant 差距分析

| 能力 | 状态 |
|------|------|
| 数据层 + 回测 | 已有（Parquet 缓存 + DailyRotationEngine）|
| Alpha158 因子 | 已有（Alpha158Calculator）|
| LightGBM + Optuna | 已有（MLRanker + run_ml_optimization）|
| 因子筛选+正交化 | **已完成 (2026-05-02)** |
| 时序特征提取 | **缺失 ← 下一步** |
| 元层优化调度 | 缺失 |
| LLM 驱动因子挖掘 | 缺失 |
| GNN / 多模态 | 缺失 |

## 下阶段路线图

### 第一优先: 时序特征层 + LightGBM (方向 2-B)

**问题**: 当前 ML 管线只使用因子截面值（t 时刻的因子快照），完全忽略时序动态：
- "波动率在上升" vs "波动率很高" — 后者可能是常态，前者才是危险信号
- "动量在衰减" vs "动量很低" — 前者暗示趋势即将反转

**方案**: 在 LightGBM 之前加一个轻量 Transformer/LSTM Encoder，提取因子时序特征

**架构**:
```
当前: 因子截面(t) → LightGBM → 预测收益
改进: 因子时序(t-20..t) → Transformer Encoder → 时序特征
      因子截面(t)         → 直接拼接            → LightGBM → 预测收益
```

**收益**: 捕捉非线性因子交互 + 时序模式 + 市场状态变化

### 第二优先: 元层优化调度

参考 RD-Agent 的 Bandit 调度器，在因子筛选/模型训练/框架优化之间动态分配计算预算。

### 第三优先: GNN / 多模态 (待评估)

---

## 相关资源

- RD-Agent(Q) 论文: [arXiv:2505.15155](https://arxiv.org/abs/2505.15155)
- RD-Agent 项目: https://github.com/microsoft/RD-Agent
- Qlib 项目: https://github.com/microsoft/qlib
