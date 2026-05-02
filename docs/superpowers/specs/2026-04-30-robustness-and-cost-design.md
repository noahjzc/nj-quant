# 稳健性检验 & 真实交易模拟增强 设计文档

## 目标

在现有每日轮动回测基础上，增加两个能力：

1. **真实交易模拟增强**：滑点模型、冲击成本（平方根模型）、T+1约束、流动性约束
2. **稳健性检验**：蒙特卡洛模拟、CSCV过拟合检测、参数敏感性分析、Deflated Sharpe/PBO

## 架构

```
nj-quant/
├── backtesting/
│   ├── analysis/
│   │   └── performance_analyzer.py    # 扩展：新增统计指标
│   └── costs/                         # 🆕 交易成本模块
│       ├── __init__.py
│       ├── cost_model.py             # 统一成本模型（费率 + 滑点 + 冲击）
│       └── market_constraints.py     # T+1、涨跌停、流动性、ST 过滤
│
├── robustness/                        # 🆕 稳健性检验模块
│   ├── __init__.py
│   ├── monte_carlo.py                # 蒙特卡洛模拟
│   ├── cscv.py                       # CSCV 过拟合检测
│   ├── sensitivity.py                # 参数敏感性分析
│   └── statistics.py                 # Deflated Sharpe、PBO 等统计指标
```

## 职责划分

- **PerformanceAnalyzer** — "回测表现如何"：基础指标 + 分布分析 + 基准对比 + 月度明细。暴露 `daily_returns` 供稳健性模块使用。
- **RobustnessAnalyzer** — "结果是否可信"：从 PerformanceAnalyzer 获取指标，做统计检验和过拟合检测。不重复计算指标。
- **CostModel** — "交易花多少钱"：统一管理费率、滑点、冲击成本，替代 `TradeExecutor` 中分散的成本计算。
- **MarketConstraints** — "能不能交易"：T+1、涨跌停、停牌、ST、流动性过滤，替代引擎中分散的约束检查。

## 模块一：成本模块

### cost_model.py

```python
class CostModel:
    # 固定费率（从 TradeExecutor 迁出）
    STAMP_DUTY = 0.001          # 印花税 0.1%
    TRANSFER_FEE = 0.00002      # 过户费 0.002%
    BROKERAGE = 0.0003          # 佣金 0.03%
    MIN_BROKERAGE = 5.0         # 最低佣金

    # 可变成本（可配置）
    slippage_bps: float = 1.0   # 滑点
    impact_model: str = 'sqrt'  # 冲击模型: 'sqrt' | 'fixed' | 'none'

    def buy_cost(price, shares, volume_today=None) -> CostBreakdown
    def sell_cost(price, shares, volume_today=None) -> CostBreakdown
    # 平方根冲击: impact = sigma * sqrt(Q / V)
```

### market_constraints.py

```python
class MarketConstraints:
    def can_buy(stock_code, price, volume_today) -> Tuple[bool, str]
        # 涨停跳过 | 停牌跳过 | ST跳过 | 当日成交额 < min_volume 跳过

    def can_sell(position, trade_date) -> Tuple[bool, str]
        # T+1: buy_date == trade_date → 不可卖
        # 跌停跳过

    def filter_pool(today_df, date) -> pd.DataFrame
        # 一站式过滤，替代引擎 _filter_stock_pool 中分散的逻辑
```

### 集成点（DailyRotationEngine）

- `_filter_stock_pool` → 委托给 `MarketConstraints.filter_pool()`
- `_check_and_sell` → 卖出前 `MarketConstraints.can_sell()` 检查 T+1
- `_execute_buy` 成本计算 → `CostModel.buy_cost()`
- `TradeExecutor` 迁移成本常量到 `CostModel`，保留为兼容层

## 模块二：PerformanceAnalyzer 扩展

### 新增指标

| 指标 | 方法 | 说明 |
|------|------|------|
| 信息比率 | `information_ratio()` | (策略年化 - 基准年化) / 跟踪误差 |
| Alpha/Beta | `alpha_beta()` | CAPM 回归，含 R² |
| 收益偏度/峰度 | `skewness_kurtosis()` | 尾部风险评估 |
| 滚动 Sharpe | `rolling_sharpe(window)` | 不同市场环境的一致性 |
| 月度收益明细 | `monthly_returns()` | 按年-月分组 |
| 最大回撤持续天数 | `max_drawdown_duration()` | 最长水下时间 |

### 接口变动

- 新增属性 `daily_returns: np.ndarray`（暴露，供 RobustnessAnalyzer 使用）
- 已有 `equity_curve` 输入保持不变
- 所有新指标仅当提供了 `benchmark_returns` 时计算基准相关指标

## 模块三：稳健性检验

### monte_carlo.py

- 对日收益率放回重采样，生成 N 条模拟净值曲线
- 非参数方法，不假设收益分布
- 输出：Sharpe 分布（均值 + 95%CI）、最大回撤分布
- 默认 `n_sim=2000`

### cscv.py

- 组合对称交叉验证（Bailey et al., 2017）
- 回测期切 S 段 → 随机选 S/2 做 IS，剩余 OOS → 重复 N 次
- 计算 IS 最优参数在 OOS 的排名衰减程度
- 输出：过拟合概率 PBO、排名衰减系数、是否通过检验

### sensitivity.py

- 对每个最优参数 ±20% 扰动，重跑回测，测 Sharpe 变化
- 输出：每个参数的敏感度（sharpe_change_pct）、整体稳定性评分 0~1
- 仅在最终验证时执行（需多次回测）

### statistics.py

- Deflated Sharpe Ratio：考虑多重测试惩罚后的 Sharpe 显著性
- PBO（Probability of Backtest Overfitting）：基于 IS/OOS 排名对比的过拟合概率

### RobustnessAnalyzer（门面）

```python
class RobustnessAnalyzer:
    def __init__(self, analyzer: PerformanceAnalyzer)
    def run_all() -> RobustnessReport

# RobustnessReport:
#   mc: MCSimulationResult
#   cscv: CSCVResult
#   sensitivity: SensitivityResult
#   pbo: float
#   deflated_sharpe: float
```

## 集成流程

```
回测 → PerformanceAnalyzer → 指标
                            → RobustnessAnalyzer → 检验报告
                            → CostModel（回测中已使用）
```

## 两个阶段

| 阶段 | 内容 | 文件数 |
|------|------|--------|
| Phase 1 | costs/ 模块 + robustness/ 模块 + PerformanceAnalyzer 扩展 | ~8 |
| Phase 2 | 优化后 Top 5 敏感性筛选 | ~1 |

Phase 2 待 Phase 1 验证通过后实施。

### Phase 2 详细设计：后置过滤

Optuna 优化结束后，取 Sharpe 最高的 Top 5 组参数，每组成规模跑敏感性分析，最后按综合评分选出最优参数。

```
Optuna 优化完成 → 取 Top 5（按 Sharpe）
  → 每组跑 SensitivityAnalyzer（每个参数 ±20%，重跑回测）
  → 综合评分 = 归一化Sharpe × 0.6 + 稳定性评分 × 0.4
  → 输出最终参数
```

在 `run_daily_rotation_optimization.py` 末尾加一个函数 `_select_by_robustness(top_params, engine_factory)`，不修改 Optuna 目标函数本身。

## 实施注意

- 不影响现有回测参数和优化流程
- `CostModel` 的滑点和冲击默认启用，可用开关关闭以对比
- `sensitivity.py` 中每个参数 ±20% 各跑一次回测，14个参数约28次回测，可在非优化时段运行
