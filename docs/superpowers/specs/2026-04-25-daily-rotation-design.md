# 每日全市场轮动回测系统设计

## 1. 概述

**目标**：实现一个**每日全市场轮动**回测模式。每天扫描全市场股票，根据技术信号生成买卖信号，动态持仓，最终输出收益率曲线和绩效指标。

**核心特点**：
- 信号模块化：买入/卖出信号可配置、可组合
- 策略抽象：可独立运行，也可作为 GA 遗传算法的适应度函数
- 配置化：持仓上限、资金分配、股票池过滤等均可配置

---

## 2. 整体架构

```
DailyRotationEngine
├── DataProvider              # 数据访问（每日全市场日线）
├── SignalEngine              # 信号引擎
│   ├── SignalFilter          # 第一层：技术指标金叉/死叉过滤
│   └── SignalRanker          # 第二层：多因子评分排序
├── PortfolioManager          # 持仓管理
│   ├── positions             # 当前持仓 {stock_code: Position}
│   └── available_capital    # 可用资金
├── TradeExecutor             # 交易执行（含手续费、印花税）
└── PerformanceTracker       # 绩效记录

DailyRotationStrategy (抽象接口)
├── run(start_date, end_date) # 独立运行模式
└── fitness(params)           # GA 适应度模式
```

---

## 3. 每日流程

```
for each_date in date_range:
    1. 获取当日全市场股票列表（含日线数据）
    2. 过滤股票池（排除 ST、涨跌停）
    3. 遍历持仓 → 检查卖出信号 → 触发则卖出
    4. 全市场扫描 → 检查买入信号 → 生成候选股列表
    5. 对候选股按信号强弱排序 → 取 TOP X 进行买入
    6. 记录当日净值、持仓、交易
```

---

## 4. 信号设计

### 4.1 两层信号架构

**第一层：信号过滤器（Signal Filter）**
- 作用：快速从全市场筛选出候选股
- 触发条件：**任一**金叉信号即进入候选（逻辑或）
- 金叉信号列表：
  - `KDJ_GOLD`：K 线从下穿越 D 线
  - `MACD_GOLD`：DIF 从下穿越 DEA
  - `MA_GOLD`：MA5 从下穿越 MA20
  - `VOL_GOLD`：VOL_MA5 上穿 VOL_MA20
  - `DMI_GOLD`：+DI 上穿 -DI
  - `BOLL_BREAK`：价格站上布林带上轨
  - `HIGH_BREAK`：价格突破 N 日高点

- 死叉信号列表（对应卖出）：
  - `KDJ_DEATH`、`MACD_DEATH`、`MA_DEATH`、`VOL_DEATH`、`DMI_DEATH`、`BOLL_BREAK_DOWN`、`HIGH_BREAK_DOWN`

**第二层：信号排序器（Signal Ranker）**
- 作用：对候选股按信号强弱排序
- 使用多因子加权评分：
  - `RSI`：RSI_1 值，越低越有反弹动力
  - `MOMENTUM`：N 日动量
  - `VOL_RATIO`：成交量放大率
  - `TREND`：均线多头排列程度
  - `VALUE`：PB、PE 估值因子
- 最终得分 = Σ(factor_weight × factor_value)

### 4.2 买入信号强度

买入信号强弱的计算：

```
signal_strength = α × RSI_score + β × MOMENTUM_score + γ × VOL_score + δ × TREND_score + ε × VALUE_score
```

其中各因子分数均做 zscore 标准化。

### 4.3 卖出信号触发

持仓股出现**任一**死叉信号即触发卖出。

---

## 5. 资金管理

**分配公式：**
```
每只持仓可用资金 = 总资产（持仓 + 现金）/ N（最大持仓数）
X = N - M（已持仓数）→ 可买入股数
每只买入金额 = 每只持仓可用资金（等权重分配）
```

**可配置项：**
- `max_positions`：最大持仓数（可作为 GA 参数）
- `initial_capital`：初始资金，默认 100 万
- `position_size_type`：`equal_weight`（等权重）或 `signal_weighted`（按信号强弱）

---

## 6. 股票池过滤

- 排除 ST 股、*ST 股
- 排除涨停股（当日 close = upper_limit）
- 排除跌停股（当日 close = lower_limit）
- 可选：排除停牌股（当日成交为 0）

---

## 7. 交易成本

沿用现有 `BacktestEngine` 的成本设置：
- 印花税：0.1%（卖出时收取）
- 过户费：0.002%（买卖都收取）
- 券商佣金：0.03%，最低 5 元

---

## 8. GA 适配接口

```python
class DailyRotationStrategy:
    def fitness(self, genome: dict) -> float:
        """
        GA 适应度函数
        genome 包含：max_positions, buy_signals, sell_signals,
                    factor_weights, stop_loss 等参数
        返回：绩效指标（Sharpe Ratio 或 总收益率）
        """
        params = self.parse_genome(genome)
        result = self.run(params)
        return result.sharpe_ratio

    def parse_genome(self, genome: dict) -> RotationConfig:
        """将 GA 的 genome 映射为策略配置"""
        ...
```

**GA 可优化的参数：**
- `max_positions`：持仓数量
- `factor_weights`：各因子权重
- `signal_thresholds`：信号触发阈值
- `buy_signal_types`：使用哪些买入信号组合

---

## 9. 输出结果

**独立运行模式：**
- 收益率曲线
-绩效指标：总收益率、年化收益率、Sharpe、Calmar、最大回撤、Win Rate
- 交易记录明细
- 持仓变化记录

**GA 模式：**
- `fitness()` 返回绩效指标（用于 GA 选择、交叉、变异）

---

## 10. 文件结构

```
back_testing/
├── rotation/
│   ├── __init__.py
│   ├── daily_rotation_engine.py    # 核心引擎
│   ├── signal_engine.py             # 信号引擎
│   │   ├── base_signal.py           # 基础信号类
│   │   ├── signal_filter.py         # 第一层信号过滤
│   │   └── signal_ranker.py         # 第二层信号排序
│   ├── portfolio_manager.py          # 持仓管理
│   ├── trade_executor.py            # 交易执行
│   ├── config.py                    # 策略配置类
│   └── strategy.py                  # 抽象策略接口
├── backtest/
│   └── run_daily_rotation.py        # 独立运行入口
└── optimization/
    └── rotation_ga_fitness.py      # GA 适应度函数
```

---

## 11. 关键设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 持仓上限 | 可配置，默认 5 | 平衡分散与管理复杂度 |
| 资金分配 | 等权重 | 简单、可解释，便于 GA 优化 |
| 买入信号 | 任一金叉即候选 | 第一层宽松，不漏机会 |
| 卖出信号 | 任一死叉即卖出 | 严格止损/止盈 |
| 信号排序 | 多因子加权评分 | 与现有因子体系一致 |
| GA 适配 | 抽象 fitness 接口 | 解耦策略与 GA |

---

## 12. 与现有系统的关系

- **复用** `DataProvider` 获取全市场日线数据
- **复用** `PerformanceAnalyzer` 计算绩效指标
- **复用** 现有 `factor_utils` 做因子标准化
- **独立** 于 `BacktestEngine`（单股票）和 `CompositeRotator`（周频）
