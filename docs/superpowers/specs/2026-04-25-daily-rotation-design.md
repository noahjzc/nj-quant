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

沿用现有 `PositionManager` 的资金管理逻辑：

```
单只持仓上限 = 总资产 × 20%
总持仓上限 = 总资产 × 90%
每只买入股数 = floor(min(单只上限, 剩余可用资金) / 单价 / 100) × 100
```

**基础配置项（均可通过配置管理）：**

| 参数 | 默认值 | 说明 | 可作为 GA 参数 |
|------|--------|------|--------------|
| `initial_capital` | 100 万 | 初始资金 | 否 |
| `max_total_pct` | 0.90 (90%) | 总仓位上限 | 是 |
| `max_position_pct` | 0.20 (20%) | 单只持仓上限 | 是 |
| `max_positions` | 5 | 最大持仓数量 | 是 |

---

## 6. 市场状态动态调节

每日根据大盘表现动态调整风险敞口参数，使策略在不同市场环境下自适应调整。

**大盘状态判断指标：**
- **大盘趋势**：`index_ma5 / index_ma20 - 1`（指数收盘价相对20日均线的位置）
- **大盘动量**：`index_return_Nd`（N日收益率，如20日）
- **大盘波动率**：`index_ATR / index_close`（ATR 相对价格的比率）

**市场状态分类：**

| 状态 | 判断条件 | 参数调整 |
|------|----------|----------|
| **强势市场** | 上涨趋势 + 高动量 + 低波动 | `max_total_pct=90%`, `max_position_pct=20%`, `max_positions=5` |
| **弱势市场** | 下跌趋势或高波动 | `max_total_pct=60%`, `max_position_pct=15%`, `max_positions=3` |
| **震荡市场** | 其他情况 | `max_total_pct=75%`, `max_position_pct=15%`, `max_positions=4` |

**参数调节示例（强势→弱势切换）：**
```python
# 强势市场 → 弱势市场
max_total_pct:    0.90 → 0.60
max_position_pct: 0.20 → 0.15
max_positions:    5    → 3
```

**配置化：** 各状态的阈值参数可通过 `MarketRegimeConfig` 配置：
```python
regime_config = MarketRegimeConfig(
    strong_trend_threshold=0.05,    # 大盘MA多头阈值（5%）
    weak_trend_threshold=-0.03,      # 大盘MA空头阈值（-3%）
    high_volatility_threshold=0.03,  # 高波动率阈值（3%）
    lookback_period=20,              # 大盘动量回溯期（N日）
    regime_params={
        'strong':  {'max_total_pct': 0.90, 'max_position_pct': 0.20, 'max_positions': 5},
        'neutral': {'max_total_pct': 0.75, 'max_position_pct': 0.15, 'max_positions': 4},
        'weak':    {'max_total_pct': 0.60, 'max_position_pct': 0.15, 'max_positions': 3},
    }
)
```

---

## 7. 股票池过滤

- 排除 ST 股、*ST 股
- 排除涨停股（当日 close = upper_limit）
- 排除跌停股（当日 close = lower_limit）
- 可选：排除停牌股（当日成交为 0）

---

## 8. 交易成本

沿用现有 `BacktestEngine` 的成本设置：
- 印花税：0.1%（卖出时收取）
- 过户费：0.002%（买卖都收取）
- 券商佣金：0.03%，最低 5 元

---

## 9. GA 适配接口

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
- `max_total_pct`：总仓位上限
- `max_position_pct`：单只持仓上限
- `factor_weights`：各因子权重
- `signal_thresholds`：信号触发阈值
- `buy_signal_types`：使用哪些买入信号组合
- `market_regime_params`：各市场状态的阈值参数

---

## 10. 输出结果

**独立运行模式：**
- 收益率曲线
-绩效指标：总收益率、年化收益率、Sharpe、Calmar、最大回撤、Win Rate
- 交易记录明细
- 持仓变化记录
- 每日市场状态记录（用于分析）

**GA 模式：**
- `fitness()` 返回绩效指标（用于 GA 选择、交叉、变异）

---

## 11. 文件结构

```
back_testing/
├── rotation/
│   ├── __init__.py
│   ├── daily_rotation_engine.py      # 核心引擎
│   ├── signal_engine/                # 信号引擎
│   │   ├── __init__.py
│   │   ├── base_signal.py           # 基础信号类
│   │   ├── signal_filter.py         # 第一层信号过滤
│   │   └── signal_ranker.py         # 第二层信号排序
│   ├── market_regime.py              # 大盘状态判断 + 动态参数调节
│   ├── config.py                    # 策略配置类（含市场状态参数）
│   └── strategy.py                  # 抽象策略接口
├── backtest/
│   └── run_daily_rotation.py        # 独立运行入口
└── optimization/
    └── rotation_ga_fitness.py       # GA 适应度函数（RotationFitnessEvaluator）
```

---

## 12. 关键设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 持仓上限 | 可配置，默认 5 | 平衡分散与管理复杂度 |
| 资金分配 | 复用 PositionManager（90%+20%） | 与现有风控体系一致 |
| 市场调节 | 动态参数（强势/中性/弱势） | 自适应风险敞口 |
| 买入信号 | 任一金叉即候选 | 第一层宽松，不漏机会 |
| 卖出信号 | 任一死叉即卖出 | 严格止损/止盈 |
| 信号排序 | 多因子加权评分 | 与现有因子体系一致 |
| GA 适配 | 抽象 fitness 接口 + 新增 Evaluator | 不破坏现有 GA 流程 |

---

## 13. 与现有系统的关系

- **复用** `DataProvider` 获取全市场日线数据
- **复用** `PerformanceAnalyzer` 计算绩效指标
- **复用** `RiskManager.check_exit()` 持仓风控监控
- **复用** `PositionManager` 资金分配逻辑
- **复用** 现有 `factor_utils` 做因子标准化
- **新增** `RotationFitnessEvaluator` 作为 GA 适应度接口（不破坏现有 `FitnessEvaluator`）
- **独立** 于 `BacktestEngine`（单股票）和 `CompositeRotator`（周频）
