# Daily Rotation 参数优化设计（Optuna）

日期: 2026-04-26
状态: 已确认

## 问题

现有 GA 优化器的染色体只能编码 12 个同质因子权重（连续实数），无法表达 `RotationConfig` 中混合类型的参数（int、List[str]、categorical）。需要一个能处理混合参数空间的优化方案。

## 方案

使用 **Optuna**（TPE Bayesian 优化）替换现有 GA，原生支持 float/int/categorical 混合参数采样。不改动现有 GA 框架，新建独立优化入口。

## 搜索空间

### 因子权重（连续，采样后归一化）

| 因子 | 范围 |
|------|------|
| RSI_1 | [0.01, 0.40] |
| RET_20 | [0.01, 0.40] |
| VOLUME_RATIO | [0.01, 0.40] |
| PB | [0.01, 0.40] |
| PE_TTM | [0.01, 0.40] |
| OVERHEAT | [0.01, 0.40] |

采样后归一化使 sum=1。

### 买入信号开关（categorical，独立 on/off）

KDJ_GOLD, MACD_GOLD, MA_GOLD, VOL_GOLD, BOLL_BREAK, HIGH_BREAK
每个单独 `suggest_categorical('signal_xxx', ['on', 'off'])`。至少保留一个。

### 信号逻辑模式（categorical）

`buy_signal_mode`: OR / AND

### 连续参数（float）

| 参数 | 范围 | 说明 |
|------|------|------|
| max_total_pct | [0.30, 1.00] | 总仓位上限 |
| max_position_pct | [0.05, 0.30] | 单只上限 |
| overheat_rsi_threshold | [60.0, 90.0] | RSI 过热阈值 |
| overheat_ret5_threshold | [0.05, 0.30] | 5日涨幅过热阈值 |
| stop_loss_mult | [1.0, 3.5] | ATR 止损倍数 |
| take_profit_mult | [2.0, 5.0] | ATR 止盈倍数 |
| trailing_pct | [0.05, 0.20] | 移动止损幅度 |
| trailing_start | [0.02, 0.10] | 移动止损启动门槛 |

### 整数参数（int）

| 参数 | 范围 | 说明 |
|------|------|------|
| max_positions | [3, 10] | 最大持仓数 |
| atr_period | [7, 21] | ATR 计算周期 |

### 固定参数（不优化）

卖出信号类型、市场状态调节、ST/涨跌停/停牌排除、基准指数

## 目标函数

- **指标**：年化 Sharpe Ratio（日频，252 日/年）
- **硬约束**：最大回撤 > 30% → fitness = 0
- equity_curve = [initial_capital] + [r.total_asset for r in results]

## Walk-Forward 支持

滚动窗口：训练 12 月、测试 6 月、步进 3 月（可配置）。每窗口独立 Optuna study（100 trials），记录各窗口最优参数和测试 Sharpe。

## 输出

- 最优参数 JSON（可回放）
- Trial 记录（参数 + Sharpe，用于分析）

## 架构

```
run_daily_rotation_optimization.py    ← 新入口
├─ sample_config(trial, base_config) → RotationConfig
├─ objective(trial) → float (Sharpe)
│   ├─ DailyRotationEngine.run()
│   └─ 计算 Sharpe + max_drawdown 约束
├─ run_single_optimization()          ← 单期优化
└─ run_walk_forward()                 ← 滚动窗口优化
```

## 文件改动

| 文件 | 改动 |
|------|------|
| `back_testing/optimization/run_daily_rotation_optimization.py` | 新建 |
| 现有 GA 文件 | 不动 |

## 依赖

```bash
pip install optuna
```

## 验证

1. `python back_testing/optimization/run_daily_rotation_optimization.py` 单期跑通无报错
2. 检查 best_params JSON 中各参数在合理范围内
3. 对比优化前后 Sharpe 是否有提升
