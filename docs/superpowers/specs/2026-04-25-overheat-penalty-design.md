# 过热度惩罚因子 (OVERHEAT) 设计

日期: 2026-04-25
状态: 已确认

## 问题

选股时经常买入短期波动大的个股，买入后 1-2 天遭遇跌停止损，频繁止损伤害收益率。

根因：当前买入信号（HIGH_BREAK、MA_GOLD 等）天然在股价大涨后触发，排序因子（RSI_1、RET_20）进一步给追高股打高分，无任何过热防护。

## 方案

在现有多因子排序系统中新增 `OVERHEAT` 惩罚因子，对 RSI 超买 + 短期涨幅双高的股票降权。

### 过热度计算公式

```
RSI分量  = max(0, (RSI_1 - 75) / (100 - 75))
涨幅分量 = min(1.0, max(0, (RET_5 - 0.15) / 0.35))
过热度   = (RSI分量 + 涨幅分量) / 2
```

仅当 RSI_1 > 75 **且** RET_5 > 15% 时计算过热度，否则为 0。值域 [0, 1]。

### 排序集成

`OVERHEAT` 作为排序因子加入 `SignalRanker`，方向 -1（越小越好），权重 0.20。

- 不过热时 OVERHEAT=0，不影响排名
- 过热时 OVERHEAT>0，通过 z-score + 方向反转机制拉低综合得分

## 改动清单

### 1. `back_testing/rotation/config.py` — RotationConfig

新增配置项：
```python
overheat_rsi_threshold: float = 75.0
overheat_ret5_threshold: float = 0.15
```

`rank_factor_weights` 追加 `'OVERHEAT': 0.20`

`rank_factor_directions` 追加 `'OVERHEAT': -1`

### 2. `back_testing/rotation/daily_rotation_engine.py` — _execute_buy

在 `factor_data_dict` 构建循环中，新增 RET_5 和 OVERHEAT 计算：
- `RET_5`: close / 5日前close - 1（与现有 RET_20 逻辑一致）
- `OVERHEAT`: 按上述公式计算

### 3. `SignalRanker` — 无需改动

已自动处理 factor_data 中的任意因子。

## 参数默认值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| overheat_rsi_threshold | 75 | RSI 超买阈值 |
| overheat_ret5_threshold | 0.15 | 5日涨幅阈值 |
| OVERHEAT 权重 | 0.20 | 惩罚因子在排序中的权重 |

## 验证

1. 运行 `python back_testing/backtest/run_daily_rotation.py --start 2024-01-01 --end 2024-06-30`，确认无报错
2. 检查日志中 [TOP] 买入候选排名，确认高 RSI + 高涨幅股票排名下降
3. 对比引入前后的 Sharpe、最大回撤、交易次数

## 后续调优

阈值和权重可在回测中调整，无需改代码逻辑。
