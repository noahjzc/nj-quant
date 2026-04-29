# 回测性能优化设计: 预计算缓存 + 引擎简化

Date: 2026-04-29
Status: draft

## 问题

每日回测约 6 秒。5 年数据 (~1250 天) 单次回测 ~2 小时，200 trials 串行 ~416 小时，不可接受。

### 根因

当前引擎用 Master DataFrame 累积每日全市场数据，后期 DataFrame 膨胀至 ~600 万行。关键瓶颈:

| 瓶颈 | 操作 | 占比 |
|------|------|------|
| `_advance_to_date` 每日 `pd.concat` | 每次复制全量 DataFrame (O(N²) 拷贝) | 30% |
| `_build_signal_features` sort + groupby.tail + rolling | 在数百万行上做分组滚动计算 | 25% |
| `_get_daily_stock_data` groupby | 全量 groupby 600 万行 | 13% |
| 因子提取逐股计算 WR、RET | Python 循环 + per-stock rolling | 8% |
| ATR 止损计算 | 每只持仓从 N 日数据算 ATR | 7% |

所有计算的回溯窗口不超过 20 个交易日，但引擎存了全部历史 (1250+ 天)。

## 方案

将滚动指标在构建缓存时预计算，回测引擎只做信号逻辑判断和排序，不做任何 rolling/window 计算。

### 缓存构建 (`daily_data_cache.py`)

`DailyDataCache.build()` 增强: 加载全量数据 → 按股票分组 → 每只计算滚动指标 → 按日期聚合写 Parquet。

新增预计算列 (9 列):

| 列 | 公式 | 替代的运行时计算 |
|----|------|-----------------|
| `vol_ma5` | `volume.rolling(5).mean()` | `_build_signal_features` 中 rolling |
| `vol_ma20` | `volume.rolling(20).mean()` | 同上 |
| `close_std_20` | `close.rolling(20).std()` | 同上 |
| `high_20_max` | `high.shift(1).rolling(20).max()` | 同上 |
| `atr_14` | `TR.rolling(14).mean()` | `_check_and_sell` 中 `calculate_atr()` |
| `wr_10` | `(H_max10 - C) / (H_max10 - L_min10) * 100` | `_execute_buy` 中 `williams_r()` |
| `wr_14` | 同上，14 天窗口 | 同上 |
| `ret_5` | `close / close.shift(5) - 1` | overheat 计算、因子排序 |
| `ret_20` | `close / close.shift(20) - 1` | 因子排序 |

NaN 处理: 每只股票前 N 天的滚动指标填 0 (新股/数据不足时信号条件自然不匹配)。

### 独立 CLI 入口 (`back_testing/data/build_daily_cache.py`)

```bash
python back_testing/data/build_daily_cache.py --start 2020-01-01 --end 2025-12-31
```

参数: `--start`, `--end`, `--cache-dir` (默认 `cache/daily_rotation`), `--benchmark-index` (默认 `sh000300`)。

### 引擎简化 (`daily_rotation_engine.py`)

#### 移除

- `_cache_df` (Master DataFrame) — 不再累积全量历史
- `_all_codes` — 不再预加载
- `_preloaded_cache` / `_has_fast_daily` — 不再需要
- `_preload_histories()` — 不再需要历史窗口
- `PRELOAD_DAYS = 30` / `MIN_TRADING_DAYS = 20` — 不再需要

#### 新增

- `_prev_df: pd.DataFrame` — 前一交易日全市场数据 (~4760 行)
- `_today_df: pd.DataFrame` — 当日全市场数据
    
#### 重写的方法

**`_advance_to_date`** — 滚动指针，2 行:
```python
self._prev_df = self._today_df
self._today_df = self.data_provider.get_daily_dataframe(date_str)
```

**`_get_daily_stock_data`** — 拼接 `_prev_df + _today_df` (~9500 行)，按股票分组为 `{code: 2-row DataFrame}`。

**`_build_signal_features`** — 纯列拷贝。从 `_today_df` 和 `_prev_df` 取预计算列，组装为特征矩阵，不做任何 rolling/sort/groupby。

#### 简化的方法

**`_check_and_sell`** — ATR 从 `atr_14` 列直接读取，停牌股从 `_prev_df` 取最后价格。

**`_execute_buy`** — `RET_20`、`RET_5` 从预计算列读取，`WR_10/WR_14` 从 `wr_10/wr_14` 列读取，OVERHEAT 用 `rsi_1` + `ret_5` 列值。

**`run()`** — 移除 `_preload_histories()` 调用。首日前读取一个交易日作为 `_prev_df` 初始化。

### 数据模型对比

```
Before:
  _cache_df:  单一大表, 累积 600 万行, 每天 concat
  _advance_to_date: concat + 全量过滤 + stale 清理
  每日 I/O: 拷贝 ~300 万行 (平均)
  每日内存: ~600 万行 × N 列

After:
  _prev_df:   前一日, ~4760 行
  _today_df:  当日, ~4760 行  
  _advance_to_date: read_parquet × 1
  每日 I/O: 读 1 个 Parquet 文件 (~2-3 MB)
  每日内存: ~1 万行 × N 列 (缩小 600 倍)
```

## 时间估算

| 步骤 | Before | After |
|------|--------|-------|
| `_advance_to_date` | concat 拷贝 300万行 (~1.8s) | `pd.read_parquet` (~0.05s) |
| `_get_daily_stock_data` | groupby 600万行 (~0.8s) | groupby 9500行 (~0.01s) |
| `_build_signal_features` | sort + groupby + rolling (~1.5s) | 列拷贝 (~0.01s) |
| 因子提取 (逐股计算) | ~0.3s | ~0.01s |
| ATR 止损检查 | ~0.2s | ~0.01s |
| 信号检测 | ~0.1s | ~0.1s |
| **每日合计** | **~6s** | **~0.2-0.3s** |

| 场景 | Before | After |
|------|--------|-------|
| 5年单次回测 | ~2 小时 | ~4-6 分钟 |
| 200 trials 串行 | ~416 小时 | ~16-20 小时 |

## 风险 & 注意

1. **缓存构建时间**: 首次构建 5 年数据 + 预计算全部列，预计 15-30 分钟。一次性开销。
2. **Parquet 文件大小**: 增加 9 列后每日文件从 ~1MB 增至 ~1.3MB，可接受。
3. **前 20 天数据**: 滚动指标为 0，信号自然不触发，不影响策略逻辑。
4. **向后兼容**: 不保留。旧缓存无法用于新引擎，需重新构建。
