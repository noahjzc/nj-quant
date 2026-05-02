# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

This is a **quantitative stock trading** system for Chinese markets (A股). It implements a daily rotation backtesting framework with Optuna-based parameter optimization, multi-factor stock ranking, and performance analysis.

## Running the Project

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows

# Single daily rotation backtest
python backtesting/run_daily_rotation.py --start 2024-01-01 --end 2024-12-31

# Optuna optimization — single-period
python optimization/optuna/run_daily_rotation_optimization.py \
    --mode single --start 2024-01-01 --end 2024-12-31 --trials 100

# Optuna optimization — walk-forward
python optimization/optuna/run_daily_rotation_optimization.py \
    --mode walkforward --start 2022-01-01 --end 2024-12-31 --trials 50

# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/strategy/rotation/test_overheat.py -v
pytest tests/optimization/optuna/test_daily_rotation_optuna.py -v
pytest tests/strategy/factors/test_multi_factor_selector.py -v
```

## Architecture

```
nj-quant/
├── backtesting/                  # 回测引擎（通用框架）
│   ├── run_daily_rotation.py    # 入口脚本
│   ├── analysis/                # 性能分析、可视化
│   │   ├── performance_analyzer.py
│   │   └── visualizer.py
│   └── risk/                    # 风险管理
│       ├── risk_manager.py
│       ├── position_manager.py
│       └── stop_loss_strategies.py
│
├── strategy/                     # 策略定义
│   ├── rotation/                # 每日轮转策略
│   │   ├── daily_rotation_engine.py  # Core engine: Master DataFrame cache, vectorized signals
│   │   ├── config.py            # RotationConfig, MarketRegimeConfig dataclasses
│   │   ├── market_regime.py     # Market state detector (strong/neutral/weak)
│   │   ├── position_manager.py  # Position sizing with regime-aware limits
│   │   ├── trade_executor.py    # Trade execution and TradeRecord
│   │   ├── strategy.py          # Strategy interface
│   │   └── signal_engine/       # 信号引擎
│   │       ├── signal_filter.py # Buy/sell signal detection (14 signal types)
│   │       └── signal_ranker.py # Multi-factor weighted ranking with z-score
│   ├── factors/                 # 因子工具
│   │   ├── factor_utils.py      # FactorProcessor (rank, zscore, winsorize)
│   │   └── factor_loader.py     # Load stock factor data
│   ├── signals/                 # 信号类型定义（预留）
│   └── ranking/                 # 多因子排名（预留）
│
├── data/                        # 数据层统一管理
│   ├── providers/               # 数据提供者
│   │   ├── data_provider.py     # Direct PostgreSQL access (SQLAlchemy)
│   │   └── index_data_provider.py
│   ├── cache/                   # Parquet 缓存
│   │   ├── daily_data_cache.py # Parquet cache + CachedProvider (cross-trial reuse)
│   │   └── build_daily_cache.py
│   ├── db/                     # 数据库模型
│   │   ├── models.py
│   │   └── connection.py
│   └── sync/                    # 数据同步客户端
│       ├── akshare_client.py
│       ├── baostock_client.py
│       ├── tushare_client.py
│       └── ...
│
├── optimization/                 # 参数优化
│   └── optuna/                  # Optuna 优化框架
│       └── run_daily_rotation_optimization.py  # Optuna CLI (single/walkforward)
│
├── signal_pipeline/             # 信号生成管线
│   ├── generators/              # 信号生成
│   │   ├── signal_generator.py
│   │   ├── intraday_signal.py
│   │   └── indicator_calculator.py
│   └── backfill/               # 数据回填
│       ├── night_backfill.py
│       ├── batch_backfill.py
│       └── data_merger.py
│
├── web/                        # 前端 + API
│   ├── frontend/
│   └── server/
│
├── scripts/                     # 运维脚本
└── tests/                       # 测试（与源码结构对齐）
```

## Daily Rotation Engine

The engine (`DailyRotationEngine`) processes one trading day at a time:

1. **Preload**: 30 days of history loaded into a single Master DataFrame (indexed by `trade_date`, with `stock_code` column), avoiding per-stock I/O
2. **Per day**: Update cache → detect market regime → build signal features (vectorized) → check exits (sell signals + ATR stops + trailing stop) → rank candidates → allocate buys
3. **Two-phase execution**: Sell phase first (free up cash), then buy phase (allocate to top-ranked candidates)
4. **Signal pipeline**: Layer 1 = binary signal detection (14 signal types), Layer 2 = multi-factor weighted ranking (z-score + direction-adjusted weighted sum)

Key performance optimizations:
- `_build_signal_features()`: Vectorized via groupby rolling transforms on the master cache `tail(21)` per stock, producing a feature matrix for all candidates in a single operation
- `_cache_df` updated incrementally per day (no repeated queries)
- `CachedProvider` reads from pre-built Parquet files (no DB queries during optimization trials)

## Configuration

`RotationConfig` is a Python dataclass. **All fields must have type annotations** — unannotated fields become class variables, not instance fields, and won't appear in `__init__`.

Key config groups:
- **Position sizing**: `max_total_pct`, `max_position_pct`, `max_positions`
- **Buy signals**: `buy_signal_types` (list), `buy_signal_mode` (`'AND'`/`'OR'`)
- **Rank factors**: `rank_factor_weights` (dict), `rank_factor_directions` (dict: 1/-1)
- **Market regime**: `MarketRegimeConfig` with strong/neutral/weak state params
- **Stops**: ATR-based stop-loss/take-profit multipliers, trailing stop thresholds
- **Overheat penalty**: RSI + 5-day return thresholds to avoid chasing overbought stocks

## Data Access: Two Providers

**`DataProvider`** (direct DB): Queries PostgreSQL via SQLAlchemy. Used for single backtests.
- `get_all_stock_codes()` → `list`
- `get_batch_histories(codes, end_date, start_date)` → `{code: DataFrame}`
- `get_index_data(index_code, start_date, end_date)` → `DataFrame`
- `get_stocks_for_date(codes, date)` → `{code: dict}` — dicts do NOT contain `stock_code` key

**`CachedProvider`** (Parquet): Reads from pre-built Parquet cache. Used by Optuna trials for cross-trial data reuse.
- Same interface as DataProvider for the three main methods
- Extra: `get_daily_dataframe(date)` → full market DataFrame (used by engine fast path)

Build cache once before optimization: `DailyDataCache.build(start, end, cache_dir)` (located at `data/cache/daily_data_cache.py`)

## Optimization

Uses **Optuna TPE** (Tree-structured Parzen Estimator) for Bayesian optimization.

14 sampled parameters in `sample_config()`: position sizing, signal mode, factor weights, ATR stops, trailing stop, overheat thresholds.

Modes:
- `single`: Optimize over one date range, minimize `-Sharpe`
- `walkforward`: Rolling windows, report per-window stats

Dependencies: `optuna` is required but NOT listed in `requirements.txt`.

## Known Pitfalls

1. **Dataclass fields need type annotations**: `field_name = value` creates a class variable, not an instance field. Use `field_name: type = value`. Missing annotations on `RotationConfig` caused all Optuna trials to fail with `unexpected keyword argument`.

2. **`get_stocks_for_date()` missing `stock_code`**: Row dicts from `DataProvider.get_stocks_for_date()` don't include `stock_code`. Engine's `_advance_to_date` must add it: `row_data['stock_code'] = stock_code`. Missing this caused zero trades (all rows invisible to stock_code-based filtering).

3. **Signal feature index alignment**: `groupby.last()` returns `stock_code` index; `groupby.nth(-2)` returns original index. Must align both to the same index before combining into a feature DataFrame.

## Important Notes

- Stock codes: Shanghai `sh` prefix, Shenzhen `sz` prefix
- Factor directions: `1` = larger is better, `-1` = smaller is better
- Database config: `config/database.ini` (PostgreSQL)
- Virtual environment: `.venv/` (Windows)


## do not use git for this