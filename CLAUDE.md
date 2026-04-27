# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **quantitative stock trading** system for Chinese markets (A股). It implements a daily rotation backtesting framework with Optuna-based parameter optimization, multi-factor stock ranking, and performance analysis.

## Running the Project

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows

# Single daily rotation backtest
python back_testing/backtest/run_daily_rotation.py --start 2024-01-01 --end 2024-12-31

# Optuna optimization — single-period
python back_testing/optimization/run_daily_rotation_optimization.py \
    --mode single --start 2024-01-01 --end 2024-12-31 --trials 100

# Optuna optimization — walk-forward
python back_testing/optimization/run_daily_rotation_optimization.py \
    --mode walkforward --start 2022-01-01 --end 2024-12-31 --trials 50

# Run all tests
pytest tests/back_testing/ -v

# Run specific test files
pytest tests/back_testing/rotation/test_overheat.py -v
pytest tests/back_testing/optimization/test_daily_rotation_optuna.py -v
pytest tests/back_testing/test_multi_factor_selector.py -v
```

## Architecture

```
back_testing/
├── rotation/                     # Daily rotation engine (active system)
│   ├── daily_rotation_engine.py  # Core engine: Master DataFrame cache, vectorized signals
│   ├── config.py                 # RotationConfig, MarketRegimeConfig dataclasses
│   ├── market_regime.py          # Market state detector (strong/neutral/weak)
│   ├── position_manager.py       # Position sizing with regime-aware limits
│   ├── trade_executor.py         # Trade execution and TradeRecord
│   ├── signal_engine/
│   │   ├── signal_filter.py      # Buy/sell signal detection (14 signal types)
│   │   └── signal_ranker.py      # Multi-factor weighted ranking with z-score
│   └── strategy.py               # Strategy interface
├── optimization/                 # Parameter optimization
│   ├── run_daily_rotation_optimization.py  # Optuna CLI (single/walkforward)
│   └── run_optimization.py       # Composite rotator optimization
├── data/                         # Data access layer
│   ├── data_provider.py          # Direct PostgreSQL access (SQLAlchemy)
│   ├── daily_data_cache.py       # Parquet cache + CachedProvider (cross-trial reuse)
│   └── db/                       # Database models and connection
├── analysis/                     # Performance analysis
│   ├── performance_analyzer.py   # Sharpe, Calmar, max drawdown, win rate
│   └── visualizer.py             # Charts and HTML reports
├── backtest/                     # Entry points
│   ├── run_daily_rotation.py     # Single daily rotation backtest
│   ├── run_composite_backtest.py # Legacy composite backtest
│   └── run_rotator_backtest.py   # Legacy rotator backtest
├── risk/                         # Risk management
│   ├── risk_manager.py
│   ├── position_manager.py
│   └── stop_loss_strategies.py   # ATR stop-loss/take-profit, trailing stop
├── factors/                      # Factor utilities
│   ├── factor_utils.py           # FactorProcessor (rank, zscore, winsorize)
│   └── factor_loader.py          # Load stock factor data
├── selectors/                    # Stock selection (legacy)
└── core/                         # Core backtest engine (legacy)
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

Build cache once before optimization: `DailyDataCache.build(start, end, cache_dir)`

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


Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
