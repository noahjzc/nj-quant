# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **quantitative stock trading** system for Chinese markets (A股). It implements a multi-factor stock selection and backtesting framework with performance analysis capabilities.

## Running the Project

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows

# Run composite backtest
python back_testing/backtest/run_composite_backtest.py

# Run rotator backtest
python back_testing/backtest/run_rotator_backtest.py

# Run full backtest
python back_testing/backtest/run_full_backtest.py

# Run tests
pytest tests/back_testing/ -v

# Run specific test file
pytest tests/back_testing/test_factor_utils.py -v
```

## Architecture

```
back_testing/
├── backtest/              # Backtest entry points
│   ├── run_composite_backtest.py   # Main composite backtest
│   ├── run_rotator_backtest.py     # Rotator-based backtest
│   └── run_full_backtest.py        # Full backtest runner
├── data/                  # Data access layer
│   ├── data_provider.py   # Unified data access (Parquet/CSV)
│   └── index_data_provider.py  # Index data for benchmark
├── factors/              # Multi-factor model
│   ├── factor_utils.py   # FactorProcessor (rank, zscore, winsorize)
│   ├── factor_config.py  # Factor weights & directions
│   └── factor_loader.py  # Load stock factor data
├── selectors/            # Stock selection
│   ├── composite_selector.py      # Original composite selector
│   ├── multi_factor_selector.py   # Multi-factor selector (new)
│   └── stock_selector.py
├── strategies/           # Trading strategies
│   ├── rsi_strategy.py, macd_strategy.py, kdj_strategy.py
│   ├── ma_strategy.py, bollinger_strategy.py
│   └── volume_strategy.py, multi_rsi_strategy.py
├── risk/                # Risk management
│   ├── risk_manager.py
│   ├── position_manager.py
│   └── stop_loss_strategies.py
├── analysis/            # Performance analysis
│   ├── performance_analyzer.py  # Sharpe, Calmar, Alpha, Beta
│   └── visualizer.py           # Charts and HTML reports
├── core/                # Core backtest engine
│   └── backtest_engine.py
└── composite_rotator.py  # Main weekly rotation controller
```

## Key Concepts

- **Stock codes**: Shanghai `sh` prefix, Shenzhen `sz` prefix
- **Data format**: Parquet (default, fast) or CSV with GBK encoding
- **Multi-factor selection**: Weighted scoring across valuation, momentum, trend, and trading factors
- **Factor directions**: `1` = larger is better, `-1` = smaller is better
- **Backtest flow**: `CompositeRotator.run_weekly()` → factor selection → rebalance → performance analysis

## Multi-Factor Model

The system supports two stock selection modes:
1. **Multi-factor (default)**: Uses `MultiFactorSelector` with configured factor weights
2. **Composite scoring**: Uses `CompositeSelector` with strategy signals

Factor configuration (`factor_config.py`):
- **Valuation**: PB, PE_TTM, PS_TTM (lower is better, direction=-1)
- **Momentum**: RSI_1, KDJ_K (stronger is better, direction=1)
- **Trend**: MA_5, MA_20 (uptrend is better, direction=1)
- **Trading**: TURNOVER, VOLUME_RATIO (more active is better, direction=1)
- **Volatility**: AMPLITUDE (lower is better, direction=-1)

## Data Provider

```python
from back_testing.data.data_provider import DataProvider

provider = DataProvider()  # Uses Parquet by default
df = provider.get_stock_data('sh600519', date='2024-01-15')
```

## Strategy Implementation

Implement `AbstractStrategy` to create a trading strategy:

```python
class MyStrategy(AbstractStrategy):
    def fill_factor(self, data: DataFrame) -> DataFrame:
        data[STRATEGY_FACTOR] = ...
        return data
```

## Performance Metrics

The `PerformanceAnalyzer` calculates:
- **Absolute**: Total return, annual return, max drawdown
- **Risk-adjusted**: Sharpe ratio, Calmar ratio, Sortino ratio
- **Relative**: Alpha, Beta, information ratio
- **Trading**: Win rate, profit/loss ratio, avg holding days, turnover

## Important Notes

- Chinese column names are defined in `data_column_names.py`
- Data files use GBK encoding for CSV
- Default data path: `data/daily_ycz/*.parquet`
- Index data path: `data/metadata/daily_ycz/index/`
