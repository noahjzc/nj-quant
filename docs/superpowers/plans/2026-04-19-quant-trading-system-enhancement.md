# Quantitative Trading System Enhancement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:
> executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance the quantitative trading backtesting system with 5 major improvements: strategy combination, risk
management, parameter optimization, portfolio backtesting, and visualization.

**Architecture:** The system will be enhanced modularly:

- `backtest_engine.py` - Core engine with risk management (stop loss/take profit)
- `strategies/` - Strategy implementations (MA, MACD, RSI, combined)
- `portfolio_optimizer.py` - Parameter optimization module
- `portfolio_backtest.py` - Multi-stock portfolio backtesting
- `visualizer.py` - Chart visualization module
- `run_full_backtest.py` - Unified entry point

**Tech Stack:** Python, pandas, numpy, matplotlib

---

## File Structure

```
back_testing/
├── backtest_engine.py      # Core engine (modified: add stop loss/take profit)
├── ma_strategy.py         # MA strategy (existing)
├── macd_strategy.py        # MACD strategy (existing)
├── rsi_strategy.py        # RSI strategy (existing)
├── combined_strategy.py    # Create: Combined trend + reversal strategy
├── stop_loss.py           # Create: Stop loss / take profit handler
├── run_backtest.py        # Single stock backtest (existing)
├── run_full_backtest.py   # Create: Portfolio backtest entry point
├── portfolio_optimizer.py # Create: Parameter optimization module
└── visualizer.py         # Create: Chart visualization

docs/superpowers/plans/
└── 2026-04-19-quant-trading-system-enhancement.md
```

---

## Task 1: Add Stop Loss / Take Profit to BacktestEngine

**Files:**

- Modify: `back_testing/backtest_engine.py:1-30` (add risk parameters)
- Modify: `back_testing/backtest_engine.py:77-165` (simulate_trades with stop loss)

- [ ] **Step 1: Add risk management parameters to BacktestEngine.__init__**

```python
def __init__(self, stock_code: str, data_path: str, initial_capital: float = 100000.0,
             benchmark_index: str = None, stop_loss: float = None, take_profit: float = None):
    # ... existing code ...
    self.stop_loss = stop_loss  # e.g., 0.10 means 10% stop loss
    self.take_profit = take_profit  # e.g., 0.20 means 20% take profit
```

- [ ] **Step 2: Modify simulate_trades to check stop loss / take profit**

In the trading loop, after recording portfolio value, add:

```python
# Check stop loss / take profit
if self.position:
    unrealized_pnl = (price - buy_price) / buy_price
    if self.stop_loss and unrealized_pnl <= -self.stop_loss:
        # Trigger stop loss - sell
        signal = 0
    elif self.take_profit and unrealized_pnl >= self.take_profit:
        # Trigger take profit - sell
        signal = 0
```

- [ ] **Step 3: Track stop loss events in trade record**

Add `'stop_loss_triggered': bool` and `'take_profit_triggered': bool` to trade records.

- [ ] **Step 4: Test stop loss on sh600519**

Run:

```bash
python -c "
from back_testing.ma_strategy import MAStrategy
engine = MAStrategy('sh600519', r'D:\workspace\code\mine\quant\data\metadata\daily_ycz', 100000, stop_loss=0.10, take_profit=0.20)
result = engine.run()
print(f'总收益: {result[\"total_return\"]:.2%}, 最大回撤: {result[\"max_drawdown\"]:.2%}')
"
```

Expected: Lower total return but significantly reduced max drawdown.

- [ ] **Step 5: Commit**

```bash
git add back_testing/backtest_engine.py
git commit -m "feat: add stop loss and take profit to backtest engine"
```

---

## Task 2: Create Combined Strategy (Trend + Reversal Filter)

**Files:**

- Create: `back_testing/combined_strategy.py`

- [ ] **Step 1: Create CombinedStrategy class**

```python
from back_testing.backtest_engine import BacktestEngine
from back_testing.ma_strategy import MAStrategy
from back_testing.rsi_strategy import RSIReversalStrategy
import numpy as np
from pandas import DataFrame


class CombinedStrategy(BacktestEngine):
    """
    Combined Trend + Reversal Strategy

    Buy: MA5 crosses above MA20 (trend signal) AND RSI < 40 (not overbought)
    Sell: MA5 crosses below MA20 (trend signal) OR RSI > 60 (overbought)

    Goal: Reduce false signals by combining trend-following with reversal filter
    """

    def generate_signals(self) -> DataFrame:
        df = self.data.copy()

        ma5_col = 'MA_5'
        ma20_col = 'MA_20'
        rsi_col = 'rsi1'

        # MA golden cross
        ma_golden = (df[ma5_col] > df[ma20_col]) & (df[ma5_col].shift(1) <= df[ma20_col].shift(1))
        # MA dead cross
        ma_dead = (df[ma5_col] < df[ma20_col]) & (df[ma5_col].shift(1) >= df[ma20_col].shift(1))

        # Buy: golden cross AND RSI not overbought
        buy_signal = ma_golden & (df[rsi_col] < 40)

        # Sell: dead cross OR RSI overbought
        sell_signal = ma_dead | (df[rsi_col] > 60)

        df['TRADE_SIGNAL'] = np.nan
        df.loc[buy_signal, 'TRADE_SIGNAL'] = 1
        df.loc[sell_signal, 'TRADE_SIGNAL'] = 0
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df
```

- [ ] **Step 2: Test combined strategy on sh600519**

Run:

```bash
python -c "
from back_testing.combined_strategy import CombinedStrategy
engine = CombinedStrategy('sh600519', r'D:\workspace\code\mine\quant\data\metadata\daily_ycz', 100000, 'sh000001')
result = engine.run()
print(f'总收益: {result[\"total_return\"]:.2%}, 胜率: {result[\"win_rate\"]:.2%}, 交易次数: {result[\"total_trades\"]}')
"
```

Expected: Fewer trades than pure MA, potentially higher win rate.

- [ ] **Step 3: Commit**

```bash
git add back_testing/combined_strategy.py
git commit -m "feat: add combined trend + reversal strategy"
```

---

## Task 3: Create Portfolio Optimizer (Parameter Optimization)

**Files:**

- Create: `back_testing/portfolio_optimizer.py`

- [ ] **Step 1: Create PortfolioOptimizer class**

```python
import pandas as pd
import numpy as np
from itertools import product


class PortfolioOptimizer:
    """
    参数优化器
    对策略参数进行网格搜索，找到历史表现最佳的参数组合
    """

    def __init__(self, strategy_class, stock_codes, data_path, benchmark_index='sh000001'):
        self.strategy_class = strategy_class
        self.stock_codes = stock_codes
        self.data_path = data_path
        self.benchmark_index = benchmark_index

    def optimize_rsi(self, rsi_buy_levels, rsi_sell_levels):
        """优化RSI策略参数"""
        results = []
        for rsi_buy, rsi_sell in product(rsi_buy_levels, rsi_sell_levels):
            total_return = 0
            total_trades = 0
            total_win_rate = 0

            for code in self.stock_codes:
                engine = self.strategy_class(
                    stock_code=code,
                    data_path=self.data_path,
                    benchmark_index=self.benchmark_index
                )
                # Modify strategy to use custom RSI levels
                result = engine.run()
                total_return += result['total_return']
                total_trades += result['total_trades']
                total_win_rate += result['win_rate']

            avg_return = total_return / len(self.stock_codes)
            avg_win_rate = total_win_rate / len(self.stock_codes)

            results.append({
                'rsi_buy': rsi_buy,
                'rsi_sell': rsi_sell,
                'avg_return': avg_return,
                'avg_win_rate': avg_win_rate,
                'avg_trades': total_trades / len(self.stock_codes)
            })

        return pd.DataFrame(results).sort_values('avg_return', ascending=False)

    def optimize_ma(self, short_periods, long_periods):
        """优化MA策略参数"""
        # Similar structure
        pass
```

- [ ] **Step 2: Run optimization for RSI (RSI < 25/30/35, RSI > 45/50/55)**

Run:

```bash
python -c "
from back_testing.portfolio_optimizer import PortfolioOptimizer
opt = PortfolioOptimizer(
    strategy_class=RSIReversalStrategy,
    stock_codes=['sh600519', 'sh600036', 'sh601318'],
    data_path=r'D:\workspace\code\mine\quant\data\metadata\daily_ycz'
)
results = opt.optimize_rsi(
    rsi_buy_levels=[25, 30, 35],
    rsi_sell_levels=[45, 50, 55]
)
print(results.head(10))
"
```

- [ ] **Step 3: Commit**

```bash
git add back_testing/portfolio_optimizer.py
git commit -m "feat: add portfolio optimizer for parameter tuning"
```

---

## Task 4: Create Portfolio Backtest (Multi-Stock with Equal Weight)

**Files:**

- Create: `back_testing/portfolio_backtest.py`

- [ ] **Step 1: Create PortfolioBacktest class**

```python
import pandas as pd
import numpy as np
from back_testing.backtest_engine import BacktestEngine


class PortfolioBacktest:
    """
    组合回测器
    每天选股，等权分配资金，回测整个组合表现
    """

    def __init__(self, strategy_class, stock_pool, data_path, initial_capital=1000000.0):
        self.strategy_class = strategy_class
        self.stock_pool = stock_pool  # List of (code, name, industry)
        self.data_path = data_path
        self.initial_capital = initial_capital
        self.per_stock_capital = initial_capital / len(stock_pool)

    def run(self):
        """运行组合回测"""
        results = []
        for code, name, industry in self.stock_pool:
            engine = self.strategy_class(
                stock_code=code,
                data_path=self.data_path,
                initial_capital=self.per_stock_capital
            )
            result = engine.run()
            results.append(result)

        # Aggregate results
        total_return = sum(r['final_capital'] for r in results) / self.initial_capital - 1
        total_trades = sum(r['total_trades'] for r in results)
        avg_win_rate = sum(r['win_rate'] for r in results) / len(results)

        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'avg_win_rate': avg_win_rate,
            'individual_results': results
        }

    def run_with_daily_selection(self):
        """
        每日选股模式
        每天根据信号选择满足条件的股票进行交易
        """
        # TODO: Implement daily selection logic
        pass
```

- [ ] **Step 2: Test portfolio backtest on all 8 stocks**

Run:

```bash
python -c "
from back_testing.portfolio_backtest import PortfolioBacktest
from back_testing.ma_strategy import MAStrategy

STOCK_POOL = [
    ('sh600519', '贵州茅台', '白酒'),
    ('sh600036', '招商银行', '银行'),
    # ... all 8 stocks
]
pb = PortfolioBacktest(MAStrategy, STOCK_POOL, r'D:\workspace\code\mine\quant\data\metadata\daily_ycz')
result = pb.run()
print(f'组合总收益: {result[\"total_return\"]:.2%}')
"
```

- [ ] **Step 3: Commit**

```bash
git add back_testing/portfolio_backtest.py
git commit -m "feat: add portfolio backtest for multi-stock testing"
```

---

## Task 5: Create Visualizer (Charts and Plots)

**Files:**

- Create: `back_testing/visualizer.py`

- [ ] **Step 1: Create Visualizer class**

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import DataFrame


class Visualizer:
    """
    回测结果可视化
    生成资金曲线、买卖点标记图
    """

    def __init__(self, result: dict, strategy_name: str):
        self.result = result
        self.strategy_name = strategy_name

    def plot_equity_curve(self, save_path=None):
        """画出资金曲线"""
        trades = self.result['trades']
        if not trades:
            return

        # Build equity timeline
        dates = []
        values = []
        capital = self.result['initial_capital']

        for trade in trades:
            dates.append(trade['date'])
            values.append(capital)
            if trade['action'] == 'BUY':
                capital = trade['capital_after']
            else:
                capital = trade['capital_after']

        plt.figure(figsize=(12, 6))
        plt.plot(dates, values)
        plt.title(f'{self.strategy_name} - {self.result["stock_code"]} 资金曲线')
        plt.xlabel('日期')
        plt.ylabel('资金')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_trades_with_prices(self, data: DataFrame, save_path=None):
        """画出价格图和买卖点"""
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot price
        ax.plot(data['交易日期'], data['后复权价'], label='价格', alpha=0.7)

        # Mark buy/sell points
        trades = self.result['trades']
        buys = [t for t in trades if t['action'] == 'BUY']
        sells = [t for t in trades if t['action'] == 'SELL']

        ax.scatter([t['date'] for t in buys], [t['price'] for t in buys],
                   marker='^', color='green', s=100, label='买入')
        ax.scatter([t['date'] for t in sells], [t['price'] for t in sells],
                   marker='v', color='red', s=100, label='卖出')

        ax.set_title(f'{self.strategy_name} - {self.result["stock_code"]} 交易记录')
        ax.set_xlabel('日期')
        ax.set_ylabel('价格')
        ax.legend()
        ax.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_strategy_comparison(self, results_dict: dict, save_path=None):
        """对比多个策略的资金曲线"""
        plt.figure(figsize=(12, 6))
        for name, result in results_dict.items():
            trades = result['trades']
            dates = [t['date'] for t in trades]
            values = []
            capital = result['initial_capital']
            for t in trades:
                values.append(capital)
                capital = t['capital_after']
            plt.plot(dates, values, label=name)

        plt.title('策略对比')
        plt.xlabel('日期')
        plt.ylabel('资金')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.close()
```

- [ ] **Step 2: Test visualizer on sh600519 MA strategy**

Run:

```bash
python -c "
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from back_testing.ma_strategy import MAStrategy
from back_testing.visualizer import Visualizer

engine = MAStrategy('sh600519', r'D:\workspace\code\mine\quant\data\metadata\daily_ycz', 100000)
result = engine.run()
viz = Visualizer(result, 'MA策略')
viz.plot_equity_curve('ma_equity.png')
viz.plot_trades_with_prices(engine.data, 'ma_trades.png')
print('Charts saved: ma_equity.png, ma_trades.png')
"
```

- [ ] **Step 3: Commit**

```bash
git add back_testing/visualizer.py
git commit -m "feat: add visualizer for charts and plots"
```

---

## Task 6: Create Unified Entry Point

**Files:**

- Create: `back_testing/run_full_backtest.py`

- [ ] **Step 1: Create unified main program**

```python
"""
完整回测系统入口
支持：单策略回测、组合回测、参数优化、可视化
"""
import argparse
import matplotlib

matplotlib.use('Agg')

from back_testing.ma_strategy import MAStrategy
from back_testing.macd_strategy import MACDStrategy
from back_testing.rsi_strategy import RSIReversalStrategy
from back_testing.combined_strategy import CombinedStrategy
from back_testing.portfolio_backtest import PortfolioBacktest
from back_testing.portfolio_optimizer import PortfolioOptimizer
from back_testing.visualizer import Visualizer

STOCK_POOL = [
    ('sh600519', '贵州茅台', '白酒'),
    ('sh600036', '招商银行', '银行'),
    ('sh601318', '中国平安', '保险'),
    ('sh688256', '寒武纪', '科创板'),
    ('sz300750', '宁德时代', '电池'),
    ('sz000001', '平安银行', '银行'),
    ('sh601899', '紫金矿业', '贵金属'),
    ('sz300059', '东方财富', '金融'),
]

DATA_PATH = r'D:\workspace\code\mine\quant\data\metadata\daily_ycz'
BENCHMARK = 'sh000001'


def run_single_stock(strategy_class, stock_code, stock_name):
    """运行单股票回测"""
    engine = strategy_class(stock_code, DATA_PATH, 100000, BENCHMARK)
    result = engine.run()
    engine.print_result(result, strategy_class.__name__)
    return result


def run_portfolio(strategy_class):
    """运行组合回测"""
    pb = PortfolioBacktest(strategy_class, STOCK_POOL, DATA_PATH)
    return pb.run()


def main():
    parser = argparse.ArgumentParser(description='量化策略回测系统')
    parser.add_argument('--mode', choices=['single', 'portfolio', 'optimize'], default='portfolio')
    parser.add_argument('--strategy', choices=['ma', 'macd', 'rsi', 'combined'], default='ma')
    args = parser.parse_args()

    strategies = {
        'ma': MAStrategy,
        'macd': MACDStrategy,
        'rsi': RSIReversalStrategy,
        'combined': CombinedStrategy
    }

    strategy_class = strategies[args.strategy]

    if args.mode == 'single':
        run_single_stock(strategy_class, 'sh600519', '贵州茅台')
    elif args.mode == 'portfolio':
        result = run_portfolio(strategy_class)
        print(f'组合总收益: {result["total_return"]:.2%}')
    elif args.mode == 'optimize':
        opt = PortfolioOptimizer(strategy_class, STOCK_POOL[:3], DATA_PATH)
        # Run optimization...


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Test the unified entry point**

Run:

```bash
python back_testing/run_full_backtest.py --mode single --strategy ma
python back_testing/run_full_backtest.py --mode portfolio --strategy ma
```

- [ ] **Step 3: Commit**

```bash
git add back_testing/run_full_backtest.py
git commit -m "feat: add unified entry point for full backtesting system"
```

---

## Task 7: Update Design Spec

**Files:**

- Modify: `docs/superpowers/specs/2026-04-18-quant-backtesting-design.md`

- [ ] **Update spec with completed features**

Add sections:

- Risk Management (stop loss / take profit)
- Combined Strategy
- Parameter Optimization
- Portfolio Backtesting
- Visualization

- [ ] **Commit spec update**

```bash
git add docs/superpowers/specs/2026-04-18-quant-backtesting-design.md
git commit -m "docs: update spec with risk management and optimization features"
```

---

## Summary

| Task | Description             | Files                  |
|------|-------------------------|------------------------|
| 1    | Stop Loss / Take Profit | backtest_engine.py     |
| 2    | Combined Strategy       | combined_strategy.py   |
| 3    | Portfolio Optimizer     | portfolio_optimizer.py |
| 4    | Portfolio Backtest      | portfolio_backtest.py  |
| 5    | Visualizer              | visualizer.py          |
| 6    | Unified Entry Point     | run_full_backtest.py   |
| 7    | Update Spec             | design spec            |

**Plan complete and saved to `docs/superpowers/plans/2026-04-19-quant-trading-system-enhancement.md`**

---

## Execution Options

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
