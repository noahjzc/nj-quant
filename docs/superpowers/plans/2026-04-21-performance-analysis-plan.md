# Performance Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add comprehensive performance analysis including risk-adjusted returns, relative returns, trading analysis, and visualization.

**Architecture:** Modular design with three new files: `IndexDataProvider` for benchmark data access, `PerformanceAnalyzer` for metrics calculation, and `PerformanceVisualizer` for charts. Integration into existing backtest output.

**Tech Stack:** Python, pandas, numpy, matplotlib

---

## Task 1: Create IndexDataProvider

**Files:**
- Create: `back_testing/index_data_provider.py`
- Test: `tests/back_testing/test_index_data_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/back_testing/test_index_data_provider.py
import pytest
import pandas as pd
from back_testing.index_data_provider import IndexDataProvider

def test_get_index_data():
    provider = IndexDataProvider(r'D:\workspace\code\mine\quant\data\metadata\daily_ycz\index')
    df = provider.get_index_data('sh000001', '2024-01-01', '2024-12-31')
    assert len(df) > 0
    assert 'close' in df.columns
    assert df['date'].iloc[0] >= pd.Timestamp('2024-01-01')

def test_get_index_return():
    provider = IndexDataProvider(r'D:\workspace\code\mine\quant\data\metadata\daily_ycz\index')
    ret = provider.get_index_return('sh000001', '2024-01-01', '2024-12-31')
    assert isinstance(ret, float)

def test_get_index_data_not_found():
    provider = IndexDataProvider(r'D:\workspace\code\mine\quant\data\metadata\daily_ycz\index')
    with pytest.raises(FileNotFoundError):
        provider.get_index_data('sh999999', '2024-01-01', '2024-12-31')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/back_testing/test_index_data_provider.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: Write minimal implementation**

```python
# back_testing/index_data_provider.py
import pandas as pd
from pathlib import Path
from typing import Optional

class IndexDataProvider:
    """指数数据提供器"""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: 指数数据目录
        """
        self.data_dir = Path(data_dir)

    def _get_file_path(self, index_code: str) -> Path:
        """获取指数数据文件路径"""
        path = self.data_dir / f'{index_code}.csv'
        if not path.exists():
            raise FileNotFoundError(f"指数数据文件不存在: {index_code}")
        return path

    def get_index_data(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数数据

        Args:
            index_code: 指数代码，如 'sh000001'
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame with columns: date, open, close, high, low, volume
        """
        file_path = self._get_file_path(index_code)
        df = pd.read_csv(file_path, encoding='gbk')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # 筛选日期范围
        df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        return df

    def get_index_return(self, index_code: str, start_date: str, end_date: str) -> float:
        """
        计算区间收益率

        Returns:
            float: 收益率（小数形式，如0.15表示15%）
        """
        df = self.get_index_data(index_code, start_date, end_date)
        if len(df) < 2:
            return 0.0
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        return (end_price - start_price) / start_price
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/back_testing/test_index_data_provider.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add back_testing/index_data_provider.py tests/back_testing/test_index_data_provider.py
git commit -m "feat: add IndexDataProvider for benchmark data access"
```

---

## Task 2: Create PerformanceAnalyzer

**Files:**
- Create: `back_testing/performance_analyzer.py`
- Test: `tests/back_testing/test_performance_analyzer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/back_testing/test_performance_analyzer.py
import pytest
import pandas as pd
from back_testing.performance_analyzer import PerformanceAnalyzer

def test_calculate_total_return():
    # 简单测试：净值从1涨到1.25，总收益25%
    trades = [{'action': 'sell', 'return': 0.25, 'shares': 100, 'price': 125}]
    analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
    metrics = analyzer.calculate_metrics()
    assert abs(metrics['total_return'] - 0.25) < 0.001

def test_calculate_win_rate():
    # 3笔交易，2笔赚钱
    trades = [
        {'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110},
        {'action': 'sell', 'return': -0.05, 'shares': 100, 'price': 95},
        {'action': 'sell', 'return': 0.15, 'shares': 100, 'price': 115},
    ]
    analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
    metrics = analyzer.calculate_metrics()
    assert metrics['win_rate'] == pytest.approx(2/3)

def test_sharpe_ratio():
    # 只有一笔交易，无法计算有意义的sharpe，波动率为0
    trades = [{'action': 'sell', 'return': 0.20, 'shares': 100, 'price': 120}]
    analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
    metrics = analyzer.calculate_metrics()
    assert metrics['sharpe_ratio'] == 0.0  # 无法计算时返回0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/back_testing/test_performance_analyzer.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: Write minimal implementation**

```python
# back_testing/performance_analyzer.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class PerformanceAnalyzer:
    """绩效分析器"""

    def __init__(
        self,
        trades: List[Dict],
        initial_capital: float = 1000000.0,
        benchmark_index: str = 'sh000001',
        risk_free_rate: float = 0.025
    ):
        """
        Args:
            trades: 交易记录列表，每条记录包含:
                - action: 'buy' | 'sell'
                - price: 成交价格
                - shares: 成交数量
                - return: 收益率（卖出时）
                - date: 交易日期（可选）
            initial_capital: 期初资金
            benchmark_index: 基准指数代码
            risk_free_rate: 无风险利率
        """
        self.trades = trades
        self.initial_capital = initial_capital
        self.benchmark_index = benchmark_index
        self.risk_free_rate = risk_free_rate

        # 解析交易
        self.sell_trades = [t for t in trades if t.get('action') == 'sell']
        self.buy_trades = [t for t in trades if t.get('action') == 'buy']

    def calculate_metrics(self) -> Dict:
        """计算所有绩效指标"""
        total_return = self._calculate_total_return()
        annual_return = self._calculate_annual_return()
        max_drawdown = self._calculate_max_drawdown()
        sharpe = self._calculate_sharpe_ratio(annual_return)
        calmar = self._calculate_calmar_ratio(annual_return, max_drawdown)
        sortino = self._calculate_sortino_ratio(annual_return)
        win_rate = self._calculate_win_rate()
        profit_loss_ratio = self._calculate_profit_loss_ratio()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'sortino_ratio': sortino,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
        }

    def _calculate_total_return(self) -> float:
        """计算总收益率"""
        if not self.sell_trades:
            return 0.0
        total_profit = sum(t['return'] * t['shares'] * t['price'] for t in self.sell_trades)
        return total_profit / self.initial_capital

    def _calculate_annual_return(self) -> float:
        """计算年化收益率"""
        total_return = self._calculate_total_return()
        # 简单按平均持仓估算天数
        n_trades = len(self.sell_trades)
        if n_trades == 0:
            return 0.0
        # 估算：假设每笔交易平均持仓天数
        avg_days_per_trade = 10  # 简化估算
        total_days = n_trades * avg_days_per_trade
        if total_days == 0:
            return 0.0
        years = total_days / 365
        if years < 0.01:
            return 0.0
        return (1 + total_return) ** (1 / years) - 1

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤（简化版）"""
        if not self.sell_trades:
            return 0.0
        # 使用交易收益率序列计算
        returns = [t['return'] for t in self.sell_trades]
        cumulative = np.cumprod(1 + np.array(returns))
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    def _calculate_sharpe_ratio(self, annual_return: float) -> float:
        """计算Sharpe比率"""
        if not self.sell_trades or len(self.sell_trades) < 2:
            return 0.0
        returns = [t['return'] for t in self.sell_trades]
        std = np.std(returns)
        if std == 0:
            return 0.0
        return (annual_return - self.risk_free_rate) / std

    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """计算Calmar比率"""
        if max_drawdown == 0:
            return 0.0
        return annual_return / max_drawdown

    def _calculate_sortino_ratio(self, annual_return: float) -> float:
        """计算Sortino比率（简化版）"""
        if not self.sell_trades or len(self.sell_trades) < 2:
            return 0.0
        returns = [t['return'] for t in self.sell_trades]
        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return 0.0
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        return (annual_return - 0) / downside_std

    def _calculate_win_rate(self) -> float:
        """计算胜率"""
        if not self.sell_trades:
            return 0.0
        wins = sum(1 for t in self.sell_trades if t['return'] > 0)
        return wins / len(self.sell_trades)

    def _calculate_profit_loss_ratio(self) -> float:
        """计算盈亏比"""
        if not self.sell_trades:
            return 0.0
        profits = [t['return'] for t in self.sell_trades if t['return'] > 0]
        losses = [abs(t['return']) for t in self.sell_trades if t['return'] < 0]
        if not profits or not losses:
            return 0.0
        avg_profit = np.mean(profits)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 0.0
        return avg_profit / avg_loss
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/back_testing/test_performance_analyzer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add back_testing/performance_analyzer.py tests/back_testing/test_performance_analyzer.py
git commit -m "feat: add PerformanceAnalyzer for metrics calculation"
```

---

## Task 3: Create PerformanceVisualizer

**Files:**
- Create: `back_testing/visualizer.py`
- Test: `tests/back_testing/test_visualizer.py` (optional, can be manual)

- [ ] **Step 1: Write the implementation**

```python
# back_testing/visualizer.py
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional

class PerformanceVisualizer:
    """绩效可视化器"""

    def __init__(self, equity_curve: pd.DataFrame, benchmark_curve: pd.DataFrame = None):
        """
        Args:
            equity_curve: 资金曲线 DataFrame，index为日期，columns为净值
            benchmark_curve: 基准曲线 DataFrame（可选）
        """
        self.equity_curve = equity_curve
        self.benchmark_curve = benchmark_curve

    def plot_equity_curve(self, save_path: Optional[str] = None):
        """资金曲线图"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, self.equity_curve.values, label='组合净值')

        if self.benchmark_curve is not None:
            # 对齐基准曲线到组合曲线日期
            aligned_benchmark = self.benchmark_curve.reindex(self.equity_curve.index, method='ffill')
            plt.plot(aligned_benchmark.index, aligned_benchmark.values, label='沪深300', alpha=0.7)

        plt.title('资金曲线')
        plt.xlabel('日期')
        plt.ylabel('净值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_drawdown(self, save_path: Optional[str] = None):
        """回撤曲线图"""
        equity = self.equity_curve.values
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100  # 转为百分比

        plt.figure(figsize=(12, 4))
        plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        plt.plot(drawdown, color='red')
        plt.title('回撤曲线')
        plt.xlabel('日期')
        plt.ylabel('回撤 (%)')
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_return_distribution(self, trades: list, save_path: Optional[str] = None):
        """收益分布直方图"""
        returns = [t['return'] * 100 for t in trades if t.get('action') == 'sell']

        plt.figure(figsize=(10, 6))
        plt.hist(returns, bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.axvline(x=np.mean(returns), color='green', linestyle='--', linewidth=2, label=f'均值: {np.mean(returns):.1f}%')
        plt.title('收益分布')
        plt.xlabel('收益率 (%)')
        plt.ylabel('频次')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def generate_report(self, trades: list, save_dir: str = None) -> str:
        """生成完整分析报告（HTML格式）"""
        if save_dir is None:
            save_dir = 'back_testing/results'

        import os
        os.makedirs(save_dir, exist_ok=True)

        # 生成各图表
        equity_path = os.path.join(save_dir, 'equity_curve.png')
        drawdown_path = os.path.join(save_dir, 'drawdown.png')
        distribution_path = os.path.join(save_dir, 'return_distribution.png')

        self.plot_equity_curve(equity_path)
        self.plot_drawdown(drawdown_path)
        self.plot_return_distribution(trades, distribution_path)

        # 生成HTML
        html_path = os.path.join(save_dir, 'performance_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>绩效分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
        .chart {{ margin: 20px 0; }}
        .chart img {{ max-width: 100%; }}
        .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .metric {{ background: #f5f5f5; padding: 15px; border-radius: 5px; min-width: 150px; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #333; }}
    </style>
</head>
<body>
    <h1>绩效分析报告</h1>

    <h2>资金曲线</h2>
    <div class="chart"><img src="equity_curve.png" alt="资金曲线"></div>

    <h2>回撤曲线</h2>
    <div class="chart"><img src="drawdown.png" alt="回撤曲线"></div>

    <h2>收益分布</h2>
    <div class="chart"><img src="return_distribution.png" alt="收益分布"></div>
</body>
</html>
            ''')

        return html_path
```

- [ ] **Step 2: Run simple verification**

```bash
python -c "
from back_testing.visualizer import PerformanceVisualizer
import pandas as pd
import numpy as np

# 测试数据
dates = pd.date_range('2024-01-01', periods=100, freq='D')
equity = pd.Series(1 + np.cumsum(np.random.randn(100) * 0.01), index=dates)

viz = PerformanceVisualizer(equity)
viz.plot_equity_curve('test_equity.png')
viz.plot_drawdown('test_drawdown.png')
print('Visualizer tests passed')
"
```

- [ ] **Step 3: Commit**

```bash
git add back_testing/visualizer.py
git commit -m "feat: add PerformanceVisualizer for charts and HTML report"
```

---

## Task 4: Integrate into run_composite_backtest

**Files:**
- Modify: `back_testing/run_composite_backtest.py`

- [ ] **Step 1: Add imports and initialization**

在文件顶部添加：
```python
from back_testing.performance_analyzer import PerformanceAnalyzer
from back_testing.visualizer import PerformanceVisualizer
```

在 `run_backtest()` 函数开头添加：
```python
from back_testing.index_data_provider import IndexDataProvider

# 指数数据目录
INDEX_DATA_DIR = r'D:\workspace\code\mine\quant\data\metadata\daily_ycz\index'
index_provider = IndexDataProvider(INDEX_DATA_DIR)
```

- [ ] **Step 2: Calculate benchmark return for the period**

在回测开始时获取基准收益：
```python
benchmark_return = index_provider.get_index_return('sh000001', start_date, end_date)
```

- [ ] **Step 3: Add performance analysis output**

在现有的交易统计后添加：
```python
# 集成绩效分析
all_trades = exit_records + rotation_sell_records
analyzer = PerformanceAnalyzer(
    trades=all_trades,
    initial_capital=initial_capital,
    benchmark_index='sh000001'
)
metrics = analyzer.calculate_metrics()

print("\n" + "=" * 50)
print("绩效分析")
print("=" * 50)
print(f"绝对收益:")
print(f"  总收益率: {metrics['total_return']:+.2%}")
print(f"  年化收益率: {metrics['annual_return']:+.2%}")
print(f"  最大回撤: {metrics['max_drawdown']:+.2%}")
print(f"\n风险调整收益:")
print(f"  Sharpe比率: {metrics['sharpe_ratio']:.2f}")
print(f"  Calmar比率: {metrics['calmar_ratio']:.2f}")
print(f"  Sortino比率: {metrics['sortino_ratio']:.2f}")
print(f"\n相对收益 (vs 沪深300):")
print(f"  基准收益: {benchmark_return:+.2%}")
print(f"  超额收益: {metrics['total_return'] - benchmark_return:+.2%}")
print(f"\n交易分析:")
print(f"  胜率: {metrics['win_rate']:.1%}")
print(f"  盈亏比: {metrics['profit_loss_ratio']:.2f}")
print("=" * 50)

# 生成可视化报告
try:
    # 构建资金曲线
    df_weeks['portfolio_value_normalized'] = df_weeks['portfolio_value'] / df_weeks['portfolio_value'].iloc[0]
    equity_curve = df_weeks.set_index('date')['portfolio_value_normalized']

    # 获取基准曲线
    benchmark_data = index_provider.get_index_data('sh000001', start_date, end_date)
    benchmark_data['normalized'] = benchmark_data['close'] / benchmark_data['close'].iloc[0]
    benchmark_curve = benchmark_data.set_index('date')['normalized']

    visualizer = PerformanceVisualizer(equity_curve, benchmark_curve)
    report_path = visualizer.generate_report(all_trades, save_dir='back_testing/results')
    print(f"\n绩效报告已生成: {report_path}")
except Exception as e:
    print(f"\n生成报告失败: {e}")
```

- [ ] **Step 4: Test the integration**

```bash
python -c "
from back_testing.run_composite_backtest import run_backtest
# 运行短期回测验证
run_backtest('2024-01-01', '2024-02-01', 1000000)
"
```

- [ ] **Step 5: Commit**

```bash
git add back_testing/run_composite_backtest.py
git commit -m "feat: integrate performance analysis into backtest output"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | IndexDataProvider | index_data_provider.py |
| 2 | PerformanceAnalyzer | performance_analyzer.py |
| 3 | PerformanceVisualizer | visualizer.py |
| 4 | Integration | run_composite_backtest.py |

**Plan complete and saved to `docs/superpowers/plans/2026-04-21-performance-analysis-plan.md`**
