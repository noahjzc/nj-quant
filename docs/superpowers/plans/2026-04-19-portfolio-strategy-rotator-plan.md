# 策略轮动量化选股系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现策略轮动量化选股系统：每周评估13种策略表现，选出最优策略，从5711只股票中筛选5只等权持仓

**Architecture:** 系统由5个模块组成：策略评估器、信号评分器、选股器、组合构建器、主控制器。各模块职责单一，通过数据流串联。每周五执行策略评估和选股，下周一开盘调仓。

**Tech Stack:** Python, pandas, numpy, back_testing框架

---

## 文件结构

```
back_testing/
├── strategy_evaluator.py      # 新增：策略评估器（4周评分）
├── signal_scorer.py           # 新增：信号评分器（各策略信号强度量化）
├── stock_selector.py          # 新增：选股器（5711股票排序筛选）
├── portfolio_rotator.py       # 新增：策略轮动主控制器
├── run_rotator_backtest.py   # 新增：回测入口
├── backtest_engine.py         # 已有：通用回测引擎
├── portfolio_backtest.py      # 已有：组合回测器
└── [13个策略文件]             # 已有：各策略实现
```

---

## Task 1: 信号评分器 (SignalScorer)

**Files:**
- Create: `back_testing/signal_scorer.py`
- Test: `tests/back_testing/test_signal_scorer.py`

- [ ] **Step 1: 编写测试用例**

```python
# tests/back_testing/test_signal_scorer.py
import pytest
from back_testing.signal_scorer import SignalScorer
import pandas as pd
import numpy as np

def test_rsi_signal_strength():
    """RSI策略：RSI值越低信号越强"""
    scorer = SignalScorer()
    # 模拟RSI数据
    df = pd.DataFrame({'rsi1': [20, 30, 50, 70, 80]})
    scores = scorer.calculate_rsi_strength(df)
    # RSI=20最强（最超卖），RSI=80最弱
    assert scores.iloc[0] > scores.iloc[1]  # 20 > 30
    assert scores.iloc[-1] < scores.iloc[0]  # 80 < 20

def test_macd_signal_strength():
    """MACD策略：DIF与DEA差值越大信号越强"""
    scorer = SignalScorer()
    df = pd.DataFrame({
        'MACD_DIF': [1.0, 2.0, 0.5, -1.0],
        'MACD_DEA': [0.5, 1.5, 0.5, -0.5]
    })
    scores = scorer.calculate_macd_strength(df)
    assert scores.iloc[1] > scores.iloc[0]  # 差值0.5 > 0.5 (DIF更大)
    assert scores.iloc[-1] < 0  # 负值表示死叉

def test_kdj_signal_strength():
    """KDJ策略：J值越低（超卖）信号越强"""
    scorer = SignalScorer()
    df = pd.DataFrame({'KDJ_J': [5, 20, 50, 80, 100]})
    scores = scorer.calculate_kdj_strength(df)
    assert scores.iloc[0] > scores.iloc[1]  # J=5最强
    assert scores.iloc[-1] < scores.iloc[0]  # J=100最弱
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/back_testing/test_signal_scorer.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: 实现信号评分器**

```python
# back_testing/signal_scorer.py
import pandas as pd
import numpy as np

class SignalScorer:
    """信号评分器：计算各策略的信号强度"""

    def calculate_rsi_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        RSI信号强度：RSI值越低（越超卖），强度越高
        强度 = 100 - RSI值（归一化到0-100）
        """
        rsi = df['rsi1']
        # RSI越低强度越高：使用 (100 - RSI) 作为强度
        strength = 100 - rsi
        # 限制范围
        strength = strength.clip(0, 100)
        return strength

    def calculate_macd_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        MACD信号强度：DIF与DEA差值越大，强度越高
        强度 = DIF - DEA（差值越大金叉越强）
        """
        dif = df['MACD_DIF']
        dea = df['MACD_DEA']
        diff = dif - dea
        # 归一化到大致0-100范围
        strength = (diff / abs(diff).mean() * 50 + 50).clip(0, 100)
        return strength

    def calculate_ma_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        MA信号强度：MA5与MA20开口角度越大，趋势越强
        强度 = (MA5 - MA20) / MA20 * 100
        """
        ma5 = df['MA_5']
        ma20 = df['MA_20']
        diff = ma5 - ma20
        strength = (diff / ma20 * 100).clip(-100, 100)
        return strength

    def calculate_kdj_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        KDJ信号强度：J值越低（超卖），强度越高
        强度 = 100 - J值
        """
        j = df['KDJ_J']
        strength = 100 - j
        strength = strength.clip(0, 100)
        return strength

    def calculate_bollinger_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        布林带信号强度：价格偏离中轨越远，强度越高
        强度 = |价格 - 中轨| / 中轨 * 100
        """
        price = df['后复权价']
        middle = df['布林线中轨']
        strength = abs(price - middle) / middle * 100
        return strength

    def calculate_volume_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        成交量信号强度：量比越大，强度越高
        """
        vol_ratio = df['量比']
        strength = (vol_ratio * 20).clip(0, 100)  # 量比2.0对应强度40
        return strength

    def calculate_combined_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        组合策略信号强度：MA + RSI 组合
        """
        ma_strength = self.calculate_ma_strength(df)
        rsi_strength = self.calculate_rsi_strength(df)
        # RSI < 40 时才考虑买入
        rsi_confirm = (df['rsi1'] < 40).astype(int) * 50
        strength = (ma_strength * 0.6 + rsi_confirm * 0.4)
        return strength.clip(0, 100)

    def get_signal_strength(self, strategy_name: str, df: pd.DataFrame) -> pd.Series:
        """
        根据策略名称获取信号强度
        """
        strength_map = {
            'MAStrategy': self.calculate_ma_strength,
            'MACDStrategy': self.calculate_macd_strength,
            'RSIReversalStrategy': self.calculate_rsi_strength,
            'CombinedStrategy': self.calculate_combined_strength,
            'BollingerStrategy': self.calculate_bollinger_strength,
            'BollingerStrictStrategy': self.calculate_bollinger_strength,
            'KDJOversoldStrategy': self.calculate_kdj_strength,
            'KDJGoldenCrossStrategy': self.calculate_kdj_strength,
            'MultiPeriodRSIStrategy': self.calculate_rsi_strength,
            'RSIReversalMultiStrategy': self.calculate_rsi_strength,
            'TrendConfirmationStrategy': self.calculate_macd_strength,
            'TrendPullbackStrategy': self.calculate_ma_strength,
            'VolumeAnomalyStrategy': self.calculate_volume_strength,
            'VolumeMAConfirmStrategy': self.calculate_volume_strength,
        }

        if strategy_name in strength_map:
            return strength_map[strategy_name](df)
        else:
            return pd.Series(50, index=df.index)  # 默认中性强度
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/back_testing/test_signal_scorer.py -v`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
git add back_testing/signal_scorer.py tests/back_testing/test_signal_scorer.py
git commit -m "feat: add signal scorer for strategy strength calculation"
```

---

## Task 2: 策略评估器 (StrategyEvaluator)

**Files:**
- Create: `back_testing/strategy_evaluator.py`
- Test: `tests/back_testing/test_strategy_evaluator.py`

- [ ] **Step 1: 编写测试用例**

```python
# tests/back_testing/test_strategy_evaluator.py
import pytest
from back_testing.strategy_evaluator import StrategyEvaluator
import pandas as pd
import numpy as np

def test_evaluate_single_strategy():
    """测试单个策略的4周表现评估"""
    evaluator = StrategyEvaluator(
        stock_codes=['sh600519', 'sz000001'],
        data_path='D:/workspace/code/mine/quant/data/metadata/daily_ycz'
    )
    scores = evaluator.evaluate_strategy('RSIReversalStrategy', weeks=4)
    assert isinstance(scores, dict)
    assert 'avg_return' in scores
    assert 'stock_returns' in scores
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/back_testing/test_strategy_evaluator.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: 实现策略评估器**

```python
# back_testing/strategy_evaluator.py
import pandas as pd
import numpy as np
from datetime import timedelta
from back_testing.backtest_engine import BacktestEngine
from back_testing.ma_strategy import MAStrategy
from back_testing.macd_strategy import MACDStrategy
from back_testing.rsi_strategy import RSIReversalStrategy
from back_testing.combined_strategy import CombinedStrategy
from back_testing.bollinger_strategy import BollingerStrategy, BollingerStrictStrategy
from back_testing.kdj_strategy import KDJOversoldStrategy, KDJGoldenCrossStrategy
from back_testing.multi_rsi_strategy import MultiPeriodRSIStrategy, RSIReversalMultiStrategy
from back_testing.trend_confirmation_strategy import TrendConfirmationStrategy, TrendPullbackStrategy
from back_testing.volume_strategy import VolumeAnomalyStrategy, VolumeMAConfirmStrategy

STRATEGY_MAP = {
    'MAStrategy': MAStrategy,
    'MACDStrategy': MACDStrategy,
    'RSIReversalStrategy': RSIReversalStrategy,
    'CombinedStrategy': CombinedStrategy,
    'BollingerStrategy': BollingerStrategy,
    'BollingerStrictStrategy': BollingerStrictStrategy,
    'KDJOversoldStrategy': KDJOversoldStrategy,
    'KDJGoldenCrossStrategy': KDJGoldenCrossStrategy,
    'MultiPeriodRSIStrategy': MultiPeriodRSIStrategy,
    'RSIReversalMultiStrategy': RSIReversalMultiStrategy,
    'TrendConfirmationStrategy': TrendConfirmationStrategy,
    'TrendPullbackStrategy': TrendPullbackStrategy,
    'VolumeAnomalyStrategy': VolumeAnomalyStrategy,
    'VolumeMAConfirmStrategy': VolumeMAConfirmStrategy,
}


class StrategyEvaluator:
    """
    策略评估器：评估各策略过去N周的表现
    """

    def __init__(self, stock_codes: list, data_path: str, initial_capital: float = 100000.0):
        self.stock_codes = stock_codes
        self.data_path = data_path
        self.initial_capital = initial_capital

    def get_recent_trading_dates(self, reference_date: pd.Timestamp, weeks: int = 4) -> tuple:
        """获取最近N周的交易日范围"""
        # 向前推N周
        start_date = reference_date - timedelta(weeks=weeks * 7)
        return start_date, reference_date

    def evaluate_strategy(self, strategy_name: str, weeks: int = 4,
                         reference_date: pd.Timestamp = None) -> dict:
        """
        评估单个策略在过去N周的表现

        Returns:
            {
                'avg_return': float,  # 平均收益率
                'stock_returns': list,  # 各股票收益率
                'num_trades': int,     # 总交易次数
            }
        """
        if reference_date is None:
            reference_date = pd.Timestamp.now()

        start_date, end_date = self.get_recent_trading_dates(reference_date, weeks)

        strategy_class = STRATEGY_MAP.get(strategy_name)
        if strategy_class is None:
            return {'avg_return': 0, 'stock_returns': [], 'num_trades': 0}

        stock_returns = []
        total_trades = 0

        for code in self.stock_codes:
            try:
                engine = strategy_class(
                    stock_code=code,
                    data_path=self.data_path,
                    initial_capital=self.initial_capital,
                    start_date=start_date.strftime('%Y-%m-%d')
                )
                result = engine.run()
                stock_returns.append(result['total_return'])
                total_trades += result['total_trades']
            except Exception as e:
                # 单只股票失败不影响整体
                continue

        avg_return = np.mean(stock_returns) if stock_returns else 0

        return {
            'avg_return': avg_return,
            'stock_returns': stock_returns,
            'num_trades': total_trades,
            'start_date': start_date,
            'end_date': end_date
        }

    def evaluate_all_strategies(self, weeks: int = 4,
                                 reference_date: pd.Timestamp = None) -> pd.DataFrame:
        """
        评估所有策略的表现，返回排名

        Returns:
            DataFrame with columns: strategy_name, avg_return, num_trades
        """
        results = []
        for strategy_name in STRATEGY_MAP.keys():
            eval_result = self.evaluate_strategy(strategy_name, weeks, reference_date)
            results.append({
                'strategy_name': strategy_name,
                'avg_return': eval_result['avg_return'],
                'num_trades': eval_result['num_trades'],
                'start_date': eval_result['start_date'],
                'end_date': eval_result['end_date']
            })

        df = pd.DataFrame(results)
        df = df.sort_values('avg_return', ascending=False)
        return df
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/back_testing/test_strategy_evaluator.py -v`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
git add back_testing/strategy_evaluator.py tests/back_testing/test_strategy_evaluator.py
git commit -m "feat: add strategy evaluator for 4-week performance scoring"
```

---

## Task 3: 选股器 (StockSelector)

**Files:**
- Create: `back_testing/stock_selector.py`
- Test: `tests/back_testing/test_stock_selector.py`

- [ ] **Step 1: 编写测试用例**

```python
# tests/back_testing/test_stock_selector.py
import pytest
from back_testing.stock_selector import StockSelector
import pandas as pd

def test_select_top_stocks():
    """测试选取信号最强的股票"""
    selector = StockSelector(
        data_path='D:/workspace/code/mine/quant/data/metadata/daily_ycz'
    )
    # 用贵州茅台测试
    selected = selector.select_top_stocks(
        strategy_name='RSIReversalStrategy',
        n=3,
        date='2024-01-15'
    )
    assert len(selected) <= 3
    assert all(isinstance(code, str) for code in selected)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/back_testing/test_stock_selector.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: 实现选股器**

```python
# back_testing/stock_selector.py
import pandas as pd
import numpy as np
import os
from back_testing.signal_scorer import SignalScorer

class StockSelector:
    """
    选股器：根据策略信号强度从全市场筛选股票
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scorer = SignalScorer()

    def get_all_stock_codes(self) -> list:
        """获取所有股票代码"""
        files = os.listdir(self.data_path)
        # 过滤出csv文件，去掉index目录
        stock_codes = [f.replace('.csv', '') for f in files
                      if f.endswith('.csv') and not f.startswith('index')]
        return stock_codes

    def calculate_stock_signal(self, stock_code: str, strategy_name: str,
                                date: pd.Timestamp) -> float:
        """
        计算单只股票在特定日期的信号强度

        Returns:
            float: 信号强度分数
        """
        file_path = f"{self.data_path}\\{stock_code}.csv"
        try:
            df = pd.read_csv(file_path, encoding='gbk')
            df['交易日期'] = pd.to_datetime(df['交易日期'])
            df = df.sort_values('交易日期')

            # 筛选到指定日期的数据
            df = df[df['交易日期'] <= date]
            if len(df) == 0:
                return 0

            # 取最后一行计算信号强度
            latest = df.iloc[-1:]
            score = self.scorer.get_signal_strength(strategy_name, latest).iloc[0]
            return score
        except Exception:
            return 0

    def select_top_stocks(self, strategy_name: str, n: int = 5,
                          date: str = None) -> list:
        """
        选取信号最强的N只股票

        Args:
            strategy_name: 策略名称
            n: 选取数量
            date: 评分日期

        Returns:
            list of stock codes
        """
        if date is None:
            date = pd.Timestamp.now()
        else:
            date = pd.to_datetime(date)

        all_codes = self.get_all_stock_codes()
        print(f"正在评估 {len(all_codes)} 只股票...")

        scores = []
        for code in all_codes:
            score = self.calculate_stock_signal(code, strategy_name, date)
            scores.append((code, score))

        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 返回前N只
        selected = [code for code, score in scores[:n]]
        print(f"选取结果: {selected}")
        return selected
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/back_testing/test_stock_selector.py -v`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
git add back_testing/stock_selector.py tests/back_testing/test_stock_selector.py
git commit -m "feat: add stock selector for signal-based screening"
```

---

## Task 4: 策略轮动主控制器 (PortfolioRotator)

**Files:**
- Create: `back_testing/portfolio_rotator.py`

- [ ] **Step 1: 实现主控制器**

```python
# back_testing/portfolio_rotator.py
import pandas as pd
import numpy as np
from back_testing.strategy_evaluator import StrategyEvaluator, STRATEGY_MAP
from back_testing.stock_selector import StockSelector
from back_testing.signal_scorer import SignalScorer

class PortfolioRotator:
    """
    策略轮动主控制器

    每周流程：
    1. 评估所有策略过去4周表现
    2. 选出最优策略
    3. 对全市场股票计算该策略信号强度
    4. 选取信号最强的5只股票
    5. 等权20%持仓
    """

    def __init__(self, data_path: str, initial_capital: float = 1000000.0,
                 n_stocks: int = 5, n_weeks: int = 4):
        self.data_path = data_path
        self.initial_capital = initial_capital
        self.n_stocks = n_stocks  # 持仓数量
        self.n_weeks = n_weeks    # 评估周期
        self.per_stock_capital = initial_capital / n_stocks

        self.strategy_evaluator = None
        self.stock_selector = StockSelector(data_path)
        self.scorer = SignalScorer()

        self.current_strategy = None
        self.current_stocks = []
        self.current_positions = {}  # {stock_code: {'shares': int, 'buy_price': float}}

        self.trade_log = []  # 交易记录
        self.portfolio_value_history = []  # 净值历史

    def select_best_strategy(self, reference_date: pd.Timestamp = None) -> str:
        """
        选出过去N周表现最好的策略
        """
        if reference_date is None:
            reference_date = pd.Timestamp.now()

        # 获取所有股票代码（取样本）
        sample_stocks = self.stock_selector.get_all_stock_codes()[:100]  # 用样本评估

        self.strategy_evaluator = StrategyEvaluator(
            stock_codes=sample_stocks,
            data_path=self.data_path,
            initial_capital=self.per_stock_capital
        )

        print(f"评估 {len(sample_stocks)} 只股票在过去{self.n_weeks}周的表现...")
        results = self.strategy_evaluator.evaluate_all_strategies(
            weeks=self.n_weeks,
            reference_date=reference_date
        )

        best_strategy = results.iloc[0]['strategy_name']
        best_return = results.iloc[0]['avg_return']

        print(f"\n策略排名（过去{self.n_weeks}周）：")
        print(results.to_string(index=False))
        print(f"\n选中策略: {best_strategy} (收益率: {best_return:.2%})")

        self.current_strategy = best_strategy
        return best_strategy

    def select_stocks(self, date: pd.Timestamp) -> list:
        """
        选取信号最强的N只股票
        """
        print(f"\n使用策略 {self.current_strategy} 筛选股票...")

        selected = self.stock_selector.select_top_stocks(
            strategy_name=self.current_strategy,
            n=self.n_stocks,
            date=date
        )

        self.current_stocks = selected
        return selected

    def rebalance(self, date: pd.Timestamp) -> dict:
        """
        执行调仓：卖出旧持仓，买入新持仓

        Returns:
            dict: 调仓详情
        """
        rebalance_detail = {
            'date': date,
            'sell_stocks': [],  # 卖出的股票
            'buy_stocks': [],    # 买入的股票
            'strategy': self.current_strategy
        }

        # 卖出不在新持仓列表中的股票
        for code in list(self.current_positions.keys()):
            if code not in self.current_stocks:
                position = self.current_positions[code]
                # 模拟卖出（使用当天收盘价）
                # 实际回测中需要加载当天数据
                rebalance_detail['sell_stocks'].append(code)
                del self.current_positions[code]

        # 买入新持仓股票
        for code in self.current_stocks:
            if code not in self.current_positions:
                rebalance_detail['buy_stocks'].append(code)
                # 初始化持仓
                self.current_positions[code] = {
                    'shares': 0,
                    'buy_price': 0
                }

        return rebalance_detail

    def calculate_portfolio_value(self, prices: dict) -> float:
        """
        计算当前组合市值
        """
        total = 0
        for code, position in self.current_positions.items():
            if position['shares'] > 0 and code in prices:
                total += position['shares'] * prices[code]
        return total

    def run_weekly(self, date: pd.Timestamp) -> dict:
        """
        执行每周流程
        """
        # 1. 选择最优策略
        self.select_best_strategy(date)

        # 2. 筛选股票
        self.select_stocks(date)

        # 3. 执行调仓
        rebalance = self.rebalance(date)

        return {
            'date': date,
            'strategy': self.current_strategy,
            'stocks': self.current_stocks,
            'rebalance': rebalance
        }
```

- [ ] **Step 2: 提交代码**

```bash
git add back_testing/portfolio_rotator.py
git commit -m "feat: add portfolio rotator main controller"
```

---

## Task 5: 回测运行器 (RotatorBacktest)

**Files:**
- Create: `back_testing/run_rotator_backtest.py`

- [ ] **Step 1: 实现回测运行器**

```python
# back_testing/run_rotator_backtest.py
"""
策略轮动量化选股系统 - 回测入口
"""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from back_testing.portfolio_rotator import PortfolioRotator

DATA_PATH = r'D:\workspace\code\mine\quant\data\metadata\daily_ycz'
INITIAL_CAPITAL = 1000000.0  # 100万
N_STOCKS = 5
N_WEEKS = 4


def get_trading_fridays(start_date: str, end_date: str) -> list:
    """
    获取回测区间内所有周五（调仓日）
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    fridays = []
    current = start
    # 找到第一个周五
    while current.weekday() != 4:  # Monday=0, Friday=4
        current += timedelta(days=1)

    while current <= end:
        fridays.append(current)
        current += timedelta(days=7)

    return fridays


def load_stock_price(stock_code: str, date: pd.Timestamp, data_path: str) -> float:
    """加载指定日期的收盘价"""
    file_path = f"{data_path}\\{stock_code}.csv"
    try:
        df = pd.read_csv(file_path, encoding='gbk')
        df['交易日期'] = pd.to_datetime(df['交易日期'])
        df = df.sort_values('交易日期')

        # 找当天或之前的最近交易日
        hist = df[df['交易日期'] <= date]
        if len(hist) == 0:
            return None
        return hist.iloc[-1]['后复权价']
    except Exception:
        return None


def run_backtest(start_date: str, end_date: str, initial_capital: float = INITIAL_CAPITAL):
    """
    运行回测
    """
    print("=" * 60)
    print("策略轮动量化选股系统 - 回测")
    print("=" * 60)
    print(f"回测区间: {start_date} ~ {end_date}")
    print(f"初始资金: {initial_capital:,.2f}")
    print(f"持仓数量: {N_STOCKS}")
    print(f"调仓周期: 每周")
    print("=" * 60)

    rotator = PortfolioRotator(
        data_path=DATA_PATH,
        initial_capital=initial_capital,
        n_stocks=N_STOCKS,
        n_weeks=N_WEEKS
    )

    # 获取所有调仓日
    fridays = get_trading_fridays(start_date, end_date)
    print(f"\n调仓日数量: {len(fridays)}")

    current_capital = initial_capital
    portfolio_value = initial_capital
    portfolio_history = []

    weekly_results = []

    for i, friday in enumerate(fridays):
        print(f"\n{'='*60}")
        print(f"第 {i+1}/{len(fridays)} 周: {friday.strftime('%Y-%m-%d')} (周五)")
        print("=" * 60)

        # 执行每周流程
        result = rotator.run_weekly(friday)

        # 计算持仓价值
        stock_values = {}
        for code in rotator.current_stocks:
            price = load_stock_price(code, friday, DATA_PATH)
            if price:
                shares = int(portfolio_value / N_STOCKS / price)
                stock_values[code] = shares * price
            else:
                stock_values[code] = 0

        # 估算组合净值
        portfolio_value = sum(stock_values.values()) + current_capital * 0.1  # 预留现金

        weekly_results.append({
            'date': friday,
            'strategy': result['strategy'],
            'stocks': result['stocks'],
            'portfolio_value': portfolio_value,
            'return': (portfolio_value - initial_capital) / initial_capital
        })

        # 打印周报
        print(f"\n持仓股票: {result['stocks']}")
        print(f"使用策略: {result['strategy']}")
        print(f"组合净值: {portfolio_value:,.2f}")
        print(f"累计收益: {(portfolio_value - initial_capital) / initial_capital:.2%}")

    # 汇总结果
    print("\n" + "=" * 60)
    print("回测结果汇总")
    print("=" * 60)

    df_weeks = pd.DataFrame(weekly_results)
    total_return = (portfolio_value - initial_capital) / initial_capital

    # 计算年化收益率
    days = (fridays[-1] - fridays[0]).days if len(fridays) > 1 else 1
    years = days / 365
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 计算最大回撤
    df_weeks['peak'] = df_weeks['portfolio_value'].cummax()
    df_weeks['drawdown'] = (df_weeks['peak'] - df_weeks['portfolio_value']) / df_weeks['peak']
    max_drawdown = df_weeks['drawdown'].max()

    print(f"回测区间: {fridays[0].strftime('%Y-%m-%d')} ~ {fridays[-1].strftime('%Y-%m-%d')}")
    print(f"总收益率: {total_return:.2%}")
    print(f"年化收益率: {annual_return:.2%}")
    print(f"最大回撤: {max_drawdown:.2%}")
    print(f"调仓次数: {len(fridays)}")

    # 策略使用统计
    strategy_counts = df_weeks['strategy'].value_counts()
    print(f"\n策略使用统计:")
    for strategy, count in strategy_counts.items():
        print(f"  {strategy}: {count}周 ({count/len(fridays):.1%})")

    # 保存结果
    output_path = f"back_testing/results/rotator_{start_date}_{end_date}.csv"
    df_weeks.to_csv(output_path, index=False)
    print(f"\n周报已保存: {output_path}")

    return df_weeks


def main():
    parser = argparse.ArgumentParser(description='策略轮动量化选股系统')
    parser.add_argument('--start', default='2020-01-01', help='回测开始日期')
    parser.add_argument('--end', default='2025-04-18', help='回测结束日期')
    parser.add_argument('--capital', type=float, default=1000000.0, help='初始资金')
    args = parser.parse_args()

    run_backtest(args.start, args.end, args.capital)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 创建结果目录**

```bash
mkdir -p back_testing/results
```

- [ ] **Step 3: 提交代码**

```bash
git add back_testing/run_rotator_backtest.py
git commit -m "feat: add rotator backtest entry point"
```

---

## Task 6: 集成测试与调优

**Files:**
- Modify: `back_testing/run_rotator_backtest.py`（根据测试结果调整）

- [ ] **Step 1: 运行小规模回测验证系统**

```bash
python -m back_testing.run_rotator_backtest --start 2024-01-01 --end 2024-03-31
```

- [ ] **Step 2: 检查输出结果**

- [ ] **Step 3: 根据结果调整参数或修复bug**

---

## 实现顺序

1. **SignalScorer** - 信号评分器（基础组件）
2. **StrategyEvaluator** - 策略评估器
3. **StockSelector** - 选股器
4. **PortfolioRotator** - 主控制器
5. **RotatorBacktest** - 回测入口
6. **集成测试** - 小规模验证后全量回测
