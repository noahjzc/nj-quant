# 稳健性检验 & 真实交易模拟增强 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有每日轮动回测系统上增加真实交易模拟（滑点/冲击/T+1/流动性）和稳健性检验（蒙特卡洛/CSCV/参数敏感性/PBO）两大能力。

**Architecture:** 新建 `backtesting/costs/`（成本模型+市场约束）和 `robustness/`（蒙特卡洛/CSCV/敏感性/统计检验）两个独立模块，扩展 `PerformanceAnalyzer` 增加进阶指标。成本模块嵌入回测引擎影响交易执行；稳健性模块作为回测后分析，从 PerformanceAnalyzer 获取日收益数据做统计推断。

**Tech Stack:** Python 3.12, numpy, pandas, scipy（线性回归/统计检验）

---

## 文件结构

| 操作 | 路径 | 职责 |
|------|------|------|
| 🆕 | `backtesting/costs/__init__.py` | 模块导出 |
| 🆕 | `backtesting/costs/cost_model.py` | 费率+滑点+平方根冲击模型 |
| 🆕 | `backtesting/costs/market_constraints.py` | T+1/涨跌停/停牌/ST/流动性过滤 |
| ✏️ | `strategy/rotation/trade_executor.py` | 迁移成本常量到 CostModel，保留兼容 |
| ✏️ | `strategy/rotation/daily_rotation_engine.py` | 集成 CostModel + MarketConstraints |
| ✏️ | `backtesting/analysis/performance_analyzer.py` | 新增6个指标 + 暴露 daily_returns |
| ✏️ | `backtesting/run_daily_rotation.py` | 输出 daily_returns + 跑稳健性检验 |
| 🆕 | `robustness/__init__.py` | 模块导出 |
| 🆕 | `robustness/monte_carlo.py` | 蒙特卡洛模拟 |
| 🆕 | `robustness/cscv.py` | CSCV 过拟合检测 |
| 🆕 | `robustness/sensitivity.py` | 参数敏感性分析 |
| 🆕 | `robustness/statistics.py` | Deflated Sharpe / PBO |
| 🆕 | `robustness/robustness_analyzer.py` | 门面类 |
| ✏️ | `optimization/optuna/run_daily_rotation_optimization.py` | Phase 2: Top5敏感性筛选 |
| 🆕 | `tests/backtesting/costs/test_cost_model.py` | CostModel 测试 |
| 🆕 | `tests/backtesting/costs/test_market_constraints.py` | MarketConstraints 测试 |
| 🆕 | `tests/robustness/test_monte_carlo.py` | 蒙特卡洛测试 |
| 🆕 | `tests/robustness/test_cscv.py` | CSCV 测试 |
| 🆕 | `tests/robustness/test_sensitivity.py` | 敏感性测试 |
| 🆕 | `tests/robustness/test_statistics.py` | 统计检验测试 |

---

### Task 1: CostModel — 统一成本模型

**Files:**
- Create: `backtesting/costs/__init__.py`
- Create: `backtesting/costs/cost_model.py`
- Create: `tests/backtesting/costs/__init__.py`
- Create: `tests/backtesting/costs/test_cost_model.py`

- [ ] **Step 1: 编写 CostModel 测试**

```python
# tests/backtesting/costs/test_cost_model.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pytest
from backtesting.costs.cost_model import CostModel, CostBreakdown


class TestCostModelDefaults:
    def setup_method(self):
        self.model = CostModel()

    def test_buy_cost_no_impact(self):
        """买入10万元，无冲击，应该只算过户费+佣金"""
        cost = self.model.buy_cost(price=10.0, shares=10000)
        # buy_value = 100000
        # transfer_fee = 100000 * 0.00002 = 2.0
        # brokerage = max(100000 * 0.0003, 5.0) = 30.0
        # slippage = 100000 * 0.0001 = 10.0
        # impact = 0 (sigma=0 since no volume info)
        # total = 2.0 + 30.0 + 10.0 = 42.0
        assert pytest.approx(cost.total, 0.01) == 42.0
        assert cost.stamp_duty == 0.0  # 买入不收印花税

    def test_sell_cost_with_stamp(self):
        """卖出10万元，应该包含印花税"""
        cost = self.model.sell_cost(price=10.0, shares=10000)
        # stamp_duty = 100000 * 0.001 = 100.0
        assert pytest.approx(cost.stamp_duty, 0.01) == 100.0
        assert cost.stamp_duty > 0

    def test_min_brokerage(self):
        """小单佣金最低5元"""
        cost = self.model.buy_cost(price=10.0, shares=100)  # buy_value = 1000
        # brokerage would be 1000 * 0.0003 = 0.3, but min is 5.0
        assert cost.brokerage == 5.0

    def test_slippage_default(self):
        """默认滑点 1bp"""
        cost = self.model.buy_cost(price=10.0, shares=10000)
        expected_slippage = 100000 * 0.0001
        assert pytest.approx(cost.slippage, 0.01) == expected_slippage

    def test_slippage_disabled(self):
        """滑点可关闭"""
        model = CostModel(slippage_bps=0)
        cost = model.buy_cost(price=10.0, shares=10000)
        assert cost.slippage == 0.0


class TestCostModelImpact:
    def setup_method(self):
        self.model = CostModel(impact_model='sqrt')

    def test_sqrt_impact_zero_when_no_volume(self):
        """无成交量数据时冲击为0"""
        cost = self.model.buy_cost(price=10.0, shares=10000)
        assert cost.impact == 0.0

    def test_sqrt_impact_small_order(self):
        """小单相对日成交占比极低，冲击趋近于0"""
        cost = self.model.buy_cost(price=10.0, shares=10000, amount_today=1e8, volatility=0.02)
        assert cost.impact < cost.slippage  # 冲击应小于滑点

    def test_sqrt_impact_large_order(self):
        """大单成交占比高，冲击显著"""
        cost = self.model.buy_cost(price=10.0, shares=50000, amount_today=5e6, volatility=0.03)
        # Q/V = 500000 / 5e6 = 0.1, sqrt(0.1) ≈ 0.316
        # impact = 0.03 * 0.316 ≈ 0.0095 → 约0.95%
        assert cost.impact > cost.slippage

    def test_impact_model_none(self):
        """可关闭冲击模型"""
        model = CostModel(impact_model='none')
        cost = model.buy_cost(price=10.0, shares=50000, amount_today=5e6, volatility=0.03)
        assert cost.impact == 0.0
```

- [ ] **Step 2: 验证测试失败**

```bash
pytest tests/backtesting/costs/test_cost_model.py -v
```
预期: ImportError (module not exist)

- [ ] **Step 3: 创建 `__init__.py`**

```python
# backtesting/costs/__init__.py
from backtesting.costs.cost_model import CostModel, CostBreakdown
from backtesting.costs.market_constraints import MarketConstraints

__all__ = ['CostModel', 'CostBreakdown', 'MarketConstraints']
```

- [ ] **Step 4: 实现 CostModel**

```python
# backtesting/costs/cost_model.py
from dataclasses import dataclass
import math


@dataclass
class CostBreakdown:
    stamp_duty: float = 0.0
    transfer_fee: float = 0.0
    brokerage: float = 0.0
    slippage: float = 0.0
    impact: float = 0.0

    @property
    def total(self) -> float:
        return self.stamp_duty + self.transfer_fee + self.brokerage + self.slippage + self.impact


class CostModel:
    """统一交易成本模型：费率 + 滑点 + 平方根冲击"""

    STAMP_DUTY = 0.001
    TRANSFER_FEE = 0.00002
    BROKERAGE = 0.0003
    MIN_BROKERAGE = 5.0

    def __init__(self, slippage_bps: float = 1.0, impact_model: str = 'sqrt'):
        self.slippage_bps = slippage_bps
        self.impact_model = impact_model  # 'sqrt' | 'fixed' | 'none'

    def buy_cost(self, price: float, shares: int,
                 amount_today: float = None, volatility: float = None) -> CostBreakdown:
        """买入总成本（买入不收印花税）"""
        return self._calc_cost(price, shares, is_buy=True,
                               amount_today=amount_today, volatility=volatility)

    def sell_cost(self, price: float, shares: int,
                  amount_today: float = None, volatility: float = None) -> CostBreakdown:
        """卖出总成本（含印花税）"""
        return self._calc_cost(price, shares, is_buy=False,
                               amount_today=amount_today, volatility=volatility)

    def _calc_cost(self, price: float, shares: int, is_buy: bool,
                   amount_today: float = None, volatility: float = None) -> CostBreakdown:
        value = price * shares

        stamp = 0.0 if is_buy else value * self.STAMP_DUTY
        transfer = value * self.TRANSFER_FEE
        brokerage = max(value * self.BROKERAGE, self.MIN_BROKERAGE)
        slippage = value * (self.slippage_bps / 10000.0)
        impact = self._calc_impact(value, amount_today, volatility)

        return CostBreakdown(
            stamp_duty=stamp,
            transfer_fee=transfer,
            brokerage=brokerage,
            slippage=slippage,
            impact=impact,
        )

    def _calc_impact(self, order_value: float, amount_today: float = None,
                     volatility: float = None) -> float:
        if self.impact_model == 'none':
            return 0.0
        if amount_today is None or amount_today <= 0:
            return 0.0
        if volatility is None:
            volatility = 0.02  # 默认2%日波动率

        if self.impact_model == 'sqrt':
            # σ * sqrt(Q / V)
            q_v = order_value / amount_today
            return volatility * math.sqrt(q_v) * order_value
        elif self.impact_model == 'fixed':
            return order_value * 0.0005  # 固定5bp
        return 0.0
```

- [ ] **Step 5: 运行测试**

```bash
pytest tests/backtesting/costs/test_cost_model.py -v
```
预期: 全部 PASS

---

### Task 2: MarketConstraints — 市场约束

**Files:**
- Create: `backtesting/costs/market_constraints.py`
- Create: `tests/backtesting/costs/test_market_constraints.py`

- [ ] **Step 1: 编写 MarketConstraints 测试**

```python
# tests/backtesting/costs/test_market_constraints.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pytest
import pandas as pd
import numpy as np
from backtesting.costs.market_constraints import MarketConstraints


class TestCanBuy:
    def setup_method(self):
        self.constraints = MarketConstraints()

    def test_normal_stock_allowed(self):
        ok, reason = self.constraints.can_buy('sh600000', price=10.0, amount_today=5e7, pct_chg=3.0,
                                               is_st=False, is_suspended=False)
        assert ok

    def test_limit_up_blocked(self):
        """涨停不能买入"""
        ok, reason = self.constraints.can_buy('sh600000', price=10.0, amount_today=5e7, pct_chg=9.95,
                                               is_st=False, is_suspended=False)
        assert not ok

    def test_st_blocked(self):
        ok, reason = self.constraints.can_buy('sh600000', price=10.0, amount_today=5e7, pct_chg=3.0,
                                               is_st=True, is_suspended=False)
        assert not ok

    def test_low_volume_blocked(self):
        """日成交额不足最低门槛不能买入"""
        ok, reason = self.constraints.can_buy('sh600000', price=10.0, amount_today=5e4, pct_chg=3.0,
                                               is_st=False, is_suspended=False)
        assert not ok


class TestCanSell:
    def setup_method(self):
        self.constraints = MarketConstraints()

    def test_normal_sell_allowed(self):
        from strategy.rotation.daily_rotation_engine import Position
        pos = Position(stock_code='sh600000', shares=1000, buy_price=10.0, buy_date='2024-01-05')
        ok, reason = self.constraints.can_sell(pos, pd.Timestamp('2024-01-08'),
                                                price=11.0, pct_chg=5.0)
        assert ok

    def test_t1_blocked_same_day(self):
        """T+1：买入当日不能卖出"""
        from strategy.rotation.daily_rotation_engine import Position
        pos = Position(stock_code='sh600000', shares=1000, buy_price=10.0, buy_date='2024-01-08')
        ok, reason = self.constraints.can_sell(pos, pd.Timestamp('2024-01-08'),
                                                price=11.0, pct_chg=5.0)
        assert not ok
        assert 'T+1' in reason

    def test_limit_down_blocked(self):
        """跌停不能卖出"""
        from strategy.rotation.daily_rotation_engine import Position
        pos = Position(stock_code='sh600000', shares=1000, buy_price=10.0, buy_date='2024-01-05')
        ok, reason = self.constraints.can_sell(pos, pd.Timestamp('2024-01-08'),
                                                price=9.0, pct_chg=-10.0)
        assert not ok


class TestFilterPool:
    def setup_method(self):
        self.constraints = MarketConstraints()

    def test_basic_filter(self):
        df = pd.DataFrame({
            'stock_code': ['sh001', 'sh002', 'sh003', 'sh004'],
            'is_st': [False, True, False, False],
            'pct_chg': [3.0, 5.0, 10.0, -10.0],
            'amount': [5e7, 1e8, 3e7, 2e7],
        })
        result = self.constraints.filter_pool(df, pd.Timestamp('2024-01-08'))
        # sh002: ST → removed, sh003: 涨停 → removed, sh004: 跌停 → removed
        # sh001: OK
        assert list(result) == ['sh001']
```

- [ ] **Step 2: 实现 MarketConstraints**

```python
# backtesting/costs/market_constraints.py
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np


class MarketConstraints:
    """A股市场硬约束"""

    LIMIT_UP_THRESHOLD = 9.5       # 涨停阈值（pct_chg >= 9.5%）
    LIMIT_DOWN_THRESHOLD = -9.5    # 跌停阈值
    MIN_AMOUNT = 1_000_000         # 最低日成交额 100万元

    # 不同板块涨跌停幅度（用于精确判断，替代当前引擎中的硬编码）
    # 主板 10%, 创业板/科创板(STAR) 20%, 北交所 30%
    _LIMIT_RULES = {
        'sh': {'600': 0.10, '601': 0.10, '603': 0.10, '605': 0.10, '688': 0.20},
        'sz': {'000': 0.10, '001': 0.10, '002': 0.10, '003': 0.10, '300': 0.20, '301': 0.20},
        'bj': {'8': 0.30, '4': 0.30},
    }

    def __init__(self, exclude_st: bool = True, exclude_limit_up: bool = True,
                 exclude_limit_down: bool = True, exclude_suspended: bool = True,
                 min_amount: float = MIN_AMOUNT):
        self.exclude_st = exclude_st
        self.exclude_limit_up = exclude_limit_up
        self.exclude_limit_down = exclude_limit_down
        self.exclude_suspended = exclude_suspended
        self.min_amount = min_amount

    def can_buy(self, stock_code: str, price: float, amount_today: float,
                pct_chg: float, is_st: bool, is_suspended: bool) -> Tuple[bool, str]:
        """检查是否可以买入"""
        if self.exclude_st and is_st:
            return False, "ST"
        if self.exclude_suspended and is_suspended:
            return False, "停牌"
        if self.exclude_limit_up and pct_chg >= self.LIMIT_UP_THRESHOLD:
            return False, "涨停"
        if amount_today < self.min_amount:
            return False, f"成交额不足({amount_today:,.0f} < {self.min_amount:,.0f})"
        return True, ""

    def can_sell(self, position, trade_date: pd.Timestamp,
                 price: float, pct_chg: float) -> Tuple[bool, str]:
        """检查是否可以卖出"""
        # T+1: 买入当日不能卖出
        buy_ts = pd.Timestamp(position.buy_date)
        if trade_date == buy_ts:
            return False, "T+1约束"
        if self.exclude_limit_down and pct_chg <= self.LIMIT_DOWN_THRESHOLD:
            return False, "跌停"
        return True, ""

    def filter_pool(self, today_df: pd.DataFrame, date: pd.Timestamp) -> List[str]:
        """一站式股票池过滤，返回可交易股票代码列表"""
        df = today_df.copy()

        if 'stock_code' not in df.columns:
            return []

        # 停牌过滤：有 price=0 或 NaN 的
        if self.exclude_suspended and 'close' in df.columns:
            df = df[df['close'].notna() & (df['close'] > 0)]

        # ST 过滤
        if self.exclude_st and 'is_st' in df.columns:
            df = df[~df['is_st'].astype(bool)]

        # 涨跌停过滤
        if 'pct_chg' in df.columns:
            if self.exclude_limit_up:
                df = df[df['pct_chg'] < self.LIMIT_UP_THRESHOLD]
            if self.exclude_limit_down:
                df = df[df['pct_chg'] > self.LIMIT_DOWN_THRESHOLD]

        # 流动性过滤
        if 'amount' in df.columns:
            df = df[df['amount'] >= self.min_amount]

        return df['stock_code'].tolist()
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/backtesting/costs/test_market_constraints.py -v
```
预期: 全部 PASS

---

### Task 3: 扩展 PerformanceAnalyzer

**Files:**
- Modify: `backtesting/analysis/performance_analyzer.py`

- [ ] **Step 1: 编写新指标的测试**

```python
# tests/backtesting/analysis/test_performance_analyzer_ext.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from backtesting.analysis.performance_analyzer import PerformanceAnalyzer


class TestNewMetrics:
    def setup_method(self):
        n = 252
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, n)
        equity = 1_000_000 * np.cumprod(1 + returns)
        equity = np.insert(equity, 0, 1_000_000)

        benchmark_returns = np.random.normal(0.0005, 0.015, n)

        self.analyzer = PerformanceAnalyzer(
            trades=[],
            initial_capital=1_000_000,
            equity_curve=equity.tolist(),
            periods_per_year=252,
            benchmark_returns=benchmark_returns,
        )
        metrics = self.analyzer.calculate_metrics()

    def test_daily_returns_exposed(self):
        assert hasattr(self.analyzer, 'daily_returns')
        assert len(self.analyzer.daily_returns) == 252

    def test_information_ratio(self):
        ir = self.analyzer.information_ratio()
        assert isinstance(ir, float)

    def test_alpha_beta(self):
        ab = self.analyzer.alpha_beta()
        assert 'alpha' in ab
        assert 'beta' in ab
        assert 'r_squared' in ab

    def test_skewness_kurtosis(self):
        sk = self.analyzer.skewness_kurtosis()
        assert 'skewness' in sk
        assert 'kurtosis' in sk

    def test_rolling_sharpe(self):
        rs = self.analyzer.rolling_sharpe(window=60)
        assert len(rs) > 0

    def test_monthly_returns(self):
        mr = self.analyzer.monthly_returns()
        assert isinstance(mr, dict)

    def test_max_drawdown_duration(self):
        dur = self.analyzer.max_drawdown_duration()
        assert dur >= 0

    def test_no_benchmark_returns(self):
        """无基准收益时，基准相关指标应返回默认值"""
        analyzer = PerformanceAnalyzer(
            trades=[],
            initial_capital=1_000_000,
            equity_curve=[1_000_000, 1_010_000, 1_020_000],
            periods_per_year=252,
        )
        analyzer.calculate_metrics()
        ir = analyzer.information_ratio()
        assert ir == 0.0
```

- [ ] **Step 2: 运行测试确认新增方法不存在**

```bash
pytest tests/backtesting/analysis/test_performance_analyzer_ext.py -v
```
预期: FAIL (AttributeError: 'PerformanceAnalyzer' object has no attribute 'daily_returns')

- [ ] **Step 3: 实现新指标**

在 `PerformanceAnalyzer.__init__` 中添加 `benchmark_returns` 参数，并在 `calculate_metrics` 中计算 `daily_returns`：

```python
# 修改 __init__，新增 benchmark_returns 参数
def __init__(
    self,
    trades: List[Dict],
    initial_capital: float = 1000000.0,
    benchmark_index: str = 'sh000001',
    risk_free_rate: float = 0.025,
    equity_curve: Optional[List[float]] = None,
    periods_per_year: int = 252,
    benchmark_returns: Optional[np.ndarray] = None,  # 🆕
):
    # ... existing init code ...
    self.benchmark_returns = benchmark_returns
    self.daily_returns = None  # 🆕 在 calculate_metrics 中赋值
```

在 `calculate_metrics` 方法中，净值曲线分支增加 `daily_returns` 计算：

```python
# 在 portfolio-level metrics 分支中增加:
if self.equity_curve is not None and len(self.equity_curve) > 1:
    equity = np.array(self.equity_curve)
    period_returns = (equity[1:] - equity[:-1]) / equity[:-1]
    self.daily_returns = period_returns  # 🆕 暴露
    # ... rest unchanged ...
```

新增以下方法：

```python
def information_ratio(self) -> float:
    """信息比率 = (策略年化 - 基准年化) / 跟踪误差"""
    if self.benchmark_returns is None or self.daily_returns is None:
        return 0.0
    if len(self.benchmark_returns) != len(self.daily_returns):
        return 0.0
    excess = self.daily_returns - self.benchmark_returns
    if np.std(excess, ddof=1) < 1e-10:
        return 0.0
    ann_excess = np.mean(excess) * self.periods_per_year
    ann_te = np.std(excess, ddof=1) * np.sqrt(self.periods_per_year)
    return ann_excess / ann_te

def alpha_beta(self) -> Dict:
    """CAPM 回归: 策略收益 ~ 基准收益，返回 alpha, beta, r_squared"""
    if self.benchmark_returns is None or self.daily_returns is None:
        return {'alpha': 0.0, 'beta': 0.0, 'r_squared': 0.0}
    if len(self.benchmark_returns) != len(self.daily_returns):
        return {'alpha': 0.0, 'beta': 0.0, 'r_squared': 0.0}
    X = self.benchmark_returns
    y = self.daily_returns
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    beta = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean) ** 2)
    alpha = y_mean - beta * X_mean
    # R²
    ss_res = np.sum((y - (alpha + beta * X)) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    return {'alpha': float(alpha), 'beta': float(beta), 'r_squared': float(r_squared)}

def skewness_kurtosis(self) -> Dict:
    """收益偏度和超额峰度"""
    if self.daily_returns is None or len(self.daily_returns) < 3:
        return {'skewness': 0.0, 'kurtosis': 0.0}
    from scipy import stats
    skew = stats.skew(self.daily_returns)
    kurt = stats.kurtosis(self.daily_returns)  # excess kurtosis (Fisher)
    return {'skewness': float(skew), 'kurtosis': float(kurt)}

def rolling_sharpe(self, window: int = 60) -> np.ndarray:
    """滚动 Sharpe 比率"""
    if self.daily_returns is None or len(self.daily_returns) < window:
        return np.array([])
    r = self.daily_returns
    rf_daily = self.risk_free_rate / self.periods_per_year
    rolling_mean = np.convolve(r - rf_daily, np.ones(window)/window, mode='valid')
    rolling_std = np.array([np.std(r[i:i+window], ddof=1) for i in range(len(r)-window+1)])
    with np.errstate(divide='ignore', invalid='ignore'):
        sharpe = np.where(rolling_std > 1e-10,
                          rolling_mean * np.sqrt(self.periods_per_year) / rolling_std,
                          0.0)
    return sharpe

def monthly_returns(self) -> Dict:
    """月度收益明细。假设 daily_returns 顺序对应 equity_curve 的日期序列。"""
    # 返回示例: {'2024-01': 0.032, '2024-02': -0.015, ...}
    # 简单实现：按21个交易日为一组近似
    if self.daily_returns is None or len(self.daily_returns) == 0:
        return {}
    monthly = {}
    n = len(self.daily_returns)
    for i, start in enumerate(range(0, n, 21)):
        end = min(start + 21, n)
        period = self.daily_returns[start:end]
        monthly_ret = np.prod(1 + period) - 1
        month_label = f"Month_{i+1:02d}"
        monthly[month_label] = float(monthly_ret)
    return monthly

def max_drawdown_duration(self) -> int:
    """最长回撤持续天数（从峰值到恢复的天数）"""
    if self.equity_curve is None or len(self.equity_curve) < 2:
        return 0
    equity = np.array(self.equity_curve)
    peak = equity[0]
    max_dur = 0
    current_dur = 0
    for value in equity[1:]:
        if value >= peak:
            peak = value
            current_dur = 0
        else:
            current_dur += 1
            if current_dur > max_dur:
                max_dur = current_dur
    return max_dur
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/backtesting/analysis/test_performance_analyzer_ext.py -v
```
预期: 全部 PASS

- [ ] **Step 5: 运行已有测试确保不回归**

```bash
pytest tests/ -v -k "performance" --tb=short
```
预期: 已有测试 PASS

---

### Task 4: 集成 CostModel 和 MarketConstraints 到引擎

**Files:**
- Modify: `strategy/rotation/daily_rotation_engine.py`
- Modify: `strategy/rotation/trade_executor.py`

- [ ] **Step 1: 更新 TradeExecutor 使用 CostModel**

```python
# strategy/rotation/trade_executor.py
from backtesting.costs.cost_model import CostModel

class TradeExecutor:
    """交易执行器，委托给 CostModel 做成本计算"""

    def __init__(self, cost_model: CostModel = None):
        self.cost_model = cost_model or CostModel()

    def execute_buy(self, stock_code: str, price: float, cash: float,
                    amount_today: float = None, volatility: float = None) -> tuple[int, float]:
        """返回值不变: (shares, total_cost)"""
        # ... 保留原有股数计算逻辑，成本改用 cost_model
        cost = self.cost_model.buy_cost(price, shares, amount_today, volatility)
        return shares, cost.total

    def execute_sell(self, stock_code: str, price: float, shares: int,
                     amount_today: float = None, volatility: float = None) -> tuple[int, float]:
        cost = self.cost_model.sell_cost(price, shares, amount_today, volatility)
        return shares, cost.total
```

- [ ] **Step 2: 修改 DailyRotationEngine.__init__**

```python
# 在 __init__ 中添加:
from backtesting.costs.cost_model import CostModel
from backtesting.costs.market_constraints import MarketConstraints

self.cost_model = getattr(config, 'cost_model', None) or CostModel()
self.market_constraints = MarketConstraints(
    exclude_st=self.config.exclude_st,
    exclude_limit_up=self.config.exclude_limit_up,
    exclude_limit_down=self.config.exclude_limit_down,
    exclude_suspended=self.config.exclude_suspended,
)
# 更新 trade_executor 使用新的 cost_model
self.trade_executor = TradeExecutor(cost_model=self.cost_model)
```

- [ ] **Step 3: 替换 _filter_stock_pool**

```python
def _filter_stock_pool(self, date: pd.Timestamp) -> set:
    codes = self.market_constraints.filter_pool(self._today_df, date)
    return set(codes)
```

- [ ] **Step 4: 在 _check_and_sell 中增加 T+1 检查**

在 `_check_and_sell` 的持仓遍历中，卖出前增加约束检查：

```python
for stock_code, position in list(self.positions.items()):
    price = current_prices.get(stock_code, 0.0)
    pct_chg = self._get_stock_pct_chg(stock_code)  # 从 _today_df 获取

    ok, reason = self.market_constraints.can_sell(position, date, price, pct_chg)
    if not ok:
        if reason == 'T+1约束':
            continue  # 静默跳过
        else:
            logger.debug(f"[SKIP] {stock_code} 不可卖出: {reason}")
            continue
    # ... 继续原有的卖出信号检查 ...
```

- [ ] **Step 5: 更新 _execute_buy 成本计算**

将 `_execute_buy` 中分散的成本计算替换为：

```python
# 旧:
# transfer_fee = buy_value * self.trade_executor.TRANSFER_FEE
# brokerage = max(buy_value * self.trade_executor.BROKERAGE, self.trade_executor.MIN_BROKERAGE)
# cost = transfer_fee + brokerage

# 新:
amount_today = row.get('amount', None)
cost = self.cost_model.buy_cost(price, shares, amount_today).total
```

- [ ] **Step 6: 运行已有测试确保引擎不回归**

```bash
pytest tests/strategy/rotation/ -v --tb=short
```
预期: 已有测试 PASS

---

### Task 5: 蒙特卡洛模拟

**Files:**
- Create: `robustness/__init__.py`
- Create: `robustness/monte_carlo.py`
- Create: `tests/robustness/__init__.py`
- Create: `tests/robustness/test_monte_carlo.py`

- [ ] **Step 1: 编写测试**

```python
# tests/robustness/test_monte_carlo.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
from robustness.monte_carlo import MonteCarloSim, MCSimulationResult


class TestMonteCarloSim:
    def setup_method(self):
        np.random.seed(42)
        n = 252
        self.daily_returns = np.random.normal(0.001, 0.02, n)

    def test_basic_run(self):
        sim = MonteCarloSim()
        result = sim.run(self.daily_returns, n_sim=200)

        assert isinstance(result, MCSimulationResult)
        assert len(result.sharpe_distribution) == 200
        assert len(result.max_dd_distribution) == 200
        # 95% CI 应有下界 < 均值 < 上界
        assert result.sharpe_95ci[0] <= result.mean_sharpe <= result.sharpe_95ci[1]

    def test_reproducible(self):
        sim = MonteCarloSim(seed=42)
        r1 = sim.run(self.daily_returns, n_sim=100)
        sim2 = MonteCarloSim(seed=42)
        r2 = sim2.run(self.daily_returns, n_sim=100)
        assert r1.mean_sharpe == r2.mean_sharpe

    def test_too_few_returns(self):
        sim = MonteCarloSim()
        result = sim.run(np.array([0.01, -0.02]), n_sim=100)
        assert result.mean_sharpe == 0.0
```

- [ ] **Step 2: 实现**

```python
# robustness/monte_carlo.py
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MCSimulationResult:
    mean_sharpe: float = 0.0
    sharpe_std: float = 0.0
    sharpe_95ci: tuple = (0.0, 0.0)
    sharpe_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_max_dd: float = 0.0
    max_dd_95ci: tuple = (0.0, 0.0)
    max_dd_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    sim_curves: list = field(default_factory=list)


class MonteCarloSim:
    """非参数蒙特卡洛模拟：对日收益率放回重采样，生成 N 条模拟净值曲线"""

    def __init__(self, rf_annual: float = 0.025, periods_per_year: int = 252, seed: int = None):
        self.rf = rf_annual
        self.ppy = periods_per_year
        self.rng = np.random.RandomState(seed)

    def run(self, daily_returns: np.ndarray, n_sim: int = 2000) -> MCSimulationResult:
        if len(daily_returns) < 5 or n_sim < 1:
            return MCSimulationResult()

        n = len(daily_returns)
        sharpes = np.empty(n_sim)
        max_dds = np.empty(n_sim)

        for i in range(n_sim):
            sampled = daily_returns[self.rng.randint(0, n, size=n)]
            equity = np.cumprod(1 + sampled)
            sharpes[i] = self._sharpe(sampled)
            max_dds[i] = self._max_drawdown(equity)

        result = MCSimulationResult(
            mean_sharpe=float(np.mean(sharpes)),
            sharpe_std=float(np.std(sharpes, ddof=1)),
            sharpe_95ci=(float(np.percentile(sharpes, 2.5)), float(np.percentile(sharpes, 97.5))),
            sharpe_distribution=sharpes,
            mean_max_dd=float(np.mean(max_dds)),
            max_dd_95ci=(float(np.percentile(max_dds, 2.5)), float(np.percentile(max_dds, 97.5))),
            max_dd_distribution=max_dds,
        )
        return result

    def _sharpe(self, returns: np.ndarray) -> float:
        excess = np.mean(returns) - self.rf / self.ppy
        vol = np.std(returns, ddof=1)
        if vol < 1e-10:
            return 0.0
        return excess / vol * np.sqrt(self.ppy)

    def _max_drawdown(self, equity: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        return float(np.max(dd))
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/robustness/test_monte_carlo.py -v
```
预期: 全部 PASS

---

### Task 6: CSCV 过拟合检测

**Files:**
- Create: `robustness/cscv.py`
- Create: `tests/robustness/test_cscv.py`

- [ ] **Step 1: 编写测试**

```python
# tests/robustness/test_cscv.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from robustness.cscv import CSCVAnalyzer, CSCVResult


class TestCSCVAnalyzer:
    def setup_method(self):
        np.random.seed(42)
        n = 252
        self.returns = np.random.normal(0.001, 0.02, n)

    def test_basic_run(self):
        analyzer = CSCVAnalyzer(seed=42)
        result = analyzer.run(self.returns, n_split=10, n_comb=50)
        assert isinstance(result, MCSimulationResult)  # reuses the type or similar
        assert 0.0 <= result.overfit_probability <= 1.0

    def test_too_few_returns(self):
        analyzer = CSCVAnalyzer()
        result = analyzer.run(np.array([0.01, -0.02]), n_split=4)
        assert result.overfit_probability == 0.0
```

- [ ] **Step 2: 实现**

```python
# robustness/cscv.py
from dataclasses import dataclass
import numpy as np


@dataclass
class CSCVResult:
    overfit_probability: float = 0.0    # PBO
    rank_decay: float = 0.0             # IS排名衰减到OOS的程度
    is_robust: bool = False             # PBO < 0.1 视为通过


class CSCVAnalyzer:
    """Combinatorially Symmetric Cross-Validation (Bailey et al. 2017)"""

    def __init__(self, rf_annual: float = 0.025, periods_per_year: int = 252, seed: int = None):
        self.rf = rf_annual
        self.ppy = periods_per_year
        self.rng = np.random.RandomState(seed)

    def run(self, daily_returns: np.ndarray, n_split: int = 16, n_comb: int = 100) -> CSCVResult:
        n = len(daily_returns)
        if n < n_split * 2:
            return CSCVResult()

        # 切成 S 段
        segment_size = n // n_split
        segments = []
        for i in range(n_split):
            seg = daily_returns[i * segment_size:(i + 1) * segment_size]
            segments.append(seg)

        is_ranks = []
        oos_ranks = []

        for _ in range(n_comb):
            # 随机选 S/2 做 IS
            indices = list(range(n_split))
            self.rng.shuffle(indices)
            half = n_split // 2
            is_idx = indices[:half]
            oos_idx = indices[half:]

            is_returns = np.concatenate([segments[i] for i in is_idx])
            oos_returns = np.concatenate([segments[i] for i in oos_idx])

            is_sharpe = self._sharpe(is_returns)
            oos_sharpe = self._sharpe(oos_returns)

            is_ranks.append(is_sharpe)
            oos_ranks.append(oos_sharpe)

        is_ranks = np.array(is_ranks)
        oos_ranks = np.array(oos_ranks)

        # PBO = IS好但OOS差的比例
        is_best = np.argmax(is_ranks)
        oos_rank_of_best = np.sum(oos_ranks > oos_ranks[is_best]) / (len(oos_ranks) - 1) if len(oos_ranks) > 1 else 0.0

        # 简化 PBO: 高 IS 排名 → 低 OOS 排名的概率
        pbo = 1.0 - oos_rank_of_best if oos_rank_of_best > 0 else 0.0

        return CSCVResult(
            overfit_probability=float(np.clip(pbo, 0.0, 1.0)),
            rank_decay=float(oos_rank_of_best),
            is_robust=pbo < 0.1,
        )

    def _sharpe(self, returns: np.ndarray) -> float:
        excess = np.mean(returns) - self.rf / self.ppy
        vol = np.std(returns, ddof=1)
        if vol < 1e-10:
            return 0.0
        return excess / vol * np.sqrt(self.ppy)
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/robustness/test_cscv.py -v
```
预期: 全部 PASS

---

### Task 7: 参数敏感性分析

**Files:**
- Create: `robustness/sensitivity.py`
- Create: `tests/robustness/test_sensitivity.py`

- [ ] **Step 1: 编写测试**

```python
# tests/robustness/test_sensitivity.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from robustness.sensitivity import SensitivityAnalyzer, SensitivityResult


class TestSensitivityAnalyzer:
    def test_empty_params(self):
        analyzer = SensitivityAnalyzer()
        result = analyzer.run({}, engine_factory=lambda p: None)
        assert result.overall_stability_score == 1.0
        assert len(result.per_param) == 0

    def test_stable_params(self):
        """参数扰动后 Sharpe 变化小，应为稳定"""
        analyzer = SensitivityAnalyzer(sharpe_fn_name='_sharpe')
        # 模拟：参数名 → (base_sharpe, perturbed_sharpes)
        params = {'a': 1.0, 'b': 2.0}
        def mock_run(cfg):
            if cfg['a'] == 1.0 and cfg['b'] == 2.0:
                return {'sharpe_ratio': 1.5}
            return {'sharpe_ratio': 1.48}  # 变化很小 → 稳定

        result = analyzer._evaluate(params, mock_run)
        assert result.overall_stability_score > 0.8
```

- [ ] **Step 2: 实现**

```python
# robustness/sensitivity.py
from dataclasses import dataclass, field
from typing import Dict, Callable, Any
from copy import deepcopy


@dataclass
class SensitivityResult:
    per_param: Dict[str, dict] = field(default_factory=dict)
    overall_stability_score: float = 1.0  # 0~1，1 = 完全稳定


class SensitivityAnalyzer:
    """参数敏感性分析：对每个参数 ±20% 扰动，对比 Sharpe 变化"""

    def __init__(self, perturbation_pct: float = 0.2):
        self.perturbation_pct = perturbation_pct

    def run(self, params: Dict[str, float],
            engine_factory: Callable[[Dict], Any]) -> SensitivityResult:
        """params: 策略参数字典
           engine_factory: 接收参数字典，返回有 shapre_ratio 的 dict 的可调用对象"""
        return self._evaluate(params, engine_factory)

    def _evaluate(self, params: Dict[str, float],
                  run_fn: Callable[[Dict], Any]) -> SensitivityResult:
        # 跑基准
        base_result = run_fn(deepcopy(params))
        base_sharpe = self._extract_sharpe(base_result)

        per_param = {}
        sharpe_changes = []

        for key, value in params.items():
            if not isinstance(value, (int, float)):
                continue

            delta = abs(value) * self.perturbation_pct if value != 0 else self.perturbation_pct
            changes = []

            for perturbed_value in [value + delta, value - delta]:
                cfg = deepcopy(params)
                cfg[key] = perturbed_value
                result = run_fn(cfg)
                p_sharpe = self._extract_sharpe(result)
                changes.append(p_sharpe)

            avg_change = abs((sum(changes) / 2 - base_sharpe) / base_sharpe) if base_sharpe != 0 else 0.0
            sharpe_changes.append(avg_change)

            per_param[key] = {
                'base_value': value,
                'delta': delta,
                'sharpe_change_pct': round(avg_change * 100, 2),
                'stable': avg_change < 0.10,  # 变化 < 10% 视为稳定
            }

        # 整体稳定性：平均敏感度的倒数，clip到[0,1]
        if sharpe_changes:
            avg_sensitivity = sum(sharpe_changes) / len(sharpe_changes)
            stability = max(0.0, min(1.0, 1.0 - avg_sensitivity * 5))
        else:
            stability = 1.0

        return SensitivityResult(per_param=per_param, overall_stability_score=round(stability, 4))

    def _extract_sharpe(self, result: Any) -> float:
        if isinstance(result, dict):
            return float(result.get('sharpe_ratio', 0))
        if hasattr(result, 'sharpe_ratio'):
            return float(result.sharpe_ratio)
        return 0.0
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/robustness/test_sensitivity.py -v
```
预期: 全部 PASS

---

### Task 8: Deflated Sharpe & PBO 统计检验

**Files:**
- Create: `robustness/statistics.py`
- Create: `tests/robustness/test_statistics.py`

- [ ] **Step 1: 编写测试**

```python
# tests/robustness/test_statistics.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from robustness.statistics import deflated_sharpe_ratio, probability_of_backtest_overfit


class TestDefaltedSharpe:
    def test_basic(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        dsr = deflated_sharpe_ratio(returns, n_trials=50)
        assert isinstance(dsr, float)

    def test_few_returns(self):
        dsr = deflated_sharpe_ratio(np.array([0.01]))
        assert dsr == 0.0


class TestPBO:
    def test_basic(self):
        np.random.seed(42)
        is_sharpes = np.random.normal(1.0, 0.3, 100)
        oos_sharpes = np.random.normal(0.5, 0.4, 100)
        pbo = probability_of_backtest_overfit(is_sharpes, oos_sharpes)
        assert 0.0 <= pbo <= 1.0

    def test_empty(self):
        pbo = probability_of_backtest_overfit(np.array([]), np.array([]))
        assert pbo == 0.0
```

- [ ] **Step 2: 实现**

```python
# robustness/statistics.py
import numpy as np
from scipy import stats


def deflated_sharpe_ratio(daily_returns: np.ndarray, n_trials: int = 100,
                          rf_annual: float = 0.025, periods_per_year: int = 252) -> float:
    """Deflated Sharpe Ratio (Harvey & Liu 2015)
    考虑多重测试惩罚：在 N 次试验中选择的最大 Sharpe 不再服从标准分布。
    """
    n = len(daily_returns)
    if n < 2:
        return 0.0

    rf_daily = rf_annual / periods_per_year
    observed_sharpe = (np.mean(daily_returns) - rf_daily) / np.std(daily_returns, ddof=1) * np.sqrt(periods_per_year)

    # 近似：在零假设（真实Sharpe=0）下，最大Sharpe的分布
    # 使用 E[max(|Z|)] ≈ sqrt(2 * log(n_trials)) 的近似
    if n_trials <= 1:
        return float(observed_sharpe)

    # 期望最大 Sharpe (零假设)
    e_max = np.sqrt(2 * np.log(n_trials))
    # 标准差
    var_max = 1.0 / n  # 渐近方差

    if var_max < 1e-10:
        return 0.0

    dsr = (observed_sharpe - e_max) / np.sqrt(var_max)
    return float(dsr)


def probability_of_backtest_overfit(is_sharpes: np.ndarray, oos_sharpes: np.ndarray) -> float:
    """PBO: IS最佳解在OOS排名后50%的概率 (Bailey et al. 2014)
    """
    if len(is_sharpes) == 0 or len(oos_sharpes) == 0:
        return 0.0

    n = len(is_sharpes)
    # 对每组IS/OOS组合，计算IS排名 vs OOS排名
    below_median = 0
    for i in range(n):
        is_better = np.sum(is_sharpes > is_sharpes[i])
        oos_better = np.sum(oos_sharpes > oos_sharpes[i])
        if oos_better > n / 2 and is_better < n / 2:
            below_median += 1

    return below_median / n
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/robustness/test_statistics.py -v
```
预期: 全部 PASS

---

### Task 9: RobustnessAnalyzer 门面

**Files:**
- Create: `robustness/robustness_analyzer.py`

- [ ] **Step 1: 实现门面类**

```python
# robustness/robustness_analyzer.py
from dataclasses import dataclass, field
import numpy as np
from robustness.monte_carlo import MonteCarloSim, MCSimulationResult
from robustness.cscv import CSCVAnalyzer, CSCVResult
from robustness.sensitivity import SensitivityAnalyzer, SensitivityResult
from robustness.statistics import deflated_sharpe_ratio


@dataclass
class RobustnessReport:
    monte_carlo: MCSimulationResult = field(default_factory=MCSimulationResult)
    cscv: CSCVResult = field(default_factory=CSCVResult)
    sensitivity: SensitivityResult = field(default_factory=SensitivityResult)
    deflated_sharpe: float = 0.0
    summary: str = ""


class RobustnessAnalyzer:
    """稳健性检验门面 — 一站式调用"""

    def __init__(self, performance_analyzer):
        """
        Args:
            performance_analyzer: PerformanceAnalyzer 实例，必须有 daily_returns 属性
        """
        self.analyzer = performance_analyzer
        self._rf = getattr(performance_analyzer, 'risk_free_rate', 0.025)
        self._ppy = getattr(performance_analyzer, 'periods_per_year', 252)

    def run_all(self, n_mc: int = 2000, n_cscv_comb: int = 100,
                sensitivity_engine_factory=None) -> RobustnessReport:
        """运行全部稳健性检验"""
        daily_returns = getattr(self.analyzer, 'daily_returns', None)

        report = RobustnessReport()

        if daily_returns is not None and len(daily_returns) > 5:
            # 蒙特卡洛
            mc = MonteCarloSim(rf_annual=self._rf, periods_per_year=self._ppy)
            report.monte_carlo = mc.run(daily_returns, n_sim=n_mc)

            # CSCV
            cscv = CSCVAnalyzer(rf_annual=self._rf, periods_per_year=self._ppy)
            report.cscv = cscv.run(daily_returns, n_comb=n_cscv_comb)

            # Deflated Sharpe
            report.deflated_sharpe = deflated_sharpe_ratio(daily_returns, rf_annual=self._rf, periods_per_year=self._ppy)

        # 敏感性（需要 engine_factory）
        if sensitivity_engine_factory is not None:
            # 从 analyzer 获取当前参数...
            # sensitivity 需要额外参数，调用方自行处理
            pass

        report.summary = self._build_summary(report)
        return report

    def run_sensitivity(self, params: dict, engine_factory) -> SensitivityResult:
        """单独运行敏感性分析"""
        sa = SensitivityAnalyzer()
        return sa.run(params, engine_factory)

    def _build_summary(self, report: RobustnessReport) -> str:
        lines = []
        mc = report.monte_carlo
        if mc.sharpe_95ci[1] > 0:
            lines.append(f"蒙特卡洛: 均值Sharpe={mc.mean_sharpe:.2f}, 95%CI=[{mc.sharpe_95ci[0]:.2f}, {mc.sharpe_95ci[1]:.2f}]")

        cscv = report.cscv
        if cscv.overfit_probability > 0:
            lines.append(f"CSCV: PBO={cscv.overfit_probability:.2%}, {'通过' if cscv.is_robust else '警告: 过拟合风险'}")

        if report.deflated_sharpe != 0:
            lines.append(f"Deflated Sharpe: {report.deflated_sharpe:.2f}")

        return "\n".join(lines)
```

```python
# robustness/__init__.py
from robustness.robustness_analyzer import RobustnessAnalyzer, RobustnessReport
from robustness.monte_carlo import MonteCarloSim, MCSimulationResult
from robustness.cscv import CSCVAnalyzer, CSCVResult
from robustness.sensitivity import SensitivityAnalyzer, SensitivityResult
from robustness.statistics import deflated_sharpe_ratio, probability_of_backtest_overfit

__all__ = [
    'RobustnessAnalyzer', 'RobustnessReport',
    'MonteCarloSim', 'MCSimulationResult',
    'CSCVAnalyzer', 'CSCVResult',
    'SensitivityAnalyzer', 'SensitivityResult',
    'deflated_sharpe_ratio', 'probability_of_backtest_overfit',
]
```

---

### Task 10: 集成到回测导出流程

**Files:**
- Modify: `backtesting/run_daily_rotation.py`

- [ ] **Step 1: 在 `_export_results` 末尾添加稳健性检验输出**

```python
# 在 _export_results 末尾添加:

# ── 7. 稳健性检验
from robustness.robustness_analyzer import RobustnessAnalyzer

robust = RobustnessAnalyzer(analyzer)
report = robust.run_all()

# 保存稳健性报告
robust_data = {
    'monte_carlo': {
        'mean_sharpe': report.monte_carlo.mean_sharpe,
        'sharpe_95ci_low': report.monte_carlo.sharpe_95ci[0],
        'sharpe_95ci_high': report.monte_carlo.sharpe_95ci[1],
        'mean_max_dd': report.monte_carlo.mean_max_dd,
        'max_dd_95ci_low': report.monte_carlo.max_dd_95ci[0],
        'max_dd_95ci_high': report.monte_carlo.max_dd_95ci[1],
    },
    'cscv': {
        'overfit_probability': report.cscv.overfit_probability,
        'rank_decay': report.cscv.rank_decay,
        'is_robust': report.cscv.is_robust,
    },
    'deflated_sharpe': report.deflated_sharpe,
    'summary': report.summary,
}
with open(out_dir / 'robustness.json', 'w', encoding='utf-8') as f:
    json.dump(robust_data, f, indent=2, ensure_ascii=False, default=str)
```

---

### Task 11 (Phase 2): Optuna 后置敏感性筛选

**Files:**
- Modify: `optimization/optuna/run_daily_rotation_optimization.py`

Phase 2 独立实施。在 Optuna 优化完成后添加函数：

```python
def _select_by_robustness(top_params: List[dict], base_config,
                          start_date, end_date, cache_dir) -> dict:
    """从 Top 5 参数中，按 Sharpe × 0.6 + 稳定性 × 0.4 选出最优"""
    from robustness.sensitivity import SensitivityAnalyzer
    from backtesting.run_daily_rotation import run

    def engine_factory(params):
        config = _params_to_config(params, base_config)
        _, results = run(start_date, end_date, config=config,
                        verbose=False, cache_dir=cache_dir)
        if results:
            equity = [r.total_asset for r in results]
            analyzer = PerformanceAnalyzer(
                trades=[], initial_capital=config.initial_capital,
                equity_curve=equity, periods_per_year=252,
            )
            metrics = analyzer.calculate_metrics()
            return metrics
        return {'sharpe_ratio': 0.0}

    sa = SensitivityAnalyzer()
    best_params = None
    best_score = -float('inf')

    for params in top_params:
        result = sa.run(params, engine_factory)
        # 从Trial中已有sharpe（在params中附带）
        sharpe_norm = (params.get('_sharpe', 0) - min_s) / (max_s - min_s)
        score = sharpe_norm * 0.6 + result.overall_stability_score * 0.4

        if score > best_score:
            best_score = score
            best_params = params

    return best_params
```

---

## 执行顺序

```
Task 1 (CostModel)      ─┐
Task 2 (MarketConstraints)┼─ 并行
Task 3 (PerfAnalyzer扩展) ─┘
        │
Task 4 (引擎集成)  ← 依赖 Task 1, 2
        │
Task 5 (蒙特卡洛)  ─┐
Task 6 (CSCV)       ├─ 并行
Task 7 (敏感性)     ├─ 并行
Task 8 (统计检验)   ─┘
        │
Task 9 (门面)  ← 依赖 Task 5-8
        │
Task 10 (导出集成) ← 依赖 Task 3, 9
        │
Task 11 (Phase 2) ← 独立，Phase 1 完成后做
```
