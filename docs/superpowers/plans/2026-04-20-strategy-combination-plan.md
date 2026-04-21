# 多策略综合评分量化选股系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现多策略综合评分系统，综合MACD、MA、RSI、KDJ、成交量五个维度评分，选取最优5只股票

**Architecture:** 新建CompositeScorer类计算综合评分，修改StockSelector支持综合评分模式，与原策略轮动系统并行运行对比

**Tech Stack:** Python, pandas, numpy, back_testing框架

---

## 文件结构

```
back_testing/
├── signal_scorer.py              # 修改：改进RSI/KDJ信号计算
├── composite_scorer.py          # 新增：综合评分器
├── composite_rotator.py          # 新增：综合评分版主控制器
└── run_composite_backtest.py     # 新增：综合评分版回测入口
```

---

## Task 1: 改进SignalScorer的RSI/KDJ信号计算

**Files:**
- Modify: `back_testing/signal_scorer.py`
- Test: `tests/back_testing/test_signal_scorer.py`

- [ ] **Step 1: 编写改进的RSI信号测试**

```python
# tests/back_testing/test_signal_scorer.py 新增测试

def test_rsi_improved_signal():
    """RSI改进信号：超卖程度 + 反弹动量"""
    scorer = SignalScorer()
    # 模拟RSI数据：超卖后反弹
    df = pd.DataFrame({
        'rsi1': [20, 25, 30, 35, 40],  # 从超卖区域反弹
        'rsi1_prev': [25, 20, 25, 30, 35]  # 前一天RSI
    })
    scores = scorer.calculate_rsi_improved(df)
    # 最低点RSI=20时应该最强（超卖+开始反弹）
    assert scores.iloc[0] > scores.iloc[2]  # 反弹中

def test_kdj_improved_signal():
    """KDJ改进信号：超卖程度 + 反弹动量"""
    scorer = SignalScorer()
    df = pd.DataFrame({
        'KDJ_J': [5, 10, 20, 50, 80],  # 从超卖区域反弹
        'KDJ_J_prev': [10, 5, 10, 20, 50]
    })
    scores = scorer.calculate_kdj_improved(df)
    # J=5时最强（超卖最严重）
    assert scores.iloc[0] > scores.iloc[1]
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/back_testing/test_signal_scorer.py -v`
Expected: FAIL - method not defined

- [ ] **Step 3: 修改SignalScorer添加改进的信号计算**

```python
# back_testing/signal_scorer.py 新增方法

def calculate_rsi_improved(self, df: pd.DataFrame) -> pd.Series:
    """
    RSI改进信号强度：超卖程度 + 反弹动量
    - 超卖程度：RSI < 30时，越低越强
    - 反弹动量：RSI转头向上的幅度
    """
    rsi = df['rsi1'].fillna(50)
    rsi_prev = df['rsi1'].shift(1).fillna(50)

    # 超卖程度：RSI < 30时越低越强
    oversold = (30 - rsi.clip(0, 30)) / 30  # 0-1之间，30时为0，0时为1

    # 反弹动量：RSI上升幅度
    momentum = (rsi - rsi_prev).clip(0, 100) / 100  # 0-1之间

    # 综合得分：超卖程度×0.6 + 反弹动量×0.4
    strength = (oversold * 0.6 + momentum * 0.4) * 100
    return strength.clip(0, 100)

def calculate_kdj_improved(self, df: pd.DataFrame) -> pd.Series:
    """
    KDJ改进信号强度：超卖程度 + 反弹动量
    """
    j = df['KDJ_J'].fillna(50)
    j_prev = df['KDJ_J'].shift(1).fillna(50)

    # 超卖程度：J < 20时越低越强
    oversold = (20 - j.clip(0, 20)) / 20  # 0-1之间

    # 反弹动量：J值上升幅度
    momentum = (j - j_prev).clip(0, 100) / 100

    strength = (oversold * 0.6 + momentum * 0.4) * 100
    return strength.clip(0, 100)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/back_testing/test_signal_scorer.py -v`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
git add back_testing/signal_scorer.py tests/back_testing/test_signal_scorer.py
git commit -m "feat: improve RSI and KDJ signal strength calculation"
```

---

## Task 2: 实现CompositeScorer综合评分器

**Files:**
- Create: `back_testing/composite_scorer.py`
- Test: `tests/back_testing/test_composite_scorer.py`

- [ ] **Step 1: 编写综合评分器测试**

```python
# tests/back_testing/test_composite_scorer.py
import pytest
from back_testing.composite_scorer import CompositeScorer
import pandas as pd
import numpy as np

def test_composite_score_calculation():
    """测试综合评分计算"""
    scorer = CompositeScorer()

    # 模拟一只股票的多维度数据
    df = pd.DataFrame({
        'MACD_DIF': [1.5, 0.8, 0.2],
        'MACD_DEA': [1.0, 0.5, 0.1],
        'MA_5': [100, 102, 105],
        'MA_20': [98, 99, 100],
        'rsi1': [30, 45, 60],
        'KDJ_J': [10, 30, 70],
        '量比': [2.0, 1.5, 1.0],
    })

    score = scorer.calculate_composite_score(df)

    # 返回应该是0-100之间的分数
    assert 0 <= score <= 100
    assert isinstance(score, float)

def test_weight_distribution():
    """测试权重总和为1"""
    scorer = CompositeScorer()
    total = (scorer.weights['macd'] + scorer.weights['ma'] +
              scorer.weights['rsi'] + scorer.weights['kdj'] +
              scorer.weights['volume'])
    assert abs(total - 1.0) < 0.001
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/back_testing/test_composite_scorer.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: 实现CompositeScorer**

```python
# back_testing/composite_scorer.py
import pandas as pd
import numpy as np
from back_testing.signal_scorer import SignalScorer

class CompositeScorer:
    """
    综合评分器：计算多策略综合评分

    评分维度：
    - MACD (35%): DIF-DEA差值，趋势强度
    - MA (20%): 均线开口，多头排列
    - RSI (15%): 超卖程度+反弹动量
    - KDJ (15%): J值超卖+反弹动量
    - 成交量 (15%): 量比，放量程度
    """

    def __init__(self, weights=None):
        self.signal_scorer = SignalScorer()

        # 默认权重配置（趋势策略权重更高）
        self.weights = weights or {
            'macd': 0.35,
            'ma': 0.20,
            'rsi': 0.15,
            'kdj': 0.15,
            'volume': 0.15
        }

    def calculate_macd_score(self, df: pd.DataFrame) -> pd.Series:
        """MACD评分：DIF-DEA差值归一化"""
        dif = df['MACD_DIF'].fillna(0)
        dea = df['MACD_DEA'].fillna(0)
        diff = dif - dea

        # 归一化到0-100：使用tanh压缩处理极端值
        score = (np.tanh(diff * 0.1) + 1) * 50  # 映射到0-100
        return score.clip(0, 100)

    def calculate_ma_score(self, df: pd.DataFrame) -> pd.Series:
        """MA评分：均线开口角度 + 多头排列"""
        ma5 = df['MA_5'].fillna(0)
        ma20 = df['MA_20'].fillna(1)
        ma60_col = df.get('MA_60', ma20)

        # 开口角度
        diff = ma5 - ma20
        angle = (diff / ma20 * 100).clip(-100, 100)

        # 多头排列：MA5 > MA20 > MA60
        bullish = ((ma5 > ma20) & (ma20 > ma60_col)).astype(int) * 30

        score = (angle + bullish).clip(0, 100)
        return score

    def calculate_rsi_score(self, df: pd.DataFrame) -> pd.Series:
        """RSI评分：使用改进的RSI信号"""
        return self.signal_scorer.calculate_rsi_improved(df)

    def calculate_kdj_score(self, df: pd.DataFrame) -> pd.Series:
        """KDJ评分：使用改进的KDJ信号"""
        return self.signal_scorer.calculate_kdj_improved(df)

    def calculate_volume_score(self, df: pd.DataFrame) -> pd.Series:
        """成交量评分：量比归一化"""
        vol_ratio = df['量比'].fillna(1)
        # 量比2.0=40分，量比5.0=100分
        score = (vol_ratio * 20).clip(0, 100)
        return score

    def calculate_composite_score(self, df: pd.DataFrame) -> float:
        """
        计算综合评分（单只股票）

        Returns:
            float: 0-100的综合评分
        """
        scores = {
            'macd': self.calculate_macd_score(df).mean(),
            'ma': self.calculate_ma_score(df).mean(),
            'rsi': self.calculate_rsi_score(df).mean(),
            'kdj': self.calculate_kdj_score(df).mean(),
            'volume': self.calculate_volume_score(df).mean(),
        }

        # 加权求和
        composite = sum(scores[k] * self.weights[k] for k in scores)
        return composite

    def calculate_composite_scores(self, df: pd.DataFrame) -> pd.Series:
        """
        计算每行的综合评分

        Returns:
            pd.Series: 每行的综合评分
        """
        macd_s = self.calculate_macd_score(df)
        ma_s = self.calculate_ma_score(df)
        rsi_s = self.calculate_rsi_score(df)
        kdj_s = self.calculate_kdj_score(df)
        volume_s = self.calculate_volume_score(df)

        composite = (
            macd_s * self.weights['macd'] +
            ma_s * self.weights['ma'] +
            rsi_s * self.weights['rsi'] +
            kdj_s * self.weights['kdj'] +
            volume_s * self.weights['volume']
        )

        return composite.clip(0, 100)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/back_testing/test_composite_scorer.py -v`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
git add back_testing/composite_scorer.py tests/back_testing/test_composite_scorer.py
git commit -m "feat: add composite scorer for multi-strategy evaluation"
```

---

## Task 3: 实现CompositeSelector综合选股器

**Files:**
- Create: `back_testing/composite_selector.py`
- Test: `tests/back_testing/test_composite_selector.py`

- [ ] **Step 1: 编写测试**

```python
# tests/back_testing/test_composite_selector.py
import pytest
from back_testing.composite_selector import CompositeSelector
import pandas as pd

def test_select_top_stocks():
    """测试选取综合评分最高的股票"""
    selector = CompositeSelector(
        data_path='D:/workspace/code/mine/quant/data/metadata/daily_ycz'
    )
    selected = selector.select_top_stocks(n=3, date='2024-01-15')
    assert len(selected) <= 3
    assert all(isinstance(code, str) for code in selected)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/back_testing/test_composite_selector.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: 实现CompositeSelector**

```python
# back_testing/composite_selector.py
import pandas as pd
import numpy as np
import os
from back_testing.composite_scorer import CompositeScorer

class CompositeSelector:
    """
    综合选股器：使用多策略综合评分从全市场筛选股票
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scorer = CompositeScorer()

    def get_all_stock_codes(self) -> list:
        """获取所有股票代码"""
        files = os.listdir(self.data_path)
        stock_codes = [f.replace('.csv', '') for f in files
                      if f.endswith('.csv') and not f.startswith('index')]
        return stock_codes

    def calculate_stock_score(self, stock_code: str, date: pd.Timestamp) -> tuple:
        """
        计算单只股票的综合评分

        Returns:
            tuple: (stock_code, composite_score)
        """
        file_path = f"{self.data_path}\\{stock_code}.csv"
        try:
            df = pd.read_csv(file_path, encoding='gbk')
            df['交易日期'] = pd.to_datetime(df['交易日期'])
            df = df.sort_values('交易日期')

            # 筛选到指定日期的数据
            df = df[df['交易日期'] <= date]
            if len(df) < 5:  # 需要足够的历史数据
                return (stock_code, 0)

            # 取最后一行计算综合评分
            latest = df.iloc[-1:]
            score = self.scorer.calculate_composite_scores(latest).iloc[0]
            return (stock_code, score)
        except Exception:
            return (stock_code, 0)

    def select_top_stocks(self, n: int = 5, date: str = None) -> list:
        """
        选取综合评分最高的N只股票

        Args:
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
        print(f"正在评估 {len(all_codes)} 只股票的综合评分...")

        scores = []
        for code in all_codes:
            stock_code, score = self.calculate_stock_score(code, date)
            scores.append((stock_code, score))

        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 返回前N只
        selected = [code for code, score in scores[:n]]
        print(f"选取结果: {selected}")
        return selected
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/back_testing/test_composite_selector.py -v`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
git add back_testing/composite_selector.py tests/back_testing/test_composite_selector.py
git commit -m "feat: add composite selector for multi-strategy stock screening"
```

---

## Task 4: 实现CompositeRotator主控制器

**Files:**
- Create: `back_testing/composite_rotator.py`

- [ ] **Step 1: 实现CompositeRotator**

```python
# back_testing/composite_rotator.py
import pandas as pd
import numpy as np
from back_testing.composite_selector import CompositeSelector
from back_testing.composite_scorer import CompositeScorer

class CompositeRotator:
    """
    综合评分轮动主控制器

    每周流程：
    1. 使用CompositeSelector对全市场股票计算综合评分
    2. 选取综合评分最高的5只股票
    3. 等权20%持仓
    """

    def __init__(self, data_path: str, initial_capital: float = 1000000.0,
                 n_stocks: int = 5):
        self.data_path = data_path
        self.initial_capital = initial_capital
        self.n_stocks = n_stocks
        self.per_stock_capital = initial_capital / n_stocks

        self.composite_selector = CompositeSelector(data_path)
        self.composite_scorer = CompositeScorer()

        self.current_stocks = []
        self.current_positions = {}

    def select_stocks(self, date: pd.Timestamp) -> list:
        """选取综合评分最高的N只股票"""
        print(f"\n使用综合评分策略筛选股票...")

        selected = self.composite_selector.select_top_stocks(
            n=self.n_stocks,
            date=date
        )

        self.current_stocks = selected
        return selected

    def rebalance(self, date: pd.Timestamp) -> dict:
        """执行调仓"""
        rebalance_detail = {
            'date': date,
            'sell_stocks': [],
            'buy_stocks': [],
            'strategy': 'CompositeScorer'
        }

        # 卖出不在新持仓列表中的股票
        for code in list(self.current_positions.keys()):
            if code not in self.current_stocks:
                rebalance_detail['sell_stocks'].append(code)
                del self.current_positions[code]

        # 买入新持仓股票
        for code in self.current_stocks:
            if code not in self.current_positions:
                rebalance_detail['buy_stocks'].append(code)
                self.current_positions[code] = {
                    'shares': 0,
                    'buy_price': 0
                }

        return rebalance_detail

    def run_weekly(self, date: pd.Timestamp) -> dict:
        """执行每周流程"""
        # 筛选股票
        self.select_stocks(date)

        # 执行调仓
        rebalance = self.rebalance(date)

        return {
            'date': date,
            'strategy': 'CompositeScorer',
            'stocks': self.current_stocks,
            'rebalance': rebalance
        }
```

- [ ] **Step 2: 验证import**

```bash
python -c "from back_testing.composite_rotator import CompositeRotator; print('Import OK')"
```

- [ ] **Step 3: 提交代码**

```bash
git add back_testing/composite_rotator.py
git commit -m "feat: add composite rotator main controller"
```

---

## Task 5: 实现CompositeRotator回测入口

**Files:**
- Create: `back_testing/run_composite_backtest.py`

- [ ] **Step 1: 实现回测入口**

```python
# back_testing/run_composite_backtest.py
"""
多策略综合评分量化选股系统 - 回测入口
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from back_testing.composite_rotator import CompositeRotator

DATA_PATH = r'D:\workspace\code\mine\quant\data\metadata\daily_ycz'
INITIAL_CAPITAL = 1000000.0
N_STOCKS = 5


def get_trading_fridays(start_date: str, end_date: str) -> list:
    """获取回测区间内所有周五"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    fridays = []
    current = start
    while current.weekday() != 4:
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
        hist = df[df['交易日期'] <= date]
        if len(hist) == 0:
            return None
        return hist.iloc[-1]['后复权价']
    except Exception:
        return None


def run_backtest(start_date: str, end_date: str, initial_capital: float = INITIAL_CAPITAL):
    """运行综合评分策略回测"""
    print("=" * 60)
    print("多策略综合评分量化选股系统 - 回测")
    print("=" * 60)
    print(f"回测区间: {start_date} ~ {end_date}")
    print(f"初始资金: {initial_capital:,.2f}")
    print(f"持仓数量: {N_STOCKS}")
    print("=" * 60)

    rotator = CompositeRotator(
        data_path=DATA_PATH,
        initial_capital=initial_capital,
        n_stocks=N_STOCKS
    )

    fridays = get_trading_fridays(start_date, end_date)
    print(f"\n调仓日数量: {len(fridays)}")

    portfolio_value = initial_capital
    weekly_results = []

    for i, friday in enumerate(fridays):
        print(f"\n{'='*60}")
        print(f"第 {i+1}/{len(fridays)} 周: {friday.strftime('%Y-%m-%d')} (周五)")
        print("=" * 60)

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

        portfolio_value = sum(stock_values.values())

        weekly_results.append({
            'date': friday,
            'stocks': result['stocks'],
            'portfolio_value': portfolio_value,
            'return': (portfolio_value - initial_capital) / initial_capital
        })

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

    days = (fridays[-1] - fridays[0]).days if len(fridays) > 1 else 1
    years = days / 365
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    df_weeks['peak'] = df_weeks['portfolio_value'].cummax()
    df_weeks['drawdown'] = (df_weeks['peak'] - df_weeks['portfolio_value']) / df_weeks['peak']
    max_drawdown = df_weeks['drawdown'].max()

    print(f"回测区间: {fridays[0].strftime('%Y-%m-%d')} ~ {fridays[-1].strftime('%Y-%m-%d')}")
    print(f"总收益率: {total_return:.2%}")
    print(f"年化收益率: {annual_return:.2%}")
    print(f"最大回撤: {max_drawdown:.2%}")
    print(f"调仓次数: {len(fridays)}")

    output_path = f"back_testing/results/composite_{start_date}_{end_date}.csv"
    df_weeks.to_csv(output_path, index=False)
    print(f"\n周报已保存: {output_path}")

    return df_weeks


def main():
    parser = argparse.ArgumentParser(description='多策略综合评分量化选股系统')
    parser.add_argument('--start', default='2024-01-01', help='回测开始日期')
    parser.add_argument('--end', default='2025-04-18', help='回测结束日期')
    parser.add_argument('--capital', type=float, default=1000000.0, help='初始资金')
    args = parser.parse_args()

    run_backtest(args.start, args.end, args.capital)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 提交代码**

```bash
git add back_testing/run_composite_backtest.py
git commit -m "feat: add composite backtest entry point"
```

---

## Task 6: 集成测试与对比回测

- [ ] **Step 1: 运行单周测试验证系统**

```bash
python -c "
from back_testing.composite_rotator import CompositeRotator
import pandas as pd

rotator = CompositeRotator(
    data_path='D:/workspace/code/mine/quant/data/metadata/daily_ycz',
    initial_capital=1000000.0,
    n_stocks=5
)

result = rotator.run_weekly(pd.Timestamp('2024-01-19'))
print(f'选股: {result[\"stocks\"]}')
print('综合评分系统集成测试 PASSED')
"
```

- [ ] **Step 2: 运行短期回测对比（如2024年）**

```bash
python -m back_testing.run_composite_backtest --start 2024-01-01 --end 2024-12-31
```

- [ ] **Step 3: 对比两种策略结果**
