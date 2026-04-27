# Multi-Factor Stock Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a multi-factor stock selection system using existing data factors.

**Architecture:** Modular design with factor utilities, factor scoring, and multi-factor selector. Replaces the existing CompositeSelector with a more flexible multi-factor approach.

**Tech Stack:** Python, pandas, numpy

---

## Task 1: Create Factor Utilities

**Files:**
- Create: `back_testing/factors/factor_utils.py`
- Create: `back_testing/factors/__init__.py`
- Test: `tests/back_testing/test_factor_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/back_testing/test_factor_utils.py
import pytest
import pandas as pd
import numpy as np
from back_testing.factors.factor_utils import FactorProcessor

def test_rank_percentile():
    """测试排名百分位计算"""
    data = pd.Series([10, 20, 30, 40, 50])
    result = FactorProcessor.rank_percentile(data)
    assert list(result) == [0.0, 0.25, 0.5, 0.75, 1.0]

def test_z_score():
    """测试Z-score标准化"""
    data = pd.Series([10, 20, 30, 40, 50])
    result = FactorProcessor.z_score(data)
    assert abs(result.mean()) < 1e-10  # 均值接近0
    assert abs(result.std() - 1.0) < 1e-10  # 标准差接近1

def test_winsorize():
    """测试去极值"""
    data = pd.Series([-100, 0, 10, 20, 100])
    result = FactorProcessor.winsorize(data, lower=0.05, upper=0.95)
    assert result.min() > -50  # 被截断
    assert result.max() < 50   # 被截断
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/back_testing/test_factor_utils.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: Write implementation**

```python
# back_testing/factors/__init__.py
from .factor_utils import FactorProcessor

# back_testing/factors/factor_utils.py
import pandas as pd
import numpy as np
from typing import Optional

class FactorProcessor:
    """因子处理器：标准化、去极值、排名"""

    @staticmethod
    def rank_percentile(series: pd.Series, ascending: bool = True) -> pd.Series:
        """
        计算排名百分位

        Args:
            series: 输入因子值
            ascending: True表示值越小得分越高（如PB），False表示值越大得分越高（如ROE）

        Returns:
            pd.Series: 排名百分位，0-1之间
        """
        if ascending:
            # 值越小排名越靠前（低估值好）
            return series.rank(pct=True, ascending=True)
        else:
            # 值越大排名越靠前（高ROE好）
            return series.rank(pct=True, ascending=False)

    @staticmethod
    def z_score(series: pd.Series) -> pd.Series:
        """
        Z-score标准化

        Returns:
            pd.Series: 标准化后的值，均值0，标准差1
        """
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(0, index=series.index)
        return (series - mean) / std

    @staticmethod
    def winsorize(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
        """
        去极值（截断到指定分位数）

        Args:
            series: 输入数据
            lower: 下界分位数
            upper: 上界分位数

        Returns:
            pd.Series: 去极值后的数据
        """
        lower_val = series.quantile(lower)
        upper_val = series.quantile(upper)
        return series.clip(lower=lower_val, upper=upper_val)

    @staticmethod
    def neutralize(series: pd.Series, market_cap: pd.Series) -> pd.Series:
        """
        市值中性化：回归残差

        去除因子中与市值的相关性，使因子更纯净
        """
        from numpy.linalg import lstsq

        # 简单线性回归：factor = a * log(market_cap) + residual
        X = np.log(market_cap.values.reshape(-1, 1))
        y = series.values
        coef, _, _, _ = lstsq(X, y, rcond=None)
        predicted = (X * coef).flatten()
        residual = y - predicted
        return pd.Series(residual, index=series.index)

    @staticmethod
    def process_factor(
        series: pd.Series,
        method: str = 'rank',
        ascending: bool = True,
        winsorize_lower: float = 0.05,
        winsorize_upper: float = 0.95
    ) -> pd.Series:
        """
        因子预处理流水线

        Args:
            series: 原始因子值
            method: 'rank' 或 'zscore'
            ascending: 值越小得分越高时为True
            winsorize_lower: 去极值下界
            winsorize_upper: 去极值上界

        Returns:
            pd.Series: 处理后的因子分数，0-1之间
        """
        # Step 1: 去极值
        result = FactorProcessor.winsorize(series, winsorize_lower, winsorize_upper)

        # Step 2: 标准化
        if method == 'zscore':
            result = FactorProcessor.z_score(result)

        # Step 3: 排名百分位
        result = FactorProcessor.rank_percentile(result, ascending=ascending)

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/back_testing/test_factor_utils.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add back_testing/factors/ tests/back_testing/test_factor_utils.py
git commit -m "feat: add FactorProcessor for factor standardization and ranking"
```

---

## Task 2: Create Multi-Factor Selector

**Files:**
- Create: `back_testing/selectors/multi_factor_selector.py`
- Test: `tests/back_testing/test_multi_factor_selector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/back_testing/test_multi_factor_selector.py
import pytest
import pandas as pd
from back_testing.selectors.multi_factor_selector import MultiFactorSelector

def test_calculate_factor_scores():
    """测试因子评分计算"""
    # 创建测试数据
    data = pd.DataFrame({
        'PB': [1.0, 2.0, 3.0, 4.0, 5.0],
        'ROE': [10.0, 20.0, 15.0, 25.0, 5.0],
    }, index=['s1', 's2', 's3', 's4', 's5'])

    # PB越低越好，ROE越高越好
    weights = {'PB': 0.5, 'ROE': 0.5}
    directions = {'PB': -1, 'ROE': 1}

    selector = MultiFactorSelector(weights, directions)
    scores = selector.calculate_factor_scores(data)

    # s1的PB最低(0分)，ROE中等(0.5分)，综合0.25
    # s4的PB最高(1分)，ROE最高(1分)，综合1.0
    assert scores['s1'] < scores['s4']

def test_select_top_stocks():
    """测试选股"""
    data = pd.DataFrame({
        'PB': [1.0, 2.0, 3.0, 4.0, 5.0],
        'ROE': [10.0, 20.0, 15.0, 25.0, 5.0],
    }, index=['sh600001', 'sh600002', 'sh600003', 'sh600004', 'sh600005'])

    weights = {'PB': 0.5, 'ROE': 0.5}
    directions = {'PB': -1, 'ROE': 1}

    selector = MultiFactorSelector(weights, directions)
    result = selector.select_top_stocks(data, n=3)

    assert len(result) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/back_testing/test_multi_factor_selector.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: Write implementation**

```python
# back_testing/selectors/multi_factor_selector.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from back_testing.factors.factor_utils import FactorProcessor

class MultiFactorSelector:
    """
    多因子选股器

    使用多个因子加权评分选择股票
    """

    def __init__(
        self,
        weights: Dict[str, float],
        directions: Dict[str, int],
        method: str = 'rank'
    ):
        """
        Args:
            weights: 因子权重字典，如 {'PB': 0.3, 'ROE': 0.2, ...}
            directions: 因子方向，1表示越大越好，-1表示越小越好
            method: 'rank' 或 'zscore' 标准化方法
        """
        self.weights = weights
        self.directions = directions
        self.method = method

        # 验证权重和为1
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            # 归一化
            self.weights = {k: v / total for k, v in weights.items()}

    def calculate_factor_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        计算各股票的因子综合得分

        Args:
            data: 股票因子数据，index为股票代码，columns为因子名

        Returns:
            pd.Series: 各股票的综合得分，0-1之间
        """
        scores = pd.Series(0.0, index=data.index)

        for factor, weight in self.weights.items():
            if factor not in data.columns:
                continue

            # 获取因子方向
            ascending = self.directions.get(factor, 1) == -1

            # 因子处理并计算排名百分位
            factor_values = data[factor].astype(float)
            processed = FactorProcessor.process_factor(
                factor_values,
                method=self.method,
                ascending=ascending
            )

            # 加权累加
            scores += processed * weight

        return scores

    def select_top_stocks(
        self,
        data: pd.DataFrame,
        n: int = 5,
        excluded: List[str] = None
    ) -> List[str]:
        """
        选取综合得分最高的N只股票

        Args:
            data: 股票因子数据
            n: 选取数量
            excluded: 排除的股票列表

        Returns:
            List[str]: 选中的股票代码列表
        """
        # 计算得分
        scores = self.calculate_factor_scores(data)

        # 排除指定股票
        if excluded:
            scores = scores.drop(excluded, errors='ignore')

        # 按得分排序
        scores = scores.sort_values(ascending=False)

        # 返回前N只
        return scores.head(n).index.tolist()

    def get_factor_contribution(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        获取各因子对综合得分的贡献

        Returns:
            DataFrame: 各股票各因子的贡献分数
        """
        contributions = pd.DataFrame(index=data.index)

        for factor, weight in self.weights.items():
            if factor not in data.columns:
                continue

            ascending = self.directions.get(factor, 1) == -1
            factor_values = data[factor].astype(float)
            processed = FactorProcessor.process_factor(
                factor_values,
                method=self.method,
                ascending=ascending
            )
            contributions[factor] = processed * weight

        contributions['total'] = contributions.sum(axis=1)
        return contributions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/back_testing/test_multi_factor_selector.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add back_testing/selectors/multi_factor_selector.py tests/back_testing/test_multi_factor_selector.py
git commit -m "feat: add MultiFactorSelector for multi-factor stock selection"
```

---

## Task 3: Create Default Factor Configuration

**Files:**
- Create: `back_testing/factors/factor_config.py`

- [ ] **Step 1: Write default configuration**

```python
# back_testing/factors/factor_config.py
"""
多因子模型默认配置

因子权重和方向配置
"""

# 默认因子配置
DEFAULT_FACTOR_CONFIG = {
    # 估值因子（低估值好）
    'PB': {
        'weight': 0.15,
        'direction': -1,
        'description': '市净率，越低越好'
    },
    'PE_TTM': {
        'weight': 0.10,
        'direction': -1,
        'description': '市盈率TTM，越低越好'
    },
    'PS_TTM': {
        'weight': 0.05,
        'direction': -1,
        'description': '市销率TTM，越低越好'
    },

    # 动量因子（强势好）
    'RSI_1': {
        'weight': 0.15,
        'direction': 1,
        'description': 'RSI短期，偏强好'
    },
    'KDJ_K': {
        'weight': 0.05,
        'direction': 1,
        'description': 'KDJ随机K，强势好'
    },

    # 趋势因子（趋势向上好）
    'MA_5': {
        'weight': 0.05,
        'direction': 1,
        'description': '5日均线偏多'
    },
    'MA_20': {
        'weight': 0.05,
        'direction': 1,
        'description': '20日均线偏多'
    },

    # 交易因子（活跃但不过度）
    'TURNOVER': {
        'weight': 0.10,
        'direction': 1,
        'description': '换手率，活跃好'
    },
    'VOLUME_RATIO': {
        'weight': 0.05,
        'direction': 1,
        'description': '量比，放量好'
    },

    # 波动因子（低波动稳健）
    'AMPLITUDE': {
        'weight': 0.05,
        'direction': -1,
        'description': '振幅，低波动好'
    },
}

def get_factor_weights(config: dict = None) -> dict:
    """获取因子权重"""
    if config is None:
        config = DEFAULT_FACTOR_CONFIG
    return {k: v['weight'] for k, v in config.items()}

def get_factor_directions(config: dict = None) -> dict:
    """获取因子方向"""
    if config is None:
        config = DEFAULT_FACTOR_CONFIG
    return {k: v['direction'] for k, v in config.items()}
```

- [ ] **Step 2: Commit**

```bash
git add back_testing/factors/factor_config.py
git commit -m "feat: add default factor configuration"
```

---

## Task 4: Create Factor Data Loader

**Files:**
- Create: `back_testing/factors/factor_loader.py`
- Test: `tests/back_testing/test_factor_loader.py`

- [ ] **Step 1: Write implementation**

```python
# back_testing/factors/factor_loader.py
"""
因子数据加载器

从DataProvider获取股票数据，提取所需因子
"""
import pandas as pd
from typing import List, Optional
from back_testing.data.data_provider import DataProvider

class FactorLoader:
    """
    因子数据加载器
    """

    # 可用的因子列名映射
    FACTOR_COLUMNS = {
        'PB': 'PB',
        'PE_TTM': 'PE_TTM',
        'PS_TTM': 'PS_TTM',
        'ROE': 'ROE_TTM',  # 如果数据中有
        'RSI_1': 'RSI_1',
        'KDJ_K': 'KDJ_K',
        'KDJ_D': 'KDJ_D',
        'MA_5': 'MA_5',
        'MA_10': 'MA_10',
        'MA_20': 'MA_20',
        'MA_30': 'MA_30',
        'TURNOVER': 'TURNOVER',
        'VOLUME_RATIO': 'VOLUME_RATIO',
        'AMPLITUDE': 'AMPLITUDE',
    }

    def __init__(self, data_provider: DataProvider = None):
        self.data_provider = data_provider or DataProvider()

    def load_stock_factors(
        self,
        stock_codes: List[str],
        date: pd.Timestamp,
        factors: List[str]
    ) -> pd.DataFrame:
        """
        加载指定股票的因子数据

        Args:
            stock_codes: 股票代码列表
            date: 评分日期
            factors: 需要加载的因子列表

        Returns:
            DataFrame: index为股票代码，columns为因子值
        """
        result_data = {}

        for code in stock_codes:
            try:
                df = self.data_provider.get_stock_data(code, date=date)
                if len(df) == 0:
                    continue

                latest = df.iloc[-1]

                # 提取各因子值
                row = {}
                for factor in factors:
                    col_name = self.FACTOR_COLUMNS.get(factor, factor)
                    if col_name in df.columns:
                        row[factor] = latest[col_name]
                    else:
                        row[factor] = None

                result_data[code] = row

            except Exception:
                continue

        result = pd.DataFrame(result_data).T

        # 填充缺失值
        for col in result.columns:
            if result[col].isna().any():
                # 用中位数填充
                median_val = result[col].median()
                result[col] = result[col].fillna(median_val)

        return result

    def load_all_stock_factors(
        self,
        date: pd.Timestamp,
        factors: List[str]
    ) -> pd.DataFrame:
        """
        加载所有股票的因子数据

        Returns:
            DataFrame: 所有股票的因子数据
        """
        all_codes = self.data_provider.get_all_stock_codes()
        return self.load_stock_factors(all_codes, date, factors)
```

- [ ] **Step 2: Commit**

```bash
git add back_testing/factors/factor_loader.py
git commit -m "feat: add FactorLoader for loading stock factor data"
```

---

## Task 5: Update CompositeRotator to Use Multi-Factor Selector

**Files:**
- Modify: `back_testing/composite_rotator.py`

- [ ] **Step 1: Add import and integrate multi-factor selector**

```python
# 在文件顶部添加
from back_testing.selectors.multi_factor_selector import MultiFactorSelector
from back_testing.factors.factor_config import get_factor_weights, get_factor_directions

# 在 __init__ 中添加
self.factor_weights = get_factor_weights()
self.factor_directions = get_factor_directions()
self.factor_selector = MultiFactorSelector(
    weights=self.factor_weights,
    directions=self.factor_directions
)
```

- [ ] **Step 2: Update run_weekly method to use factor-based selection**

```python
# 修改 run_weekly 中的选股逻辑
# 原来用 CompositeSelector 评分
# 现在用 MultiFactorSelector
```

- [ ] **Step 3: Commit**

```bash
git add back_testing/composite_rotator.py
git commit -m "feat: integrate MultiFactorSelector into CompositeRotator"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | FactorProcessor utilities | factor_utils.py |
| 2 | MultiFactorSelector | multi_factor_selector.py |
| 3 | Default factor config | factor_config.py |
| 4 | FactorLoader | factor_loader.py |
| 5 | Integration | composite_rotator.py |

**Plan complete and saved to `docs/superpowers/plans/2026-04-21-multi-factor-model-plan.md`**
