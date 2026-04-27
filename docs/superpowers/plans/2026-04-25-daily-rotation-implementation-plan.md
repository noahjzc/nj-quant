# 每日全市场轮动回测系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个每日全市场轮动回测系统，支持技术信号过滤 + 多因子排序，每日调仓，可独立运行或作为 GA 适应度函数。

**Architecture:**
- `DailyRotationEngine` 是核心引擎，管理每日流程（获取数据→检查持仓→扫描信号→买卖→记录）
- 信号引擎分两层：SignalFilter（技术指标金叉/死叉过滤）→ SignalRanker（多因子排序）
- `MarketRegime` 每日判断大盘状态，动态调整仓位参数
- 配置类全部可序列化，支持 GA genome 映射

**Tech Stack:** Python 3.12, pandas, numpy, PostgreSQL (via existing DataProvider)

---

## 文件结构

```
back_testing/
├── rotation/
│   ├── __init__.py
│   ├── config.py                    # RotationConfig, MarketRegimeConfig
│   ├── signal_engine/
│   │   ├── __init__.py
│   │   ├── base_signal.py           # SignalType 枚举, BaseSignal
│   │   ├── signal_filter.py         # SignalFilter（第一层过滤）
│   │   └── signal_ranker.py         # SignalRanker（第二层排序）
│   ├── market_regime.py             # 大盘状态判断
│   ├── position_manager.py          # RotationPositionManager（复用 PositionManager 逻辑）
│   ├── trade_executor.py            # TradeExecutor（含交易成本）
│   ├── daily_rotation_engine.py     # DailyRotationEngine 核心引擎
│   └── strategy.py                 # AbstractRotationStrategy 接口
├── backtest/
│   └── run_daily_rotation.py       # 独立运行入口
└── optimization/
    └── rotation_ga_fitness.py       # RotationFitnessEvaluator（GA 适应度）
```

---

## Task 1: 创建 rotation 包结构和配置类

**Files:**
- Create: `back_testing/rotation/__init__.py`
- Create: `back_testing/rotation/config.py`

- [ ] **Step 1: 创建 `back_testing/rotation/__init__.py`**

```python
"""每日全市场轮动回测系统"""
from back_testing.rotation.daily_rotation_engine import DailyRotationEngine
from back_testing.rotation.config import RotationConfig, MarketRegimeConfig

__all__ = ['DailyRotationEngine', 'RotationConfig', 'MarketRegimeConfig']
```

- [ ] **Step 2: 创建 `back_testing/rotation/config.py`**

```python
"""策略配置类"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MarketRegimeParams:
    """单一市场状态的参数"""
    max_total_pct: float = 0.90
    max_position_pct: float = 0.20
    max_positions: int = 5


@dataclass
class MarketRegimeConfig:
    """市场状态动态调节配置"""
    strong_trend_threshold: float = 0.05   # 大盘MA多头阈值（5%）
    weak_trend_threshold: float = -0.03     # 大盘MA空头阈值（-3%）
    high_volatility_threshold: float = 0.03  # 高波动率阈值（3%）
    lookback_period: int = 20               # 大盘动量回溯期
    regime_params: Dict[str, MarketRegimeParams] = field(default_factory=lambda: {
        'strong':  MarketRegimeParams(max_total_pct=1.00, max_position_pct=0.20, max_positions=5),
        'neutral': MarketRegimeParams(max_total_pct=0.60, max_position_pct=0.15, max_positions=4),
        'weak':    MarketRegimeParams(max_total_pct=0.30, max_position_pct=0.10, max_positions=3),
    })


@dataclass
class RotationConfig:
    """每日轮动策略配置"""
    # 基础资金
    initial_capital: float = 1_000_000.0
    # 仓位参数
    max_total_pct: float = 0.90
    max_position_pct: float = 0.20
    max_positions: int = 5
    # 信号配置
    buy_signal_types: List[str] = field(default_factory=lambda: [
        'KDJ_GOLD', 'MACD_GOLD', 'MA_GOLD', 'VOL_GOLD', 'DMI_GOLD',
        'BOLL_BREAK', 'HIGH_BREAK'
    ])
    sell_signal_types: List[str] = field(default_factory=lambda: [
        'KDJ_DEATH', 'MACD_DEATH', 'MA_DEATH', 'VOL_DEATH', 'DMI_DEATH',
        'BOLL_BREAK_DOWN', 'HIGH_BREAK_DOWN'
    ])
    # 排序因子及权重
    rank_factor_weights: Dict[str, float] = field(default_factory=lambda: {
        'RSI_1': 0.20,
        'RET_20': 0.25,
        'VOLUME_RATIO': 0.15,
        'PB': 0.20,
        'PE_TTM': 0.20,
    })
    rank_factor_directions: Dict[str, int] = field(default_factory=lambda: {
        'RSI_1': 1,      # 低RSI=超卖，反弹潜力大
        'RET_20': 1,     # 正动量=上涨趋势
        'VOLUME_RATIO': 1,
        'PB': -1,        # 低估值=价值
        'PE_TTM': -1,
    })
    # 市场状态调节
    market_regime: MarketRegimeConfig = field(default_factory=MarketRegimeConfig)
    # 股票池过滤
    exclude_st: bool = True
    exclude_limit_up: bool = True
    exclude_limit_down: bool = True
    exclude_suspended: bool = True
    # 大盘指数代码（用于市场状态判断）
    benchmark_index: str = 'sh000300'
```

- [ ] **Step 3: 运行测试验证**

```bash
python -c "from back_testing.rotation.config import RotationConfig, MarketRegimeConfig; c=RotationConfig(); print(c.max_positions)"
```
Expected: `5`

- [ ] **Step 4: Commit**

```bash
git add back_testing/rotation/__init__.py back_testing/rotation/config.py
git commit -m "feat(rotation): add rotation package structure and config classes"
```

---

## Task 2: 创建信号引擎 — SignalType 枚举和 BaseSignal

**Files:**
- Create: `back_testing/rotation/signal_engine/__init__.py`
- Create: `back_testing/rotation/signal_engine/base_signal.py`

- [ ] **Step 1: 创建 `back_testing/rotation/signal_engine/__init__.py`**

```python
"""信号引擎"""
from back_testing.rotation.signal_engine.base_signal import SignalType, BaseSignal
from back_testing.rotation.signal_engine.signal_filter import SignalFilter
from back_testing.rotation.signal_engine.signal_ranker import SignalRanker

__all__ = ['SignalType', 'BaseSignal', 'SignalFilter', 'SignalRanker']
```

- [ ] **Step 2: 创建 `back_testing/rotation/signal_engine/base_signal.py`**

```python
"""信号类型枚举和基类"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import pandas as pd


class SignalType(Enum):
    """信号类型枚举"""
    # 买入信号（金叉）
    KDJ_GOLD = 'KDJ_GOLD'
    MACD_GOLD = 'MACD_GOLD'
    MA_GOLD = 'MA_GOLD'
    VOL_GOLD = 'VOL_GOLD'
    DMI_GOLD = 'DMI_GOLD'
    BOLL_BREAK = 'BOLL_BREAK'
    HIGH_BREAK = 'HIGH_BREAK'
    # 卖出信号（死叉）
    KDJ_DEATH = 'KDJ_DEATH'
    MACD_DEATH = 'MACD_DEATH'
    MA_DEATH = 'MA_DEATH'
    VOL_DEATH = 'VOL_DEATH'
    DMI_DEATH = 'DMI_DEATH'
    BOLL_BREAK_DOWN = 'BOLL_BREAK_DOWN'
    HIGH_BREAK_DOWN = 'HIGH_BREAK_DOWN'

    @property
    def is_buy(self) -> bool:
        return self.name.endswith('_GOLD') or self.name == 'BOLL_BREAK' or self.name == 'HIGH_BREAK'

    @property
    def is_sell(self) -> bool:
        return self.name.endswith('_DEATH') or self.name in ('BOLL_BREAK_DOWN', 'HIGH_BREAK_DOWN')


@dataclass
class SignalResult:
    """信号检测结果"""
    signal_type: SignalType
    stock_code: str
    triggered: bool
    strength: float = 0.0  # 信号强度，0-1
    metadata: Optional[dict] = None


class BaseSignal:
    """信号检测基类"""

    def __init__(self, signal_type: SignalType):
        self.signal_type = signal_type

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        """检测信号"""
        raise NotImplementedError

    def _cross_up(self, series1: pd.Series, series2: pd.Series, period: int = 1) -> bool:
        """检测是否从下方穿越上方（金叉）"""
        if len(series1) < period + 1:
            return False
        current = series1.iloc[-1]
        prev = series1.iloc[-period - 1]
        current_cross = current > series2.iloc[-1]
        prev_cross = prev <= series2.iloc[-period - 1]
        return current_cross and prev_cross

    def _cross_down(self, series1: pd.Series, series2: pd.Series, period: int = 1) -> bool:
        """检测是否从上方穿越下方（死叉）"""
        if len(series1) < period + 1:
            return False
        current = series1.iloc[-1]
        prev = series1.iloc[-period - 1]
        current_cross = current < series2.iloc[-1]
        prev_cross = prev >= series2.iloc[-period - 1]
        return current_cross and prev_cross
```

- [ ] **Step 3: 运行测试验证**

```bash
python -c "from back_testing.rotation.signal_engine.base_signal import SignalType, BaseSignal; print(SignalType.KDJ_GOLD.is_buy)"
```
Expected: `True`

- [ ] **Step 4: Commit**

```bash
git add back_testing/rotation/signal_engine/__init__.py back_testing/rotation/signal_engine/base_signal.py
git commit -m "feat(rotation): add signal type enum and base signal class"
```

---

## Task 3: SignalFilter — 第一层信号过滤

**Files:**
- Create: `back_testing/rotation/signal_engine/signal_filter.py`

- [ ] **Step 1: 创建 `back_testing/rotation/signal_engine/signal_filter.py`**

```python
"""第一层信号过滤器 — 技术指标金叉/死叉过滤"""
import pandas as pd
from typing import List, Dict
from back_testing.rotation.signal_engine.base_signal import SignalType, SignalResult, BaseSignal


class KDJGoldSignal(BaseSignal):
    """KDJ 金叉检测"""

    def __init__(self):
        super().__init__(SignalType.KDJ_GOLD)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        k = df['kdj_k']
        d = df['kdj_d']
        triggered = self._cross_up(k, d)
        strength = 1.0 if triggered else 0.0
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=strength,
            metadata={'kdj_k': k.iloc[-1] if not k.empty else None, 'kdj_d': d.iloc[-1] if not d.empty else None}
        )


class KDJDeathSignal(BaseSignal):
    """KDJ 死叉检测"""

    def __init__(self):
        super().__init__(SignalType.KDJ_DEATH)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        k = df['kdj_k']
        d = df['kdj_d']
        triggered = self._cross_down(k, d)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class MACDGoldSignal(BaseSignal):
    """MACD 金叉检测（DIF 上穿 DEA）"""

    def __init__(self):
        super().__init__(SignalType.MACD_GOLD)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        dif = df['macd_dif']
        dea = df['macd_dea']
        triggered = self._cross_up(dif, dea)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class MACDDeathSignal(BaseSignal):
    """MACD 死叉检测（DIF 下穿 DEA）"""

    def __init__(self):
        super().__init__(SignalType.MACD_DEATH)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        dif = df['macd_dif']
        dea = df['macd_dea']
        triggered = self._cross_down(dif, dea)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class MAGoldSignal(BaseSignal):
    """MA 金叉检测（MA5 上穿 MA20）"""

    def __init__(self, fast: str = 'ma_5', slow: str = 'ma_20'):
        super().__init__(SignalType.MA_GOLD)
        self.fast = fast
        self.slow = slow

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if self.fast not in df.columns or self.slow not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        fast_ma = df[self.fast]
        slow_ma = df[self.slow]
        triggered = self._cross_up(fast_ma, slow_ma)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class MADeathSignal(BaseSignal):
    """MA 死叉检测（MA5 下穿 MA20）"""

    def __init__(self, fast: str = 'ma_5', slow: str = 'ma_20'):
        super().__init__(SignalType.MA_DEATH)
        self.fast = fast
        self.slow = slow

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if self.fast not in df.columns or self.slow not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        fast_ma = df[self.fast]
        slow_ma = df[self.slow]
        triggered = self._cross_down(fast_ma, slow_ma)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class VOLGoldSignal(BaseSignal):
    """VOL MA 金叉检测"""

    def __init__(self):
        super().__init__(SignalType.VOL_GOLD)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'volume_ratio' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        vol = df['volume_ratio']
        triggered = self._cross_up(vol, vol.shift(5).rolling(5).mean())
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class BollBreakSignal(BaseSignal):
    """布林带上轨突破"""

    def __init__(self):
        super().__init__(SignalType.BOLL_BREAK)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'close' not in df.columns or 'boll_upper' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        triggered = df['close'].iloc[-1] > df['boll_upper'].iloc[-1] if not df.empty else False
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class BollBreakDownSignal(BaseSignal):
    """布林带下轨突破（卖出）"""

    def __init__(self):
        super().__init__(SignalType.BOLL_BREAK_DOWN)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'close' not in df.columns or 'boll_lower' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        triggered = df['close'].iloc[-1] < df['boll_lower'].iloc[-1] if not df.empty else False
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class SignalFilter:
    """第一层信号过滤器 — 将信号类型名映射到对应检测器"""

    _SIGNAL_MAP = {
        SignalType.KDJ_GOLD: KDJGoldSignal,
        SignalType.KDJ_DEATH: KDJDeathSignal,
        SignalType.MACD_GOLD: MACDGoldSignal,
        SignalType.MACD_DEATH: MACDDeathSignal,
        SignalType.MA_GOLD: MAGoldSignal,
        SignalType.MA_DEATH: MADeathSignal,
        SignalType.VOL_GOLD: VOLGoldSignal,
        SignalType.BOLL_BREAK: BollBreakSignal,
        SignalType.BOLL_BREAK_DOWN: BollBreakDownSignal,
    }

    def __init__(self, signal_types: List[str]):
        """初始化过滤器
        Args:
            signal_types: 信号类型名列表，如 ['KDJ_GOLD', 'MACD_GOLD']
        """
        self.detectors = []
        for name in signal_types:
            try:
                sig_type = SignalType[name]
                detector_cls = self._SIGNAL_MAP.get(sig_type)
                if detector_cls:
                    self.detectors.append(detector_cls())
            except KeyError:
                pass

    def filter_buy(self, df: pd.DataFrame, stock_code: str) -> bool:
        """检查是否有任何买入信号触发"""
        for detector in self.detectors:
            if detector.signal_type.is_buy:
                result = detector.detect(df, stock_code)
                if result.triggered:
                    return True
        return False

    def filter_sell(self, df: pd.DataFrame, stock_code: str) -> bool:
        """检查是否有任何卖出信号触发"""
        for detector in self.detectors:
            if detector.signal_type.is_sell:
                result = detector.detect(df, stock_code)
                if result.triggered:
                    return True
        return False

    def scan_buy_candidates(self, stock_data_dict: Dict[str, pd.DataFrame]) -> List[str]:
        """扫描全市场，返回有买入信号的股票列表
        Args:
            stock_data_dict: {stock_code: df}，df 需包含最新日线数据
        Returns:
            有买入信号的股票代码列表
        """
        candidates = []
        for stock_code, df in stock_data_dict.items():
            if self.filter_buy(df, stock_code):
                candidates.append(stock_code)
        return candidates
```

- [ ] **Step 2: 运行测试验证**

```bash
python -c "from back_testing.rotation.signal_engine.signal_filter import SignalFilter, KDJGoldSignal; f=SignalFilter(['KDJ_GOLD']); print(len(f.detectors))"
```
Expected: `1`

- [ ] **Step 3: Commit**

```bash
git add back_testing/rotation/signal_engine/signal_filter.py
git commit -m "feat(rotation): add SignalFilter for first-layer technical signal filtering"
```

---

## Task 4: SignalRanker — 第二层信号排序

**Files:**
- Create: `back_testing/rotation/signal_engine/signal_ranker.py`

- [ ] **Step 1: 创建 `back_testing/rotation/signal_engine/signal_ranker.py`**

```python
"""第二层信号排序器 — 多因子加权评分"""
import pandas as pd
import numpy as np
from typing import List, Dict
from back_testing.factors.factor_utils import FactorProcessor


class SignalRanker:
    """
    第二层排序器：对候选股按多因子加权评分排序

    使用 zscore 标准化 + 因子方向调整 + 加权求和
    """

    def __init__(self, factor_weights: Dict[str, float], factor_directions: Dict[str, int]):
        """
        Args:
            factor_weights: 因子权重，如 {'RSI_1': 0.2, 'RET_20': 0.25}
            factor_directions: 因子方向，1=越大越好，-1=越小越好
        """
        self.factor_weights = factor_weights
        self.factor_directions = factor_directions
        self._processor = FactorProcessor()

    def rank(self, factor_data: pd.DataFrame, top_n: int = 5) -> List[str]:
        """
        对候选股排序，返回 top_n 只股票代码

        Args:
            factor_data: DataFrame，index=股票代码，columns=因子值
            top_n: 返回前 N 只

        Returns:
            排序后的股票代码列表
        """
        if factor_data.empty:
            return []

        scores = self._calculate_scores(factor_data)
        sorted_codes = scores.sort_values(ascending=False).head(top_n)
        return sorted_codes.index.tolist()

    def _calculate_scores(self, factor_data: pd.DataFrame) -> pd.Series:
        """计算每只股票的加权综合得分"""
        total_weight = sum(self.factor_weights.values())
        if total_weight == 0:
            return pd.Series(0.0, index=factor_data.index)

        composite = pd.Series(0.0, index=factor_data.index)

        for factor, weight in self.factor_weights.items():
            if factor not in factor_data.columns:
                continue

            raw = factor_data[factor].copy()
            # zscore 标准化
            normalized = self._processor.z_score(raw)
            # 方向调整：-1 则反转
            direction = self.factor_directions.get(factor, 1)
            if direction == -1:
                normalized = 1 - normalized
            # 加权
            composite += normalized * weight / total_weight

        return composite
```

- [ ] **Step 2: 运行测试验证**

```bash
python -c "
from back_testing.rotation.signal_engine.signal_ranker import SignalRanker
import pandas as pd
ranker = SignalRanker({'RSI_1': 1.0}, {'RSI_1': 1})
df = pd.DataFrame({'RSI_1': [30, 50, 70]}, index=['sh600000', 'sh600001', 'sh600002'])
result = ranker.rank(df, top_n=2)
print(result)
"
```
Expected: `['sh600000', 'sh600001']` (lower RSI scored higher because direction=1 means lower is better)

- [ ] **Step 3: Commit**

```bash
git add back_testing/rotation/signal_engine/signal_ranker.py
git commit -m "feat(rotation): add SignalRanker for multi-factor ranking"
```

---

## Task 5: 市场状态判断 — MarketRegime

**Files:**
- Create: `back_testing/rotation/market_regime.py`

- [ ] **Step 1: 创建 `back_testing/rotation/market_regime.py`**

```python
"""大盘市场状态判断和动态参数调节"""
import pandas as pd
from typing import Tuple
from back_testing.data.data_provider import DataProvider
from back_testing.rotation.config import MarketRegimeConfig, MarketRegimeParams


class MarketRegime:
    """
    每日大盘状态判断和参数动态调节

    根据大盘 MA 趋势、N 日动量、ATR 波动率判断市场状态，
    返回对应的仓位参数。
    """

    def __init__(self, config: MarketRegimeConfig, data_provider: DataProvider = None):
        self.config = config
        self.data_provider = data_provider or DataProvider()

    def get_regime(self, date: pd.Timestamp, lookback: int = None) -> Tuple[str, MarketRegimeParams]:
        """
        获取当日市场状态和对应参数

        Args:
            date: 评分日期
            lookback: 回溯天数，默认使用 config.lookback_period

        Returns:
            (状态名, 参数)，如 ('strong', MarketRegimeParams(...))
        """
        lookback = lookback or self.config.lookback_period
        index_code = 'sh000300'  # 沪深300

        start_date = (date - pd.Timedelta(days=lookback * 3)).strftime('%Y-%m-%d')
        end_date = date.strftime('%Y-%m-%d')

        index_df = self.data_provider.get_index_data(index_code, start_date=start_date, end_date=end_date)
        if index_df is None or len(index_df) < lookback + 5:
            # 数据不足，默认中性
            return 'neutral', self.config.regime_params['neutral']

        index_df = index_df.sort_index()

        close = index_df['close']
        ma_fast = close.rolling(5).mean()
        ma_slow = close.rolling(20).mean()

        # 大盘趋势：ma5 / ma20 - 1
        trend = (ma_fast.iloc[-1] / ma_slow.iloc[-1] - 1) if not (ma_fast.empty or ma_slow.empty) else 0.0

        # 大盘动量：N日收益率
        if len(close) >= lookback + 1:
            momentum = (close.iloc[-1] / close.iloc[-lookback - 1] - 1)
        else:
            momentum = 0.0

        # ATR（简化：使用日内波幅均值）
        high = index_df['high']
        low = index_df['low']
        tr1 = high - low
        atr = tr1.rolling(14).mean().iloc[-1] if len(tr1) >= 14 else tr1.mean()
        # ATR 相对价格比率
        volatility = atr / close.iloc[-1] if close.iloc[-1] > 0 else 0.0

        # 判断状态
        regime = self._classify_regime(trend, momentum, volatility)

        return regime, self.config.regime_params[regime]

    def _classify_regime(self, trend: float, momentum: float, volatility: float) -> str:
        """根据指标分类市场状态"""
        strong_threshold = self.config.strong_trend_threshold
        weak_threshold = self.config.weak_trend_threshold
        high_vol_threshold = self.config.high_volatility_threshold

        is_strong = trend > strong_threshold and momentum > 0
        is_weak = trend < weak_threshold or volatility > high_vol_threshold or momentum < -0.05

        if is_strong and not is_weak:
            return 'strong'
        elif is_weak:
            return 'weak'
        else:
            return 'neutral'
```

- [ ] **Step 2: 运行测试验证**

```bash
python -c "from back_testing.rotation.market_regime import MarketRegime; from back_testing.rotation.config import MarketRegimeConfig; m=MarketRegime(MarketRegimeConfig()); print('MarketRegime loaded OK')"
```
Expected: `MarketRegime loaded OK`

- [ ] **Step 3: Commit**

```bash
git add back_testing/rotation/market_regime.py
git commit -m "feat(rotation): add MarketRegime for market state detection"
```

---

## Task 6: 持仓管理 — RotationPositionManager

**Files:**
- Create: `back_testing/rotation/position_manager.py`

- [ ] **Step 1: 创建 `back_testing/rotation/position_manager.py`**

```python
"""轮动策略持仓管理器 — 复用 PositionManager 整手计算逻辑"""
import math
from typing import Dict, Optional


class RotationPositionManager:
    """
    持仓管理器 — 每日轮动专用

    资金分配逻辑：
    - 总仓位上限 = 总资产 × max_total_pct
    - 单只上限 = 总资产 × max_position_pct
    - 买入股数 = floor(min(单只上限, 剩余可用) / 单价 / 100) × 100
    """

    def __init__(self, total_capital: float, max_total_pct: float = 0.90,
                 max_position_pct: float = 0.20):
        self.total_capital = total_capital
        self.max_total_pct = max_total_pct
        self.max_position_pct = max_position_pct

    def update_capital(self, total_capital: float):
        """更新总资产（随每日盈亏变化）"""
        self.total_capital = total_capital

    def calculate_buy_shares(
        self,
        stock_code: str,
        current_price: float,
        existing_positions: Dict[str, int],
        prices: Optional[Dict[str, float]] = None
    ) -> int:
        """
        计算买入股数（整手，100的倍数）

        Args:
            stock_code: 股票代码
            current_price: 当前价格
            existing_positions: 已持仓字典 {stock_code: shares}
            prices: 当前持仓的市值单价字典

        Returns:
            买入股数（整手）
        """
        if current_price <= 0:
            return 0

        used_capital = self._calculate_used_capital(existing_positions, prices)
        max_total_position = self.total_capital * self.max_total_pct
        available_by_total = max_total_position - used_capital

        max_single_position = self.total_capital * self.max_position_pct
        available_capital = min(available_by_total, max_single_position)

        if available_capital <= 0:
            return 0

        shares = math.floor(available_capital / current_price / 100) * 100
        return shares

    def can_buy(
        self,
        stock_code: str,
        current_price: float,
        existing_positions: Dict[str, int],
        prices: Optional[Dict[str, float]] = None
    ) -> bool:
        """检查是否可以买入"""
        if current_price <= 0:
            return False

        used_capital = self._calculate_used_capital(existing_positions, prices)
        max_total_position = self.total_capital * self.max_total_pct
        if used_capital >= max_total_position:
            return False

        max_single_position = self.total_capital * self.max_position_pct
        existing_shares = existing_positions.get(stock_code, 0)
        existing_value = existing_shares * (prices.get(stock_code, 0) if prices else 0)
        if existing_value + current_price * 100 > max_single_position:
            return False

        return True

    def get_position_value(self, positions: Dict[str, int], prices: Dict[str, float]) -> float:
        """计算持仓总市值"""
        total = 0.0
        for code, shares in positions.items():
            if shares > 0 and code in prices:
                total += shares * prices[code]
        return total

    def get_available_capital(self, positions: Dict[str, int], prices: Dict[str, float]) -> float:
        """计算可用资金"""
        used = self.get_position_value(positions, prices)
        return self.total_capital * self.max_total_pct - used

    def _calculate_used_capital(
        self,
        positions: Dict[str, int],
        prices: Optional[Dict[str, float]] = None
    ) -> float:
        """计算已用资金"""
        if not positions or prices is None:
            return 0.0
        total = 0.0
        for code, shares in positions.items():
            if shares > 0 and code in prices:
                total += shares * prices[code]
        return total
```

- [ ] **Step 2: 运行测试验证**

```bash
python -c "
from back_testing.rotation.position_manager import RotationPositionManager
pm = RotationPositionManager(1_000_000, 0.9, 0.2)
shares = pm.calculate_buy_shares('sh600000', 100.0, {}, None)
print(shares)
"
```
Expected: `7200` (90万最大仓 / 100元 = 9000股，取整手=7200股)

- [ ] **Step 3: Commit**

```bash
git add back_testing/rotation/position_manager.py
git commit -m "feat(rotation): add RotationPositionManager for position sizing"
```

---

## Task 7: 交易执行器 — TradeExecutor

**Files:**
- Create: `back_testing/rotation/trade_executor.py`

- [ ] **Step 1: 创建 `back_testing/rotation/trade_executor.py`**

```python
"""交易执行器 — 买卖操作和成本计算"""
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TradeRecord:
    """交易记录"""
    date: str
    stock_code: str
    action: str          # 'BUY' or 'SELL'
    price: float
    shares: int
    cost: float          # 手续费+印花税+过户费
    capital_after: float  # 交易后现金


class TradeExecutor:
    """
    交易执行器

    成本设置（沿用 BacktestEngine）：
    - 印花税：0.1%（卖出时收取）
    - 过户费：0.002%（买卖都收取）
    - 券商佣金：0.03%，最低5元
    """

    STAMP_DUTY = 0.001
    TRANSFER_FEE = 0.00002
    BROKERAGE = 0.0003
    MIN_BROKERAGE = 5.0

    def __init__(self):
        pass

    def execute_buy(self, stock_code: str, price: float, cash: float) -> tuple[int, float]:
        """
        模拟买入

        Args:
            stock_code: 股票代码
            price: 买入价格
            cash: 可用资金

        Returns:
            (买入股数, 手续费总额)
        """
        if price <= 0 or cash <= 0:
            return 0, 0.0

        # 按可用资金计算最大可买股数
        max_shares = math.floor(cash / price / 100) * 100
        if max_shares == 0:
            return 0, 0.0

        buy_value = max_shares * price
        # 过户费（买卖都收）
        transfer_fee = buy_value * self.TRANSFER_FEE
        # 券商佣金
        brokerage = max(buy_value * self.BROKERAGE, self.MIN_BROKERAGE)

        total_cost = transfer_fee + brokerage
        actual_cost = total_cost + buy_value

        if actual_cost > cash:
            # 钱不够，降低股数
            available = cash - total_cost
            max_shares = math.floor(available / price / 100) * 100
            if max_shares == 0:
                return 0, 0.0
            buy_value = max_shares * price
            transfer_fee = buy_value * self.TRANSFER_FEE
            brokerage = max(buy_value * self.BROKERAGE, self.MIN_BROKERAGE)
            total_cost = transfer_fee + brokerage

        return max_shares, total_cost

    def execute_sell(self, stock_code: str, price: float, shares: int) -> tuple[int, float]:
        """
        模拟卖出

        Args:
            stock_code: 股票代码
            price: 卖出价格
            shares: 卖出股数

        Returns:
            (实际卖出股数, 手续费总额)
        """
        if price <= 0 or shares <= 0:
            return 0, 0.0

        sell_value = shares * price
        # 印花税（卖出收取）
        stamp_duty = sell_value * self.STAMP_DUTY
        # 过户费（买卖都收）
        transfer_fee = sell_value * self.TRANSFER_FEE
        # 券商佣金
        brokerage = max(sell_value * self.BROKERAGE, self.MIN_BROKERAGE)

        total_cost = stamp_duty + transfer_fee + brokerage
        net_value = sell_value - total_cost

        return shares, total_cost
```

- [ ] **Step 2: 运行测试验证**

```bash
python -c "
from back_testing.rotation.trade_executor import TradeExecutor
te = TradeExecutor()
shares, cost = te.execute_buy('sh600000', 100.0, 100000.0)
print(f'shares={shares}, cost={cost:.2f}')
"
```
Expected: `shares=7200, cost=...` (buy shares + cost > 0)

- [ ] **Step 3: Commit**

```bash
git add back_testing/rotation/trade_executor.py
git commit -m "feat(rotation): add TradeExecutor for trade simulation"
```

---

## Task 8: 核心引擎 — DailyRotationEngine

**Files:**
- Create: `back_testing/rotation/daily_rotation_engine.py`

这是最核心的文件，实现每日轮动流程。

- [ ] **Step 1: 创建 `back_testing/rotation/daily_rotation_engine.py`**

```python
"""每日全市场轮动回测核心引擎"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from back_testing.data.data_provider import DataProvider
from back_testing.rotation.config import RotationConfig
from back_testing.rotation.signal_engine.signal_filter import SignalFilter
from back_testing.rotation.signal_engine.signal_ranker import SignalRanker
from back_testing.rotation.market_regime import MarketRegime
from back_testing.rotation.position_manager import RotationPositionManager
from back_testing.rotation.trade_executor import TradeExecutor, TradeRecord


@dataclass
class Position:
    """持仓信息"""
    stock_code: str
    shares: int
    buy_price: float
    buy_date: str


@dataclass
class DailyResult:
    """每日结果"""
    date: str
    total_asset: float
    cash: float
    positions: Dict[str, Position]
    trades: List[TradeRecord]
    market_regime: str
    portfolio_value: float  # = total_asset


class DailyRotationEngine:
    """
    每日全市场轮动回测引擎

    每日流程：
    1. 获取当日全市场日线数据
    2. 过滤股票池（ST、涨跌停）
    3. 检查持仓 → 卖出信号 → 卖出
    4. 扫描全市场 → 买入信号 → 候选股
    5. 多因子排序 → 买入 TOP X
    6. 记录每日净值和交易
    """

    def __init__(self, config: RotationConfig, start_date: str, end_date: str):
        self.config = config
        self.start_date = start_date
        self.end_date = end_date

        self.data_provider = DataProvider()
        self.position_manager = RotationPositionManager(
            total_capital=config.initial_capital,
            max_total_pct=config.max_total_pct,
            max_position_pct=config.max_position_pct
        )
        self.trade_executor = TradeExecutor()
        self.buy_filter = SignalFilter(config.buy_signal_types)
        self.sell_filter = SignalFilter(config.sell_signal_types)
        self.ranker = SignalRanker(config.rank_factor_weights, config.rank_factor_directions)
        self.market_regime = MarketRegime(config.market_regime, self.data_provider)

        # 状态
        self.current_capital = config.initial_capital
        self.positions: Dict[str, Position] = {}  # stock_code -> Position
        self.daily_results: List[DailyResult] = []
        self.trade_history: List[TradeRecord] = []

    def run(self) -> List[DailyResult]:
        """运行回测"""
        dates = self._get_trading_dates()
        print(f"[DailyRotation] 回测区间: {self.start_date} ~ {self.end_date}, 共 {len(dates)} 个交易日")

        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(dates)}] {date_str} | 持仓:{len(self.positions)} | 资产:{self.current_capital:,.0f}")

            result = self._run_single_day(date)
            self.daily_results.append(result)

        print(f"[DailyRotation] 回测完成，最终资产: {self.current_capital:,.0f}")
        return self.daily_results

    def _run_single_day(self, date: pd.Timestamp) -> DailyResult:
        """每日流程"""
        date_str = date.strftime('%Y-%m-%d')

        # Step 0: 获取大盘状态，动态调整参数
        regime_name, regime_params = self.market_regime.get_regime(date)
        self.position_manager.max_total_pct = regime_params.max_total_pct
        self.position_manager.max_position_pct = regime_params.max_position_pct
        max_positions = regime_params.max_positions

        # 获取当日全市场日线
        stock_data = self._get_daily_stock_data(date)
        if not stock_data:
            return DailyResult(date_str, self.current_capital, self.current_capital, self.positions, [], regime_name)

        # 过滤股票池
        filtered_data = self._filter_stock_pool(stock_data)

        # 获取持仓快照（代码→当前价）
        current_prices = {code: df['close'].iloc[-1] for code, df in filtered_data.items() if not df.empty}
        total_asset = self.current_capital + self.position_manager.get_position_value(
            {p.stock_code: p.shares for p in self.positions.values()},
            current_prices
        )
        self.position_manager.update_capital(total_asset)

        # Step 1: 检查持仓卖出信号
        sell_trades = self._check_and_sell(date_str, filtered_data, current_prices)

        # Step 2: 扫描买入信号
        buy_candidates = self._scan_buy_candidates(filtered_data)

        # Step 3: 多因子排序，买入 TOP X
        buy_trades = self._execute_buy(date_str, filtered_data, buy_candidates, max_positions, current_prices)

        # 更新现金
        for trade in sell_trades:
            self.current_capital += trade.shares * trade.price - trade.cost
        for trade in buy_trades:
            self.current_capital -= trade.shares * trade.price + trade.cost

        all_trades = sell_trades + buy_trades

        return DailyResult(
            date=date_str,
            total_asset=total_asset,
            cash=self.current_capital,
            positions={p.stock_code: p for p in self.positions.values()},
            trades=all_trades,
            market_regime=regime_name
        )

    def _check_and_sell(
        self,
        date_str: str,
        stock_data: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float]
    ) -> List[TradeRecord]:
        """检查持仓是否有卖出信号"""
        sell_trades = []
        positions_to_close = []

        for stock_code, position in self.positions.items():
            if stock_code not in stock_data:
                continue
            df = stock_data[stock_code]
            if df.empty or len(df) < 2:
                continue

            if self.sell_filter.filter_sell(df, stock_code):
                positions_to_close.append(stock_code)

        for stock_code in positions_to_close:
            position = self.positions[stock_code]
            price = current_prices.get(stock_code, 0.0)
            if price <= 0:
                continue

            shares, cost = self.trade_executor.execute_sell(stock_code, price, position.shares)
            if shares > 0:
                trade = TradeRecord(
                    date=date_str,
                    stock_code=stock_code,
                    action='SELL',
                    price=price,
                    shares=shares,
                    cost=cost,
                    capital_after=self.current_capital
                )
                sell_trades.append(trade)
                self.trade_history.append(trade)
                del self.positions[stock_code]

        return sell_trades

    def _scan_buy_candidates(self, stock_data: Dict[str, pd.DataFrame]) -> List[str]:
        """扫描全市场，返回有买入信号的股票代码"""
        candidates = []
        for stock_code, df in stock_data.items():
            if stock_code in self.positions:
                continue
            if df.empty or len(df) < 2:
                continue
            if self.buy_filter.filter_buy(df, stock_code):
                candidates.append(stock_code)
        return candidates

    def _execute_buy(
        self,
        date_str: str,
        stock_data: Dict[str, pd.DataFrame],
        candidates: List[str],
        max_positions: int,
        current_prices: Dict[str, float]
    ) -> List[TradeRecord]:
        """对候选股排序，买入 TOP X"""
        buy_trades = []
        x = max_positions - len(self.positions)
        if x <= 0 or not candidates:
            return buy_trades

        # 提取候选股因子数据
        factor_data_dict = {}
        for stock_code in candidates:
            df = stock_data.get(stock_code)
            if df is None or df.empty:
                continue
            row = df.iloc[-1]
            factor_row = {}
            for factor in self.ranker.factor_weights.keys():
                if factor in row.index:
                    factor_row[factor] = row[factor]
            if factor_row:
                factor_data_dict[stock_code] = factor_row

        factor_df = pd.DataFrame(factor_data_dict).T
        top_stocks = self.ranker.rank(factor_df, top_n=x)

        existing_positions = {p.stock_code: p.shares for p in self.positions.values()}

        for stock_code in top_stocks:
            price = current_prices.get(stock_code, 0.0)
            if price <= 0:
                continue
            if not self.position_manager.can_buy(stock_code, price, existing_positions, current_prices):
                continue

            shares, cost = self.trade_executor.execute_buy(stock_code, price, self.current_capital)
            if shares == 0:
                continue

            trade = TradeRecord(
                date=date_str,
                stock_code=stock_code,
                action='BUY',
                price=price,
                shares=shares,
                cost=cost,
                capital_after=self.current_capital
            )
            buy_trades.append(trade)
            self.trade_history.append(trade)

            self.positions[stock_code] = Position(
                stock_code=stock_code,
                shares=shares,
                buy_price=price,
                buy_date=date_str
            )
            existing_positions[stock_code] = shares

        return buy_trades

    def _get_trading_dates(self) -> List[pd.Timestamp]:
        """获取回测区间内的交易日列表"""
        all_codes = self.data_provider.get_all_stock_codes()
        if not all_codes:
            return []

        df = self.data_provider.get_stock_data(
            all_codes[0],
            start_date=self.start_date,
            end_date=self.end_date
        )
        if df is None or df.empty:
            return []

        dates = sorted(df.index.unique())
        return [pd.Timestamp(d) for d in dates]

    def _get_daily_stock_data(self, date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """获取当日全市场日线数据"""
        date_str = date.strftime('%Y-%m-%d')
        all_codes = self.data_provider.get_all_stock_codes()
        if not all_codes:
            return {}

        result = {}
        batch_data = self.data_provider.get_batch_latest(
            all_codes, date_str, lookback_days=30
        )

        for stock_code, row_data in batch_data.items():
            try:
                # 获取该股票历史数据（含历史指标）
                hist_df = self.data_provider.get_stock_data(
                    stock_code,
                    end_date=date_str
                )
                if hist_df is not None and len(hist_df) > 0:
                    result[stock_code] = hist_df
            except Exception:
                continue

        return result

    def _filter_stock_pool(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """过滤股票池（ST、涨跌停、停牌）"""
        filtered = {}
        for stock_code, df in stock_data.items():
            if df.empty:
                continue
            latest = df.iloc[-1]

            if self.config.exclude_st:
                name = str(latest.get('stock_name', ''))
                if 'ST' in name or '*ST' in name:
                    continue

            if self.config.exclude_limit_up:
                limit_up = latest.get('limit_up', False)
                if limit_up:
                    continue

            if self.config.exclude_limit_down:
                limit_down = latest.get('limit_down', False)
                if limit_down:
                    continue

            if self.config.exclude_suspended:
                if latest.get('volume', 0) == 0:
                    continue

            filtered[stock_code] = df

        return filtered
```

- [ ] **Step 2: 运行测试验证（简单加载测试）**

```bash
python -c "from back_testing.rotation.daily_rotation_engine import DailyRotationEngine; print('DailyRotationEngine loaded OK')"
```
Expected: `DailyRotationEngine loaded OK`

- [ ] **Step 3: Commit**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "feat(rotation): add DailyRotationEngine core backtest engine"
```

---

## Task 9: 独立运行入口 — run_daily_rotation.py

**Files:**
- Create: `back_testing/backtest/run_daily_rotation.py`

- [ ] **Step 1: 创建 `back_testing/backtest/run_daily_rotation.py`**

```python
"""每日轮动策略独立运行入口"""
import argparse
from back_testing.rotation.daily_rotation_engine import DailyRotationEngine
from back_testing.rotation.config import RotationConfig, MarketRegimeConfig
from back_testing.analysis.performance_analyzer import PerformanceAnalyzer
import pandas as pd


def run(start_date: str, end_date: str, config: RotationConfig = None):
    """运行每日轮动回测"""
    print(f"=" * 60)
    print(f"每日全市场轮动回测")
    print(f"区间: {start_date} ~ {end_date}")
    print(f"=" * 60)

    engine = DailyRotationEngine(config or RotationConfig(), start_date, end_date)
    results = engine.run()

    # 输出统计
    total_return = (engine.current_capital / config.initial_capital - 1) if config else 0
    print(f"\n最终资产: {engine.current_capital:,.2f}")
    print(f"总收益率: {total_return:.2%}")
    print(f"交易次数: {len(engine.trade_history)}")

    # 绩效分析
    if results:
        df = pd.DataFrame([{
            'date': r.date,
            'total_asset': r.total_asset,
            'cash': r.cash,
            'position_value': r.total_asset - r.cash,
            'n_positions': len(r.positions),
            'regime': r.market_regime,
        } for r in results])

        analyzer = PerformanceAnalyzer(initial_capital=config.initial_capital if config else 1_000_000)
        perf = analyzer.analyze(df.set_index('date')['total_asset'])
        print(f"\n绩效指标:")
        print(f"  年化收益率: {perf.get('annual_return', 0):.2%}")
        print(f"  Sharpe: {perf.get('sharpe_ratio', 0):.2f}")
        print(f"  最大回撤: {perf.get('max_drawdown', 0):.2%}")

    return engine, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='每日全市场轮动回测')
    parser.add_argument('--start', default='2024-01-01', help='开始日期')
    parser.add_argument('--end', default='2024-12-31', help='结束日期')
    args = parser.parse_args()

    run(args.start, args.end)
```

- [ ] **Step 2: 运行测试验证**

```bash
python -c "from back_testing.backtest.run_daily_rotation import run; print('run_daily_rotation loaded OK')"
```
Expected: `run_daily_rotation loaded OK`

- [ ] **Step 3: Commit**

```bash
git add back_testing/backtest/run_daily_rotation.py
git commit -m "feat(rotation): add run_daily_rotation entry point"
```

---

## Task 10: GA 适应度接口 — RotationFitnessEvaluator

**Files:**
- Create: `back_testing/optimization/rotation_ga_fitness.py`

- [ ] **Step 1: 创建 `back_testing/optimization/rotation_ga_fitness.py`**

```python
"""RotationFitnessEvaluator — GA 适应度评估器"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from back_testing.rotation.daily_rotation_engine import DailyRotationEngine
from back_testing.rotation.config import RotationConfig, MarketRegimeConfig, MarketRegimeParams


class RotationFitnessEvaluator:
    """
    GA 适应度评估器 — 每日轮动策略版本

    给定参数配置，运行回测，返回绩效指标（Sharpe 或总收益）
    用于 GA 遗传算法优化参数搜索。
    """

    def __init__(self, max_drawdown_constraint: float = 0.30, benchmark_index: str = 'sh000300'):
        self.max_drawdown_constraint = max_drawdown_constraint
        self.benchmark_index = benchmark_index
        self._cache = {}

    def evaluate(self, genome: Dict, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
        """
        评估参数配置

        Args:
            genome: GA 参数字典，如 {
                'max_positions': 5,
                'max_total_pct': 0.9,
                'max_position_pct': 0.2,
                'factor_weights': {'RSI_1': 0.2, 'RET_20': 0.25, ...},
                'buy_signal_types': ['KDJ_GOLD', 'MACD_GOLD'],
            }
            start_date: 回测开始日期
            end_date: 回测结束日期

        Returns:
            适应度分数（Sharpe Ratio，约束违反则返回 0）
        """
        cache_key = (start_date, end_date, tuple(sorted(genome.items())))
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            config = self._genome_to_config(genome)
            engine = DailyRotationEngine(config, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            results = engine.run()

            if not results:
                return 0.0

            # 计算净值序列
            values = [r.total_asset for r in results]
            if len(values) < 2:
                return 0.0

            # 计算绩效
            perf = self._calculate_performance(values, genome)
            if perf['max_drawdown'] > self.max_drawdown_constraint:
                return 0.0

            self._cache[cache_key] = perf['sharpe']
            return perf['sharpe']

        except Exception as e:
            print(f"[RotationFitness] 评估异常: {e}")
            return 0.0

    def _genome_to_config(self, genome: Dict) -> RotationConfig:
        """将 GA genome 映射为 RotationConfig"""
        regime_params = {}
        for regime_name in ('strong', 'neutral', 'weak'):
            p = genome.get(f'regime_{regime_name}', {})
            regime_params[regime_name] = MarketRegimeParams(
                max_total_pct=p.get('max_total_pct', 0.9),
                max_position_pct=p.get('max_position_pct', 0.2),
                max_positions=p.get('max_positions', 5),
            )

        market_regime = MarketRegimeConfig(
            strong_trend_threshold=genome.get('strong_trend_threshold', 0.05),
            weak_trend_threshold=genome.get('weak_trend_threshold', -0.03),
            high_volatility_threshold=genome.get('high_volatility_threshold', 0.03),
            lookback_period=genome.get('lookback_period', 20),
            regime_params=regime_params,
        )

        return RotationConfig(
            initial_capital=genome.get('initial_capital', 1_000_000),
            max_total_pct=genome.get('max_total_pct', 0.9),
            max_position_pct=genome.get('max_position_pct', 0.2),
            max_positions=genome.get('max_positions', 5),
            buy_signal_types=genome.get('buy_signal_types', ['KDJ_GOLD', 'MACD_GOLD', 'MA_GOLD']),
            sell_signal_types=genome.get('sell_signal_types', ['KDJ_DEATH', 'MACD_DEATH', 'MA_DEATH']),
            rank_factor_weights=genome.get('factor_weights', {
                'RSI_1': 0.2, 'RET_20': 0.25, 'VOLUME_RATIO': 0.15, 'PB': 0.2, 'PE_TTM': 0.2,
            }),
            rank_factor_directions=genome.get('factor_directions', {
                'RSI_1': 1, 'RET_20': 1, 'VOLUME_RATIO': 1, 'PB': -1, 'PE_TTM': -1,
            }),
            market_regime=market_regime,
            benchmark_index=genome.get('benchmark_index', 'sh000300'),
        )

    def _calculate_performance(self, values: List[float], genome: Dict) -> Dict:
        """计算绩效指标"""
        values = np.array(values)
        returns = np.diff(values) / values[:-1]
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return {'sharpe': 0.0, 'max_drawdown': 0.0, 'total_return': 0.0}

        # 总收益率
        total_return = (values[-1] / values[0]) - 1

        # 最大回撤
        peak = values[0]
        max_drawdown = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_drawdown:
                max_drawdown = dd

        # 年化收益率
        n_years = len(values) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Sharpe（无风险利率=0）
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        return {
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'annual_return': annual_return,
        }

    def clear_cache(self):
        self._cache.clear()
```

- [ ] **Step 2: 运行测试验证**

```bash
python -c "from back_testing.optimization.rotation_ga_fitness import RotationFitnessEvaluator; print('RotationFitnessEvaluator loaded OK')"
```
Expected: `RotationFitnessEvaluator loaded OK`

- [ ] **Step 3: Commit**

```bash
git add back_testing/optimization/rotation_ga_fitness.py
git commit -m "feat(rotation): add RotationFitnessEvaluator for GA integration"
```

---

## Task 11: strategy.py — 抽象策略接口

**Files:**
- Create: `back_testing/rotation/strategy.py`

- [ ] **Step 1: 创建 `back_testing/rotation/strategy.py`**

```python
"""AbstractRotationStrategy — 策略抽象接口"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional
from back_testing.rotation.config import RotationConfig


class AbstractRotationStrategy(ABC):
    """
    每日轮动策略抽象基类

    定义策略的核心接口，支持：
    - 独立运行（run 方法）
    - GA 适应度评估（fitness 方法）
    """

    @abstractmethod
    def run(self, start_date: str, end_date: str, config: Optional[RotationConfig] = None) -> pd.DataFrame:
        """
        运行策略回测

        Args:
            start_date: 开始日期
            end_date: 结束日期
            config: 策略配置

        Returns:
            每日净值 DataFrame，index=date, columns=['total_asset', 'cash', 'position_value']
        """
        pass

    @abstractmethod
    def fitness(self, genome: Dict, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
        """
        GA 适应度函数

        Args:
            genome: GA 参数字典
            start_date: 回测开始日期
            end_date: 回测结束日期

        Returns:
            适应度分数（Sharpe Ratio）
        """
        pass

    @abstractmethod
    def get_performance(self, results_df: pd.DataFrame) -> Dict:
        """
        计算绩效指标

        Returns:
            绩效字典 {sharpe, annual_return, max_drawdown, ...}
        """
        pass
```

- [ ] **Step 2: 运行测试验证**

```bash
python -c "from back_testing.rotation.strategy import AbstractRotationStrategy; print('AbstractRotationStrategy loaded OK')"
```
Expected: `AbstractRotationStrategy loaded OK`

- [ ] **Step 3: Commit**

```bash
git add back_testing/rotation/strategy.py
git commit -m "feat(rotation): add AbstractRotationStrategy interface"
```

---

## Task 12: 更新 `rotation/__init__.py` 导出

**Files:**
- Modify: `back_testing/rotation/__init__.py`

- [ ] **Step 1: 更新 `__init__.py` 导出**

```python
"""每日全市场轮动回测系统"""
from back_testing.rotation.daily_rotation_engine import DailyRotationEngine, Position, DailyResult
from back_testing.rotation.config import RotationConfig, MarketRegimeConfig, MarketRegimeParams
from back_testing.rotation.signal_engine.signal_filter import SignalFilter
from back_testing.rotation.signal_engine.signal_ranker import SignalRanker
from back_testing.rotation.market_regime import MarketRegime
from back_testing.rotation.position_manager import RotationPositionManager
from back_testing.rotation.trade_executor import TradeExecutor, TradeRecord
from back_testing.rotation.strategy import AbstractRotationStrategy

__all__ = [
    'DailyRotationEngine', 'Position', 'DailyResult',
    'RotationConfig', 'MarketRegimeConfig', 'MarketRegimeParams',
    'SignalFilter', 'SignalRanker',
    'MarketRegime',
    'RotationPositionManager',
    'TradeExecutor', 'TradeRecord',
    'AbstractRotationStrategy',
]
```

- [ ] **Step 2: Commit**

```bash
git add back_testing/rotation/__init__.py
git commit -m "feat(rotation): update package exports"
```

---

## 自检清单

**Spec 覆盖检查：**

| Spec Section | 实现位置 |
|---|---|
| 两层信号架构 | `signal_filter.py` + `signal_ranker.py` |
| 每日流程 | `daily_rotation_engine.py` |
| 资金管理（90%+20%） | `position_manager.py` |
| 市场状态动态调节 | `market_regime.py` |
| ST/涨跌停过滤 | `daily_rotation_engine._filter_stock_pool` |
| 交易成本 | `trade_executor.py` |
| GA 适配 | `rotation_ga_fitness.py` |
| 独立运行入口 | `run_daily_rotation.py` |
| 复用 PerformanceAnalyzer | `run_daily_rotation.py` |

**Placeholder 扫描：** 无 TBD/TODO，无不完整实现。

**类型一致性：** 所有 `RotationConfig`、`MarketRegimeConfig`、`MarketRegimeParams` 字段名在 config.py、market_regime.py、rotation_ga_fitness.py 三处保持一致。

---

## 执行选项

Plan complete and saved to `docs/superpowers/plans/2026-04-25-daily-rotation-implementation-plan.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
