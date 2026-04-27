# 信号与因子增强 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 3 new signal types (KDJ_GOLD_LOW, PSY_BUY, PSY_SELL), 3 new rank factors (log circulating_mv, WR_10, WR_14), and wire them into the engine and Optuna optimizer.

**Architecture:** Bottom-up implementation — enum → config → factor utils → signal detectors → engine → optimizer. Each layer builds on the previous one. Tests are written before implementation code in each task.

**Tech Stack:** Python 3.12, pandas, numpy, pytest

---

### Task 1: Add new SignalType enum values

**Files:**
- Modify: `back_testing/rotation/signal_engine/base_signal.py:8-17` (enum), `:28-33` (is_buy property)

- [ ] **Step 1: Add test for new SignalType enums and is_buy**

Create `tests/back_testing/rotation/test_signal_types.py`:

```python
"""Tests for SignalType enum values."""
import pytest
from back_testing.rotation.signal_engine.base_signal import SignalType


def test_kdj_gold_low_is_buy():
    assert SignalType.KDJ_GOLD_LOW.is_buy is True
    assert SignalType.KDJ_GOLD_LOW.is_sell is False


def test_psy_buy_is_buy():
    assert SignalType.PSY_BUY.is_buy is True
    assert SignalType.PSY_BUY.is_sell is False


def test_psy_sell_is_sell():
    assert SignalType.PSY_SELL.is_buy is False
    assert SignalType.PSY_SELL.is_sell is True


def test_existing_signals_unchanged():
    assert SignalType.KDJ_GOLD.is_buy is True
    assert SignalType.KDJ_DEATH.is_sell is True
    assert SignalType.BOLL_BREAK.is_buy is True
    assert SignalType.BOLL_BREAK_DOWN.is_sell is True
    assert SignalType.HIGH_BREAK.is_buy is True
```

Run: `pytest tests/back_testing/rotation/test_signal_types.py -v`
Expected: FAIL — `AttributeError: KDJ_GOLD_LOW`

- [ ] **Step 2: Add enum values and fix is_buy**

In `base_signal.py`, add after `HIGH_BREAK = 'HIGH_BREAK'`:

```python
    KDJ_GOLD_LOW = 'KDJ_GOLD_LOW'
    PSY_BUY = 'PSY_BUY'
```

And after `HIGH_BREAK_DOWN = 'HIGH_BREAK_DOWN'`:

```python
    PSY_SELL = 'PSY_SELL'
```

Update `is_buy` property — the current pattern checks `name.endswith('_GOLD')` which won't match `PSY_BUY`. Change to explicit set:

```python
    @property
    def is_buy(self) -> bool:
        return self.name in ('KDJ_GOLD', 'MACD_GOLD', 'MA_GOLD', 'VOL_GOLD',
                             'DMI_GOLD', 'BOLL_BREAK', 'HIGH_BREAK',
                             'KDJ_GOLD_LOW', 'PSY_BUY')
```

Update `is_sell` similarly:

```python
    @property
    def is_sell(self) -> bool:
        return self.name in ('KDJ_DEATH', 'MACD_DEATH', 'MA_DEATH', 'VOL_DEATH',
                             'DMI_DEATH', 'BOLL_BREAK_DOWN', 'HIGH_BREAK_DOWN',
                             'PSY_SELL')
```

Run: `pytest tests/back_testing/rotation/test_signal_types.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/back_testing/rotation/test_signal_types.py back_testing/rotation/signal_engine/base_signal.py
git commit -m "feat(signal): add KDJ_GOLD_LOW, PSY_BUY, PSY_SELL SignalType enums"
```

---

### Task 2: Update RotationConfig defaults

**Files:**
- Modify: `back_testing/rotation/config.py:38-64`

- [ ] **Step 1: Write test that verifies new config defaults**

Create `tests/back_testing/rotation/test_config_defaults.py`:

```python
"""Tests for RotationConfig default values with new signals/factors."""
from back_testing.rotation.config import RotationConfig


def test_new_signal_types_in_defaults():
    config = RotationConfig()
    assert "KDJ_GOLD_LOW" in config.buy_signal_types
    assert "PSY_SELL" in config.sell_signal_types


def test_dmi_not_in_defaults():
    config = RotationConfig()
    assert "DMI_GOLD" not in config.buy_signal_types
    assert "DMI_DEATH" not in config.sell_signal_types


def test_new_factors_in_defaults():
    config = RotationConfig()
    assert "circulating_mv" in config.rank_factor_weights
    assert "WR_10" in config.rank_factor_weights
    assert "WR_14" in config.rank_factor_weights
    assert config.rank_factor_directions["circulating_mv"] == -1
    assert config.rank_factor_directions["WR_10"] == -1
    assert config.rank_factor_directions["WR_14"] == -1


def test_kdj_low_threshold_default():
    config = RotationConfig()
    assert config.kdj_low_threshold == 30.0


def test_existing_factors_unchanged():
    config = RotationConfig()
    for factor in ['RSI_1', 'RET_20', 'VOLUME_RATIO', 'PB', 'PE_TTM', 'OVERHEAT']:
        assert factor in config.rank_factor_weights
        assert factor in config.rank_factor_directions
```

Run: `pytest tests/back_testing/rotation/test_config_defaults.py -v`
Expected: FAIL — kdj_low_threshold not an attribute, new factors not in weights

- [ ] **Step 2: Update config defaults**

In `config.py`, modify:

`buy_signal_types` default → add `"KDJ_GOLD_LOW"`:
```python
    buy_signal_types: List[str] = field(default_factory=lambda: [
        "KDJ_GOLD",
        "MACD_GOLD",
        "HIGH_BREAK",
        "KDJ_GOLD_LOW",
    ])
```

`sell_signal_types` default → add `"PSY_SELL"`:
```python
    sell_signal_types: List[str] = field(default_factory=lambda: [
        'KDJ_DEATH', 'MACD_DEATH', 'MA_DEATH', 'VOL_DEATH',
        'BOLL_BREAK_DOWN', 'HIGH_BREAK_DOWN', 'PSY_SELL'
    ])
```

Add after `rank_factor_directions`:
```python
    kdj_low_threshold: float = 30.0
```

`rank_factor_weights` default → add new factors:
```python
    rank_factor_weights: Dict[str, float] = field(default_factory=lambda: {
        "RSI_1": 0.16204620597712574,
        "RET_20": 0.1639296507048362,
        "VOLUME_RATIO": 0.06851020411739067,
        "PB": 0.24923532669436346,
        "PE_TTM": 0.2650605509924776,
        "OVERHEAT": 0.0912180615138063,
        "circulating_mv": 0.15,
        "WR_10": 0.10,
        "WR_14": 0.10,
    })
```

`rank_factor_directions` default → add new factors:
```python
    rank_factor_directions: Dict[str, int] = field(default_factory=lambda: {
        'RSI_1': 1,
        'RET_20': 1,
        'VOLUME_RATIO': 1,
        'PB': -1,
        'PE_TTM': -1,
        'OVERHEAT': -1,
        'circulating_mv': -1,
        'WR_10': -1,
        'WR_14': -1,
    })
```

Run: `pytest tests/back_testing/rotation/test_config_defaults.py -v`
Expected: PASS

- [ ] **Step 3: Verify existing tests still pass**

```bash
pytest tests/back_testing/ -v --ignore=tests/back_testing/rotation/test_config_defaults.py --ignore=tests/back_testing/rotation/test_signal_types.py
```

- [ ] **Step 4: Commit**

```bash
git add tests/back_testing/rotation/test_config_defaults.py back_testing/rotation/config.py
git commit -m "feat(config): add new signals and factors to RotationConfig defaults"
```

---

### Task 3: Add williams_r to FactorProcessor

**Files:**
- Modify: `back_testing/factors/factor_utils.py` — add static method after `z_score`

- [ ] **Step 1: Write tests for williams_r**

Add to `tests/back_testing/test_factor_utils.py`:

```python
def test_williams_r_basic():
    """Test Williams %R calculation with known values."""
    import pandas as pd
    df = pd.DataFrame({
        'high': [12, 13, 14, 15, 16, 15, 14, 13, 12, 11],
        'low':  [8,  9,  10, 11, 10, 9,  8,  7,  6,  5],
        'close': [10, 11, 12, 13, 12, 11, 10, 9, 8, 7],
    })
    # WR_5: period=5, last 5 rows (indices 5-9)
    # high_n = max(15,14,13,12,11) = 15
    # low_n = min(9,8,7,6,5) = 5
    # close = 7
    # WR = (15-7)/(15-5) * -100 = 8/10 * -100 = -80
    result = FactorProcessor.williams_r(df, 5)
    assert result == -80.0


def test_williams_r_no_range():
    """Test Williams %R when high == low (no price range)."""
    import pandas as pd
    df = pd.DataFrame({
        'high': [10, 10, 10, 10, 10],
        'low':  [10, 10, 10, 10, 10],
        'close': [10, 10, 10, 10, 10],
    })
    result = FactorProcessor.williams_r(df, 5)
    assert result == -50.0


def test_williams_r_oversold():
    """Test Williams %R at extreme oversold (close near low)."""
    import pandas as pd
    df = pd.DataFrame({
        'high': [15, 15, 15, 15, 15],
        'low':  [5, 5, 5, 5, 5],
        'close': [6, 6, 6, 6, 6],
    })
    # WR = (15-6)/(15-5) * -100 = 9/10 * -100 = -90
    result = FactorProcessor.williams_r(df, 5)
    assert result == pytest.approx(-90.0)


def test_williams_r_overbought():
    """Test Williams %R at extreme overbought (close near high)."""
    import pandas as pd
    df = pd.DataFrame({
        'high': [15, 15, 15, 15, 15],
        'low':  [5, 5, 5, 5, 5],
        'close': [14, 14, 14, 14, 14],
    })
    # WR = (15-14)/(15-5) * -100 = 1/10 * -100 = -10
    result = FactorProcessor.williams_r(df, 5)
    assert result == pytest.approx(-10.0)
```

Run: `pytest tests/back_testing/test_factor_utils.py::test_williams_r_basic -v`
Expected: FAIL — `AttributeError: 'FactorProcessor' has no attribute 'williams_r'`

- [ ] **Step 2: Implement williams_r**

In `factor_utils.py`, add after the `z_score` method (after line 55):

```python
    @staticmethod
    def williams_r(df: pd.DataFrame, period: int) -> float:
        """Williams %R: measures close relative to high-low range over N periods.

        Value ranges from -100 (close at low, oversold) to 0 (close at high, overbought).

        Args:
            df: DataFrame with 'high', 'low', 'close' columns (must have >= period rows).
            period: Lookback period.

        Returns:
            Williams %R value in [-100, 0].
        """
        high_n = df['high'].tail(period).max()
        low_n = df['low'].tail(period).min()
        close = df['close'].iloc[-1]
        if high_n == low_n:
            return -50.0
        return (high_n - close) / (high_n - low_n) * -100
```

Run: `pytest tests/back_testing/test_factor_utils.py -v`
Expected: 14 passed (13 existing + 4 new - 3 that may have been already there = all pass)

- [ ] **Step 3: Commit**

```bash
git add back_testing/factors/factor_utils.py tests/back_testing/test_factor_utils.py
git commit -m "feat(factor): add williams_r static method to FactorProcessor"
```

---

### Task 4: Add new signal detectors to signal_filter.py

**Files:**
- Modify: `back_testing/rotation/signal_engine/signal_filter.py` — add 3 detector classes + update SignalFilter

- [ ] **Step 1: Write tests for new detectors**

Create `tests/back_testing/rotation/test_signal_filter.py`:

```python
"""Tests for signal detectors: KDJGoldLowSignal, PSYBuySignal, PSYSellSignal."""
import pandas as pd
from back_testing.rotation.signal_engine.signal_filter import (
    KDJGoldLowSignal, PSYBuySignal, PSYSellSignal, SignalFilter
)
from back_testing.rotation.signal_engine.base_signal import SignalType


def _make_kdj_df(k_values, d_values):
    """Build a minimal DataFrame for KDJ signal testing.
    k_values: list of k values, last two are used for cross detection.
    d_values: list of d values, same length.
    """
    return pd.DataFrame({'kdj_k': k_values, 'kdj_d': d_values})


def _make_psy_df(psy_val, psyma_val):
    """Build a minimal DataFrame for PSY signal testing."""
    return pd.DataFrame({'psy': [psy_val], 'psyma': [psyma_val]})


class TestKDJGoldLowSignal:
    def test_triggers_on_cross_up_and_low_k(self):
        detector = KDJGoldLowSignal(k_threshold=30.0)
        # k crosses above d, and k is low (20 < 30)
        df = _make_kdj_df([15, 18, 20], [16, 17, 18])
        result = detector.detect(df, 'sh600001')
        assert result.triggered is True

    def test_no_trigger_when_k_too_high(self):
        detector = KDJGoldLowSignal(k_threshold=30.0)
        # k crosses above d, but k is 35 >= 30
        df = _make_kdj_df([30, 32, 35], [31, 31, 33])
        result = detector.detect(df, 'sh600001')
        assert result.triggered is False

    def test_no_trigger_when_no_cross(self):
        detector = KDJGoldLowSignal(k_threshold=30.0)
        # k is low but no cross (k stays above d)
        df = _make_kdj_df([25, 26, 26], [24, 25, 25])
        result = detector.detect(df, 'sh600001')
        assert result.triggered is False

    def test_respects_threshold_parameter(self):
        detector_low = KDJGoldLowSignal(k_threshold=20.0)
        detector_high = KDJGoldLowSignal(k_threshold=40.0)
        # k at 25, crosses up from 22→25 above d 23→24
        df = _make_kdj_df([22, 22, 25], [21, 23, 24])
        assert detector_low.detect(df, 'sh600001').triggered is False  # 25 >= 20
        assert detector_high.detect(df, 'sh600001').triggered is True   # 25 < 40


class TestPSYBuySignal:
    def test_triggers_when_psy_low_and_above_psyma(self):
        # PSY < 25 (oversold) AND PSY > PSYMA (turning up)
        df = _make_psy_df(20, 18)
        result = PSYBuySignal().detect(df, 'sh600001')
        assert result.triggered is True

    def test_no_trigger_when_psy_not_low_enough(self):
        df = _make_psy_df(30, 28)  # PSY >= 25
        result = PSYBuySignal().detect(df, 'sh600001')
        assert result.triggered is False

    def test_no_trigger_when_psy_below_psyma(self):
        df = _make_psy_df(20, 22)  # PSY < PSYMA (still falling)
        result = PSYBuySignal().detect(df, 'sh600001')
        assert result.triggered is False


class TestPSYSellSignal:
    def test_triggers_when_psy_high_and_below_psyma(self):
        # PSY > 75 (overbought) AND PSY < PSYMA (turning down)
        df = _make_psy_df(80, 82)
        result = PSYSellSignal().detect(df, 'sh600001')
        assert result.triggered is True

    def test_no_trigger_when_psy_not_high_enough(self):
        df = _make_psy_df(70, 72)
        result = PSYSellSignal().detect(df, 'sh600001')
        assert result.triggered is False

    def test_no_trigger_when_psy_above_psyma(self):
        df = _make_psy_df(80, 78)  # PSY > PSYMA (still rising)
        result = PSYSellSignal().detect(df, 'sh600001')
        assert result.triggered is False


class TestSignalFilterWithNewSignals:
    def test_signal_map_contains_new_detectors(self):
        sf = SignalFilter(['KDJ_GOLD_LOW', 'PSY_BUY'], mode='OR')
        detector_types = [type(d).__name__ for d in sf.detectors]
        assert 'KDJGoldLowSignal' in detector_types
        assert 'PSYBuySignal' in detector_types

    def test_kdj_low_threshold_passed_through(self):
        sf = SignalFilter(['KDJ_GOLD_LOW'], mode='OR', kdj_low_threshold=25.0)
        kdj_detector = next(d for d in sf.detectors if isinstance(d, KDJGoldLowSignal))
        assert kdj_detector.k_threshold == 25.0
```

Run: `pytest tests/back_testing/rotation/test_signal_filter.py -v`
Expected: FAIL — `ImportError: cannot import name 'KDJGoldLowSignal'`

- [ ] **Step 2: Implement the 3 new detectors**

In `signal_filter.py`, add after `KDJGoldSignal` class (after line 27), before `KDJDeathSignal`:

```python
class KDJGoldLowSignal(BaseSignal):
    """KDJ 低位金叉检测 — 金叉且 K 值低于阈值"""

    def __init__(self, k_threshold: float = 30.0):
        super().__init__(SignalType.KDJ_GOLD_LOW)
        self.k_threshold = k_threshold

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        k = df['kdj_k']; d = df['kdj_d']
        triggered = self._cross_up(k, d) and k.iloc[-1] < self.k_threshold
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0,
            metadata={'kdj_k': k.iloc[-1] if not k.empty else None,
                      'threshold': self.k_threshold}
        )
```

Add after `DMIDeathSignal` class (after line 207), before `BollBreakSignal`:

```python
class PSYBuySignal(BaseSignal):
    """PSY 心理线超卖买入信号 — PSY < 25 且 PSY > PSYMA（超卖 + 趋势确认）"""

    def __init__(self):
        super().__init__(SignalType.PSY_BUY)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'psy' not in df.columns or 'psyma' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        psy_now = df['psy'].iloc[-1]
        psyma_now = df['psyma'].iloc[-1]
        triggered = psy_now < 25 and psy_now > psyma_now
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0,
            metadata={'psy': psy_now, 'psyma': psyma_now}
        )


class PSYSellSignal(BaseSignal):
    """PSY 心理线超买卖出信号 — PSY > 75 且 PSY < PSYMA（超买 + 趋势确认）"""

    def __init__(self):
        super().__init__(SignalType.PSY_SELL)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'psy' not in df.columns or 'psyma' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        psy_now = df['psy'].iloc[-1]
        psyma_now = df['psyma'].iloc[-1]
        triggered = psy_now > 75 and psy_now < psyma_now
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0,
            metadata={'psy': psy_now, 'psyma': psyma_now}
        )
```

- [ ] **Step 3: Update _SIGNAL_MAP and SignalFilter.__init__**

In `_SIGNAL_MAP`, add 3 new entries after `SignalType.HIGH_BREAK_DOWN: HighBreakDownSignal,`:

```python
        SignalType.KDJ_GOLD_LOW: KDJGoldLowSignal,
        SignalType.PSY_BUY: PSYBuySignal,
        SignalType.PSY_SELL: PSYSellSignal,
```

Update `SignalFilter.__init__` signature to accept `kdj_low_threshold`:

```python
    def __init__(self, signal_types: List[str], mode: str = 'OR', kdj_low_threshold: float = 30.0):
```

And in the detector construction loop, after `detector_cls = self._SIGNAL_MAP.get(sig_type)`, pass the threshold when creating KDJGoldLowSignal:

```python
                if detector_cls:
                    if detector_cls is KDJGoldLowSignal:
                        self.detectors.append(detector_cls(k_threshold=kdj_low_threshold))
                    else:
                        self.detectors.append(detector_cls())
```

Run: `pytest tests/back_testing/rotation/test_signal_filter.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/back_testing/rotation/test_signal_filter.py back_testing/rotation/signal_engine/signal_filter.py
git commit -m "feat(signal): add KDJGoldLowSignal, PSYBuySignal, PSYSellSignal detectors"
```

---

### Task 5: Wire new signals and factors into DailyRotationEngine

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:89` (SignalFilter construction), `:417-464` (_build_signal_features), `:389-415` (_scan_buy_candidates), `:484-519` (_execute_buy factor extraction)

- [ ] **Step 1: Write integration tests**

Create `tests/back_testing/rotation/test_engine_new_signals.py`:

```python
"""Integration tests for new signals and factors in DailyRotationEngine."""
import pandas as pd
import numpy as np
from back_testing.rotation.config import RotationConfig
from back_testing.rotation.daily_rotation_engine import DailyRotationEngine, FactorProcessor


class TestBuildSignalFeatures:
    """Test _build_signal_features includes new columns."""

    def test_psy_columns_in_features(self):
        config = RotationConfig()
        engine = DailyRotationEngine(config, '2024-01-01', '2024-01-31')

        # Build a minimal cache_df with psy/psyma columns
        dates = pd.date_range('2024-01-01', '2024-01-25', freq='B')
        rows = []
        for i, d in enumerate(dates):
            rows.append({
                'trade_date': d, 'stock_code': 'sh600001',
                'close': 10 + i * 0.1, 'open': 10 + i * 0.1,
                'high': 11 + i * 0.1, 'low': 9 + i * 0.1,
                'volume': 1000, 'kdj_k': 50, 'kdj_d': 48,
                'macd_dif': 0.5, 'macd_dea': 0.3,
                'ma_5': 10, 'ma_20': 9.5,
                'boll_mid': 10, 'psy': 30.0, 'psyma': 28.0,
            })
        engine._cache_df = pd.DataFrame(rows).set_index('trade_date')

        features = engine._build_signal_features(['sh600001'])
        assert 'psy' in features.columns
        assert 'psyma' in features.columns
        assert 'psy_p' in features.columns
        assert 'psyma_p' in features.columns


class TestNewFactorExtraction:
    """Test factor extraction with new factors."""

    def test_circulating_mv_log_transform(self):
        config = RotationConfig()
        config.rank_factor_weights = {'circulating_mv': 1.0}
        config.rank_factor_directions = {'circulating_mv': -1}
        engine = DailyRotationEngine(config, '2024-01-01', '2024-01-31')

        # Build stock_data with circulating_mv
        dates = pd.date_range('2024-01-01', '2024-01-25', freq='B')
        rows = []
        for i, d in enumerate(dates):
            rows.append({
                'trade_date': d, 'stock_code': 'sh600001',
                'close': 10 + i * 0.1, 'open': 10 + i * 0.1,
                'high': 11 + i * 0.1, 'low': 9 + i * 0.1,
                'volume': 1000, 'circulating_mv': 1e9,
            })
        engine._cache_df = pd.DataFrame(rows).set_index('trade_date')

        stock_data = engine._get_daily_stock_data(dates[-1])
        assert 'sh600001' in stock_data
        df = stock_data['sh600001']
        row = df.iloc[-1]

        val = row.get('circulating_mv', np.nan)
        log_val = np.log(val) if val > 0 else np.nan
        assert log_val == pytest.approx(np.log(1e9))

    def test_wr_factor_extraction(self):
        df = pd.DataFrame({
            'high': [12, 13, 14, 15, 16, 15, 14, 13, 12, 11,
                     12, 13, 14, 15, 16, 15, 14, 13, 12, 11],
            'low':  [8,  9,  10, 11, 10, 9,  8,  7,  6,  5,
                     8,  9,  10, 11, 10, 9,  8,  7,  6,  5],
            'close': [10, 11, 12, 13, 12, 11, 10, 9, 8, 7,
                      10, 11, 12, 13, 12, 11, 10, 9, 8, 7],
        })
        wr10 = FactorProcessor.williams_r(df, 10)
        wr14 = FactorProcessor.williams_r(df, 14)
        assert -100 <= wr10 <= 0
        assert -100 <= wr14 <= 0
```

Run: `pytest tests/back_testing/rotation/test_engine_new_signals.py -v`
Expected: FAIL on `_build_signal_features` — 'psy_p' not in features

- [ ] **Step 2: Update SignalFilter construction in __init__**

In `daily_rotation_engine.py`, change line 89:

```python
        self.buy_filter = SignalFilter(config.buy_signal_types, mode=config.buy_signal_mode,
                                        kdj_low_threshold=config.kdj_low_threshold)
```

- [ ] **Step 3: Extend _build_signal_features with psy/psyma columns**

In `_build_signal_features`, add `psy`, `psyma` to the returned DataFrame. After the line `'boll_mid': latest['boll_mid'], 'high_20_max': latest['high_20_max'],`:

```python
            'psy': latest['psy'], 'psyma': latest['psyma'],
            'psy_p': prev['psy'], 'psyma_p': prev['psyma'],
```

- [ ] **Step 4: Add KDJ_GOLD_LOW and PSY_BUY vectorized masks**

In `_scan_buy_candidates`, add after the `if 'HIGH_BREAK' in active_signals:` block:

```python
        if 'KDJ_GOLD_LOW' in active_signals:
            k_thresh = self.config.kdj_low_threshold
            masks['KDJ_GOLD_LOW'] = (
                (f['kdj_k'] > f['kdj_d']) & (f['kdj_k_p'] <= f['kdj_d_p']) & (f['kdj_k'] < k_thresh)
            )
        if 'PSY_BUY' in active_signals:
            masks['PSY_BUY'] = (f['psy'] < 25) & (f['psy'] > f['psyma'])
```

- [ ] **Step 5: Add new factor extraction in _execute_buy**

In `_execute_buy`, in the factor extraction loop (line ~497-518), add after the `elif factor in row.index:` block:

```python
                elif factor == 'circulating_mv':
                    val = row.get('circulating_mv', np.nan)
                    factor_row[factor] = np.log(val) if val > 0 else np.nan
                elif factor in ('WR_10', 'WR_14'):
                    period = 10 if factor == 'WR_10' else 14
                    factor_row[factor] = FactorProcessor.williams_r(df, period)
```

Run: `pytest tests/back_testing/rotation/test_engine_new_signals.py -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/back_testing/ -v
```

- [ ] **Step 7: Commit**

```bash
git add tests/back_testing/rotation/test_engine_new_signals.py back_testing/rotation/daily_rotation_engine.py
git commit -m "feat(engine): wire new signals and factors into DailyRotationEngine"
```

---

### Task 6: Expand Optuna optimization parameters

**Files:**
- Modify: `back_testing/optimization/run_daily_rotation_optimization.py:76-155` (ALL_SIGNAL_TYPES, FIXED_FACTOR_DIRECTIONS, sample_config), `:374-413` (_params_to_config), `:456-472` (_config_to_dict)

- [ ] **Step 1: Write test for extended sample_config**

Create `tests/back_testing/optimization/test_optuna_new_params.py`:

```python
"""Tests for new Optuna optimization parameters."""
import optuna
from back_testing.optimization.run_daily_rotation_optimization import sample_config
from back_testing.rotation.config import RotationConfig


def test_sample_config_includes_new_factors():
    study = optuna.create_study()
    trial = study.ask()

    config = sample_config(trial)
    assert 'circulating_mv' in config.rank_factor_weights
    assert 'WR_10' in config.rank_factor_weights
    assert 'WR_14' in config.rank_factor_weights
    assert 'circulating_mv' in config.rank_factor_directions
    assert config.rank_factor_directions['circulating_mv'] == -1


def test_sample_config_has_kdj_low_threshold():
    study = optuna.create_study()
    trial = study.ask()
    config = sample_config(trial)
    assert hasattr(config, 'kdj_low_threshold')
    assert 20.0 <= config.kdj_low_threshold <= 40.0


def test_kdj_gold_low_in_signal_pool():
    from back_testing.optimization.run_daily_rotation_optimization import ALL_SIGNAL_TYPES
    assert 'KDJ_GOLD_LOW' in ALL_SIGNAL_TYPES
```

Run: `pytest tests/back_testing/optimization/test_optuna_new_params.py -v`
Expected: FAIL — new factors not sampled, kdj_low_threshold not sampled

- [ ] **Step 2: Update ALL_SIGNAL_TYPES and FIXED_FACTOR_DIRECTIONS**

In `run_daily_rotation_optimization.py`:

```python
ALL_SIGNAL_TYPES = ['KDJ_GOLD', 'MACD_GOLD', 'MA_GOLD', 'VOL_GOLD', 'BOLL_BREAK', 'HIGH_BREAK', 'KDJ_GOLD_LOW']

FIXED_FACTOR_DIRECTIONS = {
    'RSI_1': 1, 'RET_20': 1, 'VOLUME_RATIO': 1,
    'PB': -1, 'PE_TTM': -1, 'OVERHEAT': -1,
    'circulating_mv': -1, 'WR_10': -1, 'WR_14': -1,
}
```

- [ ] **Step 3: Update sample_config to sample new parameters**

In `sample_config`, add after `raw_weights` sampling (add new factors to the loop that now iterates over FIXED_FACTOR_DIRECTIONS, which already includes them — no code change needed for the loop itself since it iterates over the dict keys).

Add `kdj_low_threshold` sampling after `atr_period`:

```python
    kdj_low_threshold = trial.suggest_float('kdj_low_threshold', 20.0, 40.0)
```

In the `RotationConfig(...)` call, add the new parameter:

```python
        kdj_low_threshold=kdj_low_threshold,
```

And change `rank_factor_directions=FIXED_FACTOR_DIRECTIONS` to pass the full dict (already correct since FIXED_FACTOR_DIRECTIONS now includes new factors).

- [ ] **Step 4: Update _params_to_config**

Add `kdj_low_threshold` extraction and pass it to `RotationConfig(...)`:

```python
    return RotationConfig(
        ...
        overheat_rsi_threshold=params['overheat_rsi_threshold'],
        overheat_ret5_threshold=params['overheat_ret5_threshold'],
        kdj_low_threshold=params['kdj_low_threshold'],
    )
```

- [ ] **Step 5: Update _config_to_dict**

Add `kdj_low_threshold` to the serialized dict:

```python
        'kdj_low_threshold': config.kdj_low_threshold,
```

Run: `pytest tests/back_testing/optimization/test_optuna_new_params.py -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/back_testing/ -v
```

- [ ] **Step 7: Commit**

```bash
git add tests/back_testing/optimization/test_optuna_new_params.py back_testing/optimization/run_daily_rotation_optimization.py
git commit -m "feat(optuna): expand optimization to include new signals and factors"
```

---

### Task 7: Smoke test — run a short backtest

- [ ] **Step 1: Run a short single backtest to verify everything works end-to-end**

```bash
python -c "
from back_testing.rotation.config import RotationConfig
from back_testing.rotation.daily_rotation_engine import DailyRotationEngine
from back_testing.data.data_provider import DataProvider

config = RotationConfig()
engine = DailyRotationEngine(config, '2024-12-01', '2024-12-31', data_provider=DataProvider())
results = engine.run()
print(f'Days: {len(results)}')
if results:
    print(f'Final asset: {results[-1].total_asset:,.0f}')
    print(f'Trades: {len(engine.trade_history)}')
    print(f'Last config signals: {config.buy_signal_types}')
    print(f'Factors used: {list(config.rank_factor_weights.keys())}')
"
```

- [ ] **Step 2: Commit if any fixes were needed, or just verify clean output**

---

### Task 8: Final verification — run full test suite

- [ ] **Step 1: Run all tests**

```bash
pytest tests/back_testing/ -v
```
