# Live Trading Signal Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the daily rotation backtesting system into a live trading signal pipeline: 2:30 PM intraday data → signals → Web dashboard → manual execution → nightly data backfill.

**Architecture:** The plan is organized in three layers. Layer 1 (Tasks 1-5) builds the data pipeline foundation — DB tables, indicator calculation, and data source clients. Layer 2 (Tasks 6-8) implements the two daily cron scripts and the signal generator. Layer 3 (Tasks 9-16) builds the FastAPI backend and React frontend. Layer 4 (Task 17) handles deployment configs.

**Tech Stack:** Python 3.12, PostgreSQL (existing), FastAPI, React 18 + Ant Design 5 + Vite, AKShare, Tushare Pro, pandas/numpy

---

### Task 1: Create New DB Tables

**Files:**
- Create: `signal_pipeline/schema.sql`
- Modify: (none, run SQL directly against existing PostgreSQL)

- [ ] **Step 1: Write the migration SQL**

```sql
-- signal_pipeline/schema.sql
-- Run: psql -U <user> -d <db> -f signal_pipeline/schema.sql

CREATE TABLE IF NOT EXISTS daily_signal (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    stock_code VARCHAR(10) NOT NULL,
    stock_name VARCHAR(50),
    direction VARCHAR(4) NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    target_pct NUMERIC(5,2),
    price_low NUMERIC(10,3),
    price_high NUMERIC(10,3),
    signal_reason TEXT,
    status VARCHAR(10) DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'skipped')),
    executed_price NUMERIC(10,3),
    confirmed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS position (
    id SERIAL PRIMARY KEY,
    stock_code VARCHAR(10) NOT NULL,
    stock_name VARCHAR(50),
    buy_date DATE NOT NULL,
    buy_price NUMERIC(10,3) NOT NULL,
    shares INT NOT NULL,
    sell_date DATE,
    sell_price NUMERIC(10,3),
    profit_pct NUMERIC(10,4),
    status VARCHAR(10) DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED'))
);

CREATE TABLE IF NOT EXISTS capital_ledger (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(10) NOT NULL CHECK (event_type IN ('INIT', 'DEPOSIT', 'BUY', 'SELL')),
    amount NUMERIC(15,2) NOT NULL,
    balance_after NUMERIC(15,2) NOT NULL,
    related_signal_id INT REFERENCES daily_signal(id),
    note TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cron_log (
    id SERIAL PRIMARY KEY,
    task_name VARCHAR(50) NOT NULL,
    status VARCHAR(10) NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    started_at TIMESTAMP DEFAULT NOW(),
    finished_at TIMESTAMP,
    error_message TEXT,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_signal_date ON daily_signal(trade_date);
CREATE INDEX IF NOT EXISTS idx_signal_status ON daily_signal(status);
CREATE INDEX IF NOT EXISTS idx_position_status ON position(status);
CREATE INDEX IF NOT EXISTS idx_cron_task ON cron_log(task_name, started_at);
```

- [ ] **Step 2: Run migration against PostgreSQL**

```bash
psql -U njquant -d njquant -f signal_pipeline/schema.sql
```
Expected: `CREATE TABLE` × 4, `CREATE INDEX` × 4

- [ ] **Step 3: Verify tables exist**

```bash
psql -U njquant -d njquant -c "\dt daily_signal position capital_ledger cron_log"
```
Expected: Four tables listed

- [ ] **Step 4: Insert initial capital record**

```bash
psql -U njquant -d njquant -c "INSERT INTO capital_ledger (event_type, amount, balance_after, note, created_at) VALUES ('INIT', 100000, 100000, '初始资金', NOW());"
```

- [ ] **Step 5: Commit**

```bash
git add signal_pipeline/schema.sql
git commit -m "feat: add live trading DB tables (signal, position, capital, cron)"
```

---

### Task 2: Implement Vectorized Indicator Calculator

**Files:**
- Create: `signal_pipeline/indicator_calculator.py`
- Create: `tests/signal_pipeline/test_indicator_calculator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/signal_pipeline/test_indicator_calculator.py
import pandas as pd
import numpy as np
from signal_pipeline.indicator_calculator import IndicatorCalculator


def test_calculate_all_returns_expected_columns():
    df = pd.DataFrame({
        'trade_date': pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04'] * 2),
        'stock_code': ['sh600001'] * 3 + ['sz000001'] * 3,
        'open': [10.0, 10.5, 10.3, 20.0, 20.8, 21.0],
        'high': [10.8, 10.9, 10.7, 20.5, 21.2, 21.5],
        'low': [9.8, 10.2, 10.1, 19.8, 20.5, 20.8],
        'close': [10.5, 10.3, 10.6, 20.8, 21.0, 21.3],
        'volume': [10000, 12000, 9000, 20000, 22000, 18000],
    })

    result = IndicatorCalculator.calculate_all(df)

    expected_cols = [
        'ma_5', 'ma_10', 'ma_20', 'ma_30', 'ma_60', 'ma_cross',
        'macd_dif', 'macd_dea', 'macd_hist', 'macd_cross',
        'kdj_k', 'kdj_d', 'kdj_j', 'kdj_cross',
        'boll_mid', 'boll_upper', 'boll_lower',
        'rsi_1', 'rsi_2', 'rsi_3',
        'psy', 'psyma',
        'vol_ma5', 'vol_ma20', 'close_std_20', 'high_20_max',
        'atr_14', 'wr_10', 'wr_14', 'ret_5', 'ret_20',
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"
    assert len(result) == 6


def test_ma_cross_detection():
    df = pd.DataFrame({
        'trade_date': pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04',
                                       '2024-01-05', '2024-01-06']),
        'stock_code': ['sh600001'] * 5,
        'open': [10.0] * 5, 'high': [11.0] * 5, 'low': [9.0] * 5,
        'close': [10.0, 10.5, 11.0, 11.5, 11.0],
        'volume': [10000] * 5,
    })

    result = IndicatorCalculator.calculate_all(df)
    # After rising prices, MA5 should cross above MA20 at some point
    assert 'golden_cross' in result['ma_cross'].values or 'death_cross' in result['ma_cross'].values
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/signal_pipeline/test_indicator_calculator.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'signal_pipeline'`

- [ ] **Step 3: Create `signal_pipeline/__init__.py` to fix import, then verify test still fails with `IndicatorCalculator not defined`**

```python
# signal_pipeline/__init__.py (empty)
```

- [ ] **Step 4: Write the indicator calculator**

```python
# signal_pipeline/indicator_calculator.py
import pandas as pd
import numpy as np


class IndicatorCalculator:
    """Vectorized technical indicator calculator for all stocks.
    
    Takes a DataFrame containing multiple stocks × multiple dates,
    returns the same DataFrame with all indicator columns added.
    Calculations are vectorized via groupby rolling transforms.
    """

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators. Returns df with added columns."""
        df = df.sort_values(['stock_code', 'trade_date']).copy()
        grouped = df.groupby('stock_code', sort=False)

        # --- Moving Averages ---
        for window in [5, 10, 20, 30, 60]:
            df[f'ma_{window}'] = grouped['close'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

        # MA cross detection
        df['ma_cross'] = IndicatorCalculator._detect_cross(
            df, 'ma_5', 'ma_20'
        )

        # --- MACD (12, 26, 9) ---
        df['ema_12'] = grouped['close'].transform(
            lambda x: x.ewm(span=12, adjust=False).mean()
        )
        df['ema_26'] = grouped['close'].transform(
            lambda x: x.ewm(span=26, adjust=False).mean()
        )
        df['macd_dif'] = df['ema_12'] - df['ema_26']
        df['macd_dea'] = grouped['macd_dif'].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )
        df['macd_hist'] = 2 * (df['macd_dif'] - df['macd_dea'])
        df['macd_cross'] = IndicatorCalculator._detect_cross(
            df, 'macd_dif', 'macd_dea'
        )
        df.drop(['ema_12', 'ema_26'], axis=1, inplace=True)

        # --- KDJ (9, 3, 3) ---
        low_9 = grouped['low'].transform(lambda x: x.rolling(9, min_periods=1).min())
        high_9 = grouped['high'].transform(lambda x: x.rolling(9, min_periods=1).max())
        rsv = ((df['close'] - low_9) / (high_9 - low_9 + 1e-10)) * 100

        def _calc_kdj(rsv_series):
            k = pd.Series(50.0, index=rsv_series.index)
            d = pd.Series(50.0, index=rsv_series.index)
            for i in range(1, len(rsv_series)):
                k.iloc[i] = 2/3 * k.iloc[i-1] + 1/3 * rsv_series.iloc[i]
                d.iloc[i] = 2/3 * d.iloc[i-1] + 1/3 * k.iloc[i]
            return k, d

        k_vals, d_vals = [], []
        for _, grp_rsv in grouped['close']:
            k, d = _calc_kdj(rsv.loc[grp_rsv.index])
            k_vals.append(k)
            d_vals.append(d)
        df['kdj_k'] = pd.concat(k_vals)
        df['kdj_d'] = pd.concat(d_vals)
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        df['kdj_cross'] = IndicatorCalculator._detect_cross(df, 'kdj_k', 'kdj_d')

        # --- Bollinger Bands (20, 2) ---
        df['boll_mid'] = grouped['close'].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        std_20 = grouped['close'].transform(lambda x: x.rolling(20, min_periods=1).std())
        df['boll_upper'] = df['boll_mid'] + 2 * std_20
        df['boll_lower'] = df['boll_mid'] - 2 * std_20

        # --- RSI (6, 12, 24) ---
        for period, col in [(6, 'rsi_1'), (12, 'rsi_2'), (24, 'rsi_3')]:
            delta = grouped['close'].transform(lambda x: x.diff())
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = grouped[gain.name if hasattr(gain, 'name') else None] if False else \
                       gain.groupby(df['stock_code']).transform(
                           lambda x: x.rolling(period, min_periods=1).mean())
            # Use simpler approach:
            avg_gain = gain.groupby(df['stock_code'], sort=False).transform(
                lambda x: x.ewm(span=period, adjust=False).mean()
            )
            avg_loss = loss.groupby(df['stock_code'], sort=False).transform(
                lambda x: x.ewm(span=period, adjust=False).mean()
            )
            rs = avg_gain / (avg_loss + 1e-10)
            df[col] = 100 - (100 / (1 + rs))

        # --- PSY (12) ---
        up_days = grouped['close'].transform(
            lambda x: (x.diff() > 0).rolling(12, min_periods=1).sum()
        )
        df['psy'] = up_days / 12 * 100
        df['psyma'] = df.groupby('stock_code', sort=False)['psy'].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )

        # --- Volume MAs ---
        df['vol_ma5'] = grouped['volume'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df['vol_ma20'] = grouped['volume'].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )

        # --- Close std (Bollinger width) ---
        df['close_std_20'] = grouped['close'].transform(
            lambda x: x.rolling(20, min_periods=1).std()
        )

        # --- 20-day high max (exclude today) ---
        df['high_20_max'] = grouped['high'].transform(
            lambda x: x.shift(1).rolling(20, min_periods=1).max()
        )

        # --- ATR (14) ---
        prev_close = df.groupby('stock_code', sort=False)['close'].transform(lambda x: x.shift(1))
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.groupby(df['stock_code'], sort=False).transform(
            lambda x: x.rolling(14, min_periods=1).mean()
        )

        # --- Williams %R (10, 14) ---
        for period in [10, 14]:
            high_n = grouped['high'].transform(
                lambda x: x.rolling(period, min_periods=1).max()
            )
            low_n = grouped['low'].transform(
                lambda x: x.rolling(period, min_periods=1).min()
            )
            denom = high_n - low_n
            wr = pd.Series(-50.0, index=df.index)
            mask = denom > 0
            wr[mask] = (high_n[mask] - df.loc[mask, 'close']) / denom[mask] * -100
            df[f'wr_{period}'] = wr

        # --- Returns ---
        df['ret_5'] = grouped['close'].transform(lambda x: x / x.shift(5) - 1)
        df['ret_20'] = grouped['close'].transform(lambda x: x / x.shift(20) - 1)

        # Fill NaN for insufficient history
        fill_cols = [
            'vol_ma5', 'vol_ma20', 'close_std_20', 'high_20_max',
            'atr_14', 'wr_10', 'wr_14', 'ret_5', 'ret_20',
            'boll_upper', 'boll_lower', 'rsi_1', 'rsi_2', 'rsi_3',
            'psy', 'psyma', 'kdj_k', 'kdj_d', 'kdj_j',
        ]
        df[fill_cols] = df[fill_cols].fillna(0.0)

        return df

    @staticmethod
    def _detect_cross(df: pd.DataFrame, fast_col: str, slow_col: str) -> pd.Series:
        """Detect golden/death crosses across all stocks.
        
        Returns a Series of strings: 'golden_cross', 'death_cross', or 'none'.
        """
        result = pd.Series('none', index=df.index, dtype=str)

        fast = df.groupby('stock_code', sort=False)[fast_col]
        slow = df.groupby('stock_code', sort=False)[slow_col]

        fast_prev = fast.transform(lambda x: x.shift(1))
        slow_prev = slow.transform(lambda x: x.shift(1))
        fast_cur = df[fast_col]
        slow_cur = df[slow_col]

        golden = (fast_prev <= slow_prev) & (fast_cur > slow_cur)
        death = (fast_prev >= slow_prev) & (fast_cur < slow_cur)

        result[golden] = 'golden_cross'
        result[death] = 'death_cross'
        return result
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/signal_pipeline/test_indicator_calculator.py -v
```
Expected: Both tests PASS

- [ ] **Step 6: Commit**

```bash
git add signal_pipeline/__init__.py signal_pipeline/indicator_calculator.py tests/signal_pipeline/test_indicator_calculator.py
git commit -m "feat: add vectorized indicator calculator (MA/MACD/KDJ/Boll/RSI/PSY)"
```

---

### Task 3: Tushare Pro Client

**Files:**
- Create: `signal_pipeline/data_sources/__init__.py`
- Create: `signal_pipeline/data_sources/tushare_client.py`
- Create: `tests/signal_pipeline/test_tushare_client.py`

- [ ] **Step 1: Write the test**

```python
# tests/signal_pipeline/test_tushare_client.py
import os
import pytest
from signal_pipeline.data_sources.tushare_client import TushareClient


@pytest.fixture
def client():
    token = os.environ.get('TUSHARE_TOKEN', 'test_token')
    return TushareClient(token)


def test_client_init():
    c = TushareClient('dummy_token')
    assert c.token == 'dummy_token'
    assert c.max_retries == 3


def test_retry_on_failure(mocker):
    """Simulate 2 failures then success."""
    mock_pro = mocker.patch('signal_pipeline.data_sources.tushare_client.ts.pro_api')
    mock_api = mocker.MagicMock()
    mock_api.daily.side_effect = [Exception('timeout'), Exception('timeout'), mocker.MagicMock()]
    mock_pro.return_value = mock_api

    client = TushareClient('token')
    result = client._call_with_retry(mock_api.daily, trade_date='20260428')
    assert mock_api.daily.call_count == 3
```

- [ ] **Step 2: Implement Tushare client**

```python
# signal_pipeline/data_sources/tushare_client.py
import time
import logging
import pandas as pd
from typing import Callable

logger = logging.getLogger(__name__)


class TushareClient:
    """Tushare Pro data client with retry logic."""

    def __init__(self, token: str, max_retries: int = 3, retry_delay: int = 120):
        self.token = token
        self.max_retries = max_retries
        self.retry_delay = retry_delay  # seconds between retries
        self._pro = None

    @property
    def pro(self):
        if self._pro is None:
            import tushare as ts
            self._pro = ts.pro_api(self.token)
        return self._pro

    def get_daily_all(self, trade_date: str) -> pd.DataFrame:
        """Get daily OHLCV for all stocks on a given date (YYYYMMDD)."""
        return self._call_with_retry(
            lambda: self.pro.daily(trade_date=trade_date)
        )

    def get_daily_basic_all(self, trade_date: str) -> pd.DataFrame:
        """Get daily basic (turnover_rate, PE, PB, market cap) for all stocks."""
        return self._call_with_retry(
            lambda: self.pro.daily_basic(trade_date=trade_date)
        )

    def get_adj_factor_all(self, trade_date: str) -> pd.DataFrame:
        """Get adjustment factors for all stocks (for 复权 price calculation)."""
        return self._call_with_retry(
            lambda: self.pro.adj_factor(trade_date=trade_date)
        )

    def _call_with_retry(self, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = fn()
                if isinstance(result, pd.DataFrame) and not result.empty:
                    return result
                if attempt < self.max_retries - 1:
                    logger.warning(f"Empty result, retry {attempt + 2}/{self.max_retries}")
                    time.sleep(self.retry_delay)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(f"Tushare error: {e}, retry in {self.retry_delay}s ({attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)

        raise RuntimeError(f"Tushare call failed after {self.max_retries} attempts: {last_error}")
```

- [ ] **Step 3: Run the test**

```bash
pytest tests/signal_pipeline/test_tushare_client.py -v
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add signal_pipeline/data_sources/ signal_pipeline/__init__.py tests/signal_pipeline/test_tushare_client.py
git commit -m "feat: add Tushare Pro client with retry logic"
```

---

### Task 4: AKShare Client for Intraday Snapshots

**Files:**
- Create: `signal_pipeline/data_sources/akshare_client.py`
- Create: `tests/signal_pipeline/test_akshare_client.py`

- [ ] **Step 1: Write the test**

```python
# tests/signal_pipeline/test_akshare_client.py
import pytest
from signal_pipeline.data_sources.akshare_client import AKShareClient


def test_get_spot_all_columns():
    client = AKShareClient(max_retries=1)
    # We test the column mapping, not the actual API call
    assert client.COLUMN_MAP['最新价'] == 'close'
    assert client.COLUMN_MAP['涨跌幅'] == 'change_pct'
    assert client.COLUMN_MAP['换手率'] == 'turnover_rate'


def test_retry_raises_after_max_attempts(mocker):
    mock_ak = mocker.patch('signal_pipeline.data_sources.akshare_client.ak.stock_zh_a_spot_em')
    mock_ak.side_effect = Exception('network error')

    client = AKShareClient(max_retries=2, retry_delay=0)
    with pytest.raises(RuntimeError, match='2 attempts'):
        client.get_spot_all()
```

- [ ] **Step 2: Implement AKShare client**

```python
# signal_pipeline/data_sources/akshare_client.py
import time
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class AKShareClient:
    """AKShare real-time data client for intraday snapshots."""

    # Map AKShare Chinese column names to our standard names
    COLUMN_MAP = {
        '代码': 'stock_code',
        '名称': 'stock_name',
        '最新价': 'close',
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'turnover_amount',
        '振幅': 'amplitude',
        '涨跌幅': 'change_pct',
        '换手率': 'turnover_rate',
        '量比': 'volume_ratio',
        '市盈率-动态': 'pe_ttm',
        '市净率': 'pb',
        '流通市值': 'circulating_mv',
        '总市值': 'total_mv',
    }

    def __init__(self, max_retries: int = 3, retry_delay: int = 120):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def get_spot_all(self) -> pd.DataFrame:
        """Get real-time spot data for all A-share stocks.

        Returns DataFrame with standardized column names.
        ~5000 rows, one per stock.
        """
        df = self._call_with_retry()
        df = df.rename(columns=self.COLUMN_MAP)
        # Keep only columns we have mappings for
        keep_cols = [c for c in self.COLUMN_MAP.values() if c in df.columns]
        return df[keep_cols]

    def _call_with_retry(self) -> pd.DataFrame:
        import akshare as ak
        last_error = None
        for attempt in range(self.max_retries):
            try:
                df = ak.stock_zh_a_spot_em()
                if not df.empty:
                    return df
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(f"AKShare error: {e}, retry in {self.retry_delay}s ({attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)

        raise RuntimeError(f"AKShare call failed after {self.max_retries} attempts: {last_error}")
```

- [ ] **Step 3: Run the test**

```bash
pytest tests/signal_pipeline/test_akshare_client.py -v
```
Expected: Both tests PASS

- [ ] **Step 4: Commit**

```bash
git add signal_pipeline/data_sources/akshare_client.py tests/signal_pipeline/test_akshare_client.py
git commit -m "feat: add AKShare client for intraday spot data"
```

---

### Task 5: Data Merger — Intraday + Historical

**Files:**
- Create: `signal_pipeline/data_merger.py`
- Create: `tests/signal_pipeline/test_data_merger.py`

- [ ] **Step 1: Write the test**

```python
# tests/signal_pipeline/test_data_merger.py
import pandas as pd
from signal_pipeline.data_merger import DataMerger


def test_merge_intraday_with_history():
    # Simulate yesterday's complete data from DB
    history = pd.DataFrame({
        'trade_date': pd.to_datetime(['2024-01-03', '2024-01-03']),
        'stock_code': ['sh600001', 'sz000001'],
        'open': [10.0, 20.0],
        'high': [10.5, 20.8],
        'low': [9.8, 19.5],
        'close': [10.2, 20.5],
        'volume': [10000, 20000],
        'turnover_amount': [100000, 400000],
        'amplitude': [5.0, 3.0],
        'change_pct': [2.0, 1.5],
        'turnover_rate': [1.0, 2.0],
        'volume_ratio': [1.2, 0.9],
        'circulating_mv': [1e9, 2e9],
        'total_mv': [5e9, 8e9],
        'pe_ttm': [15.0, 25.0],
        'ps_ttm': [2.0, 3.0],
        'pb': [1.5, 3.0],
    })

    # Simulate today's intraday snapshot at 2:30 PM
    intraday = pd.DataFrame({
        'stock_code': ['sh600001', 'sz000001'],
        'stock_name': ['股票A', '股票B'],
        'open': [10.3, 20.6],
        'high': [10.8, 21.0],
        'low': [10.1, 20.4],
        'close': [10.6, 20.9],  # current price as close
        'volume': [5000, 10000],
        'turnover_amount': [50000, 200000],
        'amplitude': [4.0, 2.0],
        'change_pct': [3.9, 1.95],
        'turnover_rate': [0.5, 1.0],
        'volume_ratio': [1.1, 1.0],
        'pe_ttm': [15.5, 25.5],
        'pb': [1.6, 3.1],
        'circulating_mv': [1.05e9, 2.1e9],
        'total_mv': [5.2e9, 8.2e9],
    })

    today_date = pd.Timestamp('2024-01-04')
    result = DataMerger.merge(history, intraday, today_date)

    # Should have 4 rows: 2 from history + 2 from today
    assert len(result) == 4
    # Today's rows should have the correct date
    today_rows = result[result['trade_date'] == today_date]
    assert len(today_rows) == 2
    # Today's close should be the intraday price
    assert today_rows[today_rows['stock_code'] == 'sh600001']['close'].values[0] == 10.6
```

- [ ] **Step 2: Implement DataMerger**

```python
# signal_pipeline/data_merger.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataMerger:
    """Merge intraday snapshot with historical data for signal generation."""

    @staticmethod
    def merge(
        history: pd.DataFrame,
        intraday: pd.DataFrame,
        today_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Merge yesterday's complete data with today's intraday snapshot.

        Args:
            history: Complete daily data from DB (last N days, all stocks).
                     Must have columns: trade_date, stock_code, open, high, low,
                     close, volume, and all pre-computed indicators.
            intraday: Intraday snapshot from AKShare (one row per stock).
                      Will be treated as today's "close" price.
            today_date: The trade date for today.

        Returns:
            Merged DataFrame sorted by stock_code, trade_date.
            Today's rows only have basic OHLCV + available fields from intraday.
        """
        # Ensure history doesn't already have today's data
        history = history[history['trade_date'] < today_date].copy()

        # Build today's rows from intraday snapshot
        today_rows = intraday.copy()
        today_rows['trade_date'] = today_date

        # Ensure required columns exist on today's rows
        for col in ['volume', 'turnover_amount', 'amplitude', 'change_pct',
                     'turnover_rate', 'volume_ratio', 'pe_ttm', 'pb',
                     'circulating_mv', 'total_mv', 'stock_name']:
            if col not in today_rows.columns:
                today_rows[col] = None

        # Determine which stocks appear in both history and intraday
        history_codes = set(history['stock_code'].unique())
        intraday_codes = set(intraday['stock_code'].unique())

        # Only keep stocks that have history (can't calculate indicators without it)
        common_codes = history_codes & intraday_codes
        if len(common_codes) < len(intraday_codes):
            missing = intraday_codes - common_codes
            logger.info(f"Skipping {len(missing)} stocks without history data")

        today_rows = today_rows[today_rows['stock_code'].isin(common_codes)]

        # Select common columns from history
        history_cols = [c for c in history.columns if c in today_rows.columns or c == 'trade_date']
        history_subset = history[history_cols]

        # Concat
        merged = pd.concat([history_subset, today_rows], ignore_index=True)
        merged = merged.sort_values(['stock_code', 'trade_date']).reset_index(drop=True)

        logger.info(f"Merged: {len(history_subset)} history + {len(today_rows)} today = {len(merged)} rows")

        return merged
```

- [ ] **Step 3: Run test**

```bash
pytest tests/signal_pipeline/test_data_merger.py -v
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add signal_pipeline/data_merger.py tests/signal_pipeline/test_data_merger.py
git commit -m "feat: add data merger for intraday + history combination"
```

---

### Task 6: Signal Generator — Use Existing Engine Logic

**Files:**
- Create: `signal_pipeline/signal_generator.py`
- Create: `tests/signal_pipeline/test_signal_generator.py`

**Context:** This is the core intelligence module. It reuses `SignalFilter` and `SignalRanker` from the existing codebase to detect buy/sell signals from the merged intraday data. The approach: build a feature DataFrame similar to what `_build_signal_features()` produces, then apply the same vectorized signal masks.

- [ ] **Step 1: Write the test**

```python
# tests/signal_pipeline/test_signal_generator.py
import pandas as pd
from signal_pipeline.signal_generator import SignalGenerator
from back_testing.rotation.config import RotationConfig


def test_generate_buy_signals_basic():
    config = RotationConfig()
    config.buy_signal_mode = 'OR'
    config.buy_signal_types = ['KDJ_GOLD', 'MACD_GOLD']

    generator = SignalGenerator(config)

    # Build test data with clear golden cross patterns
    today = pd.Timestamp('2024-01-10')
    yesterday = pd.Timestamp('2024-01-09')

    df = pd.DataFrame({
        'trade_date': [yesterday, today, yesterday, today],
        'stock_code': ['sh600001', 'sh600001', 'sz000001', 'sz000001'],
        'open': [10.0, 10.2, 20.0, 20.0],
        'high': [10.5, 10.6, 20.5, 20.5],
        'low': [9.8, 10.0, 19.8, 19.8],
        'close': [10.2, 10.5, 20.0, 20.2],
        'volume': [10000, 8000, 20000, 18000],
        # KDJ: K crosses above D today
        'kdj_k': [30.0, 45.0, 50.0, 40.0],
        'kdj_d': [35.0, 40.0, 45.0, 42.0],
        # MACD: DIF crosses above DEA today for sh600001
        'macd_dif': [0.05, 0.12, 0.10, 0.08],
        'macd_dea': [0.08, 0.10, 0.09, 0.09],
        'ma_5': [10.0, 10.2, 20.0, 20.0],
        'ma_20': [10.0, 10.0, 20.0, 20.0],
        'vol_ma5': [10000, 9000, 20000, 19000],
        'vol_ma20': [10000, 9500, 20000, 19500],
        'close_std_20': [0.5, 0.5, 0.3, 0.3],
        'boll_mid': [10.0, 10.1, 20.0, 20.0],
        'high_20_max': [10.8, 10.8, 20.5, 20.5],
        'psy': [50, 55, 60, 60],
        'psyma': [50, 52, 58, 59],
        'rsi_1': [50, 55, 50, 50],
        'ret_5': [0.02, 0.03, 0.01, 0.01],
        'ret_20': [0.05, 0.06, 0.02, 0.02],
        'circulating_mv': [1e9, 1.1e9, 2e9, 2.1e9],
        'pe_ttm': [15, 16, 25, 26],
        'pb': [1.5, 1.6, 3.0, 3.1],
    })

    buy_codes = generator.generate_buy_signals(df, today)
    # sh600001 should trigger (KDJ gold + MACD gold if OR mode)
    assert 'sh600001' in buy_codes
    assert len(buy_codes) > 0
```

- [ ] **Step 2: Implement SignalGenerator**

```python
# signal_pipeline/signal_generator.py
import logging
import pandas as pd
import numpy as np
from typing import List, Dict

from back_testing.rotation.config import RotationConfig
from back_testing.rotation.signal_engine.signal_filter import SignalFilter
from back_testing.rotation.signal_engine.signal_ranker import SignalRanker

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate buy/sell signals using existing engine logic.

    Reuses SignalFilter and SignalRanker from the backtesting codebase.
    Applies vectorized signal detection (same masks as engine._scan_buy_candidates).
    """

    def __init__(self, config: RotationConfig):
        self.config = config
        self.buy_filter = SignalFilter(config.buy_signal_types)
        self.ranker = SignalRanker(
            factor_weights=config.rank_factor_weights,
            factor_directions=config.rank_factor_directions,
        )

    def generate_buy_signals(
        self, df: pd.DataFrame, today: pd.Timestamp
    ) -> List[str]:
        """Return list of stock_codes with buy signals, ranked by score."""
        today_df = df[df['trade_date'] == today].copy()
        yesterday_df = df[df['trade_date'] < today].copy()

        if today_df.empty:
            return []

        # Get yesterday's data (last row per stock)
        prev_df = yesterday_df.sort_values('trade_date').groupby('stock_code').last()

        # Build feature matrix
        features = self._build_features(today_df, prev_df)
        if features.empty:
            return []

        # Apply vectorized signal masks (same logic as engine._scan_buy_candidates)
        active_signals = set(self.config.buy_signal_types)
        mode = self.config.buy_signal_mode
        masks = self._create_signal_masks(features, active_signals)

        if not masks:
            return []

        if mode == 'OR':
            combined = pd.Series(False, index=features.index)
            for m in masks.values():
                combined = combined | m.fillna(False)
        else:
            combined = pd.Series(True, index=features.index)
            for m in masks.values():
                combined = combined & m.fillna(False)

        candidates = combined[combined].index.tolist()
        if not candidates:
            return []

        # Rank candidates by multi-factor score
        factor_df = self._build_factor_df(today_df, candidates)
        ranked = self.ranker.rank(factor_df)
        return ranked.index.tolist()

    def generate_sell_signals(
        self, df: pd.DataFrame, today: pd.Timestamp, position_codes: List[str]
    ) -> List[Dict]:
        """Check if any current positions should be sold.

        Returns list of dicts: {stock_code, reason}
        """
        today_df = df[df['trade_date'] == today]
        yesterday_df = df[df['trade_date'] < today]

        if today_df.empty or not position_codes:
            return []

        prev_df = yesterday_df.sort_values('trade_date').groupby('stock_code').last()
        features = self._build_features(today_df, prev_df)
        if features.empty:
            return []

        active_signals = set(self.config.sell_signal_types)
        masks = self._create_sell_masks(features, active_signals)

        sells = []
        for signal_name, mask in masks.items():
            triggered = mask[mask.fillna(False)].index
            for code in triggered:
                if code in position_codes:
                    sells.append({'stock_code': code, 'reason': signal_name})

        return sells

    def _build_features(
        self, today_df: pd.DataFrame, prev_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Build feature matrix for signal detection.

        Same structure as engine._build_signal_features() output.
        """
        t = today_df.set_index('stock_code')
        p = prev_df

        common = t.index.intersection(p.index)
        if common.empty:
            return pd.DataFrame()

        t = t.loc[common]
        p = p.loc[common]

        return pd.DataFrame({
            'kdj_k': t['kdj_k'], 'kdj_d': t['kdj_d'],
            'kdj_k_p': p['kdj_k'], 'kdj_d_p': p['kdj_d'],
            'macd_dif': t['macd_dif'], 'macd_dea': t['macd_dea'],
            'macd_dif_p': p['macd_dif'], 'macd_dea_p': p['macd_dea'],
            'ma_5': t['ma_5'], 'ma_20': t['ma_20'],
            'ma_5_p': p['ma_5'], 'ma_20_p': p['ma_20'],
            'vol_ma5': t['vol_ma5'], 'vol_ma20': t['vol_ma20'],
            'vol_ma5_p': p['vol_ma5'], 'vol_ma20_p': p['vol_ma20'],
            'close': t['close'], 'close_std_20': t['close_std_20'],
            'boll_mid': t['boll_mid'], 'high_20_max': t['high_20_max'],
            'psy': t['psy'], 'psyma': t['psyma'],
        }, index=common)

    def _create_signal_masks(self, features, active_signals):
        """Create boolean masks for each active buy signal type."""
        f = features
        masks = {}

        if 'KDJ_GOLD' in active_signals:
            masks['KDJ_GOLD'] = (f['kdj_k'] > f['kdj_d']) & (f['kdj_k_p'] <= f['kdj_d_p'])
        if 'MACD_GOLD' in active_signals:
            masks['MACD_GOLD'] = (f['macd_dif'] > f['macd_dea']) & (f['macd_dif_p'] <= f['macd_dea_p'])
        if 'MA_GOLD' in active_signals:
            masks['MA_GOLD'] = (f['ma_5'] > f['ma_20']) & (f['ma_5_p'] <= f['ma_20_p'])
        if 'VOL_GOLD' in active_signals:
            masks['VOL_GOLD'] = (f['vol_ma5'] > f['vol_ma20']) & (f['vol_ma5_p'] <= f['vol_ma20_p'])
        if 'BOLL_BREAK' in active_signals:
            boll_upper = f['boll_mid'] + 2 * f['close_std_20']
            masks['BOLL_BREAK'] = f['close'] > boll_upper
        if 'HIGH_BREAK' in active_signals:
            masks['HIGH_BREAK'] = f['close'] >= f['high_20_max']
        if 'KDJ_GOLD_LOW' in active_signals:
            k_thresh = self.config.kdj_low_threshold
            masks['KDJ_GOLD_LOW'] = (
                (f['kdj_k'] > f['kdj_d']) & (f['kdj_k_p'] <= f['kdj_d_p']) & (f['kdj_k'] < k_thresh)
            )
        if 'PSY_BUY' in active_signals:
            masks['PSY_BUY'] = (f['psy'] < 25) & (f['psy'] > f['psyma'])

        return masks

    def _create_sell_masks(self, features, active_signals):
        """Create boolean masks for each active sell signal type."""
        f = features
        masks = {}

        if 'KDJ_DEATH' in active_signals:
            masks['KDJ_DEATH'] = (f['kdj_k'] < f['kdj_d']) & (f['kdj_k_p'] >= f['kdj_d_p'])
        if 'MACD_DEATH' in active_signals:
            masks['MACD_DEATH'] = (f['macd_dif'] < f['macd_dea']) & (f['macd_dif_p'] >= f['macd_dea_p'])
        if 'MA_DEATH' in active_signals:
            masks['MA_DEATH'] = (f['ma_5'] < f['ma_20']) & (f['ma_5_p'] >= f['ma_20_p'])
        if 'VOL_DEATH' in active_signals:
            masks['VOL_DEATH'] = (f['vol_ma5'] < f['vol_ma20']) & (f['vol_ma5_p'] >= f['vol_ma20_p'])
        if 'BOLL_BREAK_DOWN' in active_signals:
            boll_lower = f['boll_mid'] - 2 * f['close_std_20']
            masks['BOLL_BREAK_DOWN'] = f['close'] < boll_lower
        if 'HIGH_BREAK_DOWN' in active_signals:
            high_20_min = f['boll_mid']  # approximate
            masks['HIGH_BREAK_DOWN'] = f['close'] < high_20_min
        if 'PSY_SELL' in active_signals:
            masks['PSY_SELL'] = (f['psy'] > 75) & (f['psy'] < f['psyma'])

        return masks

    def _build_factor_df(
        self, today_df: pd.DataFrame, candidates: List[str]
    ) -> pd.DataFrame:
        """Build factor matrix for candidate ranking."""
        cdf = today_df[today_df['stock_code'].isin(candidates)].set_index('stock_code')
        factor_df = pd.DataFrame(index=cdf.index)

        for factor in self.ranker.factor_weights.keys():
            if factor == 'RET_20':
                factor_df[factor] = cdf['ret_20'].fillna(0.0).astype(float)
            elif factor == 'OVERHEAT':
                rsi = cdf['rsi_1']
                ret5 = cdf['ret_5'].fillna(0.0).astype(float)
                oh = pd.Series(0.0, index=cdf.index)
                rsi_t = getattr(self.config, 'overheat_rsi_threshold', 75.0)
                ret5_t = getattr(self.config, 'overheat_ret5_threshold', 0.15)
                mask = (rsi > rsi_t) & (ret5 > ret5_t)
                rsi_c = np.maximum(0.0, (rsi[mask] - rsi_t) / (100 - rsi_t))
                ret_c = np.minimum(1.0, np.maximum(0.0, (ret5[mask] - ret5_t) / 0.35))
                oh[mask] = (rsi_c + ret_c) / 2.0
                factor_df[factor] = oh
            elif factor in cdf.columns:
                factor_df[factor] = cdf[factor].fillna(0.0).astype(float)

        return factor_df
```

- [ ] **Step 3: Run test**

```bash
pytest tests/signal_pipeline/test_signal_generator.py -v
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add signal_pipeline/signal_generator.py tests/signal_pipeline/test_signal_generator.py
git commit -m "feat: add signal generator reusing engine filter/ranker logic"
```

---

### Task 7: Intraday Signal Script (14:25 cron entry)

**Files:**
- Create: `signal_pipeline/intraday_signal.py`

This is the entry point executed by cron at 14:25. It orchestrates: fetch AKShare → load history → merge → calculate indicators → generate signals → write to DB.

- [ ] **Step 1: Review the existing DataProvider `get_batch_histories` for bulk history load**

We need to load the last 60 days of history for ALL stocks at once. The existing `DataProvider` has `get_batch_histories` but let's check — actually, `DataProvider` has `get_stock_data` per-stock which is N+1. For this script we'll query directly.

- [ ] **Step 2: Write the intraday signal script**

```python
# signal_pipeline/intraday_signal.py
"""
Intraday signal generation script (cron: 25 14 * * 1-5).

1. Fetch real-time spot data via AKShare
2. Load yesterday's complete data from PostgreSQL
3. Merge + recalculate indicators
4. Generate buy/sell signals
5. Write signals to DB
"""
import sys
import logging
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from back_testing.data.db.connection import get_engine, get_session
from back_testing.data.db.models import StockDaily
from back_testing.rotation.config import RotationConfig
from signal_pipeline.data_sources.akshare_client import AKShareClient
from signal_pipeline.data_merger import DataMerger
from signal_pipeline.indicator_calculator import IndicatorCalculator
from signal_pipeline.signal_generator import SignalGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def _cron_start(task_name: str, session):
    result = session.execute(
        text("INSERT INTO cron_log (task_name, status) VALUES (:name, 'running') RETURNING id"),
        {'name': task_name}
    )
    session.commit()
    return result.fetchone()[0]


def _cron_finish(log_id: int, status: str, session, error: str = None, metadata: dict = None):
    session.execute(
        text("UPDATE cron_log SET status=:status, finished_at=NOW(), error_message=:err, metadata=:meta WHERE id=:id"),
        {'status': status, 'err': error, 'meta': str(metadata) if metadata else None, 'id': log_id}
    )
    session.commit()


def _cron_start(task_name: str, session):
    """Insert a 'running' cron_log entry, return the log id."""
    result = session.execute(
        text("INSERT INTO cron_log (task_name, status) VALUES (:name, 'running') RETURNING id"),
        {'name': task_name}
    )
    session.commit()
    return result.fetchone()[0]


def _cron_finish(log_id: int, status: str, session, error: str = None, metadata: dict = None):
    session.execute(
        text("UPDATE cron_log SET status=:status, finished_at=NOW(), error_message=:err, metadata=:meta WHERE id=:id"),
        {'status': status, 'err': error, 'meta': str(metadata) if metadata else None, 'id': log_id}
    )
    session.commit()


def main():
    logger.info("=" * 50)
    logger.info("盘中信号生成开始")
    logger.info("=" * 50)

    Session = get_session()
    session = Session()
    log_id = _cron_start('intraday_signal', session)

    try:
        # 1. Fetch intraday snapshot
        logger.info("Step 1/5: 获取 AKShare 实时快照...")
        akshare = AKShareClient(max_retries=3, retry_delay=60)
        intraday_df = akshare.get_spot_all()
        logger.info(f"获取到 {len(intraday_df)} 只股票")

        # 2. Load history from DB (last 60 days for all stocks)
        logger.info("Step 2/5: 加载历史数据...")
        today = date.today()
        start_date = today - timedelta(days=60)
        hist_df = pd.read_sql(
            text("SELECT * FROM stock_daily WHERE trade_date >= :start"),
            get_engine(),
            params={'start': start_date}
        )
        logger.info(f"历史数据: {len(hist_df)} 行")

        # 3. Merge
        logger.info("Step 3/5: 合并日内+历史数据...")
        today_ts = pd.Timestamp(today)
        merged_df = DataMerger.merge(hist_df, intraday_df, today_ts)
        logger.info(f"合并后: {len(merged_df)} 行")

        # 4. Calculate indicators
        logger.info("Step 4/5: 重算技术指标...")
        merged_df = IndicatorCalculator.calculate_all(merged_df)
        logger.info("指标计算完成")

        # 5. Generate signals
        logger.info("Step 5/5: 生成交易信号...")
        config = RotationConfig()
        generator = SignalGenerator(config)

        # Get current positions for sell signal checking
        position_codes = [
            r[0] for r in session.execute(
                text("SELECT stock_code FROM position WHERE status='OPEN'")
            ).fetchall()
        ]
        logger.info(f"当前持仓: {position_codes}")

        buy_codes = generator.generate_buy_signals(merged_df, today_ts)
        sell_signals = generator.generate_sell_signals(merged_df, today_ts, position_codes)

        logger.info(f"买入信号: {len(buy_codes)} 只")
        logger.info(f"卖出信号: {len(sell_signals)} 只")

        # 6. Write signals to DB
        today_df = merged_df[merged_df['trade_date'] == today_ts]
        today_lookup = today_df.set_index('stock_code') if not today_df.empty else {}

        # Clear today's previous signals (if re-running)
        session.execute(
            text("DELETE FROM daily_signal WHERE trade_date = :d"),
            {'d': today}
        )

        signal_count = 0

        # Write buy signals
        for i, code in enumerate(buy_codes):
            if not today_lookup.empty and code in today_lookup.index:
                row = today_lookup.loc[code]
                name = row.get('stock_name', code)
                price = float(row['close'])
            else:
                name = code
                price = 0.0

            session.execute(
                text("""
                    INSERT INTO daily_signal
                    (trade_date, stock_code, stock_name, direction, target_pct,
                     price_low, price_high, signal_reason, status)
                    VALUES (:d, :code, :name, 'BUY', :pct, :lo, :hi, :reason, 'pending')
                """),
                {
                    'd': today, 'code': code, 'name': str(name),
                    'pct': float(config.max_position_pct * 100),
                    'lo': price * 0.985, 'hi': price * 1.015,
                    'reason': f'多因子排名 #{i + 1}'
                }
            )
            signal_count += 1

        # Write sell signals
        for s in sell_signals:
            code = s['stock_code']
            session.execute(
                text("""
                    INSERT INTO daily_signal
                    (trade_date, stock_code, stock_name, direction, target_pct,
                     price_low, price_high, signal_reason, status)
                    VALUES (:d, :code, :code, 'SELL', NULL, NULL, NULL, :reason, 'pending')
                """),
                {'d': today, 'code': code, 'reason': s['reason']}
            )
            signal_count += 1

        session.commit()

        _cron_finish(log_id, 'success', session, metadata={
            'buy_count': len(buy_codes),
            'sell_count': len(sell_signals),
            'total_signals': signal_count,
            'intraday_stocks': len(intraday_df),
        })
        logger.info(f"信号写入完成: {signal_count} 条")
        logger.info("盘中信号生成完成")

    except Exception as e:
        logger.error(f"盘中信号生成失败: {e}", exc_info=True)
        _cron_finish(log_id, 'failed', session, error=str(e))
        sys.exit(1)
    finally:
        session.close()


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Commit**

```bash
git add signal_pipeline/intraday_signal.py
git commit -m "feat: add intraday signal script (14:25 cron entry)"
```

---

### Task 8: Night Backfill Script (18:00 cron entry)

**Files:**
- Create: `signal_pipeline/night_backfill.py`

- [ ] **Step 1: Write the night backfill script**

This script: (1) fetches complete daily data from Tushare, (2) merges with daily_basic for valuation, (3) gets adj_factor for 复权, (4) calculates indicators, (5) upserts to DB, (6) incrementally builds Parquet cache, (7) logs to cron_log.

```python
# signal_pipeline/night_backfill.py
"""
Nightly data backfill script (cron: 0 18 * * 1-5).

1. Fetch complete daily data from Tushare Pro
2. Calculate all technical indicators
3. Upsert into stock_daily table
4. Incrementally build Parquet cache
5. Update cron_log
"""
import os
import sys
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from back_testing.data.db.connection import get_engine, get_session
from back_testing.data.db.models import StockDaily
from back_testing.data.daily_data_cache import DailyDataCache
from signal_pipeline.data_sources.tushare_client import TushareClient
from signal_pipeline.indicator_calculator import IndicatorCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def _get_last_trade_date() -> date:
    """Get the latest trade_date with <5% null indicators (i.e. properly backfilled)."""
    from back_testing.data.db.connection import get_engine
    result = pd.read_sql(
        "SELECT trade_date FROM stock_daily WHERE macd_dif IS NOT NULL GROUP BY 1 ORDER BY 1 DESC LIMIT 1",
        get_engine()
    )
    return result['trade_date'].iloc[0] if not result.empty else date.today() - timedelta(days=30)


def main():
    logger.info("=" * 50)
    logger.info("盘后数据补全开始")
    logger.info("=" * 50)

    Session = get_session()
    session = Session()
    log_id = _cron_start('night_backfill', session)

    token = os.environ.get('TUSHARE_TOKEN')
    if not token:
        logger.error("TUSHARE_TOKEN not set")
        _cron_finish(log_id, 'failed', session, error='TUSHARE_TOKEN not set')
        sys.exit(1)

    try:
        tushare = TushareClient(token)
        today = date.today()
        today_str = today.strftime('%Y%m%d')

        # 1. Fetch daily OHLCV from Tushare
        logger.info("Step 1/5: 拉取 Tushare 日线数据...")
        daily_df = tushare.get_daily_all(today_str)
        if daily_df.empty:
            logger.warning("Tushare daily 返回空（可能今日非交易日）")
            _cron_finish(log_id, 'success', session, metadata={'note': 'no trading day'})
            return

        # Convert ts_code to our format: 000001.SZ → sz000001
        daily_df['stock_code'] = daily_df['ts_code'].apply(_convert_ts_code)
        daily_df['trade_date'] = pd.to_datetime(daily_df['trade_date'])
        daily_df = daily_df.rename(columns={
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'vol': 'volume', 'amount': 'turnover_amount',
            'pct_chg': 'change_pct',
        })

        # 2. Fetch daily_basic (valuation + market cap)
        logger.info("Step 2/5: 拉取 Tushare daily_basic...")
        basic_df = tushare.get_daily_basic_all(today_str)
        if not basic_df.empty:
            basic_df['stock_code'] = basic_df['ts_code'].apply(_convert_ts_code)
            daily_df = daily_df.merge(
                basic_df[['stock_code', 'turnover_rate', 'volume_ratio',
                           'pe_ttm', 'pb', 'ps_ttm', 'total_mv', 'circ_mv']],
                on='stock_code', how='left'
            )
            daily_df = daily_df.rename(columns={
                'circ_mv': 'circulating_mv',
            })

        # 3. Calculate indicators
        logger.info("Step 3/5: 计算技术指标...")
        # Load last 60 days for context
        hist_start = today - timedelta(days=60)
        hist_df = pd.read_sql(
            text("SELECT * FROM stock_daily WHERE trade_date >= :start"),
            get_engine(),
            params={'start': hist_start}
        )

        # Combine history + today, calculate indicators
        combined = pd.concat([hist_df, daily_df], ignore_index=True)
        combined = combined.sort_values(['stock_code', 'trade_date'])
        combined = IndicatorCalculator.calculate_all(combined)

        # Extract only today's rows with indicators
        today_with_indicators = combined[combined['trade_date'] == pd.Timestamp(today)]
        logger.info(f"今日数据: {len(today_with_indicators)} 行（含指标）")

        # 4. Upsert into stock_daily
        logger.info("Step 4/5: 写入 PostgreSQL...")
        upsert_count = 0
        for _, row in today_with_indicators.iterrows():
            data = row.to_dict()
            # Convert numpy types
            for k, v in data.items():
                if isinstance(v, (np.integer,)):
                    data[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    data[k] = float(v)
                elif pd.isna(v):
                    data[k] = None

            stmt = pg_insert(StockDaily).values(**data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['stock_code', 'trade_date'],
                set_={k: stmt.excluded[k] for k in data if k not in ('stock_code', 'trade_date')}
            )
            session.execute(stmt)
            upsert_count += 1

        session.commit()
        logger.info(f"写入/更新: {upsert_count} 行")

        # 5. Incremental Parquet cache build
        logger.info("Step 5/5: 增量构建 Parquet 缓存...")
        cache_dir = Path(os.environ.get('CACHE_DIR', 'cache/daily_rotation'))
        daily_dir = cache_dir / 'daily'
        daily_dir.mkdir(parents=True, exist_ok=True)

        today_parquet = daily_dir / f'{today}.parquet'
        today_with_indicators.to_parquet(today_parquet, index=False)

        # Update trading_dates.parquet
        dates_path = cache_dir / 'trading_dates.parquet'
        if dates_path.exists():
            existing_dates = pd.read_parquet(dates_path)
            new_dates = pd.DataFrame({'trade_date': [pd.Timestamp(today)]})
            all_dates = pd.concat([existing_dates, new_dates]).drop_duplicates()
            all_dates.to_parquet(dates_path, index=False)
        else:
            pd.DataFrame({'trade_date': [pd.Timestamp(today)]}).to_parquet(dates_path, index=False)

        logger.info(f"缓存已更新: {today_parquet}")

        _cron_finish(log_id, 'success', session, metadata={
            'stocks_updated': upsert_count,
            'tushare_rows': len(daily_df),
        })
        logger.info("盘后数据补全完成")

    except Exception as e:
        logger.error(f"盘后补全失败: {e}", exc_info=True)
        _cron_finish(log_id, 'failed', session, error=str(e))
        sys.exit(1)
    finally:
        session.close()


def _convert_ts_code(ts_code: str) -> str:
    """000001.SZ → sz000001, 600519.SH → sh600519"""
    code, exchange = ts_code.split('.')
    prefix = {'SH': 'sh', 'SZ': 'sz', 'BJ': 'bj'}.get(exchange, exchange.lower())
    return f'{prefix}{code}'


def _cron_start(task_name: str, session):
    result = session.execute(
        text("INSERT INTO cron_log (task_name, status) VALUES (:name, 'running') RETURNING id"),
        {'name': task_name}
    )
    session.commit()
    return result.fetchone()[0]


def _cron_finish(log_id: int, status: str, session, error: str = None, metadata: dict = None):
    session.execute(
        text("UPDATE cron_log SET status=:status, finished_at=NOW(), error_message=:err, metadata=:meta WHERE id=:id"),
        {'status': status, 'err': error, 'meta': str(metadata) if metadata else None, 'id': log_id}
    )
    session.commit()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add signal_pipeline/night_backfill.py
git commit -m "feat: add night backfill script with Tushare fetch + cache build"
```

---

### Task 9: FastAPI Server Setup + Pydantic Schemas

**Files:**
- Create: `web/server/main.py`
- Create: `web/server/__init__.py`
- Create: `web/server/models/schemas.py`
- Create: `web/server/models/__init__.py`

- [ ] **Step 1: Write Pydantic schemas**

```python
# web/server/models/schemas.py
from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime


class SignalOut(BaseModel):
    id: int
    trade_date: date
    stock_code: str
    stock_name: Optional[str]
    direction: str
    target_pct: Optional[float]
    price_low: Optional[float]
    price_high: Optional[float]
    signal_reason: Optional[str]
    status: str
    executed_price: Optional[float]
    confirmed_at: Optional[datetime]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class SignalConfirm(BaseModel):
    executed_price: float


class PositionOut(BaseModel):
    id: int
    stock_code: str
    stock_name: Optional[str]
    buy_date: date
    buy_price: float
    shares: int
    sell_date: Optional[date]
    sell_price: Optional[float]
    profit_pct: Optional[float]
    status: str

    class Config:
        from_attributes = True


class CapitalLedgerOut(BaseModel):
    id: int
    event_type: str
    amount: float
    balance_after: float
    related_signal_id: Optional[int]
    note: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class DepositRequest(BaseModel):
    amount: float
    note: Optional[str] = None


class CronLogOut(BaseModel):
    id: int
    task_name: str
    status: str
    started_at: datetime
    finished_at: Optional[datetime]
    error_message: Optional[str]
    metadata: Optional[dict]

    class Config:
        from_attributes = True


class AssetOverview(BaseModel):
    total_asset: float
    available_cash: float
    position_value: float
    total_profit: float
    total_profit_pct: float


class StockDailyOut(BaseModel):
    stock_code: str
    trade_date: date
    stock_name: Optional[str]
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: Optional[float]
    change_pct: Optional[float]
    turnover_rate: Optional[float]
    pe_ttm: Optional[float]
    pb: Optional[float]
    circulating_mv: Optional[float]

    class Config:
        from_attributes = True
```

- [ ] **Step 2: Write FastAPI server entry**

```python
# web/server/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from web.server.api import signals, positions, data_browser, cron_status

app = FastAPI(title="NJ Quant Signal Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(positions.router, prefix="/api/positions", tags=["positions"])
app.include_router(data_browser.router, prefix="/api/data", tags=["data"])
app.include_router(cron_status.router, prefix="/api/cron", tags=["cron"])


@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": __import__('datetime').datetime.now().isoformat()}
```

- [ ] **Step 3: Commit**

```bash
git add web/server/
git commit -m "feat: add FastAPI server setup and Pydantic schemas"
```

---

### Task 10: API — Signals Endpoints

**Files:**
- Create: `web/server/api/signals.py`
- Create: `web/server/api/__init__.py`

- [ ] **Step 1: Write signals API**

```python
# web/server/api/signals.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List, Optional

from back_testing.data.db.connection import get_session as _get_session
from web.server.models.schemas import SignalOut, SignalConfirm

router = APIRouter()


def get_session():
    session = _get_session()()
    try:
        yield session
    finally:
        session.close()


@router.get("/", response_model=List[SignalOut])
def list_signals(
    trade_date: Optional[str] = None,
    status: Optional[str] = None,
    session: Session = Depends(get_session),
):
    """List signals, optionally filtered by date and status."""
    query = "SELECT * FROM daily_signal WHERE 1=1"
    params = {}
    if trade_date:
        query += " AND trade_date = :date"
        params['date'] = trade_date
    if status:
        query += " AND status = :status"
        params['status'] = status
    query += " ORDER BY created_at DESC LIMIT 200"

    result = session.execute(text(query), params)
    return [dict(row._mapping) for row in result]


@router.post("/{signal_id}/confirm")
def confirm_signal(
    signal_id: int,
    body: SignalConfirm,
    session: Session = Depends(get_session),
):
    """Confirm execution of a signal with actual price. Updates capital ledger and positions."""
    signal = session.execute(
        text("SELECT * FROM daily_signal WHERE id = :id"),
        {'id': signal_id}
    ).fetchone()
    if not signal:
        raise HTTPException(404, "Signal not found")
    if signal.status != 'pending':
        raise HTTPException(400, f"Signal already {signal.status}")

    s = dict(signal._mapping)

    # Get current balance
    balance_row = session.execute(
        text("SELECT balance_after FROM capital_ledger ORDER BY id DESC LIMIT 1")
    ).fetchone()
    current_balance = float(balance_row[0]) if balance_row else 0

    if s['direction'] == 'BUY':
        # Calculate shares from target_pct
        buy_amount = current_balance * (float(s['target_pct'] or 5) / 100)
        shares = int(buy_amount / body.executed_price / 100) * 100  # round to lot
        if shares <= 0:
            raise HTTPException(400, "Insufficient funds for even 1 lot")

        cost = shares * body.executed_price
        new_balance = current_balance - cost

        # Create position
        session.execute(
            text("""
                INSERT INTO position (stock_code, stock_name, buy_date, buy_price, shares)
                VALUES (:code, :name, :date, :price, :shares)
            """),
            {'code': s['stock_code'], 'name': s['stock_name'], 'date': s['trade_date'],
             'price': body.executed_price, 'shares': shares}
        )

        # Update capital ledger
        session.execute(
            text("""
                INSERT INTO capital_ledger (event_type, amount, balance_after, related_signal_id)
                VALUES ('BUY', :amount, :balance, :sid)
            """),
            {'amount': -cost, 'balance': new_balance, 'sid': signal_id}
        )

        session.execute(
            text("UPDATE daily_signal SET status='confirmed', executed_price=:price, confirmed_at=NOW() WHERE id=:id"),
            {'price': body.executed_price, 'id': signal_id}
        )

    elif s['direction'] == 'SELL':
        # Find the open position for this stock
        pos = session.execute(
            text("SELECT * FROM position WHERE stock_code=:code AND status='OPEN' ORDER BY buy_date DESC LIMIT 1"),
            {'code': s['stock_code']}
        ).fetchone()
        if not pos:
            raise HTTPException(400, "No open position found for this stock")

        p = dict(pos._mapping)
        revenue = p['shares'] * body.executed_price
        new_balance = current_balance + revenue

        # Close position
        profit_pct = (body.executed_price - float(p['buy_price'])) / float(p['buy_price']) * 100
        session.execute(
            text("""
                UPDATE position SET sell_date=:date, sell_price=:price,
                profit_pct=:profit, status='CLOSED' WHERE id=:id
            """),
            {'date': s['trade_date'], 'price': body.executed_price,
             'profit': profit_pct, 'id': p['id']}
        )

        # Update capital ledger
        session.execute(
            text("""
                INSERT INTO capital_ledger (event_type, amount, balance_after, related_signal_id)
                VALUES ('SELL', :amount, :balance, :sid)
            """),
            {'amount': revenue, 'balance': new_balance, 'sid': signal_id}
        )

        session.execute(
            text("UPDATE daily_signal SET status='confirmed', executed_price=:price, confirmed_at=NOW() WHERE id=:id"),
            {'price': body.executed_price, 'id': signal_id}
        )

    session.commit()
    return {"ok": True}


@router.post("/{signal_id}/skip")
def skip_signal(signal_id: int, session: Session = Depends(get_session)):
    """Mark a signal as skipped (放弃)."""
    session.execute(
        text("UPDATE daily_signal SET status='skipped', confirmed_at=NOW() WHERE id=:id"),
        {'id': signal_id}
    )
    session.commit()
    return {"ok": True}
```

- [ ] **Step 2: Commit**

```bash
git add web/server/api/
git commit -m "feat: add signals API (list, confirm, skip)"
```

---

### Task 11: API — Positions + Capital Endpoints

**Files:**
- Create: `web/server/api/positions.py`

- [ ] **Step 1: Write positions API**

```python
# web/server/api/positions.py
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List

from back_testing.data.db.connection import get_session as _get_session
from web.server.models.schemas import (
    PositionOut, CapitalLedgerOut, AssetOverview, DepositRequest
)

router = APIRouter()


def get_session():
    return _get_session()()


@router.get("/overview", response_model=AssetOverview)
def asset_overview(session: Session = Depends(get_session)):
    """Get total asset overview."""
    # Available cash
    balance = session.execute(
        text("SELECT balance_after FROM capital_ledger ORDER BY id DESC LIMIT 1")
    ).fetchone()
    available_cash = float(balance[0]) if balance else 0

    # Position market value (approximate: use buy_price since we don't have live prices here)
    positions = session.execute(
        text("SELECT shares, buy_price FROM position WHERE status='OPEN'")
    ).fetchall()
    position_value = sum(float(p.shares) * float(p.buy_price) for p in positions)

    total_asset = available_cash + position_value

    # Total profit from closed positions
    profit = session.execute(
        text("SELECT SUM(profit_pct * shares * buy_price / 100) FROM position WHERE status='CLOSED'")
    ).fetchone()[0] or 0

    return AssetOverview(
        total_asset=total_asset,
        available_cash=available_cash,
        position_value=position_value,
        total_profit=profit,
        total_profit_pct=(profit / 100000 * 100) if profit else 0,
    )


@router.get("/", response_model=List[PositionOut])
def list_positions(
    status: str = None,
    session: Session = Depends(get_session),
):
    query = "SELECT * FROM position"
    params = {}
    if status:
        query += " WHERE status = :status"
        params['status'] = status
    query += " ORDER BY buy_date DESC LIMIT 100"

    result = session.execute(text(query), params)
    return [dict(row._mapping) for row in result]


@router.get("/capital", response_model=List[CapitalLedgerOut])
def capital_history(session: Session = Depends(get_session)):
    result = session.execute(
        text("SELECT * FROM capital_ledger ORDER BY id DESC LIMIT 50")
    )
    return [dict(row._mapping) for row in result]


@router.post("/capital/deposit")
def deposit(body: DepositRequest, session: Session = Depends(get_session)):
    """Add funds to available capital."""
    balance_row = session.execute(
        text("SELECT balance_after FROM capital_ledger ORDER BY id DESC LIMIT 1")
    ).fetchone()
    current = float(balance_row[0]) if balance_row else 0
    new_balance = current + body.amount

    session.execute(
        text("""
            INSERT INTO capital_ledger (event_type, amount, balance_after, note)
            VALUES ('DEPOSIT', :amount, :balance, :note)
        """),
        {'amount': body.amount, 'balance': new_balance, 'note': body.note}
    )
    session.commit()
    return {"ok": True, "balance_after": new_balance}
```

- [ ] **Step 2: Commit**

```bash
git add web/server/api/positions.py
git commit -m "feat: add positions & capital API (overview, deposit, history)"
```

---

### Task 12: API — Data Browser + Cron Status Endpoints

**Files:**
- Create: `web/server/api/data_browser.py`
- Create: `web/server/api/cron_status.py`

- [ ] **Step 1: Write data browser API**

```python
# web/server/api/data_browser.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List, Optional

from back_testing.data.db.connection import get_session as _get_session
from web.server.models.schemas import StockDailyOut

router = APIRouter()


def get_session():
    return _get_session()()


@router.get("/stocks")
def list_stocks(
    search: Optional[str] = Query(None, description="代码/名称搜索"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
):
    """Paginated stock list."""
    offset = (page - 1) * page_size
    where = ""
    params = {'limit': page_size, 'offset': offset}
    if search:
        where = "WHERE stock_code ILIKE :search OR stock_name ILIKE :search2"
        params['search'] = f"%{search}%"
        params['search2'] = f"%{search}%"

    total = session.execute(
        text(f"SELECT COUNT(DISTINCT stock_code) FROM stock_daily {where}"), params
    ).fetchone()[0]

    rows = session.execute(
        text(f"""
            SELECT DISTINCT ON (stock_code) stock_code, stock_name, industry, close, change_pct,
                   turnover_rate, pe_ttm, pb, circulating_mv, trade_date
            FROM stock_daily {where}
            ORDER BY stock_code, trade_date DESC
            LIMIT :limit OFFSET :offset
        """),
        params
    ).fetchall()

    return {
        'total': total,
        'page': page,
        'data': [dict(r._mapping) for r in rows],
    }


@router.get("/stocks/{stock_code}")
def get_stock_detail(
    stock_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    session: Session = Depends(get_session),
):
    """Get daily data for a specific stock."""
    query = "SELECT * FROM stock_daily WHERE stock_code = :code"
    params = {'code': stock_code}
    if start_date:
        query += " AND trade_date >= :start"
        params['start'] = start_date
    if end_date:
        query += " AND trade_date <= :end"
        params['end'] = end_date
    query += " ORDER BY trade_date ASC LIMIT 500"

    result = session.execute(text(query), params)
    return [dict(r._mapping) for r in result]
```

- [ ] **Step 2: Write cron status API**

```python
# web/server/api/cron_status.py
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List

from back_testing.data.db.connection import get_session as _get_session
from web.server.models.schemas import CronLogOut

router = APIRouter()


def get_session():
    return _get_session()()


@router.get("/", response_model=List[CronLogOut])
def list_cron_logs(session: Session = Depends(get_session)):
    result = session.execute(
        text("SELECT *, metadata::text as metadata FROM cron_log ORDER BY id DESC LIMIT 100")
    )
    return [dict(row._mapping) for row in result]


@router.get("/status")
def data_completeness(session: Session = Depends(get_session)):
    """Quick check: when was data last updated?"""
    last_daily = session.execute(
        text("SELECT MAX(trade_date) FROM stock_daily")
    ).fetchone()[0]

    last_signal = session.execute(
        text("SELECT MAX(trade_date) FROM daily_signal")
    ).fetchone()[0]

    last_backfill = session.execute(
        text("SELECT status, finished_at FROM cron_log WHERE task_name='night_backfill' ORDER BY id DESC LIMIT 1")
    ).fetchone()

    return {
        'last_daily_date': str(last_daily) if last_daily else None,
        'last_signal_date': str(last_signal) if last_signal else None,
        'last_backfill': {
            'status': last_backfill.status if last_backfill else None,
            'finished_at': str(last_backfill.finished_at) if last_backfill else None,
        }
    }
```

- [ ] **Step 3: Commit**

```bash
git add web/server/api/data_browser.py web/server/api/cron_status.py
git commit -m "feat: add data browser and cron status API endpoints"
```

---

### Task 13: React Frontend — Project Setup + App Shell

**Files:**
- Create: `web/frontend/package.json`
- Create: `web/frontend/vite.config.ts`
- Create: `web/frontend/tsconfig.json`
- Create: `web/frontend/index.html`
- Create: `web/frontend/src/main.tsx`
- Create: `web/frontend/src/App.tsx`

- [ ] **Step 1: Initialize React project with Vite**

```bash
cd web/frontend
npm create vite@latest . -- --template react-ts
npm install antd @ant-design/icons @ant-design/charts axios react-router-dom
```

- [ ] **Step 2: Write App shell with Ant Design layout**

```tsx
// web/frontend/src/App.tsx
import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Layout, Menu, ConfigProvider } from 'antd';
import {
  TableOutlined,
  ClockCircleOutlined,
  RiseOutlined,
  PieChartOutlined,
} from '@ant-design/icons';
import zhCN from 'antd/locale/zh_CN';
import CronTracker from './pages/CronTracker';
import SignalTable from './pages/SignalTable';
import Positions from './pages/Positions';
import DataBrowser from './pages/DataBrowser';

const { Header, Content, Sider } = Layout;

const menuItems = [
  { key: '/signals', icon: <RiseOutlined />, label: <Link to="/signals">交易信号</Link> },
  { key: '/positions', icon: <PieChartOutlined />, label: <Link to="/positions">持仓管理</Link> },
  { key: '/data', icon: <TableOutlined />, label: <Link to="/data">数据浏览</Link> },
  { key: '/cron', icon: <ClockCircleOutlined />, label: <Link to="/cron">任务追踪</Link> },
];

const App: React.FC = () => {
  return (
    <ConfigProvider locale={zhCN} theme={{ token: { colorPrimary: '#1677ff' } }}>
      <BrowserRouter>
        <Layout style={{ minHeight: '100vh' }}>
          <Sider breakpoint="lg" collapsedWidth="0">
            <div style={{ height: 48, margin: 16, color: '#fff', fontSize: 18, fontWeight: 'bold', textAlign: 'center' }}>
              NJ Quant
            </div>
            <Menu theme="dark" mode="inline" items={menuItems} />
          </Sider>
          <Layout>
            <Header style={{ background: '#fff', padding: '0 24px', fontSize: 16, fontWeight: 500 }}>
              量化交易信号看板
            </Header>
            <Content style={{ margin: 16, padding: 24, background: '#fff', borderRadius: 8 }}>
              <Routes>
                <Route path="/" element={<SignalTable />} />
                <Route path="/signals" element={<SignalTable />} />
                <Route path="/positions" element={<Positions />} />
                <Route path="/data" element={<DataBrowser />} />
                <Route path="/cron" element={<CronTracker />} />
              </Routes>
            </Content>
          </Layout>
        </Layout>
      </BrowserRouter>
    </ConfigProvider>
  );
};

export default App;
```

- [ ] **Step 3: Configure Vite proxy for API**

```typescript
// web/frontend/vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
});
```

- [ ] **Step 4: Verify dev server starts**

```bash
cd web/frontend && npm run dev
```
Expected: Vite dev server on :3000, proxying /api to :8080

- [ ] **Step 5: Commit**

```bash
git add web/frontend/
git commit -m "feat: add React + Ant Design project shell with routing"
```

---

### Task 14: React Frontend — SignalTable Page

**Files:**
- Create: `web/frontend/src/pages/SignalTable/index.tsx`

This is the most complex page — it displays today's signals, allows confirm/skip actions, and shows filterable status.

- [ ] **Step 1: Write the SignalTable page**

```tsx
// web/frontend/src/pages/SignalTable/index.tsx
import React, { useEffect, useState } from 'react';
import {
  Table, Tag, Button, Modal, InputNumber, Space, message, DatePicker, Select, Card,
} from 'antd';
import { CheckOutlined, CloseOutlined } from '@ant-design/icons';
import axios from 'axios';
import dayjs from 'dayjs';

const STATUS_MAP: Record<string, { color: string; label: string }> = {
  pending: { color: 'processing', label: '待执行' },
  confirmed: { color: 'success', label: '已确认' },
  skipped: { color: 'default', label: '已放弃' },
};

const DIRECTION_MAP: Record<string, { color: string; label: string }> = {
  BUY: { color: 'red', label: '买入' },
  SELL: { color: 'green', label: '卖出' },
};

const SignalTable: React.FC = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [date, setDate] = useState(dayjs().format('YYYY-MM-DD'));
  const [status, setStatus] = useState<string | undefined>();
  const [confirmModal, setConfirmModal] = useState<{ id: number; price?: number } | null>(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const params: any = { trade_date: date };
      if (status) params.status = status;
      const res = await axios.get('/api/signals/', { params });
      setData(res.data);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, [date, status]);

  const handleConfirm = async (id: number) => {
    if (!confirmModal?.price) return;
    try {
      await axios.post(`/api/signals/${id}/confirm`, { executed_price: confirmModal.price });
      message.success('信号已确认');
      setConfirmModal(null);
      fetchData();
    } catch (e: any) {
      message.error(e.response?.data?.detail || '确认失败');
    }
  };

  const handleSkip = async (id: number) => {
    try {
      await axios.post(`/api/signals/${id}/skip`);
      message.success('已放弃该信号');
      fetchData();
    } catch {
      message.error('操作失败');
    }
  };

  const columns = [
    { title: '日期', dataIndex: 'trade_date', key: 'date', width: 100 },
    { title: '代码', dataIndex: 'stock_code', key: 'code', width: 100 },
    { title: '名称', dataIndex: 'stock_name', key: 'name', width: 100 },
    {
      title: '方向', dataIndex: 'direction', key: 'dir', width: 80,
      render: (d: string) => <Tag color={DIRECTION_MAP[d]?.color}>{DIRECTION_MAP[d]?.label}</Tag>,
    },
    {
      title: '建议仓位%', dataIndex: 'target_pct', key: 'pct', width: 90,
      render: (v: number) => v ? `${v.toFixed(1)}%` : '-',
    },
    {
      title: '建议价格', key: 'price', width: 130,
      render: (_: any, r: any) =>
        r.price_low ? `${r.price_low?.toFixed(2)} ~ ${r.price_high?.toFixed(2)}` : '-',
    },
    { title: '信号原因', dataIndex: 'signal_reason', key: 'reason', ellipsis: true },
    {
      title: '状态', dataIndex: 'status', key: 'status', width: 90,
      render: (s: string) => <Tag color={STATUS_MAP[s]?.color}>{STATUS_MAP[s]?.label}</Tag>,
    },
    {
      title: '成交价', dataIndex: 'executed_price', key: 'exec', width: 90,
      render: (v: number) => v?.toFixed(2) || '-',
    },
    {
      title: '操作', key: 'action', width: 160, fixed: 'right' as const,
      render: (_: any, r: any) => (
        <Space>
          {r.status === 'pending' && (
            <>
              <Button
                type="primary" size="small" icon={<CheckOutlined />}
                onClick={() => setConfirmModal({ id: r.id })}
              >
                确认
              </Button>
              <Button
                size="small" icon={<CloseOutlined />} danger
                onClick={() => handleSkip(r.id)}
              >
                放弃
              </Button>
            </>
          )}
        </Space>
      ),
    },
  ];

  return (
    <Card
      title="每日交易信号"
      extra={
        <Space>
          <DatePicker value={dayjs(date)} onChange={(d) => setDate(d?.format('YYYY-MM-DD') || '')} />
          <Select
            allowClear placeholder="状态筛选" style={{ width: 120 }}
            value={status} onChange={setStatus}
            options={[
              { value: 'pending', label: '待执行' },
              { value: 'confirmed', label: '已确认' },
              { value: 'skipped', label: '已放弃' },
            ]}
          />
          <Button onClick={fetchData}>刷新</Button>
        </Space>
      }
    >
      <Table
        columns={columns}
        dataSource={data}
        rowKey="id"
        loading={loading}
        scroll={{ x: 1100 }}
        pagination={{ pageSize: 20 }}
      />
      <Modal
        title="确认执行"
        open={!!confirmModal}
        onOk={() => confirmModal && handleConfirm(confirmModal.id)}
        onCancel={() => setConfirmModal(null)}
      >
        <p>请输入实际成交价格：</p>
        <InputNumber
          style={{ width: '100%' }}
          min={0}
          step={0.01}
          value={confirmModal?.price}
          onChange={(v) => setConfirmModal({ ...confirmModal!, price: v || undefined })}
          placeholder="成交价"
        />
      </Modal>
    </Card>
  );
};

export default SignalTable;
```

- [ ] **Step 2: Commit**

```bash
git add web/frontend/src/pages/SignalTable/
git commit -m "feat: add SignalTable page with confirm/skip actions"
```

---

### Task 15: React Frontend — Positions Page

**Files:**
- Create: `web/frontend/src/pages/Positions/index.tsx`

- [ ] **Step 1: Write the Positions page**

```tsx
// web/frontend/src/pages/Positions/index.tsx
import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Statistic, Table, Tag, Button, Modal, InputNumber, Space, message } from 'antd';
import { WalletOutlined, DollarOutlined, StockOutlined, RiseOutlined } from '@ant-design/icons';
import axios from 'axios';

const Positions: React.FC = () => {
  const [overview, setOverview] = useState<any>({});
  const [positions, setPositions] = useState([]);
  const [history, setHistory] = useState([]);
  const [capitalLogs, setCapitalLogs] = useState([]);
  const [depositVisible, setDepositVisible] = useState(false);
  const [depositAmount, setDepositAmount] = useState<number>();
  const [loading, setLoading] = useState(false);

  const fetchAll = async () => {
    setLoading(true);
    try {
      const [ov, pos, hist, cap] = await Promise.all([
        axios.get('/api/positions/overview'),
        axios.get('/api/positions/', { params: { status: 'OPEN' } }),
        axios.get('/api/positions/', { params: { status: 'CLOSED' } }),
        axios.get('/api/positions/capital'),
      ]);
      setOverview(ov.data);
      setPositions(pos.data);
      setHistory(hist.data);
      setCapitalLogs(cap.data);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchAll(); }, []);

  const handleDeposit = async () => {
    if (!depositAmount) return;
    await axios.post('/api/positions/capital/deposit', { amount: depositAmount });
    message.success('资金补充成功');
    setDepositVisible(false);
    setDepositAmount(undefined);
    fetchAll();
  };

  const posColumns = [
    { title: '代码', dataIndex: 'stock_code', width: 100 },
    { title: '名称', dataIndex: 'stock_name', width: 100 },
    { title: '买入日', dataIndex: 'buy_date', width: 100 },
    {
      title: '买入价', dataIndex: 'buy_price', width: 90,
      render: (v: number) => v?.toFixed(2),
    },
    { title: '股数', dataIndex: 'shares', width: 80 },
    {
      title: '市值', key: 'mv', width: 100,
      render: (_: any, r: any) => (r.shares * r.buy_price).toFixed(2),
    },
  ];

  const historyColumns = [
    ...posColumns.filter(c => c.dataIndex !== 'shares'),
    {
      title: '卖出日', dataIndex: 'sell_date', width: 100,
    },
    {
      title: '卖出价', dataIndex: 'sell_price', width: 90,
      render: (v: number) => v?.toFixed(2),
    },
    {
      title: '收益率', dataIndex: 'profit_pct', width: 90,
      render: (v: number) => (
        <Tag color={v >= 0 ? 'red' : 'green'}>
          {v != null ? `${v.toFixed(2)}%` : '-'}
        </Tag>
      ),
    },
  ];

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card><Statistic title="总资产" value={overview.total_asset} precision={2} prefix={<WalletOutlined />} /></Card>
        </Col>
        <Col span={6}>
          <Card><Statistic title="可用资金" value={overview.available_cash} precision={2} prefix={<DollarOutlined />} /></Card>
        </Col>
        <Col span={6}>
          <Card><Statistic title="持仓市值" value={overview.position_value} precision={2} prefix={<StockOutlined />} /></Card>
        </Col>
        <Col span={6}>
          <Card><Statistic title="累计收益" value={overview.total_profit} precision={2} prefix={<RiseOutlined />} /></Card>
        </Col>
      </Row>

      <Card
        title="当前持仓"
        extra={
          <Space>
            <Button type="primary" onClick={() => setDepositVisible(true)}>补充资金</Button>
            <Button onClick={fetchAll} loading={loading}>刷新</Button>
          </Space>
        }
        style={{ marginBottom: 24 }}
      >
        <Table columns={posColumns} dataSource={positions} rowKey="id" pagination={false}
          locale={{ emptyText: '暂无持仓' }} />
      </Card>

      <Card title="历史交易" style={{ marginBottom: 24 }}>
        <Table columns={historyColumns} dataSource={history} rowKey="id"
          pagination={{ pageSize: 10 }} />
      </Card>

      <Card title="资金流水">
        <Table
          dataSource={capitalLogs} rowKey="id" pagination={{ pageSize: 10 }}
          columns={[
            { title: '类型', dataIndex: 'event_type', width: 80 },
            {
              title: '金额', dataIndex: 'amount', width: 120,
              render: (v: number) => <span style={{ color: v >= 0 ? '#3f8600' : '#cf1322' }}>{v.toFixed(2)}</span>,
            },
            { title: '余额', dataIndex: 'balance_after', width: 120, render: (v: number) => v?.toFixed(2) },
            { title: '备注', dataIndex: 'note' },
            { title: '时间', dataIndex: 'created_at', width: 170 },
          ]}
        />
      </Card>

      <Modal
        title="补充资金"
        open={depositVisible}
        onOk={handleDeposit}
        onCancel={() => setDepositVisible(false)}
      >
        <InputNumber
          style={{ width: '100%' }} min={0} step={1000}
          value={depositAmount} onChange={(v) => setDepositAmount(v || undefined)}
          placeholder="金额"
        />
      </Modal>
    </div>
  );
};

export default Positions;
```

- [ ] **Step 2: Commit**

```bash
git add web/frontend/src/pages/Positions/
git commit -m "feat: add Positions page with asset overview and capital management"
```

---

### Task 16: React Frontend — CronTracker + DataBrowser Pages

**Files:**
- Create: `web/frontend/src/pages/CronTracker/index.tsx`
- Create: `web/frontend/src/pages/DataBrowser/index.tsx`

- [ ] **Step 1: Write CronTracker page**

```tsx
// web/frontend/src/pages/CronTracker/index.tsx
import React, { useEffect, useState } from 'react';
import { Card, Table, Tag, Timeline, Descriptions, Space, Button } from 'antd';
import { CheckCircleOutlined, CloseCircleOutlined, SyncOutlined } from '@ant-design/icons';
import axios from 'axios';

const STATUS_ICON: Record<string, React.ReactNode> = {
  success: <CheckCircleOutlined style={{ color: '#52c41a' }} />,
  failed: <CloseCircleOutlined style={{ color: '#ff4d4f' }} />,
  running: <SyncOutlined spin style={{ color: '#1677ff' }} />,
};

const CronTracker: React.FC = () => {
  const [logs, setLogs] = useState([]);
  const [completeness, setCompleteness] = useState<any>({});
  const [loading, setLoading] = useState(false);

  const fetch = async () => {
    setLoading(true);
    try {
      const [logRes, statusRes] = await Promise.all([
        axios.get('/api/cron/'),
        axios.get('/api/cron/status'),
      ]);
      setLogs(logRes.data);
      setCompleteness(statusRes.data);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetch(); }, []);

  return (
    <div>
      <Card title="数据完整性" style={{ marginBottom: 24 }}>
        <Descriptions column={3}>
          <Descriptions.Item label="最近日线数据">{completeness.last_daily_date || '-'}</Descriptions.Item>
          <Descriptions.Item label="最近信号日期">{completeness.last_signal_date || '-'}</Descriptions.Item>
          <Descriptions.Item label="最近补全状态">
            {completeness.last_backfill?.status ? (
              <Tag icon={STATUS_ICON[completeness.last_backfill.status]}>
                {completeness.last_backfill.status} ({completeness.last_backfill.finished_at})
              </Tag>
            ) : '-'}
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Card
        title="任务执行记录"
        extra={<Button onClick={fetch} loading={loading}>刷新</Button>}
      >
        <Table
          dataSource={logs} rowKey="id" pagination={{ pageSize: 15 }}
          columns={[
            { title: '任务', dataIndex: 'task_name', width: 150 },
            {
              title: '状态', dataIndex: 'status', width: 100,
              render: (s: string) => <Tag icon={STATUS_ICON[s]} color={s === 'success' ? 'success' : s === 'failed' ? 'error' : 'processing'}>{s}</Tag>,
            },
            { title: '开始', dataIndex: 'started_at', width: 170 },
            { title: '结束', dataIndex: 'finished_at', width: 170, render: (v: string) => v || '-' },
            { title: '错误', dataIndex: 'error_message', ellipsis: true, render: (v: string) => v || '-' },
            { title: '详情', dataIndex: 'metadata', ellipsis: true, render: (v: any) => typeof v === 'string' ? v : JSON.stringify(v) || '-' },
          ]}
        />
      </Card>
    </div>
  );
};

export default CronTracker;
```

- [ ] **Step 2: Write DataBrowser page**

```tsx
// web/frontend/src/pages/DataBrowser/index.tsx
import React, { useEffect, useState } from 'react';
import { Card, Table, Input, Button, Space, DatePicker, Drawer } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import axios from 'axios';

const DataBrowser: React.FC = () => {
  const [data, setData] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const [detailVisible, setDetailVisible] = useState(false);
  const [detailCode, setDetailCode] = useState('');
  const [detailData, setDetailData] = useState([]);

  const fetch = async (p: number = 1) => {
    setLoading(true);
    try {
      const res = await axios.get('/api/data/stocks', {
        params: { page: p, search: search || undefined, page_size: 50 },
      });
      setData(res.data.data);
      setTotal(res.data.total);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetch(page); }, [page]);

  const handleDetail = async (code: string) => {
    setDetailCode(code);
    setDetailVisible(true);
    const res = await axios.get(`/api/data/stocks/${code}`);
    setDetailData(res.data);
  };

  return (
    <Card
      title="全量数据浏览"
      extra={
        <Space>
          <Input.Search
            placeholder="代码/名称" prefix={<SearchOutlined />}
            value={search} onChange={(e) => setSearch(e.target.value)}
            onSearch={() => { setPage(1); fetch(1); }}
            style={{ width: 200 }}
          />
          <Button onClick={() => fetch(page)} loading={loading}>刷新</Button>
        </Space>
      }
    >
      <Table
        dataSource={data} rowKey="stock_code" loading={loading}
        pagination={{ total, current: page, pageSize: 50, onChange: setPage }}
        columns={[
          { title: '代码', dataIndex: 'stock_code', width: 100 },
          { title: '名称', dataIndex: 'stock_name', width: 100 },
          { title: '行业', dataIndex: 'industry', width: 100 },
          { title: '日期', dataIndex: 'trade_date', width: 100 },
          { title: '收盘价', dataIndex: 'close', width: 90 },
          {
            title: '涨跌幅', dataIndex: 'change_pct', width: 80,
            render: (v: number) => <span style={{ color: v >= 0 ? '#cf1322' : '#3f8600' }}>{v}%</span>,
          },
          { title: '换手率', dataIndex: 'turnover_rate', width: 80 },
          { title: 'PE', dataIndex: 'pe_ttm', width: 70 },
          { title: 'PB', dataIndex: 'pb', width: 70 },
          { title: '流通市值(亿)', dataIndex: 'circulating_mv', width: 110, render: (v: number) => (v / 1e8).toFixed(1) },
          {
            title: '操作', width: 80, fixed: 'right' as const,
            render: (_: any, r: any) => <Button size="small" onClick={() => handleDetail(r.stock_code)}>详情</Button>,
          },
        ]}
        scroll={{ x: 1000 }}
      />

      <Drawer
        title={`${detailCode} 历史数据`} open={detailVisible}
        onClose={() => setDetailVisible(false)} width={800}
      >
        <Table
          dataSource={detailData} rowKey={(r: any) => `${r.stock_code}-${r.trade_date}`}
          pagination={{ pageSize: 20 }}
          columns={[
            { title: '日期', dataIndex: 'trade_date', width: 100 },
            { title: '开', dataIndex: 'open', width: 70 }, { title: '高', dataIndex: 'high', width: 70 },
            { title: '低', dataIndex: 'low', width: 70 }, { title: '收', dataIndex: 'close', width: 70 },
            { title: '量', dataIndex: 'volume', width: 90 },
            { title: 'PE', dataIndex: 'pe_ttm', width: 60 }, { title: 'PB', dataIndex: 'pb', width: 60 },
          ]}
          scroll={{ x: 600 }}
        />
      </Drawer>
    </Card>
  );
};

export default DataBrowser;
```

- [ ] **Step 3: Commit**

```bash
git add web/frontend/src/pages/CronTracker/ web/frontend/src/pages/DataBrowser/
git commit -m "feat: add CronTracker and DataBrowser pages"
```

---

### Task 17: Deployment Configs

**Files:**
- Create: `deploy/nj-quant-web.service` (systemd)
- Create: `deploy/nginx.conf` (nginx)
- Create: `deploy/crontab.txt` (cron)
- Create: `deploy/.env.example`
- Create: `scripts/start.sh`

- [ ] **Step 1: Write systemd unit**

```ini
# deploy/nj-quant-web.service
[Unit]
Description=NJ Quant Web Dashboard
After=network.target postgresql.service

[Service]
Type=simple
User=njquant
WorkingDirectory=/home/njquant/nj-quant
EnvironmentFile=/home/njquant/nj-quant/.env
ExecStart=/home/njquant/nj-quant/.venv/bin/uvicorn web.server.main:app --host 127.0.0.1 --port 8080
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Write nginx config**

```nginx
# deploy/nginx.conf
server {
    listen 80;
    server_name _;

    # React static files
    root /home/njquant/nj-quant/web/frontend/dist;
    index index.html;

    # API proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

- [ ] **Step 3: Write crontab**

```bash
# deploy/crontab.txt
TUSHARE_TOKEN=your_token_here

# 14:25 — Intraday signal generation
25 14 * * 1-5  cd /home/njquant/nj-quant && .venv/bin/python signal_pipeline/intraday_signal.py >> logs/intraday.log 2>&1

# 18:00 — Night data backfill + cache build
0 18 * * 1-5  cd /home/njquant/nj-quant && .venv/bin/python signal_pipeline/night_backfill.py >> logs/backfill.log 2>&1

# 09:00 — Web health check
0 9 * * 1-5  systemctl is-active --quiet nj-quant-web || systemctl restart nj-quant-web
```

- [ ] **Step 4: Write env example**

```bash
# deploy/.env.example
# Copy to /home/njquant/nj-quant/.env and fill in values
TUSHARE_TOKEN=your_tushare_token
CACHE_DIR=cache/daily_rotation
DB_HOST=localhost
DB_PORT=5432
DB_NAME=njquant
DB_USER=njquant
DB_PASSWORD=your_password
```

- [ ] **Step 5: Write start script**

```bash
#!/bin/bash
# scripts/start.sh — One-click start for development

set -e

# Activate venv
source .venv/bin/activate

# Start FastAPI
echo "Starting FastAPI on :8080..."
uvicorn web.server.main:app --host 0.0.0.0 --port 8080 --reload &
API_PID=$!

# Start React dev server
echo "Starting React on :3000..."
cd web/frontend && npm run dev &
UI_PID=$!

echo "API: http://localhost:8080"
echo "UI:  http://localhost:3000"
echo "Press Ctrl+C to stop"

trap "kill $API_PID $UI_PID 2>/dev/null" EXIT
wait
```

- [ ] **Step 6: Commit**

```bash
git add deploy/ scripts/
git commit -m "feat: add deployment configs (systemd, nginx, cron, start script)"
```

---

### Task 18: Add Missing Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add new dependencies**

Add to requirements.txt:
```
tushare
akshare>=1.12.0
fastapi
uvicorn
```

- [ ] **Step 2: Install**

```bash
pip install tushare akshare fastapi uvicorn
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add live trading dependencies (tushare, akshare, fastapi)"
```
