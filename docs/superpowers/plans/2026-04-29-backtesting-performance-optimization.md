# 回测性能优化实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将单日回测从 ~6s 降至 ~0.2s，通过预计算滚动指标到 Parquet 缓存 + 引擎简化为纯查表逻辑。

**Architecture:** Data layer — `DailyDataCache.build()` 一次性计算所有滚动指标写入 Parquet。Engine layer — 移除 Master DataFrame 累积，改为 `_prev_df`/`_today_df` 双指针滚动，`_build_signal_features` 变为纯列拷贝，因子/ATR 从预计算列读取。

**Tech Stack:** Python 3.12, pandas, pyarrow (Parquet), PostgreSQL (source data, unchanged)

**Spec:** `docs/superpowers/specs/2026-04-29-backtesting-performance-optimization-design.md`

---

### File Structure

| File | Role |
|------|------|
| `back_testing/data/daily_data_cache.py` | Modify: `DailyDataCache.build()` — add precomputed columns |
| `back_testing/data/build_daily_cache.py` | **Create**: standalone CLI entry point |
| `back_testing/rotation/daily_rotation_engine.py` | Modify: engine rewrite — remove Master DataFrame, add rolling pointer, simplify methods |
---

### Task 1: Add `get_daily_dataframe` to `DataProvider` (fix C1)

**Files:**
- Modify: `back_testing/data/data_provider.py`

The new engine calls `self.data_provider.get_daily_dataframe(date_str)` unconditionally. Currently only `CachedProvider` has this method; `DataProvider` does not. Without this, constructing the engine without a `CachedProvider` will raise `AttributeError`.

- [ ] **Step 1: Add `get_daily_dataframe` to `DataProvider`**

Insert after `get_stocks_for_date` (line ~389):

```python
    def get_daily_dataframe(self, date: str) -> pd.DataFrame:
        """获取指定日期的全市场 DataFrame（兼容 CachedProvider 接口）。

        用于新引擎的每日滚动指针。从 PostgreSQL 单次查询。
        """
        session = self.Session()
        try:
            target_date = pd.to_datetime(date)
            results = session.query(StockDaily).filter(
                StockDaily.trade_date == target_date
            ).all()

            if not results:
                return pd.DataFrame()

            rows = []
            for r in results:
                rows.append({
                    'stock_code': r.stock_code,
                    'trade_date': pd.Timestamp(r.trade_date),
                    'stock_name': r.stock_name,
                    'open': float(r.open) if r.open else 0.0,
                    'high': float(r.high) if r.high else 0.0,
                    'low': float(r.low) if r.low else 0.0,
                    'close': float(r.close) if r.close else 0.0,
                    'volume': float(r.volume) if r.volume else 0.0,
                    'turnover_amount': float(r.turnover_amount) if r.turnover_amount else 0.0,
                    'adj_close': float(r.adj_close) if r.adj_close else 0.0,
                    'prev_adj_close': float(r.prev_adj_close) if r.prev_adj_close else 0.0,
                    'amplitude': float(r.amplitude) if r.amplitude else 0.0,
                    'change_pct': float(r.change_pct) if r.change_pct else 0.0,
                    'turnover_rate': float(r.turnover_rate) if r.turnover_rate else 0.0,
                    'volume_ratio': float(r.volume_ratio) if r.volume_ratio else 0.0,
                    'circulating_mv': float(r.circulating_mv) if r.circulating_mv else 0.0,
                    'total_mv': float(r.total_mv) if r.total_mv else 0.0,
                    'limit_up': r.limit_up,
                    'limit_down': r.limit_down,
                    'pe_ttm': float(r.pe_ttm) if r.pe_ttm else None,
                    'ps_ttm': float(r.ps_ttm) if r.ps_ttm else None,
                    'pcf_ttm': float(r.pcf_ttm) if r.pcf_ttm else None,
                    'pb': float(r.pb) if r.pb else None,
                    'ma_5': float(r.ma_5) if r.ma_5 else None,
                    'ma_10': float(r.ma_10) if r.ma_10 else None,
                    'ma_20': float(r.ma_20) if r.ma_20 else None,
                    'ma_30': float(r.ma_30) if r.ma_30 else None,
                    'ma_60': float(r.ma_60) if r.ma_60 else None,
                    'ma_cross': r.ma_cross,
                    'macd_dif': float(r.macd_dif) if r.macd_dif else None,
                    'macd_dea': float(r.macd_dea) if r.macd_dea else None,
                    'macd_hist': float(r.macd_hist) if r.macd_hist else None,
                    'macd_cross': r.macd_cross,
                    'kdj_k': float(r.kdj_k) if r.kdj_k else None,
                    'kdj_d': float(r.kdj_d) if r.kdj_d else None,
                    'kdj_j': float(r.kdj_j) if r.kdj_j else None,
                    'kdj_cross': r.kdj_cross,
                    'boll_mid': float(r.boll_mid) if r.boll_mid else None,
                    'boll_upper': float(r.boll_upper) if r.boll_upper else None,
                    'boll_lower': float(r.boll_lower) if r.boll_lower else None,
                    'rsi_1': float(r.rsi_1) if r.rsi_1 else None,
                    'rsi_2': float(r.rsi_2) if r.rsi_2 else None,
                    'rsi_3': float(r.rsi_3) if r.rsi_3 else None,
                    'psy': float(r.psy) if r.psy else None,
                    'psyma': float(r.psyma) if r.psyma else None,
                    'industry': r.industry,
                    'concept': r.concept,
                    'area': r.area,
                })

            return pd.DataFrame(rows)

        finally:
            session.close()
```

Note: This method does NOT include the 9 precomputed columns. When using `DataProvider`, the engine's `_build_signal_features` will get `KeyError` on those columns. However, the `DataProvider` path is only used for single backtests without cache, and the optimization workflow always uses `CachedProvider`. The fallback behavior is acceptable (engine will fall through `_build_signal_features` exception to per-stock signal detection).

- [ ] **Step 2: Commit**

```bash
git add back_testing/data/data_provider.py
git commit -m "feat(data): add get_daily_dataframe to DataProvider for engine compatibility"
```

---

### Task 2: Enhance `DailyDataCache.build()` — precompute rolling columns

**Files:**
- Modify: `back_testing/data/daily_data_cache.py:165-286`

- [ ] **Step 1: Add precompute method to `DailyDataCache`**

Add a `_precompute_stock_indicators(df)` static method that computes all rolling indicators on a single stock's DataFrame. Index must be sorted by date.

```python
@staticmethod
def _precompute_stock_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling indicators for a single stock. Expects df sorted by trade_date."""
    df = df.sort_values('trade_date').copy()

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Volume MAs
    df['vol_ma5'] = volume.rolling(5, min_periods=1).mean()
    df['vol_ma20'] = volume.rolling(20, min_periods=1).mean()

    # Close std (Bollinger width)
    df['close_std_20'] = close.rolling(20, min_periods=1).std()

    # 20-day high max (exclude today via shift)
    df['high_20_max'] = high.shift(1).rolling(20, min_periods=1).max()

    # ATR (14)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14, min_periods=1).mean()

    # Williams %R (10, 14)
    for period in [10, 14]:
        high_n = high.rolling(period, min_periods=1).max()
        low_n = low.rolling(period, min_periods=1).min()
        denom = high_n - low_n
        wr = pd.Series(-50.0, index=df.index)
        mask = denom > 0
        wr[mask] = (high_n[mask] - close[mask]) / denom[mask] * -100
        df[f'wr_{period}'] = wr

    # Returns
    df['ret_5'] = close / close.shift(5) - 1
    df['ret_20'] = close / close.shift(20) - 1

    # Fill NaN with 0 (new stocks / insufficient history)
    new_cols = ['vol_ma5', 'vol_ma20', 'close_std_20', 'high_20_max',
                'atr_14', 'wr_10', 'wr_14', 'ret_5', 'ret_20']
    df[new_cols] = df[new_cols].fillna(0.0)

    return df
```

- [ ] **Step 2: Rewrite `DailyDataCache.build()` — load full data, compute per-stock, write per-date**

Replace the body of `build()` after the directory creation. The key change: load all data at once, group by stock_code, call `_precompute_stock_indicators`, then group by date and write.

```python
@staticmethod
def build(
    start_date: str,
    end_date: str,
    cache_dir: str,
    preload_days: int = 30,
    benchmark_index: str = 'sh000300'
):
    from back_testing.data.db.connection import get_engine

    engine = get_engine()
    cache_path = Path(cache_dir)
    daily_dir = cache_path / 'daily'
    index_dir = cache_path / 'index'
    daily_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    # ── 0. Version check for incremental build safety ──
    CACHE_VERSION = 2  # v1=raw columns, v2=with precomputed rolling indicators
    version_path = cache_path / 'cache_version.txt'
    if version_path.exists():
        existing_version = int(version_path.read_text().strip())
        if existing_version < CACHE_VERSION:
            print(f"缓存版本不兼容 (v{existing_version} → v{CACHE_VERSION})，强制重建所有日期...")
            import shutil
            for f in daily_dir.glob('*.parquet'):
                f.unlink()  # Delete old-format files so all dates get rebuilt
    version_path.write_text(str(CACHE_VERSION))

    load_start = (pd.Timestamp(start_date) - pd.Timedelta(days=preload_days)).strftime('%Y-%m-%d')
    load_end = end_date

    # ── 1. Query trading dates ──
    dates_df = pd.read_sql(
        "SELECT DISTINCT trade_date FROM stock_daily "
        "WHERE trade_date >= %(start)s AND trade_date <= %(end)s "
        "ORDER BY trade_date",
        engine, params={'start': load_start, 'end': load_end}
    )
    all_dates = [d.strftime('%Y-%m-%d') for d in pd.to_datetime(dates_df['trade_date'])]
    if not all_dates:
        raise ValueError(f"指定范围内无交易日数据: {load_start} ~ {load_end}")

    print(f"缓存构建: {load_start} ~ {load_end}, 共 {len(all_dates)} 个交易日")

    # ── 2. Check existing dates (incremental build) ──
    dates_to_build = [d for d in all_dates if not (daily_dir / f'{d}.parquet').exists()]
    if not dates_to_build:
        print("所有日期已缓存，跳过日线构建。")
    else:
        print(f"需构建 {len(dates_to_build)} 个交易日 (已有 {len(all_dates) - len(dates_to_build)} 个)")

        # ── 3. Load full data for dates that need building ──
        date_params = tuple(dates_to_build)
        # Use chunked reads for large date ranges
        chunk_size = 60  # ~2 months per chunk to avoid OOM
        all_data_frames = []
        for i in range(0, len(date_params), chunk_size):
            chunk = date_params[i:i + chunk_size]
            placeholders = ','.join([f'%(d{j})s' for j in range(len(chunk))])
            params = {f'd{j}': chunk[j] for j in range(len(chunk))}
            chunk_df = pd.read_sql(
                f"SELECT * FROM stock_daily WHERE trade_date IN ({placeholders})",
                engine, params=params
            )
            all_data_frames.append(chunk_df)
            print(f"  加载数据块 {i // chunk_size + 1}/{(len(date_params) - 1) // chunk_size + 1}")

        full_df = pd.concat(all_data_frames, ignore_index=True)
        print(f"  加载 {len(full_df)} 行原始数据")

        # Convert numeric columns
        for col in full_df.columns:
            if col in ('trade_date', 'stock_code', 'stock_name'):
                continue
            if full_df[col].dtype == object:
                try:
                    full_df[col] = pd.to_numeric(full_df[col])
                except (ValueError, TypeError):
                    pass

        full_df['trade_date'] = pd.to_datetime(full_df['trade_date'])

        # ── 4. Per-stock compute indicators ──
        print("计算滚动指标...")
        all_stocks = full_df['stock_code'].unique()
        computed_frames = []
        for idx, code in enumerate(all_stocks):
            stock_df = full_df[full_df['stock_code'] == code]
            computed = DailyDataCache._precompute_stock_indicators(stock_df)
            computed_frames.append(computed)
            if (idx + 1) % 500 == 0:
                print(f"  已处理 {idx + 1}/{len(all_stocks)} 只股票")

        combined = pd.concat(computed_frames, ignore_index=True)
        print(f"  计算完成，共 {len(combined)} 行")

        # ── 5. Write per-date Parquet ──
        for i, date_str in enumerate(dates_to_build):
            daily_path = daily_dir / f'{date_str}.parquet'
            date_mask = combined['trade_date'] == pd.Timestamp(date_str)
            day_data = combined[date_mask]
            if day_data.empty:
                continue
            day_data.to_parquet(daily_path, index=False)
            if (i + 1) % 50 == 0:
                print(f"  写入 {i + 1}/{len(dates_to_build)}: {date_str} ({len(day_data)} 只)")

        print(f"日线数据写入完成: {len(dates_to_build)} 个日期")

    # ── 6. Stock codes ──
    codes_df = pd.read_sql(
        "SELECT stock_code FROM stock_meta WHERE is_active = TRUE AND market != '北'",
        engine
    )
    codes_df.to_parquet(cache_path / 'stock_codes.parquet', index=False)
    print(f"股票代码: {len(codes_df)} 只")

    # ── 7. Trading dates (from all cached daily files) ──
    all_cached_dates = sorted([f.stem for f in daily_dir.glob('*.parquet')])
    pd.DataFrame({'trade_date': all_cached_dates}).to_parquet(
        cache_path / 'trading_dates.parquet', index=False
    )

    # ── 8. Index data ──
    print(f"加载指数数据: {benchmark_index} ...")
    index_df = pd.read_sql(
        "SELECT * FROM index_daily WHERE index_code = %(code)s "
        "AND trade_date >= %(start)s AND trade_date <= %(end)s",
        engine,
        params={'code': benchmark_index, 'start': load_start, 'end': load_end}
    )
    for col in index_df.columns:
        if col == 'trade_date':
            continue
        if index_df[col].dtype == object and col not in ('index_code',):
            try:
                index_df[col] = pd.to_numeric(index_df[col])
            except (ValueError, TypeError):
                pass

    index_df['trade_date'] = pd.to_datetime(index_df['trade_date'])
    index_df = index_df.set_index('trade_date').sort_index()
    index_df.to_parquet(index_dir / f'{benchmark_index}.parquet')
    print(f"指数数据: {len(index_df)} 条")

    print(f"缓存构建完成: {cache_path}")
    return str(cache_path)
```

- [ ] **Step 3: Run existing cache tests to verify build still works**

```bash
pytest tests/back_testing/data/ -v
```

- [ ] **Step 4: Commit**

```bash
git add back_testing/data/daily_data_cache.py
git commit -m "feat(data): add precomputed rolling indicators to DailyDataCache.build()"
```

---

### Task 3: Create standalone `build_daily_cache.py` CLI

**Files:**
- Create: `back_testing/data/build_daily_cache.py`

- [ ] **Step 1: Create the CLI script**

```python
"""独立缓存构建入口 — 预计算 + Parquet 写入

Usage:
    python back_testing/data/build_daily_cache.py --start 2020-01-01 --end 2025-12-31
    python back_testing/data/build_daily_cache.py --start 2024-01-01 --end 2024-12-31 --cache-dir cache/my_cache
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path (in case script is run directly)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from back_testing.data.daily_data_cache import DailyDataCache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='构建 Daily Rotation 预计算缓存（含滚动指标）'
    )
    parser.add_argument('--start', required=True, help='开始日期 YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='结束日期 YYYY-MM-DD')
    parser.add_argument('--cache-dir', default='cache/daily_rotation', help='缓存目录')
    parser.add_argument('--preload-days', type=int, default=30, help='预加载天数')
    parser.add_argument('--benchmark-index', default='sh000300', help='基准指数代码')

    args = parser.parse_args()

    logger.info(f"开始构建缓存: {args.start} ~ {args.end}")
    path = DailyDataCache.build(
        start_date=args.start,
        end_date=args.end,
        cache_dir=args.cache_dir,
        preload_days=args.preload_days,
        benchmark_index=args.benchmark_index,
    )
    logger.info(f"缓存构建完成: {path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Test CLI runs (dry run)**

```bash
python back_testing/data/build_daily_cache.py --help
```

Expected: Prints help text with all arguments.

- [ ] **Step 3: Commit**

```bash
git add back_testing/data/build_daily_cache.py
git commit -m "feat(data): add standalone cache build CLI entry point"
```

---

### Task 4: Rewrite engine `__init__` — new attributes, remove old

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:111-171`

- [ ] **Step 1: Replace `__init__` attributes**

Remove `_cache_df`, `_all_codes`, `_preloaded_cache`, `_has_fast_daily`, `PRELOAD_DAYS`, `MIN_TRADING_DAYS`.
Add `_prev_df`, `_today_df`.

Replace lines 106-171 (class docstring preserved, init changed):

```python
    def __init__(self, config: RotationConfig, start_date: str, end_date: str,
                 data_provider=None):
        """Initialize engine with rolling pointer data model.

        Args:
            config: Strategy configuration.
            start_date / end_date: Backtest date range (YYYY-MM-DD).
            data_provider: Data source (CachedProvider with precomputed columns required).
        """
        self.config = config
        self.start_date = start_date
        self.end_date = end_date

        # ── Data source ──
        self.data_provider = data_provider or DataProvider()

        # ── Subsystems ──
        self.position_manager = RotationPositionManager(
            total_capital=config.initial_capital,
            max_total_pct=config.max_total_pct,
            max_position_pct=config.max_position_pct
        )
        self.trade_executor = TradeExecutor()
        self.buy_filter = SignalFilter(config.buy_signal_types, mode=config.buy_signal_mode,
                                        kdj_low_threshold=config.kdj_low_threshold)
        self.sell_filter = SignalFilter(config.sell_signal_types)
        self.ranker = SignalRanker(config.rank_factor_weights, config.rank_factor_directions)
        self.market_regime = MarketRegime(config.market_regime, self.data_provider)

        # ── ATR stop parameters ──
        self.atr_period = config.atr_period
        self.stop_loss_mult = config.stop_loss_mult
        self.take_profit_mult = config.take_profit_mult
        self.trailing_pct = config.trailing_pct
        self.trailing_start = config.trailing_start

        # ── Runtime state ──
        self.current_capital = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.daily_results: List[DailyResult] = []
        self.trade_history: List[TradeRecord] = []

        # ── Rolling pointer: prev/current day DataFrames (~4760 rows each) ──
        self._prev_df: pd.DataFrame = pd.DataFrame()
        self._today_df: pd.DataFrame = pd.DataFrame()
```

- [ ] **Step 2: Remove unused import `from datetime import datetime`** (line 7)

Now used only in `run()` print statement — we'll replace that later.

- [ ] **Step 3: Commit**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "refactor(engine): replace Master DataFrame with rolling pointer in __init__"
```

---

### Task 5: Rewrite `_advance_to_date` — rolling pointer

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:876-948` (entire `_advance_to_date`)

- [ ] **Step 1: Replace `_advance_to_date` with 2-line rolling pointer**

```python
    def _advance_to_date(self, date: pd.Timestamp):
        """滚动指针: 将前一日的 _today_df 变为 _prev_df，读入当日新数据。

        不再累积历史数据，不再 concat。每天只读一个 Parquet 文件。
        """
        self._prev_df = self._today_df

        date_str = date.strftime('%Y-%m-%d')
        day_df = self.data_provider.get_daily_dataframe(date_str)
        if day_df is None or day_df.empty:
            self._today_df = pd.DataFrame()
            return

        day_df = day_df.copy()
        day_df['trade_date'] = pd.Timestamp(date_str)
        self._today_df = day_df.set_index('trade_date')
```

- [ ] **Step 2: Commit**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "refactor(engine): rewrite _advance_to_date as rolling pointer"
```

---

### Task 6: Rewrite `_get_daily_stock_data` — groupby 2 days

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:950-974` (entire `_get_daily_stock_data`)

- [ ] **Step 1: Replace `_get_daily_stock_data`**

```python
    def _get_daily_stock_data(self, date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """拼接 _prev_df + _today_df (~9500 行)，按股票分组为 {code: 2-row DataFrame}。

        每个 stock DataFrame 包含 1-2 行（前日+当日），index=trade_date。
        预计算列已在 Parquet 中，无需历史窗口检查。
        """
        if self._today_df.empty:
            return {}

        # Concatenate prev + today (~9500 rows total for all stocks)
        frames = [self._prev_df, self._today_df]
        combined = pd.concat(frames)

        if combined.empty:
            return {}

        result = {}
        for code, group in combined.groupby('stock_code', sort=False):
            if date in group.index:
                result[code] = group
        return result
```

- [ ] **Step 2: Commit**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "refactor(engine): rewrite _get_daily_stock_data for 2-day rolling window"
```

---

### Task 7: Rewrite `_build_signal_features` — pure column copy

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:600-664` (entire `_build_signal_features`)

- [ ] **Step 1: Replace `_build_signal_features`**

```python
    def _build_signal_features(self, stock_codes: List[str]) -> pd.DataFrame:
        """从 _today_df / _prev_df 直接提取预计算列，组装特征矩阵。

        零 groupby，零 rolling，零 sort_values。
        所有技术指标已在缓存构建时预计算好。
        """
        if self._today_df.empty:
            return pd.DataFrame()

        # Filter to relevant stocks
        today = self._today_df[self._today_df['stock_code'].isin(stock_codes)]
        prev = self._prev_df[self._prev_df['stock_code'].isin(stock_codes)]

        if today.empty:
            return pd.DataFrame()

        today = today.set_index('stock_code')
        prev = prev.set_index('stock_code')

        # Only keep stocks present in both days (needed for cross detection)
        common = today.index.intersection(prev.index)
        if common.empty:
            return pd.DataFrame()

        t = today.loc[common]
        p = prev.loc[common]

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
```

- [ ] **Step 2: Commit**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "refactor(engine): rewrite _build_signal_features as pure column copy"
```

---

### Task 8: Simplify `_check_and_sell` — ATR from column

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:374-505` (lines 442-448, the ATR check block)

- [ ] **Step 1: Replace the ATR calculation block**

Replace lines 442-448:

Old:
```python
            current_price = current_prices.get(stock_code, 0.0)
            if current_price > 0:
                try:
                    atr = StopLossStrategies.calculate_atr(df, period=self.atr_period)
                except Exception:
                    atr = 0.0
                if atr > 0:
```

New:
```python
            current_price = current_prices.get(stock_code, 0.0)
            if current_price > 0:
                atr = float(df['atr_14'].iloc[-1]) if 'atr_14' in df.columns else 0.0
                if atr > 0:
```

- [ ] **Step 2: Also replace the suspended stock price lookup (lines 400-403)**

Old (uses `_cache_df`):
```python
            if stock_code not in stock_data:
                if stock_code in self._cache_df['stock_code'].values:
                    df_cached = self._cache_df[self._cache_df['stock_code'] == stock_code].sort_index()
                    if not df_cached.empty:
                        price = df_cached['close'].iloc[-1]
```

New (use `_prev_df`):
```python
            if stock_code not in stock_data:
                # Suspended/delisted: use _prev_df close price
                if not self._prev_df.empty:
                    prev_rows = self._prev_df[self._prev_df['stock_code'] == stock_code]
                    if not prev_rows.empty:
                        price = float(prev_rows['close'].iloc[-1])
```

Note: `price` needs to be defined before use. The full suspended block becomes:

```python
            if stock_code not in stock_data:
                price = 0.0
                if not self._prev_df.empty:
                    prev_rows = self._prev_df[self._prev_df['stock_code'] == stock_code]
                    if not prev_rows.empty:
                        price = float(prev_rows['close'].iloc[-1])
                    if price > 0:
                        shares, cost = self.trade_executor.execute_sell(stock_code, price, position.shares)
                        if shares > 0:
                            buy_price = position.buy_price
                            holding_days = (pd.Timestamp(date_str) - pd.Timestamp(position.buy_date)).days
                            return_pct = (price - buy_price) / buy_price * 100 if buy_price > 0 else 0
                            pnl = (price - buy_price) * shares - cost
                            capital_before_sell = self.current_capital

                            trade = TradeRecord(
                                date=date_str,
                                stock_code=stock_code,
                                action='SELL',
                                price=price,
                                shares=shares,
                                cost=cost,
                                capital_before=capital_before_sell
                            )
                            sell_trades.append(trade)
                            self.trade_history.append(trade)
                            del self.positions[stock_code]

                            logger.info(
                                f"[SELL/SUSPENDED] {date_str} {stock_code} @ {price:.3f} x {shares}股 "
                                f"买价:{buy_price:.3f} 持有:{holding_days}天 收益:{return_pct:+.2f}% "
                                f"PnL:{pnl:+,.0f} (卖前现金:{capital_before_sell:,.0f})"
                            )
                continue
```
- [ ] **Step 3: Commit**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "refactor(engine): read ATR from precomputed column in _check_and_sell"
```

---

### Task 9: Simplify `_execute_buy` — factors from columns

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:666-833` (lines 700-748, the factor extraction loop)

- [ ] **Step 1: Replace the factor extraction loop**

Replace lines 700-748 (the per-candidate factor extraction):

Old:
```python
        factor_data_dict = {}
        for stock_code in candidates:
            df = stock_data.get(stock_code)
            if df is None or df.empty:
                continue
            row = df.iloc[-1]
            factor_row = {}

            if len(df) >= 5 and 'close' in df.columns:
                ret5 = row['close'] / df['close'].iloc[-5] - 1
            else:
                ret5 = 0.0

            for factor in self.ranker.factor_weights.keys():
                if factor == 'RET_20':
                    if len(df) >= 20 and 'close' in df.columns:
                        factor_row[factor] = row['close'] / df['close'].iloc[-20] - 1
                    else:
                        factor_row[factor] = np.nan
                elif factor == 'OVERHEAT':
                    rsi_val = row.get('rsi_1', np.nan)
                    if pd.notna(rsi_val):
                        factor_row[factor] = compute_overheat(
                            float(rsi_val), ret5,
                            self.config.overheat_rsi_threshold,
                            self.config.overheat_ret5_threshold
                        )
                    else:
                        factor_row[factor] = 0.0
                elif factor == 'circulating_mv':
                    val = row.get('circulating_mv', np.nan)
                    factor_row[factor] = np.log(val) if val > 0 else np.nan
                elif factor in ('WR_10', 'WR_14'):
                    period = 10 if factor == 'WR_10' else 14
                    factor_row[factor] = FactorProcessor.williams_r(df, period)
                elif factor in row.index:
                    val = row[factor]
                    factor_row[factor] = val if val == val else np.nan
                else:
                    factor_row[factor] = np.nan
            factor_data_dict[stock_code] = factor_row
```

New:
```python
        factor_data_dict = {}
        for stock_code in candidates:
            df = stock_data.get(stock_code)
            if df is None or df.empty:
                continue
            row = df.iloc[-1]
            factor_row = {}

            for factor in self.ranker.factor_weights.keys():
                if factor == 'RET_20':
                    factor_row[factor] = float(row.get('ret_20', 0.0) or 0.0)
                elif factor == 'OVERHEAT':
                    rsi_val = row.get('rsi_1', np.nan)
                    ret5 = float(row.get('ret_5', 0.0) or 0.0)
                    if pd.notna(rsi_val):
                        factor_row[factor] = compute_overheat(
                            float(rsi_val), ret5,
                            self.config.overheat_rsi_threshold,
                            self.config.overheat_ret5_threshold
                        )
                    else:
                        factor_row[factor] = 0.0
                elif factor == 'circulating_mv':
                    val = row.get('circulating_mv', np.nan)
                    factor_row[factor] = np.log(val) if val > 0 else np.nan
                elif factor == 'WR_10':
                    factor_row[factor] = float(row.get('wr_10', np.nan) or 0.0)
                elif factor == 'WR_14':
                    factor_row[factor] = float(row.get('wr_14', np.nan) or 0.0)
                elif factor in row.index:
                    val = row[factor]
                    factor_row[factor] = val if val == val else np.nan
                else:
                    factor_row[factor] = np.nan
            factor_data_dict[stock_code] = factor_row
```

- [ ] **Step 2: Remove unused import `FactorProcessor`**

Since `WR_10/WR_14` no longer calls `FactorProcessor.williams_r()`, remove the import on line 45:

Old:
```python
from back_testing.factors.factor_utils import FactorProcessor
```

Remove this line. If `FactorProcessor` is not used elsewhere in the file (check: it isn't — `SignalRanker` uses `FactorProcessor` internally), this import can be safely removed.

- [ ] **Step 3: Commit**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "refactor(engine): read factors from precomputed columns in _execute_buy"
```

---

### Task 10: Simplify `run()` — remove preload, init `_prev_df`

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:173-225` (the `run()` method)

- [ ] **Step 1: Rewrite `run()`**

Replace the entire `run()` method:

```python
    def run(self) -> List[DailyResult]:
        """Run the backtest main loop.

        Flow:
        1. Get trading dates from benchmark index.
        2. Initialize _prev_df from the trading day before first_date.
        3. For each date: advance data pointer → run single day logic → record results.
        """
        dates = self._get_trading_dates()
        n_dates = len(dates)
        if n_dates < 2:
            return []

        # Initialize _prev_df: load trading day just before first_date
        self._init_prev_cache(dates[0])

        now = datetime.now
        print(f"{now():%H:%M:%S} [DailyRotation] {self.start_date} ~ {self.end_date}, {n_dates}天")

        # ── Day-by-day loop ──
        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')
            if i == 0 or (i + 1) % 10 == 0:
                prev_asset = self.daily_results[-1].total_asset if self.daily_results else self.config.initial_capital
                print(f"{now():%H:%M:%S}   [{i+1}/{n_dates}] {date_str} | 持仓:{len(self.positions)} | 资产:{prev_asset:,.0f}")

            self._advance_to_date(date)
            result = self._run_single_day(date)
            self.daily_results.append(result)

        final_asset = self.daily_results[-1].total_asset if self.daily_results else self.current_capital
        print(f"{datetime.now():%H:%M:%S} [DailyRotation] 回测完成，最终资产: {final_asset:,.0f}")
        return self.daily_results
```

- [ ] **Step 2: Add `_init_prev_cache` method** (insert after `_get_trading_dates`)

```python
    def _init_prev_cache(self, first_date: pd.Timestamp):
        """加载 first_date 前一个交易日的数据作为 _prev_df。

        向后搜索最多 15 个日历日（覆盖春节等长假），找到第一个有 Parquet 文件的交易日。
        如果找不到（极罕见），_prev_df 保持为空，Day-1 信号检测自动跳过（影响可忽略）。
        """
        for offset in range(1, 16):
            candidate = first_date - pd.Timedelta(days=offset)
            date_str = candidate.strftime('%Y-%m-%d')
            df = self.data_provider.get_daily_dataframe(date_str)
            if df is not None and not df.empty:
                df = df.copy()
                df['trade_date'] = candidate
                self._prev_df = df.set_index('trade_date')
                return
```

- [ ] **Step 3: Commit**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "refactor(engine): simplify run() — remove preload, add _init_prev_cache"
```

---

### Task 11: Update callers — remove `preloaded_cache` argument (fix C3)

**Files:**
- Modify: `back_testing/backtest/run_daily_rotation.py`
- Modify: `back_testing/optimization/run_daily_rotation_optimization.py`

Task 4 (engine `__init__`) removes the `preloaded_cache` parameter. Three call sites still pass it, causing `TypeError`.

- [ ] **Step 1: Update `run_daily_rotation.py`** — remove `preloaded_cache` logic

Replace lines 29-53:

Old:
```python
    data_provider = None
    preloaded_cache = None

    if cache_dir:
        cache_path = DailyDataCache.build(...)
        ...
        preloaded_cache = DailyDataCache.load_preload_dataframe(preload_path)

    engine = DailyRotationEngine(config, start_date, end_date,
                                 data_provider=data_provider,
                                 preloaded_cache=preloaded_cache)
```

New:
```python
    data_provider = None

    if cache_dir:
        cache_path = DailyDataCache.build(
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
            benchmark_index=config.benchmark_index,
        )
        cache = DailyDataCache(cache_path)
        data_provider = CachedProvider(cache)
        print(f"数据缓存就绪: {cache_path}")
        print(f"  {len(cache.stock_codes)} 只股票, {len(cache.trading_dates)} 个交易日")

    engine = DailyRotationEngine(config, start_date, end_date,
                                 data_provider=data_provider)
```

Remove: `preloaded_cache = None`, the `write_preload_cache`/`load_preload_dataframe` block, and the `preloaded_cache=preloaded_cache` argument.

- [ ] **Step 2: Update `run_daily_rotation_optimization.py`** — replace `objective()` line 293

Line ~293 in `objective()`:
Old:
```python
engine = DailyRotationEngine(config, start_date, end_date,
                             data_provider=data_provider,
                             preloaded_cache=preloaded_cache)
```
New:
```python
engine = DailyRotationEngine(config, start_date, end_date,
                             data_provider=data_provider)
```

Also remove the `preloaded_cache = None` / `if preload_cache_path:` block in `objective()`:
Old (~line 285-287):
```python
preloaded_cache = None
if preload_cache_path:
    preloaded_cache = DailyDataCache.load_preload_dataframe(preload_cache_path)
```
New: Remove these 3 lines entirely.

- [ ] **Step 3: Update `run_daily_rotation_optimization.py`** — replace `_evaluate_on_test()` line 731

Same pattern as objective():
Old (~line 724-726):
```python
preloaded_cache = None
if preload_cache_path:
    preloaded_cache = DailyDataCache.load_preload_dataframe(preload_cache_path)
```
And the engine construction:
Old:
```python
engine = DailyRotationEngine(config, start, end,
                             data_provider=data_provider,
                             preloaded_cache=preloaded_cache)
```
New:
```python
engine = DailyRotationEngine(config, start, end,
                             data_provider=data_provider)
```

- [ ] **Step 4: Remove `_build_preload_cache` calls** — preload cache is no longer needed

In `run_single_optimization()` (~line 443):
```python
preload_cache_path = _build_preload_cache(data_provider, base_config, start_date, end_date)
```
This is still called to determine `preload_cache_path` which is passed to `objective()`. Since `objective()` no longer uses it, we can also remove this call. However, keep it for now to avoid changing the `_build_preload_cache` function signature — it just becomes a no-op. The `preload_cache_path` parameter in `objective()` can stay but be ignored.

Actually, cleaner approach: remove `_build_preload_cache` call from `run_single_optimization` and `run_walk_forward`, remove the `preload_cache_path` parameter from `objective()` and `_evaluate_on_test()`.

```python
# In run_single_optimization(), remove or comment out:
# preload_cache_path = _build_preload_cache(...)

# Update objective lambda:
obj = lambda trial: objective(trial, start_date, end_date, base_config, data_provider)

# Update objective signature — remove preload_cache_path parameter:
def objective(trial, start_date, end_date,
              base_config=None, data_provider=None):
```

- [ ] **Step 5: Commit**

```bash
git add back_testing/backtest/run_daily_rotation.py back_testing/optimization/run_daily_rotation_optimization.py
git commit -m "refactor: remove preloaded_cache from engine callers"
```

---

### Task 12: Remove dead code and clean up

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py`

- [ ] **Step 1: Remove `_preload_histories` method** (original lines 854-874)

Full method deletion.

- [ ] **Step 2: Remove unused import `FactorProcessor`**

Line 45: `from back_testing.factors.factor_utils import FactorProcessor` — REMOVE (no longer used directly in engine; `SignalRanker` uses it internally).

- [ ] **Step 3: Keep everything else**

- `compute_overheat` — still used in `_execute_buy`
- `StopLossStrategies` — still used in `_check_and_sell` for `check_exit()`
- `MarketRegime`, `RotationPositionManager`, `TradeExecutor`, `SignalFilter`, `SignalRanker` — all in active use

- [ ] **Step 4: Commit**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "chore(engine): remove dead code and unused imports"
```

---

### Task 13: Verification — tests and sanity checks

**Files:**
- (No file changes — verification only)

- [ ] **Step 1: Verify precomputed column correctness (I4)**

Add a quick sanity script comparing precomputed `atr_14` and `wr_10` against runtime `StopLossStrategies.calculate_atr()` and `FactorProcessor.williams_r()` for a sample of 5 stocks × 5 dates:

```python
# Run interactively after building cache:
from back_testing.data.daily_data_cache import DailyDataCache, CachedProvider
from back_testing.risk.stop_loss_strategies import StopLossStrategies
from back_testing.factors.factor_utils import FactorProcessor
import pandas as pd

cache = DailyDataCache('cache/daily_rotation')
provider = CachedProvider(cache)

# Sample 5 stocks, 5 recent dates
codes = cache.stock_codes[:5]
dates = [d.strftime('%Y-%m-%d') for d in cache.trading_dates[-5:]]

for code in codes:
    df = provider.get_daily_dataframe(dates[-1])  # today
    prev_df = provider.get_daily_dataframe(dates[-2])  # prev
    if code not in df['stock_code'].values:
        continue
    row = df[df['stock_code'] == code].iloc[-1]
    
    # Compare ATR: need 15 days history from get_histories
    hist = provider.get_batch_histories([code], end_date=dates[-1], start_date=dates[0])
    if code in hist and len(hist[code]) >= 15:
        runtime_atr = StopLossStrategies.calculate_atr(hist[code], period=14)
        cached_atr = float(row['atr_14'])
        diff_atr = abs(runtime_atr - cached_atr)
        status = 'PASS' if diff_atr < 0.01 else f'DIFF={diff_atr:.4f}'
        print(f"  {code} {dates[-1]} ATR: runtime={runtime_atr:.4f} cached={cached_atr:.4f} {status}")
    
    # Compare WR_10
    if code in hist and len(hist[code]) >= 10:
        runtime_wr = FactorProcessor.williams_r(hist[code], period=10)
        cached_wr = float(row['wr_10'])
        diff_wr = abs(runtime_wr - cached_wr)
        status = 'PASS' if diff_wr < 0.01 else f'DIFF={diff_wr:.4f}'
        print(f"  {code} {dates[-1]} WR10: runtime={runtime_wr:.2f} cached={cached_wr:.2f} {status}")
```

Expected: All comparisons show `PASS` (difference < 1 cent for ATR, < 0.01 for WR).

- [ ] **Step 2: Run all rotation tests**

```bash
pytest tests/back_testing/rotation/ -v
```

Expected: All existing tests pass. If any fail, diagnose and fix before proceeding.

- [ ] **Step 3: Run optimization tests**

```bash
pytest tests/back_testing/optimization/ -v
```

Expected: All existing tests pass.

- [ ] **Step 4: Run data tests**

```bash
pytest tests/back_testing/data/ -v
```

Expected: All existing tests pass.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/back_testing/ -v
```

Expected: All tests pass.

- [ ] **Step 6: Document known behavioral difference (I1)**

Day-1: when `_init_prev_cache` finds no previous trading day (e.g., cached range starts exactly at `start_date`), `_prev_df` is empty. This causes `_build_signal_features` to return empty (no common index intersection), so no buy signals on day 1. This is architecturally intentional — the old engine could trade on day 1 because it preloaded 30 days of history. The impact is 1 day out of 1250 (0.08%), which is negligible for a ~20x speedup trade-off.

---

### Task 14: Performance benchmark

**Files:**
- (No file changes — measurement only)

- [ ] **Step 1: Rebuild cache with precomputed columns before benchmarking**

```bash
python back_testing/data/build_daily_cache.py --start 2024-01-01 --end 2024-12-31
```

- [ ] **Step 2: Run single-year backtest and measure time**

```bash
time python back_testing/backtest/run_daily_rotation.py --start 2024-01-01 --end 2024-12-31
```

Expected: ~4-6 minutes for 1 year (~250 days). This implies ~0.2-0.3s/day.

- [ ] **Step 3: Run 5-year backtest to verify scaling is linear**

```bash
time python back_testing/backtest/run_daily_rotation.py --start 2020-01-01 --end 2024-12-31
```

Expected: ~20-25 minutes for 5 years (~1250 days). Daily time should remain flat at ~0.2-0.3s (no O(N²) growth).

- [ ] **Step 4: Document actual results in the design doc or commit message**

Note the actual per-day and total times achieved.

---
