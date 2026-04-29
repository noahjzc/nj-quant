"""Daily Rotation 数据缓存 — 基于 Parquet 的跨 Trial 数据复用层

用法:
    # 构建缓存（一次性，从数据库加载）
    DailyDataCache.build('2023-01-01', '2023-06-01', 'cache/daily_rotation')

    # 读取（跨 Trial / 跨进程共享）
    cache = DailyDataCache('cache/daily_rotation/2023-01-01_2023-06-01')
    provider = CachedProvider(cache)

    # 传给 DailyRotationEngine
    engine = DailyRotationEngine(config, start, end, data_provider=provider)
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DailyDataCache:
    """管理 Parquet 缓存的构建与读取。

    缓存结构:
        {cache_dir}/
            stock_codes.parquet
            trading_dates.parquet
            index/{index_code}.parquet
            daily/{YYYY-MM-DD}.parquet
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self._daily_dir = self.cache_dir / 'daily'
        self._index_dir = self.cache_dir / 'index'

        self._stock_codes: Optional[List[str]] = None
        self._trading_dates: Optional[List[pd.Timestamp]] = None
        self._index_cache: Dict[str, pd.DataFrame] = {}

    # ── 属性（懒加载） ──────────────────────────────

    @property
    def stock_codes(self) -> List[str]:
        if self._stock_codes is None:
            df = pd.read_parquet(self.cache_dir / 'stock_codes.parquet')
            self._stock_codes = df['stock_code'].tolist()
        return self._stock_codes

    @property
    def trading_dates(self) -> List[pd.Timestamp]:
        if self._trading_dates is None:
            df = pd.read_parquet(self.cache_dir / 'trading_dates.parquet')
            self._trading_dates = [pd.Timestamp(d) for d in df['trade_date']]
        return self._trading_dates

    # ── 读取接口 ────────────────────────────────────

    def get_daily(self, date: str) -> pd.DataFrame:
        """读取某日所有股票数据"""
        path = self._daily_dir / f'{date}.parquet'
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def get_histories(
        self,
        codes: List[str],
        end_date: str,
        start_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """批量获取历史数据，返回 {code: DataFrame}"""
        code_set = set(codes)
        frames = []
        for f in sorted(self._daily_dir.glob('*.parquet')):
            date_str = f.stem
            if start_date and date_str < start_date:
                continue
            if date_str >= end_date:
                break
            df = pd.read_parquet(f)
            mask = df['stock_code'].isin(code_set)
            if mask.any():
                frames.append(df[mask])

        if not frames:
            return {}

        combined = pd.concat(frames, ignore_index=True)
        if combined.empty:
            return {}

        combined = combined.set_index('trade_date').sort_index()

        result = {}
        for code, stock_df in combined.groupby('stock_code', sort=False):
            if not stock_df.empty:
                result[code] = stock_df.copy()

        return result

    def get_index_data(
        self,
        index_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取指数历史数据"""
        if index_code not in self._index_cache:
            path = self._index_dir / f'{index_code}.parquet'
            if path.exists():
                self._index_cache[index_code] = pd.read_parquet(path)
            else:
                return pd.DataFrame()

        df = self._index_cache[index_code]
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        return df

    # ── 预加载缓存（跨 Trial 复用） ──────────────────

    PRELOAD_DAYS = 30

    def write_preload_cache(self, first_date: str, output_path: str) -> str:
        """将引擎预加载窗口保存为单个 Parquet 文件。

        各 Trial 可通过 load_preload_cache() 快速读取，
        避免每个 Trial 从 30+ 个日文件中重复读取和拼接。
        """
        start = (pd.Timestamp(first_date) - pd.Timedelta(days=self.PRELOAD_DAYS)).strftime('%Y-%m-%d')
        histories = self.get_histories(self.stock_codes, end_date=first_date, start_date=start)

        if not histories:
            return ''

        combined = pd.concat(histories.values(), ignore_index=False)
        combined.to_parquet(output_path)
        return output_path

    @staticmethod
    def load_preload_cache(path: str) -> Dict[str, pd.DataFrame]:
        """从预加载缓存文件快速重建 {code: DataFrame} 字典"""
        df = pd.read_parquet(path)
        result = {}
        for code, group in df.groupby('stock_code', sort=False):
            result[code] = group.copy()
        return result

    @staticmethod
    def load_preload_dataframe(path: str) -> pd.DataFrame:
        """从预加载缓存文件直接读取为 master DataFrame（无 groupby 开销）。

        Engine 可直接用作 _cache_df，避免 4761 次 concat 的转换成本。
        """
        return pd.read_parquet(path)

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

    # ── 构建 ────────────────────────────────────────

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
        needs_rebuild = False
        if version_path.exists():
            existing_version = int(version_path.read_text().strip())
            if existing_version < CACHE_VERSION:
                print(f"缓存版本不兼容 (v{existing_version} → v{CACHE_VERSION})，强制重建所有日期...")
                needs_rebuild = True
        elif any(daily_dir.glob('*.parquet')):
            # No version marker + existing files = old v1 cache, must rebuild
            print(f"检测到旧版缓存（无版本标记），强制重建所有日期...")
            needs_rebuild = True
        if needs_rebuild:
            import shutil
            for f in daily_dir.glob('*.parquet'):
                f.unlink()
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


class CachedProvider:
    """与 DataProvider 接口兼容的缓存读取器。

    用于替换 DataProvider，从 Parquet 缓存读取而非数据库查询。
    可安全地在多进程间共享（只读文件操作）。
    """

    def __init__(self, cache: DailyDataCache):
        self.cache = cache

    # ── DataProvider 兼容接口 ────────────────────────

    def get_all_stock_codes(self) -> list:
        return self.cache.stock_codes

    def get_batch_histories(
        self,
        stock_codes: list,
        end_date: str,
        start_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        return self.cache.get_histories(stock_codes, end_date, start_date)

    def get_stocks_for_date(self, stock_codes: list, date: str) -> dict:
        """返回 {stock_code: row_dict}，与 DataProvider 接口兼容"""
        df = self.cache.get_daily(date)
        if df.empty:
            return {}

        code_set = set(stock_codes)
        df = df[df['stock_code'].isin(code_set)]
        if df.empty:
            return {}

        # 用 to_dict('records') 比 iterrows 快 10x+
        result = {}
        for row in df.to_dict('records'):
            code = row['stock_code']
            # 确保 trade_date 是 Timestamp
            row['trade_date'] = pd.Timestamp(date)
            result[code] = row
        return result

    def get_daily_dataframe(self, date: str) -> pd.DataFrame:
        """获取某日全市场 DataFrame（优化路径，避免 dict 转换）"""
        return self.cache.get_daily(date)

    def get_index_data(
        self,
        index_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        return self.cache.get_index_data(index_code, start_date, end_date)
