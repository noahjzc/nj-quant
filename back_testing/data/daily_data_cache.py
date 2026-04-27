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

    # ── 构建 ────────────────────────────────────────

    @staticmethod
    def build(
        start_date: str,
        end_date: str,
        cache_dir: str,
        preload_days: int = 30,
        benchmark_index: str = 'sh000300'
    ):
        """从数据库一次性加载全量数据，写入 Parquet 缓存。

        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
            cache_dir: 缓存根目录
            preload_days: 预加载天数（用于引擎的 _preload_histories）
            benchmark_index: 基准指数代码
        """
        from back_testing.data.db.connection import get_engine

        engine = get_engine()
        cache_path = Path(cache_dir) / f'{start_date}_{end_date}'
        daily_dir = cache_path / 'daily'
        index_dir = cache_path / 'index'
        daily_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)

        # 加载范围（含预加载窗口）
        load_start = (pd.Timestamp(start_date) - pd.Timedelta(days=preload_days)).strftime('%Y-%m-%d')
        load_end = end_date

        # ── 1. 股票代码 ──
        logger.info("加载股票代码...")
        codes_df = pd.read_sql(
            "SELECT stock_code FROM stock_meta WHERE is_active = TRUE AND market != '北'",
            engine
        )
        codes_df.to_parquet(cache_path / 'stock_codes.parquet', index=False)
        all_codes = codes_df['stock_code'].tolist()
        logger.info(f"  共 {len(all_codes)} 只")

        # ── 2. 日线数据（一次性查询全量） ──
        logger.info(f"加载日线数据: {load_start} ~ {load_end} ...")
        daily_df = pd.read_sql(
            "SELECT * FROM stock_daily WHERE trade_date >= %(start)s AND trade_date <= %(end)s",
            engine,
            params={'start': load_start, 'end': load_end}
        )
        # 将 Numeric → float，避免 Parquet 写入 Decimal 报错
        for col in daily_df.columns:
            if col == 'trade_date':
                continue
            if daily_df[col].dtype == object:
                try:
                    daily_df[col] = pd.to_numeric(daily_df[col])
                except (ValueError, TypeError):
                    pass

        # 转换日期列
        daily_df['trade_date'] = pd.to_datetime(daily_df['trade_date'])

        # 按日期分组写入（保留 trade_date 列，get_histories 等需要用它做 index）
        date_groups = daily_df.groupby(daily_df['trade_date'].dt.strftime('%Y-%m-%d'))
        file_count = 0
        for date_str, group in date_groups:
            group.to_parquet(daily_dir / f'{date_str}.parquet', index=False)
            file_count += 1
        logger.info(f"  写入 {file_count} 个日文件")

        # ── 3. 交易日列表（从日线数据推导） ──
        trading_dates = sorted(daily_df['trade_date'].dt.strftime('%Y-%m-%d').unique())
        pd.DataFrame({'trade_date': trading_dates}).to_parquet(
            cache_path / 'trading_dates.parquet', index=False
        )

        # ── 4. 指数数据 ──
        logger.info(f"加载指数数据: {benchmark_index} ...")
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
        logger.info(f"  共 {len(index_df)} 条")

        logger.info(f"缓存构建完成: {cache_path}")
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
