"""
数据提供器 - 统一管理股票数据读取

支持两种数据源：
- PostgreSQL（生产环境，默认）
- 本地 Parquet/CSV（回退/历史兼容）

使用方式：
    from back_testing.data.data_provider import DataProvider

    provider = DataProvider()  # 默认使用数据库
    # 或者
    provider = DataProvider(use_db=False)  # 使用本地文件

    df = provider.get_stock_data('sh600519')
    df = provider.get_stock_data('sh600519', date='2024-01-15')  # 筛选到指定日期
"""
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from back_testing.data.db.connection import get_engine, get_session
from back_testing.data.db.models import StockDaily, StockMeta


class DataProvider:
    """
    统一数据访问层

    支持两种数据源：
    - PostgreSQL（生产环境，默认）
    - 本地 Parquet/CSV（回退/历史兼容）
    """

    def __init__(
        self,
        use_db: bool = True,
        data_dir: Optional[str] = None
    ):
        """
        Args:
            use_db: 是否使用数据库（默认 True）
            data_dir: 数据目录（仅 use_db=False 时使用）
        """
        self.use_db = use_db

        if use_db:
            self.engine = get_engine()
            self.Session = get_session()
        else:
            self.use_parquet = True
            if data_dir is None:
                project_root = Path(__file__).parent.parent
                self.data_dir = project_root / 'data' / 'daily_ycz'
            else:
                self.data_dir = Path(data_dir)
            self.csv_dir = Path(r'D:\workspace\code\mine\quant\data\metadata\daily_ycz')

    def _get_from_db(
        self,
        stock_code: str,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """从数据库获取股票数据"""
        session = self.Session()

        try:
            query = session.query(StockDaily).filter(StockDaily.stock_code == stock_code)

            if date is not None:
                date_ts = pd.to_datetime(date)
                query = query.filter(StockDaily.trade_date < date_ts)

            if start_date is not None:
                query = query.filter(StockDaily.trade_date >= pd.to_datetime(start_date))

            if end_date is not None:
                query = query.filter(StockDaily.trade_date <= pd.to_datetime(end_date))

            query = query.order_by(StockDaily.trade_date.asc())

            results = query.all()

            if not results:
                return pd.DataFrame()

            data = {
                'stock_code': [r.stock_code for r in results],
                'trade_date': [r.trade_date for r in results],
                'open': [float(r.open) if r.open else 0 for r in results],
                'high': [float(r.high) if r.high else 0 for r in results],
                'low': [float(r.low) if r.low else 0 for r in results],
                'close': [float(r.close) if r.close else 0 for r in results],
                'volume': [float(r.volume) if r.volume else 0 for r in results],
                'turnover': [float(r.turnover) if r.turnover else 0 for r in results],
                'amplitude': [float(r.amplitude) if r.amplitude else 0 for r in results],
                'change_pct': [float(r.change_pct) if r.change_pct else 0 for r in results],
            }

            return pd.DataFrame(data).set_index('trade_date')

        finally:
            session.close()

    def _get_from_file(
        self,
        stock_code: str,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """从本地文件获取股票数据（原有逻辑）"""
        file_path = self._get_file_path(stock_code)

        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, encoding='gbk')

        date_col = '交易日期'
        if date_col not in df.columns:
            for col in ['date', 'Date', 'DATE', '交易日期']:
                if col in df.columns:
                    date_col = col
                    break

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        if date is not None:
            date_ts = pd.to_datetime(date)
            df = df[df[date_col] < date_ts]

        if start_date is not None:
            df = df[df[date_col] >= pd.to_datetime(start_date)]

        if end_date is not None:
            df = df[df[date_col] <= pd.to_datetime(end_date)]

        return df

    def _get_file_path(self, stock_code: str) -> Path:
        """获取数据文件路径"""
        if self.use_parquet:
            parquet_path = self.data_dir / f'{stock_code}.parquet'
            if parquet_path.exists():
                return parquet_path
            csv_path = self.csv_dir / f'{stock_code}.csv'
            if csv_path.exists():
                return csv_path
            raise FileNotFoundError(f"找不到数据文件: {stock_code}")
        else:
            csv_path = self.csv_dir / f'{stock_code}.csv'
            if csv_path.exists():
                return csv_path
            raise FileNotFoundError(f"找不到数据文件: {stock_code}")

    def get_stock_data(
        self,
        stock_code: str,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取股票历史数据"""
        if self.use_db:
            return self._get_from_db(stock_code, date, start_date, end_date)
        else:
            return self._get_from_file(stock_code, date, start_date, end_date)

    def get_stock_price(
        self,
        stock_code: str,
        date: Union[str, pd.Timestamp]
    ) -> Optional[float]:
        """获取指定日期的收盘价（后复权）"""
        if self.use_db:
            session = self.Session()
            try:
                date_ts = pd.to_datetime(date)

                result = session.query(StockDaily).filter(
                    StockDaily.stock_code == stock_code,
                    StockDaily.trade_date <= date_ts
                ).order_by(StockDaily.trade_date.desc()).first()

                return float(result.close) if result and result.close else None
            finally:
                session.close()
        else:
            date_ts = pd.to_datetime(date)
            df = self._get_from_file(stock_code)
            if '交易日期' in df.columns:
                df['trade_date'] = pd.to_datetime(df['交易日期'])
            else:
                return None
            hist = df[df['trade_date'] <= date_ts]
            if len(hist) == 0:
                return None
            return hist.iloc[-1].get('后复权价')

    def get_all_stock_codes(self) -> list:
        """获取所有股票代码（排除北交所、退市股）"""
        if self.use_db:
            session = self.Session()
            try:
                codes = [r[0] for r in session.query(StockMeta.stock_code).filter(
                    StockMeta.is_active == True,
                    StockMeta.market != '北'
                ).all()]
                return codes
            finally:
                session.close()
        else:
            if self.use_parquet and self.data_dir.exists():
                files = list(self.data_dir.glob('*.parquet'))
                return [f.stem for f in files if not f.stem.startswith('bj')]
            elif self.csv_dir.exists():
                files = list(self.csv_dir.glob('*.csv'))
                return [f.stem for f in files if not f.stem.startswith('index') and not f.stem.startswith('bj')]
            return []

    def get_latest_price(self, stock_code: str) -> Optional[float]:
        """获取最近收盘价"""
        return self.get_stock_price(stock_code, pd.Timestamp.now())

    def get_latest_trade_date(self, stock_code: str):
        """获取最近交易日"""
        if self.use_db:
            session = self.Session()
            try:
                result = session.query(StockDaily.trade_date).filter(
                    StockDaily.stock_code == stock_code
                ).order_by(StockDaily.trade_date.desc()).first()
                return result[0] if result else None
            finally:
                session.close()
        else:
            df = self.get_stock_data(stock_code)
            if len(df) == 0:
                return None
            if '交易日期' in df.columns:
                return df['交易日期'].iloc[-1]
            return None


# 全局默认实例
_default_provider: Optional[DataProvider] = None


def get_provider() -> DataProvider:
    """获取默认数据提供器（单例）"""
    global _default_provider
    if _default_provider is None:
        _default_provider = DataProvider(use_db=True)
    return _default_provider
