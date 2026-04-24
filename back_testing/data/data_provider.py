"""
数据提供器 - 统一管理股票数据读取（PostgreSQL only）

使用方式：
    from back_testing.data.data_provider import DataProvider

    provider = DataProvider()
    df = provider.get_stock_data('sh600519')
    df = provider.get_stock_data('sh600519', date='2024-01-15')  # 筛选到指定日期
"""
from typing import Optional

import pandas as pd

from back_testing.data.db.connection import get_engine, get_session
from back_testing.data.db.models import StockDaily, StockMeta


class DataProvider:
    """
    统一数据访问层（PostgreSQL）

    所有数据均从 PostgreSQL 数据库读取，不再支持本地文件。
    """

    def __init__(self):
        self.engine = get_engine()
        self.Session = get_session()

    def get_stock_data(
        self,
        stock_code: str,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取股票历史数据"""
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
                'trade_date': [pd.Timestamp(r.trade_date) for r in results],
                # Basic OHLCV
                'open': [float(r.open) if r.open else 0 for r in results],
                'high': [float(r.high) if r.high else 0 for r in results],
                'low': [float(r.low) if r.low else 0 for r in results],
                'close': [float(r.close) if r.close else 0 for r in results],
                'volume': [float(r.volume) if r.volume else 0 for r in results],
                'turnover_amount': [float(r.turnover_amount) if r.turnover_amount else 0 for r in results],
                # Adjustment prices
                'adj_close': [float(r.adj_close) if r.adj_close else 0 for r in results],
                'prev_adj_close': [float(r.prev_adj_close) if r.prev_adj_close else 0 for r in results],
                # Price metrics
                'amplitude': [float(r.amplitude) if r.amplitude else 0 for r in results],
                'change_pct': [float(r.change_pct) if r.change_pct else 0 for r in results],
                # Trading indicators
                'turnover_rate': [float(r.turnover_rate) if r.turnover_rate else 0 for r in results],
                'volume_ratio': [float(r.volume_ratio) if r.volume_ratio else 0 for r in results],
                'circulating_mv': [float(r.circulating_mv) if r.circulating_mv else 0 for r in results],
                'total_mv': [float(r.total_mv) if r.total_mv else 0 for r in results],
                'limit_up': [r.limit_up for r in results],
                'limit_down': [r.limit_down for r in results],
                # Valuation factors
                'pe_ttm': [float(r.pe_ttm) if r.pe_ttm else None for r in results],
                'ps_ttm': [float(r.ps_ttm) if r.ps_ttm else None for r in results],
                'pcf_ttm': [float(r.pcf_ttm) if r.pcf_ttm else None for r in results],
                'pb': [float(r.pb) if r.pb else None for r in results],
                # MA indicators
                'ma_5': [float(r.ma_5) if r.ma_5 else None for r in results],
                'ma_10': [float(r.ma_10) if r.ma_10 else None for r in results],
                'ma_20': [float(r.ma_20) if r.ma_20 else None for r in results],
                'ma_30': [float(r.ma_30) if r.ma_30 else None for r in results],
                'ma_60': [float(r.ma_60) if r.ma_60 else None for r in results],
                'ma_cross': [r.ma_cross for r in results],
                # MACD indicators
                'macd_dif': [float(r.macd_dif) if r.macd_dif else None for r in results],
                'macd_dea': [float(r.macd_dea) if r.macd_dea else None for r in results],
                'macd_hist': [float(r.macd_hist) if r.macd_hist else None for r in results],
                'macd_cross': [r.macd_cross for r in results],
                # KDJ indicators
                'kdj_k': [float(r.kdj_k) if r.kdj_k else None for r in results],
                'kdj_d': [float(r.kdj_d) if r.kdj_d else None for r in results],
                'kdj_j': [float(r.kdj_j) if r.kdj_j else None for r in results],
                'kdj_cross': [r.kdj_cross for r in results],
                # Bollinger bands
                'boll_mid': [float(r.boll_mid) if r.boll_mid else None for r in results],
                'boll_upper': [float(r.boll_upper) if r.boll_upper else None for r in results],
                'boll_lower': [float(r.boll_lower) if r.boll_lower else None for r in results],
                # RSI indicators
                'rsi_1': [float(r.rsi_1) if r.rsi_1 else None for r in results],
                'rsi_2': [float(r.rsi_2) if r.rsi_2 else None for r in results],
                'rsi_3': [float(r.rsi_3) if r.rsi_3 else None for r in results],
                # Psychology indicators
                'psy': [float(r.psy) if r.psy else None for r in results],
                'psyma': [float(r.psyma) if r.psyma else None for r in results],
                # Metadata
                'stock_name': [r.stock_name for r in results],
                'industry': [r.industry for r in results],
                'concept': [r.concept for r in results],
                'area': [r.area for r in results],
            }

            return pd.DataFrame(data).set_index('trade_date')

        finally:
            session.close()

    def get_stock_price(
        self,
        stock_code: str,
        date: str
    ) -> Optional[float]:
        """获取指定日期的收盘价（复权）"""
        session = self.Session()
        try:
            date_ts = pd.to_datetime(date)

            result = session.query(StockDaily).filter(
                StockDaily.stock_code == stock_code,
                StockDaily.trade_date <= date_ts
            ).order_by(StockDaily.trade_date.desc()).first()

            return float(result.adj_close) if result and result.adj_close else None
        finally:
            session.close()

    def get_all_stock_codes(self) -> list:
        """获取所有股票代码（排除北交所、退市股）"""
        session = self.Session()
        try:
            codes = [r[0] for r in session.query(StockMeta.stock_code).filter(
                StockMeta.is_active == True,
                StockMeta.market != '北'
            ).all()]
            return codes
        finally:
            session.close()

    def get_latest_price(self, stock_code: str) -> Optional[float]:
        """获取最近收盘价（后复权）"""
        session = self.Session()
        try:
            result = session.query(StockDaily).filter(
                StockDaily.stock_code == stock_code
            ).order_by(StockDaily.trade_date.desc()).first()

            return float(result.adj_close) if result and result.adj_close else None
        finally:
            session.close()

    def get_latest_trade_date(self, stock_code: str):
        """获取最近交易日"""
        session = self.Session()
        try:
            result = session.query(StockDaily.trade_date).filter(
                StockDaily.stock_code == stock_code
            ).order_by(StockDaily.trade_date.desc()).first()
            return result[0] if result else None
        finally:
            session.close()


# 全局默认实例
_default_provider: Optional[DataProvider] = None
_provider_lock = __import__('threading').Lock()


def get_provider() -> DataProvider:
    """获取默认数据提供器（单例，线程安全）"""
    global _default_provider
    with _provider_lock:
        if _default_provider is None:
            _default_provider = DataProvider()
        return _default_provider


def close_provider():
    """关闭默认数据提供器"""
    global _default_provider
    with _provider_lock:
        if _default_provider is not None:
            from back_testing.data.db.connection import close_session
            close_session()
            _default_provider = None
