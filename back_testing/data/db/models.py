"""数据库表模型"""
from datetime import date, datetime
from sqlalchemy import Column, String, Date, Numeric, Boolean, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class StockDaily(Base):
    """日线行情表"""
    __tablename__ = 'stock_daily'

    stock_code = Column(String(10), primary_key=True)
    trade_date = Column(Date, primary_key=True)
    open = Column(Numeric(10, 3))
    high = Column(Numeric(10, 3))
    low = Column(Numeric(10, 3))
    close = Column(Numeric(10, 3))
    volume = Column(Numeric(15, 2))
    turnover = Column(Numeric(15, 2))
    amplitude = Column(Numeric(10, 4))
    change_pct = Column(Numeric(10, 4))
    is_trading = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index('idx_stock_trade_date', 'stock_code', 'trade_date'),
    )


class StockMeta(Base):
    """股票元数据表"""
    __tablename__ = 'stock_meta'

    stock_code = Column(String(10), primary_key=True)
    stock_name = Column(String(50))
    list_date = Column(Date)
    delist_date = Column(Date, nullable=True)
    industry = Column(String(50))
    market = Column(String(10))
    is_active = Column(Boolean, default=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    created_at = Column(DateTime, default=datetime.now)


class StockFinancial(Base):
    """财务数据表"""
    __tablename__ = 'stock_financial'

    stock_code = Column(String(10), primary_key=True)
    report_period = Column(Date, primary_key=True)
    publish_date = Column(Date)
    roe = Column(Numeric(10, 4), nullable=True)
    roa = Column(Numeric(10, 4), nullable=True)
    gross_margin = Column(Numeric(10, 4), nullable=True)
    net_margin = Column(Numeric(10, 4), nullable=True)
    debt_ratio = Column(Numeric(10, 4), nullable=True)
    current_ratio = Column(Numeric(10, 4), nullable=True)
    quick_ratio = Column(Numeric(10, 4), nullable=True)
    pe_ttm = Column(Numeric(12, 4), nullable=True)
    pb = Column(Numeric(10, 4), nullable=True)
    ps_ttm = Column(Numeric(10, 4), nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index('idx_stock_publish', 'stock_code', 'publish_date'),
    )


class IndexDaily(Base):
    """指数行情表"""
    __tablename__ = 'index_daily'

    index_code = Column(String(10), primary_key=True)
    trade_date = Column(Date, primary_key=True)
    open = Column(Numeric(12, 4))
    high = Column(Numeric(12, 4))
    low = Column(Numeric(12, 4))
    close = Column(Numeric(12, 4))
    volume = Column(Numeric(15, 2))
    turnover = Column(Numeric(15, 2))
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index('idx_index_trade_date', 'index_code', 'trade_date'),
    )