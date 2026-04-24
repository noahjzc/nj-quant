"""数据库表模型"""
from datetime import date, datetime
from sqlalchemy import Column, String, Date, Numeric, Boolean, DateTime, Index, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class StockDaily(Base):
    """日线行情表"""
    __tablename__ = 'stock_daily'

    # Primary key
    stock_code = Column(String(10), primary_key=True)
    trade_date = Column(Date, primary_key=True)

    # Basic OHLCV
    open = Column(Numeric(15, 3))
    high = Column(Numeric(15, 3))
    low = Column(Numeric(15, 3))
    close = Column(Numeric(15, 3))
    volume = Column(Numeric(15, 2))
    turnover_amount = Column(Numeric(15, 2))  # 成交额

    # Adjustment prices
    adj_close = Column(Numeric(15, 3))  # 后复权价
    prev_adj_close = Column(Numeric(15, 3))  # 前复权价

    # Price metrics
    amplitude = Column(Numeric(10, 4))  # 振幅
    change_pct = Column(Numeric(10, 4))  # 涨跌幅

    # Trading indicators
    turnover_rate = Column(Numeric(10, 4))  # 换手率
    volume_ratio = Column(Numeric(10, 4))  # 量比
    circulating_mv = Column(Numeric(15, 2))  # 流通市值
    total_mv = Column(Numeric(15, 2))  # 总市值

    # Limit up/down
    limit_up = Column(Boolean, default=False)  # 是否涨停
    limit_down = Column(Boolean, default=False)  # 是否跌停

    # Valuation factors
    pe_ttm = Column(Numeric(15, 4))  # 市盈率TTM
    ps_ttm = Column(Numeric(15, 4))  # 市销率TTM
    pcf_ttm = Column(Numeric(15, 4))  # 市现率TTM
    pb = Column(Numeric(10, 4))  # 市净率

    # MA indicators
    ma_5 = Column(Numeric(15, 3))
    ma_10 = Column(Numeric(15, 3))
    ma_20 = Column(Numeric(15, 3))
    ma_30 = Column(Numeric(15, 3))
    ma_60 = Column(Numeric(15, 3))
    ma_cross = Column(Text)  # MA金叉死叉

    # MACD indicators
    macd_dif = Column(Numeric(15, 6))
    macd_dea = Column(Numeric(15, 6))
    macd_hist = Column(Numeric(15, 6))
    macd_cross = Column(Text)  # MACD金叉死叉

    # KDJ indicators
    kdj_k = Column(Numeric(10, 4))
    kdj_d = Column(Numeric(10, 4))
    kdj_j = Column(Numeric(10, 4))
    kdj_cross = Column(Text)  # KDJ金叉死叉

    # Bollinger bands
    boll_mid = Column(Numeric(15, 3))
    boll_upper = Column(Numeric(15, 3))
    boll_lower = Column(Numeric(15, 3))

    # Psychology indicators
    psy = Column(Numeric(10, 4))
    psyma = Column(Numeric(10, 4))

    # RSI indicators
    rsi_1 = Column(Numeric(10, 4))
    rsi_2 = Column(Numeric(10, 4))
    rsi_3 = Column(Numeric(10, 4))

    # Metadata
    stock_name = Column(String(50))
    industry = Column(String(100))
    concept = Column(Text)
    area = Column(String(50))

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