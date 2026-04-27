-- 创建 stock_daily 表
CREATE TABLE IF NOT EXISTS stock_daily (
    stock_code VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    open DECIMAL(10,3),
    high DECIMAL(10,3),
    low DECIMAL(10,3),
    close DECIMAL(10,3),
    volume DECIMAL(15,2),
    turnover DECIMAL(15,2),
    amplitude DECIMAL(10,4),
    change_pct DECIMAL(10,4),
    is_trading BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_code, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_stock_trade_date ON stock_daily(stock_code, trade_date);

-- 创建 stock_meta 表
CREATE TABLE IF NOT EXISTS stock_meta (
    stock_code VARCHAR(10) PRIMARY KEY,
    stock_name VARCHAR(50),
    list_date DATE,
    delist_date DATE,
    industry VARCHAR(50),
    market VARCHAR(10),
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建 stock_financial 表
CREATE TABLE IF NOT EXISTS stock_financial (
    stock_code VARCHAR(10) NOT NULL,
    report_period DATE NOT NULL,
    publish_date DATE,
    roe DECIMAL(10,4),
    roa DECIMAL(10,4),
    gross_margin DECIMAL(10,4),
    net_margin DECIMAL(10,4),
    debt_ratio DECIMAL(10,4),
    current_ratio DECIMAL(10,4),
    quick_ratio DECIMAL(10,4),
    pe_ttm DECIMAL(12,4),
    pb DECIMAL(10,4),
    ps_ttm DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_code, report_period)
);

CREATE INDEX IF NOT EXISTS idx_stock_publish ON stock_financial(stock_code, publish_date);

-- 创建 index_daily 表
CREATE TABLE IF NOT EXISTS index_daily (
    index_code VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume DECIMAL(15,2),
    turnover DECIMAL(15,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (index_code, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_index_trade_date ON index_daily(index_code, trade_date);