-- ============================================================
-- ECS 初始化步骤 2: 创建表结构（用 quant_user 在 quant_db 上执行）
-- 在服务器上执行:
--   psql -U quant_user -d quant_db -f scripts/server_setup_02_tables.sql
-- ============================================================

-- 股票日线行情表
CREATE TABLE IF NOT EXISTS stock_daily (
    stock_code        VARCHAR(10)     NOT NULL,
    trade_date        DATE            NOT NULL,
    open              NUMERIC(15,3),
    high              NUMERIC(15,3),
    low               NUMERIC(15,3),
    close             NUMERIC(15,3),
    volume            NUMERIC(15,2),
    turnover_amount   NUMERIC(15,2),
    adj_close         NUMERIC(15,3),
    prev_adj_close    NUMERIC(15,3),
    amplitude         NUMERIC(10,4),
    change_pct        NUMERIC(10,4),
    turnover_rate     NUMERIC(10,4),
    volume_ratio      NUMERIC(10,4),
    circulating_mv    NUMERIC(15,2),
    total_mv          NUMERIC(15,2),
    limit_up          BOOLEAN         DEFAULT FALSE,
    limit_down        BOOLEAN         DEFAULT FALSE,
    pe_ttm            NUMERIC(15,4),
    ps_ttm            NUMERIC(15,4),
    pcf_ttm           NUMERIC(15,4),
    pb                NUMERIC(10,4),
    ma_5              NUMERIC(15,3),
    ma_10             NUMERIC(15,3),
    ma_20             NUMERIC(15,3),
    ma_30             NUMERIC(15,3),
    ma_60             NUMERIC(15,3),
    ma_cross          TEXT,
    macd_dif          NUMERIC(15,6),
    macd_dea          NUMERIC(15,6),
    macd_hist         NUMERIC(15,6),
    macd_cross        TEXT,
    kdj_k             NUMERIC(10,4),
    kdj_d             NUMERIC(10,4),
    kdj_j             NUMERIC(10,4),
    kdj_cross         TEXT,
    boll_mid          NUMERIC(15,3),
    boll_upper        NUMERIC(15,3),
    boll_lower        NUMERIC(15,3),
    psy               NUMERIC(10,4),
    psyma             NUMERIC(10,4),
    rsi_1             NUMERIC(10,4),
    rsi_2             NUMERIC(10,4),
    rsi_3             NUMERIC(10,4),
    stock_name        VARCHAR(50),
    industry          VARCHAR(100),
    concept           TEXT,
    area              VARCHAR(50),
    PRIMARY KEY (stock_code, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_stock_trade_date ON stock_daily (stock_code, trade_date);

-- 股票元数据表
CREATE TABLE IF NOT EXISTS stock_meta (
    stock_code   VARCHAR(50)  PRIMARY KEY,
    stock_name   VARCHAR(50),
    list_date    DATE,
    delist_date  DATE,
    industry     VARCHAR(50),
    market       VARCHAR(10),
    is_active    BOOLEAN DEFAULT TRUE,
    updated_at   TIMESTAMP,
    created_at   TIMESTAMP
);

-- 财务数据表
CREATE TABLE IF NOT EXISTS stock_financial (
    stock_code    VARCHAR(10)  NOT NULL,
    report_period DATE         NOT NULL,
    publish_date  DATE,
    roe           NUMERIC(10,4),
    roa           NUMERIC(10,4),
    gross_margin  NUMERIC(10,4),
    net_margin    NUMERIC(10,4),
    debt_ratio    NUMERIC(10,4),
    current_ratio NUMERIC(10,4),
    quick_ratio   NUMERIC(10,4),
    pe_ttm        NUMERIC(12,4),
    pb            NUMERIC(10,4),
    ps_ttm        NUMERIC(10,4),
    created_at    TIMESTAMP,
    PRIMARY KEY (stock_code, report_period)
);

CREATE INDEX IF NOT EXISTS idx_stock_publish ON stock_financial (stock_code, publish_date);

-- 指数行情表
CREATE TABLE IF NOT EXISTS index_daily (
    index_code  VARCHAR(10)  NOT NULL,
    trade_date  DATE         NOT NULL,
    open        NUMERIC(12,4),
    high        NUMERIC(12,4),
    low         NUMERIC(12,4),
    close       NUMERIC(12,4),
    volume      NUMERIC(15,2),
    turnover    NUMERIC(15,2),
    created_at  TIMESTAMP,
    PRIMARY KEY (index_code, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_index_trade_date ON index_daily (index_code, trade_date);


GRANT ALL PRIVILEGES ON index_daily TO quant_user;