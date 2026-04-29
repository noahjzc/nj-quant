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
