# Live Trading Signal Pipeline Design

## Overview

将现有的 Daily Rotation 回测系统扩展为 **每日实盘信号管线**。核心思路：下午 2:30 用日内数据生成信号，收盘前人工执行，晚间补全当日完整数据。

## Daily Workflow

```
14:25  cron → intraday_signal.py
       ├─ AKShare 全市场实时快照
       ├─ DB 加载昨日完整数据
       ├─ 合并 + 向量化重算全部技术指标
       ├─ 运行信号引擎
       └─ 信号写入 DB

14:35  信号就绪，Web 可查看

14:35～15:00  用户审核信号，手动下单，Web 确认执行或放弃

18:00  cron → night_backfill.py
       ├─ Tushare Pro 批量拉取当日完整日线
       ├─ 更新 DB stock_daily
       ├─ 重算当日完整技术指标写入 DB
       ├─ 增量追加当日 Parquet 缓存
       └─ 更新 cron 执行状态

09:00  cron → 检查 Web 存活
```

## Project Structure (新增部分)

```
signal_pipeline/
├── intraday_signal.py            # 14:25 入口
├── night_backfill.py             # 18:00 入口
├── indicator_calculator.py       # 向量化指标计算（MA/MACD/KDJ/Boll/RSI/PSY）
├── data_merger.py                # 日内数据 + 历史数据合并
└── data_sources/
    ├── tushare_client.py         # Tushare Pro（盘后批量）
    └── akshare_client.py         # AKShare（盘中实时快照）

web/
├── server/
│   ├── main.py                   # FastAPI 入口
│   ├── api/
│   │   ├── signals.py            # 信号 CRUD
│   │   ├── positions.py          # 持仓 & 收益
│   │   ├── data_browser.py       # 数据浏览
│   │   └── cron_status.py        # 任务追踪
│   └── models/
│       └── schemas.py            # Pydantic models
├── frontend/                     # React + Ant Design
│   ├── src/
│   │   ├── pages/
│   │   │   ├── DataBrowser/
│   │   │   ├── CronTracker/
│   │   │   ├── SignalTable/
│   │   │   └── Positions/
│   │   ├── components/
│   │   └── App.tsx
│   └── package.json
└── start.sh
```

## Data Sources

| 场景 | 数据源 | 接口 | 说明 |
|------|--------|------|------|
| 盘中快照 | AKShare | `stock_zh_a_spot_em()` | 全市场实时行情，免费 |
| 盘后补全 | Tushare Pro | `daily` + `daily_basic` + `adj_factor` | 批量日线，2000积分档 |
| 历史数据 | 现有 DB | PostgreSQL | BaoStock 导入，已有完整指标 |

Tushare 只提供原始 OHLCV + 估值 + 市值，技术指标全部由 `indicator_calculator.py` 自己算。

## Indicator Calculator

一次性向量化计算全市场所有股票的技术指标，避免逐只循环。

输入：DataFrame（全市场 × 近 60 天 OHLCV），按 `stock_code` 分组。
输出：每只股票的 MA_5/10/20/30/60、MACD、KDJ、Boll、RSI、PSY 及金叉/死叉信号。

14:25 场景：合并昨日完整数据 + 今日部分数据（最新价作为 close），重算所有指标。
18:00 场景：当日完整数据写入 DB 后，重算并回写。

## Web Modules

后端 FastAPI，前端 React + Ant Design。

### a. 全量数据查看
- 股票列表分页表格（按代码/名称/行业筛选）
- 点击进入详情：K 线 + 指标走势（ECharts/AntV）
- 支持日期范围选择

### b. Cron 任务追踪
- 时间线：每次任务执行的时间、状态（running/success/failed）、耗时
- 失败任务红色高亮，可展开查看错误详情
- 数据完整性指示：最近交易日是否补全、缓存是否已构建

### c. 每日交易信号表
- 当天信号列表：代码/名称/方向/建议仓位%/建议价格区间/触发信号
- 每行两个操作按钮：「确认执行」「放弃」
- 确认执行弹出表单：实际成交价
- 信号三态：待执行 / 已确认 / 已放弃
- 已确认和已放弃的信号可筛选查看

### d. 持仓 & 收益
- 顶部资产总览卡片：总资产 / 可用资金 / 持仓市值 / 累计收益
- 「补充资金」按钮，弹出表单填入金额，写入资金流水表
- 当前持仓表：代码/名称/成本价/现价/盈亏%/持有天数
- 历史交易记录：买入日/卖出日/买入价/卖出价/收益率
- 基于信号确认自动维护：确认买入 → 扣减可用资金 + 新增持仓，确认卖出 → 增加可用资金 + 更新盈亏

## New DB Tables

```sql
CREATE TABLE daily_signal (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    stock_code VARCHAR(10) NOT NULL,
    stock_name VARCHAR(50),
    direction VARCHAR(4),           -- 'BUY' / 'SELL'
    target_pct NUMERIC(5,2),        -- 建议仓位%
    price_low NUMERIC(10,3),
    price_high NUMERIC(10,3),
    signal_reason TEXT,
    status VARCHAR(10) DEFAULT 'pending',  -- pending / confirmed / skipped
    executed_price NUMERIC(10,3),
    confirmed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE position (
    id SERIAL PRIMARY KEY,
    stock_code VARCHAR(10) NOT NULL,
    stock_name VARCHAR(50),
    buy_date DATE NOT NULL,
    buy_price NUMERIC(10,3) NOT NULL,
    shares INT NOT NULL,
    sell_date DATE,
    sell_price NUMERIC(10,3),
    profit_pct NUMERIC(10,4),
    status VARCHAR(10) DEFAULT 'OPEN'
);

CREATE TABLE capital_ledger (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(10) NOT NULL,   -- 'INIT' / 'DEPOSIT' / 'BUY' / 'SELL'
    amount NUMERIC(15,2) NOT NULL,     -- 正数=入金，负数=出金
    balance_after NUMERIC(15,2) NOT NULL,
    related_signal_id INT,
    note TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE cron_log (
    id SERIAL PRIMARY KEY,
    task_name VARCHAR(50),
    status VARCHAR(10),
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    error_message TEXT,
    metadata JSONB
);
```

## Cache Building

晚间 `night_backfill.py` 完成后，增量追加当日 Parquet：

```
DB 更新完 → 构建 daily/{YYYY-MM-DD}.parquet → 更新 trading_dates.parquet
```

只追加新的一天，不重建全量。复用现有 `DailyDataCache`。

## Deployment

```
Linux 服务器
├── FastAPI    →  systemd 常驻 :8080
├── React      →  nginx 静态托管
├── PostgreSQL →  复用现有 DB
├── Python venv
└── cron       →  14:25 / 18:00 / 09:00
```

无 Docker，无消息队列。

## Fault Tolerance

- Tushare / AKShare 调用失败 → 等 2 分钟重试，最多 3 次 → 仍失败写 cron_log 并 Web 标红
- AKShare 14:30 前必须成功，否则当天无信号（标记到 cron_log）
- 晚间补全失败 → 次日 cron 可补跑，18:00 重试的是前一交易日数据
- systemd `Restart=always` 保 Web 存活
- 补全脚本幂等：insert on conflict 不重复写入
