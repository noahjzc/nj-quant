# 预测者网数据同步方案

## 概述

预测者网（yucezhe.com）提供 A 股全市场日线数据推送服务（含完整技术指标），与现有数据源的关系：

| 数据源 | 用途 | 时间 |
|--------|------|------|
| 预测者网 | 日线数据主要来源（带指标） | 每日 18:00 |
| akshare | 盘中实时数据 | 盘中 |
| Tushare Pro | 财务数据扩展 | 按需 |

## 数据格式

### ZIP 包结构（每日推送）
```
overview-push-YYYYMMDD/
  ├── stock overview.csv    # 全市场股票数据（单文件，含所有股票）
  ├── index data.csv       # 指数数据
  ├── industry overview.csv # 行业统计
  └── readme.txt
```

### stock overview.csv 字段
与现有 `stock_daily` 表字段完全一致，含：股票代码、名称、OHLCV、复权价、换手率、市值、PE/PB/PS、PCF、MA/MACD/KDJ/BOLL/RSI/PSY 等指标。

**字段名对照**：股票代码格式为 `sh600000`（直接可用）。

### index data.csv 字段
```
index_code, date, open, close, low, high, volume, money, change
```
对应 `index_daily` 表字段。

## 程序设计

### `sync_overview.py`

统一入口，支持两种模式：

```
# 每日 cron 使用（下载 ZIP）
python back_testing/data/sync/sync_overview.py --mode daily

# 一次性历史补全（本地 CSV 目录）
python back_testing/data/sync/sync_overview.py --mode backfill --data-dir /path/to/stock/csv
```

#### 模式 1：daily（每日定时）

```
1. API 下载 ZIP
   - GET https://yucezhe.com/api/v1/data/today?name=overview-data-push&email=...&key=...
   - 成功 → 下载 ZIP；未就绪 → 轮询等待（30s 间隔，最多 10 次）
   - ZIP 保存到临时目录

2. 解压并解析
   - stock overview.csv → DataFrame
   - index data.csv → DataFrame

3. 获取实际交易日期
   - 从 stock overview.csv 首行读取 trade_date（即数据的实际交易日）
   - 因为节假日，实际可能是上一交易日

4. 逐股票 upsert
   - 对每只股票：查 DB 中该股的已有日期 → 只 upsert 不存在的日期
   - stock_code + trade_date 为键，ON CONFLICT DO UPDATE

5. 指数数据 upsert
   - index data.csv → index_daily 表
   - volume 单位：手 → 股（×100），money 单位千元 → 元（×1000）

6. 构建缓存
   - 从 DB 查询当天全市场数据
   - 写入 DAILY_CACHE_DIR/{date}.parquet
   - 更新 trading_dates.parquet
```

#### 模式 2：backfill（一次性历史补全）

```
1. 扫描 CSV 目录
   - 目录下每股票一个 .csv 文件（如 sh600000.csv）

2. 对每只股票
   - 从 DB 读取该股票已有日期的最大日期
   - 读取 CSV，只取 trade_date > DB最大日期 的记录
   - upsert 到 DB

3. 构建缓存（同 daily 模式）
```

### 字段映射

直接复用 `import_overview_data.py` 中的 `CSV_TO_DB_MAP`：

```python
CSV_TO_DB = {
    "股票代码": "stock_code",
    "股票名称": "stock_name",
    "交易日期": "trade_date",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "后复权价": "adj_close",
    "前复权价": "prev_adj_close",
    "涨跌幅": "change_pct",
    "成交量": "volume",
    "成交额": "turnover_amount",
    "换手率": "turnover_rate",
    "流通市值": "circulating_mv",
    "总市值": "total_mv",
    "是否涨停": "limit_up",
    "是否跌停": "limit_down",
    "市盈率TTM": "pe_ttm",
    "市销率TTM": "ps_ttm",
    "市现率TTM": "pcf_ttm",
    "市净率": "pb",
    "MA_5": "ma_5",
    "MA_10": "ma_10",
    "MA_20": "ma_20",
    "MA_30": "ma_30",
    "MA_60": "ma_60",
    "MA金叉死叉": "ma_cross",
    "MACD_DIF": "macd_dif",
    "MACD_DEA": "macd_dea",
    "MACD_MACD": "macd_hist",
    "MACD_金叉死叉": "macd_cross",
    "KDJ_K": "kdj_k",
    "KDJ_D": "kdj_d",
    "KDJ_J": "kdj_j",
    "KDJ_金叉死叉": "kdj_cross",
    "布林线中轨": "boll_mid",
    "布林线上轨": "boll_upper",
    "布林线下轨": "boll_lower",
    "psy": "psy",
    "psyma": "psyma",
    "rsi1": "rsi_1",
    "rsi2": "rsi_2",
    "rsi3": "rsi_3",
    "振幅": "amplitude",
    "量比": "volume_ratio",
}
```

### 关键设计决策

| 决策 | 原因 |
|------|------|
| Upsert 而非 Delete+Insert | 避免数据覆盖，可重复运行 |
| 逐股票判断缺失日期 | 比全量对比更高效，内存占用低 |
| 数据来自 CSV 不算指标 | 预测者网已含完整技术指标，无需重算 |
| ZIP 保存临时目录 | 下载完成即可解析，不需要常驻 |

### 指数映射

| index_code（数据源） | DB 存储 |
|----------------------|---------|
| sh000001 | 上证综指 |
| sh000300 | 沪深300 |
| sz399001 | 深证成指 |
| sz399006 | 创业板指 |

## 文件结构

```
back_testing/data/sync/
├── sync_overview.py          # 统一入口
├── backfill_overview.py     # （旧文件，废弃）
├── import_overview_data.py  # （旧文件，废弃）
└── overview_client.py        # 新增：API 客户端
```

## 环境配置

```bash
# 环境变量
YUCEZHE_EMAIL=your_email@example.com
YUCEZHE_API_KEY=your_api_key
YUCEZHE_PRODUCT=overview-data-push
```

## Cron 配置

```bash
# 每日 20:00 运行
0 20 * * 1-5 cd /path/to/nj-quant && python back_testing/data/sync/sync_overview.py --mode daily >> logs/sync_overview.log 2>&1
```

## 错误处理

| 情况 | 处理 |
|------|------|
| API 返回数据未就绪 | 轮询等待 30s×10 次后退出（退出码 1） |
| ZIP 解压失败 | 打印错误，退出 |
| 某只股票 upsert 失败 | 跳过该股票，记录警告，继续其他 |
| DB 连接失败 | 退出（退出码 1） |
| 缓存写入失败 | 警告但不退出 |

## 依赖

无新依赖。复用现有 `sqlalchemy`、`pandas`、`requests/urllib`。
