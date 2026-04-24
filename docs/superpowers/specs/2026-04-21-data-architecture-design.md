# 数据架构设计文档

**日期**: 2026-04-21
**主题**: PostgreSQL 数据层架构
**状态**: 待用户确认

---

## 1. 背景与目标

当前项目使用本地 Parquet/CSV 文件存储历史行情数据，已实现多因子选股和回测功能。为支持更丰富的数据维度、新鲜数据实时获取、以及未来云端部署，需要将数据层迁移至 PostgreSQL 数据库。

### 1.1 需求摘要

| 维度 | 内容 |
|------|------|
| **数据源** | akshare（全市场A股） |
| **存储** | PostgreSQL（本地 Windows 安装，未来迁移云端） |
| **数据范围** | 全市场A股，2021年至今（约5年历史） |
| **历史同步** | 一次性初始化脚本 |
| **日常更新** | 盘中(14:30)监控 + 收盘后(15:30)决策 |
| **财务数据** | 按季度，按个股发布日期同步 |
| **停牌/退市** | 需要处理 |
| **价格类型** | 后复权价（与现有系统一致） |

---

## 2. 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    akshare API                          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              数据同步层 (Python Scripts)                  │
│  ┌─────────────────┐    ┌─────────────────────────┐   │
│  │ init_history.py  │    │ daily_update.py         │   │
│  │ 一次性初始化      │    │ 每日定时增量更新          │   │
│  └─────────────────┘    └─────────────────────────┘   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              PostgreSQL 数据库                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ stock_daily  │  │ stock_meta  │  │ index_daily  │    │
│  │ stock_financial                                      │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│          back_testing/data/data_provider.py              │
│              统一数据访问层（兼容现有接口）                 │
└─────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│          back_testing/factors/factor_utils.py             │
│        技术指标实时计算（KDJ/MACD/BOLL/RSI等）              │
│        FactorProcessor 在选股/回测时实时计算                │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 数据库表结构

### 3.1 日线行情表 `stock_daily`

存储股票每日交易数据（后复权价）。

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| stock_code | VARCHAR(10) | PK | 股票代码，如 sh600519 |
| trade_date | DATE | PK | 交易日期 |
| open | DECIMAL(10,3) | | 开盘价（后复权） |
| high | DECIMAL(10,3) | | 最高价（后复权） |
| low | DECIMAL(10,3) | | 最低价（后复权） |
| close | DECIMAL(10,3) | | 收盘价（后复权） |
| volume | DECIMAL(15,2) | | 成交量（手） |
| turnover | DECIMAL(15,2) | | 成交额（元） |
| amplitude | DECIMAL(10,4) | | 振幅（%） |
| change_pct | DECIMAL(10,4) | | 涨跌幅（%） |
| is_trading | BOOLEAN | DEFAULT true | 当日是否正常交易 |
| created_at | TIMESTAMP | DEFAULT now() | 记录创建时间 |

**主键**: `(stock_code, trade_date)`
**索引**: `idx_stock_trade_date` on `(stock_code, trade_date)`

### 3.2 股票元数据表 `stock_meta`

存储股票基本信息，用于筛选和过滤。

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| stock_code | VARCHAR(10) | PK | 股票代码 |
| stock_name | VARCHAR(50) | | 股票名称 |
| list_date | DATE | | 上市日期 |
| delist_date | DATE | NULL | 退市日期，NULL表示未退市 |
| industry | VARCHAR(50) | | 所属行业 |
| market | VARCHAR(10) | | 市场（沪/深/北） |
| is_active | BOOLEAN | DEFAULT true | 是否在交易 |
| updated_at | TIMESTAMP | | 最后更新时间 |
| created_at | TIMESTAMP | DEFAULT now() | 记录创建时间 |

### 3.3 财务数据表 `stock_financial`

存储股票财务数据，按财报周期存储。

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| stock_code | VARCHAR(10) | PK, PART | 股票代码 |
| report_period | DATE | PK | 财报周期（如 2024-03-31） |
| publish_date | DATE | | 实际发布日期（用于回测过滤） |
| roe | DECIMAL(10,4) | | 净资产收益率（%） |
| roa | DECIMAL(10,4) | | 资产收益率（%） |
| gross_margin | DECIMAL(10,4) | | 毛利率（%） |
| net_margin | DECIMAL(10,4) | | 净利率（%） |
| debt_ratio | DECIMAL(10,4) | | 资产负债率（%） |
| current_ratio | DECIMAL(10,4) | | 流动比率 |
| quick_ratio | DECIMAL(10,4) | | 速动比率 |
| pe_ttm | DECIMAL(12,4) | | 市盈率TTM |
| pb | DECIMAL(10,4) | | 市净率 |
| ps_tmm | DECIMAL(10,4) | | 市销率TTM |
| created_at | TIMESTAMP | DEFAULT now() | 记录创建时间 |

**主键**: `(stock_code, report_period)`
**索引**: `idx_stock_publish` on `(stock_code, publish_date)`

### 3.4 指数数据表 `index_daily`

存储指数每日行情数据。

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| index_code | VARCHAR(10) | PK | 指数代码，如 sh000001 |
| trade_date | DATE | PK | 交易日期 |
| open | DECIMAL(12,4) | | 开盘点位 |
| high | DECIMAL(12,4) | | 最高点位 |
| low | DECIMAL(12,4) | | 最低点位 |
| close | DECIMAL(12,4) | | 收盘点位 |
| volume | DECIMAL(15,2) | | 成交量 |
| turnover | DECIMAL(15,2) | | 成交额 |
| created_at | TIMESTAMP | DEFAULT now() | 记录创建时间 |

**主键**: `(index_code, trade_date)`

---

## 4. 数据同步策略

### 4.1 历史数据初始化 (`init_history.py`)

一次性脚本，用于初始化 2021 年至今的全量数据。

**执行步骤**:
1. 获取全市场股票列表（akshare 接口）
2. 获取全市场指数列表
3. 批量拉取 2021-01-01 至昨日 的日线数据
4. 批量拉取 历史财务数据（年报+季报）
5. 写入 PostgreSQL（使用 `INSERT ... ON CONFLICT DO UPDATE` 防止重复）

**执行方式**:
```bash
python back_testing/data/init_history.py
```

**预计耗时**: 全量初始化约 2-4 小时（4000+ 股票 × 1200+ 交易日）

### 4.2 每日增量更新 (`daily_update.py`)

定时任务，每日两次执行。

**14:30 盘中更新**（用于风险监控）:
- 获取持仓股票的当日数据
- 写入/更新 `stock_daily`

**15:30 收盘后更新**（用于次日决策）:
- 获取全市场当日行情数据
- 更新 `stock_meta` 中的停牌/上市状态
- 检查财务数据更新（季度财报发布期增量同步）

**执行方式**:
```bash
# 14:30 盘中更新
python back_testing/data/daily_update.py --mode intraday

# 15:30 收盘后更新
python back_testing/data/daily_update.py --mode close
```

**Windows 定时任务配置**:
```
schtasks /create /tn "QuantDataUpdate-Intraday" /tr "python back_testing/data/daily_update.py --mode intraday" /sc daily /st 14:30
schtasks /create /tn "QuantDataUpdate-Close" /tr "python back_testing/data/daily_update.py --mode close" /sc daily /st 15:30
```

---

## 5. 数据访问层适配

### 5.1 新的 DataProvider 接口

```python
class DataProvider:
    """
    统一数据访问层

    支持两种数据源：
    - PostgreSQL（生产环境）
    - 本地文件（回退/历史兼容）
    """

    def __init__(self, use_db: bool = True):
        self.use_db = use_db
        if use_db:
            self.engine = create_engine(DB_URL)

    def get_stock_data(
        self,
        stock_code: str,
        date: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """获取股票历史数据"""
        if self.use_db:
            return self._get_from_db(stock_code, date, start_date, end_date)
        else:
            return self._get_from_file(stock_code, date, start_date, end_date)

    def get_stock_price(self, stock_code: str, date: str) -> Optional[float]:
        """获取指定日期收盘价（后复权）"""

    def get_all_stock_codes(self) -> list:
        """获取所有股票代码（排除北交所、退市股）"""

    def get_latest_trade_date(self, stock_code: str) -> Optional[date]:
        """获取最近交易日"""
```

### 5.2 回测兼容性

现有回测代码无需修改，DataProvider 默认使用数据库：
- `get_stock_data()` → 数据库查询
- `get_stock_price()` → 数据库查询
- `get_all_stock_codes()` → 数据库查询

如需回退到本地文件：
```python
provider = DataProvider(use_db=False)  # 使用原有 Parquet/CSV
```

---

## 6. 配置管理

数据库配置通过环境变量或配置文件管理：

```ini
# config/database.ini
[postgresql]
host = localhost
port = 5432
database = quant_db
user = quant_user
password = your_password

[akshare]
rate_limit = 10  # 每秒请求数限制
retry_times = 3
```

---

## 7. 目录结构

```
back_testing/
├── data/
│   ├── __init__.py
│   ├── data_provider.py      # 统一数据访问层（改造）
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py     # 数据库连接管理
│   │   ├── models.py         # SQLAlchemy 模型定义
│   │   └── migrations/       # 数据库迁移脚本
│   ├── sync/
│   │   ├── __init__.py
│   │   ├── init_history.py  # 历史数据初始化
│   │   ├── daily_update.py   # 每日增量更新
│   │   └── akshare_client.py # akshare 封装
│   └── scripts/
│       └── setup_database.sql # 建表脚本
├── config/
│   └── database.ini          # 数据库配置
└── ...
```

---

## 8. 实施计划（待后续细化）

### Phase 1: 数据库搭建
1. Windows 安装 PostgreSQL
2. 创建数据库和用户
3. 执行建表脚本

### Phase 2: 数据同步层
4. 实现 akshare_client 封装
5. 实现 init_history.py
6. 实现 daily_update.py

### Phase 3: 数据访问层
7. 改造 DataProvider 支持数据库
8. 保留文件回退能力

### Phase 4: 定时任务
9. 配置 Windows 定时任务
10. 日志和监控

---

## 9. 已知约束与后续决策点

| 约束 | 说明 | 决策 |
|------|------|------|------|
| akshare 限流 | 免费接口有请求频率限制 | 限速保护 + 重试机制 |
| 财报发布日期不统一 | 需按个股追踪发布日期 | 首次全量后按季增量 |
| 停牌期间数据 | 停牌日 `is_trading=false` | 已考虑 |
| 北交所数据 | 初期排除在外 | 未来可扩展 |
| 技术指标存储 | KDJ/MACD/BOLL 等 | **实时计算，不预存储** |

---

## 10. 文档版本

| 版本 | 日期 | 说明 |
|------|------|------|
| 0.1 | 2026-04-21 | 初始版本，待用户确认 |
