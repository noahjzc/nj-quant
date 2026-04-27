# PostgreSQL 数据层实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:
> executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将项目数据层从本地 Parquet/CSV 文件迁移至 PostgreSQL 数据库，实现全市场A股数据存储和每日增量更新。

**Architecture:**

- 数据同步层：通过 akshare API 获取数据，写入 PostgreSQL
- 数据访问层：改造现有 DataProvider，支持数据库和文件双模式
- 初始化方式：历史数据一次性初始化 + 每日增量更新

**Tech Stack:** PostgreSQL, SQLAlchemy, akshare, pandas

---

## 阶段一：数据库基础设施

### Task 1: 安装依赖和配置

**Files:**

- Create: `back_testing/data/db/__init__.py`
- Create: `back_testing/data/db/connection.py`
- Create: `back_testing/data/db/models.py`
- Create: `back_testing/config/database.ini`
- Modify: `requirements.txt`

- [ ] **Step 1: 添加 SQLAlchemy 到依赖**

```bash
pip install sqlalchemy psycopg2-binary
```

- [ ] **Step 2: 创建数据库模块目录结构**

```bash
mkdir -p back_testing/data/db
mkdir -p back_testing/data/sync
mkdir -p back_testing/data/scripts
touch back_testing/data/db/__init__.py
touch back_testing/data/sync/__init__.py
```

- [ ] **Step 3: 创建数据库配置文件**

`back_testing/config/database.ini`:

```ini
[postgresql]
host = localhost
port = 5432
database = quant_db
user = quant_user
password = 123456

[akshare]
rate_limit = 10
retry_times = 3
timeout = 30
```

- [ ] **Step 4: 创建数据库连接管理模块**

`back_testing/data/db/connection.py`:

```python
"""数据库连接管理"""
import os
from configparser import ConfigParser
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

_config = None
_engine = None
_session_factory = None


def get_db_config() -> dict:
    """读取数据库配置"""
    global _config
    if _config is None:
        config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'database.ini'
        parser = ConfigParser()
        parser.read(config_path)
        _config = dict(parser.items('postgresql'))
    return _config


def get_engine():
    """获取数据库引擎（单例）"""
    global _engine
    if _engine is None:
        config = get_db_config()
        db_url = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
        _engine = create_engine(db_url, pool_size=5, max_overflow=10)
    return _engine


def get_session():
    """获取数据库会话"""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = sessionmaker(bind=engine)
    return scoped_session(_session_factory)


def close_session():
    """关闭会话"""
    global _session_factory
    if _session_factory is not None:
        _session_factory = None
```

- [ ] **Step 5: 创建 SQLAlchemy 模型**

`back_testing/data/db/models.py`:

```python
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
```

- [ ] **Step 6: 创建数据库初始化脚本**

`back_testing/data/scripts/setup_database.sql`:

```sql
-- 创建 stock_daily 表
CREATE TABLE IF NOT EXISTS stock_daily
(
    stock_code
    VARCHAR
(
    10
) NOT NULL,
    trade_date DATE NOT NULL,
    open DECIMAL
(
    10,
    3
),
    high DECIMAL
(
    10,
    3
),
    low DECIMAL
(
    10,
    3
),
    close DECIMAL
(
    10,
    3
),
    volume DECIMAL
(
    15,
    2
),
    turnover DECIMAL
(
    15,
    2
),
    amplitude DECIMAL
(
    10,
    4
),
    change_pct DECIMAL
(
    10,
    4
),
    is_trading BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY
(
    stock_code,
    trade_date
)
    );

CREATE INDEX IF NOT EXISTS idx_stock_trade_date ON stock_daily(stock_code, trade_date);

-- 创建 stock_meta 表
CREATE TABLE IF NOT EXISTS stock_meta
(
    stock_code
    VARCHAR
(
    10
) PRIMARY KEY,
    stock_name VARCHAR
(
    50
),
    list_date DATE,
    delist_date DATE,
    industry VARCHAR
(
    50
),
    market VARCHAR
(
    10
),
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

-- 创建 stock_financial 表
CREATE TABLE IF NOT EXISTS stock_financial
(
    stock_code
    VARCHAR
(
    10
) NOT NULL,
    report_period DATE NOT NULL,
    publish_date DATE,
    roe DECIMAL
(
    10,
    4
),
    roa DECIMAL
(
    10,
    4
),
    gross_margin DECIMAL
(
    10,
    4
),
    net_margin DECIMAL
(
    10,
    4
),
    debt_ratio DECIMAL
(
    10,
    4
),
    current_ratio DECIMAL
(
    10,
    4
),
    quick_ratio DECIMAL
(
    10,
    4
),
    pe_ttm DECIMAL
(
    12,
    4
),
    pb DECIMAL
(
    10,
    4
),
    ps_ttm DECIMAL
(
    10,
    4
),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY
(
    stock_code,
    report_period
)
    );

CREATE INDEX IF NOT EXISTS idx_stock_publish ON stock_financial(stock_code, publish_date);

-- 创建 index_daily 表
CREATE TABLE IF NOT EXISTS index_daily
(
    index_code
    VARCHAR
(
    10
) NOT NULL,
    trade_date DATE NOT NULL,
    open DECIMAL
(
    12,
    4
),
    high DECIMAL
(
    12,
    4
),
    low DECIMAL
(
    12,
    4
),
    close DECIMAL
(
    12,
    4
),
    volume DECIMAL
(
    15,
    2
),
    turnover DECIMAL
(
    15,
    2
),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY
(
    index_code,
    trade_date
)
    );

CREATE INDEX IF NOT EXISTS idx_index_trade_date ON index_daily(index_code, trade_date);
```

- [ ] **Step 7: 执行建表脚本**

```powershell
psql -U quant_user -d quant_db -h localhost -f back_testing/data/scripts/setup_database.sql
```

- [ ] **Step 8: 提交代码**

```bash
git add back_testing/data/db back_testing/config/database.ini requirements.txt
git commit -m "feat(data): add PostgreSQL database models and connection management"
```

---

### Task 2: akshare 客户端封装

**Files:**

- Create: `back_testing/data/sync/akshare_client.py`
- Create: `tests/back_testing/data/test_akshare_client.py`

- [ ] **Step 1: 创建 akshare_client.py**

`back_testing/data/sync/akshare_client.py`:

```python
"""akshare API 封装层"""
import time
import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)


class AkshareClient:
    """akshare API 封装，提供限流和错误处理"""

    def __init__(self, rate_limit: int = 10, retry_times: int = 3, timeout: int = 30):
        """
        Args:
            rate_limit: 每秒请求数限制
            retry_times: 重试次数
            timeout: 超时秒数
        """
        self.rate_limit = rate_limit
        self.retry_times = retry_times
        self.timeout = timeout
        self._min_interval = 1.0 / rate_limit
        self._last_request_time = 0.0

    def _rate_limit_wait(self):
        """等待直到可以发送请求"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _retry_request(self, func, *args, **kwargs):
        """带重试的请求"""
        last_error = None
        for attempt in range(self.retry_times):
            try:
                self._rate_limit_wait()
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"请求失败（第 {attempt + 1}/{self.retry_times} 次）: {e}")
                if attempt < self.retry_times - 1:
                    time.sleep(2 ** attempt)  # 指数退避
        raise last_error

    def get_stock_list(self) -> pd.DataFrame:
        """获取全市场A股列表"""
        df = self._retry_request(ak.stock_zh_a_spot_em)
        # 过滤只需要的基本信息
        result = df[['代码', '名称', '板块', '市值', '上市时间']].copy()
        result.columns = ['stock_code', 'stock_name', 'industry', 'market_cap', 'list_date']
        # 添加市场前缀
        result['stock_code'] = result['stock_code'].apply(
            lambda x: f'sh{x}' if str(x).startswith('6') else f'sz{x}'
        )
        return result

    def get_stock_daily(
            self,
            stock_code: str,
            start_date: str,
            end_date: str,
            adjust: str = "qfq"
    ) -> pd.DataFrame:
        """
        获取股票日线数据（后复权）

        Args:
            stock_code: 股票代码，如 sh600519
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD
            adjust: 复权类型，qfq=后复权，hfq=前复权，None=不复权

        Returns:
            DataFrame with columns: stock_code, trade_date, open, high, low, close, volume, turnover, amplitude, change_pct
        """
        # akshare 需要不带前缀的代码
        plain_code = stock_code[2:] if stock_code.startswith(('sh', 'sz')) else stock_code

        df = self._retry_request(
            ak.stock_zh_a_hist,
            symbol=plain_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # 重命名列
        df = df.rename(columns={
            '日期': 'trade_date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'turnover',
            '振幅': 'amplitude',
            '涨跌幅': 'change_pct'
        })

        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
        df['stock_code'] = stock_code
        df['is_trading'] = True

        return df[['stock_code', 'trade_date', 'open', 'high', 'low', 'close',
                   'volume', 'turnover', 'amplitude', 'change_pct', 'is_trading']]

    def get_index_daily(
            self,
            index_code: str,
            start_date: str,
            end_date: str
    ) -> pd.DataFrame:
        """
        获取指数日线数据

        Args:
            index_code: 指数代码，如 sh000001
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD
        """
        df = self._retry_request(
            ak.index_zh_a_hist,
            symbol=index_code[2:],
            period="daily",
            start_date=start_date,
            end_date=end_date
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            '日期': 'trade_date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'turnover'
        })

        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
        df['index_code'] = index_code

        return df[['index_code', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'turnover']]

    def get_stock_financial(
            self,
            stock_code: str,
            start_year: int = None,
            end_year: int = None
    ) -> pd.DataFrame:
        """
        获取股票财务数据

        Args:
            stock_code: 股票代码
            start_year: 起始年份
            end_year: 结束年份
        """
        plain_code = stock_code[2:] if stock_code.startswith(('sh', 'sz')) else stock_code

        try:
            df = self._retry_request(
                ak.stock_financial_analysis_indicator,
                symbol=plain_code
            )
        except Exception as e:
            logger.warning(f"获取财务数据失败 {stock_code}: {e}")
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # 只保留需要的列
        cols = ['股票代码', '报告日期', '净资产收益率', '资产报酬率', '销售毛利率',
                '销售净利率', '资产负债率', '流动比率', '速动比率',
                '市盈率(TTM)', '市净率', '市销率(TTM)']
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols]

        df = df.rename(columns={
            '股票代码': 'stock_code',
            '报告日期': 'report_period',
            '净资产收益率': 'roe',
            '资产报酬率': 'roa',
            '销售毛利率': 'gross_margin',
            '销售净利率': 'net_margin',
            '资产负债率': 'debt_ratio',
            '流动比率': 'current_ratio',
            '速动比率': 'quick_ratio',
            '市盈率(TTM)': 'pe_ttm',
            '市净率': 'pb',
            '市销率(TTM)': 'ps_ttm'
        })

        df['stock_code'] = stock_code
        df['report_period'] = pd.to_datetime(df['report_period']).dt.date

        # 尝试获取发布日期（使用报告期后推2个月作为预估）
        df['publish_date'] = df['report_period'].apply(
            lambda x: date(x.year + (x.month // 12), (x.month % 12) + 1, 1) if x.month < 12
            else date(x.year + 1, 1, 1)
        )

        return df
```

- [ ] **Step 2: 创建测试文件**

`tests/back_testing/data/test_akshare_client.py`:

```python
"""akshare_client 测试"""
import pytest
from datetime import date, timedelta
from back_testing.data.sync.akshare_client import AkshareClient


class TestAkshareClient:
    """测试 AkshareClient"""

    @pytest.fixture
    def client(self):
        return AkshareClient(rate_limit=1)

    def test_init(self):
        """测试初始化"""
        client = AkshareClient()
        assert client.rate_limit == 10
        assert client.retry_times == 3

    def test_rate_limit_wait(self, client):
        """测试限速等待"""
        import time
        start = time.time()
        client._rate_limit_wait()
        client._rate_limit_wait()
        elapsed = time.time() - start
        assert elapsed >= 0.09  # 至少等待 0.1 秒

    def test_get_stock_list(self, client):
        """测试获取股票列表"""
        df = client.get_stock_list()
        assert not df.empty
        assert 'stock_code' in df.columns
        assert 'stock_name' in df.columns
        # 验证格式
        for code in df['stock_code'].head():
            assert code.startswith('sh') or code.startswith('sz')
```

- [ ] **Step 3: 运行测试验证**

```powershell
cd D:\workspace\code\mine\quant\nj-quant
pytest tests/back_testing/data/test_akshare_client.py -v
```

- [ ] **Step 4: 提交代码**

```bash
git add back_testing/data/sync/akshare_client.py tests/back_testing/data/test_akshare_client.py
git commit -m "feat(data): add akshare client wrapper with rate limiting"
```

---

### Task 3: 历史数据初始化脚本

**Files:**

- Create: `back_testing/data/sync/init_history.py`
- Create: `tests/back_testing/data/test_init_history.py`

- [ ] **Step 1: 创建 init_history.py**

`back_testing/data/sync/init_history.py`:

```python
"""
历史数据初始化脚本

用法:
    python back_testing/data/sync/init_history.py --start 20210101 --end 20260420
"""
import argparse
import logging
import sys
from datetime import datetime

# 添加项目根目录到 path
sys.path.insert(0, str(__file__).rsplit('back_testing', 1)[0])

from back_testing.data.db.connection import get_engine, get_session
from back_testing.data.db.models import StockDaily, StockMeta, StockFinancial, IndexDaily
from back_testing.data.sync.akshare_client import AkshareClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoryInitializer:
    """历史数据初始化器"""

    def __init__(self, client: AkshareClient):
        self.client = client
        self.engine = get_engine()
        self.Session = get_session()

    def init_stock_meta(self) -> int:
        """初始化股票元数据"""
        logger.info("开始获取股票列表...")
        df = self.client.get_stock_list()

        if df.empty:
            logger.error("获取股票列表失败")
            return 0

        session = self.Session()
        count = 0

        try:
            for _, row in df.iterrows():
                stock_code = row['stock_code']

                # 判断市场
                if stock_code.startswith('sh'):
                    market = '沪'
                elif stock_code.startswith('sz'):
                    market = '深'
                else:
                    market = '北'

                meta = StockMeta(
                    stock_code=stock_code,
                    stock_name=row.get('stock_name'),
                    industry=row.get('industry'),
                    market=market,
                    is_active=True
                )
                session.merge(meta)  # 使用 merge 避免主键冲突
                count += 1

            session.commit()
            logger.info(f"股票元数据初始化完成，共 {count} 只")
        except Exception as e:
            session.rollback()
            logger.error(f"股票元数据初始化失败: {e}")
        finally:
            session.close()

        return count

    def init_stock_daily(self, stock_codes: list, start_date: str, end_date: str) -> int:
        """初始化股票日线数据"""
        total = len(stock_codes)
        success_count = 0
        fail_count = 0

        session = self.Session()

        for i, code in enumerate(stock_codes):
            if (i + 1) % 50 == 0:
                logger.info(f"进度: {i + 1}/{total}")

            try:
                df = self.client.get_stock_daily(code, start_date, end_date)

                if df.empty:
                    fail_count += 1
                    continue

                for _, row in df.iterrows():
                    daily = StockDaily(
                        stock_code=row['stock_code'],
                        trade_date=row['trade_date'],
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        turnover=row['turnover'],
                        amplitude=row['amplitude'],
                        change_pct=row['change_pct'],
                        is_trading=row.get('is_trading', True)
                    )
                    session.merge(daily)

                session.commit()
                success_count += 1

            except Exception as e:
                session.rollback()
                logger.warning(f"获取 {code} 日线数据失败: {e}")
                fail_count += 1

        session.close()
        logger.info(f"日线数据初始化完成: 成功 {success_count}, 失败 {fail_count}")
        return success_count

    def init_index_daily(self, index_codes: list, start_date: str, end_date: str) -> int:
        """初始化指数日线数据"""
        session = self.Session()
        success_count = 0

        for code in index_codes:
            try:
                df = self.client.get_index_daily(code, start_date, end_date)

                if df.empty:
                    continue

                for _, row in df.iterrows():
                    index_daily = IndexDaily(
                        index_code=row['index_code'],
                        trade_date=row['trade_date'],
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        turnover=row['turnover']
                    )
                    session.merge(index_daily)

                session.commit()
                success_count += 1

            except Exception as e:
                session.rollback()
                logger.warning(f"获取 {code} 指数数据失败: {e}")

        session.close()
        logger.info(f"指数数据初始化完成: {success_count}/{len(index_codes)}")
        return success_count

    def init_financial(self, stock_codes: list, years: list = None) -> int:
        """初始化财务数据"""
        if years is None:
            years = [2021, 2022, 2023, 2024, 2025]

        total = len(stock_codes)
        success_count = 0

        session = self.Session()

        for i, code in enumerate(stock_codes):
            if (i + 1) % 100 == 0:
                logger.info(f"财务数据进度: {i + 1}/{total}")

            try:
                df = self.client.get_stock_financial(code)

                if df.empty:
                    continue

                # 过滤指定年份
                if years:
                    df = df[df['report_period'].apply(lambda x: x.year in years)]

                for _, row in df.iterrows():
                    financial = StockFinancial(
                        stock_code=code,
                        report_period=row['report_period'],
                        publish_date=row.get('publish_date'),
                        roe=row.get('roe'),
                        roa=row.get('roa'),
                        gross_margin=row.get('gross_margin'),
                        net_margin=row.get('net_margin'),
                        debt_ratio=row.get('debt_ratio'),
                        current_ratio=row.get('current_ratio'),
                        quick_ratio=row.get('quick_ratio'),
                        pe_ttm=row.get('pe_ttm'),
                        pb=row.get('pb'),
                        ps_ttm=row.get('ps_ttm')
                    )
                    session.merge(financial)

                session.commit()
                success_count += 1

            except Exception as e:
                session.rollback()
                logger.warning(f"获取 {code} 财务数据失败: {e}")

        session.close()
        logger.info(f"财务数据初始化完成: {success_count}/{total}")
        return success_count

    def run(self, start_date: str, end_date: str):
        """执行完整初始化"""
        logger.info(f"=" * 50)
        logger.info(f"历史数据初始化开始")
        logger.info(f"日期范围: {start_date} ~ {end_date}")
        logger.info(f"=" * 50)

        # 1. 初始化股票元数据
        count = self.init_stock_meta()

        # 2. 获取股票列表
        session = self.Session()
        stock_codes = [r[0] for r in session.query(StockMeta.stock_code).filter(
            StockMeta.is_active == True
        ).all()]
        session.close()

        # 3. 初始化日线数据
        logger.info("开始初始化日线数据（这可能需要较长时间）...")
        self.init_stock_daily(stock_codes, start_date, end_date)

        # 4. 初始化指数数据
        logger.info("开始初始化指数数据...")
        index_codes = ['sh000001', 'sh000300', 'sz399001', 'sz399006']  # 沪深300, 上证, 深证, 创业板
        self.init_index_daily(index_codes, start_date, end_date)

        # 5. 初始化财务数据
        logger.info("开始初始化财务数据...")
        self.init_financial(stock_codes)

        logger.info(f"=" * 50)
        logger.info(f"历史数据初始化完成")
        logger.info(f"=" * 50)


def main():
    parser = argparse.ArgumentParser(description='历史数据初始化')
    parser.add_argument('--start', default='20210101', help='开始日期 YYYYMMDD')
    parser.add_argument('--end', default=None, help='结束日期 YYYYMMDD')
    args = parser.parse_args()

    end_date = args.end or datetime.now().strftime('%Y%m%d')

    client = AkshareClient(rate_limit=5)  # 降低速率避免限流
    initializer = HistoryInitializer(client)
    initializer.run(args.start, end_date)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 创建测试**

`tests/back_testing/data/test_init_history.py`:

```python
"""历史初始化脚本测试"""
import pytest
from unittest.mock import Mock, patch
from back_testing.data.sync.init_history import HistoryInitializer


class TestHistoryInitializer:
    """测试 HistoryInitializer"""

    @pytest.fixture
    def mock_initializer(self):
        with patch('back_testing.data.sync.init_history.get_engine'),
                patch('back_testing.data.sync.init_history.get_session'):
            client = Mock()
            initializer = HistoryInitializer(client)
            return initializer

    def test_init_stock_meta_empty(self, mock_initializer):
        """测试空股票列表处理"""
        with patch.object(mock_initializer.client, 'get_stock_list', return_value=None):
            result = mock_initializer.init_stock_meta()
            assert result == 0
```

- [ ] **Step 3: 提交代码**

```bash
git add back_testing/data/sync/init_history.py tests/back_testing/data/test_init_history.py
git commit -m "feat(data): add history initialization script"
```

---

### Task 4: 每日增量更新脚本

**Files:**

- Create: `back_testing/data/sync/daily_update.py`

- [ ] **Step 1: 创建 daily_update.py**

`back_testing/data/sync/daily_update.py`:

```python
"""
每日增量更新脚本

用法:
    # 盘中更新（持仓股票）
    python back_testing/data/sync/daily_update.py --mode intraday --portfolio sh600519,sh600036

    # 收盘后更新（全市场）
    python back_testing/data/sync/daily_update.py --mode close
"""
import argparse
import logging
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, str(__file__).rsplit('back_testing', 1)[0])

from back_testing.data.db.connection import get_session
from back_testing.data.db.models import StockDaily, StockMeta, IndexDaily
from back_testing.data.sync.akshare_client import AkshareClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DailyUpdater:
    """每日数据更新器"""

    def __init__(self, client: AkshareClient):
        self.client = client
        self.Session = get_session()

    def get_last_trade_date(self, stock_code: str) -> date:
        """获取某股票最近交易日期"""
        session = self.Session()
        try:
            result = session.query(StockDaily.trade_date).filter(
                StockDaily.stock_code == stock_code
            ).order_by(StockDaily.trade_date.desc()).first()
            return result[0] if result else None
        finally:
            session.close()

    def get_all_active_stocks(self) -> list:
        """获取所有活跃股票"""
        session = self.Session()
        try:
            codes = [r[0] for r in session.query(StockMeta.stock_code).filter(
                StockMeta.is_active == True
            ).all()]
            return codes
        finally:
            session.close()

    def update_stock_daily(self, stock_code: str, start_date: date) -> bool:
        """更新单只股票日线数据"""
        start_str = start_date.strftime('%Y%m%d') if start_date else '20210101'
        today_str = datetime.now().strftime('%Y%m%d')

        try:
            df = self.client.get_stock_daily(stock_code, start_str, today_str)

            if df.empty:
                return False

            session = self.Session()
            try:
                for _, row in df.iterrows():
                    if row['trade_date'] <= start_date:
                        continue  # 跳过已有数据

                    daily = StockDaily(
                        stock_code=row['stock_code'],
                        trade_date=row['trade_date'],
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        turnover=row['turnover'],
                        amplitude=row['amplitude'],
                        change_pct=row['change_pct'],
                        is_trading=row.get('is_trading', True)
                    )
                    session.merge(daily)
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.warning(f"更新 {stock_code} 失败: {e}")
                return False
            finally:
                session.close()

        except Exception as e:
            logger.warning(f"获取 {stock_code} 数据失败: {e}")
            return False

    def update_portfolio(self, portfolio: list) -> dict:
        """盘中更新持仓股票"""
        logger.info(f"盘中更新: {len(portfolio)} 只股票")
        results = {'success': 0, 'fail': 0}

        for code in portfolio:
            last_date = self.get_last_trade_date(code)
            if last_date is None:
                last_date = date.today() - timedelta(days=30)

            if self.update_stock_daily(code, last_date):
                results['success'] += 1
            else:
                results['fail'] += 1

        logger.info(f"盘中更新完成: 成功 {results['success']}, 失败 {results['fail']}")
        return results

    def update_full_market(self) -> dict:
        """收盘后全市场更新"""
        logger.info("开始全市场收盘后更新...")

        # 获取所有活跃股票
        stocks = self.get_all_active_stocks()
        logger.info(f"全市场股票数量: {len(stocks)}")

        # 计算起始日期（获取最近30天数据，确保不遗漏）
        start_date = date.today() - timedelta(days=30)

        results = {'success': 0, 'fail': 0}

        for i, code in enumerate(stocks):
            if (i + 1) % 100 == 0:
                logger.info(f"进度: {i + 1}/{len(stocks)}")

            if self.update_stock_daily(code, start_date):
                results['success'] += 1
            else:
                results['fail'] += 1

        # 更新指数数据
        index_codes = ['sh000001', 'sh000300', 'sz399001', 'sz399006']
        self.update_index_daily(index_codes, start_date)

        logger.info(f"全市场更新完成: 成功 {results['success']}, 失败 {results['fail']}")
        return results

    def update_index_daily(self, index_codes: list, start_date: date) -> bool:
        """更新指数数据"""
        start_str = start_date.strftime('%Y%m%d')
        today_str = datetime.now().strftime('%Y%m%d')

        session = self.Session()
        try:
            for code in index_codes:
                try:
                    df = self.client.get_index_daily(code, start_str, today_str)

                    if df.empty:
                        continue

                    for _, row in df.iterrows():
                        index_daily = IndexDaily(
                            index_code=row['index_code'],
                            trade_date=row['trade_date'],
                            open=row['open'],
                            high=row['high'],
                            low=row['low'],
                            close=row['close'],
                            volume=row['volume'],
                            turnover=row['turnover']
                        )
                        session.merge(index_daily)

                    session.commit()
                    logger.info(f"指数 {code} 更新完成")

                except Exception as e:
                    session.rollback()
                    logger.warning(f"指数 {code} 更新失败: {e}")

            return True

        finally:
            session.close()

    def run_intraday(self, portfolio: list):
        """运行盘中更新"""
        logger.info(f"=" * 50)
        logger.info("盘中数据更新开始")
        logger.info(f"时间: {datetime.now()}")
        logger.info(f"=" * 50)

        self.update_portfolio(portfolio)

        logger.info(f"=" * 50)
        logger.info("盘中数据更新完成")
        logger.info(f"=" * 50)

    def run_close(self):
        """运行收盘后更新"""
        logger.info(f"=" * 50)
        logger.info("收盘后数据更新开始")
        logger.info(f"时间: {datetime.now()}")
        logger.info(f"=" * 50)

        self.update_full_market()

        logger.info(f"=" * 50)
        logger.info("收盘后数据更新完成")
        logger.info(f"=" * 50)


def main():
    parser = argparse.ArgumentParser(description='每日数据更新')
    parser.add_argument('--mode', choices=['intraday', 'close'], required=True,
                        help='更新模式: intraday=盘中, close=收盘后')
    parser.add_argument('--portfolio', type=str, default=None,
                        help='持仓股票列表，逗号分隔，如 sh600519,sh600036')
    args = parser.parse_args()

    client = AkshareClient(rate_limit=10)
    updater = DailyUpdater(client)

    if args.mode == 'intraday':
        portfolio = args.portfolio.split(',') if args.portfolio else []
        if not portfolio:
            logger.error("盘中模式需要指定 --portfolio 参数")
            sys.exit(1)
        updater.run_intraday(portfolio)
    else:
        updater.run_close()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 提交代码**

```bash
git add back_testing/data/sync/daily_update.py
git commit -m "feat(data): add daily update script"
```

---

### Task 5: 改造 DataProvider 支持数据库

**Files:**

- Modify: `back_testing/data/data_provider.py`

- [ ] **Step 1: 阅读现有 DataProvider 实现**

确保理解现有接口：

```python
def get_stock_data(self, stock_code: str, date=None, start_date=None, end_date=None) -> pd.DataFrame


    def get_stock_price(self, stock_code: str, date) -> Optional[float]


    def get_all_stock_codes(self) -> list


    def get_latest_price(self, stock_code: str) -> Optional[float]
```

- [ ] **Step 2: 改造 DataProvider**

`back_testing/data/data_provider.py`:

```python
"""
数据提供器 - 统一管理股票数据读取

支持两种数据源：
- PostgreSQL（生产环境，默认）
- 本地 Parquet/CSV（回退/历史兼容）
"""
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from back_testing.data.db.connection import get_engine, get_session
from back_testing.data.db.models import StockDaily, IndexDaily, StockMeta


class DataProvider:
    """
    统一数据访问层
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
            df['trade_date'] = pd.to_datetime(df['交易日期'])
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
            return df.index[-1]


# 全局默认实例
_default_provider: Optional[DataProvider] = None


def get_provider() -> DataProvider:
    """获取默认数据提供器（单例）"""
    global _default_provider
    if _default_provider is None:
        _default_provider = DataProvider(use_db=True)
    return _default_provider
```

- [ ] **Step 3: 运行测试验证**

```powershell
pytest tests/back_testing/test_factor_loader.py tests/back_testing/test_factor_utils.py -v
```

- [ ] **Step 4: 提交代码**

```bash
git add back_testing/data/data_provider.py
git commit -m "feat(data): support PostgreSQL in DataProvider"
```

---

## 阶段二：定时任务配置

### Task 6: Windows 定时任务配置

**Files:**

- Create: `back_testing/data/scripts/setup_scheduled_tasks.bat`

- [ ] **Step 1: 创建定时任务配置脚本**

`back_testing/data/scripts/setup_scheduled_tasks.bat`:

```batch
@echo off
REM Windows 定时任务配置脚本
REM 用管理员权限运行

echo 正在创建定时任务...

REM 14:30 盘中更新任务
schtasks /create /tn "Quant-Intraday-Update" ^
    /tr "python D:\workspace\code\mine\quant\nj-quant\back_testing\data\sync\daily_update.py --mode intraday --portfolio sh600519,sh600036" ^
    /sc daily /st 14:30 ^
    /f

REM 15:30 收盘后更新任务
schtasks /create /tn "Quant-Close-Update" ^
    /tr "python D:\workspace\code\mine\quant\nj-quant\back_testing\data\sync\daily_update.py --mode close" ^
    /sc daily /st 15:30 ^
    /f

echo 定时任务创建完成
schtasks /query /tn "Quant-Intraday-Update"
schtasks /query /tn "Quant-Close-Update"

pause
```

- [ ] **Step 2: 提交代码**

```bash
git add back_testing/data/scripts/setup_scheduled_tasks.bat
git commit -m "feat(data): add Windows scheduled task configuration"
```

---

## 实施检查清单

| 步骤 | 任务                      | 状态        |
|----|-------------------------|-----------|
| 1  | PostgreSQL 安装与配置        | ✅ 已完成（用户） |
| 2  | 创建数据库和用户                | ✅ 已完成（用户） |
| 3  | Task 1: 数据库模型和连接        | ⬜         |
| 4  | Task 2: akshare 客户端     | ⬜         |
| 5  | Task 3: 历史数据初始化         | ⬜         |
| 6  | Task 4: 每日增量更新          | ⬜         |
| 7  | Task 5: DataProvider 改造 | ⬜         |
| 8  | Task 6: 定时任务配置          | ⬜         |
| 9  | 集成测试                    | ⬜         |

---

## 执行后验证

完成所有任务后，执行以下验证：

```powershell
# 1. 验证数据库连接
psql -U quant_user -d quant_db -h localhost -c "\dt"

# 2. 验证 akshare 可用
python -c "import akshare as ak; print(ak.__version__)"

# 3. 初始化少量历史数据测试
python back_testing/data/sync/init_history.py --start 20260101 --end 20260110

# 4. 验证 DataProvider 从数据库读取
python -c "from back_testing.data.data_provider import DataProvider; p = DataProvider(use_db=True); print(p.get_all_stock_codes()[:5])"
```
