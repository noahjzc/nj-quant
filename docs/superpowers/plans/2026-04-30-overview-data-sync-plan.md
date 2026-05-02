# 预测者网数据同步 实施计划

**Goal:** 实现两个程序：`sync_overview.py`（统一入口，支持每日定时和一次性历史补全）和 `overview_client.py`（API 客户端）

**Architecture:**
- `overview_client.py`：封装预测者网 API 下载和解压逻辑
- `sync_overview.py`：统一入口，`--mode daily` 下载 ZIP 同步，`--mode backfill` 本地 CSV 目录补全
- 核心 upsert 逻辑复用，两种模式共用

**Tech Stack:** Python, pandas, sqlalchemy, zipfile, requests/urllib

---

## 文件结构

```
back_testing/data/sync/
├── overview_client.py    # 新增：API 客户端（下载+解压）
├── sync_overview.py     # 新增：统一入口
└── backfill_overview.py # 已有（废弃，由 sync_overview.py 替代）
```

---

### Task 1: 创建 `overview_client.py`

**Files:**
- Create: `back_testing/data/sync/overview_client.py`

- [ ] **Step 1: 创建 overview_client.py 框架**

```python
"""预测者网 API 客户端

Usage:
    from back_testing.data.sync.overview_client import OverviewClient
    client = OverviewClient(email="...", api_key="...")
    zip_path = client.download_today("overview-data-push")  # 阻塞等待
    stock_df, index_df = client.parse_zip(zip_path)
"""
import logging
import os
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger("overview_client")

API_URL = "https://yucezhe.com/api/v1/data/today"


class OverviewClient:
    def __init__(self, email: str, api_key: str, product: str = "overview-data-push"):
        self.email = email
        self.api_key = api_key
        self.product = product

    def _call_api(self) -> str | None:
        """调用预测者网 API，返回下载 URL 或 None。"""
        try:
            from urllib.parse import urlencode
            from urllib.request import urlopen
        except ImportError:
            from urllib import urlencode, urlopen

        params = urlencode({"name": self.product, "email": self.email, "key": self.api_key})
        response = urlopen(f"{API_URL}?{params}", timeout=30)
        return response.read().decode("utf-8").strip()

    def download_today(self, max_retries: int = 10, retry_interval: int = 30) -> str:
        """下载当日数据包，阻塞等待数据就绪。

        Returns:
            临时 ZIP 文件路径（用完后自动删除）

        Raises:
            RuntimeError: 轮询超时
        """
        for attempt in range(max_retries):
            url = self._call_api()
            if url and "data.yucezhe.com" in url:
                logger.info(f"下载数据: {url}")
                tmp_path = os.path.join(tempfile.gettempdir(), f"overview_push_{int(time.time())}.zip")
                from urllib.request import urlretrieve
                urlretrieve(url, tmp_path)
                logger.info(f"ZIP 保存至: {tmp_path}")
                return tmp_path
            else:
                msg = url or "(空响应)"
                if attempt < max_retries - 1:
                    logger.warning(f"数据未就绪（第 {attempt+1} 次）: {msg}，{retry_interval}s 后重试")
                    time.sleep(retry_interval)
                else:
                    raise RuntimeError(f"数据获取超时: {msg}")
        raise RuntimeError("不可能到达")

    @staticmethod
    def parse_zip(zip_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """解析 ZIP 包，返回 (stock_df, index_df)。

        Returns:
            (stock overview DataFrame, index data DataFrame)
        """
        with zipfile.ZipFile(zip_path, "r") as z:
            names = z.namelist()
            stock_file = next(n for n in names if n.endswith("stock overview.csv"))
            index_file = next(n for n in names if n.endswith("index data.csv"))

            with z.open(stock_file) as f:
                stock_df = pd.read_csv(f, encoding="gbk")
            with z.open(index_file) as f:
                index_df = pd.read_csv(f, encoding="gbk")

        return stock_df, index_df

    @staticmethod
    def get_trade_date(stock_df: pd.DataFrame) -> str:
        """从 stock DataFrame 提取交易日期（第一行）。"""
        # 第一行即数据的交易日
        return stock_df.iloc[0]["交易日期"]
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from back_testing.data.sync.overview_client import OverviewClient; print('OK')"`
Expected: `OK`

---

### Task 2: 创建 `sync_overview.py` 入口

**Files:**
- Create: `back_testing/data/sync/sync_overview.py`

- [ ] **Step 1: 创建 sync_overview.py 框架**

```python
"""预测者网数据同步

Usage:
    # 每日 cron（下载 ZIP）
    python back_testing/data/sync/sync_overview.py --mode daily

    # 一次性历史补全（本地 CSV 目录）
    python back_testing/data/sync/sync_overview.py --mode backfill --data-dir /path/to/stock/csv
"""
import argparse
import logging
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from back_testing.data.db.connection import get_engine
from back_testing.data.db.models import StockDaily, IndexDaily
from back_testing.data.sync.overview_client import OverviewClient
from sqlalchemy.dialects.postgresql import insert as pg_insert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("sync_overview")


# ── 字段映射 ──────────────────────────────────────────────────
CSV_TO_DB = {
    "股票代码": "stock_code",
    "股票名称": "stock_name",
    "交易日期": "trade_date",
    "新浪行业": "industry",
    "新浪概念": "concept",
    "新浪地域": "area",
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


def _na_to_none(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if pd.isna(val) or abs(val) == float("inf"):
            return None
    if pd.isna(val):
        return None
    return val


def _process_stock_df(df: pd.DataFrame) -> pd.DataFrame:
    """处理 stock overview DataFrame。"""
    df = df.rename(columns=CSV_TO_DB)
    keep_cols = [c for c in df.columns if c in CSV_TO_DB.values()]
    df = df[keep_cols]
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    if "limit_up" in df.columns:
        df["limit_up"] = df["limit_up"].apply(
            lambda x: str(x).strip() == "是" if pd.notna(x) else False
        )
    if "limit_down" in df.columns:
        df["limit_down"] = df["limit_down"].apply(
            lambda x: str(x).strip() == "是" if pd.notna(x) else False
        )
    df = df.replace("", None)
    return df


def _upsert_stock(df: pd.DataFrame, engine) -> int:
    """将 DataFrame upsert 到 stock_daily 表。返回写入行数。"""
    n = 0
    with engine.begin() as conn:
        for _, row in df.iterrows():
            trade_date = row["trade_date"]
            insert_data = {k: _na_to_none(v) for k, v in row.items() if hasattr(StockDaily, k)}
            insert_data["stock_code"] = row["stock_code"]
            insert_data["trade_date"] = trade_date
            stmt = pg_insert(StockDaily).values(**insert_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["stock_code", "trade_date"],
                set_=insert_data,
            )
            conn.execute(stmt)
            n += 1
    return n


def _upsert_index(df: pd.DataFrame, engine) -> int:
    """将 index DataFrame upsert 到 index_daily 表。"""
    # index data.csv 列: index_code, date, open, close, low, high, volume, money, change
    n = 0
    with engine.begin() as conn:
        for _, row in df.iterrows():
            trade_date = pd.to_datetime(row["date"]).date()
            insert_data = {
                "index_code": row["index_code"],
                "trade_date": trade_date,
                "open": _na_to_none(row["open"]),
                "high": _na_to_none(row["high"]),
                "low": _na_to_none(row["low"]),
                "close": _na_to_none(row["close"]),
                "volume": _na_to_none(row["volume"] * 100) if pd.notna(row.get("volume")) else None,  # 手→股
                "turnover": _na_to_none(row["money"] * 1000) if pd.notna(row.get("money")) else None,  # 千元→元
            }
            stmt = pg_insert(IndexDaily).values(**insert_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["index_code", "trade_date"],
                set_=insert_data,
            )
            conn.execute(stmt)
            n += 1
    return n


def _build_cache(trade_date: date, engine):
    """构建指定日期的 Parquet 缓存。"""
    from back_testing.data.daily_data_cache import DAILY_CACHE_DIR

    trade_date_str = trade_date.strftime("%Y-%m-%d")

    with engine.connect() as conn:
        cache_df = pd.read_sql(
            text("""
                SELECT * FROM stock_daily WHERE trade_date = :td ORDER BY stock_code
            """),
            conn,
            params={"td": trade_date_str},
        )

    if cache_df.empty:
        logger.warning(f"缓存构建：无数据 {trade_date_str}")
        return

    cache_df["trade_date"] = pd.to_datetime(cache_df["trade_date"]).dt.strftime("%Y-%m-%d")
    DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = DAILY_CACHE_DIR / f"{trade_date_str}.parquet"
    cache_df.to_parquet(parquet_path, index=False)
    logger.info(f"缓存写入: {parquet_path} ({len(cache_df)} 条)")


def run_daily():
    """每日模式：从预测者网下载 ZIP 并同步。"""
    email = os.environ.get("YUCEZHE_EMAIL")
    api_key = os.environ.get("YUCEZHE_API_KEY")
    if not email or not api_key:
        logger.error("请设置 YUCEZHE_EMAIL 和 YUCEZHE_API_KEY 环境变量")
        sys.exit(1)

    client = OverviewClient(email, api_key)
    zip_path = client.download_today()
    try:
        stock_df, index_df = client.parse_zip(zip_path)
    finally:
        os.unlink(zip_path)

    trade_date_str = client.get_trade_date(stock_df)
    logger.info(f"交易日期: {trade_date_str}")

    stock_df = _process_stock_df(stock_df)
    stock_df = stock_df[stock_df["trade_date"] == pd.to_datetime(trade_date_str).date()]

    engine = get_engine()

    n_stock = _upsert_stock(stock_df, engine)
    logger.info(f"股票 upsert: {n_stock} 条")

    n_index = _upsert_index(index_df, engine)
    logger.info(f"指数 upsert: {n_index} 条")

    trade_date = pd.to_datetime(trade_date_str).date()
    _build_cache(trade_date, engine)


def run_backfill(data_dir: str):
    """历史补全模式：扫描本地 CSV 目录，补全缺失数据。"""
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"目录不存在: {data_dir}")
        sys.exit(1)

    engine = get_engine()
    csv_files = sorted(data_path.glob("*.csv"))
    logger.info(f"找到 {len(csv_files)} 个 CSV 文件")

    total_upsert = 0
    for i, fp in enumerate(csv_files):
        stock_code = fp.stem
        df = pd.read_csv(fp, encoding="gbk")
        df = _process_stock_df(df)

        # 从 DB 读该股已有最大日期
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT MAX(trade_date) FROM stock_daily WHERE stock_code = :code"),
                {"code": stock_code},
            )
            db_max = result.scalar()

        if db_max is not None:
            df = df[df["trade_date"] > db_max]

        if df.empty:
            continue

        n = _upsert_stock(df, engine)
        total_upsert += n
        logger.info(f"  {stock_code}: 写入 {n} 条")

        if (i + 1) % 500 == 0:
            logger.info(f"进度: {i + 1}/{len(csv_files)}")

    logger.info(f"历史补全完成: 共写入 {total_upsert} 条")

    # 缓存：取最新导入的日期
    with engine.connect() as conn:
        result = conn.execute(text("SELECT MAX(trade_date) FROM stock_daily"))
        latest = result.scalar()
    if latest:
        _build_cache(latest, engine)


def main():
    parser = argparse.ArgumentParser(description="预测者网数据同步")
    parser.add_argument("--mode", choices=["daily", "backfill"], required=True)
    parser.add_argument("--data-dir", help="backfill 模式：CSV 目录路径")
    args = parser.parse_args()

    if args.mode == "daily":
        run_daily()
    else:
        if not args.data_dir:
            parser.error("--mode backfill 需要 --data-dir")
        run_backfill(args.data_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile back_testing/data/sync/sync_overview.py && echo "OK"`
Expected: `OK`

---

### Task 3: 编写测试

**Files:**
- Create: `tests/back_testing/data/sync/test_sync_overview.py`

- [ ] **Step 1: 写测试**

```python
"""Tests for sync_overview.py"""
import os
import tempfile
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestProcessStockDf:
    """测试 _process_stock_df 字段映射和类型转换。"""

    def test_columns_renamed(self):
        from back_testing.data.sync.sync_overview import _process_stock_df

        raw = pd.DataFrame({
            "股票代码": ["sh600519", "sz000001"],
            "股票名称": ["贵州茅台", "平安银行"],
            "交易日期": ["2025-04-23", "2025-04-23"],
            "开盘价": [1800.0, 12.0],
            "最高价": [1850.0, 12.5],
            "最低价": [1790.0, 11.8],
            "收盘价": [1830.0, 12.2],
            "后复权价": [1830.0, 12.2],
            "前复权价": [1800.0, 12.0],
            "涨跌幅": [1.5, 0.8],
            "成交量": [100000.0, 500000.0],
            "成交额": [1000000.0, 5000000.0],
            "换手率": [0.5, 0.3],
            "流通市值": [1e9, 2e9],
            "总市值": [2e9, 3e9],
            "是否涨停": ["否", "是"],
            "是否跌停": ["否", "否"],
            "市盈率TTM": [30.0, 8.0],
            "市销率TTM": [10.0, 5.0],
            "市现率TTM": [20.0, 15.0],
            "市净率": [5.0, 1.2],
            "MA_5": [1820.0, 12.1],
            "MA_10": [1810.0, 12.0],
            "MA_20": [1800.0, 11.9],
            "MA_30": [1795.0, 11.8],
            "MA_60": [1780.0, 11.5],
            "MA金叉死叉": ["金叉", "死叉"],
            "MACD_DIF": [1.5, 0.3],
            "MACD_DEA": [1.2, 0.2],
            "MACD_MACD": [0.6, 0.2],
            "MACD_金叉死叉": ["金叉", "死叉"],
            "KDJ_K": [60.0, 50.0],
            "KDJ_D": [55.0, 48.0],
            "KDJ_J": [70.0, 54.0],
            "KDJ_金叉死叉": ["金叉", "死叉"],
            "布林线中轨": [1800.0, 12.0],
            "布林线上轨": [1850.0, 12.5],
            "布林线下轨": [1750.0, 11.5],
            "psy": [50.0, 45.0],
            "psyma": [48.0, 44.0],
            "rsi1": [60.0, 55.0],
            "rsi2": [58.0, 53.0],
            "rsi3": [56.0, 51.0],
            "振幅": [3.0, 5.0],
            "量比": [1.2, 0.9],
        })

        df = _process_stock_df(raw)

        assert list(df.columns) == [
            "stock_code", "stock_name", "trade_date", "open", "high", "low", "close",
            "adj_close", "prev_adj_close", "change_pct", "volume", "turnover_amount",
            "turnover_rate", "circulating_mv", "total_mv", "limit_up", "limit_down",
            "pe_ttm", "ps_ttm", "pcf_ttm", "pb",
            "ma_5", "ma_10", "ma_20", "ma_30", "ma_60", "ma_cross",
            "macd_dif", "macd_dea", "macd_hist", "macd_cross",
            "kdj_k", "kdj_d", "kdj_j", "kdj_cross",
            "boll_mid", "boll_upper", "boll_lower",
            "psy", "psyma", "rsi_1", "rsi_2", "rsi_3",
            "amplitude", "volume_ratio",
        ]
        assert df.iloc[0]["stock_code"] == "sh600519"
        assert df.iloc[0]["limit_up"] is False
        assert df.iloc[1]["limit_up"] is True
        assert df.iloc[0]["trade_date"] == date(2025, 4, 23)


class TestNaToNone:
    """测试 _na_to_none。"""

    def test_normal_passthrough(self):
        from back_testing.data.sync.sync_overview import _na_to_none
        assert _na_to_none(42) == 42
        assert _na_to_none(3.14) == 3.14

    def test_nan_becomes_none(self):
        from back_testing.data.sync.sync_overview import _na_to_none
        assert _na_to_none(float("nan")) is None
        assert _na_to_none(float("inf")) is None
        assert _na_to_none(-float("inf")) is None

    def test_pd_na(self):
        from back_testing.data.sync.sync_overview import _na_to_none
        assert _na_to_none(pd.NA) is None


class TestOverviewClientParse:
    """测试 OverviewClient.parse_zip。"""

    def test_parse_zip(self):
        from back_testing.data.sync.overview_client import OverviewClient

        # 构造模拟 ZIP
        stock_csv = "股票代码,股票名称,交易日期,开盘价,最高价,最低价,收盘价\nsh600519,贵州茅台,2025-04-23,1800,1850,1790,1830"
        index_csv = "index_code,date,open,close,low,high,volume,money,change\nsh000001,2025-04-23,3400,3500,3380,3510,100000,100000000,0.02"

        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as z:
            z.writestr("overview-push-20250423/stock overview.csv", stock_csv)
            z.writestr("overview-push-20250423/index data.csv", index_csv)

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(zip_buf.getvalue())
            tmp_path = f.name

        try:
            stock_df, index_df = OverviewClient.parse_zip(tmp_path)
            assert len(stock_df) == 1
            assert len(index_df) == 1
            assert index_df.iloc[0]["index_code"] == "sh000001"
        finally:
            os.unlink(tmp_path)
```

- [ ] **Step 2: 运行测试**

Run: `pytest tests/back_testing/data/sync/test_sync_overview.py -v`
Expected: 5 passed

---

## 验证清单

| 检查项 | 命令 |
|--------|------|
| 语法正确 | `python -m py_compile back_testing/data/sync/sync_overview.py && echo OK` |
| 导入无报错 | `python -c "from back_testing.data.sync.overview_client import OverviewClient; print('OK')"` |
| 测试通过 | `pytest tests/back_testing/data/sync/test_sync_overview.py -v` |
| 日志格式 | `python back_testing/data/sync/sync_overview.py --help` |

---

## Spec 覆盖检查

- [x] 每日下载 ZIP → Task 1 `download_today` + Task 2 `run_daily`
- [x] 历史补全 CSV 目录 → Task 2 `run_backfill`
- [x] Upsert 防重 → `_upsert_stock` / `_upsert_index`
- [x] 自动检测缺失日期 → `run_backfill` 中 `db_max` 过滤
- [x] 指数数据同步 → `_upsert_index`
- [x] 缓存构建 → `_build_cache`
- [x] CSV_TO_DB 映射复用 → Task 2 直接复用
