# Tushare 批量数据补全实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `batch_backfill.py`，支持 `--start` / `--end` 指定日期范围，逐日从 Tushare 拉取数据并写入 PostgreSQL，含完整技术指标。

**Architecture:** 抽取 `night_backfill.py` 中"单日 DB 写入"逻辑为独立函数 `backfill_single_day()`，批量脚本循环调用，失败即停。Parquet 写入和 cron_log 由夜间定时任务负责，批量脚本专注 DB 数据完整性。

**Tech Stack:** Python, pandas, tushare, sqlalchemy (PostgreSQL upsert)

---

## 文件结构

```
signal_pipeline/
├── night_backfill.py      # 不改动
└── batch_backfill.py      # 新建：批量补全入口
```

---

## Task 1: 创建 `batch_backfill.py` 框架

**Files:**
- Create: `signal_pipeline/batch_backfill.py`

- [ ] **Step 1: 写框架（import + argparser + 交易日生成）**

```python
"""Batch Backfill — 批量从 Tushare 拉取日线数据写入 PostgreSQL。

Usage:
    python signal_pipeline/batch_backfill.py --start 2025-04-23 --end 2025-04-30
"""
import argparse
import logging
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, str(__file__.parent))
from signal_pipeline.night_backfill import (
    TushareClient,
    _convert_ts_code,
    _na_to_none,
    _get_engine,
    _get_session,
    _backfill_single_day,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("batch_backfill")


def _generate_trading_days(start: date, end: date) -> list[date]:
    """生成 start ~ end 范围内的所有交易日（跳过周末）。"""
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon-Fri
            days.append(current)
        current = date(current.year, current.month, current.day + 1)
        # naive date arithmetic, handle month boundary manually
        from datetime import timedelta
        current = start + timedelta(days=(current - start).days)
        if current > end:
            break
    return days


def main():
    parser = argparse.ArgumentParser(description="批量补全 Tushare 日线数据")
    parser.add_argument("--start", required=True, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD")
    args = parser.parse_args()

    start_date = pd.to_datetime(args.start).date()
    end_date = pd.to_datetime(args.end).date()

    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        logger.error("TUSHARE_TOKEN environment variable not set")
        sys.exit(1)

    days = _generate_trading_days(start_date, end_date)
    logger.info(f"=== Batch Backfill {start_date} ~ {end_date} ===")
    logger.info(f"Trading days: {[str(d) for d in days]}")

    succeeded, failed = 0, []
    for d in days:
        ok, rows, err = _backfill_single_day(d, token)
        if ok:
            logger.info(f"{d}  ✓  {rows} rows")
            succeeded += 1
        else:
            logger.error(f"{d}  ✗  {err}")
            failed.append((d, err))
            break  # B 策略：失败即停

    if failed:
        logger.error(f"=== FAILED: {failed[0][0]} — {failed[0][1]} ===")
        logger.error(f"Total: {succeeded} succeeded, {len(failed)} failed")
        sys.exit(1)
    else:
        logger.info(f"=== All {succeeded} days succeeded ===")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行框架验证 import 不报错**

Run: `python -c "import signal_pipeline.batch_backfill" 2>&1 | head -20`
Expected: `ModuleNotFoundError: cannot import name '_backfill_single_day'` (函数尚不存在)

- [ ] **Step 3: Commit**

```bash
git add signal_pipeline/batch_backfill.py
git commit -m "feat: scaffold batch_backfill.py framework"
```

---

## Task 2: 在 `night_backfill.py` 中抽取 `_backfill_single_day()` 函数

**Files:**
- Modify: `signal_pipeline/night_backfill.py:1-345` — 在 `if __name__ == "__main__": main()` 前插入新函数

- [ ] **Step 1: 添加 `_get_engine` / `_get_session` 顶层访问函数**

在 `_get_previous_trading_day()` 前添加：

```python
def _get_engine():
    from back_testing.data.db.connection import get_engine as _ge
    return _ge()


def _get_session():
    from back_testing.data.db.connection import get_session as _gs
    return _gs
```

- [ ] **Step 2: 添加 `_backfill_single_day()` 函数**

在 `if __name__ == "__main__":` 前插入：

```python
def _backfill_single_day(target_date: date, token: str) -> tuple[bool, int, str | None]:
    """单日补全逻辑（不写 Parquet，不写 cron_log）。

    Returns:
        (success, rows_written, error_message)
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from back_testing.data.db.models import StockDaily

    client = TushareClient(token)
    tushare_date = target_date.strftime("%Y%m%d")

    try:
        daily_df = client.get_daily_all(tushare_date)
        if daily_df.empty:
            return False, 0, f"Tushare daily returned empty for {tushare_date}"

        basic_df = client.get_daily_basic_all(tushare_date)
        adj_df = client.get_adj_factor_all(tushare_date)

        # Convert ts_code → stock_code
        daily_df["stock_code"] = daily_df["ts_code"].apply(_convert_ts_code)
        basic_df["stock_code"] = basic_df["ts_code"].apply(_convert_ts_code)
        adj_df["stock_code"] = adj_df["ts_code"].apply(_convert_ts_code)

        # Merge
        merged = daily_df.merge(
            basic_df[["stock_code", "turnover_rate", "volume_ratio",
                      "circulating_mv", "total_mv", "pe_ttm", "ps_ttm", "pcf_ttm", "pb"]],
            on="stock_code", how="left",
        )
        adj_latest = adj_df.sort_values("trade_date").groupby("stock_code", sort=False).last().reset_index()
        merged = merged.merge(adj_latest[["stock_code", "adj_factor"]], on="stock_code", how="left")

        # 后复权
        merged["adj_close"] = merged["close"] * merged["adj_factor"]
        merged = merged.sort_values(["stock_code", "trade_date"])
        merged["prev_adj_close"] = merged.groupby("stock_code")["adj_close"].shift(1)

        # 计算指标
        for col in ["stock_code", "trade_date", "open", "high", "low", "close", "volume"]:
            if col not in merged.columns:
                return False, 0, f"Missing required column: {col}"
        merged["trade_date"] = pd.to_datetime(merged["trade_date"])
        merged = merged.sort_values(["stock_code", "trade_date"])
        merged = IndicatorCalculator.calculate_all(merged)

        # 准备字段
        merged["amplitude"] = (merged["high"] - merged["low"]) / merged["low"] * 100
        merged["change_pct"] = merged["pct_chg"]
        merged["limit_up"] = merged["pct_chg"] >= 9.9
        merged["limit_down"] = merged["pct_chg"] <= -9.9

        # 写 DB
        record_cols = [
            "stock_code", "trade_date", "open", "high", "low", "close", "volume",
            "turnover_amount", "adj_close", "prev_adj_close", "amplitude", "change_pct",
            "turnover_rate", "volume_ratio", "circulating_mv", "total_mv",
            "limit_up", "limit_down", "pe_ttm", "ps_ttm", "pcf_ttm", "pb",
            "ma_5", "ma_10", "ma_20", "ma_30", "ma_60", "ma_cross",
            "macd_dif", "macd_dea", "macd_hist", "macd_cross",
            "kdj_k", "kdj_d", "kdj_j", "kdj_cross",
            "boll_mid", "boll_upper", "boll_lower",
            "psy", "psyma", "rsi_1", "rsi_2", "rsi_3",
        ]
        available_cols = [c for c in record_cols if c in merged.columns]
        db_df = merged[available_cols].copy()
        for col in db_df.columns:
            if col in ("stock_code", "trade_date", "ma_cross", "macd_cross", "kdj_cross"):
                continue
            db_df[col] = db_df[col].apply(_na_to_none)
        db_df["trade_date"] = pd.to_datetime(db_df["trade_date"]).dt.strftime("%Y-%m-%d")

        engine = _get_engine()
        with engine.begin() as conn:
            for record in db_df.to_dict("records"):
                record_copy = dict(record)
                stock_code = record_copy.pop("stock_code")
                trade_date = record_copy.pop("trade_date")
                insert_data = {k: v for k, v in record_copy.items() if hasattr(StockDaily, k)}
                insert_data["stock_code"] = stock_code
                insert_data["trade_date"] = trade_date
                stmt = pg_insert(StockDaily).values(**insert_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["stock_code", "trade_date"],
                    set_=insert_data,
                )
                conn.execute(stmt)

        return True, len(db_df), None

    except Exception as e:
        return False, 0, str(e)
```

- [ ] **Step 3: 验证 night_backfill.py 语法正确**

Run: `python -m py_compile signal_pipeline/night_backfill.py && echo "OK"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add signal_pipeline/night_backfill.py
git commit -m "refactor: extract _backfill_single_day() for reuse in batch script"
```

---

## Task 3: 补全 `batch_backfill.py` 并测试

**Files:**
- Modify: `signal_pipeline/batch_backfill.py`

- [ ] **Step 1: 修复交易日生成逻辑**

pandas 的 `pd.bdate_range()` 更简洁：

```python
from pandas.tseries.offsets import BDay

def _generate_trading_days(start: date, end: date) -> list[date]:
    return pd.bdate_range(start, end).tolist()
```

- [ ] **Step 2: 运行语法检查**

Run: `python -m py_compile signal_pipeline/batch_backfill.py && echo "OK"`
Expected: `OK`

- [ ] **Step 3: 测试交易日生成**

Run: `python -c "from signal_pipeline.batch_backfill import _generate_trading_days; from datetime import date; days = _generate_trading_days(date(2025,4,23), date(2025,4,30)); print([str(d.date()) for d in days])"`
Expected: 包含 4月23/24/28/29/30（4个工作日）

- [ ] **Step 4: Commit**

```bash
git add signal_pipeline/batch_backfill.py
git commit -m "feat: complete batch_backfill with trading day generation"
```

---

## Task 4: 写测试

**Files:**
- Create: `tests/signal_pipeline/test_batch_backfill.py`

- [ ] **Step 1: 写测试**

```python
import pytest
from datetime import date
from unittest.mock import patch, MagicMock

from signal_pipeline.batch_backfill import _generate_trading_days


class TestGenerateTradingDays:
    def test_filters_weekends(self):
        """Thu-Fri-Mon range should include Thu/Fri, skip Sat/Sun, include Mon."""
        days = _generate_trading_days(date(2025, 4, 24), date(2025, 4, 28))
        day_strs = [str(d.date()) for d in days]
        assert "2025-04-24" in day_strs  # Thursday
        assert "2025-04-25" in day_strs  # Friday
        assert "2025-04-26" not in day_strs  # Saturday
        assert "2025-04-27" not in day_strs  # Sunday
        assert "2025-04-28" in day_strs  # Monday

    def test_single_day_weekday(self):
        days = _generate_trading_days(date(2025, 4, 23), date(2025, 4, 23))
        assert len(days) == 1

    def test_single_day_weekend(self):
        days = _generate_trading_days(date(2025, 4, 26), date(2025, 4, 26))  # Saturday
        assert len(days) == 0
```

- [ ] **Step 2: 运行测试**

Run: `pytest tests/signal_pipeline/test_batch_backfill.py -v`
Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add tests/signal_pipeline/test_batch_backfill.py
git commit -m "test: add batch_backfill trading day generation tests"
```

---

## 验证清单

| 检查项 | 命令 |
|--------|------|
| 语法正确 | `python -m py_compile signal_pipeline/batch_backfill.py` |
| 测试通过 | `pytest tests/signal_pipeline/test_batch_backfill.py -v` |
| import 无报错 | `python -c "from signal_pipeline.batch_backfill import _generate_trading_days"` |
| 交易日计算正确 | `python -c "from signal_pipeline.batch_backfill import _generate_trading_days; from datetime import date; print([str(d.date()) for d in _generate_trading_days(date(2025,4,23), date(2025,4,30))])"` |

---

## Spec 覆盖检查

- [x] `--start` / `--end` 参数指定日期范围 → Task 1
- [x] 自动跳过非交易日（周末） → Task 3 `_generate_trading_days`
- [x] 失败即停（B 策略） → Task 1 `break`
- [x] 复用 TushareClient / IndicatorCalculator → Task 2 `_backfill_single_day`
- [x] 技术指标完整计算 → Task 2 调用 `IndicatorCalculator.calculate_all`
- [x] Parquet 不在批量脚本中写入 → 设计决定，Task 2 只写 DB
