"""Batch Backfill — 批量从 Tushare 拉取日线数据写入 PostgreSQL。

Usage:
    python signal_pipeline/batch_backfill.py --start 2025-04-23 --end 2025-04-30
"""
import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from signal_pipeline.night_backfill import (
    TushareClient,
    _convert_ts_code,
    _na_to_none,
    _get_engine,
    _backfill_single_day,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("batch_backfill")


def _generate_trading_days(start: date, end: date) -> list:
    """生成 start ~ end 范围内的所有交易日（跳过周末）。"""
    return pd.bdate_range(start, end).tolist()


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
    logger.info(f"Trading days: {[str(d.date()) for d in days]}")

    succeeded, failed = 0, []
    for d in days:
        ok, rows, err = _backfill_single_day(d, token)
        if ok:
            logger.info(f"{d.date()}  ✓  {rows} rows")
            succeeded += 1
        else:
            logger.error(f"{d.date()}  ✗  {err}")
            failed.append((d, err))
            break  # 失败即停

    if failed:
        logger.error(f"=== FAILED: {failed[0][0].date()} — {failed[0][1]} ===")
        logger.error(f"Total: {succeeded} succeeded, {len(failed)} failed")
        sys.exit(1)
    else:
        logger.info(f"=== All {succeeded} days succeeded ===")


if __name__ == "__main__":
    main()
