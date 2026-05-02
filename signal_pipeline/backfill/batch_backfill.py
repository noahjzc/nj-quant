"""Batch Backfill — 批量从 Tushare 拉取日线数据写入 PostgreSQL。

Usage:
    python signal_pipeline/batch_backfill.py --date 2025-04-23
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
    _backfill_single_day,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("batch_backfill")


def main():
    parser = argparse.ArgumentParser(description="补全指定日期的 Tushare 日线数据")
    parser.add_argument("--date", required=True, help="目标日期 YYYY-MM-DD")
    args = parser.parse_args()

    target_date = pd.to_datetime(args.date).date()

    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        logger.error("TUSHARE_TOKEN environment variable not set")
        sys.exit(1)

    logger.info(f"=== Batch Backfill {target_date} ===")

    ok, rows, err = _backfill_single_day(target_date, token)
    if ok:
        logger.info(f"{target_date}  ok  {rows} stocks processed")
    else:
        logger.error(f"{target_date}  failed  {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
