"""Night Backfill Script — cron 18:00 daily entry point.

Steps:
1. Fetch complete daily data from Tushare (previous trading day)
2. Merge with daily_basic for valuation fields
3. Get adj_factor for 复权
4. Calculate technical indicators
5. Upsert to DB (stock_daily table)
6. Incrementally build Parquet cache
7. Log to cron_log
"""
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.dialects.postgresql import insert as pg_insert

from back_testing.data.db.connection import get_engine, get_session
from back_testing.data.db.models import StockDaily
from signal_pipeline.data_sources.tushare_client import TushareClient
from signal_pipeline.indicator_calculator import IndicatorCalculator

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("night_backfill")

# ── Cache dir ─────────────────────────────────────────────────────────────────

CACHE_DIR = Path("cache/daily_rotation")
DAILY_CACHE_DIR = CACHE_DIR / "daily"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _convert_ts_code(ts_code: str) -> str:
    """Convert Tushare code format to local format.

    000001.SZ → sz000001
    600519.SH → sh600519
    """
    if ts_code.endswith(".SZ"):
        return "sz" + ts_code[:-3]
    if ts_code.endswith(".SH"):
        return "sh" + ts_code[:-3]
    raise ValueError(f"Unknown Tushare code suffix: {ts_code}")


def _na_to_none(val):
    """Convert numpy/pandas NA to None for PostgreSQL compatibility."""
    if val is None:
        return None
    if isinstance(val, (np.integer, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return None
        return val.item()
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    if pd.isna(val):
        return None
    return val


def _cron_start(session, task_name: str) -> int:
    """Log cron task start. Returns log id."""
    from back_testing.data.db.models import Base
    from sqlalchemy import Column, Integer, String, Date, Numeric, Boolean, DateTime, Index, Text

    class CronLog(Base):
        __tablename__ = "cron_log"

        id = Column(Integer, primary_key=True)
        task_name = Column(String(50), nullable=False)
        status = Column(String(10), nullable=False)
        started_at = Column(DateTime)
        finished_at = Column(DateTime)
        error_message = Column(Text)
        metadata = Column(Text)

    result = session.execute(
        CronLog.__table__.insert().values(
            task_name=task_name,
            status="running",
            started_at=session.bind.func.now(),
        )
    )
    session.commit()
    return result.inserted_primary_key[0]


def _cron_finish(session, log_id: int, status: str, error_message: str = None):
    """Log cron task finish."""
    from sqlalchemy import text

    session.execute(
        text("UPDATE cron_log SET status = :status, finished_at = NOW()"
             " WHERE id = :id"),
        {"status": status, "id": log_id},
    )
    if error_message:
        session.execute(
            text("UPDATE cron_log SET error_message = :msg WHERE id = :id"),
            {"msg": error_message, "id": log_id},
        )
    session.commit()


def _get_engine():
    from back_testing.data.db.connection import get_engine as _ge
    return _ge()


def _get_previous_trading_day() -> date:
    """Return the previous trading day (weekday, backfill at 18:00)."""
    today = date.today()
    yesterday = today - timedelta(days=1)
    if yesterday.weekday() < 5:  # Mon–Fri
        return yesterday
    # Weekend → Friday
    return yesterday - timedelta(days=yesterday.weekday() - 4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 0. Token check
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        logger.error("TUSHARE_TOKEN environment variable not set")
        sys.exit(1)

    # 0. Target date (previous trading day by default)
    if len(sys.argv) > 1:
        target_date = pd.to_datetime(sys.argv[1]).date()
    else:
        target_date = _get_previous_trading_day()

    logger.info(f"Night backfill for {target_date}")

    # 1. Init Tushare client
    client = TushareClient(token)

    # 2. Cron logging
    engine = get_engine()
    Session = get_session()
    session = Session()
    log_id = _cron_start(session, "night_backfill")

    try:
        # ── 3. Fetch from Tushare ───────────────────────────────────────────
        tushare_date = target_date.strftime("%Y%m%d")  # "20250428"

        logger.info(f"Fetching daily data for {tushare_date}...")
        daily_df = client.get_daily_all(tushare_date)
        logger.info(f"  daily: {len(daily_df)} rows")

        logger.info(f"Fetching daily_basic for {tushare_date}...")
        basic_df = client.get_daily_basic_all(tushare_date)
        logger.info(f"  daily_basic: {len(basic_df)} rows")

        logger.info(f"Fetching adj_factor for {tushare_date}...")
        adj_df = client.get_adj_factor_all(tushare_date)
        logger.info(f"  adj_factor: {len(adj_df)} rows")

        # ── 4. Merge ───────────────────────────────────────────────────────
        # Convert ts_code → local stock_code
        daily_df["stock_code"] = daily_df["ts_code"].apply(_convert_ts_code)
        basic_df["stock_code"] = basic_df["ts_code"].apply(_convert_ts_code)
        adj_df["stock_code"] = adj_df["ts_code"].apply(_convert_ts_code)

        # Merge daily + basic on stock_code
        merged = daily_df.merge(
            basic_df[["stock_code", "turnover_rate", "volume_ratio",
                      "circulating_mv", "total_mv", "pe_ttm", "ps_ttm",
                      "pcf_ttm", "pb"]],
            on="stock_code",
            how="left",
        )

        # Merge adj_factor
        adj_df_sorted = adj_df.sort_values("trade_date")
        # Keep only the latest adj_factor per stock (most recent trading day)
        adj_latest = adj_df_sorted.groupby("stock_code", sort=False).last().reset_index()
        merged = merged.merge(
            adj_latest[["stock_code", "adj_factor"]],
            on="stock_code",
            how="left",
        )

        # ── 5. Compute adjusted close (后复权) ──────────────────────────────
        # adj_close = close * adj_factor
        merged["adj_close"] = merged["close"] * merged["adj_factor"]

        # prev_adj_close: use previous day's adj_close from adj_factor history
        # For simplicity, use (close * adj_factor) shifted per stock
        merged = merged.sort_values(["stock_code", "trade_date"])
        merged["prev_adj_close"] = merged.groupby("stock_code")["adj_close"].shift(1)

        # ── 6. Calculate indicators ────────────────────────────────────────
        # Ensure required columns exist
        for col in ["stock_code", "trade_date", "open", "high", "low", "close", "volume"]:
            if col not in merged.columns:
                raise ValueError(f"Missing required column: {col}")

        merged["trade_date"] = pd.to_datetime(merged["trade_date"])
        merged = merged.sort_values(["stock_code", "trade_date"])

        logger.info("Calculating technical indicators...")
        merged = IndicatorCalculator.calculate_all(merged)
        logger.info(f"  indicators computed, shape: {merged.shape}")

        # ── 7. Prepare for DB upsert ────────────────────────────────────────
        trade_date_str = target_date.strftime("%Y-%m-%d")

        # Build amplitude and change_pct
        merged["amplitude"] = (merged["high"] - merged["low"]) / merged["low"] * 100
        merged["change_pct"] = merged["pct_chg"]  # from Tushare daily

        # limit_up / limit_down flags (涨停/跌停)
        merged["limit_up"] = merged["pct_chg"] >= 9.9
        merged["limit_down"] = merged["pct_chg"] <= -9.9

        # Select and rename columns for stock_daily
        record_cols = [
            "stock_code", "trade_date", "open", "high", "low", "close", "volume",
            "turnover_amount", "adj_close", "prev_adj_close", "amplitude", "change_pct",
            "turnover_rate", "volume_ratio", "circulating_mv", "total_mv",
            "limit_up", "limit_down",
            "pe_ttm", "ps_ttm", "pcf_ttm", "pb",
            "ma_5", "ma_10", "ma_20", "ma_30", "ma_60", "ma_cross",
            "macd_dif", "macd_dea", "macd_hist", "macd_cross",
            "kdj_k", "kdj_d", "kdj_j", "kdj_cross",
            "boll_mid", "boll_upper", "boll_lower",
            "psy", "psyma", "rsi_1", "rsi_2", "rsi_3",
        ]

        # Only keep columns that exist
        available_cols = [c for c in record_cols if c in merged.columns]
        db_df = merged[available_cols].copy()

        # Convert numpy types and NA → None
        for col in db_df.columns:
            if col in ("stock_code", "trade_date", "ma_cross", "macd_cross", "kdj_cross"):
                continue
            db_df[col] = db_df[col].apply(_na_to_none)

        # trade_date → string for DB
        db_df["trade_date"] = pd.to_datetime(db_df["trade_date"]).dt.strftime("%Y-%m-%d")

        records = db_df.to_dict("records")
        logger.info(f"Prepared {len(records)} records for upsert")

        # ── 8. Upsert to DB ────────────────────────────────────────────────
        logger.info("Upserting to stock_daily table...")

        with engine.begin() as conn:
            for record in records:
                record_copy = dict(record)
                stock_code = record_copy.pop("stock_code")
                trade_date = record_copy.pop("trade_date")

                # Remove keys not in StockDaily table
                insert_data = {
                    k: v for k, v in record_copy.items()
                    if hasattr(StockDaily, k)
                }
                insert_data["stock_code"] = stock_code
                insert_data["trade_date"] = trade_date

                stmt = pg_insert(StockDaily).values(**insert_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["stock_code", "trade_date"],
                    set_=insert_data,
                )
                conn.execute(stmt)

        logger.info("DB upsert complete")

        # ── 9. Write Parquet cache ─────────────────────────────────────────
        DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Prepare parquet DataFrame (use original column names)
        pq_df = merged.copy()
        pq_df["trade_date"] = pd.to_datetime(pq_df["trade_date"]).dt.strftime("%Y-%m-%d")
        # Select all available indicator columns plus base columns
        pq_cols = [
            "trade_date", "stock_code", "open", "high", "low", "close", "volume",
            "turnover_amount", "adj_close", "prev_adj_close", "amplitude", "change_pct",
            "turnover_rate", "volume_ratio", "circulating_mv", "total_mv",
            "limit_up", "limit_down", "pe_ttm", "ps_ttm", "pcf_ttm", "pb",
            "ma_5", "ma_10", "ma_20", "ma_30", "ma_60", "ma_cross",
            "macd_dif", "macd_dea", "macd_hist", "macd_cross",
            "kdj_k", "kdj_d", "kdj_j", "kdj_cross",
            "boll_mid", "boll_upper", "boll_lower",
            "psy", "psyma", "rsi_1", "rsi_2", "rsi_3",
            "vol_ma5", "vol_ma20", "close_std_20", "high_20_max",
            "atr_14", "wr_10", "wr_14", "ret_5", "ret_20",
        ]
        pq_available = [c for c in pq_cols if c in pq_df.columns]
        pq_df = pq_df[pq_available]

        parquet_path = DAILY_CACHE_DIR / f"{trade_date_str}.parquet"
        pq_df.to_parquet(parquet_path, index=False)
        logger.info(f"Parquet written: {parquet_path}")

        # ── 10. Update trading_dates.parquet ────────────────────────────────
        trading_dates_path = CACHE_DIR / "trading_dates.parquet"
        if trading_dates_path.exists():
            dates_df = pd.read_parquet(trading_dates_path)
            if trade_date_str not in dates_df["trade_date"].values:
                dates_df = pd.concat(
                    [dates_df, pd.DataFrame({"trade_date": [trade_date_str]})],
                    ignore_index=True,
                )
                dates_df = dates_df.sort_values("trade_date").reset_index(drop=True)
                dates_df.to_parquet(trading_dates_path, index=False)
                logger.info(f"Updated trading_dates.parquet with {trade_date_str}")
        else:
            pd.DataFrame({"trade_date": [trade_date_str]}).to_parquet(
                trading_dates_path, index=False
            )
            logger.info(f"Created trading_dates.parquet with {trade_date_str}")

        # ── 11. Cron finish ─────────────────────────────────────────────────
        _cron_finish(session, log_id, "success")
        logger.info(f"Night backfill for {target_date} completed successfully")

    except Exception as e:
        logger.exception(f"Night backfill failed: {e}")
        _cron_finish(session, log_id, "failed", str(e))
        sys.exit(1)
    finally:
        session.close()


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

        if basic_df.empty:
            logger.warning(f"daily_basic returned empty for {tushare_date}")
        if adj_df.empty:
            logger.warning(f"adj_factor returned empty for {tushare_date}")

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


if __name__ == "__main__":
    main()
