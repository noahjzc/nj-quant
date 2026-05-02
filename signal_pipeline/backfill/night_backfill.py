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

from data.db.connection import get_engine, get_session
from data.db.models import StockDaily
from data.sync.tushare_client import TushareClient
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
    920017.BJ → bj920017
    """
    if ts_code.endswith(".SZ"):
        return "sz" + ts_code[:-3]
    if ts_code.endswith(".SH"):
        return "sh" + ts_code[:-3]
    if ts_code.endswith(".BJ"):
        return "bj" + ts_code[:-3]
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
    from data.db.models import Base
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
    from data.db.connection import get_engine as _ge
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
        # Tushare daily columns: vol=手, amount=千元 → DB: volume=股, turnover_amount=元
        daily_df = daily_df.rename(columns={'vol': 'vol_raw', 'amount': 'amount_raw'})
        daily_df['volume'] = daily_df['vol_raw'] * 100        # 手 → 股
        daily_df['turnover_amount'] = daily_df['amount_raw'] * 1000  # 千元 → 元
        daily_df = daily_df.drop(columns=['vol_raw', 'amount_raw'])
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
        # Map Tushare column names to DB column names
        column_map = {
            "turnover_rate": "turnover_rate",
            "volume_ratio": "volume_ratio",
            "total_mv": "total_mv",
            "pe_ttm": "pe_ttm",
            "ps_ttm": "ps_ttm",
            "pcf_ttm": "pcf_ttm",
            "pb": "pb",
            "circulating_mv": "circulating_mv",
            "float_mv": "circulating_mv",
            "circ_mv": "circulating_mv",
        }
        ts_to_db = {k: v for k, v in column_map.items() if k in basic_df.columns}
        select_cols = ["stock_code"] + list(ts_to_db.keys())
        merged = daily_df.merge(
            basic_df[select_cols].rename(columns=ts_to_db),
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


def _backfill_index_data(client: TushareClient, target_date: date) -> None:
    """从 Tushare 获取指定日期的主要指数数据，写入 index_daily 表。

    覆盖的指数：上证综指、沪深300、深证成指、创业板指。
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from data.db.models import IndexDaily

    # 本地代码 → Tushare 代码
    LOCAL_TO_TUSHARE = {
        "sh000001": "000001.SH",  # 上证综指
        "sh000300": "000300.SH",  # 沪深300
        "sz399001": "399001.SZ",  # 深证成指
        "sz399006": "399006.SZ",  # 创业板指
    }
    TUSHARE_TO_LOCAL = {v: k for k, v in LOCAL_TO_TUSHARE.items()}

    tushare_date = target_date.strftime("%Y%m%d")
    engine = _get_engine()

    for local_code, ts_code in LOCAL_TO_TUSHARE.items():
        try:
            # 用默认参数捕获 ts_code，避免闭包引用问题
            df = client._call_with_retry(
                lambda tc=ts_code: client.pro.index_daily(
                    ts_code=tc, start_date=tushare_date, end_date=tushare_date
                )
            )
            if df.empty:
                logger.warning(f"Index {ts_code} returned empty for {tushare_date}")
                continue

            for _, row in df.iterrows():
                # Tushare 返回 trade_date 是字符串 "YYYYMMDD"，需转成 date
                trade_date_val = row["trade_date"]
                if isinstance(trade_date_val, str):
                    trade_date_val = pd.to_datetime(trade_date_val).date()
                insert_data = {
                    "index_code": TUSHARE_TO_LOCAL.get(row["ts_code"], row["ts_code"].lower().replace(".", "")),
                    "trade_date": trade_date_val,
                    "open": _na_to_none(row["open"]),
                    "high": _na_to_none(row["high"]),
                    "low": _na_to_none(row["low"]),
                    "close": _na_to_none(row["close"]),
                    "volume": _na_to_none(row["vol"]) * 100 if row.get("vol") is not None else None,  # 手 → 股
                    "turnover": _na_to_none(row.get("amount")) * 1000 if row.get("amount") else None,  # 千元 → 元
                }
                stmt = pg_insert(IndexDaily).values(**insert_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["index_code", "trade_date"],
                    set_=insert_data,
                )
                with engine.begin() as conn:
                    conn.execute(stmt)

            logger.info(f"Index {ts_code} upserted for {tushare_date}")

        except Exception as e:
            logger.warning(f"Failed to fetch index {ts_code}: {e}")


def _backfill_single_day(target_date: date, token: str) -> tuple[bool, int, str | None]:
    """单日补全逻辑（不写 Parquet，不写 cron_log）。

    流程：
      1. 从 Tushare 获取当天全市场数据（单日）
      2. 逐股票处理：
         - 从 DB 读取该股票最新 65 条记录
         - 追加新数据，构造 66 行 DataFrame
         - 调用 IndicatorCalculator.calculate_all() 计算指标
         - 提取目标日期那行，upsert 到 DB
      3. 构建缓存

    Returns:
        (success, rows_written, error_message)
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy import text
    from data.db.models import StockDaily
    from data.db.connection import get_engine

    client = TushareClient(token)
    tushare_date = target_date.strftime("%Y%m%d")
    engine = get_engine()

    try:
        # ── 1. 获取当天全市场数据 ────────────────────────────────────────
        daily_df = client.get_daily_all(tushare_date)
        if daily_df.empty:
            return False, 0, f"Tushare daily returned empty for {tushare_date}"

        basic_df = client.get_daily_basic_all(tushare_date)
        adj_df = client.get_adj_factor_all(tushare_date)

        if basic_df.empty:
            logger.warning(f"daily_basic returned empty for {tushare_date}")
        if adj_df.empty:
            logger.warning(f"adj_factor returned empty for {tushare_date}")

        # Tushare: vol=手 → volume=股, amount=千元 → turnover_amount=元
        daily_df = daily_df.rename(columns={'vol': 'vol_raw', 'amount': 'amount_raw'})
        daily_df['volume'] = daily_df['vol_raw'] * 100
        daily_df['turnover_amount'] = daily_df['amount_raw'] * 1000
        daily_df = daily_df.drop(columns=['vol_raw', 'amount_raw'])

        # ts_code → stock_code
        daily_df["stock_code"] = daily_df["ts_code"].apply(_convert_ts_code)
        basic_df["stock_code"] = basic_df["ts_code"].apply(_convert_ts_code)
        adj_df["stock_code"] = adj_df["ts_code"].apply(_convert_ts_code)

        # ── 2. 合并数据 ─────────────────────────────────────────────────
        column_map = {
            "turnover_rate": "turnover_rate",
            "volume_ratio": "volume_ratio",
            "total_mv": "total_mv",
            "pe_ttm": "pe_ttm",
            "ps_ttm": "ps_ttm",
            "pcf_ttm": "pcf_ttm",
            "pb": "pb",
            "circulating_mv": "circulating_mv",
            "float_mv": "circulating_mv",
            "circ_mv": "circulating_mv",
        }
        ts_to_db = {k: v for k, v in column_map.items() if k in basic_df.columns}
        select_cols = ["stock_code"] + list(ts_to_db.keys())
        merged = daily_df.merge(
            basic_df[select_cols].rename(columns=ts_to_db),
            on="stock_code", how="left",
        )

        # 复权因子：取当天（adj_factor 是累计复权因子，当天的值即为当前复权因子）
        adj_latest = adj_df[adj_df["trade_date"] == tushare_date].groupby(
            "stock_code", sort=False
        ).last().reset_index()
        merged = merged.merge(
            adj_latest[["stock_code", "adj_factor"]],
            on="stock_code", how="left",
        )

        # adj_close = close * adj_factor
        merged["adj_close"] = merged["close"] * merged["adj_factor"]
        # amplitude / limit_up / limit_down
        merged["amplitude"] = (merged["high"] - merged["low"]) / merged["low"] * 100
        merged["change_pct"] = merged["pct_chg"]
        merged["limit_up"] = merged["pct_chg"] >= 9.9
        merged["limit_down"] = merged["pct_chg"] <= -9.9

        trade_date_str = target_date.strftime("%Y-%m-%d")

        # ── 3. 逐股票：读历史 → 追加新数据 → 计算指标 → upsert ─────────
        upsert_count = 0
        for _, new_row in merged.iterrows():
            stock_code = new_row["stock_code"]

            # 3.1 从 DB 读取该股票最新 65 条基础数据（不读指标列，由 calculate_all 重算）
            with engine.connect() as conn:
                hist = pd.read_sql(
                    text("""
                        SELECT trade_date, open, high, low, close, volume,
                               turnover_amount, adj_close
                        FROM stock_daily
                        WHERE stock_code = :code
                        ORDER BY trade_date DESC
                        LIMIT 65
                    """),
                    conn,
                    params={"code": stock_code},
                )

            # 3.2 构造 66 行 DataFrame（65 条历史 + 1 条新数据）
            # 只设基础列；指标列由 IndicatorCalculator.calculate_all 全部重新计算
            # prev_adj_close = 前一日的 adj_close（用于 DB 存储字段）
            prev_adj = hist.iloc[0]["adj_close"] if len(hist) > 0 else None
            # 确保 trade_date 类型一致：hist 来自 SQL 是 datetime.date，统一转 Timestamp
            hist["trade_date"] = pd.to_datetime(hist["trade_date"])
            new_record = {
                "trade_date": pd.to_datetime(trade_date_str),
                "open": new_row["open"],
                "high": new_row["high"],
                "low": new_row["low"],
                "close": new_row["close"],
                "volume": new_row["volume"],
                "turnover_amount": new_row["turnover_amount"],
                "adj_close": new_row["adj_close"],
                "prev_adj_close": prev_adj,
            }

            df66 = pd.concat(
                [hist, pd.DataFrame([new_record])],
                ignore_index=True,
            )
            df66 = df66.sort_values("trade_date").reset_index(drop=True)

            # 3.3 计算技术指标（在整个 66 行窗口上）
            df66 = IndicatorCalculator.calculate_all(df66)

            # 3.4 提取目标日期那行，写入 DB
            target_row = df66[df66["trade_date"] == pd.to_datetime(trade_date_str)]
            if target_row.empty:
                logger.warning(f"No record for {trade_date_str} after calculation: {stock_code}")
                continue

            record = target_row.iloc[0].to_dict()

            for col in list(record.keys()):
                if col in ("stock_code", "trade_date"):
                    continue
                record[col] = _na_to_none(record.get(col))

            insert_data = {
                k: v for k, v in record.items()
                if hasattr(StockDaily, k)
            }
            insert_data["stock_code"] = stock_code
            insert_data["trade_date"] = trade_date_str

            stmt = pg_insert(StockDaily).values(**insert_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["stock_code", "trade_date"],
                set_=insert_data,
            )
            with engine.begin() as conn:
                conn.execute(stmt)

            upsert_count += 1

        logger.info(f"DB upsert complete: {upsert_count} stocks")

        # ── 4. 构建缓存：从 DB 查询当天数据，写入 Parquet ───────────────
        DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # 从 DB 读取当天全市场数据（带完整指标）
        with engine.connect() as conn:
            cache_df = pd.read_sql(
                text("""
                    SELECT stock_code, trade_date, open, high, low, close, volume,
                           turnover_amount, adj_close, prev_adj_close, amplitude, change_pct,
                           turnover_rate, volume_ratio, circulating_mv, total_mv,
                           limit_up, limit_down, pe_ttm, ps_ttm, pcf_ttm, pb,
                           ma_5, ma_10, ma_20, ma_30, ma_60,
                           macd_dif, macd_dea, macd_hist,
                           kdj_k, kdj_d, kdj_j,
                           boll_mid, boll_upper, boll_lower,
                           psy, psyma, rsi_1, rsi_2, rsi_3
                    FROM stock_daily
                    WHERE trade_date = :td
                    ORDER BY stock_code
                """),
                conn,
                params={"td": trade_date_str},
            )

        if not cache_df.empty:
            cache_df["trade_date"] = pd.to_datetime(cache_df["trade_date"]).dt.strftime("%Y-%m-%d")
            parquet_path = DAILY_CACHE_DIR / f"{trade_date_str}.parquet"
            cache_df.to_parquet(parquet_path, index=False)
            logger.info(f"Parquet written: {parquet_path} ({len(cache_df)} stocks)")

            # 更新 trading_dates.parquet
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

        # ── 5. 获取指数数据 ───────────────────────────────────────────
        _backfill_index_data(client, target_date)

        return True, upsert_count, None

    except Exception as e:
        logger.exception(f"_backfill_single_day failed: {e}")
        return False, 0, str(e)


if __name__ == "__main__":
    main()
