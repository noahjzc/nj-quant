"""
预测者网历史数据补全脚本

用法:
    # 补全所有缺失数据（扫描全量CSV）
    python back_testing/data/sync/backfill_overview.py --data-dir D:/path/to/stock/csv

    # 只补特定日期区间
    python back_testing/data/sync/backfill_overview.py --data-dir D:/path/to/stock/csv --start 2025-04-01 --end 2025-04-30

    # 补单只股票（调试用）
    python back_testing/data/sync/backfill_overview.py --data-dir D:/path/to/stock/csv --stock sh600519
"""
import argparse
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from back_testing.data.db.connection import get_engine
from back_testing.data.db.models import StockDaily
from sqlalchemy.dialects.postgresql import insert as pg_insert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backfill_overview")


# ── 字段映射：CSV列名 → DB列名 ────────────────────────────────
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


def _process_csv(filepath: str) -> pd.DataFrame:
    """解析单个股票CSV文件，返回处理好的DataFrame。"""
    df = pd.read_csv(filepath, encoding="gbk")

    # 重命名列
    df = df.rename(columns=CSV_TO_DB)

    # 过滤不需要的列（不在映射中）
    keep_cols = [c for c in df.columns if c in CSV_TO_DB.values()]
    df = df[keep_cols]

    # stock_code 已是 sh600000 格式（直接可用）
    df["stock_code"] = df["stock_code"].astype(str).str.strip()

    # trade_date: 字符串 "2025-04-23" → date
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    # limit_up / limit_down: "是" → True, 其他 → False
    if "limit_up" in df.columns:
        df["limit_up"] = df["limit_up"].apply(
            lambda x: str(x).strip() == "是" if pd.notna(x) else False
        )
    if "limit_down" in df.columns:
        df["limit_down"] = df["limit_down"].apply(
            lambda x: str(x).strip() == "是" if pd.notna(x) else False
        )

    # 空字符串 → None
    df = df.replace("", None)

    return df


def _get_db_max_date(engine, stock_code: str) -> date | None:
    """查询某股票在DB中的最大日期。"""
    with engine.connect() as conn:
        result = conn.execute(
            pd.text("""
                SELECT MAX(trade_date) FROM stock_daily WHERE stock_code = :code
            """),
            {"code": stock_code},
        )
        row = result.fetchone()
        return row[0] if row and row[0] else None


def _na_to_none(val):
    """Convert NaN/inf to None for PostgreSQL."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if pd.isna(val) or val != val:  # NaN check
            return None
        if abs(val) == float("inf"):
            return None
    if pd.isna(val):
        return None
    return val


def backfill_stock(
    filepath: str,
    engine,
    start_date: date | None = None,
    end_date: date | None = None,
) -> int:
    """对单只股票补全数据，返回写入行数。"""
    stock_code = Path(filepath).stem  # 文件名即 stock_code
    df = _process_csv(filepath)

    # 按日期过滤
    if start_date:
        df = df[df["trade_date"] >= start_date]
    if end_date:
        df = df[df["trade_date"] <= end_date]

    if df.empty:
        return 0

    # 检查DB最新日期，只补缺失部分
    db_max = _get_db_max_date(engine, stock_code)
    if db_max is not None:
        df = df[df["trade_date"] > db_max]

    if df.empty:
        return 0

    # upsert
    records = df.to_dict("records")
    n = 0
    with engine.begin() as conn:
        for record in records:
            trade_date = record.pop("trade_date")
            insert_data = {k: _na_to_none(v) for k, v in record.items() if hasattr(StockDaily, k)}
            insert_data["stock_code"] = stock_code
            insert_data["trade_date"] = trade_date

            stmt = pg_insert(StockDaily).values(**insert_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["stock_code", "trade_date"],
                set_=insert_data,
            )
            conn.execute(stmt)
            n += 1

    return n


def main():
    parser = argparse.ArgumentParser(description="预测者网历史数据补全")
    parser.add_argument("--data-dir", required=True, help="CSV文件所在目录")
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--stock", default=None, help="仅处理指定股票（如 sh600519）")
    parser.add_argument("--dry-run", action="store_true", help="仅扫描，不写入")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"目录不存在: {data_dir}")
        sys.exit(1)

    start_date = pd.to_datetime(args.start).date() if args.start else None
    end_date = pd.to_datetime(args.end).date() if args.end else None

    engine = get_engine()

    # 收集所有CSV文件
    csv_files = sorted(data_dir.glob("*.csv"))
    if args.stock:
        csv_files = [f for f in csv_files if f.stem == args.stock]
        if not csv_files:
            logger.error(f"未找到股票 {args.stock} 的CSV文件")
            sys.exit(1)

    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    if start_date or end_date:
        logger.info(f"日期范围: {start_date or '开始'} ~ {end_date or '结束'}")

    total_upsert = 0
    total_files = 0
    skipped = 0

    for i, filepath in enumerate(csv_files):
        stock_code = filepath.stem

        # 先查DB最新日期
        db_max = _get_db_max_date(engine, stock_code)

        if args.dry_run:
            df_sample = pd.read_csv(filepath, encoding="gbk", nrows=2)
            df_sample = df_sample.rename(columns=CSV_TO_DB)
            file_min = pd.to_datetime(df_sample["交易日期"].iloc[-1]).date()
            file_max = pd.to_datetime(df_sample["交易日期"].iloc[0]).date()
            logger.info(
                f"[dry-run] {stock_code}: DB最新={db_max}, "
                f"文件范围={file_min}~{file_max}"
            )
            continue

        try:
            n = backfill_stock(str(filepath), engine, start_date, end_date)
            if n > 0:
                total_upsert += n
                total_files += 1
                logger.info(f"  {stock_code}: 写入 {n} 条")
            else:
                skipped += 1
        except Exception as e:
            logger.error(f"  {stock_code}: 失败 - {e}")

        if (i + 1) % 500 == 0:
            logger.info(f"进度: {i + 1}/{len(csv_files)}")

    logger.info("=" * 50)
    if args.dry_run:
        logger.info(f"Dry run 完成: {len(csv_files)} 个文件扫描完毕")
    else:
        logger.info(f"补全完成: {total_files} 只股票写入, {skipped} 只无需更新, 共 {total_upsert} 条新记录")


if __name__ == "__main__":
    main()
