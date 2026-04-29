# signal_pipeline/intraday_signal.py
"""
Intraday signal generation script (cron: 25 14 * * 1-5).

1. Fetch real-time spot data via AKShare
2. Load yesterday's complete data from PostgreSQL
3. Merge + recalculate indicators
4. Generate buy/sell signals
5. Write signals to DB
"""
import sys
import logging
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from back_testing.data.db.connection import get_engine, get_session
from back_testing.rotation.config import RotationConfig
from signal_pipeline.data_sources.akshare_client import AKShareClient
from signal_pipeline.data_merger import DataMerger
from signal_pipeline.indicator_calculator import IndicatorCalculator
from signal_pipeline.signal_generator import SignalGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def _cron_start(task_name: str, session):
    """Insert a 'running' cron_log entry, return the log id."""
    result = session.execute(
        text("INSERT INTO cron_log (task_name, status) VALUES (:name, 'running') RETURNING id"),
        {'name': task_name}
    )
    session.commit()
    return result.fetchone()[0]


def _cron_finish(log_id: int, status: str, session, error: str = None, metadata: dict = None):
    session.execute(
        text("UPDATE cron_log SET status=:status, finished_at=NOW(), error_message=:err, metadata=:meta WHERE id=:id"),
        {'status': status, 'err': error, 'meta': str(metadata) if metadata else None, 'id': log_id}
    )
    session.commit()


def main():
    logger.info("=" * 50)
    logger.info("盘中信号生成开始")
    logger.info("=" * 50)

    Session = get_session()
    session = Session()
    log_id = _cron_start('intraday_signal', session)

    try:
        # 1. Fetch intraday snapshot
        logger.info("Step 1/5: 获取 AKShare 实时快照...")
        akshare = AKShareClient(max_retries=3, retry_delay=60)
        intraday_df = akshare.get_spot_all()
        logger.info(f"获取到 {len(intraday_df)} 只股票")

        # 2. Load history from DB (last 60 days for all stocks)
        logger.info("Step 2/5: 加载历史数据...")
        today = date.today()
        start_date = today - timedelta(days=60)
        hist_df = pd.read_sql(
            text("SELECT * FROM stock_daily WHERE trade_date >= :start"),
            get_engine(),
            params={'start': start_date}
        )
        logger.info(f"历史数据: {len(hist_df)} 行")

        # 3. Merge
        logger.info("Step 3/5: 合并日内+历史数据...")
        today_ts = pd.Timestamp(today)
        merged_df = DataMerger.merge(hist_df, intraday_df, today_ts)
        logger.info(f"合并后: {len(merged_df)} 行")

        # 4. Calculate indicators
        logger.info("Step 4/5: 重算技术指标...")
        merged_df = IndicatorCalculator.calculate_all(merged_df)
        logger.info("指标计算完成")

        # 5. Generate signals
        logger.info("Step 5/5: 生成交易信号...")
        config = RotationConfig()
        generator = SignalGenerator(config)

        # Get current positions for sell signal checking
        position_codes = [
            r[0] for r in session.execute(
                text("SELECT stock_code FROM position WHERE status='OPEN'")
            ).fetchall()
        ]
        logger.info(f"当前持仓: {position_codes}")

        buy_codes = generator.generate_buy_signals(merged_df, today_ts)
        sell_signals = generator.generate_sell_signals(merged_df, today_ts, position_codes)

        logger.info(f"买入信号: {len(buy_codes)} 只")
        logger.info(f"卖出信号: {len(sell_signals)} 只")

        # 6. Write signals to DB
        today_df = merged_df[merged_df['trade_date'] == today_ts]
        today_lookup = today_df.set_index('stock_code') if not today_df.empty else pd.DataFrame()

        # Clear today's previous signals (if re-running)
        session.execute(
            text("DELETE FROM daily_signal WHERE trade_date = :d"),
            {'d': today}
        )

        signal_count = 0

        # Write buy signals
        for i, code in enumerate(buy_codes):
            if not today_lookup.empty and code in today_lookup.index:
                row = today_lookup.loc[code]
                name = row.get('stock_name', code)
                price = float(row['close'])
            else:
                name = code
                price = 0.0

            session.execute(
                text("""
                    INSERT INTO daily_signal
                    (trade_date, stock_code, stock_name, direction, target_pct,
                     price_low, price_high, signal_reason, status)
                    VALUES (:d, :code, :name, 'BUY', :pct, :lo, :hi, :reason, 'pending')
                """),
                {
                    'd': today, 'code': code, 'name': str(name),
                    'pct': float(config.max_position_pct * 100),
                    'lo': price * 0.985, 'hi': price * 1.015,
                    'reason': f'多因子排名 #{i + 1}'
                }
            )
            signal_count += 1

        # Write sell signals
        for s in sell_signals:
            code = s['stock_code']
            session.execute(
                text("""
                    INSERT INTO daily_signal
                    (trade_date, stock_code, stock_name, direction, target_pct,
                     price_low, price_high, signal_reason, status)
                    VALUES (:d, :code, :code, 'SELL', NULL, NULL, NULL, :reason, 'pending')
                """),
                {'d': today, 'code': code, 'reason': s['reason']}
            )
            signal_count += 1

        session.commit()

        _cron_finish(log_id, 'success', session, metadata={
            'buy_count': len(buy_codes),
            'sell_count': len(sell_signals),
            'total_signals': signal_count,
            'intraday_stocks': len(intraday_df),
        })
        logger.info(f"信号写入完成: {signal_count} 条")
        logger.info("盘中信号生成完成")

    except Exception as e:
        logger.error(f"盘中信号生成失败: {e}", exc_info=True)
        _cron_finish(log_id, 'failed', session, error=str(e))
        sys.exit(1)
    finally:
        session.close()


if __name__ == '__main__':
    main()
