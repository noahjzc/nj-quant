from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List

from data.db.connection import get_session as _get_session
from web.server.models.schemas import CronLogOut

router = APIRouter()


def get_session():
    session = _get_session()()
    try:
        yield session
    finally:
        session.close()


@router.get("/", response_model=List[CronLogOut])
def list_cron_logs(session: Session = Depends(get_session)):
    result = session.execute(
        text("SELECT *, metadata::text as metadata FROM cron_log ORDER BY id DESC LIMIT 100")
    )
    return [dict(row._mapping) for row in result]


@router.get("/status")
def data_completeness(session: Session = Depends(get_session)):
    """Quick check: when was data last updated?"""
    last_daily = session.execute(
        text("SELECT MAX(trade_date) FROM stock_daily")
    ).fetchone()[0]

    last_signal = session.execute(
        text("SELECT MAX(trade_date) FROM daily_signal")
    ).fetchone()[0]

    last_backfill = session.execute(
        text("SELECT status, finished_at FROM cron_log WHERE task_name='night_backfill' ORDER BY id DESC LIMIT 1")
    ).fetchone()

    return {
        'last_daily_date': str(last_daily) if last_daily else None,
        'last_signal_date': str(last_signal) if last_signal else None,
        'last_backfill': {
            'status': last_backfill.status if last_backfill else None,
            'finished_at': str(last_backfill.finished_at) if last_backfill else None,
        }
    }
