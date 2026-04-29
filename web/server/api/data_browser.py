from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import Optional

from back_testing.data.db.connection import get_session as _get_session

router = APIRouter()


def get_session():
    session = _get_session()()
    try:
        yield session
    finally:
        session.close()


@router.get("/stocks")
def list_stocks(
    search: Optional[str] = Query(None, description="代码/名称搜索"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
):
    """Paginated stock list."""
    offset = (page - 1) * page_size
    where = ""
    params = {'limit': page_size, 'offset': offset}
    if search:
        where = "WHERE stock_code ILIKE :search OR stock_name ILIKE :search2"
        params['search'] = f"%{search}%"
        params['search2'] = f"%{search}%"

    total = session.execute(
        text(f"SELECT COUNT(DISTINCT stock_code) FROM stock_daily {where}"), params
    ).fetchone()[0]

    rows = session.execute(
        text(f"""
            SELECT DISTINCT ON (stock_code) stock_code, stock_name, industry, close, change_pct,
                   turnover_rate, pe_ttm, pb, circulating_mv, trade_date
            FROM stock_daily {where}
            ORDER BY stock_code, trade_date DESC
            LIMIT :limit OFFSET :offset
        """),
        params
    ).fetchall()

    return {
        'total': total,
        'page': page,
        'data': [dict(r._mapping) for r in rows],
    }


@router.get("/stocks/{stock_code}")
def get_stock_detail(
    stock_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    session: Session = Depends(get_session),
):
    """Get daily data for a specific stock."""
    query = "SELECT * FROM stock_daily WHERE stock_code = :code"
    params = {'code': stock_code}
    if start_date:
        query += " AND trade_date >= :start"
        params['start'] = start_date
    if end_date:
        query += " AND trade_date <= :end"
        params['end'] = end_date
    query += " ORDER BY trade_date ASC LIMIT 500"

    result = session.execute(text(query), params)
    return [dict(r._mapping) for r in result]
