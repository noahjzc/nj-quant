from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List, Optional

from back_testing.data.db.connection import get_session as _get_session
from web.server.models.schemas import SignalOut, SignalConfirm

router = APIRouter()


def get_session():
    session = _get_session()()
    try:
        yield session
    finally:
        session.close()


@router.get("/", response_model=List[SignalOut])
def list_signals(
    trade_date: Optional[str] = None,
    status: Optional[str] = None,
    session: Session = Depends(get_session),
):
    """List signals, optionally filtered by date and status."""
    query = "SELECT * FROM daily_signal WHERE 1=1"
    params = {}
    if trade_date:
        query += " AND trade_date = :date"
        params['date'] = trade_date
    if status:
        query += " AND status = :status"
        params['status'] = status
    query += " ORDER BY created_at DESC LIMIT 200"

    result = session.execute(text(query), params)
    return [dict(row._mapping) for row in result]


@router.post("/{signal_id}/confirm")
def confirm_signal(
    signal_id: int,
    body: SignalConfirm,
    session: Session = Depends(get_session),
):
    """Confirm execution with actual price. Updates capital ledger and positions."""
    signal = session.execute(
        text("SELECT * FROM daily_signal WHERE id = :id"),
        {'id': signal_id}
    ).fetchone()
    if not signal:
        raise HTTPException(404, "Signal not found")
    if signal.status != 'pending':
        raise HTTPException(400, f"Signal already {signal.status}")

    s = dict(signal._mapping)

    balance_row = session.execute(
        text("SELECT balance_after FROM capital_ledger ORDER BY id DESC LIMIT 1")
    ).fetchone()
    current_balance = float(balance_row[0]) if balance_row else 0

    if s['direction'] == 'BUY':
        buy_amount = current_balance * (float(s['target_pct'] or 5) / 100)
        shares = int(buy_amount / body.executed_price / 100) * 100
        if shares <= 0:
            raise HTTPException(400, "Insufficient funds for even 1 lot")

        cost = shares * body.executed_price
        new_balance = current_balance - cost

        session.execute(
            text("""
                INSERT INTO position (stock_code, stock_name, buy_date, buy_price, shares)
                VALUES (:code, :name, :date, :price, :shares)
            """),
            {'code': s['stock_code'], 'name': s['stock_name'], 'date': s['trade_date'],
             'price': body.executed_price, 'shares': shares}
        )

        session.execute(
            text("""
                INSERT INTO capital_ledger (event_type, amount, balance_after, related_signal_id)
                VALUES ('BUY', :amount, :balance, :sid)
            """),
            {'amount': -cost, 'balance': new_balance, 'sid': signal_id}
        )

        session.execute(
            text("UPDATE daily_signal SET status='confirmed', executed_price=:price, confirmed_at=NOW() WHERE id=:id"),
            {'price': body.executed_price, 'id': signal_id}
        )

    elif s['direction'] == 'SELL':
        pos = session.execute(
            text("SELECT * FROM position WHERE stock_code=:code AND status='OPEN' ORDER BY buy_date DESC LIMIT 1"),
            {'code': s['stock_code']}
        ).fetchone()
        if not pos:
            raise HTTPException(400, "No open position found for this stock")

        p = dict(pos._mapping)
        revenue = p['shares'] * body.executed_price
        new_balance = current_balance + revenue
        profit_pct = (body.executed_price - float(p['buy_price'])) / float(p['buy_price']) * 100

        session.execute(
            text("""
                UPDATE position SET sell_date=:date, sell_price=:price,
                profit_pct=:profit, status='CLOSED' WHERE id=:id
            """),
            {'date': s['trade_date'], 'price': body.executed_price,
             'profit': profit_pct, 'id': p['id']}
        )

        session.execute(
            text("""
                INSERT INTO capital_ledger (event_type, amount, balance_after, related_signal_id)
                VALUES ('SELL', :amount, :balance, :sid)
            """),
            {'amount': revenue, 'balance': new_balance, 'sid': signal_id}
        )

        session.execute(
            text("UPDATE daily_signal SET status='confirmed', executed_price=:price, confirmed_at=NOW() WHERE id=:id"),
            {'price': body.executed_price, 'id': signal_id}
        )

    session.commit()
    return {"ok": True}


@router.post("/{signal_id}/skip")
def skip_signal(signal_id: int, session: Session = Depends(get_session)):
    """Mark a signal as skipped."""
    session.execute(
        text("UPDATE daily_signal SET status='skipped', confirmed_at=NOW() WHERE id=:id"),
        {'id': signal_id}
    )
    session.commit()
    return {"ok": True}
