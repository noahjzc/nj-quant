from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List

from back_testing.data.db.connection import get_session as _get_session
from web.server.models.schemas import (
    PositionOut, CapitalLedgerOut, AssetOverview, DepositRequest
)

router = APIRouter()


def get_session():
    session = _get_session()()
    try:
        yield session
    finally:
        session.close()


@router.get("/overview", response_model=AssetOverview)
def asset_overview(session: Session = Depends(get_session)):
    balance = session.execute(
        text("SELECT balance_after FROM capital_ledger ORDER BY id DESC LIMIT 1")
    ).fetchone()
    available_cash = float(balance[0]) if balance else 0

    positions = session.execute(
        text("SELECT shares, buy_price FROM position WHERE status='OPEN'")
    ).fetchall()
    position_value = sum(float(p.shares) * float(p.buy_price) for p in positions)

    total_asset = available_cash + position_value

    profit = session.execute(
        text("SELECT SUM(profit_pct * shares * buy_price / 100) FROM position WHERE status='CLOSED'")
    ).fetchone()[0] or 0

    return AssetOverview(
        total_asset=total_asset,
        available_cash=available_cash,
        position_value=position_value,
        total_profit=float(profit),
        total_profit_pct=(float(profit) / 100000 * 100) if profit else 0,
    )


@router.get("/", response_model=List[PositionOut])
def list_positions(
    status: str = None,
    session: Session = Depends(get_session),
):
    query = "SELECT * FROM position"
    params = {}
    if status:
        query += " WHERE status = :status"
        params['status'] = status
    query += " ORDER BY buy_date DESC LIMIT 100"

    result = session.execute(text(query), params)
    return [dict(row._mapping) for row in result]


@router.get("/capital", response_model=List[CapitalLedgerOut])
def capital_history(session: Session = Depends(get_session)):
    result = session.execute(
        text("SELECT * FROM capital_ledger ORDER BY id DESC LIMIT 50")
    )
    return [dict(row._mapping) for row in result]


@router.post("/capital/deposit")
def deposit(body: DepositRequest, session: Session = Depends(get_session)):
    balance_row = session.execute(
        text("SELECT balance_after FROM capital_ledger ORDER BY id DESC LIMIT 1")
    ).fetchone()
    current = float(balance_row[0]) if balance_row else 0
    new_balance = current + body.amount

    session.execute(
        text("""
            INSERT INTO capital_ledger (event_type, amount, balance_after, note)
            VALUES ('DEPOSIT', :amount, :balance, :note)
        """),
        {'amount': body.amount, 'balance': new_balance, 'note': body.note}
    )
    session.commit()
    return {"ok": True, "balance_after": new_balance}
