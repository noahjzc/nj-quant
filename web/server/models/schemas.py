from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime


class SignalOut(BaseModel):
    id: int
    trade_date: date
    stock_code: str
    stock_name: Optional[str]
    direction: str
    target_pct: Optional[float]
    price_low: Optional[float]
    price_high: Optional[float]
    signal_reason: Optional[str]
    status: str
    executed_price: Optional[float]
    confirmed_at: Optional[datetime]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class SignalConfirm(BaseModel):
    executed_price: float


class PositionOut(BaseModel):
    id: int
    stock_code: str
    stock_name: Optional[str]
    buy_date: date
    buy_price: float
    shares: int
    sell_date: Optional[date]
    sell_price: Optional[float]
    profit_pct: Optional[float]
    status: str

    class Config:
        from_attributes = True


class CapitalLedgerOut(BaseModel):
    id: int
    event_type: str
    amount: float
    balance_after: float
    related_signal_id: Optional[int]
    note: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class DepositRequest(BaseModel):
    amount: float
    note: Optional[str] = None


class CronLogOut(BaseModel):
    id: int
    task_name: str
    status: str
    started_at: datetime
    finished_at: Optional[datetime]
    error_message: Optional[str]
    metadata: Optional[dict]

    class Config:
        from_attributes = True


class AssetOverview(BaseModel):
    total_asset: float
    available_cash: float
    position_value: float
    total_profit: float
    total_profit_pct: float


class StockDailyOut(BaseModel):
    stock_code: str
    trade_date: date
    stock_name: Optional[str]
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: Optional[float]
    change_pct: Optional[float]
    turnover_rate: Optional[float]
    pe_ttm: Optional[float]
    pb: Optional[float]
    circulating_mv: Optional[float]

    class Config:
        from_attributes = True