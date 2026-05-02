from typing import Tuple, List
import pandas as pd


class MarketConstraints:
    LIMIT_UP_THRESHOLD = 9.5
    LIMIT_DOWN_THRESHOLD = -9.5
    MIN_AMOUNT = 1_000_000  # 100万元

    def __init__(self, exclude_st: bool = True, exclude_limit_up: bool = True,
                 exclude_limit_down: bool = True, exclude_suspended: bool = True,
                 min_amount: float = MIN_AMOUNT):
        self.exclude_st = exclude_st
        self.exclude_limit_up = exclude_limit_up
        self.exclude_limit_down = exclude_limit_down
        self.exclude_suspended = exclude_suspended
        self.min_amount = min_amount

    def can_buy(self, stock_code: str, price: float, amount_today: float,
                pct_chg: float, is_st: bool, is_suspended: bool) -> Tuple[bool, str]:
        if self.exclude_st and is_st:
            return False, "ST"
        if self.exclude_suspended and is_suspended:
            return False, "停牌"
        if self.exclude_limit_up and pct_chg >= self.LIMIT_UP_THRESHOLD:
            return False, "涨停"
        if amount_today < self.min_amount:
            return False, f"成交额不足"
        return True, ""

    def can_sell(self, position, trade_date: pd.Timestamp,
                 price: float, pct_chg: float) -> Tuple[bool, str]:
        buy_ts = pd.Timestamp(position.buy_date)
        if trade_date == buy_ts:
            return False, "T+1约束"
        if self.exclude_limit_down and pct_chg <= self.LIMIT_DOWN_THRESHOLD:
            return False, "跌停"
        return True, ""

    def filter_pool(self, today_df: pd.DataFrame, date: pd.Timestamp) -> List[str]:
        df = today_df.copy()

        if 'stock_code' not in df.columns:
            return []

        if self.exclude_suspended and 'close' in df.columns:
            df = df[df['close'].notna() & (df['close'] > 0)]

        if self.exclude_st and 'is_st' in df.columns:
            df = df[~df['is_st'].astype(bool)]

        if 'pct_chg' in df.columns:
            if self.exclude_limit_up:
                df = df[df['pct_chg'] < self.LIMIT_UP_THRESHOLD]
            if self.exclude_limit_down:
                df = df[df['pct_chg'] > self.LIMIT_DOWN_THRESHOLD]

        if 'amount' in df.columns:
            df = df[df['amount'] >= self.min_amount]

        return df['stock_code'].tolist()
