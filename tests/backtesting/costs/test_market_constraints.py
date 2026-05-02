import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pytest
import pandas as pd
from backtesting.costs.market_constraints import MarketConstraints


class TestCanBuy:
    def setup_method(self):
        self.constraints = MarketConstraints()

    def test_normal_stock_allowed(self):
        ok, reason = self.constraints.can_buy('sh600000', price=10.0, amount_today=5e7,
                                               pct_chg=3.0, is_st=False, is_suspended=False)
        assert ok

    def test_limit_up_blocked(self):
        ok, reason = self.constraints.can_buy('sh600000', price=10.0, amount_today=5e7,
                                               pct_chg=9.95, is_st=False, is_suspended=False)
        assert not ok

    def test_st_blocked(self):
        ok, reason = self.constraints.can_buy('sh600000', price=10.0, amount_today=5e7,
                                               pct_chg=3.0, is_st=True, is_suspended=False)
        assert not ok

    def test_low_volume_blocked(self):
        ok, reason = self.constraints.can_buy('sh600000', price=10.0, amount_today=5e4,
                                               pct_chg=3.0, is_st=False, is_suspended=False)
        assert not ok


class TestCanSell:
    def setup_method(self):
        self.constraints = MarketConstraints()
        from dataclasses import dataclass

        @dataclass
        class Position:
            stock_code: str
            shares: int
            buy_price: float
            buy_date: str
            highest_price: float = 0.0

        self.Position = Position

    def test_normal_sell_allowed(self):
        pos = self.Position(stock_code='sh600000', shares=1000, buy_price=10.0, buy_date='2024-01-05')
        ok, reason = self.constraints.can_sell(pos, pd.Timestamp('2024-01-08'),
                                                price=11.0, pct_chg=5.0)
        assert ok

    def test_t1_blocked_same_day(self):
        pos = self.Position(stock_code='sh600000', shares=1000, buy_price=10.0, buy_date='2024-01-08')
        ok, reason = self.constraints.can_sell(pos, pd.Timestamp('2024-01-08'),
                                                price=11.0, pct_chg=5.0)
        assert not ok
        assert 'T+1' in reason

    def test_limit_down_blocked(self):
        pos = self.Position(stock_code='sh600000', shares=1000, buy_price=10.0, buy_date='2024-01-05')
        ok, reason = self.constraints.can_sell(pos, pd.Timestamp('2024-01-08'),
                                                price=9.0, pct_chg=-10.0)
        assert not ok


class TestFilterPool:
    def setup_method(self):
        self.constraints = MarketConstraints()

    def test_basic_filter(self):
        df = pd.DataFrame({
            'stock_code': ['sh001', 'sh002', 'sh003', 'sh004'],
            'is_st': [False, True, False, False],
            'pct_chg': [3.0, 5.0, 10.0, -10.0],
            'amount': [5e7, 1e8, 3e7, 2e7],
        })
        result = self.constraints.filter_pool(df, pd.Timestamp('2024-01-08'))
        assert result == ['sh001']
