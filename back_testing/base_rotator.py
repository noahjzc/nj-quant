"""Base rotator — template method for weekly rebalance workflows."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


class BaseRotator(ABC):
    """轮动基类：提供调仓骨架和状态管理，选股逻辑由子类实现。

    模板方法: run_weekly(date, prices) → {'date', 'stocks', 'rebalance'}
    子类必须实现: select_stocks(date) → List[str]
    """

    def __init__(self, initial_capital: float = 1_000_000.0, n_stocks: int = 5):
        self.initial_capital = initial_capital
        self.n_stocks = n_stocks
        self.per_stock_capital = initial_capital / n_stocks

        self.current_stocks: List[str] = []
        self.current_positions: Dict[str, dict] = {}  # {code: {'shares', 'buy_price'}}
        self.strategy_name: str = self.__class__.__name__

    @abstractmethod
    def select_stocks(self, date: pd.Timestamp, **kwargs) -> List[str]:
        """子类实现各自的选股逻辑。"""

    def rebalance(self, date: pd.Timestamp,
                  prices: Optional[Dict[str, float]] = None) -> dict:
        """统一的调仓逻辑：卖出→买入。

        Args:
            date: 调仓日期
            prices: 可选的价格字典，传入时会计算具体买入股数。

        Returns:
            dict: {'date', 'sell_stocks': [], 'buy_stocks': []}
        """
        rebalance_detail = {
            'date': date,
            'sell_stocks': [],
            'buy_stocks': [],
        }

        # 卖出不在新持仓列表中的股票
        for code in list(self.current_positions.keys()):
            if code not in self.current_stocks:
                rebalance_detail['sell_stocks'].append(code)
                del self.current_positions[code]

        # 买入新持仓列表中的股票
        for code in self.current_stocks:
            if code not in self.current_positions:
                rebalance_detail['buy_stocks'].append(code)
                if prices and code in prices and prices[code] > 0:
                    shares = int(self.per_stock_capital / prices[code])
                else:
                    shares = 0
                self.current_positions[code] = {
                    'shares': shares,
                    'buy_price': prices.get(code, 0) if prices else 0,
                }

        return rebalance_detail

    def calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """计算当前持仓市值（供子类/调用方使用）。"""
        total = 0.0
        for code, pos in self.current_positions.items():
            if pos['shares'] > 0 and code in prices:
                total += pos['shares'] * prices[code]
        return total

    def run_weekly(self, date: pd.Timestamp,
                   prices: Optional[Dict[str, float]] = None) -> dict:
        """每周流程模板：选股 → 调仓。

        Args:
            date: 当前日期（周五）
            prices: 可选的价格映射，传给 rebalance。

        Returns:
            dict: {'date', 'stocks', 'rebalance'}
        """
        self.select_stocks(date)
        rebalance = self.rebalance(date, prices)

        return {
            'date': date,
            'strategy': self.strategy_name,
            'stocks': self.current_stocks,
            'rebalance': rebalance,
        }
