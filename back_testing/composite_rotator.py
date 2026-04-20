import pandas as pd
import numpy as np
from back_testing.composite_selector import CompositeSelector
from back_testing.composite_scorer import CompositeScorer

class CompositeRotator:
    """
    综合评分轮动主控制器

    每周流程：
    1. 使用CompositeSelector对全市场股票计算综合评分
    2. 选取综合评分最高的5只股票
    3. 等权20%持仓
    """

    def __init__(self, data_path: str, initial_capital: float = 1000000.0,
                 n_stocks: int = 5):
        self.data_path = data_path
        self.initial_capital = initial_capital
        self.n_stocks = n_stocks
        self.per_stock_capital = initial_capital / n_stocks

        self.composite_selector = CompositeSelector(data_path)
        self.composite_scorer = CompositeScorer()

        self.current_stocks = []
        self.current_positions = {}

    def select_stocks(self, date: pd.Timestamp) -> list:
        """选取综合评分最高的N只股票"""
        print(f"\n使用综合评分策略筛选股票...")

        selected = self.composite_selector.select_top_stocks(
            n=self.n_stocks,
            date=date
        )

        self.current_stocks = selected
        return selected

    def rebalance(self, date: pd.Timestamp) -> dict:
        """执行调仓"""
        rebalance_detail = {
            'date': date,
            'sell_stocks': [],
            'buy_stocks': [],
            'strategy': 'CompositeScorer'
        }

        # 卖出不在新持仓列表中的股票
        for code in list(self.current_positions.keys()):
            if code not in self.current_stocks:
                rebalance_detail['sell_stocks'].append(code)
                del self.current_positions[code]

        # 买入新持仓股票
        for code in self.current_stocks:
            if code not in self.current_positions:
                rebalance_detail['buy_stocks'].append(code)
                self.current_positions[code] = {
                    'shares': 0,
                    'buy_price': 0
                }

        return rebalance_detail

    def run_weekly(self, date: pd.Timestamp) -> dict:
        """执行每周流程"""
        # 筛选股票
        self.select_stocks(date)

        # 执行调仓
        rebalance = self.rebalance(date)

        return {
            'date': date,
            'strategy': 'CompositeScorer',
            'stocks': self.current_stocks,
            'rebalance': rebalance
        }
