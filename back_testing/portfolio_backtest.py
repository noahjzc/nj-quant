import pandas as pd
import numpy as np
from back_testing.core.backtest_engine import BacktestEngine

class PortfolioBacktest:
    """
    组合回测器
    每天选股，等权分配资金，回测整个组合表现
    """

    def __init__(self, strategy_class, stock_pool, initial_capital=1000000.0, start_date=None):
        self.strategy_class = strategy_class
        self.stock_pool = stock_pool  # List of (code, name, industry)
        self.initial_capital = initial_capital
        self.per_stock_capital = initial_capital / len(stock_pool)
        self.start_date = start_date

    def run(self):
        """运行组合回测"""
        results = []
        for code, name, industry in self.stock_pool:
            engine = self.strategy_class(
                stock_code=code,
                initial_capital=self.per_stock_capital,
                start_date=self.start_date
            )
            result = engine.run()
            results.append(result)

        # Aggregate results
        total_return = sum(r['final_capital'] for r in results) / self.initial_capital - 1
        total_trades = sum(r['total_trades'] for r in results)
        avg_win_rate = sum(r['win_rate'] for r in results) / len(results)

        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'avg_win_rate': avg_win_rate,
            'individual_results': results
        }