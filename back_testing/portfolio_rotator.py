import pandas as pd
import numpy as np
from back_testing.strategy_evaluator import StrategyEvaluator, STRATEGY_MAP
from back_testing.stock_selector import StockSelector
from back_testing.signal_scorer import SignalScorer


class PortfolioRotator:
    """
    策略轮动主控制器

    每周流程：
    1. 评估所有策略过去4周表现
    2. 选出最优策略
    3. 对全市场股票计算该策略信号强度
    4. 选取信号最强的5只股票
    5. 等权20%持仓
    """

    def __init__(self, data_path: str, initial_capital: float = 1000000.0,
                 n_stocks: int = 5, n_weeks: int = 4):
        self.data_path = data_path
        self.initial_capital = initial_capital
        self.n_stocks = n_stocks  # 持仓数量
        self.n_weeks = n_weeks    # 评估周期
        self.per_stock_capital = initial_capital / n_stocks

        self.strategy_evaluator = None
        self.stock_selector = StockSelector(data_path)
        self.scorer = SignalScorer()

        self.current_strategy = None
        self.current_stocks = []
        self.current_positions = {}  # {stock_code: {'shares': int, 'buy_price': float}}

        self.trade_log = []  # 交易记录
        self.portfolio_value_history = []  # 净值历史

    def select_best_strategy(self, reference_date: pd.Timestamp = None) -> str:
        """
        选出过去N周表现最好的策略
        """
        if reference_date is None:
            reference_date = pd.Timestamp.now()

        # 获取所有股票代码（取样本进行评估）
        sample_stocks = self.stock_selector.get_all_stock_codes()[:100]  # 用样本评估

        self.strategy_evaluator = StrategyEvaluator(
            stock_codes=sample_stocks,
            data_path=self.data_path,
            initial_capital=self.per_stock_capital
        )

        print(f"评估 {len(sample_stocks)} 只股票在过去{self.n_weeks}周的表现...")
        results = self.strategy_evaluator.evaluate_all_strategies(
            weeks=self.n_weeks,
            reference_date=reference_date
        )

        best_strategy = results.iloc[0]['strategy_name']
        best_return = results.iloc[0]['avg_return']

        print(f"\n策略排名（过去{self.n_weeks}周）：")
        print(results.to_string(index=False))
        print(f"\n选中策略: {best_strategy} (收益率: {best_return:.2%})")

        self.current_strategy = best_strategy
        return best_strategy

    def select_stocks(self, date: pd.Timestamp) -> list:
        """
        选取信号最强的N只股票
        """
        print(f"\n使用策略 {self.current_strategy} 筛选股票...")

        selected = self.stock_selector.select_top_stocks(
            strategy_name=self.current_strategy,
            n=self.n_stocks,
            date=date
        )

        self.current_stocks = selected
        return selected

    def rebalance(self, date: pd.Timestamp) -> dict:
        """
        执行调仓：卖出旧持仓，买入新持仓

        Returns:
            dict: 调仓详情
        """
        rebalance_detail = {
            'date': date,
            'sell_stocks': [],  # 卖出的股票
            'buy_stocks': [],    # 买入的股票
            'strategy': self.current_strategy
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

    def calculate_portfolio_value(self, prices: dict) -> float:
        """
        计算当前组合市值
        """
        total = 0
        for code, position in self.current_positions.items():
            if position['shares'] > 0 and code in prices:
                total += position['shares'] * prices[code]
        return total

    def run_weekly(self, date: pd.Timestamp) -> dict:
        """
        执行每周流程
        """
        # 1. 选择最优策略
        self.select_best_strategy(date)

        # 2. 筛选股票
        self.select_stocks(date)

        # 3. 执行调仓
        rebalance = self.rebalance(date)

        return {
            'date': date,
            'strategy': self.current_strategy,
            'stocks': self.current_stocks,
            'rebalance': rebalance
        }