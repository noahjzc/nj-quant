import pandas as pd

from back_testing.base_rotator import BaseRotator
from back_testing.strategy_evaluator import StrategyEvaluator, STRATEGY_MAP
from back_testing.selectors.stock_selector import StockSelector
from back_testing.signal_scorer import SignalScorer


class PortfolioRotator(BaseRotator):
    """
    策略轮动主控制器

    每周流程：
    1. 评估所有策略过去4周表现
    2. 选出最优策略
    3. 对全市场股票计算该策略信号强度
    4. 选取信号最强的5只股票
    5. 等权20%持仓
    """

    def __init__(self, initial_capital: float = 1000000.0,
                 n_stocks: int = 5, n_weeks: int = 4):
        super().__init__(initial_capital, n_stocks)
        self.n_weeks = n_weeks

        self.strategy_evaluator = None
        self.stock_selector = StockSelector()
        self.scorer = SignalScorer()

        self.current_strategy = None
        self.trade_log = []
        self.portfolio_value_history = []

    def select_best_strategy(self, reference_date: pd.Timestamp = None) -> str:
        """选出过去N周表现最好的策略"""
        if reference_date is None:
            reference_date = pd.Timestamp.now()

        # 获取所有股票代码（取样本进行评估）
        sample_stocks = self.stock_selector.get_all_stock_codes()[:100]

        self.strategy_evaluator = StrategyEvaluator(
            stock_codes=sample_stocks,
            initial_capital=self.per_stock_capital,
        )

        print(f"评估 {len(sample_stocks)} 只股票在过去{self.n_weeks}周的表现...")
        results = self.strategy_evaluator.evaluate_all_strategies(
            weeks=self.n_weeks,
            reference_date=reference_date,
        )

        best_strategy = results.iloc[0]['strategy_name']
        best_return = results.iloc[0]['avg_return']

        print(f"\n策略排名（过去{self.n_weeks}周）：")
        print(results.to_string(index=False))
        print(f"\n选中策略: {best_strategy} (收益率: {best_return:.2%})")

        self.current_strategy = best_strategy
        return best_strategy

    def select_stocks(self, date: pd.Timestamp, **kwargs) -> list:
        """选取信号最强的N只股票"""
        print(f"\n使用策略 {self.current_strategy} 筛选股票...")

        selected = self.stock_selector.select_top_stocks(
            strategy_name=self.current_strategy,
            n=self.n_stocks,
            date=date,
        )

        self.current_stocks = selected
        return selected

    def run_weekly(self, date: pd.Timestamp,
                   prices: dict = None) -> dict:
        """执行每周流程（策略评估 + 选股 + 调仓）"""
        # 1. 选择最优策略
        self.select_best_strategy(date)

        # 2. 执行模板流程（选股 + 调仓）
        result = super().run_weekly(date, prices)

        # 3. 补充策略信息
        result['strategy'] = self.current_strategy
        return result
