import pandas as pd
import numpy as np
from back_testing.selectors.composite_selector import CompositeSelector
from back_testing.selectors.multi_factor_selector import MultiFactorSelector
from back_testing.factors.factor_config import get_factor_weights, get_factor_directions
from back_testing.factors.factor_loader import FactorLoader
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
                 n_stocks: int = 5, use_multi_factor: bool = True):
        self.data_path = data_path
        self.initial_capital = initial_capital
        self.n_stocks = n_stocks
        self.per_stock_capital = initial_capital / n_stocks
        self.use_multi_factor = use_multi_factor

        self.composite_selector = CompositeSelector(data_path)
        self.composite_scorer = CompositeScorer()

        # Multi-factor selector
        if use_multi_factor:
            self.factor_weights = get_factor_weights()
            self.factor_directions = get_factor_directions()
            self.factor_selector = MultiFactorSelector(
                weights=self.factor_weights,
                directions=self.factor_directions
            )
            self.factor_loader = FactorLoader()

        self.current_stocks = []
        self.current_positions = {}

    def select_stocks(self, date: pd.Timestamp) -> list:
        """选取综合评分最高的N只股票"""
        if self.use_multi_factor:
            print(f"\n使用多因子策略筛选股票...")
            return self._select_stocks_multi_factor(date)
        else:
            print(f"\n使用综合评分策略筛选股票...")
            return self._select_stocks_composite(date)

    def _select_stocks_multi_factor(self, date: pd.Timestamp) -> list:
        """使用多因子选股"""
        try:
            # 加载所有股票因子数据
            factors = list(self.factor_weights.keys())
            factor_data = self.factor_loader.load_all_stock_factors(date, factors)

            if len(factor_data) == 0:
                print("警告：未获取到因子数据，切换到综合评分策略")
                return self._select_stocks_composite(date)

            # 使用多因子选择器选股
            selected = self.factor_selector.select_top_stocks(
                data=factor_data,
                n=self.n_stocks,
                excluded=self.current_stocks  # 排除已有持仓
            )

            self.current_stocks = selected
            print(f"多因子选取结果: {selected}")
            return selected
        except Exception as e:
            print(f"多因子选股失败: {e}，切换到综合评分策略")
            return self._select_stocks_composite(date)

    def _select_stocks_composite(self, date: pd.Timestamp) -> list:
        """使用综合评分选股（原有逻辑）"""
        selected = self.composite_selector.select_top_stocks(
            n=self.n_stocks,
            date=date
        )

        self.current_stocks = selected
        return selected

    def rebalance(self, date: pd.Timestamp, prices: dict = None) -> dict:
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
                # 计算实际持仓份额
                if prices and code in prices and prices[code] > 0:
                    shares = int(self.per_stock_capital / prices[code])
                else:
                    shares = 0
                self.current_positions[code] = {
                    'shares': shares,
                    'buy_price': prices.get(code, 0) if prices else 0
                }

        return rebalance_detail

    def run_weekly(self, date: pd.Timestamp, prices: dict = None) -> dict:
        """执行每周流程"""
        # 筛选股票
        self.select_stocks(date)

        # 执行调仓
        rebalance = self.rebalance(date, prices)

        return {
            'date': date,
            'strategy': 'CompositeScorer',
            'stocks': self.current_stocks,
            'rebalance': rebalance
        }
