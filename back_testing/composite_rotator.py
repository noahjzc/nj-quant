import pandas as pd

from back_testing.base_rotator import BaseRotator
from back_testing.selectors.multi_factor_selector import MultiFactorSelector
from back_testing.selectors.stock_selector import StockSelector
from back_testing.factors.factor_config import get_factor_weights, get_factor_directions
from back_testing.factors.factor_loader import FactorLoader
from back_testing.composite_scorer import CompositeScorer
from back_testing.data.data_provider import DataProvider


class CompositeRotator(BaseRotator):
    """
    综合评分轮动主控制器

    每周流程：
    1. 使用多因子或 CompositeSelector 选股
    2. 选取评分最高的 N 只股票
    3. 等权 20% 持仓
    """

    def __init__(self, initial_capital: float = 1_000_000.0,
                 n_stocks: int = 5, use_multi_factor: bool = True):
        super().__init__(initial_capital, n_stocks)
        self.use_multi_factor = use_multi_factor

        self.composite_selector = StockSelector()
        self.composite_scorer = CompositeScorer()

        # Multi-factor selector
        if use_multi_factor:
            self.factor_weights = get_factor_weights()
            self.factor_directions = get_factor_directions()
            self.factor_loader = FactorLoader(data_provider=DataProvider())
            self.factor_selector = MultiFactorSelector(
                weights=self.factor_weights,
                directions=self.factor_directions,
                data_provider=self.factor_loader.data_provider,
            )

        self.current_strategy = 'CompositeScorer'

    def select_stocks(self, date: pd.Timestamp, **kwargs) -> list:
        """选取综合评分最高的 N 只股票"""
        if self.use_multi_factor:
            print(f"\n使用多因子策略筛选股票...")
            return self._select_stocks_multi_factor(date)
        else:
            print(f"\n使用综合评分策略筛选股票...")
            return self._select_stocks_composite(date)

    def _select_stocks_multi_factor(self, date: pd.Timestamp) -> list:
        """使用多因子选股"""
        try:
            print(f"    [多因子] 正在加载全市场股票因子数据... date={date.strftime('%Y-%m-%d')}", flush=True)
            factors = list(self.factor_weights.keys())
            factor_data = self.factor_loader.load_all_stock_factors(date, factors)
            print(f"    [多因子] 因子数据加载完成，共 {len(factor_data)} 只股票", flush=True)

            if len(factor_data) == 0:
                print("警告：未获取到因子数据，切换到综合评分策略")
                return self._select_stocks_composite(date)

            print(f"    [多因子] 正在计算因子评分并选股...", flush=True)
            selected = self.factor_selector.select_top_stocks(
                data=factor_data,
                n=self.n_stocks,
                excluded=self.current_stocks,
                date=date,
            )
            print(f"    [多因子] 选取完成，结果: {selected}", flush=True)

            self.current_stocks = selected
            print(f"多因子选取结果: {selected}")
            return selected
        except Exception as e:
            print(f"多因子选股失败: {e}，切换到综合评分策略")
            return self._select_stocks_composite(date)

    def _select_stocks_composite(self, date: pd.Timestamp) -> list:
        """使用综合评分选股（旧逻辑）"""
        selected = self.composite_selector.select_top_stocks(
            n=self.n_stocks,
            date=date,
        )
        self.current_stocks = selected
        return selected
