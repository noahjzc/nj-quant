"""综合选股器（兼容包装）

内部委托给 StockSelector（统一选股器）执行批量评分，
避免逐只查询的 N+1 问题。
"""
import pandas as pd

from back_testing.selectors.stock_selector import StockSelector


class CompositeSelector:
    """综合选股器（兼容包装）

    保留原接口，内部使用 StockSelector + CompositeScorer 批量评分。
    """

    def __init__(self):
        self._selector = StockSelector()

    def get_all_stock_codes(self) -> list:
        return self._selector.data_provider.get_all_stock_codes()

    def select_top_stocks(self, n: int = 5, date: str = None) -> list:
        if date is None:
            date = pd.Timestamp.now()
        else:
            date = pd.to_datetime(date)

        return self._selector.select_top_stocks(n=n, date=date)
