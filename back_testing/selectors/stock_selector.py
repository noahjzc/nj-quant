"""统一选股器：批量评分，支持 CompositeScorer 和 SignalScorer 两种模式。

用 get_batch_latest() 一次性获取所有股票的最新数据行，
避免原来逐只查询的 N+1 问题。
"""
from typing import Optional, List

import pandas as pd

from back_testing.composite_scorer import CompositeScorer
from back_testing.data.data_provider import DataProvider
from back_testing.signal_scorer import SignalScorer


class StockSelector:
    """统一选股器

    两种评分模式：
    - CompositeScorer（默认）：多策略综合评分
    - SignalScorer：按策略名称计算信号强度

    Examples:
        >>> selector = StockSelector()
        >>> selector.select_top_stocks(n=5, date='2024-01-15')
        >>> selector.select_top_stocks(n=5, date='2024-01-15', strategy_name='RSIReversalStrategy')
    """

    def __init__(self, scorer=None):
        self.data_provider = DataProvider()
        self._composite_scorer = scorer if isinstance(scorer, CompositeScorer) else CompositeScorer()
        self._signal_scorer = SignalScorer()

    def select_top_stocks(
        self,
        n: int = 5,
        date: Optional[str] = None,
        strategy_name: Optional[str] = None,
    ) -> List[str]:
        """选取评分最高的 N 只股票

        Args:
            n: 选取数量
            date: 评分日期
            strategy_name: 策略名称（None=使用综合评分）

        Returns:
            List[str]: 股票代码列表
        """
        if date is None:
            date = pd.Timestamp.now()
        else:
            date = pd.to_datetime(date)

        all_codes = self.data_provider.get_all_stock_codes()
        print(f"正在批量获取 {len(all_codes)} 只股票的数据...", flush=True)

        # 一条 SQL 批量获取所有股票的最新数据行
        batch_data = self.data_provider.get_batch_latest(all_codes, date)

        scores = []
        for code in all_codes:
            if code not in batch_data:
                scores.append((code, 0.0))
                continue

            row = batch_data[code]
            df = pd.DataFrame([row])

            try:
                if strategy_name:
                    score = self._signal_scorer.get_signal_strength(strategy_name, df).iloc[0]
                else:
                    score = self._composite_scorer.calculate_composite_scores(df).iloc[0]
            except Exception:
                score = 0.0

            scores.append((code, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [code for code, _ in scores[:n]]
        print(f"选取结果: {selected}")
        return selected
