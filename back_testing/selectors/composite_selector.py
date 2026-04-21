import sys
import pandas as pd
import numpy as np
from back_testing.composite_scorer import CompositeScorer
from back_testing.data.data_provider import DataProvider


class CompositeSelector:
    """
    综合选股器：使用多策略综合评分从全市场筛选股票
    """

    def __init__(self, data_path: str = None, use_parquet: bool = True):
        self.data_provider = DataProvider(data_dir=data_path, use_parquet=use_parquet)
        self.scorer = CompositeScorer()

    def get_all_stock_codes(self) -> list:
        """获取所有股票代码"""
        return self.data_provider.get_all_stock_codes()

    def calculate_stock_score(self, stock_code: str, date: pd.Timestamp) -> tuple:
        """
        计算单只股票的综合评分

        Returns:
            tuple: (stock_code, composite_score)
        """
        try:
            # 使用DataProvider读取数据
            df = self.data_provider.get_stock_data(stock_code, date=date)
            if len(df) < 5:  # 需要足够的历史数据
                return (stock_code, 0)

            # 取最后一行计算综合评分
            latest = df.iloc[-1:]
            score = self.scorer.calculate_composite_scores(latest).iloc[0]
            return (stock_code, score)
        except Exception:
            return (stock_code, 0)

    def select_top_stocks(self, n: int = 5, date: str = None) -> list:
        """
        选取综合评分最高的N只股票

        Args:
            n: 选取数量
            date: 评分日期

        Returns:
            list of stock codes
        """
        if date is None:
            date = pd.Timestamp.now()
        else:
            date = pd.to_datetime(date)

        all_codes = self.get_all_stock_codes()
        total = len(all_codes)
        print(f"正在评估 {total} 只股票的综合评分...", flush=True)

        scores = []
        for i, code in enumerate(all_codes):
            stock_code, score = self.calculate_stock_score(code, date)
            scores.append((stock_code, score))
            # 每评估10%打印一次进度
            if (i + 1) % (total // 10 + 1) == 0 or i == total - 1:
                progress = (i + 1) / total * 100
                print(f"\r进度: {i+1}/{total} ({progress:.1f}%)", end="", flush=True)

        print()  # 换行
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 返回前N只
        selected = [code for code, score in scores[:n]]
        print(f"选取结果: {selected}")
        return selected
