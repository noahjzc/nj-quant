import pandas as pd
import numpy as np
import os
from back_testing.composite_scorer import CompositeScorer

class CompositeSelector:
    """
    综合选股器：使用多策略综合评分从全市场筛选股票
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scorer = CompositeScorer()

    def get_all_stock_codes(self) -> list:
        """获取所有股票代码"""
        files = os.listdir(self.data_path)
        stock_codes = [f.replace('.csv', '') for f in files
                      if f.endswith('.csv') and not f.startswith('index')]
        return stock_codes

    def calculate_stock_score(self, stock_code: str, date: pd.Timestamp) -> tuple:
        """
        计算单只股票的综合评分

        Returns:
            tuple: (stock_code, composite_score)
        """
        file_path = f"{self.data_path}\\{stock_code}.csv"
        try:
            df = pd.read_csv(file_path, encoding='gbk')
            df['交易日期'] = pd.to_datetime(df['交易日期'])
            df = df.sort_values('交易日期')

            # 筛选到指定日期的数据
            df = df[df['交易日期'] <= date]
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
        print(f"正在评估 {len(all_codes)} 只股票的综合评分...")

        scores = []
        for code in all_codes:
            stock_code, score = self.calculate_stock_score(code, date)
            scores.append((stock_code, score))

        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 返回前N只
        selected = [code for code, score in scores[:n]]
        print(f"选取结果: {selected}")
        return selected