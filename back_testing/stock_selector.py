import pandas as pd
import numpy as np
import os
from back_testing.signal_scorer import SignalScorer

class StockSelector:
    """选股器：根据策略信号强度从全市场筛选股票"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scorer = SignalScorer()

    def get_all_stock_codes(self) -> list:
        """获取所有股票代码"""
        files = os.listdir(self.data_path)
        stock_codes = [f.replace('.csv', '') for f in files
                      if f.endswith('.csv') and not f.startswith('index')]
        return stock_codes

    def calculate_stock_signal(self, stock_code: str, strategy_name: str,
                                date: pd.Timestamp) -> float:
        """计算单只股票在特定日期的信号强度"""
        file_path = f"{self.data_path}\\{stock_code}.csv"
        try:
            df = pd.read_csv(file_path, encoding='gbk')
            df['交易日期'] = pd.to_datetime(df['交易日期'])
            df = df.sort_values('交易日期')

            # 筛选到指定日期的数据
            df = df[df['交易日期'] <= date]
            if len(df) == 0:
                return 0

            # 取最后一行计算信号强度
            latest = df.iloc[-1:]
            score = self.scorer.get_signal_strength(strategy_name, latest).iloc[0]
            return score
        except Exception:
            return 0

    def select_top_stocks(self, strategy_name: str, n: int = 5,
                          date: str = None) -> list:
        """选取信号最强的N只股票"""
        if date is None:
            date = pd.Timestamp.now()
        else:
            date = pd.to_datetime(date)

        all_codes = self.get_all_stock_codes()
        print(f"正在评估 {len(all_codes)} 只股票...")

        scores = []
        for code in all_codes:
            score = self.calculate_stock_signal(code, strategy_name, date)
            scores.append((code, score))

        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 返回前N只
        selected = [code for code, score in scores[:n]]
        print(f"选取结果: {selected}")
        return selected