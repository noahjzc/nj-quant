import os
from typing import Optional

import pandas as pd


class IndexDataProvider:
    """指数数据提供者，用于读取指数CSV数据文件"""

    def __init__(self, data_dir: str):
        """data_dir: 指数数据目录"""
        self.data_dir = data_dir

    def _get_index_file_path(self, index_code: str) -> str:
        """获取指数文件路径"""
        return os.path.join(self.data_dir, f"{index_code}.csv")

    def get_index_data(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指数数据，返回包含 date, open, close, high, low, volume 列的DataFrame"""
        file_path = self._get_index_file_path(index_code)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Index data file not found: {file_path}")

        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])

        # Filter by date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        mask = (df['date'] >= start) & (df['date'] <= end)
        df = df.loc[mask].copy()

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Return only required columns
        return df[['date', 'open', 'close', 'high', 'low', 'volume']]

    def get_index_return(self, index_code: str, start_date: str, end_date: str) -> float:
        """计算区间收益率，返回小数形式如 0.15 表示 15%"""
        df = self.get_index_data(index_code, start_date, end_date)
        if len(df) < 2:
            raise ValueError(f"Not enough data to calculate return for {index_code} between {start_date} and {end_date}")

        start_price = df.iloc[0]['close']
        end_price = df.iloc[-1]['close']
        return (end_price - start_price) / start_price
