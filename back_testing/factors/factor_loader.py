"""
因子数据加载器

从DataProvider获取股票数据，提取所需因子
"""
import pandas as pd
from typing import List, Optional
from back_testing.data.data_provider import DataProvider


class FactorLoader:
    """
    因子数据加载器
    """

    # 可用的因子列名映射
    FACTOR_COLUMNS = {
        'PB': 'PB',
        'PE_TTM': 'PE_TTM',
        'PS_TTM': 'PS_TTM',
        'ROE': 'ROE_TTM',  # 如果数据中有
        'RSI_1': 'RSI_1',
        'KDJ_K': 'KDJ_K',
        'KDJ_D': 'KDJ_D',
        'MA_5': 'MA_5',
        'MA_10': 'MA_10',
        'MA_20': 'MA_20',
        'MA_30': 'MA_30',
        'TURNOVER': 'TURNOVER',
        'VOLUME_RATIO': 'VOLUME_RATIO',
        'AMPLITUDE': 'AMPLITUDE',
    }

    def __init__(self, data_provider: DataProvider = None):
        self.data_provider = data_provider or DataProvider()

    def load_stock_factors(
        self,
        stock_codes: List[str],
        date: pd.Timestamp,
        factors: List[str]
    ) -> pd.DataFrame:
        """
        加载指定股票的因子数据

        Args:
            stock_codes: 股票代码列表
            date: 评分日期
            factors: 需要加载的因子列表

        Returns:
            DataFrame: index为股票代码，columns为因子值
        """
        result_data = {}

        for code in stock_codes:
            try:
                df = self.data_provider.get_stock_data(code, date=date)
                if len(df) == 0:
                    continue

                latest = df.iloc[-1]

                # 提取各因子值
                row = {}
                for factor in factors:
                    col_name = self.FACTOR_COLUMNS.get(factor, factor)
                    if col_name in df.columns:
                        row[factor] = latest[col_name]
                    else:
                        row[factor] = None

                result_data[code] = row

            except Exception:
                continue

        result = pd.DataFrame(result_data).T

        # 填充缺失值
        for col in result.columns:
            if result[col].isna().any():
                # 用中位数填充
                median_val = result[col].median()
                result[col] = result[col].fillna(median_val)

        return result

    def load_all_stock_factors(
        self,
        date: pd.Timestamp,
        factors: List[str]
    ) -> pd.DataFrame:
        """
        加载所有股票的因子数据

        Returns:
            DataFrame: 所有股票的因子数据
        """
        all_codes = self.data_provider.get_all_stock_codes()
        return self.load_stock_factors(all_codes, date, factors)