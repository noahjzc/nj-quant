"""
因子数据加载器

从DataProvider获取股票数据，提取所需因子
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Optional
from back_testing.data.data_provider import DataProvider

logger = logging.getLogger(__name__)


class FactorLoader:
    """
    因子数据加载器
    """

    # 可用的因子列名映射 (数据库实际列名)
    FACTOR_COLUMNS = {
        # 估值因子
        'PB': 'pb',
        'PE_TTM': 'pe_ttm',
        'PS_TTM': 'ps_ttm',
        'PCF_TTM': 'pcf_ttm',
        # RSI 指标
        'RSI_1': 'rsi_1',
        'RSI_2': 'rsi_2',
        'RSI_3': 'rsi_3',
        # KDJ 指标
        'KDJ_K': 'kdj_k',
        'KDJ_D': 'kdj_d',
        'KDJ_J': 'kdj_j',
        # 均线
        'MA_5': 'ma_5',
        'MA_10': 'ma_10',
        'MA_20': 'ma_20',
        'MA_30': 'ma_30',
        'MA_60': 'ma_60',
        # 交易类
        'TURNOVER': 'turnover_amount',  # 数据库实际列名是 turnover_amount
        'VOLUME_RATIO': 'volume_ratio',
        'AMPLITUDE': 'amplitude',
        # 动量/规模（需计算）
        'RET_20': 'ret_20',      # 20日收益率（需计算，非数据库列）
        'RET_60': 'ret_60',      # 60日收益率（需计算）
        'LN_MCAP': 'ln_mcap',    # 对数市值（需计算）
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

        total = len(stock_codes)
        for idx, code in enumerate(stock_codes):
            try:
                df = self.data_provider.get_stock_data(code, date=date)
                if len(df) == 0:
                    continue

                latest = df.iloc[-1]

                # 提取各因子值
                row = {}
                for factor in factors:
                    col_name = self.FACTOR_COLUMNS.get(factor, factor)
                    # 需要计算的因子
                    if factor in ('RET_20', 'RET_60'):
                        period = 20 if factor == 'RET_20' else 60
                        row[factor] = self._calculate_return(code, date, period)
                    elif factor == 'LN_MCAP':
                        row[factor] = self._calculate_ln_mcap(latest)
                    elif col_name in df.columns:
                        row[factor] = latest[col_name]
                    else:
                        row[factor] = None

                result_data[code] = row

                # 每500只打印一次进度
                if (idx + 1) % 500 == 0:
                    print(f"    [因子加载] 已处理 {idx + 1}/{total} 只股票...", flush=True)

            except Exception as e:
                logger.warning(f"Failed to load factors for {code}: {e}")
                continue

        result = pd.DataFrame(result_data).T

        # 填充缺失值
        for col in result.columns:
            if result[col].isna().any():
                # 先转换为数值类型（避免 object dtype 的 FutureWarning）
                numeric_col = pd.to_numeric(result[col], errors='coerce')
                median_val = numeric_col.median()
                # 只有当中位数不是 NaN 时才填充（全空数据时跳过）
                if pd.notna(median_val):
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
        print(f"    [因子加载] 获取全市场股票列表...", flush=True)
        all_codes = self.data_provider.get_all_stock_codes()
        print(f"    [因子加载] 股票列表共 {len(all_codes)} 只，正在逐只加载因子...", flush=True)
        result = self.load_stock_factors(all_codes, date, factors)
        print(f"    [因子加载] 完成，成功加载 {len(result)} 只股票的因子数据", flush=True)
        return result

    def load_stock_turnover(
        self,
        stock_codes: List[str],
        date: pd.Timestamp
    ) -> pd.Series:
        """
        获取指定日期的成交额

        Args:
            stock_codes: 股票代码列表
            date: 日期

        Returns:
            Series: index为股票代码，value为成交额（元）
        """
        result = {}
        for code in stock_codes:
            try:
                df = self.data_provider.get_stock_data(code, date=date)
                if len(df) == 0:
                    continue
                # 成交额列：尝试多种可能的列名
                turnover_col = None
                for col in df.columns:
                    col_upper = col.upper() if isinstance(col, str) else ''
                    if 'TURNOVER' in col_upper or '成交额' in col:
                        turnover_col = col
                        break
                if turnover_col and turnover_col in df.columns:
                    result[code] = df[turnover_col].iloc[-1]
            except Exception:
                continue
        return pd.Series(result)

    def _calculate_return(self, stock_code: str, date: pd.Timestamp, period: int) -> float:
        """
        计算过去N日收益率

        Args:
            stock_code: 股票代码
            date: 当前日期
            period: 回看天数 (20 或 60)

        Returns:
            收益率 (小数，如 0.15 表示 15%)
        """
        end_date = date.strftime('%Y-%m-%d')
        start_date = (date - pd.Timedelta(days=period * 3)).strftime('%Y-%m-%d')

        df = self.data_provider.get_stock_data(stock_code, start_date=start_date, end_date=end_date)
        if len(df) < period + 1:
            return 0.0

        prices = df['adj_close'].values
        if len(prices) < period + 1:
            return 0.0

        start_price = prices[-(period + 1)]
        end_price = prices[-1]
        if start_price == 0:
            return 0.0
        return (end_price - start_price) / start_price

    def _calculate_ln_mcap(self, row) -> float:
        """
        计算对数市值

        Returns:
            对数市值 (circulating_mv 或 total_mv 的自然对数)
        """
        mv = row.get('circulating_mv') or row.get('total_mv', 0)
        if mv and mv > 0:
            return np.log(mv)
        return 0.0