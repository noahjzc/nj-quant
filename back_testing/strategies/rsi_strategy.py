from pandas import DataFrame
from back_testing.core.backtest_engine import BacktestEngine
from data_column_names import RSI_1
import numpy as np


class RSIReversalStrategy(BacktestEngine):
    """
    RSI均值回归策略

    买入信号：RSI1 < 30（超卖区域）
    卖出信号：RSI1 > 50（回到正常区间）

    这是一个均值回归策略，假设RSI跌到30以下是被低估了，
    短期内会反弹到50左右的正常水平。
    """

    def generate_signals(self) -> DataFrame:
        """生成RSI均值回归交易信号"""
        df = self.data.copy()

        rsi_col = RSI_1

        # 买入信号：RSI < 30 超卖
        buy_signal = df[rsi_col] < 30

        # 卖出信号：RSI > 50
        sell_signal = df[rsi_col] > 50

        # 初始化信号列
        df['TRADE_SIGNAL'] = np.nan

        # 标记信号：1=买入，0=卖出
        df.loc[buy_signal, 'TRADE_SIGNAL'] = 1
        df.loc[sell_signal, 'TRADE_SIGNAL'] = 0

        # 去除NaN信号
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df


class RSIExtremaStrategy(BacktestEngine):
    """
    RSI极值策略（更严格的版本）

    买入信号：RSI1 < 25（深度超卖）
    卖出信号：RSI1 > 60 或 RSI从低位回升20%以上

    追求更高胜率，但交易次数更少
    """

    def generate_signals(self) -> DataFrame:
        """生成RSI极值交易信号"""
        df = self.data.copy()

        rsi_col = RSI_1

        # 买入信号：RSI < 25 深度超卖
        buy_signal = df[rsi_col] < 25

        # 卖出信号：RSI > 60
        sell_signal = df[rsi_col] > 60

        # 初始化信号列
        df['TRADE_SIGNAL'] = np.nan

        # 标记信号：1=买入，0=卖出
        df.loc[buy_signal, 'TRADE_SIGNAL'] = 1
        df.loc[sell_signal, 'TRADE_SIGNAL'] = 0

        # 去除NaN信号
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df
