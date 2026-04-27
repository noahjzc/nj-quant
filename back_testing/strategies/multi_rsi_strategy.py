from pandas import DataFrame
from back_testing.core.backtest_engine import BacktestEngine
from data_column_names import RSI_1, RSI_2, RSI_3, POST_ADJUSTED_CLOSE_PRICE
import numpy as np


class MultiPeriodRSIStrategy(BacktestEngine):
    """
    多周期RSI策略

    同时参考RSI(6)、RSI(12)、RSI(24)三个周期
    买入信号：短期RSI超卖且中长周期RSI也处于低位（共振）
    卖出信号：任一RSI进入超买区域

    RSI参数：数据已预计算rsi1, rsi2, rsi3
    """

    def generate_signals(self) -> DataFrame:
        """生成多周期RSI交易信号"""
        df = self.data.copy()

        rsi_short = RSI_1   # 短期RSI (如RSI6)
        rsi_mid = RSI_2     # 中期RSI (如RSI12)
        rsi_long = RSI_3    # 长期RSI (如RSI24)

        # 多周期共振买入：短期RSI < 30 且 中期RSI < 40
        buy_signal = (df[rsi_short] < 30) & (df[rsi_mid] < 40)

        # 卖出信号：任一RSI > 70
        sell_signal = (df[rsi_short] > 70) | (df[rsi_mid] > 70) | (df[rsi_long] > 70)

        df['TRADE_SIGNAL'] = np.nan
        df.loc[buy_signal, 'TRADE_SIGNAL'] = 1
        df.loc[sell_signal, 'TRADE_SIGNAL'] = 0
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df


class RSIReversalMultiStrategy(BacktestEngine):
    """
    RSI多周期共振反转策略

    更严格的买入条件：多个周期同时超卖
    买入信号：三个周期RSI都 < 35（强共振）
    卖出信号：短期RSI > 65 或 中期RSI > 60
    """

    def generate_signals(self) -> DataFrame:
        """生成RSI多周期共振反转交易信号"""
        df = self.data.copy()

        rsi_short = RSI_1
        rsi_mid = RSI_2
        rsi_long = RSI_3

        # 强共振买入：三个周期都处于超卖区域
        buy_signal = (df[rsi_short] < 35) & (df[rsi_mid] < 35) & (df[rsi_long] < 35)

        # RSI转头向上（从前一天低点反弹）
        rsi_reversal = df[rsi_short] > df[rsi_short].shift(1)

        # 买入信号：共振超卖 + RSI开始反弹
        buy_signal = buy_signal & rsi_reversal

        # 卖出信号：任一RSI进入超买
        sell_signal = (df[rsi_short] > 65) | (df[rsi_mid] > 60) | (df[rsi_long] > 60)

        df['TRADE_SIGNAL'] = np.nan
        df.loc[buy_signal, 'TRADE_SIGNAL'] = 1
        df.loc[sell_signal, 'TRADE_SIGNAL'] = 0
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df
