from pandas import DataFrame
from back_testing.core.backtest_engine import BacktestEngine
from data_column_names import BOLL_UPPER, BOLL_MIDDLE, BOLL_LOWER, POST_ADJUSTED_CLOSE_PRICE
import numpy as np


class BollingerStrategy(BacktestEngine):
    """
    布林带策略

    买入信号：价格触及布林带下轨
    卖出信号：价格触及布林带中轨

    布林带参数：数据已预计算布林线上轨, 布林线中轨, 布林线下轨
    """

    def generate_signals(self) -> DataFrame:
        """生成布林带交易信号"""
        df = self.data.copy()

        lower_col = BOLL_LOWER
        middle_col = BOLL_MIDDLE
        upper_col = BOLL_UPPER
        price_col = POST_ADJUSTED_CLOSE_PRICE

        # 买入信号：价格触及或跌破布林带下轨
        buy_signal = df[price_col] <= df[lower_col]

        # 卖出信号：价格触及或突破布林带中轨
        sell_signal = df[price_col] >= df[middle_col]

        df['TRADE_SIGNAL'] = np.nan
        df.loc[buy_signal, 'TRADE_SIGNAL'] = 1
        df.loc[sell_signal, 'TRADE_SIGNAL'] = 0
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df


class BollingerStrictStrategy(BacktestEngine):
    """
    布林带策略（严格版）

    买入信号：价格跌破布林带下轨（超卖）
    卖出信号：价格突破布林带上轨（超买）或回到中轨

    更严格的布林带策略，减少假信号
    """

    def generate_signals(self) -> DataFrame:
        """生成布林带交易信号（严格版）"""
        df = self.data.copy()

        lower_col = BOLL_LOWER
        middle_col = BOLL_MIDDLE
        upper_col = BOLL_UPPER
        price_col = POST_ADJUSTED_CLOSE_PRICE

        # 买入信号：价格跌破布林带下轨
        buy_signal = (df[price_col] < df[lower_col]) & (df[price_col].shift(1) >= df[lower_col].shift(1))

        # 卖出信号：价格突破布林带上轨
        sell_signal = (df[price_col] > df[upper_col]) & (df[price_col].shift(1) <= df[upper_col].shift(1))

        df['TRADE_SIGNAL'] = np.nan
        df.loc[buy_signal, 'TRADE_SIGNAL'] = 1
        df.loc[sell_signal, 'TRADE_SIGNAL'] = 0
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df
