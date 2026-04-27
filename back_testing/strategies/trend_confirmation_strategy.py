from pandas import DataFrame
from back_testing.core.backtest_engine import BacktestEngine
from data_column_names import MA_5, MA_20, MA_60, MACD_DIF, MACD_DEA, POST_ADJUSTED_CLOSE_PRICE
import numpy as np


class TrendConfirmationStrategy(BacktestEngine):
    """
    趋势确认策略

    使用MA确定趋势方向，MACD确认入场时机
    买入条件：MA5 > MA20（上升趋势）且 MA20 > MA60（多头排列），MACD金叉
    卖出条件：MA5 < MA20（下降趋势）或 MACD死叉

    策略特点：趋势跟踪型，减少假突破
    """

    def generate_signals(self) -> DataFrame:
        """生成趋势确认交易信号"""
        df = self.data.copy()

        ma5 = MA_5
        ma20 = MA_20
        ma60 = MA_60
        dif = MACD_DIF
        dea = MACD_DEA
        price = POST_ADJUSTED_CLOSE_PRICE

        # 上升趋势：MA5 > MA20 > MA60（多头排列）
        uptrend = (df[ma5] > df[ma20]) & (df[ma20] > df[ma60])

        # MACD金叉：DIF从下方穿过DEA
        golden_cross = (df[dif] > df[dea]) & (df[dif].shift(1) <= df[dea].shift(1))

        # MACD死叉：DIF从上方穿过DEA
        dead_cross = (df[dif] < df[dea]) & (df[dif].shift(1) >= df[dea].shift(1))

        # 买入信号：多头排列 + MACD金叉
        buy_signal = uptrend & golden_cross

        # 卖出信号：空头排列（MA5 < MA20）或 MACD死叉
        downtrend = df[ma5] < df[ma20]
        sell_signal = downtrend | dead_cross

        df['TRADE_SIGNAL'] = np.nan
        df.loc[buy_signal, 'TRADE_SIGNAL'] = 1
        df.loc[sell_signal, 'TRADE_SIGNAL'] = 0
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df


class TrendPullbackStrategy(BacktestEngine):
    """
    趋势回踩策略

    确认上升趋势后，等待价格回踩MA20后再次上涨时买入
    买入条件：MA5上穿MA20后，价格回踩MA20并反弹
    卖出条件：MA5下穿MA20

    策略特点：趋势中的回撤买入，降低追高风险
    """

    def generate_signals(self) -> DataFrame:
        """生成趋势回踩交易信号"""
        df = self.data.copy()

        ma5 = MA_5
        ma20 = MA_20
        price = POST_ADJUSTED_CLOSE_PRICE

        # MA5上穿MA20（金叉）
        ma_golden = (df[ma5] > df[ma20]) & (df[ma5].shift(1) <= df[ma20].shift(1))

        # 价格回踩MA20：价格在MA20附近且从前低点上反弹
        pullback = (df[price] >= df[ma20] * 0.98) & (df[price] <= df[ma20] * 1.02)
        price_rebound = df[price] > df[price].shift(1)

        # 买入信号：金叉后回踩MA20并反弹
        buy_signal = ma_golden & pullback & price_rebound

        # 卖出信号：MA5下穿MA20（死叉）
        ma_dead = (df[ma5] < df[ma20]) & (df[ma5].shift(1) >= df[ma20].shift(1))
        sell_signal = ma_dead

        df['TRADE_SIGNAL'] = np.nan
        df.loc[buy_signal, 'TRADE_SIGNAL'] = 1
        df.loc[sell_signal, 'TRADE_SIGNAL'] = 0
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df
