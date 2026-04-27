from pandas import DataFrame
from back_testing.core.backtest_engine import BacktestEngine
from data_column_names import VOLUME_RATIO, POST_ADJUSTED_CLOSE_PRICE, MA_5, MA_20
import numpy as np


class VolumeAnomalyStrategy(BacktestEngine):
    """
    成交量异常策略

    基于量比（VOLUME_RATIO）检测异常放量
    买入信号：量比突然放大（> 2倍）且价格同时上涨
    卖出信号：量比萎缩（< 0.5）或价格下跌

    策略特点：关注资金推动的行情启动点
    """

    def generate_signals(self) -> DataFrame:
        """生成成交量异常交易信号"""
        df = self.data.copy()

        vol_ratio = VOLUME_RATIO
        price = POST_ADJUSTED_CLOSE_PRICE

        # 异常放量：量比突然放大到2倍以上
        volume_surge = df[vol_ratio] > 2.0

        # 价格同时上涨
        price_rise = df[price] > df[price].shift(1)

        # 买入信号：异常放量 + 价格同步上涨
        buy_signal = volume_surge & price_rise

        # 卖出信号：量比萎缩或价格下跌
        volume_shrink = df[vol_ratio] < 0.5
        price_fall = df[price] < df[price].shift(1)
        sell_signal = volume_shrink | price_fall

        df['TRADE_SIGNAL'] = np.nan
        df.loc[buy_signal, 'TRADE_SIGNAL'] = 1
        df.loc[sell_signal, 'TRADE_SIGNAL'] = 0
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df


class VolumeMAConfirmStrategy(BacktestEngine):
    """
    放量确认策略

    结合趋势（MA）和放量确认信号
    买入条件：上升趋势中（MA多头排列）且突然放量（量比>1.5）
    卖出条件：MA死叉或量比萎缩

    策略特点：趋势确认 + 放量验证，双重过滤假信号
    """

    def generate_signals(self) -> DataFrame:
        """生成放量确认交易信号"""
        df = self.data.copy()

        vol_ratio = VOLUME_RATIO
        ma5 = MA_5
        ma20 = MA_20
        price = POST_ADJUSTED_CLOSE_PRICE

        # 上升趋势：MA5 > MA20
        uptrend = df[ma5] > df[ma20]

        # 放量：量比 > 1.5
        volume_surge = df[vol_ratio] > 1.5

        # 价格同时上涨
        price_rise = df[price] > df[price].shift(1)

        # 买入信号：上升趋势 + 放量确认
        buy_signal = uptrend & volume_surge & price_rise

        # 卖出信号：MA死叉 或 量比萎缩
        ma_dead = (df[ma5] < df[ma20]) & (df[ma5].shift(1) >= df[ma20].shift(1))
        volume_shrink = df[vol_ratio] < 0.6
        sell_signal = ma_dead | volume_shrink

        df['TRADE_SIGNAL'] = np.nan
        df.loc[buy_signal, 'TRADE_SIGNAL'] = 1
        df.loc[sell_signal, 'TRADE_SIGNAL'] = 0
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df
