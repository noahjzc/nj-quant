from pandas import DataFrame
from back_testing.core.backtest_engine import BacktestEngine
from data_column_names import KDJ_K, KDJ_D, KDJ_J, POST_ADJUSTED_CLOSE_PRICE
import numpy as np


class KDJOversoldStrategy(BacktestEngine):
    """
    KDJ超卖策略

    买入信号：K值或J值进入超卖区域（< 20），随后反转
    卖出信号：K值或J值进入超买区域（> 80），或死叉

    KDJ参数：数据已预计算KDJ_K, KDJ_D, KDJ_J
    """

    def generate_signals(self) -> DataFrame:
        """生成KDJ交易信号"""
        df = self.data.copy()

        k_col = KDJ_K
        d_col = KDJ_D
        j_col = KDJ_J
        price_col = POST_ADJUSTED_CLOSE_PRICE

        # KDJ超卖策略：J值超卖后反转买入
        # 超卖区域：J < 20
        oversold = df[j_col] < 20
        # 前一天更超卖（J值比今天低）
        oversold_reversal = (df[j_col] > df[j_col].shift(1)) & oversold

        # KDJ死叉：K从上方穿过D
        dead_cross = (df[k_col] < df[d_col]) & (df[k_col].shift(1) >= df[d_col].shift(1))

        # 买入信号：J值从超卖区域反弹
        buy_signal = oversold_reversal

        # 卖出信号：KD死叉 或 J值进入超买区域
        sell_signal = dead_cross | (df[j_col] > 80)

        df['TRADE_SIGNAL'] = np.nan
        df.loc[buy_signal, 'TRADE_SIGNAL'] = 1
        df.loc[sell_signal, 'TRADE_SIGNAL'] = 0
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df


class KDJGoldenCrossStrategy(BacktestEngine):
    """
    KDJ金叉策略

    买入信号：K值从下方穿过D值（金叉），且K值 < 50（低位）
    卖出信号：K值从上方穿过D值（死叉），或K值 > 80（超买）

    趋势型KDJ策略，减少假信号
    """

    def generate_signals(self) -> DataFrame:
        """生成KDJ金叉交易信号"""
        df = self.data.copy()

        k_col = KDJ_K
        d_col = KDJ_D
        j_col = KDJ_J

        # 金叉：K从下方穿过D，且在低位区（K < 50）
        golden_cross = (df[k_col] > df[d_col]) & (df[k_col].shift(1) <= df[d_col].shift(1))
        buy_signal = golden_cross & (df[k_col] < 50)

        # 死叉：K从上方穿过D
        dead_cross = (df[k_col] < df[d_col]) & (df[k_col].shift(1) >= df[d_col].shift(1))

        # 卖出信号：死叉 或 K进入超买区
        sell_signal = dead_cross | (df[k_col] > 80)

        df['TRADE_SIGNAL'] = np.nan
        df.loc[buy_signal, 'TRADE_SIGNAL'] = 1
        df.loc[sell_signal, 'TRADE_SIGNAL'] = 0
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df
