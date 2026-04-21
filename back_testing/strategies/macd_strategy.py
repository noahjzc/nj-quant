from pandas import DataFrame
from back_testing.core.backtest_engine import BacktestEngine
import numpy as np


class MACDStrategy(BacktestEngine):
    """
    MACD策略

    买入信号：DIF从下方穿过DEA（金叉）
    卖出信号：DIF从上方穿过DEA（死叉）

    DIF = EMA(close, 12) - EMA(close, 26)
    DEA = EMA(DIF, 9)
    """

    def generate_signals(self) -> DataFrame:
        """生成MACD交易信号"""
        df = self.data.copy()

        dif_col = 'MACD_DIF'
        dea_col = 'MACD_DEA'

        # 金叉：DIF上穿DEA
        golden_cross = (df[dif_col] > df[dea_col]) & (df[dif_col].shift(1) <= df[dea_col].shift(1))

        # 死叉：DIF下穿DEA
        dead_cross = (df[dif_col] < df[dea_col]) & (df[dif_col].shift(1) >= df[dea_col].shift(1))

        # 初始化信号列
        df['TRADE_SIGNAL'] = np.nan

        # 标记信号：1=买入，0=卖出
        df.loc[golden_cross, 'TRADE_SIGNAL'] = 1
        df.loc[dead_cross, 'TRADE_SIGNAL'] = 0

        # 去除NaN信号
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df
