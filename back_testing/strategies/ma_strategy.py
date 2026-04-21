from pandas import DataFrame
from back_testing.core.backtest_engine import BacktestEngine
import numpy as np


class MAStrategy(BacktestEngine):
    """
    MA均线策略

    买入信号：MA5从下方穿过MA20（金叉）
    卖出信号：MA5从上方穿过MA20（死叉）

    使用后复权价计算信号和收益率
    """

    def generate_signals(self) -> DataFrame:
        """生成MA均线交易信号"""
        df = self.data.copy()

        ma5_col = 'MA_5'
        ma20_col = 'MA_20'

        # 金叉：MA5上穿MA20
        golden_cross = (df[ma5_col] > df[ma20_col]) & (df[ma5_col].shift(1) <= df[ma20_col].shift(1))

        # 死叉：MA5下穿MA20
        dead_cross = (df[ma5_col] < df[ma20_col]) & (df[ma5_col].shift(1) >= df[ma20_col].shift(1))

        # 初始化信号列
        df['TRADE_SIGNAL'] = np.nan

        # 标记信号：1=买入，0=卖出
        df.loc[golden_cross, 'TRADE_SIGNAL'] = 1
        df.loc[dead_cross, 'TRADE_SIGNAL'] = 0

        # 去除NaN信号
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df
