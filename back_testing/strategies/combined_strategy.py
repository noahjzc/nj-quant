from back_testing.core.backtest_engine import BacktestEngine
from data_column_names import MA_5, MA_20, RSI_1
import numpy as np
from pandas import DataFrame

class CombinedStrategy(BacktestEngine):
    """
    Combined Trend + Reversal Strategy

    Buy: MA5 crosses above MA20 (trend signal) AND RSI < 40 (not overbought)
    Sell: MA5 crosses below MA20 (trend signal) OR RSI > 60 (overbought)

    Goal: Reduce false signals by combining trend-following with reversal filter
    """

    def generate_signals(self) -> DataFrame:
        df = self.data.copy()

        ma5_col = MA_5
        ma20_col = MA_20
        rsi_col = RSI_1

        # MA golden cross
        ma_golden = (df[ma5_col] > df[ma20_col]) & (df[ma5_col].shift(1) <= df[ma20_col].shift(1))
        # MA dead cross
        ma_dead = (df[ma5_col] < df[ma20_col]) & (df[ma5_col].shift(1) >= df[ma20_col].shift(1))

        # Buy: golden cross AND RSI not overbought
        buy_signal = ma_golden & (df[rsi_col] < 40)

        # Sell: dead cross OR RSI overbought
        sell_signal = ma_dead | (df[rsi_col] > 60)

        df['TRADE_SIGNAL'] = np.nan
        df.loc[buy_signal, 'TRADE_SIGNAL'] = 1
        df.loc[sell_signal, 'TRADE_SIGNAL'] = 0
        df = df.dropna(subset=['TRADE_SIGNAL'])

        return df
