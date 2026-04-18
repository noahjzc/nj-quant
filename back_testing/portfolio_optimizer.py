import pandas as pd
import numpy as np
from itertools import product
from back_testing.backtest_engine import BacktestEngine


class PortfolioOptimizer:
    """
    参数优化器
    对策略参数进行网格搜索，找到历史表现最佳的参数组合
    """

    def __init__(self, strategy_class, stock_codes, data_path, benchmark_index='sh000001'):
        self.strategy_class = strategy_class
        self.stock_codes = stock_codes
        self.data_path = data_path
        self.benchmark_index = benchmark_index

    def optimize_rsi(self, rsi_buy_levels, rsi_sell_levels):
        """优化RSI策略参数"""
        # Create a parameterized RSI strategy class
        class RSICustomStrategy(BacktestEngine):
            def __init__(self, stock_code, data_path, initial_capital, benchmark_index, rsi_buy, rsi_sell):
                super().__init__(stock_code, data_path, initial_capital, benchmark_index)
                self.rsi_buy = rsi_buy
                self.rsi_sell = rsi_sell

            def generate_signals(self):
                df = self.data.copy()
                buy_signal = df['rsi1'] < self.rsi_buy
                sell_signal = df['rsi1'] > self.rsi_sell
                df['TRADE_SIGNAL'] = np.nan
                df.loc[buy_signal, 'TRADE_SIGNAL'] = 1
                df.loc[sell_signal, 'TRADE_SIGNAL'] = 0
                return df.dropna(subset=['TRADE_SIGNAL'])

        results = []
        for rsi_buy, rsi_sell in product(rsi_buy_levels, rsi_sell_levels):
            total_return = 0
            total_trades = 0
            total_win_rate = 0

            for code in self.stock_codes:
                engine = RSICustomStrategy(
                    stock_code=code,
                    data_path=self.data_path,
                    initial_capital=100000,
                    benchmark_index=self.benchmark_index,
                    rsi_buy=rsi_buy,
                    rsi_sell=rsi_sell
                )
                result = engine.run()
                total_return += result['total_return']
                total_trades += result['total_trades']
                total_win_rate += result['win_rate']

            avg_return = total_return / len(self.stock_codes)
            avg_win_rate = total_win_rate / len(self.stock_codes)

            results.append({
                'rsi_buy': rsi_buy,
                'rsi_sell': rsi_sell,
                'avg_return': avg_return,
                'avg_win_rate': avg_win_rate,
                'avg_trades': total_trades / len(self.stock_codes)
            })

        return pd.DataFrame(results).sort_values('avg_return', ascending=False)