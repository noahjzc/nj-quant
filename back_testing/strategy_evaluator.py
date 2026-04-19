import pandas as pd
import numpy as np
from datetime import timedelta
from back_testing.backtest_engine import BacktestEngine
from back_testing.ma_strategy import MAStrategy
from back_testing.macd_strategy import MACDStrategy
from back_testing.rsi_strategy import RSIReversalStrategy
from back_testing.combined_strategy import CombinedStrategy
from back_testing.bollinger_strategy import BollingerStrategy, BollingerStrictStrategy
from back_testing.kdj_strategy import KDJOversoldStrategy, KDJGoldenCrossStrategy
from back_testing.multi_rsi_strategy import MultiPeriodRSIStrategy, RSIReversalMultiStrategy
from back_testing.trend_confirmation_strategy import TrendConfirmationStrategy, TrendPullbackStrategy
from back_testing.volume_strategy import VolumeAnomalyStrategy, VolumeMAConfirmStrategy

STRATEGY_MAP = {
    'MAStrategy': MAStrategy,
    'MACDStrategy': MACDStrategy,
    'RSIReversalStrategy': RSIReversalStrategy,
    'CombinedStrategy': CombinedStrategy,
    'BollingerStrategy': BollingerStrategy,
    'BollingerStrictStrategy': BollingerStrictStrategy,
    'KDJOversoldStrategy': KDJOversoldStrategy,
    'KDJGoldenCrossStrategy': KDJGoldenCrossStrategy,
    'MultiPeriodRSIStrategy': MultiPeriodRSIStrategy,
    'RSIReversalMultiStrategy': RSIReversalMultiStrategy,
    'TrendConfirmationStrategy': TrendConfirmationStrategy,
    'TrendPullbackStrategy': TrendPullbackStrategy,
    'VolumeAnomalyStrategy': VolumeAnomalyStrategy,
    'VolumeMAConfirmStrategy': VolumeMAConfirmStrategy,
}


class StrategyEvaluator:
    """策略评估器：评估各策略过去N周的表现"""

    def __init__(self, stock_codes: list, data_path: str, initial_capital: float = 100000.0):
        self.stock_codes = stock_codes
        self.data_path = data_path
        self.initial_capital = initial_capital

    def get_recent_trading_dates(self, reference_date: pd.Timestamp, weeks: int = 4) -> tuple:
        """获取最近N周的交易日范围"""
        start_date = reference_date - timedelta(weeks=weeks * 7)
        return start_date, reference_date

    def evaluate_strategy(self, strategy_name: str, weeks: int = 4,
                         reference_date: pd.Timestamp = None) -> dict:
        """评估单个策略在过去N周的表现"""
        if reference_date is None:
            reference_date = pd.Timestamp.now()

        start_date, end_date = self.get_recent_trading_dates(reference_date, weeks)

        strategy_class = STRATEGY_MAP.get(strategy_name)
        if strategy_class is None:
            return {'avg_return': 0, 'stock_returns': [], 'num_trades': 0}

        stock_returns = []
        total_trades = 0

        for code in self.stock_codes:
            try:
                engine = strategy_class(
                    stock_code=code,
                    data_path=self.data_path,
                    initial_capital=self.initial_capital,
                    start_date=start_date.strftime('%Y-%m-%d')
                )
                result = engine.run()
                stock_returns.append(result['total_return'])
                total_trades += result['total_trades']
            except Exception:
                continue

        avg_return = np.mean(stock_returns) if stock_returns else 0

        return {
            'avg_return': avg_return,
            'stock_returns': stock_returns,
            'num_trades': total_trades,
            'start_date': start_date,
            'end_date': end_date
        }

    def evaluate_all_strategies(self, weeks: int = 4,
                                 reference_date: pd.Timestamp = None) -> pd.DataFrame:
        """评估所有策略的表现，返回排名"""
        results = []
        for strategy_name in STRATEGY_MAP.keys():
            eval_result = self.evaluate_strategy(strategy_name, weeks, reference_date)
            results.append({
                'strategy_name': strategy_name,
                'avg_return': eval_result['avg_return'],
                'num_trades': eval_result['num_trades'],
                'start_date': eval_result['start_date'],
                'end_date': eval_result['end_date']
            })

        df = pd.DataFrame(results)
        df = df.sort_values('avg_return', ascending=False)
        return df