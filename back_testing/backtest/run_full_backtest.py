"""
完整回测系统入口
支持：单策略回测、组合回测、参数优化、可视化
"""
import argparse
import matplotlib
matplotlib.use('Agg')

from back_testing.strategies.ma_strategy import MAStrategy
from back_testing.strategies.macd_strategy import MACDStrategy
from back_testing.strategies.rsi_strategy import RSIReversalStrategy
from back_testing.strategies.combined_strategy import CombinedStrategy
from back_testing.portfolio_backtest import PortfolioBacktest
from back_testing.core.portfolio_optimizer import PortfolioOptimizer
from back_testing.strategies.bollinger_strategy import BollingerStrategy, BollingerStrictStrategy
from back_testing.strategies.kdj_strategy import KDJOversoldStrategy, KDJGoldenCrossStrategy
from back_testing.strategies.multi_rsi_strategy import MultiPeriodRSIStrategy, RSIReversalMultiStrategy
from back_testing.strategies.trend_confirmation_strategy import TrendConfirmationStrategy, TrendPullbackStrategy
from back_testing.strategies.volume_strategy import VolumeAnomalyStrategy, VolumeMAConfirmStrategy

STOCK_POOL = [
    ('sh600519', '贵州茅台', '白酒'),
    ('sh600036', '招商银行', '银行'),
    ('sh601318', '中国平安', '保险'),
    ('sh688256', '寒武纪', '科创板'),
    ('sz300750', '宁德时代', '电池'),
    ('sz000001', '平安银行', '银行'),
    ('sh601899', '紫金矿业', '贵金属'),
    ('sz300059', '东方财富', '金融'),
]

DATA_PATH = r'D:\workspace\code\mine\quant\data\metadata\daily_ycz'
BENCHMARK = 'sh000001'

def run_single_stock(strategy_class, stock_code, stock_name, start_date=None):
    """运行单股票回测"""
    engine = strategy_class(stock_code, DATA_PATH, 100000, BENCHMARK, start_date=start_date)
    result = engine.run()
    engine.print_result(result, strategy_class.__name__)
    return result

def run_portfolio(strategy_class, start_date=None):
    """运行组合回测"""
    pb = PortfolioBacktest(strategy_class, STOCK_POOL, DATA_PATH, start_date=start_date)
    return pb.run()

def main():
    parser = argparse.ArgumentParser(description='量化策略回测系统')
    parser.add_argument('--mode', choices=['single', 'portfolio', 'optimize'], default='portfolio')
    parser.add_argument('--strategy', choices=[
        'ma', 'macd', 'rsi', 'combined',
        'bollinger', 'bollinger_strict',
        'kdj_oversold', 'kdj_golden',
        'multi_rsi', 'multi_rsi_v2',
        'trend_confirm', 'trend_pullback',
        'volume', 'volume_ma'
    ], default='ma')
    parser.add_argument('--start', default='2020-01-01', help='回测开始日期，如 2020-01-01')
    args = parser.parse_args()

    strategies = {
        'ma': MAStrategy,
        'macd': MACDStrategy,
        'rsi': RSIReversalStrategy,
        'combined': CombinedStrategy,
        'bollinger': BollingerStrategy,
        'bollinger_strict': BollingerStrictStrategy,
        'kdj_oversold': KDJOversoldStrategy,
        'kdj_golden': KDJGoldenCrossStrategy,
        'multi_rsi': MultiPeriodRSIStrategy,
        'multi_rsi_v2': RSIReversalMultiStrategy,
        'trend_confirm': TrendConfirmationStrategy,
        'trend_pullback': TrendPullbackStrategy,
        'volume': VolumeAnomalyStrategy,
        'volume_ma': VolumeMAConfirmStrategy,
    }

    strategy_class = strategies[args.strategy]

    if args.mode == 'single':
        run_single_stock(strategy_class, 'sh600519', '贵州茅台', start_date=args.start)
    elif args.mode == 'portfolio':
        result = run_portfolio(strategy_class, start_date=args.start)
        print(f'组合总收益: {result["total_return"]:.2%}')
    elif args.mode == 'optimize':
        print('优化功能待实现')

if __name__ == '__main__':
    main()