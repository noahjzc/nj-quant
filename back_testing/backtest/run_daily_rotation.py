"""每日轮动策略独立运行入口"""
import argparse
from back_testing.rotation.daily_rotation_engine import DailyRotationEngine
from back_testing.rotation.config import RotationConfig, MarketRegimeConfig
from back_testing.analysis.performance_analyzer import PerformanceAnalyzer
import pandas as pd


def run(start_date: str, end_date: str, config: RotationConfig = None):
    """运行每日轮动回测"""
    print(f"=" * 60)
    print(f"每日全市场轮动回测")
    print(f"区间: {start_date} ~ {end_date}")
    print(f"=" * 60)

    config = config or RotationConfig()
    engine = DailyRotationEngine(config, start_date, end_date)
    results = engine.run()

    # 输出统计
    total_return = (engine.current_capital / config.initial_capital - 1) if config else 0
    print(f"\n最终资产: {engine.current_capital:,.2f}")
    print(f"总收益率: {total_return:.2%}")
    print(f"交易次数: {len(engine.trade_history)}")

    # 绩效分析
    if results:
        df = pd.DataFrame([{
            'date': r.date,
            'total_asset': r.total_asset,
            'cash': r.cash,
            'position_value': r.total_asset - r.cash,
            'n_positions': len(r.positions),
            'regime': r.market_regime,
        } for r in results])

        analyzer = PerformanceAnalyzer(initial_capital=config.initial_capital if config else 1_000_000)
        perf = analyzer.analyze(df.set_index('date')['total_asset'])
        print(f"\n绩效指标:")
        print(f"  年化收益率: {perf.get('annual_return', 0):.2%}")
        print(f"  Sharpe: {perf.get('sharpe_ratio', 0):.2f}")
        print(f"  最大回撤: {perf.get('max_drawdown', 0):.2%}")

    return engine, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='每日全市场轮动回测')
    parser.add_argument('--start', default='2024-01-01', help='开始日期')
    parser.add_argument('--end', default='2024-12-31', help='结束日期')
    args = parser.parse_args()

    run(args.start, args.end)
