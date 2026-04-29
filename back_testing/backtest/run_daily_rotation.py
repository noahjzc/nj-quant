"""每日轮动策略独立运行入口"""
import argparse
import json
import logging
from back_testing.rotation.daily_rotation_engine import DailyRotationEngine
from back_testing.rotation.config import RotationConfig, MarketRegimeConfig
from back_testing.analysis.performance_analyzer import PerformanceAnalyzer
from back_testing.data.daily_data_cache import DailyDataCache, CachedProvider
from back_testing.optimization.run_daily_rotation_optimization import _params_to_config
import pandas as pd


def run(start_date: str, end_date: str, config: RotationConfig = None, verbose: bool = False,
        cache_dir: str = None, config_file: str = None):
    """运行每日轮动回测"""
    # 配置日志
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    print(f"=" * 60)
    print(f"每日全市场轮动回测")
    print(f"区间: {start_date} ~ {end_date}")
    print(f"=" * 60)

    if config_file:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        params = data.get('params', data)
        config = _params_to_config(params)
        print(f"从配置文件加载: {config_file}")
    config = config or RotationConfig()
    data_provider = None

    if cache_dir:
        from pathlib import Path
        cache_path = Path(cache_dir)
        # 如果缓存已存在则直接复用，否则构建
        if not (cache_path / 'daily').is_dir():
            cache_path = Path(DailyDataCache.build(
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                benchmark_index=config.benchmark_index,
            ))
        cache = DailyDataCache(str(cache_path))
        data_provider = CachedProvider(cache)
        print(f"数据缓存就绪: {cache_path}")
        print(f"  {len(cache.stock_codes)} 只股票, {len(cache.trading_dates)} 个交易日")

    engine = DailyRotationEngine(config, start_date, end_date,
                                 data_provider=data_provider)
    results = engine.run()

    # 输出统计
    # 计算最终总资产：现金 + 持仓市值
    final_result = results[-1] if results else None
    final_total_asset = final_result.total_asset if final_result else engine.current_capital
    total_return = (final_total_asset / config.initial_capital - 1) if config else 0
    print(f"\n最终资产: {final_total_asset:,.2f}")
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

        # Convert TradeRecord to dict for PerformanceAnalyzer compatibility
        trades_dicts = [
            {'action': t.action.lower(), 'price': t.price, 'shares': t.shares,
             'return': 0.0}  # win_rate uses equity_curve; return from trades is secondary
            for t in engine.trade_history
        ]

        analyzer = PerformanceAnalyzer(
            trades=trades_dicts,
            initial_capital=config.initial_capital,
            equity_curve=[config.initial_capital] + df['total_asset'].tolist(),
            periods_per_year=252
        )
        perf = analyzer.calculate_metrics()
        print(f"\n绩效指标:")
        print(f"  年化收益率: {perf.get('annual_return', 0):.2%}")
        print(f"  Sharpe: {perf.get('sharpe_ratio', 0):.2f}")
        print(f"  最大回撤: {perf.get('max_drawdown', 0):.2%}")

    return engine, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='每日全市场轮动回测')
    parser.add_argument('--start', default='2024-01-01', help='开始日期')
    parser.add_argument('--end', default='2024-12-31', help='结束日期')
    parser.add_argument('--verbose', action='store_true', help='输出详细日志')
    parser.add_argument('--cache-dir', default='cache/daily_rotation',
                        help='Parquet 缓存目录（默认: cache/daily_rotation）')
    parser.add_argument('--no-cache', action='store_true', help='不使用缓存（每次从 DB 查询）')
    parser.add_argument('--config', default=None, help='参数 JSON 文件（如 output/best_params_xxx.json）')
    args = parser.parse_args()

    run(args.start, args.end, verbose=args.verbose,
        cache_dir=None if args.no_cache else args.cache_dir,
        config_file=args.config)
