"""每日轮动策略独立运行入口"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import os
from datetime import datetime

import pandas as pd

from strategy.rotation.daily_rotation_engine import DailyRotationEngine
from strategy.rotation.config import RotationConfig, MarketRegimeConfig
from backtesting.analysis.performance_analyzer import PerformanceAnalyzer
from backtesting.analysis.visualizer import PerformanceVisualizer
from data.cache.daily_data_cache import DailyDataCache, CachedProvider
from optimization.optuna.run_daily_rotation_optimization import _params_to_config


def run(start_date: str, end_date: str, config: RotationConfig = None, verbose: bool = False,
        cache_dir: str = None, config_file: str = None, ml_model: str = None):
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

    ranker = None
    if ml_model:
        from strategy.ml.ml_ranker import MLRanker
        ranker = MLRanker(ml_model)
        print(f"ML 排名器已加载: {ml_model}")

    engine = DailyRotationEngine(config, start_date, end_date,
                                 data_provider=data_provider, ranker=ranker)
    results = engine.run()

    # 绩效分析 & 导出
    if results:
        _export_results(results, engine, config, start_date, end_date)

    return engine, results


def _export_results(results, engine, config, start_date, end_date):
    """导出回测结果到时间戳文件夹。"""
    ts = datetime.now().strftime('%Y_%m_%d_%H_%M')
    out_dir = Path(f'results/{ts}_performance')
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 构建净值 DataFrame ──
    equity_df = pd.DataFrame([{
        'date': r.date, 'total_asset': r.total_asset, 'cash': r.cash,
        'position_value': r.total_asset - r.cash,
        'n_positions': len(r.positions), 'regime': r.market_regime,
    } for r in results])
    equity_df.to_csv(out_dir / 'equity_curve.csv', index=False, encoding='utf-8-sig')

    # ── 2. 交易明细 CSV ──
    trades_df = pd.DataFrame([{
        'date': t.date, 'stock_code': t.stock_code, 'action': t.action,
        'price': t.price, 'shares': t.shares, 'cost': t.cost,
        'capital_before': t.capital_before,
    } for t in engine.trade_history])
    if not trades_df.empty:
        trades_df.to_csv(out_dir / 'trades.csv', index=False, encoding='utf-8-sig')

    # ── 3. 每日持仓 CSV ──
    pos_rows = []
    for r in results:
        for code, pos in r.positions.items():
            pos_rows.append({
                'date': r.date, 'stock_code': code, 'shares': pos.shares,
                'buy_price': pos.buy_price, 'buy_date': pos.buy_date,
                'highest_price': pos.highest_price,
            })
    if pos_rows:
        pd.DataFrame(pos_rows).to_csv(out_dir / 'positions_daily.csv', index=False, encoding='utf-8-sig')

    # ── 4. 绩效指标 JSON ──
    analyzer = PerformanceAnalyzer(
        trades=[{'action': t.action.lower(), 'price': t.price, 'shares': t.shares, 'return': 0.0}
                for t in engine.trade_history],
        initial_capital=config.initial_capital,
        equity_curve=[config.initial_capital] + equity_df['total_asset'].tolist(),
        periods_per_year=252,
    )
    metrics = analyzer.calculate_metrics()
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    # ── 7. 稳健性检验
    from robustness.robustness_analyzer import RobustnessAnalyzer

    robust = RobustnessAnalyzer(analyzer)
    report = robust.run_all()

    robust_data = {
        'monte_carlo': {
            'mean_sharpe': report.monte_carlo.mean_sharpe,
            'sharpe_95ci_low': report.monte_carlo.sharpe_95ci[0],
            'sharpe_95ci_high': report.monte_carlo.sharpe_95ci[1],
            'mean_max_dd': report.monte_carlo.mean_max_dd,
            'max_dd_95ci_low': report.monte_carlo.max_dd_95ci[0],
            'max_dd_95ci_high': report.monte_carlo.max_dd_95ci[1],
        },
        'cscv': {
            'overfit_probability': report.cscv.overfit_probability,
            'rank_decay': report.cscv.rank_decay,
            'is_robust': report.cscv.is_robust,
        },
        'deflated_sharpe': report.deflated_sharpe,
        'summary': report.summary,
    }
    with open(out_dir / 'robustness.json', 'w', encoding='utf-8') as f:
        json.dump(robust_data, f, indent=2, ensure_ascii=False, default=str)

    # ── 5. HTML 报告 + 图表 ──
    equity_curve = pd.Series(
        [config.initial_capital] + equity_df['total_asset'].tolist(),
        index=pd.to_datetime([start_date] + equity_df['date'].tolist()),
        name='equity',
    )
    try:
        benchmark_df = engine.data_provider.get_index_data(
            config.benchmark_index,
            start_date=start_date,
            end_date=end_date,
        )
        if not benchmark_df.empty and 'close' in benchmark_df.columns:
            benchmark_norm = benchmark_df['close'] / benchmark_df['close'].iloc[0] * config.initial_capital
            benchmark_curve = benchmark_norm.rename('benchmark')
        else:
            benchmark_curve = None
    except Exception:
        benchmark_curve = None

    visualizer = PerformanceVisualizer(equity_curve, benchmark_curve)
    visualizer.generate_report(
        trades=[{'return': 0.0} for _ in engine.trade_history],
        save_dir=str(out_dir),
    )

    # ── 6. 控制台摘要 ──
    final_result = results[-1]
    total_return = (final_result.total_asset / config.initial_capital - 1)
    print(f"\n最终资产: {final_result.total_asset:,.2f}")
    print(f"总收益率: {total_return:.2%}")
    print(f"交易次数: {len(engine.trade_history)}")
    print(f"年化收益: {metrics.get('annual_return', 0):.2%}  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}  最大回撤: {metrics.get('max_drawdown', 0):.2%}")
    print(f"结果已导出: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='每日全市场轮动回测')
    parser.add_argument('--start', default='2024-01-01', help='开始日期')
    parser.add_argument('--end', default='2024-12-31', help='结束日期')
    parser.add_argument('--verbose', action='store_true', help='输出详细日志')
    parser.add_argument('--cache-dir', default='cache/daily_rotation',
                        help='Parquet 缓存目录（默认: cache/daily_rotation）')
    parser.add_argument('--no-cache', action='store_true', help='不使用缓存（每次从 DB 查询）')
    parser.add_argument('--config', default=None, help='参数 JSON 文件（如 output/best_params_xxx.json）')
    parser.add_argument('--ml-model', default=None, help='ML 模型路径（使用 MLRanker 替代 SignalRanker）')
    args = parser.parse_args()

    run(args.start, args.end, verbose=args.verbose,
        cache_dir=None if args.no_cache else args.cache_dir,
        config_file=args.config, ml_model=args.ml_model)
