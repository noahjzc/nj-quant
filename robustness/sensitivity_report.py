"""参数稳定性筛选 CLI

用法:
    python -m robustness.sensitivity_report \
        --params output/best_params_xxx.json \
        --start 2024-01-01 --end 2024-12-31

    python -m robustness.sensitivity_report \
        --params output/best_params_xxx.json \
        --start 2024-01-01 --end 2024-12-31 \
        --ml-model output/best_model.pkl
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='参数稳定性筛选 (Sensitivity Analysis)')
    parser.add_argument('--params', required=True, help='最优参数 JSON 路径 (best_params_xxx.json)')
    parser.add_argument('--start', required=True, help='回测开始日期')
    parser.add_argument('--end', required=True, help='回测结束日期')
    parser.add_argument('--cache-dir', default='cache/daily_rotation', help='Parquet 缓存目录')
    parser.add_argument('--ml-model', default=None, help='ML 模型路径 (MLRanker/TemporalMLRanker)')
    parser.add_argument('--factors', default=None, help='因子列表 JSON (selected_factors.json)')
    parser.add_argument('--output', default='.', help='结果输出目录')
    parser.add_argument('--perturbation', type=float, default=0.2, help='扰动比例 (默认 20%%)')

    args = parser.parse_args()

    from robustness.sensitivity import SensitivityAnalyzer
    from strategy.rotation.daily_rotation_engine import DailyRotationEngine
    from strategy.rotation.config import RotationConfig
    from backtesting.analysis.performance_analyzer import PerformanceAnalyzer
    from optimization.optuna.run_daily_rotation_optimization import _params_to_config
    from data.cache.daily_data_cache import DailyDataCache, CachedProvider

    # 加载参数
    with open(args.params, 'r', encoding='utf-8') as f:
        data = json.load(f)
    params = data.get('best_params', data)
    base_config = RotationConfig()

    # 覆盖因子方向
    if args.factors and Path(args.factors).exists():
        with open(args.factors, 'r', encoding='utf-8') as f:
            factor_data = json.load(f)
        raw_factors = factor_data.get('raw', [])
        if raw_factors:
            ic_path = Path(args.output) / 'factor_ic_report.csv'
            if ic_path.exists():
                import pandas as pd
                ic_df = pd.read_csv(ic_path, index_col=0)
                directions = {f: (1 if ic_df.loc[f, 'ic_mean'] > 0 else -1)
                             for f in raw_factors if f in ic_df.index}
            else:
                directions = {f: 1 for f in raw_factors}
            base_config.rank_factor_weights = {f: 1.0 / len(raw_factors) for f in raw_factors}
            base_config.rank_factor_directions = directions

    # 加载 ranker
    ranker = None
    if args.ml_model:
        model_path = Path(args.ml_model)
        encoder_path = model_path.parent / 'temporal_encoder.pt'
        if encoder_path.exists():
            from strategy.ml.temporal.temporal_ranker import TemporalMLRanker
            ranker = TemporalMLRanker(str(model_path), str(encoder_path))
            print(f"时序增强排名器已加载")
        else:
            from strategy.ml.ml_ranker import MLRanker
            ranker = MLRanker(str(model_path))
            print(f"ML 排名器已加载")

    # 数据
    cache_path = DailyDataCache.build(args.start, args.end, args.cache_dir)
    cache = DailyDataCache(cache_path)
    cached_provider = CachedProvider(cache)
    print(f"数据缓存就绪: {len(cache.stock_codes)} 只股票, {len(cache.trading_dates)} 个交易日")

    # engine_factory
    def engine_factory(p):
        config = _params_to_config(p, base_config)
        engine = DailyRotationEngine(config, args.start, args.end,
                                     data_provider=cached_provider, ranker=ranker)
        results = engine.run()
        if not results:
            return {'sharpe_ratio': 0.0}
        equity = [config.initial_capital] + [r.total_asset for r in results]
        analyzer = PerformanceAnalyzer(
            trades=[], initial_capital=config.initial_capital,
            equity_curve=equity, periods_per_year=252,
        )
        return analyzer.calculate_metrics()

    # 运行分析
    n_numeric = len([v for v in params.values() if isinstance(v, (int, float))])
    n_framework = len([k for k, v in params.items()
                       if isinstance(v, (int, float)) and not k.startswith('weight_')])
    logger.info(f"分析 {n_framework} 个框架参数 (跳过 {n_numeric - n_framework} 个因子权重)")
    logger.info(f"预计回测 {1 + n_framework * 2} 次")

    sa = SensitivityAnalyzer(perturbation_pct=args.perturbation)
    result = sa.run(params, engine_factory)

    # 打印结果
    print(f"\n{'='*60}")
    print(f"参数敏感性分析报告")
    print(f"区间: {args.start} ~ {args.end}")
    print(f"整体稳定性: {result.overall_stability_score:.4f}")
    print(f"{'='*60}")
    print(f"\n{'参数':<25} {'基准值':>10} {'Sharpe变化':>10} {'稳定性':>8}")
    print(f"{'-'*55}")
    for key, info in sorted(result.per_param.items(),
                             key=lambda x: x[1]['sharpe_change_pct'], reverse=True):
        status = '✓' if info['stable'] else '⚠'
        print(f"{key:<25} {info['base_value']:>10.4g} {info['sharpe_change_pct']:>9.1f}% {status:>7}")

    # 保存
    output = Path(args.output)
    import time
    report = {
        'date_range': f"{args.start} ~ {args.end}",
        'overall_stability': result.overall_stability_score,
        'per_param': {k: v for k, v in result.per_param.items()},
    }
    path = output / f'sensitivity_report_{int(time.time())}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {path}")


if __name__ == '__main__':
    main()
