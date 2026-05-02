"""ML 超参优化 CLI

用法:
    python optimization/optuna/run_ml_optimization.py \
        --train-start 2020-01-01 --train-end 2022-12-31 \
        --cache-dir cache/daily_rotation \
        --factors output/selected_factors.json \
        --trials 50 --output output/
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strategy.ml.ml_optuna import run_ml_optimization
from strategy.factors.alpha158 import Alpha158Calculator


def _get_all_alpha_columns() -> list:
    """获取全部 Alpha158 因子列名。"""
    import pandas as pd
    calc = Alpha158Calculator()
    dummy = pd.DataFrame({
        'open': [10.0, 10.1, 10.2], 'high': [10.3, 10.2, 10.4],
        'low': [9.8, 9.9, 9.7], 'close': [10.0, 10.1, 10.0],
        'volume': [1e6, 1.1e6, 0.9e6],
    })
    return list(calc.compute(dummy).columns)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
    )

    parser = argparse.ArgumentParser(description='ML 超参优化 (LightGBM + Optuna)')
    parser.add_argument('--train-start', required=True, help='训练开始日期')
    parser.add_argument('--train-end', required=True, help='训练截止日期')
    parser.add_argument('--cache-dir', default='cache/daily_rotation', help='Parquet缓存目录')
    parser.add_argument('--factors', default=None,
                        help='因子列表JSON路径 (selected_factors.json)，不指定则用全量Alpha158因子')
    parser.add_argument('--factor-type', choices=['raw', 'orthogonal'], default='orthogonal',
                        help='使用哪套因子: raw=原始精选, orthogonal=正交化')
    parser.add_argument('--trials', type=int, default=50, help='Optuna Trial 数')
    parser.add_argument('--output', default='.', help='输出目录')
    parser.add_argument('--storage', default=None, help='Optuna 持久化存储URL')
    parser.add_argument('--study-name', default=None, help='Optuna Study 名称')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--encoder', default=None,
                        help='预训练 TemporalEncoder 路径 (.pt)，用于时序特征增强')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 加载因子列表
    if args.factors and Path(args.factors).exists():
        with open(args.factors, 'r', encoding='utf-8') as f:
            factor_data = json.load(f)
        factor_columns = factor_data[args.factor_type]
        print(f"因子列: {len(factor_columns)} ({args.factor_type})")
    else:
        factor_columns = _get_all_alpha_columns()
        print(f"因子列: {len(factor_columns)} (全量 Alpha158)")

    if args.encoder and Path(args.encoder).exists():
        # 时序模式: Encoder 提取特征 + LightGBM 直接训练（不经过 Optuna）
        from strategy.ml.temporal.temporal_trainer import TemporalTrainer
        trainer = TemporalTrainer(args.cache_dir, args.encoder)
        model_path = trainer.train(
            train_start=args.train_start,
            train_end=args.train_end,
            output_path=args.output,
        )
        print(f"\n时序增强模型已保存: {model_path}")
        return  # 时序模式不跑 Optuna，直接结束

    run_ml_optimization(
        train_start=args.train_start,
        train_end=args.train_end,
        cache_dir=args.cache_dir,
        factor_columns=factor_columns,
        n_trials=args.trials,
        output_dir=args.output,
        study_name=args.study_name,
        storage_url=args.storage,
    )


if __name__ == '__main__':
    main()
