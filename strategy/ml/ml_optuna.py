"""ML 超参优化 — Optuna 搜索 LightGBM 最优超参"""
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import optuna
import joblib
import lightgbm as lgb

from strategy.ml.trainer import MLRankerTrainer

logger = logging.getLogger(__name__)

# 搜索空间定义
ML_SEARCH_SPACE = {
    'num_leaves':            ('int', 31, 255),
    'learning_rate':         ('float', 0.01, 0.20),
    'n_estimators':          ('int', 100, 800),
    'min_child_samples':     ('int', 50, 500),
    'subsample':             ('float', 0.5, 1.0),
    'colsample_bytree':      ('float', 0.5, 1.0),
    'reg_alpha':             ('log_float', 1e-4, 1.0),
    'reg_lambda':            ('log_float', 1e-4, 1.0),
}


def sample_ml_params(trial: optuna.Trial) -> dict:
    """从 Optuna Trial 采样 LightGBM 超参。"""
    params = {}
    for name, (kind, low, high) in ML_SEARCH_SPACE.items():
        if kind == 'int':
            params[name] = trial.suggest_int(name, low, high)
        elif kind == 'float':
            params[name] = trial.suggest_float(name, low, high)
        elif kind == 'log_float':
            params[name] = trial.suggest_float(name, low, high, log=True)
    return params


def ml_objective(
    trial: optuna.Trial,
    train_start: str,
    train_end: str,
    cache_dir: str,
    factor_columns: list,
    purge_days: int = 5,
) -> float:
    """Optuna 目标函数: 采样ML超参 → 训练模型 → 返回验证集 RMSE。"""
    params = sample_ml_params(trial)
    trainer = MLRankerTrainer(cache_dir, factor_columns=factor_columns)
    X, y = trainer.build_dataset(train_start, train_end, purge_days=purge_days)

    # 时间序列分割: 前80%训练, 后20%验证 (加2% purge gap)
    purge_size = max(1, int(len(X) * 0.02))
    split_idx = int(len(X) * 0.8)
    val_start = min(split_idx + purge_size, len(X) - 1)

    if val_start >= len(X) or split_idx < 10:
        return 1e9

    X_train, X_val = X.iloc[:split_idx], X.iloc[val_start:]
    y_train, y_val = y[:split_idx], y[val_start:]

    model_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'early_stopping_rounds': 50,
        **params,
    }

    model = lgb.LGBMRegressor(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    best_rmse = model.best_score_['valid_0']['rmse']
    return best_rmse


def run_ml_optimization(
    train_start: str,
    train_end: str,
    cache_dir: str,
    factor_columns: list,
    n_trials: int = 50,
    output_dir: str = '.',
    study_name: Optional[str] = None,
    storage_url: Optional[str] = None,
) -> Tuple[str, float, optuna.Study]:
    """运行 ML 超参优化，返回 (model_path, best_rmse, study)。"""
    storage_kwargs = {}
    if storage_url:
        storage_kwargs['storage'] = storage_url

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=study_name,
        load_if_exists=True,
        **storage_kwargs,
    )

    obj = lambda trial: ml_objective(
        trial, train_start, train_end, cache_dir, factor_columns
    )
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

    # 用最优参数重新训练并保存模型
    best_params = study.best_params
    final_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        **best_params,
    }

    trainer = MLRankerTrainer(cache_dir, factor_columns=factor_columns)
    X, y = trainer.build_dataset(train_start, train_end)

    model = lgb.LGBMRegressor(**final_params)
    model.fit(X, y)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = str(output / 'best_model.pkl')
    joblib.dump(model, model_path)

    # 保存最优参数 JSON
    params_path = output / 'best_ml_params.json'
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train_start': train_start,
            'train_end': train_end,
            'best_rmse': study.best_value,
            'best_params': best_params,
            'n_factors': len(factor_columns),
        }, f, indent=2, ensure_ascii=False)

    # 保存 Trial 记录
    trials_path = output / 'ml_optuna_trials.csv'
    study.trials_dataframe().to_csv(trials_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*60}")
    print(f"ML 超参优化完成")
    print(f"最优 RMSE: {study.best_value:.6f}")
    print(f"模型已保存: {model_path}")
    print(f"{'='*60}")

    return model_path, study.best_value, study


def run_temporal_optimization(
    train_start: str,
    train_end: str,
    cache_dir: str,
    encoder_path: str,
    factor_columns: list,
    n_trials: int = 50,
    output_dir: str = '.',
    study_name: Optional[str] = None,
    storage_url: Optional[str] = None,
) -> Tuple[str, float, optuna.Study]:
    """时序模式下运行 ML 超参优化: 冻结 Encoder 提取特征 + Optuna 搜索 LightGBM。

    Returns:
        (model_path, best_rmse, study)
    """
    from strategy.ml.temporal.temporal_trainer import TemporalTrainer

    print(f"\n提取时序特征中...")
    # 用 TemporalTrainer 一次性提取时序特征
    tt = TemporalTrainer(cache_dir, encoder_path)

    # 加载数据并逐日提取特征
    import numpy as np
    from collections import deque

    all_dates = sorted([f.stem for f in Path(cache_dir, 'daily').glob('*.parquet')])
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    dates = [d for d in all_dates if train_start <= d <= train_end]

    purge_days = 5
    if purge_days > 0:
        end_idx = date_to_idx[dates[-1]]
        cutoff_idx = max(0, end_idx - purge_days)
        dates = [d for d in dates if d <= all_dates[cutoff_idx]]

    if len(dates) < 30:
        raise ValueError(f"训练日期不足: {len(dates)} 天")

    import pandas as pd
    history: dict = {}
    M = len(factor_columns)
    X_chunks = []
    y_chunks = []

    for i, date_str in enumerate(dates):
        idx = date_to_idx[date_str]
        if idx < tt.seq_len - 1:
            continue
        target_idx = idx + 5
        if target_idx >= len(all_dates):
            continue

        daily_path = tt.daily_dir / f'{date_str}.parquet'
        target_path = tt.daily_dir / f'{all_dates[target_idx]}.parquet'
        if not target_path.exists():
            continue

        df = pd.read_parquet(daily_path).set_index('stock_code')
        target_df = pd.read_parquet(target_path).set_index('stock_code')

        cols = [c for c in factor_columns if c in df.columns]
        if len(cols) < len(factor_columns) * 0.5:
            continue
        common = df.index.intersection(target_df.index)
        if len(common) < 100:
            continue

        for s in common:
            if s not in history:
                history[s] = deque(maxlen=tt.seq_len)
            history[s].append(np.array(
                [float(df.loc[s].get(c, 0.0)) for c in factor_columns],
                dtype=np.float32
            ))

        stock_list = list(common)
        hist_array = np.zeros((len(stock_list), tt.seq_len, M), dtype=np.float32)
        for j, s in enumerate(stock_list):
            h = list(history[s])
            for t_idx in range(min(len(h), tt.seq_len)):
                hist_array[j, tt.seq_len - len(h) + t_idx] = h[t_idx]

        # 归一化
        hist_array = (hist_array - tt.factor_mean) / tt.factor_std
        hist_array = np.nan_to_num(hist_array, nan=0.0, posinf=0.0, neginf=0.0)

        import torch
        with torch.no_grad():
            temporal_feats = tt.encoder(torch.tensor(hist_array)).numpy()

        cross_feats = np.array([
            [float(df.loc[s].get(c, 0.0)) for c in factor_columns]
            for s in stock_list
        ], dtype=np.float32)

        X = np.concatenate([cross_feats, temporal_feats], axis=1)
        y = np.array([
            (target_df.loc[s, 'close'] - df.loc[s, 'close']) / df.loc[s, 'close']
            for s in stock_list
        ], dtype=np.float32)

        mask = (y > -0.5) & (y < 0.5)
        if mask.sum() < 100:
            continue
        X_chunks.append(X[mask])
        y_chunks.append(y[mask])

        if (i + 1) % 50 == 0:
            logger.info(f"  特征提取: {i+1}/{len(dates)}, 样本: {sum(len(c) for c in X_chunks)}")

    X_all = np.vstack(X_chunks)
    y_all = np.concatenate(y_chunks)
    logger.info(f"时序特征提取完成: X={X_all.shape}")

    # Optuna 搜索 LightGBM 超参
    def temporal_objective(trial):
        params = sample_ml_params(trial)
        lgb_params = {
            'objective': 'regression', 'metric': 'rmse',
            'boosting_type': 'gbdt', 'verbosity': -1,
            'random_state': 42, 'early_stopping_rounds': 50, **params,
        }
        split_idx = int(len(X_all) * 0.8)
        purge_size = max(1, int(len(X_all) * 0.02))
        val_start = min(split_idx + purge_size, len(X_all) - 1)

        if val_start >= len(X_all) or split_idx < 10:
            return 1e9

        import lightgbm as lgb
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_all[:split_idx], y_all[:split_idx],
                  eval_set=[(X_all[val_start:], y_all[val_start:])])
        return model.best_score_['valid_0']['rmse']

    storage_kwargs = {}
    if storage_url:
        storage_kwargs['storage'] = storage_url

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=study_name,
        load_if_exists=True,
        **storage_kwargs,
    )
    study.optimize(temporal_objective, n_trials=n_trials, show_progress_bar=True)

    # 用最优参数训练最终模型
    best_params = study.best_params
    final_params = {
        'objective': 'regression', 'metric': 'rmse',
        'boosting_type': 'gbdt', 'verbosity': -1,
        'random_state': 42, **best_params,
    }
    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(X_all, y_all)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = str(output / 'best_model.pkl')
    joblib.dump(final_model, model_path)

    params_path = output / 'best_ml_params.json'
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train_start': train_start, 'train_end': train_end,
            'best_rmse': study.best_value, 'best_params': best_params,
            'n_factors': len(factor_columns), 'temporal_features': tt.d_model,
        }, f, indent=2, ensure_ascii=False)

    trials_path = output / 'ml_optuna_trials.csv'
    study.trials_dataframe().to_csv(trials_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*60}")
    print(f"时序增强 ML 优化完成")
    print(f"最优 RMSE: {study.best_value:.6f}")
    print(f"模型已保存: {model_path}")
    print(f"{'='*60}")

    return model_path, study.best_value, study
