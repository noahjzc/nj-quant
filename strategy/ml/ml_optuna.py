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
