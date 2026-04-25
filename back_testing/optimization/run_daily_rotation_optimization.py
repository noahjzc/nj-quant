"""Daily Rotation 参数优化 — Optuna 贝叶斯优化

用法:
    # 单期优化
    python back_testing/optimization/run_daily_rotation_optimization.py \
        --mode single --start 2024-01-01 --end 2024-12-31 --trials 100

    # Walk-Forward 优化
    python back_testing/optimization/run_daily_rotation_optimization.py \
        --mode walkforward --start 2022-01-01 --end 2024-12-31 --trials 50
"""
import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import optuna
from back_testing.rotation.config import RotationConfig
from back_testing.rotation.daily_rotation_engine import DailyRotationEngine

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# 绩效计算
# ═══════════════════════════════════════════════

def max_drawdown(equity: List[float]) -> float:
    """从净值序列计算最大回撤"""
    if len(equity) < 2:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_sharpe(equity: List[float], periods_per_year: int = 252) -> float:
    """从净值序列计算年化 Sharpe Ratio（无风险利率=0）"""
    if len(equity) < 2:
        return 0.0
    values = np.array(equity)
    returns = np.diff(values) / values[:-1]
    returns = returns[~np.isnan(returns)]
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


# ═══════════════════════════════════════════════
# 参数采样
# ═══════════════════════════════════════════════

# 可优化信号的完整列表
ALL_SIGNAL_TYPES = ['KDJ_GOLD', 'MACD_GOLD', 'MA_GOLD', 'VOL_GOLD', 'BOLL_BREAK', 'HIGH_BREAK']

# 固定参数（不参与优化）
FIXED_FACTOR_DIRECTIONS = {
    'RSI_1': 1, 'RET_20': 1, 'VOLUME_RATIO': 1,
    'PB': -1, 'PE_TTM': -1, 'OVERHEAT': -1,
}


def sample_config(trial: optuna.Trial, base_config: RotationConfig = None) -> RotationConfig:
    """从 Optuna Trial 采样一个 RotationConfig

    Args:
        trial: Optuna trial 对象
        base_config: 基础配置（固定参数从此继承），None 则用默认值

    Returns:
        采样后的 RotationConfig
    """
    base = base_config or RotationConfig()

    # --- 因子权重：独立采样后归一化 ---
    raw_weights = {}
    for factor in FIXED_FACTOR_DIRECTIONS:
        raw_weights[factor] = trial.suggest_float(f'weight_{factor}', 0.01, 0.40)
    total = sum(raw_weights.values())
    rank_factor_weights = {k: v / total for k, v in raw_weights.items()}

    # --- 买入信号开关（至少保留一个） ---
    active_signals = []
    for sig in ALL_SIGNAL_TYPES:
        on = trial.suggest_categorical(f'signal_{sig}', ['on', 'off'])
        if on == 'on':
            active_signals.append(sig)
    # 始终建议 fallback 信号以保持参数空间一致性（TPE 需要）
    fallback_signal = trial.suggest_categorical('fallback_signal', ALL_SIGNAL_TYPES)
    if not active_signals:
        active_signals.append(fallback_signal)

    # --- 信号逻辑模式 ---
    buy_signal_mode = trial.suggest_categorical('buy_signal_mode', ['OR', 'AND'])

    # --- 连续参数 ---
    max_total_pct = trial.suggest_float('max_total_pct', 0.30, 1.00)
    max_position_pct = trial.suggest_float('max_position_pct', 0.05, 0.30)
    overheat_rsi_threshold = trial.suggest_float('overheat_rsi_threshold', 60.0, 90.0)
    overheat_ret5_threshold = trial.suggest_float('overheat_ret5_threshold', 0.05, 0.30)
    stop_loss_mult = trial.suggest_float('stop_loss_mult', 1.0, 3.5)
    take_profit_mult = trial.suggest_float('take_profit_mult', 2.0, 5.0)
    trailing_pct = trial.suggest_float('trailing_pct', 0.05, 0.20)
    trailing_start = trial.suggest_float('trailing_start', 0.02, 0.10)

    # --- 整数参数 ---
    max_positions = trial.suggest_int('max_positions', 3, 10)
    atr_period = trial.suggest_int('atr_period', 7, 21)

    return RotationConfig(
        initial_capital=base.initial_capital,
        max_total_pct=max_total_pct,
        max_position_pct=max_position_pct,
        max_positions=max_positions,
        buy_signal_types=active_signals,
        buy_signal_mode=buy_signal_mode,
        sell_signal_types=base.sell_signal_types,
        rank_factor_weights=rank_factor_weights,
        rank_factor_directions=FIXED_FACTOR_DIRECTIONS,
        market_regime=base.market_regime,
        exclude_st=base.exclude_st,
        exclude_limit_up=base.exclude_limit_up,
        exclude_limit_down=base.exclude_limit_down,
        exclude_suspended=base.exclude_suspended,
        benchmark_index=base.benchmark_index,
        atr_period=atr_period,
        stop_loss_mult=stop_loss_mult,
        take_profit_mult=take_profit_mult,
        trailing_pct=trailing_pct,
        trailing_start=trailing_start,
        overheat_rsi_threshold=overheat_rsi_threshold,
        overheat_ret5_threshold=overheat_ret5_threshold,
    )


# ═══════════════════════════════════════════════
# 目标函数
# ═══════════════════════════════════════════════

MAX_DRAWDOWN_LIMIT = 0.30


def objective(trial: optuna.Trial,
              start_date: str,
              end_date: str,
              base_config: RotationConfig = None) -> float:
    """Optuna 目标函数：给定 Trial，运行回测，返回年化 Sharpe

    Args:
        trial: Optuna trial
        start_date: 回测开始日期 'YYYY-MM-DD'
        end_date: 回测结束日期 'YYYY-MM-DD'
        base_config: 基础配置

    Returns:
        年化 Sharpe Ratio（max_drawdown > 30% 时返回 0）
    """
    config = sample_config(trial, base_config)

    try:
        engine = DailyRotationEngine(config, start_date, end_date)
        results = engine.run()

        if not results or len(results) < 2:
            return 0.0

        equity = [config.initial_capital] + [r.total_asset for r in results]

        dd = max_drawdown(equity)
        if dd > MAX_DRAWDOWN_LIMIT:
            return 0.0

        sharpe = compute_sharpe(equity, periods_per_year=252)
        return sharpe if not math.isnan(sharpe) else 0.0

    except Exception:
        logger.debug(f"Trial {trial.number} failed", exc_info=True)
        return 0.0


# ═══════════════════════════════════════════════
# Walk-Forward 窗口生成
# ═══════════════════════════════════════════════

def generate_windows(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_months: int = 12,
    test_months: int = 6,
    step_months: int = 3,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """生成 Walk-Forward 滚动窗口

    Returns:
        List of (train_start, train_end, test_start, test_end)
    """
    windows = []
    train_start = start_date
    while True:
        train_end = train_start + pd.DateOffset(months=train_months) - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.DateOffset(days=1)

        if test_end > end_date:
            break

        windows.append((train_start, train_end, test_start, test_end))
        train_start = train_start + pd.DateOffset(months=step_months)

    return windows


# ═══════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════

def run_single_optimization(
    start_date: str,
    end_date: str,
    n_trials: int = 100,
    base_config: RotationConfig = None,
    study_name: str = None,
    output_dir: str = None,
) -> Tuple[RotationConfig, float, optuna.Study]:
    """单期优化：在固定日期范围内搜索最优参数

    Returns:
        (best_config, best_sharpe, study)
    """
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=study_name,
    )

    obj = lambda trial: objective(trial, start_date, end_date, base_config)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

    # 从 best_params 重建 best_config
    best_config = _params_to_config(study.best_params, base_config)
    best_sharpe = study.best_value

    print(f"\n{'=' * 60}")
    print(f"单期优化完成")
    print(f"区间: {start_date} ~ {end_date}")
    print(f"最优 Sharpe: {best_sharpe:.4f}")
    print(f"{'=' * 60}")

    # 保存结果
    _save_results(study, best_config, best_sharpe, start_date, end_date, output_dir)

    return best_config, best_sharpe, study


def run_walk_forward(
    start_date: str,
    end_date: str,
    n_trials: int = 100,
    base_config: RotationConfig = None,
    train_months: int = 12,
    test_months: int = 6,
    step_months: int = 3,
    output_dir: str = None,
) -> List[Dict]:
    """Walk-Forward 滚动优化

    Returns:
        List of dicts，每个窗口一条记录
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    windows = generate_windows(start_ts, end_ts, train_months, test_months, step_months)

    if not windows:
        print("错误: 日期范围内无法生成 Walk-Forward 窗口")
        return []

    print(f"\nWalk-Forward 优化: {len(windows)} 个窗口")
    print(f"训练={train_months}月 测试={test_months}月 步进={step_months}月")

    records = []
    for i, (train_s, train_e, test_s, test_e) in enumerate(windows):
        print(f"\n{'─' * 50}")
        print(f"窗口 {i + 1}/{len(windows)}")
        print(f"训练: {train_s.strftime('%Y-%m-%d')} ~ {train_e.strftime('%Y-%m-%d')}")
        print(f"测试: {test_s.strftime('%Y-%m-%d')} ~ {test_e.strftime('%Y-%m-%d')}")

        # 训练期优化
        train_start_str = train_s.strftime('%Y-%m-%d')
        train_end_str = train_e.strftime('%Y-%m-%d')
        best_config, train_sharpe, study = run_single_optimization(
            train_start_str, train_end_str, n_trials, base_config,
            study_name=f"wf_window_{i}",
            output_dir=output_dir,
        )

        # 测试期评估
        test_start_str = test_s.strftime('%Y-%m-%d')
        test_end_str = test_e.strftime('%Y-%m-%d')
        test_sharpe = _evaluate_on_test(best_config, test_start_str, test_end_str)

        record = {
            'window': i,
            'train_start': train_start_str,
            'train_end': train_end_str,
            'test_start': test_start_str,
            'test_end': test_end_str,
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'best_params': _config_to_dict(best_config),
        }
        records.append(record)

    # 汇总
    _print_wf_summary(records)
    _save_wf_results(records, output_dir)

    return records


# ═══════════════════════════════════════════════
# 内部辅助函数
# ═══════════════════════════════════════════════

def _params_to_config(params: Dict, base_config: RotationConfig = None) -> RotationConfig:
    """从 Optuna best_params 字典重建 RotationConfig（不依赖 Trial）"""
    base = base_config or RotationConfig()

    # 重建因子权重
    raw_weights = {factor: params[f'weight_{factor}']
                   for factor in FIXED_FACTOR_DIRECTIONS}
    total = sum(raw_weights.values())
    rank_factor_weights = {k: v / total for k, v in raw_weights.items()}

    # 重建信号列表
    active_signals = [sig for sig in ALL_SIGNAL_TYPES
                      if params.get(f'signal_{sig}') == 'on']
    if not active_signals:
        active_signals = [params.get('fallback_signal', ALL_SIGNAL_TYPES[0])]

    return RotationConfig(
        initial_capital=base.initial_capital,
        max_total_pct=params['max_total_pct'],
        max_position_pct=params['max_position_pct'],
        max_positions=params['max_positions'],
        buy_signal_types=active_signals,
        buy_signal_mode=params['buy_signal_mode'],
        sell_signal_types=base.sell_signal_types,
        rank_factor_weights=rank_factor_weights,
        rank_factor_directions=FIXED_FACTOR_DIRECTIONS,
        market_regime=base.market_regime,
        exclude_st=base.exclude_st,
        exclude_limit_up=base.exclude_limit_up,
        exclude_limit_down=base.exclude_limit_down,
        exclude_suspended=base.exclude_suspended,
        benchmark_index=base.benchmark_index,
        atr_period=params['atr_period'],
        stop_loss_mult=params['stop_loss_mult'],
        take_profit_mult=params['take_profit_mult'],
        trailing_pct=params['trailing_pct'],
        trailing_start=params['trailing_start'],
        overheat_rsi_threshold=params['overheat_rsi_threshold'],
        overheat_ret5_threshold=params['overheat_ret5_threshold'],
    )


def _evaluate_on_test(config: RotationConfig, start: str, end: str) -> float:
    """在测试集上评估给定配置，返回 Sharpe"""
    try:
        engine = DailyRotationEngine(config, start, end)
        results = engine.run()
        if not results or len(results) < 2:
            return 0.0
        equity = [config.initial_capital] + [r.total_asset for r in results]
        return compute_sharpe(equity)
    except Exception:
        return 0.0


def _config_to_dict(config: RotationConfig) -> Dict:
    """将 RotationConfig 序列化为可 JSON 化的 dict"""
    return {
        'max_total_pct': config.max_total_pct,
        'max_position_pct': config.max_position_pct,
        'max_positions': config.max_positions,
        'buy_signal_types': config.buy_signal_types,
        'buy_signal_mode': config.buy_signal_mode,
        'rank_factor_weights': config.rank_factor_weights,
        'atr_period': config.atr_period,
        'stop_loss_mult': config.stop_loss_mult,
        'take_profit_mult': config.take_profit_mult,
        'trailing_pct': config.trailing_pct,
        'trailing_start': config.trailing_start,
        'overheat_rsi_threshold': config.overheat_rsi_threshold,
        'overheat_ret5_threshold': config.overheat_ret5_threshold,
    }


def _save_results(study: optuna.Study, best_config: RotationConfig,
                  best_sharpe: float, start: str, end: str,
                  output_dir: str = None):
    """保存优化结果到 JSON 和 CSV"""
    output_dir = Path(output_dir or '.')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 最优参数
    best_path = output_dir / 'best_params.json'
    best_data = {
        'start_date': start,
        'end_date': end,
        'best_sharpe': best_sharpe,
        'best_params': _config_to_dict(best_config),
    }
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(best_data, f, indent=2, ensure_ascii=False)
    print(f"最优参数已保存: {best_path}")

    # Trial 记录
    trials_path = output_dir / 'optuna_trials.csv'
    df = study.trials_dataframe()
    df.to_csv(trials_path, index=False, encoding='utf-8-sig')
    print(f"Trial 记录已保存: {trials_path} ({len(df)} 条)")


def _print_wf_summary(records: List[Dict]):
    """打印 Walk-Forward 汇总"""
    print(f"\n{'=' * 60}")
    print("Walk-Forward 汇总")
    print(f"{'=' * 60}")
    test_sharpes = [r['test_sharpe'] for r in records]
    positive_count = sum(1 for s in test_sharpes if s > 0)
    print(f"窗口数: {len(records)}")
    print(f"测试 Sharpe 均值: {np.mean(test_sharpes):.4f}")
    print(f"测试 Sharpe 中位数: {np.median(test_sharpes):.4f}")
    print(f"测试 Sharpe 标准差: {np.std(test_sharpes):.4f}")
    print(f"正 Sharpe 窗口: {positive_count}/{len(records)}")


def _save_wf_results(records: List[Dict], output_dir: str = None):
    """保存 Walk-Forward 结果"""
    output_dir = Path(output_dir or '.')
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'walkforward_results.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Walk-Forward 结果已保存: {path}")
