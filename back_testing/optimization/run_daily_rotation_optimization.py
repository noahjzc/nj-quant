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
