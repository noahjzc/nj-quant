"""daily_rotation Optuna 优化 单元测试"""
import numpy as np
import optuna
from back_testing.optimization.run_daily_rotation_optimization import (
    compute_sharpe,
    max_drawdown,
    sample_config,
)
from back_testing.rotation.config import RotationConfig


def test_max_drawdown_simple():
    equity = [1.0, 1.1, 0.9, 1.0, 0.8, 1.05]
    dd = max_drawdown(equity)
    # peak=1.1, trough=0.8 → dd=0.3/1.1≈0.2727
    assert abs(dd - 0.2727) < 0.001


def test_max_drawdown_no_drawdown():
    equity = [1.0, 1.1, 1.2, 1.3]
    assert max_drawdown(equity) == 0.0


def test_max_drawdown_empty():
    assert max_drawdown([1.0]) == 0.0


def test_compute_sharpe_positive():
    # 构造收益率序列：每天固定 +0.001，有微小波动
    equity = [1.0 + i * 0.001 if i % 2 == 0 else 1.0 + i * 0.001 + 0.0005
              for i in range(253)]
    sharpe = compute_sharpe(equity)
    assert sharpe > 1.0


def test_compute_sharpe_constant():
    equity = [1.0] * 253
    assert compute_sharpe(equity) == 0.0


def test_sample_config_basic():
    study = optuna.create_study()
    trial = study.ask()

    config = sample_config(trial)

    # 信号至少有一个开启
    assert any(s in config.buy_signal_types for s in [
        'KDJ_GOLD', 'MACD_GOLD', 'MA_GOLD', 'VOL_GOLD', 'BOLL_BREAK', 'HIGH_BREAK'
    ]), "at least one buy signal must be on"

    # 因子权重和为 1（允许浮点误差）
    w = sum(config.rank_factor_weights.values())
    assert abs(w - 1.0) < 0.001, f"weights sum to {w}"

    # 连续参数在范围内
    assert 0.30 <= config.max_total_pct <= 1.00
    assert 0.05 <= config.max_position_pct <= 0.30
    assert 3 <= config.max_positions <= 10
    assert 7 <= config.atr_period <= 21
    assert 1.0 <= config.stop_loss_mult <= 3.5
    assert 2.0 <= config.take_profit_mult <= 5.0
    assert 0.05 <= config.trailing_pct <= 0.20
    assert 0.02 <= config.trailing_start <= 0.10
    assert 60.0 <= config.overheat_rsi_threshold <= 90.0
    assert 0.05 <= config.overheat_ret5_threshold <= 0.30

    assert config.buy_signal_mode in ('OR', 'AND')

    # 固定参数不变
    assert config.initial_capital == 1_000_000.0
    assert config.benchmark_index == 'sh000300'
    assert config.exclude_st is True


def test_sample_config_overrides_base():
    base = RotationConfig(
        max_total_pct=0.50,
        initial_capital=500_000,
    )
    study = optuna.create_study()
    trial = study.ask()
    config = sample_config(trial, base_config=base)
    assert config.initial_capital == 500_000
