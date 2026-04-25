"""daily_rotation Optuna 优化 单元测试"""
import numpy as np
from back_testing.optimization.run_daily_rotation_optimization import (
    compute_sharpe,
    max_drawdown,
)


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
