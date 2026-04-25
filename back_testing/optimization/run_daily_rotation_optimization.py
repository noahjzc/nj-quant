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
