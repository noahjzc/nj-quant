import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from strategy.factors.factor_screening import FactorScreener


def test_compute_rank_ic_basic():
    """Rank IC 计算: 正向因子应得正IC，反向因子应得负IC"""
    n = 50
    stocks = [f'sh00000{i}' for i in range(n)]
    forward_ret = pd.Series(
        np.linspace(0.10, -0.10, n), index=stocks
    )
    f_pos = pd.Series(np.linspace(0.8, 0.1, n), index=stocks)
    f_neg = pd.Series(np.linspace(0.1, 0.8, n), index=stocks)
    factor_df = pd.DataFrame({'factor_pos': f_pos, 'factor_neg': f_neg})

    screener = FactorScreener.__new__(FactorScreener)
    ic = screener._compute_cross_sectional_ic(factor_df, forward_ret)

    assert ic['factor_pos'] > 0.3, f"正相关因子IC应>0.3, got {ic['factor_pos']}"
    assert ic['factor_neg'] < -0.3, f"负相关因子IC应<-0.3, got {ic['factor_neg']}"


def test_screen_factors_empty():
    """空IC DataFrame 返回空列表"""
    screener = FactorScreener.__new__(FactorScreener)
    ic_df = pd.DataFrame({'ic_mean': [], 'ic_std': [], 'icir': [], 'ic_positive_ratio': []})
    ic_df.index.name = 'factor'
    raw, orth = screener.screen_factors(ic_df, min_abs_ic=0.01, min_icir=0.1)
    assert raw == []
    assert orth == []
