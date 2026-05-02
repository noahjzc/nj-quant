import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest
from strategy.factors.alpha158 import Alpha158Calculator


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    close = 10 + np.cumsum(np.random.normal(0, 0.3, n))
    close = np.maximum(close, 0.1)
    return pd.DataFrame({
        'open': close * (1 + np.random.normal(0, 0.01, n)),
        'high': close * (1 + np.abs(np.random.normal(0, 0.02, n))),
        'low': close * (1 - np.abs(np.random.normal(0, 0.02, n))),
        'close': close,
        'volume': np.random.lognormal(15, 0.5, n),
    })


class TestAlpha158Calculator:
    def test_compute_returns_dataframe(self, sample_df):
        calc = Alpha158Calculator()
        result = calc.compute(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    def test_factor_count(self, sample_df):
        calc = Alpha158Calculator()
        result = calc.compute(sample_df)
        # 9 kbar + 3 price + 145 rolling = 157
        assert len(result.columns) == 157

    def test_no_inf_values(self, sample_df):
        calc = Alpha158Calculator()
        result = calc.compute(sample_df)
        assert not result.replace(np.nan, 0).isin([np.inf, -np.inf]).any().any()

    def test_kbar_factors_present(self, sample_df):
        calc = Alpha158Calculator()
        result = calc.compute(sample_df)
        expected = ['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2',
                    'KLOW', 'KLOW2', 'KSFT', 'KSFT2']
        for col in expected:
            assert col in result.columns, f'{col} missing'

    def test_price_factors_present(self, sample_df):
        calc = Alpha158Calculator()
        result = calc.compute(sample_df)
        for col in ['OPEN0', 'HIGH0', 'LOW0']:
            assert col in result.columns, f'{col} missing'

    def test_rolling_factors_all_windows(self, sample_df):
        calc = Alpha158Calculator()
        result = calc.compute(sample_df)
        for d in [5, 10, 20, 30, 60]:
            assert f'MA{d}' in result.columns
            assert f'STD{d}' in result.columns
            assert f'ROC{d}' in result.columns

    def test_kbar_near_zero_for_flat_day(self):
        """平开平收平盘日，KBar 应接近 0"""
        df = pd.DataFrame({
            'open': [10.0, 10.0],
            'high': [10.1, 10.0],
            'low': [9.9, 10.0],
            'close': [10.0, 10.0],
            'volume': [1e6, 1e6],
        })
        calc = Alpha158Calculator()
        result = calc.compute(df)
        # KMID = (close-open)/open = 0
        assert abs(result['KMID'].iloc[-1]) < 0.01

    def test_custom_windows(self, sample_df):
        calc = Alpha158Calculator(windows=[10, 20])
        result = calc.compute(sample_df)
        # 9 + 3 + 2 windows * 29 = 70
        assert len(result.columns) == 70
        assert 'MA10' in result.columns
        assert 'MA20' in result.columns
        assert 'MA5' not in result.columns

    def test_min_data(self):
        """单行数据不应崩溃"""
        df = pd.DataFrame({
            'open': [10.0], 'high': [10.1], 'low': [9.9],
            'close': [10.0], 'volume': [1e6],
        })
        calc = Alpha158Calculator()
        result = calc.compute(df)
        assert len(result) == 1
        assert len(result.columns) == 157
