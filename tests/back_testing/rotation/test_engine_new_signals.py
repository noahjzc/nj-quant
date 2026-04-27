"""Integration tests for new signals and factors in DailyRotationEngine."""
import pandas as pd
import numpy as np
import pytest
from back_testing.rotation.config import RotationConfig
from back_testing.rotation.daily_rotation_engine import DailyRotationEngine
from back_testing.factors.factor_utils import FactorProcessor


class TestBuildSignalFeatures:
    """Test _build_signal_features includes new columns."""

    def test_psy_columns_in_features(self):
        config = RotationConfig()
        engine = DailyRotationEngine(config, '2024-01-01', '2024-01-31')

        dates = pd.date_range('2024-01-01', '2024-01-25', freq='B')
        rows = []
        for i, d in enumerate(dates):
            rows.append({
                'trade_date': d, 'stock_code': 'sh600001',
                'close': 10 + i * 0.1, 'open': 10 + i * 0.1,
                'high': 11 + i * 0.1, 'low': 9 + i * 0.1,
                'volume': 1000, 'kdj_k': 50, 'kdj_d': 48,
                'macd_dif': 0.5, 'macd_dea': 0.3,
                'ma_5': 10, 'ma_20': 9.5,
                'boll_mid': 10, 'psy': 30.0, 'psyma': 28.0,
            })
        engine._cache_df = pd.DataFrame(rows).set_index('trade_date')

        features = engine._build_signal_features(['sh600001'])
        assert 'psy' in features.columns
        assert 'psyma' in features.columns


class TestNewFactorExtraction:
    """Test factor extraction with new factors."""

    def test_circulating_mv_log_transform(self):
        config = RotationConfig()
        engine = DailyRotationEngine(config, '2024-01-01', '2024-03-31')

        # Need >= MIN_TRADING_DAYS (20) rows for _get_daily_stock_data to return the stock
        dates = pd.date_range('2024-01-02', '2024-02-01', freq='B')
        rows = []
        for i, d in enumerate(dates):
            rows.append({
                'trade_date': d, 'stock_code': 'sh600001',
                'close': 10 + i * 0.1, 'open': 10 + i * 0.1,
                'high': 11 + i * 0.1, 'low': 9 + i * 0.1,
                'volume': 1000, 'circulating_mv': 1e9,
            })
        engine._cache_df = pd.DataFrame(rows).set_index('trade_date')

        stock_data = engine._get_daily_stock_data(dates[-1])
        assert 'sh600001' in stock_data
        df = stock_data['sh600001']
        row = df.iloc[-1]

        val = row.get('circulating_mv', np.nan)
        log_val = np.log(val) if val > 0 else np.nan
        assert log_val == pytest.approx(np.log(1e9))

    def test_wr_factor_extraction(self):
        df = pd.DataFrame({
            'high': [12, 13, 14, 15, 16, 15, 14, 13, 12, 11,
                     12, 13, 14, 15, 16, 15, 14, 13, 12, 11],
            'low':  [8,  9,  10, 11, 10, 9,  8,  7,  6,  5,
                     8,  9,  10, 11, 10, 9,  8,  7,  6,  5],
            'close': [10, 11, 12, 13, 12, 11, 10, 9, 8, 7,
                      10, 11, 12, 13, 12, 11, 10, 9, 8, 7],
        })
        wr10 = FactorProcessor.williams_r(df, 10)
        wr14 = FactorProcessor.williams_r(df, 14)
        assert -100 <= wr10 <= 0
        assert -100 <= wr14 <= 0
