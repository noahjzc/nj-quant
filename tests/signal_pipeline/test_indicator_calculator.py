import pandas as pd
import numpy as np
from signal_pipeline.indicator_calculator import IndicatorCalculator


def test_calculate_all_returns_expected_columns():
    df = pd.DataFrame({
        'trade_date': pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04'] * 2),
        'stock_code': ['sh600001'] * 3 + ['sz000001'] * 3,
        'open': [10.0, 10.5, 10.3, 20.0, 20.8, 21.0],
        'high': [10.8, 10.9, 10.7, 20.5, 21.2, 21.5],
        'low': [9.8, 10.2, 10.1, 19.8, 20.5, 20.8],
        'close': [10.5, 10.3, 10.6, 20.8, 21.0, 21.3],
        'volume': [10000, 12000, 9000, 20000, 22000, 18000],
    })

    result = IndicatorCalculator.calculate_all(df)

    expected_cols = [
        'ma_5', 'ma_10', 'ma_20', 'ma_30', 'ma_60', 'ma_cross',
        'macd_dif', 'macd_dea', 'macd_hist', 'macd_cross',
        'kdj_k', 'kdj_d', 'kdj_j', 'kdj_cross',
        'boll_mid', 'boll_upper', 'boll_lower',
        'rsi_1', 'rsi_2', 'rsi_3',
        'psy', 'psyma',
        'vol_ma5', 'vol_ma20', 'close_std_20', 'high_20_max',
        'atr_14', 'wr_10', 'wr_14', 'ret_5', 'ret_20',
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"
    assert len(result) == 6


def test_ma_cross_detection():
    # Need at least 20 data points for ma_20 to exist and cross to occur
    dates = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06',
                            '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10', '2024-01-11',
                            '2024-01-12', '2024-01-13', '2024-01-14', '2024-01-15', '2024-01-16',
                            '2024-01-17', '2024-01-18', '2024-01-19', '2024-01-20', '2024-01-21'])
    df = pd.DataFrame({
        'trade_date': dates,
        'stock_code': ['sh600001'] * 20,
        'open': [10.0] * 20, 'high': [11.0] * 20, 'low': [9.0] * 20,
        # Rising price to trigger golden cross: ma_5 crosses above ma_20
        'close': [10.0, 10.3, 10.5, 10.7, 10.9, 11.1, 11.3, 11.5, 11.7, 11.9,
                  12.1, 12.4, 12.8, 13.2, 13.5, 13.8, 14.0, 14.2, 14.4, 14.6],
        'volume': [10000] * 20,
    })

    result = IndicatorCalculator.calculate_all(df)
    assert 'golden_cross' in result['ma_cross'].values or 'death_cross' in result['ma_cross'].values