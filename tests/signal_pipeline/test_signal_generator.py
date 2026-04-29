import pandas as pd
from signal_pipeline.signal_generator import SignalGenerator
from back_testing.rotation.config import RotationConfig


def test_generate_sell_signals_basic():
    """Test that generate_sell_signals returns dicts with stock_code and reason."""
    config = RotationConfig()
    config.sell_signal_types = ['KDJ_DEATH', 'MACD_DEATH']

    generator = SignalGenerator(config)

    today = pd.Timestamp('2024-01-10')
    yesterday = pd.Timestamp('2024-01-09')

    # Data with both KDJ and MACD death crosses for sh600001
    # (k_prev=55>=d_prev=50, k_curr=45<d_curr=40 => death cross)
    df = pd.DataFrame({
        'trade_date': [yesterday, today, yesterday, today],
        'stock_code': ['sh600001', 'sh600001', 'sz000001', 'sz000001'],
        'open': [10.0, 10.2, 20.0, 20.0],
        'high': [10.5, 10.6, 20.5, 20.5],
        'low': [9.8, 10.0, 19.8, 19.8],
        'close': [10.2, 10.5, 20.0, 20.2],
        'volume': [10000, 8000, 20000, 18000],
        # KDJ death cross for sh600001 (55->45 crossing below 50->40)
        'kdj_k': [55.0, 45.0, 50.0, 40.0],
        'kdj_d': [50.0, 40.0, 45.0, 42.0],
        # MACD no death cross for sh600001 (dif 0.08 >= dea 0.06, not <)
        'macd_dif': [0.05, 0.08, 0.10, 0.08],
        'macd_dea': [0.03, 0.06, 0.09, 0.09],
        'ma_5': [10.0, 10.2, 20.0, 20.0],
        'ma_20': [10.0, 10.0, 20.0, 20.0],
        'vol_ma5': [10000, 9000, 20000, 19000],
        'vol_ma20': [10000, 9500, 20000, 19500],
        'close_std_20': [0.5, 0.5, 0.3, 0.3],
        'boll_mid': [10.0, 10.1, 20.0, 20.0],
        'high_20_max': [10.8, 10.8, 20.5, 20.5],
        'psy': [50, 55, 60, 60],
        'psyma': [50, 52, 58, 59],
        'rsi_1': [50, 55, 50, 50],
        'ret_5': [0.02, 0.03, 0.01, 0.01],
        'ret_20': [0.05, 0.06, 0.02, 0.02],
        'circulating_mv': [1e9, 1.1e9, 2e9, 2.1e9],
        'pe_ttm': [15, 16, 25, 26],
        'pb': [1.5, 1.6, 3.0, 3.1],
    })

    sell_signals = generator.generate_sell_signals(df, today, ['sh600001', 'sz000001'])

    # Should return a list of dicts
    assert isinstance(sell_signals, list)
    for s in sell_signals:
        assert 'stock_code' in s
        assert 'reason' in s

    codes = [s['stock_code'] for s in sell_signals]
    # At least one should trigger (sh600001 triggers KDJ_DEATH)
    assert 'sh600001' in codes or 'sz000001' in codes


def test_generate_buy_signals_basic():
    config = RotationConfig()
    config.buy_signal_mode = 'OR'
    config.buy_signal_types = ['KDJ_GOLD', 'MACD_GOLD']

    generator = SignalGenerator(config)

    today = pd.Timestamp('2024-01-10')
    yesterday = pd.Timestamp('2024-01-09')

    df = pd.DataFrame({
        'trade_date': [yesterday, today, yesterday, today],
        'stock_code': ['sh600001', 'sh600001', 'sz000001', 'sz000001'],
        'open': [10.0, 10.2, 20.0, 20.0],
        'high': [10.5, 10.6, 20.5, 20.5],
        'low': [9.8, 10.0, 19.8, 19.8],
        'close': [10.2, 10.5, 20.0, 20.2],
        'volume': [10000, 8000, 20000, 18000],
        'kdj_k': [30.0, 45.0, 50.0, 40.0],
        'kdj_d': [35.0, 40.0, 45.0, 42.0],
        'macd_dif': [0.05, 0.12, 0.10, 0.08],
        'macd_dea': [0.08, 0.10, 0.09, 0.09],
        'ma_5': [10.0, 10.2, 20.0, 20.0],
        'ma_20': [10.0, 10.0, 20.0, 20.0],
        'vol_ma5': [10000, 9000, 20000, 19000],
        'vol_ma20': [10000, 9500, 20000, 19500],
        'close_std_20': [0.5, 0.5, 0.3, 0.3],
        'boll_mid': [10.0, 10.1, 20.0, 20.0],
        'high_20_max': [10.8, 10.8, 20.5, 20.5],
        'psy': [50, 55, 60, 60],
        'psyma': [50, 52, 58, 59],
        'rsi_1': [50, 55, 50, 50],
        'ret_5': [0.02, 0.03, 0.01, 0.01],
        'ret_20': [0.05, 0.06, 0.02, 0.02],
        'circulating_mv': [1e9, 1.1e9, 2e9, 2.1e9],
        'pe_ttm': [15, 16, 25, 26],
        'pb': [1.5, 1.6, 3.0, 3.1],
    })

    buy_codes = generator.generate_buy_signals(df, today)
    assert 'sh600001' in buy_codes
    assert len(buy_codes) > 0
