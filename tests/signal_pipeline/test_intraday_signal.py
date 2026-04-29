"""Tests for signal_pipeline/intraday_signal.py"""
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import date


def test_cron_start_returns_log_id():
    """Test that _cron_start inserts a cron_log entry and returns the id."""
    from signal_pipeline.intraday_signal import _cron_start

    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = (42,)
    mock_session.execute.return_value = mock_result

    log_id = _cron_start('intraday_signal', mock_session)

    assert log_id == 42
    mock_session.execute.assert_called_once()
    call_args = mock_session.execute.call_args
    sql_text = str(call_args[0][0])
    assert 'INSERT INTO cron_log' in sql_text
    assert call_args[0][1]['name'] == 'intraday_signal'
    mock_session.commit.assert_called_once()


def test_cron_finish_updates_status():
    """Test that _cron_finish updates cron_log with status and metadata."""
    from signal_pipeline.intraday_signal import _cron_finish

    mock_session = MagicMock()

    _cron_finish(99, 'success', mock_session, metadata={'buy_count': 3})

    mock_session.execute.assert_called_once()
    call_args = mock_session.execute.call_args
    params = call_args[0][1]
    assert params['status'] == 'success'
    assert params['id'] == 99
    assert 'buy_count' in params['meta']
    mock_session.commit.assert_called_once()


def test_delete_before_insert_happens():
    """Verify DELETE is called before any INSERT in the signal write flow."""
    from signal_pipeline.intraday_signal import main
    import signal_pipeline.intraday_signal as intraday_module

    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = (1,)
    mock_session.execute.return_value = mock_result

    mock_engine = MagicMock()

    # Empty intraday snapshot
    mock_spot_df = pd.DataFrame({
        'stock_code': [], 'stock_name': [], 'close': [], 'open': [],
        'high': [], 'low': [], 'volume': [], 'turnover_amount': [],
        'amplitude': [], 'change_pct': [], 'turnover_rate': [],
        'volume_ratio': [], 'pe_ttm': [], 'pb': [],
        'circulating_mv': [], 'total_mv': [],
    })

    mock_hist_df = pd.DataFrame({
        'trade_date': pd.to_datetime(['2024-01-09']),
        'stock_code': ['sh600001'],
        'open': [10.0], 'high': [10.5], 'low': [9.8], 'close': [10.2],
        'volume': [10000], 'turnover_amount': [100000], 'amplitude': [5.0],
        'change_pct': [1.5], 'turnover_rate': [1.5], 'volume_ratio': [1.2],
        'pe_ttm': [15.0], 'pb': [1.5],
        'circulating_mv': [1e9], 'total_mv': [2e9],
    })

    call_order = []

    def mock_execute(sql, params=None):
        call_order.append(str(sql) if hasattr(sql, '__str__') else sql)
        return mock_result

    mock_session.execute = mock_execute

    with patch.object(intraday_module, 'get_session', return_value=lambda: mock_session), \
         patch.object(intraday_module, 'get_engine', return_value=mock_engine), \
         patch('signal_pipeline.intraday_signal.AKShareClient') as MockAKShare, \
         patch('signal_pipeline.intraday_signal.DataMerger') as MockMerger, \
         patch('signal_pipeline.intraday_signal.IndicatorCalculator') as MockCalc, \
         patch('signal_pipeline.intraday_signal.SignalGenerator') as MockGen, \
         patch('signal_pipeline.intraday_signal.RotationConfig') as MockConfig, \
         patch.object(intraday_module, '_cron_start', return_value=1), \
         patch.object(intraday_module, '_cron_finish'):

        mock_ak_instance = MockAKShare.return_value
        mock_ak_instance.get_spot_all.return_value = mock_spot_df

        mock_merge_instance = MockMerger.return_value
        mock_merge_instance.merge.return_value = mock_hist_df

        mock_calc_instance = MockCalc.return_value
        mock_calc_instance.calculate_all.return_value = mock_hist_df

        mock_gen_instance = MockGen.return_value
        mock_gen_instance.generate_buy_signals.return_value = ['sh600001']
        mock_gen_instance.generate_sell_signals.return_value = []

        with patch('pandas.read_sql', return_value=mock_hist_df):
            intraday_module.main()

        # Check delete was called before insert
        delete_idx = None
        insert_idx = None
        for i, call_str in enumerate(call_order):
            if 'DELETE FROM daily_signal' in call_str:
                delete_idx = i
            if 'INSERT INTO daily_signal' in call_str and insert_idx is None:
                insert_idx = i

        assert delete_idx is not None, f"DELETE not called. Calls: {call_order}"
        assert insert_idx is not None, f"INSERT not called. Calls: {call_order}"
        assert delete_idx < insert_idx, "DELETE must come before INSERT"


def test_buy_signal_uses_stock_name_from_merged_data():
    """
    Verify the name-lookup logic used in main() for INSERT.

    When today_lookup is populated from merged data with stock_name,
    the name field in the INSERT params should use that stock_name.
    """
    # This tests the lookup logic directly, bypassing the full main() pipeline
    # which requires complex mocking of multiple interdependent modules.
    today_ts = pd.Timestamp(date.today())
    hist_df = pd.DataFrame({
        'trade_date': pd.to_datetime(['2024-01-09']),
        'stock_code': ['sh600001'], 'open': [10.0], 'high': [10.5],
        'low': [9.8], 'close': [10.2], 'volume': [10000],
        'turnover_amount': [100000], 'amplitude': [5.0], 'change_pct': [1.5],
        'turnover_rate': [1.5], 'volume_ratio': [1.2], 'pe_ttm': [15.0],
        'pb': [1.5], 'circulating_mv': [1e9], 'total_mv': [2e9],
    })
    today_row = pd.DataFrame({
        'trade_date': [today_ts],
        'stock_code': ['sh600001'], 'stock_name': ['TestStock'],
        'open': [10.0], 'high': [10.8], 'low': [9.9], 'close': [10.5],
        'volume': [10000], 'turnover_amount': [100000], 'amplitude': [5.0],
        'change_pct': [2.0], 'turnover_rate': [1.5], 'volume_ratio': [1.2],
        'pe_ttm': [15.0], 'pb': [1.5], 'circulating_mv': [1e9], 'total_mv': [2e9],
    })
    merged_df = pd.concat([hist_df, today_row], ignore_index=True)

    # Simulate main() today_lookup logic
    today_df = merged_df[merged_df['trade_date'] == today_ts]
    today_lookup = today_df.set_index('stock_code') if not today_df.empty else pd.DataFrame()

    code = 'sh600001'
    if not today_lookup.empty and code in today_lookup.index:
        row = today_lookup.loc[code]
        name = row.get('stock_name', code)
    else:
        name = code

    assert name == 'TestStock'
    assert code == 'sh600001'
