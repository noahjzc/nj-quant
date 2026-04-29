import pandas as pd
from signal_pipeline.data_merger import DataMerger


def test_merge_intraday_with_history():
    history = pd.DataFrame({
        'trade_date': pd.to_datetime(['2024-01-03', '2024-01-03']),
        'stock_code': ['sh600001', 'sz000001'],
        'open': [10.0, 20.0],
        'high': [10.5, 20.8],
        'low': [9.8, 19.5],
        'close': [10.2, 20.5],
        'volume': [10000, 20000],
        'turnover_amount': [100000, 400000],
        'amplitude': [5.0, 3.0],
        'change_pct': [2.0, 1.5],
        'turnover_rate': [1.0, 2.0],
        'volume_ratio': [1.2, 0.9],
        'circulating_mv': [1e9, 2e9],
        'total_mv': [5e9, 8e9],
        'pe_ttm': [15.0, 25.0],
        'ps_ttm': [2.0, 3.0],
        'pb': [1.5, 3.0],
    })

    intraday = pd.DataFrame({
        'stock_code': ['sh600001', 'sz000001'],
        'stock_name': ['股票A', '股票B'],
        'open': [10.3, 20.6],
        'high': [10.8, 21.0],
        'low': [10.1, 20.4],
        'close': [10.6, 20.9],
        'volume': [5000, 10000],
        'turnover_amount': [50000, 200000],
        'amplitude': [4.0, 2.0],
        'change_pct': [3.9, 1.95],
        'turnover_rate': [0.5, 1.0],
        'volume_ratio': [1.1, 1.0],
        'pe_ttm': [15.5, 25.5],
        'pb': [1.6, 3.1],
        'circulating_mv': [1.05e9, 2.1e9],
        'total_mv': [5.2e9, 8.2e9],
    })

    today_date = pd.Timestamp('2024-01-04')
    result = DataMerger.merge(history, intraday, today_date)

    assert len(result) == 4
    today_rows = result[result['trade_date'] == today_date]
    assert len(today_rows) == 2
    assert today_rows[today_rows['stock_code'] == 'sh600001']['close'].values[0] == 10.6
