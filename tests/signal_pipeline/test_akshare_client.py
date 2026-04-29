import pytest
from signal_pipeline.data_sources.akshare_client import AKShareClient


def test_get_spot_all_columns():
    client = AKShareClient(max_retries=1)
    assert client.COLUMN_MAP['最新价'] == 'close'
    assert client.COLUMN_MAP['涨跌幅'] == 'change_pct'
    assert client.COLUMN_MAP['换手率'] == 'turnover_rate'


def test_retry_raises_after_max_attempts(mocker):
    mock_fn = mocker.patch('akshare.stock_zh_a_spot_em')
    mock_fn.side_effect = Exception('network error')

    client = AKShareClient(max_retries=2, retry_delay=0)
    with pytest.raises(RuntimeError, match='2 attempts'):
        client.get_spot_all()
