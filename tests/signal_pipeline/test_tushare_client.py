# tests/signal_pipeline/test_tushare_client.py
import os
import pandas as pd
import pytest
from signal_pipeline.data_sources.tushare_client import TushareClient


@pytest.fixture
def client():
    token = os.environ.get('TUSHARE_TOKEN', 'test_token')
    return TushareClient(token)


def test_client_init():
    c = TushareClient('dummy_token')
    assert c.token == 'dummy_token'
    assert c.max_retries == 3


def test_retry_on_failure(mocker):
    """Simulate 2 failures then success."""
    mock_df = pd.DataFrame({'code': ['000001'], 'close': [10.0]})
    mock_pro = mocker.patch('signal_pipeline.data_sources.tushare_client.ts.pro_api')
    mock_api = mocker.MagicMock()
    mock_api.daily.side_effect = [
        Exception('timeout'),
        Exception('timeout'),
        mock_df,
    ]
    mock_pro.return_value = mock_api

    client = TushareClient('token', retry_delay=0)
    result = client._call_with_retry(lambda: mock_api.daily(trade_date='20260428'))
    assert mock_api.daily.call_count == 3
    assert not result.empty
