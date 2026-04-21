import pytest
import pandas as pd
from back_testing.data.index_data_provider import IndexDataProvider


def test_get_index_data():
    provider = IndexDataProvider(r'D:\workspace\code\mine\quant\data\metadata\daily_ycz\index')
    df = provider.get_index_data('sh000001', '2024-01-01', '2024-12-31')
    assert len(df) > 0
    assert 'close' in df.columns


def test_get_index_return():
    provider = IndexDataProvider(r'D:\workspace\code\mine\quant\data\metadata\daily_ycz\index')
    ret = provider.get_index_return('sh000001', '2024-01-01', '2024-12-31')
    assert isinstance(ret, float)


def test_file_not_found():
    provider = IndexDataProvider(r'D:\workspace\code\mine\quant\data\metadata\daily_ycz\index')
    with pytest.raises(FileNotFoundError):
        provider.get_index_data('sh999999', '2024-01-01', '2024-12-31')