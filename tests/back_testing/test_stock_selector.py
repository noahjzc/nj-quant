import pytest
from back_testing.stock_selector import StockSelector
import pandas as pd

def test_select_top_stocks():
    """测试选取信号最强的股票"""
    selector = StockSelector(
        data_path='D:/workspace/code/mine/quant/data/metadata/daily_ycz'
    )
    # 用贵州茅台测试
    selected = selector.select_top_stocks(
        strategy_name='RSIReversalStrategy',
        n=3,
        date='2024-01-15'
    )
    assert len(selected) <= 3
    assert all(isinstance(code, str) for code in selected)