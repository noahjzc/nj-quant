import pytest
from back_testing.selectors.composite_selector import CompositeSelector
import pandas as pd

def test_select_top_stocks():
    """测试选取综合评分最高的股票"""
    selector = CompositeSelector(
        data_path='D:/workspace/code/mine/quant/data/metadata/daily_ycz'
    )
    selected = selector.select_top_stocks(n=3, date='2024-01-15')
    assert len(selected) <= 3
    assert all(isinstance(code, str) for code in selected)