import pytest
from back_testing.strategy_evaluator import StrategyEvaluator
import pandas as pd
import numpy as np

def test_evaluate_single_strategy():
    """测试单个策略的4周表现评估"""
    evaluator = StrategyEvaluator(
        stock_codes=['sh600519', 'sz000001'],
        data_path='D:/workspace/code/mine/quant/data/metadata/daily_ycz'
    )
    scores = evaluator.evaluate_strategy('RSIReversalStrategy', weeks=4)
    assert isinstance(scores, dict)
    assert 'avg_return' in scores
    assert 'stock_returns' in scores