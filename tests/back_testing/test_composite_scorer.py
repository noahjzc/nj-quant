import pytest
from back_testing.composite_scorer import CompositeScorer
import pandas as pd
import numpy as np

def test_composite_score_calculation():
    """测试综合评分计算"""
    scorer = CompositeScorer()

    # 模拟一只股票的多维度数据
    df = pd.DataFrame({
        'MACD_DIF': [1.5, 0.8, 0.2],
        'MACD_DEA': [1.0, 0.5, 0.1],
        'MA_5': [100, 102, 105],
        'MA_20': [98, 99, 100],
        'rsi1': [30, 45, 60],
        'KDJ_J': [10, 30, 70],
        '量比': [2.0, 1.5, 1.0],
    })

    score = scorer.calculate_composite_score(df)

    # 返回应该是0-100之间的分数
    assert 0 <= score <= 100
    assert isinstance(score, float)

def test_weight_distribution():
    """测试权重总和为1"""
    scorer = CompositeScorer()
    total = (scorer.weights['macd'] + scorer.weights['ma'] +
              scorer.weights['rsi'] + scorer.weights['kdj'] +
              scorer.weights['volume'])
    assert abs(total - 1.0) < 0.001

def test_macd_score_range():
    """MACD评分应该在0-100之间"""
    scorer = CompositeScorer()
    df = pd.DataFrame({
        'MACD_DIF': [0, 1, -1],
        'MACD_DEA': [0, 0.5, -0.5],
    })
    score = scorer.calculate_macd_score(df)
    assert score.min() >= 0
    assert score.max() <= 100
