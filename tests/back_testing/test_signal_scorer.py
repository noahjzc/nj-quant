import pytest
from back_testing.signal_scorer import SignalScorer
import pandas as pd
import numpy as np

def test_rsi_signal_strength():
    """RSI策略：RSI值越低信号越强"""
    scorer = SignalScorer()
    df = pd.DataFrame({'rsi1': [20, 30, 50, 70, 80]})
    scores = scorer.calculate_rsi_strength(df)
    assert scores.iloc[0] > scores.iloc[1]  # 20 > 30 (更超卖更强)
    assert scores.iloc[-1] < scores.iloc[0]  # 80 < 20

def test_macd_signal_strength():
    """MACD策略：DIF与DEA差值越大信号越强"""
    scorer = SignalScorer()
    df = pd.DataFrame({
        'MACD_DIF': [1.0, 2.0, 0.5, -1.0],
        'MACD_DEA': [0.5, 1.5, 0.5, -0.5]
    })
    scores = scorer.calculate_macd_strength(df)
    # 死亡交叉比金叉弱
    assert scores.iloc[-1] < scores.iloc[0]
    assert scores.iloc[-1] < 20
    assert scores.iloc[-1] < scores.iloc[1]  # 死亡交叉比金叉更弱

def test_kdj_signal_strength():
    """KDJ策略：J值越低（超卖）信号越强"""
    scorer = SignalScorer()
    df = pd.DataFrame({'KDJ_J': [5, 20, 50, 80, 100]})
    scores = scorer.calculate_kdj_strength(df)
    assert scores.iloc[0] > scores.iloc[1]  # J=5最强
    assert scores.iloc[-1] < scores.iloc[0]  # J=100最弱