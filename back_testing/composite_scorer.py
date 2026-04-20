import pandas as pd
import numpy as np
from back_testing.signal_scorer import SignalScorer

class CompositeScorer:
    """
    综合评分器：计算多策略综合评分

    评分维度：
    - MACD (35%): DIF-DEA差值，趋势强度
    - MA (20%): 均线开口，多头排列
    - RSI (15%): 超卖程度+反弹动量
    - KDJ (15%): J值超卖+反弹动量
    - 成交量 (15%): 量比，放量程度
    """

    def __init__(self, weights=None):
        self.signal_scorer = SignalScorer()

        # 默认权重配置（趋势策略权重更高）
        self.weights = weights or {
            'macd': 0.35,
            'ma': 0.20,
            'rsi': 0.15,
            'kdj': 0.15,
            'volume': 0.15
        }

    def calculate_macd_score(self, df: pd.DataFrame) -> pd.Series:
        """MACD评分：DIF-DEA差值归一化"""
        dif = df['MACD_DIF'].fillna(0)
        dea = df['MACD_DEA'].fillna(0)
        diff = dif - dea

        # 归一化到0-100：使用tanh压缩处理极端值
        score = (np.tanh(diff * 0.1) + 1) * 50  # 映射到0-100
        return score.clip(0, 100)

    def calculate_ma_score(self, df: pd.DataFrame) -> pd.Series:
        """MA评分：均线开口角度 + 多头排列"""
        ma5 = df['MA_5'].fillna(0)
        ma20 = df['MA_20'].fillna(1)
        ma60_col = df.get('MA_60', ma20)

        # 开口角度
        diff = ma5 - ma20
        angle = (diff / ma20 * 100).clip(-100, 100)

        # 多头排列：MA5 > MA20 > MA60
        bullish = ((ma5 > ma20) & (ma20 > ma60_col)).astype(int) * 30

        score = (angle + bullish).clip(0, 100)
        return score

    def calculate_rsi_score(self, df: pd.DataFrame) -> pd.Series:
        """RSI评分：使用改进的RSI信号"""
        return self.signal_scorer.calculate_rsi_improved(df)

    def calculate_kdj_score(self, df: pd.DataFrame) -> pd.Series:
        """KDJ评分：使用改进的KDJ信号"""
        return self.signal_scorer.calculate_kdj_improved(df)

    def calculate_volume_score(self, df: pd.DataFrame) -> pd.Series:
        """成交量评分：量比归一化"""
        vol_ratio = df['量比'].fillna(1)
        # 量比2.0=40分，量比5.0=100分
        score = (vol_ratio * 20).clip(0, 100)
        return score

    def calculate_composite_score(self, df: pd.DataFrame) -> float:
        """
        计算综合评分（单只股票）

        Returns:
            float: 0-100的综合评分
        """
        scores = {
            'macd': self.calculate_macd_score(df).mean(),
            'ma': self.calculate_ma_score(df).mean(),
            'rsi': self.calculate_rsi_score(df).mean(),
            'kdj': self.calculate_kdj_score(df).mean(),
            'volume': self.calculate_volume_score(df).mean(),
        }

        # 加权求和
        composite = sum(scores[k] * self.weights[k] for k in scores)
        return composite

    def calculate_composite_scores(self, df: pd.DataFrame) -> pd.Series:
        """
        计算每行的综合评分

        Returns:
            pd.Series: 每行的综合评分
        """
        macd_s = self.calculate_macd_score(df)
        ma_s = self.calculate_ma_score(df)
        rsi_s = self.calculate_rsi_score(df)
        kdj_s = self.calculate_kdj_score(df)
        volume_s = self.calculate_volume_score(df)

        composite = (
            macd_s * self.weights['macd'] +
            ma_s * self.weights['ma'] +
            rsi_s * self.weights['rsi'] +
            kdj_s * self.weights['kdj'] +
            volume_s * self.weights['volume']
        )

        return composite.clip(0, 100)
