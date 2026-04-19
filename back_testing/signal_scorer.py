import pandas as pd
import numpy as np

class SignalScorer:
    """信号评分器：计算各策略的信号强度"""

    def calculate_rsi_strength(self, df: pd.DataFrame) -> pd.Series:
        """RSI信号强度：RSI值越低（越超卖），强度越高"""
        rsi = df['rsi1'].fillna(50)  # NaN用中性值填充
        strength = 100 - rsi
        return strength.clip(0, 100)

    def calculate_macd_strength(self, df: pd.DataFrame) -> pd.Series:
        """MACD信号强度：DIF与DEA差值越大，强度越高，DIF绝对值越大趋势越强"""
        dif = df['MACD_DIF'].fillna(0)
        dea = df['MACD_DEA'].fillna(0)
        diff = dif - dea
        # 避免零除
        mean_diff = abs(diff).mean()
        if mean_diff == 0 or pd.isna(mean_diff):
            mean_diff = 1.0
        strength = (diff / mean_diff * 50 + 50).clip(0, 100)
        return strength

    def calculate_ma_strength(self, df: pd.DataFrame) -> pd.Series:
        """MA信号强度：MA5与MA20开口角度越大，趋势越强"""
        ma5 = df['MA_5'].fillna(0)
        ma20 = df['MA_20'].fillna(1)  # 避免零除，用1替代
        diff = ma5 - ma20
        strength = (diff / ma20 * 100).clip(-100, 100)
        return strength

    def calculate_kdj_strength(self, df: pd.DataFrame) -> pd.Series:
        """KDJ信号强度：J值越低（超卖），强度越高"""
        j = df['KDJ_J'].fillna(50)  # NaN用中性值填充
        strength = 100 - j
        return strength.clip(0, 100)

    def calculate_bollinger_strength(self, df: pd.DataFrame) -> pd.Series:
        """布林带信号强度：价格偏离中轨越远，强度越高"""
        price = df['后复权价'].fillna(0)
        middle = df['布林线中轨'].fillna(1)
        # 避免零除：使用 replace 将 0 也替换为 1
        middle = middle.replace(0, 1)
        strength = abs(price - middle) / middle * 100
        return strength

    def calculate_volume_strength(self, df: pd.DataFrame) -> pd.Series:
        """成交量信号强度：量比越大，强度越高"""
        vol_ratio = df['量比'].fillna(1)  # NaN用正常值填充
        strength = (vol_ratio * 20).clip(0, 100)
        return strength

    def calculate_combined_strength(self, df: pd.DataFrame) -> pd.Series:
        """组合策略信号强度：MA + RSI 组合"""
        ma_strength = self.calculate_ma_strength(df)
        rsi_strength = self.calculate_rsi_strength(df)
        rsi_confirm = (df['rsi1'] < 40).astype(int) * 50
        strength = (ma_strength * 0.6 + rsi_confirm * 0.4)
        return strength.clip(0, 100)

    def get_signal_strength(self, strategy_name: str, df: pd.DataFrame) -> pd.Series:
        """根据策略名称获取信号强度"""
        strength_map = {
            'MAStrategy': self.calculate_ma_strength,
            'MACDStrategy': self.calculate_macd_strength,
            'RSIReversalStrategy': self.calculate_rsi_strength,
            'CombinedStrategy': self.calculate_combined_strength,
            'BollingerStrategy': self.calculate_bollinger_strength,
            'BollingerStrictStrategy': self.calculate_bollinger_strength,
            'KDJOversoldStrategy': self.calculate_kdj_strength,
            'KDJGoldenCrossStrategy': self.calculate_kdj_strength,
            'MultiPeriodRSIStrategy': self.calculate_rsi_strength,
            'RSIReversalMultiStrategy': self.calculate_rsi_strength,
            'TrendConfirmationStrategy': self.calculate_macd_strength,
            'TrendPullbackStrategy': self.calculate_ma_strength,
            'VolumeAnomalyStrategy': self.calculate_volume_strength,
            'VolumeMAConfirmStrategy': self.calculate_volume_strength,
        }

        if strategy_name in strength_map:
            return strength_map[strategy_name](df)
        else:
            return pd.Series(50, index=df.index)