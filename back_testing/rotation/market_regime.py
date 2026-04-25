"""大盘市场状态判断和动态参数调节"""
import pandas as pd
from typing import Tuple
from back_testing.data.data_provider import DataProvider
from back_testing.rotation.config import MarketRegimeConfig, MarketRegimeParams


class MarketRegime:
    """
    每日大盘状态判断和参数动态调节

    根据大盘 MA 趋势、N 日动量、ATR 波动率判断市场状态，
    返回对应的仓位参数。
    """

    def __init__(self, config: MarketRegimeConfig, data_provider: DataProvider = None):
        self.config = config
        self.data_provider = data_provider or DataProvider()

    def get_regime(self, date: pd.Timestamp, lookback: int = None) -> Tuple[str, MarketRegimeParams]:
        """
        获取当日市场状态和对应参数

        Args:
            date: 评分日期
            lookback: 回溯天数，默认使用 config.lookback_period

        Returns:
            (状态名, 参数)，如 ('strong', MarketRegimeParams(...))
        """
        lookback = lookback or self.config.lookback_period
        index_code = 'sh000300'  # 沪深300

        start_date = (date - pd.Timedelta(days=lookback * 3)).strftime('%Y-%m-%d')
        end_date = date.strftime('%Y-%m-%d')

        index_df = self.data_provider.get_index_data(index_code, start_date=start_date, end_date=end_date)
        if index_df is None or len(index_df) < lookback + 5:
            # 数据不足，默认中性
            return 'neutral', self.config.regime_params['neutral']

        index_df = index_df.sort_index()

        close = index_df['close']
        ma_fast = close.rolling(5).mean()
        ma_slow = close.rolling(20).mean()

        # 大盘趋势：ma5 / ma20 - 1
        trend = (ma_fast.iloc[-1] / ma_slow.iloc[-1] - 1) if not (ma_fast.empty or ma_slow.empty) else 0.0

        # 大盘动量：N日收益率
        if len(close) >= lookback + 1:
            momentum = (close.iloc[-1] / close.iloc[-lookback - 1] - 1)
        else:
            momentum = 0.0

        # ATR（简化：使用日内波幅均值）
        high = index_df['high']
        low = index_df['low']
        tr1 = high - low
        atr = tr1.rolling(14).mean().iloc[-1] if len(tr1) >= 14 else tr1.mean()
        # ATR 相对价格比率
        volatility = atr / close.iloc[-1] if close.iloc[-1] > 0 else 0.0

        # 判断状态
        regime = self._classify_regime(trend, momentum, volatility)

        return regime, self.config.regime_params[regime]

    def _classify_regime(self, trend: float, momentum: float, volatility: float) -> str:
        """根据指标分类市场状态"""
        strong_threshold = self.config.strong_trend_threshold
        weak_threshold = self.config.weak_trend_threshold
        high_vol_threshold = self.config.high_volatility_threshold

        is_strong = trend > strong_threshold and momentum > 0
        is_weak = trend < weak_threshold or volatility > high_vol_threshold or momentum < -0.05

        if is_strong and not is_weak:
            return 'strong'
        elif is_weak:
            return 'weak'
        else:
            return 'neutral'
