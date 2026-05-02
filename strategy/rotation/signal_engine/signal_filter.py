"""第一层信号过滤器 — 技术指标金叉/死叉过滤"""
import logging
import pandas as pd
from typing import List, Dict
from strategy.rotation.signal_engine.base_signal import SignalType, SignalResult, BaseSignal

logger = logging.getLogger(__name__)


class KDJGoldSignal(BaseSignal):
    """KDJ 金叉检测"""

    def __init__(self):
        super().__init__(SignalType.KDJ_GOLD)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        k = df['kdj_k']
        d = df['kdj_d']
        triggered = self._cross_up(k, d)
        strength = 1.0 if triggered else 0.0
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=strength,
            metadata={'kdj_k': k.iloc[-1] if not k.empty else None, 'kdj_d': d.iloc[-1] if not d.empty else None}
        )


class KDJGoldLowSignal(BaseSignal):
    """KDJ 低位金叉检测 — 金叉且 K 值低于阈值"""

    def __init__(self, k_threshold: float = 30.0):
        super().__init__(SignalType.KDJ_GOLD_LOW)
        self.k_threshold = k_threshold

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        k = df['kdj_k']; d = df['kdj_d']
        triggered = bool(self._cross_up(k, d) and k.iloc[-1] < self.k_threshold)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0,
            metadata={'kdj_k': k.iloc[-1] if not k.empty else None,
                      'threshold': self.k_threshold}
        )


class KDJDeathSignal(BaseSignal):
    """KDJ 死叉检测"""

    def __init__(self):
        super().__init__(SignalType.KDJ_DEATH)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        k = df['kdj_k']
        d = df['kdj_d']
        triggered = self._cross_down(k, d)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class MACDGoldSignal(BaseSignal):
    """MACD 金叉检测（DIF 上穿 DEA）"""

    def __init__(self):
        super().__init__(SignalType.MACD_GOLD)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        dif = df['macd_dif']
        dea = df['macd_dea']
        triggered = self._cross_up(dif, dea)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class MACDDeathSignal(BaseSignal):
    """MACD 死叉检测（DIF 下穿 DEA）"""

    def __init__(self):
        super().__init__(SignalType.MACD_DEATH)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        dif = df['macd_dif']
        dea = df['macd_dea']
        triggered = self._cross_down(dif, dea)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class MAGoldSignal(BaseSignal):
    """MA 金叉检测（MA5 上穿 MA20）"""

    def __init__(self, fast: str = 'ma_5', slow: str = 'ma_20'):
        super().__init__(SignalType.MA_GOLD)
        self.fast = fast
        self.slow = slow

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if self.fast not in df.columns or self.slow not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        fast_ma = df[self.fast]
        slow_ma = df[self.slow]
        triggered = self._cross_up(fast_ma, slow_ma)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class MADeathSignal(BaseSignal):
    """MA 死叉检测（MA5 下穿 MA20）"""

    def __init__(self, fast: str = 'ma_5', slow: str = 'ma_20'):
        super().__init__(SignalType.MA_DEATH)
        self.fast = fast
        self.slow = slow

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if self.fast not in df.columns or self.slow not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        fast_ma = df[self.fast]
        slow_ma = df[self.slow]
        triggered = self._cross_down(fast_ma, slow_ma)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class VOLGoldSignal(BaseSignal):
    """VOL MA 金叉检测（VOL_MA5 上穿 VOL_MA20）"""

    def __init__(self):
        super().__init__(SignalType.VOL_GOLD)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'volume' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        vol = df['volume']
        vol_ma5 = vol.rolling(5).mean()
        vol_ma20 = vol.rolling(20).mean()
        triggered = self._cross_up(vol_ma5, vol_ma20)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class VOLDeathSignal(BaseSignal):
    """VOL MA 死叉检测（VOL_MA5 下穿 VOL_MA20）"""

    def __init__(self):
        super().__init__(SignalType.VOL_DEATH)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'vol_ma5' not in df.columns or 'vol_ma20' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        triggered = self._cross_down(df['vol_ma5'], df['vol_ma20'])
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class DMIGoldSignal(BaseSignal):
    """DMI 金叉检测（+DI 上穿 -DI）"""

    def __init__(self):
        super().__init__(SignalType.DMI_GOLD)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'dmi_plus_di' not in df.columns or 'dmi_minus_di' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        plus_di = df['dmi_plus_di']
        minus_di = df['dmi_minus_di']
        triggered = self._cross_up(plus_di, minus_di)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class DMIDeathSignal(BaseSignal):
    """DMI 死叉检测（+DI 下穿 -DI）"""

    def __init__(self):
        super().__init__(SignalType.DMI_DEATH)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'dmi_plus_di' not in df.columns or 'dmi_minus_di' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        plus_di = df['dmi_plus_di']
        minus_di = df['dmi_minus_di']
        triggered = self._cross_down(plus_di, minus_di)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class PSYBuySignal(BaseSignal):
    """PSY 心理线超卖买入信号 — PSY < 25 且 PSY > PSYMA（超卖 + 趋势确认）"""

    def __init__(self):
        super().__init__(SignalType.PSY_BUY)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'psy' not in df.columns or 'psyma' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        psy_now = df['psy'].iloc[-1]
        psyma_now = df['psyma'].iloc[-1]
        triggered = bool(psy_now < 25 and psy_now > psyma_now)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0,
            metadata={'psy': psy_now, 'psyma': psyma_now}
        )


class PSYSellSignal(BaseSignal):
    """PSY 心理线超买卖出信号 — PSY > 75 且 PSY < PSYMA（超买 + 趋势确认）"""

    def __init__(self):
        super().__init__(SignalType.PSY_SELL)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'psy' not in df.columns or 'psyma' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        psy_now = df['psy'].iloc[-1]
        psyma_now = df['psyma'].iloc[-1]
        triggered = bool(psy_now > 75 and psy_now < psyma_now)
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0,
            metadata={'psy': psy_now, 'psyma': psyma_now}
        )


class BollBreakSignal(BaseSignal):
    """布林带上轨突破（从 boll_mid 计算，标准差回溯 20 日）"""

    BOLL_PERIOD = 20

    def __init__(self):
        super().__init__(SignalType.BOLL_BREAK)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'close' not in df.columns or 'boll_mid' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        if len(df) < self.BOLL_PERIOD:
            return SignalResult(self.signal_type, stock_code, False, 0.0)

        close = df['close']
        boll_mid = df['boll_mid']
        # 计算布林带：用 close 的历史标准差（回溯 BOLL_PERIOD）
        rolling_std = close.rolling(window=self.BOLL_PERIOD).std()
        boll_upper = boll_mid + 2 * rolling_std
        boll_lower = boll_mid - 2 * rolling_std

        current_close = close.iloc[-1]
        triggered = current_close > boll_upper.iloc[-1]
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class BollBreakDownSignal(BaseSignal):
    """布林带下轨突破（从 boll_mid 计算）"""

    BOLL_PERIOD = 20

    def __init__(self):
        super().__init__(SignalType.BOLL_BREAK_DOWN)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'close' not in df.columns or 'boll_mid' not in df.columns or 'close_std_20' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        if len(df) < 2:
            return SignalResult(self.signal_type, stock_code, False, 0.0)

        current = df.iloc[-1]
        boll_lower = current['boll_mid'] - 2 * current['close_std_20']
        triggered = current['close'] < boll_lower
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=bool(triggered),
            strength=1.0 if triggered else 0.0
        )


class HighBreakSignal(BaseSignal):
    """N 日高点突破（回溯 20 日）"""

    LOOKBACK = 20

    def __init__(self, lookback: int = 20):
        super().__init__(SignalType.HIGH_BREAK)
        self.lookback = lookback

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'close' not in df.columns or 'high' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        if len(df) < self.lookback + 1:
            return SignalResult(self.signal_type, stock_code, False, 0.0)

        high_series = df['high']
        close_series = df['close']
        # 前 N 日最高价（不含今日）
        past_high = high_series.iloc[-self.lookback-1:-1].max()
        current_close = close_series.iloc[-1]
        triggered = current_close >= past_high
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class HighBreakDownSignal(BaseSignal):
    """N 日低点跌破（回溯 20 日）"""

    LOOKBACK = 20

    def __init__(self, lookback: int = 20):
        super().__init__(SignalType.HIGH_BREAK_DOWN)
        self.lookback = lookback

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'close' not in df.columns or 'low' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        if len(df) < self.lookback + 1:
            return SignalResult(self.signal_type, stock_code, False, 0.0)

        low_series = df['low']
        close_series = df['close']
        # 前 N 日最低价（不含今日）
        past_low = low_series.iloc[-self.lookback-1:-1].min()
        current_close = close_series.iloc[-1]
        triggered = current_close <= past_low
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class SignalFilter:
    """第一层信号过滤器 — 将信号类型名映射到对应检测器"""

    _SIGNAL_MAP = {
        SignalType.KDJ_GOLD: KDJGoldSignal,
        SignalType.KDJ_DEATH: KDJDeathSignal,
        SignalType.MACD_GOLD: MACDGoldSignal,
        SignalType.MACD_DEATH: MACDDeathSignal,
        SignalType.MA_GOLD: MAGoldSignal,
        SignalType.MA_DEATH: MADeathSignal,
        SignalType.VOL_GOLD: VOLGoldSignal,
        SignalType.VOL_DEATH: VOLDeathSignal,
        SignalType.DMI_GOLD: DMIGoldSignal,
        SignalType.DMI_DEATH: DMIDeathSignal,
        SignalType.BOLL_BREAK: BollBreakSignal,
        SignalType.BOLL_BREAK_DOWN: BollBreakDownSignal,
        SignalType.HIGH_BREAK: HighBreakSignal,
        SignalType.HIGH_BREAK_DOWN: HighBreakDownSignal,
        SignalType.KDJ_GOLD_LOW: KDJGoldLowSignal,
        SignalType.PSY_BUY: PSYBuySignal,
        SignalType.PSY_SELL: PSYSellSignal,
    }

    def __init__(self, signal_types: List[str], mode: str = 'OR', kdj_low_threshold: float = 30.0):
        """
        Args:
            signal_types: 信号类型列表
            mode: 'OR' — 任意信号触发即通过 | 'AND' — 所有信号同时触发才通过
            kdj_low_threshold: KDJ_GOLD_LOW 信号的 K 值阈值
        """
        self.mode = mode
        self.detectors = []
        unknown_signals = []
        for name in signal_types:
            try:
                sig_type = SignalType[name]
                detector_cls = self._SIGNAL_MAP.get(sig_type)
                if detector_cls:
                    if detector_cls is KDJGoldLowSignal:
                        self.detectors.append(detector_cls(k_threshold=kdj_low_threshold))
                    else:
                        self.detectors.append(detector_cls())
                else:
                    unknown_signals.append(name)
            except KeyError:
                unknown_signals.append(name)

        if unknown_signals:
            logger.warning(f"SignalFilter: unknown signal types will be ignored: {unknown_signals}")

    def filter_buy(self, df: pd.DataFrame, stock_code: str) -> bool:
        """检查是否有买入信号触发"""
        buy_detectors = [d for d in self.detectors if d.signal_type.is_buy]
        if not buy_detectors:
            return False
        if self.mode == 'AND':
            return all(d.detect(df, stock_code).triggered for d in buy_detectors)
        return any(d.detect(df, stock_code).triggered for d in buy_detectors)

    def filter_sell(self, df: pd.DataFrame, stock_code: str) -> bool:
        """检查是否有卖出信号触发"""
        for detector in self.detectors:
            if detector.signal_type.is_sell:
                result = detector.detect(df, stock_code)
                if result.triggered:
                    return True
        return False

    def scan_buy_candidates(self, stock_data_dict: Dict[str, pd.DataFrame]) -> List[str]:
        """扫描全市场，返回有买入信号的股票列表"""
        candidates = []
        for stock_code, df in stock_data_dict.items():
            if self.filter_buy(df, stock_code):
                candidates.append(stock_code)
        return candidates