"""第一层信号过滤器 — 技术指标金叉/死叉过滤"""
import pandas as pd
from typing import List, Dict
from back_testing.rotation.signal_engine.base_signal import SignalType, SignalResult, BaseSignal


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
    """VOL MA 金叉检测"""

    def __init__(self):
        super().__init__(SignalType.VOL_GOLD)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'volume_ratio' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        vol = df['volume_ratio']
        triggered = self._cross_up(vol, vol.shift(5).rolling(5).mean())
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class BollBreakSignal(BaseSignal):
    """布林带上轨突破"""

    def __init__(self):
        super().__init__(SignalType.BOLL_BREAK)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'close' not in df.columns or 'boll_upper' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        triggered = df['close'].iloc[-1] > df['boll_upper'].iloc[-1] if not df.empty else False
        return SignalResult(
            signal_type=self.signal_type,
            stock_code=stock_code,
            triggered=triggered,
            strength=1.0 if triggered else 0.0
        )


class BollBreakDownSignal(BaseSignal):
    """布林带下轨突破（卖出）"""

    def __init__(self):
        super().__init__(SignalType.BOLL_BREAK_DOWN)

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        if 'close' not in df.columns or 'boll_lower' not in df.columns:
            return SignalResult(self.signal_type, stock_code, False, 0.0)
        triggered = df['close'].iloc[-1] < df['boll_lower'].iloc[-1] if not df.empty else False
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
        SignalType.BOLL_BREAK: BollBreakSignal,
        SignalType.BOLL_BREAK_DOWN: BollBreakDownSignal,
    }

    def __init__(self, signal_types: List[str]):
        self.detectors = []
        for name in signal_types:
            try:
                sig_type = SignalType[name]
                detector_cls = self._SIGNAL_MAP.get(sig_type)
                if detector_cls:
                    self.detectors.append(detector_cls())
            except KeyError:
                pass

    def filter_buy(self, df: pd.DataFrame, stock_code: str) -> bool:
        """检查是否有任何买入信号触发"""
        for detector in self.detectors:
            if detector.signal_type.is_buy:
                result = detector.detect(df, stock_code)
                if result.triggered:
                    return True
        return False

    def filter_sell(self, df: pd.DataFrame, stock_code: str) -> bool:
        """检查是否有任何卖出信号触发"""
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
