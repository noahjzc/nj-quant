"""信号类型枚举和基类"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import pandas as pd


class SignalType(Enum):
    """信号类型枚举"""
    # 买入信号（金叉）
    KDJ_GOLD = 'KDJ_GOLD'
    MACD_GOLD = 'MACD_GOLD'
    MA_GOLD = 'MA_GOLD'
    VOL_GOLD = 'VOL_GOLD'
    DMI_GOLD = 'DMI_GOLD'
    BOLL_BREAK = 'BOLL_BREAK'
    HIGH_BREAK = 'HIGH_BREAK'
    # 卖出信号（死叉）
    KDJ_DEATH = 'KDJ_DEATH'
    MACD_DEATH = 'MACD_DEATH'
    MA_DEATH = 'MA_DEATH'
    VOL_DEATH = 'VOL_DEATH'
    DMI_DEATH = 'DMI_DEATH'
    BOLL_BREAK_DOWN = 'BOLL_BREAK_DOWN'
    HIGH_BREAK_DOWN = 'HIGH_BREAK_DOWN'

    @property
    def is_buy(self) -> bool:
        return self.name.endswith('_GOLD') or self.name == 'BOLL_BREAK' or self.name == 'HIGH_BREAK'

    @property
    def is_sell(self) -> bool:
        return self.name.endswith('_DEATH') or self.name in ('BOLL_BREAK_DOWN', 'HIGH_BREAK_DOWN')


@dataclass
class SignalResult:
    """信号检测结果"""
    signal_type: SignalType
    stock_code: str
    triggered: bool
    strength: float = 0.0  # 信号强度，0-1
    metadata: Optional[dict] = None


class BaseSignal:
    """信号检测基类"""

    def __init__(self, signal_type: SignalType):
        self.signal_type = signal_type

    def detect(self, df: pd.DataFrame, stock_code: str) -> SignalResult:
        """检测信号"""
        raise NotImplementedError

    def _cross_up(self, series1: pd.Series, series2: pd.Series, period: int = 1) -> bool:
        """检测是否从下方穿越上方（金叉）"""
        if len(series1) < period + 1:
            return False
        current = series1.iloc[-1]
        prev = series1.iloc[-period - 1]
        current_cross = current > series2.iloc[-1]
        prev_cross = prev <= series2.iloc[-period - 1]
        return current_cross and prev_cross

    def _cross_down(self, series1: pd.Series, series2: pd.Series, period: int = 1) -> bool:
        """检测是否从上方穿越下方（死叉）"""
        if len(series1) < period + 1:
            return False
        current = series1.iloc[-1]
        prev = series1.iloc[-period - 1]
        current_cross = current < series2.iloc[-1]
        prev_cross = prev >= series2.iloc[-period - 1]
        return current_cross and prev_cross
