"""每日全市场轮动回测系统"""
from strategy.rotation.daily_rotation_engine import DailyRotationEngine, Position, DailyResult
from strategy.rotation.config import RotationConfig, MarketRegimeConfig, MarketRegimeParams
from strategy.rotation.signal_engine.signal_filter import SignalFilter
from strategy.rotation.signal_engine.signal_ranker import SignalRanker
from strategy.rotation.market_regime import MarketRegime
from strategy.rotation.position_manager import RotationPositionManager
from strategy.rotation.trade_executor import TradeExecutor, TradeRecord
from strategy.rotation.strategy import AbstractRotationStrategy

__all__ = [
    'DailyRotationEngine', 'Position', 'DailyResult',
    'RotationConfig', 'MarketRegimeConfig', 'MarketRegimeParams',
    'SignalFilter', 'SignalRanker',
    'MarketRegime',
    'RotationPositionManager',
    'TradeExecutor', 'TradeRecord',
    'AbstractRotationStrategy',
]