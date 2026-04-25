"""每日全市场轮动回测系统"""
from back_testing.rotation.config import RotationConfig, MarketRegimeConfig

# Deferred import to avoid circular dependency during package setup
__all__ = ['RotationConfig', 'MarketRegimeConfig']
