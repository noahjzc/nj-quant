"""策略配置类"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MarketRegimeParams:
    """单一市场状态的参数"""
    max_total_pct: float = 0.90
    max_position_pct: float = 0.20
    max_positions: int = 5


@dataclass
class MarketRegimeConfig:
    """市场状态动态调节配置"""
    strong_trend_threshold: float = 0.05   # 大盘MA多头阈值（5%）
    weak_trend_threshold: float = -0.03     # 大盘MA空头阈值（-3%）
    high_volatility_threshold: float = 0.03  # 高波动率阈值（3%）
    lookback_period: int = 20               # 大盘动量回溯期
    regime_params: Dict[str, MarketRegimeParams] = field(default_factory=lambda: {
        'strong':  MarketRegimeParams(max_total_pct=1.00, max_position_pct=0.20, max_positions=5),
        'neutral': MarketRegimeParams(max_total_pct=0.60, max_position_pct=0.15, max_positions=4),
        'weak':    MarketRegimeParams(max_total_pct=0.30, max_position_pct=0.10, max_positions=3),
    })


@dataclass
class RotationConfig:
    """每日轮动策略配置"""
    # 基础资金
    initial_capital: float = 1_000_000.0
    # 仓位参数
    max_total_pct: float = 0.90
    max_position_pct: float = 0.20
    max_positions: int = 5
    # 信号配置
    buy_signal_types: List[str] = field(default_factory=lambda: [
        'KDJ_GOLD', 'MACD_GOLD', 'MA_GOLD', 'VOL_GOLD', 'DMI_GOLD',
        'BOLL_BREAK', 'HIGH_BREAK'
    ])
    sell_signal_types: List[str] = field(default_factory=lambda: [
        'KDJ_DEATH', 'MACD_DEATH', 'MA_DEATH', 'VOL_DEATH', 'DMI_DEATH',
        'BOLL_BREAK_DOWN', 'HIGH_BREAK_DOWN'
    ])
    # 排序因子及权重
    rank_factor_weights: Dict[str, float] = field(default_factory=lambda: {
        'RSI_1': 0.20,
        'RET_20': 0.25,
        'VOLUME_RATIO': 0.15,
        'PB': 0.20,
        'PE_TTM': 0.20,
    })
    rank_factor_directions: Dict[str, int] = field(default_factory=lambda: {
        'RSI_1': 1,
        'RET_20': 1,
        'VOLUME_RATIO': 1,
        'PB': -1,
        'PE_TTM': -1,
    })
    # 市场状态调节
    market_regime: MarketRegimeConfig = field(default_factory=MarketRegimeConfig)
    # 股票池过滤
    exclude_st: bool = True
    exclude_limit_up: bool = True
    exclude_limit_down: bool = True
    exclude_suspended: bool = True
    # 大盘指数代码
    benchmark_index: str = 'sh000300'
