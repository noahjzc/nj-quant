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
    strong_trend_threshold: float = 0.05  # 大盘MA多头阈值（5%）
    weak_trend_threshold: float = -0.03  # 大盘MA空头阈值（-3%）
    high_volatility_threshold: float = 0.03  # 高波动率阈值（3%）
    lookback_period: int = 20  # 大盘动量回溯期
    regime_params: Dict[str, MarketRegimeParams] = field(default_factory=lambda: {
        'strong': MarketRegimeParams(max_total_pct=1.00, max_position_pct=0.20, max_positions=8),
        'neutral': MarketRegimeParams(max_total_pct=0.60, max_position_pct=0.15, max_positions=6),
        'weak': MarketRegimeParams(max_total_pct=0.30, max_position_pct=0.10, max_positions=4),
    })


@dataclass
class RotationConfig:
    """每日轮动策略配置"""
    # 基础资金
    initial_capital: float = 1_000_000.0
    # 仓位参数
    max_total_pct: float = 0.9818022008269246
    max_position_pct: float = 0.24170239102375576
    max_positions: int = 3
    # 信号配置
    buy_signal_types: List[str] = field(default_factory=lambda: [
        "KDJ_GOLD",
        "MACD_GOLD",
        "HIGH_BREAK",
        "KDJ_GOLD_LOW",
    ])
    buy_signal_mode: str = 'AND'  # 'OR': 任意信号触发 | 'AND': 所有信号同时触发
    sell_signal_types: List[str] = field(default_factory=lambda: [
        'KDJ_DEATH', 'MACD_DEATH', 'MA_DEATH', 'VOL_DEATH',
        'BOLL_BREAK_DOWN', 'HIGH_BREAK_DOWN', 'PSY_SELL'
    ])
    # 排序因子及权重
    rank_factor_weights: Dict[str, float] = field(default_factory=lambda: {
        "RSI_1": 0.16204620597712574,
        "RET_20": 0.1639296507048362,
        "VOLUME_RATIO": 0.06851020411739067,
        "PB": 0.24923532669436346,
        "PE_TTM": 0.2650605509924776,
        "OVERHEAT": 0.0912180615138063,
        "circulating_mv": 0.15,
        "WR_10": 0.10,
        "WR_14": 0.10,
    })
    rank_factor_directions: Dict[str, int] = field(default_factory=lambda: {
        'RSI_1': 1,
        'RET_20': 1,
        'VOLUME_RATIO': 1,
        'PB': -1,
        'PE_TTM': -1,
        'OVERHEAT': -1,
        'circulating_mv': -1,
        'WR_10': -1,
        'WR_14': -1,
    })
    # KDJ 低位金叉阈值
    kdj_low_threshold: float = 30.0
    # 市场状态调节
    market_regime: MarketRegimeConfig = field(default_factory=MarketRegimeConfig)
    # 股票池过滤
    exclude_st: bool = True
    exclude_limit_up: bool = True
    exclude_limit_down: bool = True
    exclude_suspended: bool = True
    # 大盘指数代码
    benchmark_index: str = 'sh000300'
    # ATR 止损止盈参数（StopLossStrategies）
    atr_period: int = 8
    stop_loss_mult: float = 1.8039991280707548
    take_profit_mult: float = 4.317584349749643
    trailing_pct: float = 0.09498461878539541
    trailing_start: float = 0.029083753533516884
    # 过热度惩罚
    overheat_rsi_threshold: float = 70.80430452511213  # RSI 超买阈值
    overheat_ret5_threshold: float = 0.11334939772795069  # 5日涨幅阈值
