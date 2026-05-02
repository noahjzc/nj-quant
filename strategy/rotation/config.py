"""策略配置类"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Alpha158 因子默认方向（基于金融直觉，非 IC 分析）
# 1 = 越大越好（正向因子），-1 = 越小越好（反向因子）
ALPHA158_DIRECTIONS = {
    # KBar: 阳线实体、上影线短 → 正向
    'KMID': 1, 'KLEN': -1, 'KMID2': 1,
    'KUP': -1, 'KUP2': -1, 'KLOW': 1, 'KLOW2': 1,
    'KSFT': 1, 'KSFT2': 1,
    # Price ratios: 价格相对位置高 → 正向
    'OPEN0': 1, 'HIGH0': 1, 'LOW0': -1,
    # Rolling — 动量类: 正
    'ROC': 1, 'MA': 1,
    # Rolling — 波动类: 负
    'STD': -1, 'RESI': -1,
    # Rolling — 趋势强度: 正
    'BETA': 1, 'RSQR': 1,
    # Rolling — 极值
    'MAX': 1, 'MIN': -1, 'QTLU': 1, 'QTLD': -1,
    # Rolling — 位置/排名
    'RANK': 1, 'RSV': 1,
    # Rolling — 路径
    'IMAX': -1, 'IMIN': 1, 'IMXD': -1,
    # Rolling — 价量相关
    'CORR': 1, 'CORD': 1,
    # Rolling — 日内涨跌比例
    'CNTP': 1, 'CNTN': -1, 'CNTD': 1,
    # Rolling — 价格动量 RSI
    'SUMP': 1, 'SUMN': -1, 'SUMD': 1,
    # Rolling — 成交量
    'VMA': -1, 'VSTD': -1, 'WVMA': -1,
    'VSUMP': 1, 'VSUMN': -1, 'VSUMD': 1,
}


def resolve_alpha_direction(factor_name: str) -> int:
    """根据因子名前缀推断方向。

    例如 'MA5' → 前缀 'MA' → 方向 1
        'STD20' → 前缀 'STD' → 方向 -1
    """
    if factor_name in ALPHA158_DIRECTIONS:
        return ALPHA158_DIRECTIONS[factor_name]
    # 尝试前缀匹配（如 MA5 → MA）
    for prefix, direction in sorted(ALPHA158_DIRECTIONS.items(),
                                     key=lambda x: -len(x[0])):
        if factor_name.startswith(prefix):
            return direction
    return 1  # 默认正向


def add_alpha158_factors(config, weight: float = 0.01, top_n: int = None):
    """向配置中添加 Alpha158 因子。

    从缓存中读取因子列名，按等权加入 rank_factor_weights。
    需要缓存已构建（含 alpha 因子列）。

    Args:
        config: RotationConfig 实例
        weight: 每个 alpha 因子的初始权重（默认 0.01，优化时可被 Optuna 覆盖）
        top_n: 仅添加前 N 个因子（None = 全部 157 个）

    Returns:
        config（原地修改）
    """
    from strategy.factors.alpha158 import Alpha158Calculator
    from copy import deepcopy

    calc = Alpha158Calculator()
    # 获取因子名列表（通过计算一个虚拟 DataFrame）
    import pandas as pd
    import numpy as np
    dummy = pd.DataFrame({
        'open': [10.0, 10.1], 'high': [10.3, 10.2],
        'low': [9.8, 9.9], 'close': [10.0, 10.1],
        'volume': [1e6, 1.1e6],
    })
    result = calc.compute(dummy)
    all_factors = list(result.columns)

    if top_n:
        all_factors = all_factors[:top_n]

    weights = dict(config.rank_factor_weights)
    directions = dict(config.rank_factor_directions)

    for f in all_factors:
        if f not in weights:
            weights[f] = weight
            directions[f] = resolve_alpha_direction(f)

    config.rank_factor_weights = weights
    config.rank_factor_directions = directions
    return config


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
        "PSY_BUY",
    ])
    buy_signal_mode: str = 'AND'  # 'OR': 任意信号触发 | 'AND': 所有信号同时触发
    sell_signal_types: List[str] = field(default_factory=lambda: [
        'KDJ_DEATH', 'MACD_DEATH', 'MA_DEATH', 'VOL_DEATH',
        'BOLL_BREAK_DOWN', 'HIGH_BREAK_DOWN', 'PSY_SELL'
    ])
    # 排序因子及权重
    rank_factor_weights: Dict[str, float] = field(default_factory=lambda: {
        "RSI_1": 0.12003422664972277,
        "RET_20": 0.12142937089247126,
        "VOLUME_RATIO": 0.05074829934621531,
        "PB": 0.18461876051434330,
        "PE_TTM": 0.19634114888331674,
        "OVERHEAT": 0.06756893445467133,
        "circulating_mv": 0.11111111111111110,
        "WR_10": 0.07407407407407407,
        "WR_14": 0.07407407407407407,
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
    exclude_new_stocks: bool = True
    exclude_limit_up: bool = True
    exclude_limit_down: bool = True
    exclude_suspended: bool = True
    # 大盘指数代码
    benchmark_index: str = 'sh000300'
    # Trial 提前淘汰：资产低于初始资本的该比例时触发 Optuna pruning
    min_asset_ratio: float = 0.5
    # ATR 止损止盈参数（StopLossStrategies）
    atr_period: int = 8
    stop_loss_mult: float = 1.8039991280707548
    take_profit_mult: float = 4.317584349749643
    trailing_pct: float = 0.09498461878539541
    trailing_start: float = 0.029083753533516884
    # 过热度惩罚
    overheat_rsi_threshold: float = 70.80430452511213  # RSI 超买阈值
    overheat_ret5_threshold: float = 0.11334939772795069  # 5日涨幅阈值