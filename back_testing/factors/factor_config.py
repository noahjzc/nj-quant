"""
多因子模型默认配置

因子权重和方向配置
"""

# 默认因子配置
DEFAULT_FACTOR_CONFIG = {
    # 动量因子（强势好）
    'RSI_1': {
        'weight': 0.20,
        'direction': 1,
        'description': 'RSI短期，偏强好'
    },
    'RSI_2': {
        'weight': 0.10,
        'direction': 1,
        'description': 'RSI中期，偏强好'
    },
    'RSI_3': {
        'weight': 0.05,
        'direction': 1,
        'description': 'RSI长期，偏强好'
    },
    'KDJ_K': {
        'weight': 0.15,
        'direction': 1,
        'description': 'KDJ随机K，强势好'
    },
    'KDJ_D': {
        'weight': 0.05,
        'direction': 1,
        'description': 'KDJ随机D，强势好'
    },

    # 趋势因子（趋势向上好）
    'MA_5': {
        'weight': 0.15,
        'direction': 1,
        'description': '5日均线偏多'
    },
    'MA_10': {
        'weight': 0.10,
        'direction': 1,
        'description': '10日均线偏多'
    },
    'MA_20': {
        'weight': 0.10,
        'direction': 1,
        'description': '20日均线偏多'
    },
    'MA_30': {
        'weight': 0.10,
        'direction': 1,
        'description': '30日均线偏多'
    },

    # 新增动量因子
    'RET_20': {
        'weight': 0.10,
        'direction': 1,
        'description': '20日价格动量，越强越好'
    },
    'RET_60': {
        'weight': 0.10,
        'direction': 1,
        'description': '60日价格动量，越强越好'
    },
    'LN_MCAP': {
        'weight': 0.05,
        'direction': -1,
        'description': '对数市值，越小越好'
    },
}

def get_factor_weights(config: dict = None) -> dict:
    """获取因子权重"""
    if config is None:
        config = DEFAULT_FACTOR_CONFIG
    return {k: v['weight'] for k, v in config.items()}

def get_factor_directions(config: dict = None) -> dict:
    """获取因子方向"""
    if config is None:
        config = DEFAULT_FACTOR_CONFIG
    return {k: v['direction'] for k, v in config.items()}