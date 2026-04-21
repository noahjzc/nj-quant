"""
多因子模型默认配置

因子权重和方向配置
"""

# 默认因子配置
DEFAULT_FACTOR_CONFIG = {
    # 估值因子（低估值好）
    'PB': {
        'weight': 0.15,
        'direction': -1,
        'description': '市净率，越低越好'
    },
    'PE_TTM': {
        'weight': 0.10,
        'direction': -1,
        'description': '市盈率TTM，越低越好'
    },
    'PS_TTM': {
        'weight': 0.05,
        'direction': -1,
        'description': '市销率TTM，越低越好'
    },

    # 动量因子（强势好）
    'RSI_1': {
        'weight': 0.15,
        'direction': 1,
        'description': 'RSI短期，偏强好'
    },
    'KDJ_K': {
        'weight': 0.05,
        'direction': 1,
        'description': 'KDJ随机K，强势好'
    },

    # 趋势因子（趋势向上好）
    'MA_5': {
        'weight': 0.05,
        'direction': 1,
        'description': '5日均线偏多'
    },
    'MA_20': {
        'weight': 0.05,
        'direction': 1,
        'description': '20日均线偏多'
    },

    # 交易因子（活跃但不过度）
    'TURNOVER': {
        'weight': 0.10,
        'direction': 1,
        'description': '换手率，活跃好'
    },
    'VOLUME_RATIO': {
        'weight': 0.05,
        'direction': 1,
        'description': '量比，放量好'
    },

    # 波动因子（低波动稳健）
    'AMPLITUDE': {
        'weight': 0.05,
        'direction': -1,
        'description': '振幅，低波动好'
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