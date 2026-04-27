"""RotationFitnessEvaluator — GA 适应度评估器"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from back_testing.rotation.daily_rotation_engine import DailyRotationEngine
from back_testing.rotation.config import RotationConfig, MarketRegimeConfig, MarketRegimeParams


class RotationFitnessEvaluator:
    """
    GA 适应度评估器 — 每日轮动策略版本

    给定参数配置，运行回测，返回绩效指标（Sharpe 或总收益）
    用于 GA 遗传算法优化参数搜索。
    """

    def __init__(self, max_drawdown_constraint: float = 0.30, benchmark_index: str = 'sh000300'):
        self.max_drawdown_constraint = max_drawdown_constraint
        self.benchmark_index = benchmark_index
        self._cache = {}

    def evaluate(self, genome: Dict, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
        """
        评估参数配置

        Args:
            genome: GA 参数字典，如 {
                'max_positions': 5,
                'max_total_pct': 0.9,
                'max_position_pct': 0.2,
                'factor_weights': {'RSI_1': 0.2, 'RET_20': 0.25, ...},
                'buy_signal_types': ['KDJ_GOLD', 'MACD_GOLD'],
            }
            start_date: 回测开始日期
            end_date: 回测结束日期

        Returns:
            适应度分数（Sharpe Ratio，约束违反则返回 0）
        """
        cache_key = (start_date, end_date, tuple(sorted(genome.items())))
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            config = self._genome_to_config(genome)
            engine = DailyRotationEngine(config, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            results = engine.run()

            if not results:
                return 0.0

            # 计算净值序列
            values = [r.total_asset for r in results]
            if len(values) < 2:
                return 0.0

            # 计算绩效
            perf = self._calculate_performance(values, genome)
            if perf['max_drawdown'] > self.max_drawdown_constraint:
                return 0.0

            self._cache[cache_key] = perf['sharpe']
            return perf['sharpe']

        except Exception as e:
            print(f"[RotationFitness] 评估异常: {e}")
            return 0.0

    def _genome_to_config(self, genome: Dict) -> RotationConfig:
        """将 GA genome 映射为 RotationConfig"""
        regime_params = {}
        for regime_name in ('strong', 'neutral', 'weak'):
            p = genome.get(f'regime_{regime_name}', {})
            regime_params[regime_name] = MarketRegimeParams(
                max_total_pct=p.get('max_total_pct', 0.9),
                max_position_pct=p.get('max_position_pct', 0.2),
                max_positions=p.get('max_positions', 5),
            )

        market_regime = MarketRegimeConfig(
            strong_trend_threshold=genome.get('strong_trend_threshold', 0.05),
            weak_trend_threshold=genome.get('weak_trend_threshold', -0.03),
            high_volatility_threshold=genome.get('high_volatility_threshold', 0.03),
            lookback_period=genome.get('lookback_period', 20),
            regime_params=regime_params,
        )

        return RotationConfig(
            initial_capital=genome.get('initial_capital', 1_000_000),
            max_total_pct=genome.get('max_total_pct', 0.9),
            max_position_pct=genome.get('max_position_pct', 0.2),
            max_positions=genome.get('max_positions', 5),
            buy_signal_types=genome.get('buy_signal_types', ['KDJ_GOLD', 'MACD_GOLD', 'MA_GOLD']),
            sell_signal_types=genome.get('sell_signal_types', ['KDJ_DEATH', 'MACD_DEATH', 'MA_DEATH']),
            rank_factor_weights=genome.get('factor_weights', {
                'RSI_1': 0.2, 'RET_20': 0.25, 'VOLUME_RATIO': 0.15, 'PB': 0.2, 'PE_TTM': 0.2,
            }),
            rank_factor_directions=genome.get('factor_directions', {
                'RSI_1': 1, 'RET_20': 1, 'VOLUME_RATIO': 1, 'PB': -1, 'PE_TTM': -1,
            }),
            market_regime=market_regime,
            benchmark_index=genome.get('benchmark_index', 'sh000300'),
        )

    def _calculate_performance(self, values: List[float], genome: Dict) -> Dict:
        """计算绩效指标"""
        values = np.array(values)
        returns = np.diff(values) / values[:-1]
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return {'sharpe': 0.0, 'max_drawdown': 0.0, 'total_return': 0.0}

        # 总收益率
        total_return = (values[-1] / values[0]) - 1

        # 最大回撤
        peak = values[0]
        max_drawdown = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_drawdown:
                max_drawdown = dd

        # 年化收益率
        n_years = len(values) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Sharpe（无风险利率=0）
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        return {
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'annual_return': annual_return,
        }

    def clear_cache(self):
        self._cache.clear()
