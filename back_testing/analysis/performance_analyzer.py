"""
Performance Analyzer - 绩效分析器

计算交易策略的绩效指标，包括：
- 总收益率 (total_return)
- 年化收益率 (annual_return)
- 最大回撤 (max_drawdown)
- Sharpe比率 (sharpe_ratio)
- Calmar比率 (calmar_ratio)
- Sortino比率 (sortino_ratio)
- 胜率 (win_rate)
- 盈亏比 (profit_loss_ratio)
"""

import math
from typing import Dict, List, Optional

import numpy as np


class PerformanceAnalyzer:
    """绩效分析器"""

    def __init__(
        self,
        trades: List[Dict],
        initial_capital: float = 1000000.0,
        benchmark_index: str = 'sh000001',
        risk_free_rate: float = 0.025
    ):
        """
        Args:
            trades: 交易记录列表，每条记录包含:
                - action: 'buy' | 'sell'
                - price: 成交价格
                - shares: 成交数量
                - return: 收益率（卖出时）
            initial_capital: 期初资金
            benchmark_index: 基准指数代码
            risk_free_rate: 无风险利率
        """
        self.trades = trades
        self.initial_capital = initial_capital
        self.benchmark_index = benchmark_index
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(self) -> Dict:
        """计算所有绩效指标"""
        # Extract sell trades with returns
        sell_trades = [t for t in self.trades if t.get('action') == 'sell' and 'return' in t]
        returns = [t['return'] for t in sell_trades]

        metrics = {
            'total_return': self._calculate_total_return(returns),
            'annual_return': self._calculate_annual_return(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'win_rate': self._calculate_win_rate(returns),
            'profit_loss_ratio': self._calculate_profit_loss_ratio(returns),
        }

        return metrics

    def _calculate_total_return(self, returns: List[float]) -> float:
        """计算总收益率"""
        if not returns:
            return 0.0

        # Compound returns
        total_return = 1.0
        for r in returns:
            total_return *= (1 + r)

        return total_return - 1.0

    def _calculate_annual_return(self, returns: List[float]) -> float:
        """计算年化收益率"""
        if not returns:
            return 0.0

        total_return = self._calculate_total_return(returns)

        # Assume each trade represents approximately 1 year
        # So annual return is approximately the total return
        n_years = len(returns) if len(returns) > 0 else 1

        if n_years == 0:
            return 0.0

        # Annualized return using compound method
        annual_return = (1 + total_return) ** (1 / n_years) - 1

        return annual_return

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """计算最大回撤"""
        if not returns:
            return 0.0

        # Build equity curve
        equity = [self.initial_capital]
        for r in returns:
            equity.append(equity[-1] * (1 + r))

        # Calculate drawdown at each point
        max_drawdown = 0.0
        peak = equity[0]

        for value in equity:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """计算Sharpe比率"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)  # Sample standard deviation

        if std_return == 0:
            return 0.0

        # Annualize: assume each return period is 1 year
        sharpe = (mean_return - self.risk_free_rate) / std_return

        return sharpe

    def _calculate_calmar_ratio(self, returns: List[float]) -> float:
        """计算Calmar比率"""
        if not returns:
            return 0.0

        total_return = self._calculate_total_return(returns)
        max_drawdown = self._calculate_max_drawdown(returns)

        if max_drawdown == 0:
            return float('inf') if total_return > 0 else 0.0

        return total_return / max_drawdown

    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """计算Sortino比率"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)

        # Only consider downside returns (below 0)
        downside_returns = returns_array[returns_array < 0]

        if len(downside_returns) == 0:
            return float('inf') if mean_return > self.risk_free_rate else 0.0

        # Use ddof=0 for single downside return to avoid NaN
        if len(downside_returns) == 1:
            # Single downside return: use its absolute value as the deviation
            downside_std = abs(downside_returns[0])
        else:
            downside_std = np.std(downside_returns, ddof=1)

        if downside_std == 0:
            return 0.0

        sortino = (mean_return - self.risk_free_rate) / downside_std

        return sortino

    def _calculate_win_rate(self, returns: List[float]) -> float:
        """计算胜率"""
        if not returns:
            return 0.0

        wins = sum(1 for r in returns if r > 0)
        total = len(returns)

        return wins / total

    def _calculate_profit_loss_ratio(self, returns: List[float]) -> float:
        """计算盈亏比"""
        if not returns:
            return 0.0

        wins = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]

        if not wins:
            return 0.0

        if not losses:
            return float('inf')

        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)

        if avg_loss == 0:
            return float('inf')

        return avg_win / avg_loss
