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
        risk_free_rate: float = 0.025,
        equity_curve: Optional[List[float]] = None,
        periods_per_year: int = 52,
        benchmark_returns: Optional[np.ndarray] = None,
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
            equity_curve: 组合净值序列（含起始值），用于正确计算总收益/回撤/Sharpe等。
                          当提供时，以下指标从净值曲线推导而非从个股交易收益率复利：
                          total_return, annual_return, max_drawdown, sharpe_ratio,
                          sortino_ratio, calmar_ratio。
                          win_rate 和 profit_loss_ratio 仍从个股交易计算。
            periods_per_year: 每年期数，用于年化。周频=52，日频=252，月频=12。
            benchmark_returns: 基准每日收益率序列，用于计算信息比率、alpha/beta等扩展指标。
        """
        self.trades = trades
        self.initial_capital = initial_capital
        self.benchmark_index = benchmark_index
        self.risk_free_rate = risk_free_rate
        self.equity_curve = equity_curve
        self.periods_per_year = periods_per_year
        self.benchmark_returns = benchmark_returns
        self.daily_returns: Optional[np.ndarray] = None

    def calculate_metrics(self) -> Dict:
        """计算所有绩效指标"""
        # Extract completed trades with returns (any action type with a return field)
        completed_trades = [t for t in self.trades if 'return' in t]
        returns = [t['return'] for t in completed_trades]

        # Trade-level metrics (always from individual trades)
        win_rate = self._calculate_win_rate(returns)
        profit_loss_ratio = self._calculate_profit_loss_ratio(returns)

        # Portfolio-level metrics
        if self.equity_curve is not None and len(self.equity_curve) > 1:
            equity = np.array(self.equity_curve)
            period_returns = (equity[1:] - equity[:-1]) / equity[:-1]
            self.daily_returns = period_returns  # expose for external consumers

            total_return = (equity[-1] / equity[0]) - 1

            n_periods = len(period_returns)
            annual_return = (1 + total_return) ** (self.periods_per_year / n_periods) - 1

            max_drawdown = self._max_drawdown_from_equity(equity)

            # Sharpe
            excess_annual = np.mean(period_returns) * self.periods_per_year - self.risk_free_rate
            annual_vol = np.std(period_returns, ddof=1) * np.sqrt(self.periods_per_year)
            sharpe = excess_annual / annual_vol if annual_vol > 1e-10 else 0.0

            # Sortino
            downside = period_returns[period_returns < 0]
            if len(downside) == 0:
                sortino = float('inf') if excess_annual > 0 else 0.0
            elif len(downside) == 1:
                ds_annual = abs(downside[0]) * np.sqrt(self.periods_per_year)
                sortino = excess_annual / ds_annual if ds_annual > 1e-10 else 0.0
            else:
                ds_annual = np.std(downside, ddof=1) * np.sqrt(self.periods_per_year)
                sortino = excess_annual / ds_annual if ds_annual > 1e-10 else 0.0

            # Calmar
            calmar = annual_return / max_drawdown if max_drawdown > 0 else (
                float('inf') if annual_return > 0 else 0.0
            )
        else:
            total_return = self._calculate_total_return(returns)
            annual_return = self._calculate_annual_return(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            sharpe = self._calculate_sharpe_ratio(returns)
            sortino = self._calculate_sortino_ratio(returns)
            calmar = self._calculate_calmar_ratio(returns)

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'sortino_ratio': sortino,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
        }

        return metrics

    def information_ratio(self) -> float:
        """信息比率 = (策略年化 - 基准年化) / 跟踪误差"""
        if self.benchmark_returns is None or self.daily_returns is None:
            return 0.0
        if len(self.benchmark_returns) != len(self.daily_returns):
            return 0.0
        excess = self.daily_returns - self.benchmark_returns
        if np.std(excess, ddof=1) < 1e-10:
            return 0.0
        ann_excess = np.mean(excess) * self.periods_per_year
        ann_te = np.std(excess, ddof=1) * np.sqrt(self.periods_per_year)
        return ann_excess / ann_te

    def alpha_beta(self) -> Dict:
        """CAPM 回归: 策略收益 ~ 基准收益，返回 alpha, beta, r_squared"""
        if self.benchmark_returns is None or self.daily_returns is None:
            return {'alpha': 0.0, 'beta': 0.0, 'r_squared': 0.0}
        if len(self.benchmark_returns) != len(self.daily_returns):
            return {'alpha': 0.0, 'beta': 0.0, 'r_squared': 0.0}
        X = self.benchmark_returns
        y = self.daily_returns
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        denom = np.sum((X - X_mean) ** 2)
        if denom < 1e-10:
            return {'alpha': 0.0, 'beta': 0.0, 'r_squared': 0.0}
        beta = np.sum((X - X_mean) * (y - y_mean)) / denom
        alpha = y_mean - beta * X_mean
        ss_res = np.sum((y - (alpha + beta * X)) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        return {'alpha': float(alpha), 'beta': float(beta), 'r_squared': float(r_squared)}

    def skewness_kurtosis(self) -> Dict:
        """收益偏度和超额峰度"""
        if self.daily_returns is None or len(self.daily_returns) < 3:
            return {'skewness': 0.0, 'kurtosis': 0.0}
        from scipy import stats
        skew = stats.skew(self.daily_returns)
        kurt = stats.kurtosis(self.daily_returns)
        return {'skewness': float(skew), 'kurtosis': float(kurt)}

    def rolling_sharpe(self, window: int = 60) -> np.ndarray:
        """滚动 Sharpe 比率"""
        if self.daily_returns is None or len(self.daily_returns) < window:
            return np.array([])
        r = self.daily_returns
        rf_daily = self.risk_free_rate / self.periods_per_year
        rolling_mean = np.convolve(r - rf_daily, np.ones(window) / window, mode='valid')
        result = np.zeros(len(rolling_mean))
        for i in range(len(rolling_mean)):
            seg = r[i:i + window]
            seg_std = np.std(seg, ddof=1)
            if seg_std > 1e-10:
                result[i] = rolling_mean[i] * np.sqrt(self.periods_per_year) / seg_std
        return result

    def monthly_returns(self) -> Dict:
        """月度收益明细（按21个交易日分组）"""
        if self.daily_returns is None or len(self.daily_returns) == 0:
            return {}
        monthly = {}
        n = len(self.daily_returns)
        for i, start in enumerate(range(0, n, 21)):
            end = min(start + 21, n)
            period = self.daily_returns[start:end]
            monthly_ret = float(np.prod(1 + period) - 1)
            month_label = f"Month_{i + 1:02d}"
            monthly[month_label] = monthly_ret
        return monthly

    def max_drawdown_duration(self) -> int:
        """最长回撤持续天数"""
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return 0
        equity = np.array(self.equity_curve)
        peak = equity[0]
        max_dur = 0
        current_dur = 0
        for value in equity[1:]:
            if value >= peak:
                peak = value
                current_dur = 0
            else:
                current_dur += 1
                if current_dur > max_dur:
                    max_dur = current_dur
        return max_dur

    @staticmethod
    def _max_drawdown_from_equity(equity: np.ndarray) -> float:
        """从净值序列计算最大回撤"""
        peak = equity[0]
        max_dd = 0.0
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

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
