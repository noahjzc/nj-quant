from dataclasses import dataclass, field
import numpy as np


@dataclass
class MCSimulationResult:
    mean_sharpe: float = 0.0
    sharpe_std: float = 0.0
    sharpe_95ci: tuple = (0.0, 0.0)
    sharpe_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_max_dd: float = 0.0
    max_dd_95ci: tuple = (0.0, 0.0)
    max_dd_distribution: np.ndarray = field(default_factory=lambda: np.array([]))


class MonteCarloSim:
    """非参数蒙特卡洛模拟：对日收益率放回重采样，生成 N 条模拟净值曲线"""

    def __init__(self, rf_annual: float = 0.025, periods_per_year: int = 252, seed: int = None):
        self.rf = rf_annual
        self.ppy = periods_per_year
        self.rng = np.random.RandomState(seed)

    def run(self, daily_returns: np.ndarray, n_sim: int = 2000) -> MCSimulationResult:
        if len(daily_returns) < 5 or n_sim < 1:
            return MCSimulationResult()

        n = len(daily_returns)
        sharpes = np.empty(n_sim)
        max_dds = np.empty(n_sim)

        for i in range(n_sim):
            idx = self.rng.randint(0, n, size=n)
            sampled = daily_returns[idx]
            equity = np.cumprod(1 + sampled)
            sharpes[i] = self._sharpe(sampled)
            max_dds[i] = self._max_drawdown(equity)

        return MCSimulationResult(
            mean_sharpe=float(np.mean(sharpes)),
            sharpe_std=float(np.std(sharpes, ddof=1)),
            sharpe_95ci=(float(np.percentile(sharpes, 2.5)), float(np.percentile(sharpes, 97.5))),
            sharpe_distribution=sharpes,
            mean_max_dd=float(np.mean(max_dds)),
            max_dd_95ci=(float(np.percentile(max_dds, 2.5)), float(np.percentile(max_dds, 97.5))),
            max_dd_distribution=max_dds,
        )

    def _sharpe(self, returns: np.ndarray) -> float:
        excess = np.mean(returns) - self.rf / self.ppy
        vol = np.std(returns, ddof=1)
        if vol < 1e-10:
            return 0.0
        return excess / vol * np.sqrt(self.ppy)

    def _max_drawdown(self, equity: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        return float(np.max(dd))
