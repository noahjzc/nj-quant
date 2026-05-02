from dataclasses import dataclass
import numpy as np


@dataclass
class CSCVResult:
    overfit_probability: float = 0.0
    rank_decay: float = 0.0
    is_robust: bool = False


class CSCVAnalyzer:
    """CSCV 过拟合检测 (Bailey et al. 2017)"""

    def __init__(self, rf_annual: float = 0.025, periods_per_year: int = 252, seed: int = None):
        self.rf = rf_annual
        self.ppy = periods_per_year
        self.rng = np.random.RandomState(seed)

    def run(self, daily_returns: np.ndarray, n_split: int = 16, n_comb: int = 100) -> CSCVResult:
        n = len(daily_returns)
        if n < n_split * 2:
            return CSCVResult()

        segment_size = n // n_split
        segments = []
        for i in range(n_split):
            seg = daily_returns[i * segment_size:(i + 1) * segment_size]
            segments.append(seg)

        is_ranks = []
        oos_ranks = []

        for _ in range(n_comb):
            indices = list(range(n_split))
            self.rng.shuffle(indices)
            half = n_split // 2
            is_idx = indices[:half]
            oos_idx = indices[half:]

            is_returns = np.concatenate([segments[i] for i in is_idx])
            oos_returns = np.concatenate([segments[i] for i in oos_idx])

            is_sharpe = self._sharpe(is_returns)
            oos_sharpe = self._sharpe(oos_returns)

            is_ranks.append(is_sharpe)
            oos_ranks.append(oos_sharpe)

        is_ranks = np.array(is_ranks)
        oos_ranks = np.array(oos_ranks)

        is_best = np.argmax(is_ranks)
        oos_rank_of_best = np.sum(oos_ranks > oos_ranks[is_best]) / (len(oos_ranks) - 1) if len(oos_ranks) > 1 else 0.0

        pbo = 1.0 - oos_rank_of_best if oos_rank_of_best > 0 else 0.0

        return CSCVResult(
            overfit_probability=float(np.clip(pbo, 0.0, 1.0)),
            rank_decay=float(oos_rank_of_best),
            is_robust=pbo < 0.1,
        )

    def _sharpe(self, returns: np.ndarray) -> float:
        excess = np.mean(returns) - self.rf / self.ppy
        vol = np.std(returns, ddof=1)
        if vol < 1e-10:
            return 0.0
        return excess / vol * np.sqrt(self.ppy)
