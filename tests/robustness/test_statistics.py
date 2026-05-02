import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from robustness.statistics import deflated_sharpe_ratio, probability_of_backtest_overfit


class TestDeflatedSharpe:
    def test_basic(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        dsr = deflated_sharpe_ratio(returns, n_trials=50)
        assert isinstance(dsr, float)

    def test_single_trial(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        dsr = deflated_sharpe_ratio(returns, n_trials=1)
        assert isinstance(dsr, float)

    def test_few_returns(self):
        dsr = deflated_sharpe_ratio(np.array([0.01]))
        assert dsr == 0.0

    def test_no_variance(self):
        dsr = deflated_sharpe_ratio(np.array([0.001, 0.001, 0.001]), n_trials=100)
        assert isinstance(dsr, float)


class TestPBO:
    def test_basic(self):
        np.random.seed(42)
        is_sharpes = np.random.normal(1.0, 0.3, 100)
        oos_sharpes = np.random.normal(0.5, 0.4, 100)
        pbo = probability_of_backtest_overfit(is_sharpes, oos_sharpes)
        assert 0.0 <= pbo <= 1.0

    def test_empty(self):
        pbo = probability_of_backtest_overfit(np.array([]), np.array([]))
        assert pbo == 0.0

    def test_perfect_alignment(self):
        """IS and OOS perfectly correlated → low PBO"""
        np.random.seed(42)
        is_sharpes = np.arange(10, dtype=float)
        oos_sharpes = np.arange(10, dtype=float)
        pbo = probability_of_backtest_overfit(is_sharpes, oos_sharpes)
        assert pbo == 0.0

    def test_reverse_alignment(self):
        """IS and OOS perfectly reversed → high PBO"""
        np.random.seed(42)
        is_sharpes = np.arange(10, dtype=float)
        oos_sharpes = np.arange(9, -1, -1, dtype=float)
        pbo = probability_of_backtest_overfit(is_sharpes, oos_sharpes)
        # Perfect reversal: top-half IS → bottom-half OOS, PBO = 0.5
        assert pbo >= 0.5
