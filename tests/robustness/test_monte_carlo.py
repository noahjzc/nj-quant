import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
from robustness.monte_carlo import MonteCarloSim, MCSimulationResult


class TestMonteCarloSim:
    def setup_method(self):
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 252)

    def test_basic_run(self):
        sim = MonteCarloSim()
        result = sim.run(self.returns, n_sim=200)
        assert isinstance(result, MCSimulationResult)
        assert len(result.sharpe_distribution) == 200
        assert len(result.max_dd_distribution) == 200
        assert result.sharpe_95ci[0] <= result.mean_sharpe <= result.sharpe_95ci[1]

    def test_reproducible(self):
        sim1 = MonteCarloSim(seed=42)
        r1 = sim1.run(self.returns, n_sim=100)
        sim2 = MonteCarloSim(seed=42)
        r2 = sim2.run(self.returns, n_sim=100)
        assert r1.mean_sharpe == r2.mean_sharpe

    def test_too_few_returns(self):
        sim = MonteCarloSim()
        result = sim.run(np.array([0.01, -0.02]), n_sim=100)
        assert result.mean_sharpe == 0.0
