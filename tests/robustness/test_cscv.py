import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from robustness.cscv import CSCVAnalyzer, CSCVResult


class TestCSCVAnalyzer:
    def setup_method(self):
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 252)

    def test_basic_run(self):
        analyzer = CSCVAnalyzer(seed=42)
        result = analyzer.run(self.returns, n_split=10, n_comb=50)
        assert isinstance(result, CSCVResult)
        assert 0.0 <= result.overfit_probability <= 1.0

    def test_too_few_returns(self):
        analyzer = CSCVAnalyzer()
        result = analyzer.run(np.array([0.01, -0.02]), n_split=4)
        assert result.overfit_probability == 0.0
