import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from robustness.sensitivity import SensitivityAnalyzer, SensitivityResult


class TestSensitivityAnalyzer:
    def test_empty_params(self):
        analyzer = SensitivityAnalyzer()
        result = analyzer.run({}, engine_factory=lambda p: None)
        assert result.overall_stability_score == 1.0
        assert len(result.per_param) == 0

    def test_stable_params(self):
        analyzer = SensitivityAnalyzer()
        params = {'a': 1.0, 'b': 2.0}

        def mock_run(cfg):
            if cfg['a'] == 1.0 and cfg['b'] == 2.0:
                return {'sharpe_ratio': 1.5}
            return {'sharpe_ratio': 1.48}

        result = analyzer.run(params, mock_run)
        assert result.overall_stability_score > 0.8
        assert len(result.per_param) == 2
        assert result.per_param['a']['stable'] is True

    def test_unstable_params(self):
        analyzer = SensitivityAnalyzer()
        params = {'x': 10.0}

        def mock_run(cfg):
            if cfg['x'] == 10.0:
                return {'sharpe_ratio': 1.5}
            return {'sharpe_ratio': 0.3}  # massive drop → unstable

        result = analyzer.run(params, mock_run)
        assert result.per_param['x']['stable'] is False
        assert result.overall_stability_score < 0.5

    def test_zero_base_sharpe(self):
        analyzer = SensitivityAnalyzer()
        params = {'a': 1.0}

        def mock_run(cfg):
            return {'sharpe_ratio': 0.0}

        result = analyzer.run(params, mock_run)
        # When base is 0, change is computed as 0% → stable
        assert result.per_param['a']['sharpe_change_pct'] == 0.0
