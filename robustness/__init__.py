# robustness/__init__.py
from robustness.monte_carlo import MonteCarloSim, MCSimulationResult

__all__ = [
    'MonteCarloSim', 'MCSimulationResult',
]

# Lazy imports for modules that will be added later
def __getattr__(name):
    if name == 'CSCVAnalyzer':
        from robustness.cscv import CSCVAnalyzer
        return CSCVAnalyzer
    if name == 'CSCVResult':
        from robustness.cscv import CSCVResult
        return CSCVResult
    if name == 'SensitivityAnalyzer':
        from robustness.sensitivity import SensitivityAnalyzer
        return SensitivityAnalyzer
    if name == 'SensitivityResult':
        from robustness.sensitivity import SensitivityResult
        return SensitivityResult
    if name == 'RobustnessAnalyzer':
        from robustness.robustness_analyzer import RobustnessAnalyzer
        return RobustnessAnalyzer
    if name == 'RobustnessReport':
        from robustness.robustness_analyzer import RobustnessReport
        return RobustnessReport
    if name == 'deflated_sharpe_ratio':
        from robustness.statistics import deflated_sharpe_ratio
        return deflated_sharpe_ratio
    if name == 'probability_of_backtest_overfit':
        from robustness.statistics import probability_of_backtest_overfit
        return probability_of_backtest_overfit
    raise AttributeError(f"module 'robustness' has no attribute {name!r}")
