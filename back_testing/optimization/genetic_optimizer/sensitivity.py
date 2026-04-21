"""Sensitivity Analysis: Factor importance evaluation"""
import numpy as np
import pandas as pd
from typing import Dict


class SensitivityAnalyzer:
    """
    Analyze factor importance by testing weight sensitivity.

    Method:
    1. For each factor, vary its weight while keeping others fixed
    2. Measure change in Calmar ratio
    3. Higher sensitivity = more important factor
    """

    def __init__(self, evaluator):
        """
        Args:
            evaluator: FitnessEvaluator instance
        """
        self.evaluator = evaluator

    def analyze_factor_importance(self,
                                  optimal_weights: Dict[str, float],
                                  data,
                                  factor_range: tuple = (0.0, 0.5),
                                  steps: int = 10) -> pd.DataFrame:
        """
        Analyze importance of each factor.

        Args:
            optimal_weights: Optimal weights from GA
            data: (start_date, end_date) tuple for evaluation
            factor_range: Weight range to test
            steps: Number of steps in range

        Returns:
            DataFrame with factor importance scores
        """
        results = []

        for factor, base_weight in optimal_weights.items():
            sensitivities = []

            for pct in np.linspace(factor_range[0], factor_range[1], steps):
                test_weights = optimal_weights.copy()
                test_weights[factor] = pct

                total = sum(test_weights.values())
                test_weights = {k: v / total for k, v in test_weights.items()}

                fitness = self.evaluator.evaluate(test_weights, data[0], data[1])
                sensitivities.append(fitness)

            sensitivity = np.std(sensitivities)
            results.append({
                'factor': factor,
                'base_weight': base_weight,
                'sensitivity': sensitivity,
                'fitness_range': max(sensitivities) - min(sensitivities)
            })

        return pd.DataFrame(results).sort_values('sensitivity', ascending=False)
