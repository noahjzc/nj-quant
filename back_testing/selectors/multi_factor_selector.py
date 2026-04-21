"""Multi-factor stock selector using weighted scoring."""
from typing import Dict, List, Optional

import pandas as pd

from back_testing.factors.factor_utils import FactorProcessor


class MultiFactorSelector:
    """Multi-factor stock selector based on weighted scoring.

    This class implements a multi-factor selection model that:
    1. Standardizes factors using rank or z-score method
    2. Adjusts for factor direction (positive = larger is better, negative = smaller is better)
    3. Computes weighted composite scores
    4. Selects top N stocks by composite score

    Args:
        weights: Factor weight dictionary, e.g., {'PB': 0.3, 'ROE': 0.2, ...}
        directions: Factor direction, 1 means larger is better, -1 means smaller is better
        method: 'rank' or 'zscore' standardization method
    """

    def __init__(self, weights: Dict[str, float], directions: Dict[str, int], method: str = 'rank'):
        """Initialize MultiFactorSelector.

        Args:
            weights: Factor weight dictionary.
            directions: Factor direction dictionary.
            method: Standardization method - 'rank' or 'zscore'.

        Raises:
            ValueError: If method is not 'rank' or 'zscore'.
        """
        if method not in ('rank', 'zscore'):
            raise ValueError(f"Invalid method '{method}'. Use 'rank' or 'zscore'.")

        self.weights = weights
        self.directions = directions
        self.method = method
        self._factor_processor = FactorProcessor()

    def calculate_factor_scores(self, data: pd.DataFrame) -> pd.Series:
        """Calculate composite factor scores for all stocks.

        Args:
            data: DataFrame with factor columns as keys and stock data as values.
                  Index should be stock codes.

        Returns:
            Series with stock codes as index and composite scores as values.
        """
        if data.empty:
            return pd.Series(dtype=float)

        # Get factor columns that exist in data
        factor_columns = [f for f in self.weights.keys() if f in data.columns]
        if not factor_columns:
            return pd.Series(0.0, index=data.index)

        # Handle single stock case - return neutral score of 0.5
        if len(data) == 1:
            return pd.Series([0.5], index=data.index)

        composite_scores = pd.Series(0.0, index=data.index)
        total_weight = sum(self.weights[f] for f in factor_columns)

        for factor in factor_columns:
            # Get factor direction (default to 1 if not specified)
            direction = self.directions.get(factor, 1)

            # For direction=-1 (smaller is better): ascending=True so lower values rank first
            # For direction=1 (larger is better): ascending=False so higher values rank first
            ascending = True if direction == -1 else False

            # Standardize factor using specified method
            if self.method == 'rank':
                processed = self._factor_processor.rank_percentile(
                    data[factor], ascending=ascending
                )
            else:  # zscore
                # For zscore with direction=-1, we need to negate BEFORE z_score
                # so that lower original values get higher z-scores (which normalize higher)
                if direction == -1:
                    processed = self._factor_processor.z_score(-data[factor])
                else:
                    processed = self._factor_processor.z_score(data[factor])

            # For rank method with direction=1, flip so higher original values get higher scores
            # (ascending=False ranks higher values first with lower percentile)
            if self.method == 'rank' and direction == 1:
                processed = 1 - processed

            # Add weighted contribution to composite score
            composite_scores += processed * self.weights[factor] / total_weight

        return composite_scores

    def select_top_stocks(self, data: pd.DataFrame, n: int = 5, excluded: List[str] = None) -> List[str]:
        """Select top N stocks by composite factor score.

        Args:
            data: DataFrame with factor columns and stock data.
            n: Number of stocks to select.
            excluded: List of stock codes to exclude from selection.

        Returns:
            List of selected stock codes sorted by score (highest first).
        """
        if excluded is None:
            excluded = []

        # Calculate scores
        scores = self.calculate_factor_scores(data)

        # Filter out excluded stocks
        available = scores.drop(index=[s for s in excluded if s in scores.index], errors='ignore')

        if available.empty:
            return []

        # Sort by score descending and select top N
        sorted_scores = available.sort_values(ascending=False)
        result = sorted_scores.head(n).index.tolist()

        return result

    def get_factor_contribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get each factor's contribution to the composite score.

        Args:
            data: DataFrame with factor columns and stock data.

        Returns:
            DataFrame with factor names as columns and stock codes as index.
            Each cell contains the weighted contribution of that factor.
        """
        if data.empty:
            return pd.DataFrame()

        # Get factor columns that exist in data
        factor_columns = [f for f in self.weights.keys() if f in data.columns]
        if not factor_columns:
            return pd.DataFrame(index=data.index)

        # Handle single stock case
        if len(data) == 1:
            contributions = pd.DataFrame(0.5, index=data.index, columns=factor_columns)
            for factor in factor_columns:
                direction = self.directions.get(factor, 1)
                if direction == -1:
                    contributions.loc[data.index[0], factor] = 0.0
                else:
                    contributions.loc[data.index[0], factor] = 1.0
                contributions.loc[data.index[0], factor] *= self.weights[factor]
            return contributions

        total_weight = sum(self.weights[f] for f in factor_columns)
        contributions = pd.DataFrame(0.0, index=data.index, columns=factor_columns)

        for factor in factor_columns:
            direction = self.directions.get(factor, 1)
            ascending = True if direction == -1 else False

            # Standardize factor
            if self.method == 'rank':
                processed = self._factor_processor.rank_percentile(
                    data[factor], ascending=ascending
                )
            else:
                # For zscore with direction=-1, negate before z_score
                if direction == -1:
                    processed = self._factor_processor.z_score(-data[factor])
                else:
                    processed = self._factor_processor.z_score(data[factor])

            # For rank method with direction=1, flip so higher original values get higher scores
            if self.method == 'rank' and direction == 1:
                processed = 1 - processed

            # Store weighted contribution
            contributions[factor] = processed * self.weights[factor] / total_weight

        return contributions
