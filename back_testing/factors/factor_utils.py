"""Factor processing utilities for standardization, ranking, and neutralization."""
import numpy as np
import pandas as pd


class FactorProcessor:
    """Utility class for factor data processing including standardization,
    ranking, winsorization, and market neutralization.
    """

    @staticmethod
    def rank_percentile(series: pd.Series, ascending: bool = True) -> pd.Series:
        """Calculate percentile rank (0-1) of the series values.

        Args:
            series: Input data series.
            ascending: If True, lower values get lower scores (good for PB).
                      If False, higher values get lower scores.

        Returns:
            Series with percentile ranks in [0, 1].
        """
        n = len(series)
        if n <= 1:
            return pd.Series(np.zeros(n), index=series.index)

        rank = series.rank(method='average', ascending=ascending)
        # Convert to 0-based percentile: (rank - 1) / (n - 1)
        return (rank - 1) / (n - 1)

    @staticmethod
    def z_score(series: pd.Series) -> pd.Series:
        """Z-score standardization normalized to [0, 1] range.

        Computes z-scores (mean=0, std=1) and then normalizes to [0, 1]
        using min-max scaling for consistency with rank_percentile output.

        Args:
            series: Input data series.

        Returns:
            Series with values normalized to [0, 1] range.
        """
        mean = series.mean()
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(np.zeros(len(series)), index=series.index)
        z = (series - mean) / std
        # Normalize to [0, 1] range for consistency with rank_percentile
        z_min = z.min()
        z_max = z.max()
        if z_max == z_min:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (z - z_min) / (z_max - z_min)

    @staticmethod
    def winsorize(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
        """Clip extreme values to percentile bounds.

        Args:
            series: Input data series.
            lower: Lower percentile bound (default 0.05).
            upper: Upper percentile bound (default 0.95).

        Returns:
            Series with extreme values clipped to percentile bounds.
        """
        lower_bound = series.quantile(lower)
        upper_bound = series.quantile(upper)
        # Convert to float first to avoid FutureWarning from clip downcasting
        return series.astype(float).clip(lower=lower_bound, upper=upper_bound)

    @staticmethod
    def neutralize(series: pd.Series, market_cap: pd.Series) -> pd.Series:
        """Market neutralization via regression residual.

        Regresses the factor on log(market cap) and returns the residual,
        making the factor orthogonal to market cap. Using log(market cap)
        is standard quant practice as market cap typically has a log-normal
        distribution.

        Args:
            series: Factor values to neutralize.
            market_cap: Market cap values for regression.

        Returns:
            Market-neutral factor (residual from regression).
        """
        # Align indices
        common_idx = series.index.intersection(market_cap.index)
        if len(common_idx) == 0:
            return series.copy()

        y = series.loc[common_idx].values.astype(float)
        # Use log(market cap) - standard quant practice
        x = np.log(market_cap.loc[common_idx].values)

        # Add constant for intercept: y = alpha + beta * x + residual
        # Using least squares: (X'X)^-1 * X'y
        X = np.column_stack([np.ones(len(x)), x])
        # Solve normal equation: X'X * beta = X'y
        # beta = (X'X)^-1 * X'y
        XtX = X.T @ X
        Xty = X.T @ y
        beta = np.linalg.solve(XtX, Xty)

        # Predicted: alpha + beta * x
        predicted = beta[0] + beta[1] * x

        # Residual = actual - predicted
        residual = y - predicted

        result = pd.Series(residual, index=common_idx)

        # Preserve original index and fill missing with original values
        full_result = series.astype(float).copy()
        full_result.loc[common_idx] = result

        return full_result

    @staticmethod
    def process_factor(
        series: pd.Series,
        method: str = 'rank',
        ascending: bool = True,
        winsorize_lower: float = 0.05,
        winsorize_upper: float = 0.95
    ) -> pd.Series:
        """Full processing pipeline for factor data.

        Args:
            series: Input factor series.
            method: Processing method - 'rank' or 'zscore'.
            ascending: If True, lower values get lower scores (for rank).
            winsorize_lower: Lower percentile for winsorization.
            winsorize_upper: Upper percentile for winsorization.

        Returns:
            Processed factor series.

        Raises:
            ValueError: If method is not 'rank' or 'zscore'.
        """
        # Apply winsorization first
        winsorized = FactorProcessor.winsorize(
            series, lower=winsorize_lower, upper=winsorize_upper
        )

        # Apply specified method
        if method == 'rank':
            return FactorProcessor.rank_percentile(winsorized, ascending=ascending)
        elif method == 'zscore':
            return FactorProcessor.z_score(winsorized)
        else:
            raise ValueError(f"Invalid method '{method}'. Use 'rank' or 'zscore'.")
