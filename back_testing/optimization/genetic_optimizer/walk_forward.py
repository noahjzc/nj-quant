"""Walk-Forward Analysis for robust optimization"""
import pandas as pd
from typing import List, Dict, Tuple


class WalkForwardAnalyzer:
    """
    Walk-Forward window generator and weight aggregator.

    Window structure:
    - Train: 3 years (GA optimization)
    - Validation: 1 year (early stopping)
    - Test: 1 year (final evaluation)

    Rolls forward every 3 months.
    """

    def __init__(self,
                 train_window_years: int = 3,
                 val_window_years: int = 1,
                 test_window_years: int = 1,
                 step_months: int = 3):
        """
        Args:
            train_window_years: Training window length in years
            val_window_years: Validation window length in years
            test_window_years: Test window length in years
            step_months: Rolling step in months
        """
        self.train_window_years = train_window_years
        self.val_window_years = val_window_years
        self.test_window_years = test_window_years
        self.step_months = step_months

    def get_windows(self, start_date: pd.Timestamp,
                   end_date: pd.Timestamp) -> List[Dict]:
        """
        Generate walk-forward windows.

        Args:
            start_date: Data start date
            end_date: Data end date

        Returns:
            List of windows: [{'train': (start, end), 'val': ..., 'test': ...}, ...]
        """
        windows = []
        current = pd.Timestamp(start_date)

        train_months = self.train_window_years * 12
        val_months = self.val_window_years * 12
        test_months = self.test_window_years * 12

        while True:
            train_end = current + pd.DateOffset(months=train_months)
            val_end = train_end + pd.DateOffset(months=val_months)
            test_end = val_end + pd.DateOffset(months=test_months)

            if test_end > end_date:
                break

            windows.append({
                'train': (current, train_end),
                'val': (train_end, val_end),
                'test': (val_end, test_end)
            })

            current = current + pd.DateOffset(months=self.step_months)

        return windows

    def aggregate_weights(self, weights_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate weights from multiple windows.

        Args:
            weights_list: List of optimal weights from each window

        Returns:
            Aggregated weights (simple average)
        """
        if not weights_list:
            return {}

        n = len(weights_list)
        aggregated = {}
        all_keys = set()
        for w in weights_list:
            all_keys.update(w.keys())

        for key in all_keys:
            values = [w.get(key, 0) for w in weights_list]
            aggregated[key] = sum(values) / n

        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v / total for k, v in aggregated.items()}

        return aggregated
