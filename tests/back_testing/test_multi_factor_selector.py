"""Tests for MultiFactorSelector class."""
import pandas as pd
import numpy as np
import pytest
from back_testing.selectors.multi_factor_selector import MultiFactorSelector


class TestMultiFactorSelector:
    """Test cases for MultiFactorSelector."""

    def test_calculate_factor_scores(self):
        """Test factor score calculation with rank method."""
        data = pd.DataFrame({
            'PB': [1.0, 2.0, 3.0, 4.0, 5.0],
            'ROE': [10.0, 20.0, 15.0, 25.0, 5.0],
        }, index=['s1', 's2', 's3', 's4', 's5'])

        weights = {'PB': 0.5, 'ROE': 0.5}
        directions = {'PB': -1, 'ROE': 1}

        selector = MultiFactorSelector(weights, directions)
        scores = selector.calculate_factor_scores(data)

        # s1: PB=1 (lowest), ROE=10 (second lowest)
        # s4: PB=4 (second highest), ROE=25 (highest)
        # s1 composite: 0.5*PB_low + 0.5*ROE_low = 0.5*0.0 + 0.5*0.25 = 0.125
        # s4 composite: 0.5*PB_mid + 0.5*ROE_high = 0.5*0.75 + 0.5*1.0 = 0.875
        assert scores['s1'] < scores['s4'], f"Expected s1 < s4, got s1={scores['s1']}, s4={scores['s4']}"
        # s2 should be in the middle
        assert scores['s2'] > scores['s1'], f"Expected s2 score > s1 score"
        assert scores['s2'] < scores['s4'], f"Expected s2 score < s4 score"

    def test_calculate_factor_scores_zscore_method(self):
        """Test factor score calculation with zscore method."""
        data = pd.DataFrame({
            'PB': [1.0, 2.0, 3.0, 4.0, 5.0],
            'ROE': [10.0, 20.0, 15.0, 25.0, 5.0],
        }, index=['s1', 's2', 's3', 's4', 's5'])

        weights = {'PB': 0.5, 'ROE': 0.5}
        directions = {'PB': -1, 'ROE': 1}

        selector = MultiFactorSelector(weights, directions, method='zscore')
        scores = selector.calculate_factor_scores(data)

        # With equal weights on opposite directions, s1 and s4 should have equal scores
        assert scores['s1'] == pytest.approx(scores['s4'], abs=0.01)

    def test_select_top_stocks(self):
        """Test top stock selection."""
        data = pd.DataFrame({
            'PB': [1.0, 2.0, 3.0, 4.0, 5.0],
            'ROE': [10.0, 20.0, 15.0, 25.0, 5.0],
        }, index=['sh600001', 'sh600002', 'sh600003', 'sh600004', 'sh600005'])

        weights = {'PB': 0.5, 'ROE': 0.5}
        directions = {'PB': -1, 'ROE': 1}

        selector = MultiFactorSelector(weights, directions)
        result = selector.select_top_stocks(data, n=3)

        assert len(result) == 3
        # sh600004 should be first (best combined score: best PB and best ROE)
        assert result[0] == 'sh600004'

    def test_select_top_stocks_with_excluded(self):
        """Test top stock selection with excluded stocks."""
        data = pd.DataFrame({
            'PB': [1.0, 2.0, 3.0, 4.0, 5.0],
            'ROE': [10.0, 20.0, 15.0, 25.0, 5.0],
        }, index=['sh600001', 'sh600002', 'sh600003', 'sh600004', 'sh600005'])

        weights = {'PB': 0.5, 'ROE': 0.5}
        directions = {'PB': -1, 'ROE': 1}

        selector = MultiFactorSelector(weights, directions)
        result = selector.select_top_stocks(data, n=3, excluded=['sh600004'])

        assert len(result) == 3
        # sh600004 should not be in result since it was excluded
        assert 'sh600004' not in result
        # sh600002 should be first now (best among remaining)
        assert result[0] == 'sh600002'

    def test_get_factor_contribution(self):
        """Test factor contribution calculation."""
        data = pd.DataFrame({
            'PB': [1.0, 2.0, 3.0, 4.0, 5.0],
            'ROE': [10.0, 20.0, 15.0, 25.0, 5.0],
        }, index=['s1', 's2', 's3', 's4', 's5'])

        weights = {'PB': 0.5, 'ROE': 0.5}
        directions = {'PB': -1, 'ROE': 1}

        selector = MultiFactorSelector(weights, directions)
        contributions = selector.get_factor_contribution(data)

        assert 'PB' in contributions.columns
        assert 'ROE' in contributions.columns
        assert len(contributions) == 5
        # For PB direction=-1 (ascending=True), lowest PB (s1) gets percentile 0.0
        # s4 PB=4 is 4th rank, percentile = (4-1)/4 = 0.75
        # PB contribution = percentile * weight
        # s1 = 0.0 * 0.5 = 0.0, s4 = 0.75 * 0.5 = 0.375
        assert contributions.loc['s1', 'PB'] == pytest.approx(0.0, abs=0.01)
        assert contributions.loc['s4', 'PB'] == pytest.approx(0.375, abs=0.01)

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        weights = {'PB': 0.5, 'ROE': 0.5}
        directions = {'PB': -1, 'ROE': 1}

        with pytest.raises(ValueError):
            MultiFactorSelector(weights, directions, method='invalid')

    def test_weights_normalized_implicitly(self):
        """Test that weights are used correctly regardless of sum."""
        data = pd.DataFrame({
            'PB': [1.0, 2.0, 3.0],
            'ROE': [10.0, 20.0, 5.0],
        }, index=['s1', 's2', 's3'])

        # Non-normalized weights (sum to 1.5)
        weights = {'PB': 0.75, 'ROE': 0.75}
        directions = {'PB': -1, 'ROE': 1}

        selector = MultiFactorSelector(weights, directions)
        scores = selector.calculate_factor_scores(data)

        # s1 has lowest PB and lowest ROE, so should have lowest score
        # s2 has middle PB and highest ROE, so should have middle score
        # s3 has highest PB and lowest ROE, so should have highest PB score but lowest ROE
        assert scores['s1'] < scores['s2']

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        data = pd.DataFrame({
            'PB': [],
            'ROE': [],
        })

        weights = {'PB': 0.5, 'ROE': 0.5}
        directions = {'PB': -1, 'ROE': 1}

        selector = MultiFactorSelector(weights, directions)
        scores = selector.calculate_factor_scores(data)

        assert len(scores) == 0

    def test_single_stock(self):
        """Test handling of single stock."""
        data = pd.DataFrame({
            'PB': [1.0],
            'ROE': [10.0],
        }, index=['s1'])

        weights = {'PB': 0.5, 'ROE': 0.5}
        directions = {'PB': -1, 'ROE': 1}

        selector = MultiFactorSelector(weights, directions)
        scores = selector.calculate_factor_scores(data)

        assert len(scores) == 1
        # Single stock should get a neutral score of 0.5
        assert scores['s1'] == pytest.approx(0.5, abs=0.01)

    def test_select_top_stocks_more_than_available(self):
        """Test selecting more stocks than available."""
        data = pd.DataFrame({
            'PB': [1.0, 2.0],
            'ROE': [10.0, 20.0],
        }, index=['sh600001', 'sh600002'])

        weights = {'PB': 0.5, 'ROE': 0.5}
        directions = {'PB': -1, 'ROE': 1}

        selector = MultiFactorSelector(weights, directions)
        result = selector.select_top_stocks(data, n=10)

        # Should return all available stocks
        assert len(result) == 2
