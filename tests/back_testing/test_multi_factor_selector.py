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

        # PB direction=-1 (smaller is better): ascending=False → smallest→1, largest→0
        # ROE direction=1 (larger is better): ascending=True → smallest→0, largest→1
        # s1: PB=1 (best PB → 1.0), ROE=10 (2nd lowest → 0.25) → total = 0.5*1.0+0.5*0.25 = 0.625
        # s4: PB=4 (2nd highest → 0.25), ROE=25 (best ROE → 1.0) → total = 0.5*0.25+0.5*1.0 = 0.625
        # With equal weights, s1 (best PB) and s4 (best ROE) are tied
        assert abs(scores['s1'] - scores['s4']) < 0.01, f"Expected s1 ≈ s4, got s1={scores['s1']}, s4={scores['s4']}"
        # s2 (good ROE, moderate PB) should be ranked highest, s5 (worst of both) lowest
        assert scores['s2'] > scores['s1'], f"Expected s2 score > s1 score"

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
        # Use n=10 so the 70% main-board limit (7) exceeds the 5 available stocks
        result = selector.select_top_stocks(data, n=10)

        assert len(result) == 5
        # sh600002 should be first (best combined: moderate PB + best ROE)
        assert result[0] == 'sh600002'

    def test_select_top_stocks_with_excluded(self):
        """Test top stock selection with excluded stocks."""
        data = pd.DataFrame({
            'PB': [1.0, 2.0, 3.0, 4.0, 5.0],
            'ROE': [10.0, 20.0, 15.0, 25.0, 5.0],
        }, index=['sh600001', 'sh600002', 'sh600003', 'sh600004', 'sh600005'])

        weights = {'PB': 0.5, 'ROE': 0.5}
        directions = {'PB': -1, 'ROE': 1}

        selector = MultiFactorSelector(weights, directions)
        # Use n=10 so main-board 70% limit (7) exceeds the 4 remaining stocks
        result = selector.select_top_stocks(data, n=10, excluded=['sh600004'])

        assert len(result) == 4
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
        # PB direction=-1 (smaller is better): ascending=False → smallest→1, largest→0
        # s1 PB=1 (smallest) → rank 5 → percentile 1.0 → contribution 1.0*0.5 = 0.5
        # s4 PB=4 (2nd largest) → rank 2 → percentile 0.25 → contribution 0.25*0.5 = 0.125
        assert contributions.loc['s1', 'PB'] == pytest.approx(0.5, abs=0.01)
        assert contributions.loc['s4', 'PB'] == pytest.approx(0.125, abs=0.01)

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

        # Equal weights: PB and ROE each contribute 50% after normalization
        # s1: best PB (→1.0) + middle ROE (→0.5) = 0.75
        # s2: middle PB (→0.5) + best ROE (→1.0) = 0.75
        # s3: worst PB (→0.0) + worst ROE (→0.0) = 0.0
        # s1 and s2 are tied (best PB vs best ROE cancel out), s3 is worst
        assert abs(scores['s1'] - scores['s2']) < 0.01, f"Expected s1 ≈ s2, got s1={scores['s1']}, s2={scores['s2']}"
        assert scores['s3'] < scores['s1'], f"Expected s3 < s1, got s3={scores['s3']}, s1={scores['s1']}"

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
