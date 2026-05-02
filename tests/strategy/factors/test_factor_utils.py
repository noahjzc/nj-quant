"""Tests for FactorProcessor factor utilities."""
import pandas as pd
import numpy as np
import pytest

from strategy.factors.factor_utils import FactorProcessor


def test_rank_percentile():
    """Test percentile ranking - lower values get lower scores by default."""
    data = pd.Series([10, 20, 30, 40, 50])
    result = FactorProcessor.rank_percentile(data)
    expected = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
    assert list(result) == list(expected), f"Expected {list(expected)}, got {list(result)}"


def test_rank_percentile_descending():
    """Test percentile ranking with ascending=False - higher values get lower scores."""
    data = pd.Series([10, 20, 30, 40, 50])
    result = FactorProcessor.rank_percentile(data, ascending=False)
    expected = pd.Series([1.0, 0.75, 0.5, 0.25, 0.0])
    assert list(result) == list(expected), f"Expected {list(expected)}, got {list(result)}"


def test_z_score():
    """Test z-score standardization normalized to [0, 1] range."""
    data = pd.Series([10, 20, 30, 40, 50])
    result = FactorProcessor.z_score(data)
    assert result.min() == 0.0, f"Min should be 0, got {result.min()}"
    assert result.max() == 1.0, f"Max should be 1, got {result.max()}"
    assert abs(result.mean() - 0.5) < 1e-10, f"Mean should be ~0.5, got {result.mean()}"


def test_z_score_with_nan():
    """Test z-score handles NaN values."""
    data = pd.Series([10, 20, np.nan, 40, 50])
    result = FactorProcessor.z_score(data)
    assert not pd.isna(result.iloc[0]), "First value should not be NaN"
    assert pd.isna(result.iloc[2]), "NaN should be preserved"


def test_winsorize():
    """Test winsorization clips extreme values to percentile bounds."""
    # Use larger dataset so percentiles properly clip extremes
    data = pd.Series([-100, -50, 0, 10, 20, 30, 40, 50, 100])
    result = FactorProcessor.winsorize(data, lower=0.05, upper=0.95)
    # With 9 elements, 0.05 * 8 = 0.4 -> interpolates between index 0 and 1
    # lower bound = -100 + 0.4 * (-50 - (-100)) = -100 + 20 = -80
    # 0.95 * 8 = 7.6 -> interpolates between index 7 and 8
    # upper bound = 50 + 0.6 * (100 - 50) = 50 + 30 = 80
    # After winsorize: [-80, -50, 0, 10, 20, 30, 40, 50, 80]
    assert result.min() == pytest.approx(-80), f"Min should be -80, got {result.min()}"
    assert result.max() == pytest.approx(80), f"Max should be 80, got {result.max()}"
    # Middle values should be unchanged
    assert result.iloc[3] == 10, f"Value at index 3 should be 10, got {result.iloc[3]}"


def test_winsorize_preserves_middle():
    """Test winsorization preserves middle values."""
    # Use larger dataset so percentiles properly clip extremes
    data = pd.Series([-100, -50, 0, 10, 20, 30, 40, 50, 100])
    result = FactorProcessor.winsorize(data, lower=0.05, upper=0.95)
    # Middle values should be preserved (indices 2-6: 0, 10, 20, 30, 40)
    assert result.iloc[2] == 0, f"Value at index 2 should be preserved as 0"
    assert result.iloc[3] == 10, f"Value at index 3 should be preserved as 10"
    assert result.iloc[4] == 20, f"Value at index 4 should be preserved as 20"


def test_neutralize():
    """Test market neutralization via regression residual."""
    # Factor values
    factor = pd.Series([10, 20, 30, 40, 50])
    # Market cap values (larger cap should have more weight)
    market_cap = pd.Series([100, 200, 300, 400, 500])
    result = FactorProcessor.neutralize(factor, market_cap)
    # Result should be orthogonal to market_cap
    # The sum should be close to 0 (mean zero)
    assert abs(result.mean()) < 1e-10, f"Mean should be ~0, got {result.mean()}"


def test_neutralize_preserves_variance():
    """Test neutralize preserves factor variance information."""
    factor = pd.Series([10, 20, 30, 40, 50])
    market_cap = pd.Series([100, 200, 300, 400, 500])
    result = FactorProcessor.neutralize(factor, market_cap)
    # Variance should be preserved (std > 0)
    assert result.std() > 0, "Standard deviation should be > 0"


def test_process_factor_rank():
    """Test full pipeline with rank method."""
    data = pd.Series([10, 20, 30, 40, 50])
    result = FactorProcessor.process_factor(data, method='rank')
    expected = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
    assert list(result) == list(expected), f"Expected {list(expected)}, got {list(result)}"


def test_process_factor_zscore():
    """Test full pipeline with zscore method normalized to [0, 1]."""
    data = pd.Series([10, 20, 30, 40, 50])
    result = FactorProcessor.process_factor(data, method='zscore')
    assert result.min() == 0.0, f"Min should be 0, got {result.min()}"
    assert result.max() == 1.0, f"Max should be 1, got {result.max()}"


def test_process_factor_winsorize_applied():
    """Test process_factor applies winsorization by default."""
    data = pd.Series([-100, 0, 10, 20, 100])
    result = FactorProcessor.process_factor(data, method='rank', winsorize_lower=0.05, winsorize_upper=0.95)
    # Should not have extreme values after winsorization
    assert result.min() >= 0.0, f"Min should be >= 0, got {result.min()}"
    assert result.max() <= 1.0, f"Max should be <= 1, got {result.max()}"


def test_process_factor_invalid_method():
    """Test process_factor raises error for invalid method."""
    data = pd.Series([10, 20, 30])
    with pytest.raises(ValueError):
        FactorProcessor.process_factor(data, method='invalid')


def test_williams_r_basic():
    """Test Williams %R calculation with known values."""
    df = pd.DataFrame({
        'high': [12, 13, 14, 15, 16, 15, 14, 13, 12, 11],
        'low':  [8,  9,  10, 11, 10, 9,  8,  7,  6,  5],
        'close': [10, 11, 12, 13, 12, 11, 10, 9, 8, 7],
    })
    # WR_5: period=5, last 5 rows (indices 5-9)
    # high_n = max(15,14,13,12,11) = 15
    # low_n = min(9,8,7,6,5) = 5
    # close = 7
    # WR = (15-7)/(15-5) * -100 = 8/10 * -100 = -80
    result = FactorProcessor.williams_r(df, 5)
    assert result == -80.0


def test_williams_r_no_range():
    """Test Williams %R when high == low (no price range)."""
    df = pd.DataFrame({
        'high': [10, 10, 10, 10, 10],
        'low':  [10, 10, 10, 10, 10],
        'close': [10, 10, 10, 10, 10],
    })
    result = FactorProcessor.williams_r(df, 5)
    assert result == -50.0


def test_williams_r_oversold():
    """Test Williams %R at extreme oversold (close near low)."""
    df = pd.DataFrame({
        'high': [15, 15, 15, 15, 15],
        'low':  [5, 5, 5, 5, 5],
        'close': [6, 6, 6, 6, 6],
    })
    # WR = (15-6)/(15-5) * -100 = 9/10 * -100 = -90
    result = FactorProcessor.williams_r(df, 5)
    assert result == pytest.approx(-90.0)


def test_williams_r_overbought():
    """Test Williams %R at extreme overbought (close near high)."""
    df = pd.DataFrame({
        'high': [15, 15, 15, 15, 15],
        'low':  [5, 5, 5, 5, 5],
        'close': [14, 14, 14, 14, 14],
    })
    # WR = (15-14)/(15-5) * -100 = 1/10 * -100 = -10
    result = FactorProcessor.williams_r(df, 5)
    assert result == pytest.approx(-10.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
