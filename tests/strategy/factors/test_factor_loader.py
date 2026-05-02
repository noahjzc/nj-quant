"""Tests for FactorLoader class."""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from strategy.factors.factor_loader import FactorLoader


class TestFactorLoader:
    """Test cases for FactorLoader."""

    def test_load_stock_factors(self):
        """Test loading factors for specific stocks."""
        # Mock DataProvider
        mock_provider = MagicMock()

        # Create mock stock data
        mock_df = pd.DataFrame({
            'PB': [1.5, 1.6],
            'PE_TTM': [10.0, 11.0],
            'RSI_1': [60.0, 65.0],
            'MA_5': [10.0, 10.2],
        }, index=pd.to_datetime(['2024-01-01', '2024-01-02']))

        mock_provider.get_stock_data.return_value = mock_df

        loader = FactorLoader(data_provider=mock_provider)
        result = loader.load_stock_factors(
            stock_codes=['sh600001'],
            date=pd.Timestamp('2024-01-02'),
            factors=['PB', 'PE_TTM']
        )

        assert len(result) == 1
        assert 'PB' in result.columns
        assert 'PE_TTM' in result.columns
        # Latest values
        assert result.loc['sh600001', 'PB'] == 1.6
        assert result.loc['sh600001', 'PE_TTM'] == 11.0

    def test_load_all_stock_factors(self):
        """Test loading factors for all stocks."""
        mock_provider = MagicMock()
        mock_provider.get_all_stock_codes.return_value = ['sh600001', 'sh600002']
        mock_provider.get_stock_data.return_value = pd.DataFrame({
            'PB': [1.5],
            'PE_TTM': [10.0],
        }, index=pd.to_datetime(['2024-01-02']))

        loader = FactorLoader(data_provider=mock_provider)
        result = loader.load_all_stock_factors(
            date=pd.Timestamp('2024-01-02'),
            factors=['PB', 'PE_TTM']
        )

        assert len(result) == 2
        mock_provider.get_all_stock_codes.assert_called_once()

    def test_missing_factor_filled_with_median(self):
        """Test that missing factor values are filled with median."""
        mock_provider = MagicMock()

        # Stock with missing PB
        mock_df = pd.DataFrame({
            'PB': [None, 1.6],
            'PE_TTM': [10.0, 11.0],
        }, index=pd.to_datetime(['2024-01-01', '2024-01-02']))

        mock_provider.get_stock_data.return_value = mock_df

        loader = FactorLoader(data_provider=mock_provider)
        result = loader.load_stock_factors(
            stock_codes=['sh600001'],
            date=pd.Timestamp('2024-01-02'),
            factors=['PB', 'PE_TTM']
        )

        # Should fill with median (1.6 since first row is NaN)
        assert result.loc['sh600001', 'PB'] == 1.6

    def test_factor_column_mapping(self):
        """Test that factor column names are correctly mapped."""
        loader = FactorLoader()

        # Verify known mappings
        assert loader.FACTOR_COLUMNS['PB'] == 'PB'
        assert loader.FACTOR_COLUMNS['RSI_1'] == 'RSI_1'
        assert loader.FACTOR_COLUMNS['ROE'] == 'ROE_TTM'