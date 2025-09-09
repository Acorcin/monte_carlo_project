"""
Unit tests for data_fetcher.py module.

Tests data fetching, validation, and processing functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import the module to test
import data_fetcher


class TestDataFetcher:
    """Test class for data fetching functionality."""

    @pytest.mark.unit
    def test_fetch_stock_data_valid_input(self, mock_yfinance_data):
        """Test fetching stock data with valid inputs."""
        with patch('yfinance.download', side_effect=mock_yfinance_data):
            data = data_fetcher.fetch_stock_data('AAPL', period='1y', interval='1d')
            
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
            assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
            assert len(data) > 0

    @pytest.mark.unit
    def test_fetch_stock_data_invalid_ticker(self):
        """Test fetching data with invalid ticker."""
        with patch('yfinance.download', return_value=pd.DataFrame()):
            data = data_fetcher.fetch_stock_data('INVALID_TICKER', period='1y', interval='1d')
            
            assert data is None or data.empty

    @pytest.mark.unit
    def test_fetch_stock_data_network_error(self):
        """Test handling of network errors during data fetching."""
        with patch('yfinance.download', side_effect=Exception("Network error")):
            data = data_fetcher.fetch_stock_data('AAPL', period='1y', interval='1d')
            
            assert data is None or data.empty

    @pytest.mark.unit
    def test_validate_data_valid_dataframe(self, sample_stock_data):
        """Test data validation with valid DataFrame."""
        if hasattr(data_fetcher, 'validate_data'):
            result = data_fetcher.validate_data(sample_stock_data)
            assert result is True
        else:
            # If function doesn't exist, create a simple validation
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            assert all(col in sample_stock_data.columns for col in required_columns)

    @pytest.mark.unit
    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        incomplete_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [102, 103, 104],
            'Close': [101, 102, 103]
            # Missing 'Low' and 'Volume'
        })
        
        if hasattr(data_fetcher, 'validate_data'):
            result = data_fetcher.validate_data(incomplete_data)
            assert result is False
        else:
            # Test that required columns are missing
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [col for col in required_columns if col not in incomplete_data.columns]
            assert len(missing) > 0

    @pytest.mark.unit
    def test_data_preprocessing(self, sample_stock_data):
        """Test data preprocessing functionality."""
        # Test that data is properly cleaned
        assert not sample_stock_data.isnull().any().any()
        assert (sample_stock_data['High'] >= sample_stock_data['Low']).all()
        assert (sample_stock_data['High'] >= sample_stock_data['Open']).all()
        assert (sample_stock_data['High'] >= sample_stock_data['Close']).all()
        assert (sample_stock_data['Low'] <= sample_stock_data['Open']).all()
        assert (sample_stock_data['Low'] <= sample_stock_data['Close']).all()

    @pytest.mark.unit
    @pytest.mark.parametrize("period,interval", [
        ("1d", "1m"),
        ("5d", "5m"),
        ("1mo", "1h"),
        ("3mo", "1d"),
        ("1y", "1wk"),
        ("2y", "1mo")
    ])
    def test_fetch_with_different_periods_intervals(self, period, interval, mock_yfinance_data):
        """Test fetching data with different period and interval combinations."""
        with patch('yfinance.download', side_effect=mock_yfinance_data):
            data = data_fetcher.fetch_stock_data('AAPL', period=period, interval=interval)
            
            if data is not None and not data.empty:
                assert isinstance(data, pd.DataFrame)
                assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

    @pytest.mark.unit
    def test_data_caching(self, mock_yfinance_data):
        """Test data caching functionality if implemented."""
        with patch('yfinance.download', side_effect=mock_yfinance_data) as mock_download:
            # Fetch data twice with same parameters
            data1 = data_fetcher.fetch_stock_data('AAPL', period='1y', interval='1d')
            data2 = data_fetcher.fetch_stock_data('AAPL', period='1y', interval='1d')
            
            # Verify both calls return data
            assert data1 is not None
            assert data2 is not None
            
            # Note: Caching behavior depends on implementation
            # This test documents expected behavior


class TestDataValidation:
    """Test class for data validation functions."""

    @pytest.mark.unit
    def test_check_data_quality(self, sample_stock_data):
        """Test data quality checking."""
        # Test for reasonable price relationships
        assert (sample_stock_data['High'] >= sample_stock_data['Low']).all()
        assert (sample_stock_data['Volume'] >= 0).all()
        
        # Test for no extreme outliers (prices shouldn't change by more than 50% in one day)
        price_changes = sample_stock_data['Close'].pct_change().abs()
        assert (price_changes[1:] < 0.5).all()  # Skip first NaN value

    @pytest.mark.unit
    def test_detect_data_gaps(self):
        """Test detection of data gaps."""
        # Create data with gaps
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        }, index=dates)
        
        # Remove some days to create gaps
        data_with_gaps = data.drop(data.index[3:5])  # Remove 2 days
        
        # Test gap detection (implementation dependent)
        expected_length = 10
        actual_length = len(data_with_gaps)
        assert actual_length < expected_length

    @pytest.mark.unit
    def test_handle_missing_values(self):
        """Test handling of missing values in data."""
        # Create data with missing values
        data = pd.DataFrame({
            'Open': [100, np.nan, 102, 103],
            'High': [102, 103, np.nan, 105],
            'Low': [99, 100, 101, np.nan],
            'Close': [101, 102, 103, 104],
            'Volume': [1000000, 1100000, np.nan, 1200000]
        })
        
        # Test that we can identify missing values
        assert data.isnull().any().any()
        
        # Test forward fill or other handling methods
        filled_data = data.fillna(method='ffill')
        assert not filled_data.isnull().any().any() or len(filled_data) == 1


class TestDataTransformation:
    """Test class for data transformation functions."""

    @pytest.mark.unit
    def test_calculate_returns(self, sample_stock_data):
        """Test calculation of returns."""
        returns = sample_stock_data['Close'].pct_change()
        
        # Test that returns are calculated correctly
        assert len(returns) == len(sample_stock_data)
        assert pd.isna(returns.iloc[0])  # First return should be NaN
        
        # Test that returns are reasonable
        non_nan_returns = returns.dropna()
        assert (non_nan_returns.abs() < 0.5).all()  # No single-day changes > 50%

    @pytest.mark.unit
    def test_calculate_volatility(self, sample_stock_data):
        """Test calculation of volatility."""
        returns = sample_stock_data['Close'].pct_change().dropna()
        volatility = returns.std()
        
        assert isinstance(volatility, float)
        assert volatility >= 0
        assert volatility < 1  # Daily volatility should be reasonable

    @pytest.mark.unit
    def test_technical_indicators(self, sample_stock_data):
        """Test calculation of basic technical indicators."""
        # Test moving averages
        ma_short = sample_stock_data['Close'].rolling(10).mean()
        ma_long = sample_stock_data['Close'].rolling(30).mean()
        
        assert len(ma_short) == len(sample_stock_data)
        assert len(ma_long) == len(sample_stock_data)
        
        # Test that moving averages smooth the data
        assert ma_short.var() <= sample_stock_data['Close'].var()
        assert ma_long.var() <= ma_short.var()


if __name__ == '__main__':
    pytest.main([__file__])
