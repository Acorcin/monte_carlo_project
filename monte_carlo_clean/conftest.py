"""
Pytest configuration file for Monte Carlo Trading Application tests.

This file contains shared fixtures and test configuration.
"""

import pytest
import pandas as pd
import numpy as np
import tkinter as tk
from unittest.mock import MagicMock, patch
import tempfile
import os
from datetime import datetime, timedelta


@pytest.fixture
def sample_stock_data():
    """Generate sample stock data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)  # For reproducible tests
    
    # Generate realistic stock price data
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
    data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
    
    return data


@pytest.fixture
def sample_trading_signals():
    """Generate sample trading signals for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Generate signals: 1=buy, -1=sell, 0=hold
    signals = np.random.choice([1, -1, 0], size=len(dates), p=[0.1, 0.1, 0.8])
    return pd.Series(signals, index=dates)


@pytest.fixture
def mock_algorithm():
    """Create a mock trading algorithm for testing."""
    algorithm = MagicMock()
    algorithm.name = "TestAlgorithm"
    algorithm.get_algorithm_type.return_value = "momentum"
    
    # Mock the generate_signals method
    def mock_generate_signals(data):
        np.random.seed(42)
        signals = np.random.choice([1, -1, 0], size=len(data), p=[0.15, 0.15, 0.7])
        return pd.Series(signals, index=data.index)
    
    algorithm.generate_signals = mock_generate_signals
    return algorithm


@pytest.fixture
def mock_backtest_results():
    """Create mock backtest results for testing."""
    return {
        'algorithm_name': 'TestAlgorithm',
        'initial_capital': 10000.0,
        'final_capital': 11500.0,
        'total_return': 0.15,
        'metrics': {
            'total_trades': 50,
            'win_rate': 0.6,
            'avg_return': 0.003,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.08,
            'profit_factor': 1.5,
            'drawdown_peak_value': 12000.0,
            'drawdown_trough_value': 11040.0,
            'drawdown_duration_days': 15,
            'avg_drawdown': 0.03,
            'time_underwater_pct': 25.0
        },
        'trades': [
            {
                'entry_date': datetime(2024, 1, 15),
                'exit_date': datetime(2024, 1, 20),
                'direction': 'long',
                'entry_price': 100.0,
                'exit_price': 102.0,
                'return': 0.02,
                'duration': timedelta(days=5)
            },
            {
                'entry_date': datetime(2024, 2, 10),
                'exit_date': datetime(2024, 2, 15),
                'direction': 'short',
                'entry_price': 105.0,
                'exit_price': 103.0,
                'return': 0.019,
                'duration': timedelta(days=5)
            }
        ],
        'returns': [0.02, -0.01, 0.015, -0.005, 0.01, 0.008, -0.003, 0.012]
    }


@pytest.fixture
def mock_gui_root():
    """Create a mock Tkinter root for GUI testing."""
    # Don't create actual Tkinter objects in tests
    root = MagicMock()
    root.winfo_screenwidth.return_value = 1920
    root.winfo_screenheight.return_value = 1080
    return root


@pytest.fixture
def temp_algorithm_file():
    """Create a temporary algorithm file for testing."""
    algorithm_code = '''
from algorithms.base_algorithm import TradingAlgorithm
import pandas as pd
import numpy as np

class TestCustomAlgorithm(TradingAlgorithm):
    def __init__(self):
        super().__init__("Test Custom Algorithm")
        self.short_window = 10
        self.long_window = 30
    
    def generate_signals(self, data):
        if len(data) < self.long_window:
            return pd.Series(0, index=data.index)
        
        short_ma = data['Close'].rolling(self.short_window).mean()
        long_ma = data['Close'].rolling(self.long_window).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[short_ma > long_ma] = 1
        signals[short_ma < long_ma] = -1
        
        return signals
    
    def get_algorithm_type(self):
        return "trend_following"
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(algorithm_code)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)


@pytest.fixture
def mock_yfinance_data():
    """Mock yfinance data for testing data fetching."""
    def mock_download(*args, **kwargs):
        # Return sample data that looks like yfinance output
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        initial_price = 150.0
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'Open': prices,
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        return data
    
    return mock_download


@pytest.fixture
def mock_monte_carlo_results():
    """Generate mock Monte Carlo simulation results."""
    np.random.seed(42)
    num_simulations = 1000
    num_periods = 252  # One year of trading days
    
    # Generate multiple simulation paths
    results = pd.DataFrame()
    for i in range(num_simulations):
        returns = np.random.normal(0.0008, 0.02, num_periods)  # Daily returns
        cum_returns = (1 + returns).cumprod()
        portfolio_values = 10000 * cum_returns  # Starting with $10,000
        results[f'sim_{i}'] = portfolio_values
    
    return results


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data files."""
    temp_dir = tempfile.mkdtemp(prefix="monte_carlo_tests_")
    yield temp_dir
    
    # Cleanup after all tests
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "gui: mark test as a GUI test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Skip GUI tests if no display available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GUI tests when appropriate."""
    skip_gui = pytest.mark.skip(reason="No display available for GUI tests")
    
    for item in items:
        if "gui" in item.keywords:
            # Check if we're in a headless environment
            if os.environ.get('CI') or os.environ.get('HEADLESS'):
                item.add_marker(skip_gui)
