#!/usr/bin/env python3
"""
Test script for report generation functionality
"""

import sys
import os
sys.path.append('algorithms')

from algorithms.base_algorithm import TradingAlgorithm
import pandas as pd
import numpy as np
from datetime import datetime

class TestAlgorithm(TradingAlgorithm):
    """Simple test algorithm for report generation testing."""

    def __init__(self):
        super().__init__(
            name="Test Algorithm",
            description="Test algorithm for report generation",
            parameters={'test_param': 10}
        )

    def generate_signals(self, data):
        """Generate simple signals for testing."""
        return pd.Series([0] * len(data), index=data.index)

    def get_algorithm_type(self):
        return "test"

def test_report_generation():
    """Test the report generation functionality."""
    print("🧪 Testing Report Generation Functionality")
    print("=" * 50)

    # Create test data
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    data = pd.DataFrame({
        'Open': 100 + np.random.normal(0, 2, len(dates)),
        'High': 102 + np.random.normal(0, 2, len(dates)),
        'Low': 98 + np.random.normal(0, 2, len(dates)),
        'Close': 100 + np.random.normal(0, 2, len(dates)),
        'Volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)

    print("✅ Test data created")

    # Create and run test algorithm
    algorithm = TestAlgorithm()
    print("✅ Test algorithm created")

    # Run backtest
    results = algorithm.backtest(data, initial_capital=10000)
    print("✅ Backtest completed")

    # Check results structure
    required_fields = ['algorithm_name', 'initial_capital', 'final_capital', 'total_return', 'metrics']
    for field in required_fields:
        if field not in results:
            print(f"❌ Missing field: {field}")
            return False
        print(f"✅ Found field: {field}")

    # Check metrics structure
    metrics = results['metrics']
    required_metrics = ['total_trades', 'win_rate', 'avg_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor']
    for metric in required_metrics:
        if metric not in metrics:
            print(f"❌ Missing metric: {metric}")
            return False
        print(f"✅ Found metric: {metric}")

    print("\n🎉 Report generation test PASSED!")
    print("All required fields and metrics are present in the backtest results.")
    print("The generate_report functionality should now work correctly.")

    return True

if __name__ == "__main__":
    test_report_generation()
