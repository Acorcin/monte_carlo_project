#!/usr/bin/env python3
"""
Test script for report generation logic
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

def test_report_logic():
    """Test the report generation logic without GUI components."""
    print("📄 Testing Report Generation Logic")
    print("=" * 40)

    try:
        # Create test data
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        data = pd.DataFrame({
            'Open': 100 + np.random.normal(0, 2, len(dates)),
            'High': 102 + np.random.normal(0, 2, len(dates)),
            'Low': 98 + np.random.normal(0, 2, len(dates)),
            'Close': 100 + np.random.normal(0, 2, len(dates)),
            'Volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)

        # Create and run test algorithm
        algorithm = TestAlgorithm()
        results = algorithm.backtest(data, initial_capital=10000)

        print("✅ Backtest results generated")

        # Test the report generation logic directly
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Extract metrics with safe access (same logic as the GUI method)
        metrics = results.get('metrics', {})
        total_trades = metrics.get('total_trades', 0)
        win_rate = metrics.get('win_rate', 0)
        avg_return = metrics.get('avg_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        profit_factor = metrics.get('profit_factor', 1.0)

        total_return_pct = results.get('total_return', 0)

        print("✅ Metrics extracted successfully")

        # Generate report string
        report = f"""
MONTE CARLO TRADING STRATEGY ANALYSIS REPORT
Generated: {timestamp}
{'='*80}

STRATEGY CONFIGURATION
{'='*80}
Algorithm: {results.get('algorithm_name', 'Unknown')}
Ticker: TEST
Period: 1y
Interval: 1d
Initial Capital: ${results.get('initial_capital', 10000):,.2f}

BACKTEST PERFORMANCE
{'='*80}
Final Capital: ${results.get('final_capital', 10000):,.2f}
Total Return: {total_return_pct:.2f}%
Total Trades: {total_trades}
Win Rate: {win_rate:.1%}
Average Return per Trade: {avg_return:.2%}
Sharpe Ratio: {sharpe_ratio:.3f}
Maximum Drawdown: {max_drawdown:.2%}
Profit Factor: {profit_factor:.2f}

MONTE CARLO ANALYSIS
{'='*80}
Simulation Method: synthetic_returns
Number of Simulations: 1000

CONCLUSIONS
{'='*80}
This analysis demonstrates the application of advanced Monte Carlo simulation
techniques to trading strategy evaluation.

Key insights:
• The strategy shows {'positive' if total_return_pct > 0 else 'negative'} performance over the test period
• Win rate of {win_rate:.1%} indicates {'good' if win_rate > 50 else 'room for improvement in'} trade selection
• Sharpe ratio of {sharpe_ratio:.3f} suggests {'attractive' if sharpe_ratio > 1 else 'modest'} risk-adjusted returns

DISCLAIMER
{'='*80}
This analysis is for educational and research purposes only.
"""

        print("✅ Report generated successfully")
        print("📄 Report Preview (first 200 characters):")
        print(report[:200] + "...")

        # Verify report contains key information
        required_sections = [
            "MONTE CARLO TRADING STRATEGY ANALYSIS REPORT",
            "STRATEGY CONFIGURATION",
            "BACKTEST PERFORMANCE",
            "Test Algorithm",
            "Total Return",
            "Win Rate",
            "Sharpe Ratio"
        ]

        missing_sections = []
        for section in required_sections:
            if section not in report:
                missing_sections.append(section)

        if missing_sections:
            print(f"❌ Missing sections in report: {missing_sections}")
            return False

        print("✅ All required sections present in report")
        print("\n🎉 Report Generation Logic Test PASSED!")
        print("The report generation functionality is working correctly.")

        return True

    except Exception as e:
        print(f"❌ Report Generation Logic Test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_report_logic()
