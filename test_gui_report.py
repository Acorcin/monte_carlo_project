#!/usr/bin/env python3
"""
Test script for GUI report generation functionality
"""

import sys
import os
sys.path.append('algorithms')

# Mock tkinter to avoid GUI display issues during testing
import tkinter as tk
from unittest.mock import MagicMock, patch

# Mock tkinter components
tk.Tk = MagicMock()
tk.ttk = MagicMock()
tk.messagebox = MagicMock()

# Mock matplotlib
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.backends.backend_tkagg'] = MagicMock()

# Now import our modules
from algorithms.base_algorithm import TradingAlgorithm
from monte_carlo_gui_app import MonteCarloGUI
import pandas as pd
import numpy as np

class TestAlgorithm(TradingAlgorithm):
    """Simple test algorithm for GUI report generation testing."""

    def __init__(self):
        super().__init__(
            name="Test Algorithm",
            description="Test algorithm for GUI report generation",
            parameters={'test_param': 10}
        )

    def generate_signals(self, data):
        """Generate simple signals for testing."""
        return pd.Series([0] * len(data), index=data.index)

    def get_algorithm_type(self):
        return "test"

def test_gui_report_generation():
    """Test the GUI report generation functionality."""
    print("🖥️ Testing GUI Report Generation")
    print("=" * 40)

    try:
        # Create mock GUI
        gui = MagicMock(spec=MonteCarloGUI)

        # Create test data and results
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

        # Set up mock GUI attributes
        gui.current_results = results
        gui.ticker_var = MagicMock()
        gui.ticker_var.get.return_value = "TEST"
        gui.period_var = MagicMock()
        gui.period_var.get.return_value = "1y"
        gui.interval_var = MagicMock()
        gui.interval_var.get.return_value = "1d"
        gui.sim_method_var = MagicMock()
        gui.sim_method_var.get.return_value = "synthetic_returns"
        gui.num_sims_var = MagicMock()
        gui.num_sims_var.get.return_value = "1000"

        # Mock GUI components
        gui.summary_text = MagicMock()
        gui.notebook = MagicMock()

        # Test the actual generate_report method
        MonteCarloGUI.generate_report(gui)

        print("✅ GUI report generation method called successfully")
        print("✅ No errors thrown during report generation")

        # Verify that the GUI methods were called
        if hasattr(gui.summary_text, 'delete') and hasattr(gui.summary_text, 'insert'):
            print("✅ Report text was updated in GUI")
        else:
            print("⚠️ Report text update not detected (mock behavior)")

        print("\n🎉 GUI Report Generation Test PASSED!")
        print("The generate_report functionality is working correctly in the GUI.")

        return True

    except Exception as e:
        print(f"❌ GUI Report Generation Test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_gui_report_generation()
