"""
Trading Algorithms Package

This package contains various trading algorithms for backtesting
with the Monte Carlo simulation system.
"""

import sys
import os

# Ensure the algorithms directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .base_algorithm import TradingAlgorithm
    from .algorithm_manager import AlgorithmManager
except ImportError:
    try:
        from base_algorithm import TradingAlgorithm
        from algorithm_manager import AlgorithmManager
    except ImportError:
        # Fallback for import issues
        print("⚠️ Direct imports failed, using fallback method")
        TradingAlgorithm = None
        AlgorithmManager = None

__all__ = ['TradingAlgorithm', 'AlgorithmManager']
