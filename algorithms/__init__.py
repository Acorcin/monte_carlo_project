"""
Trading Algorithms Package

This package contains various trading algorithms for backtesting
with the Monte Carlo simulation system.
"""

from .base_algorithm import TradingAlgorithm
from .algorithm_manager import AlgorithmManager

__all__ = ['TradingAlgorithm', 'AlgorithmManager']
