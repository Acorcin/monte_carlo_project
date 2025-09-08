"""
Price Momentum Algorithm

Momentum strategy that buys when price momentum is strong upward
and sells when momentum turns downward.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_algorithm import TradingAlgorithm

class PriceMomentum(TradingAlgorithm):
    """
    Price Momentum Strategy.
    
    Buy when price momentum > buy_threshold.
    Sell when price momentum < sell_threshold.
    """
    
    def __init__(self, momentum_period: int = 10, buy_threshold: float = 0.02, 
                 sell_threshold: float = -0.01):
        """
        Initialize the Price Momentum algorithm.
        
        Args:
            momentum_period (int): Period for momentum calculation
            buy_threshold (float): Momentum threshold for buy signals
            sell_threshold (float): Momentum threshold for sell signals
        """
        parameters = {
            'momentum_period': momentum_period,
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold
        }
        
        super().__init__(
            name="Price Momentum",
            description=f"Buy when {momentum_period}-period momentum > {buy_threshold:.1%}",
            parameters=parameters
        )
    
    def calculate_momentum(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate price momentum.
        
        Args:
            prices (pd.Series): Price series
            period (int): Momentum period
            
        Returns:
            pd.Series: Momentum values (price change over period)
        """
        return prices.pct_change(periods=period)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on price momentum.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            pd.Series: Trading signals (1=buy, -1=sell, 0=hold)
        """
        momentum_period = self.parameters['momentum_period']
        buy_threshold = self.parameters['buy_threshold']
        sell_threshold = self.parameters['sell_threshold']
        
        # Calculate momentum
        momentum = self.calculate_momentum(data['Close'], momentum_period)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Buy signal: momentum crosses above buy threshold
        buy_signals = (momentum > buy_threshold) & (momentum.shift(1) <= buy_threshold)
        
        # Sell signal: momentum crosses below sell threshold
        sell_signals = (momentum < sell_threshold) & (momentum.shift(1) >= sell_threshold)
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals
    
    def get_algorithm_type(self) -> str:
        """Return the algorithm type."""
        return "momentum"
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Return parameter information for configuration."""
        return {
            'momentum_period': {
                'type': 'int',
                'default': 10,
                'min': 1,
                'max': 50,
                'description': 'Period for momentum calculation'
            },
            'buy_threshold': {
                'type': 'float',
                'default': 0.02,
                'min': 0.001,
                'max': 0.1,
                'description': 'Momentum threshold for buy signals'
            },
            'sell_threshold': {
                'type': 'float',
                'default': -0.01,
                'min': -0.1,
                'max': -0.001,
                'description': 'Momentum threshold for sell signals'
            }
        }
