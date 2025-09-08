"""
Moving Average Crossover Algorithm

Classic trend-following algorithm that generates buy signals when fast MA
crosses above slow MA, and sell signals when fast MA crosses below slow MA.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_algorithm import TradingAlgorithm

class MovingAverageCrossover(TradingAlgorithm):
    """
    Moving Average Crossover Strategy.
    
    Buy when fast MA crosses above slow MA.
    Sell when fast MA crosses below slow MA.
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        """
        Initialize the Moving Average Crossover algorithm.
        
        Args:
            fast_period (int): Period for fast moving average
            slow_period (int): Period for slow moving average
        """
        parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period
        }
        
        super().__init__(
            name="Moving Average Crossover",
            description=f"Buy when {fast_period}-period MA crosses above {slow_period}-period MA",
            parameters=parameters
        )
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on moving average crossover.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            pd.Series: Trading signals (1=buy, -1=sell, 0=hold)
        """
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        
        # Calculate moving averages
        fast_ma = data['Close'].rolling(window=fast_period).mean()
        slow_ma = data['Close'].rolling(window=slow_period).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Buy signal: fast MA crosses above slow MA
        buy_signals = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        
        # Sell signal: fast MA crosses below slow MA
        sell_signals = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals
    
    def get_algorithm_type(self) -> str:
        """Return the algorithm type."""
        return "trend_following"
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Return parameter information for configuration."""
        return {
            'fast_period': {
                'type': 'int',
                'default': 10,
                'min': 1,
                'max': 100,
                'description': 'Period for fast moving average'
            },
            'slow_period': {
                'type': 'int',
                'default': 30,
                'min': 2,
                'max': 200,
                'description': 'Period for slow moving average'
            }
        }
