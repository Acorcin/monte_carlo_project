"""
RSI Oversold/Overbought Algorithm

Mean reversion strategy that buys when RSI is oversold and sells when overbought.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_algorithm import TradingAlgorithm

class RSIOversoldOverbought(TradingAlgorithm):
    """
    RSI Oversold/Overbought Strategy.
    
    Buy when RSI < oversold_threshold.
    Sell when RSI > overbought_threshold.
    """
    
    def __init__(self, rsi_period: int = 14, oversold_threshold: float = 30, 
                 overbought_threshold: float = 70):
        """
        Initialize the RSI Oversold/Overbought algorithm.
        
        Args:
            rsi_period (int): Period for RSI calculation
            oversold_threshold (float): RSI level considered oversold
            overbought_threshold (float): RSI level considered overbought
        """
        parameters = {
            'rsi_period': rsi_period,
            'oversold_threshold': oversold_threshold,
            'overbought_threshold': overbought_threshold
        }
        
        super().__init__(
            name="RSI Oversold/Overbought",
            description=f"Buy when RSI < {oversold_threshold}, sell when RSI > {overbought_threshold}",
            parameters=parameters
        )
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            prices (pd.Series): Price series
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on RSI oversold/overbought levels.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            pd.Series: Trading signals (1=buy, -1=sell, 0=hold)
        """
        rsi_period = self.parameters['rsi_period']
        oversold = self.parameters['oversold_threshold']
        overbought = self.parameters['overbought_threshold']
        
        # Calculate RSI
        rsi = self.calculate_rsi(data['Close'], rsi_period)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Buy signal: RSI crosses below oversold threshold
        buy_signals = (rsi < oversold) & (rsi.shift(1) >= oversold)
        
        # Sell signal: RSI crosses above overbought threshold
        sell_signals = (rsi > overbought) & (rsi.shift(1) <= overbought)
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals
    
    def get_algorithm_type(self) -> str:
        """Return the algorithm type."""
        return "mean_reversion"
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Return parameter information for configuration."""
        return {
            'rsi_period': {
                'type': 'int',
                'default': 14,
                'min': 2,
                'max': 50,
                'description': 'Period for RSI calculation'
            },
            'oversold_threshold': {
                'type': 'float',
                'default': 30.0,
                'min': 10.0,
                'max': 40.0,
                'description': 'RSI level considered oversold'
            },
            'overbought_threshold': {
                'type': 'float',
                'default': 70.0,
                'min': 60.0,
                'max': 90.0,
                'description': 'RSI level considered overbought'
            }
        }
