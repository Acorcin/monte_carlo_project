"""
Sample Trading Algorithm Template

This is a template file that shows how to create a custom trading algorithm
that can be uploaded to the Monte Carlo GUI application.

To create your own algorithm:
1. Copy this template
2. Rename the class (MyCustomStrategy -> YourStrategyName)
3. Implement your trading logic in the generate_signals method
4. Update the docstring and algorithm type
5. Upload via the GUI

Requirements:
- Must inherit from TradingAlgorithm
- Must implement generate_signals() method
- Must implement get_algorithm_type() method
"""

import pandas as pd
import numpy as np
from algorithms.base_algorithm import TradingAlgorithm

class MyCustomStrategy(TradingAlgorithm):
    """
    My Custom Trading Strategy
    
    This is a sample strategy that demonstrates how to create a custom
    trading algorithm for the Monte Carlo simulation system.
    
    Strategy Description:
    - Uses simple moving averages for trend detection
    - Generates buy signals when short MA crosses above long MA
    - Generates sell signals when short MA crosses below long MA
    - Includes basic risk management
    
    Parameters:
    - short_window: Period for short moving average (default: 10)
    - long_window: Period for long moving average (default: 30)
    - risk_threshold: Maximum position size as fraction of capital (default: 0.1)
    """
    
    def __init__(self, name="My Custom Strategy", short_window=10, long_window=30, risk_threshold=0.1):
        """
        Initialize the custom strategy.
        
        Args:
            name (str): Name of the strategy
            short_window (int): Period for short moving average
            long_window (int): Period for long moving average
            risk_threshold (float): Maximum position size as fraction of capital
        """
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
        self.risk_threshold = risk_threshold
        
        print(f"🎯 Initialized {name}")
        print(f"   📊 Short MA: {short_window} periods")
        print(f"   📈 Long MA: {long_window} periods")
        print(f"   ⚠️ Risk threshold: {risk_threshold:.1%}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on moving average crossover.
        
        This method is required by the TradingAlgorithm base class.
        
        Args:
            data (pd.DataFrame): OHLCV market data with columns:
                - Open, High, Low, Close, Volume
                - Index should be datetime
        
        Returns:
            pd.Series: Trading signals aligned with data index
                - 1 = Buy signal
                - -1 = Sell signal  
                - 0 = Hold/No signal
        """
        
        # Validate input data
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            print(f"❌ Missing required columns. Found: {list(data.columns)}")
            return pd.Series(0, index=data.index, name='signal')
        
        if len(data) < self.long_window:
            print(f"❌ Insufficient data: {len(data)} rows, need {self.long_window}")
            return pd.Series(0, index=data.index, name='signal')
        
        print(f"🔍 Generating signals for {len(data)} data points...")
        
        # Calculate moving averages
        short_ma = data['Close'].rolling(window=self.short_window).mean()
        long_ma = data['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index, name='signal')
        
        # Buy signal: short MA crosses above long MA
        buy_condition = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        signals[buy_condition] = 1
        
        # Sell signal: short MA crosses below long MA
        sell_condition = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        signals[sell_condition] = -1
        
        # Count signals
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        
        print(f"📈 Generated {buy_signals} buy signals")
        print(f"📉 Generated {sell_signals} sell signals")
        
        return signals
    
    def get_algorithm_type(self) -> str:
        """
        Return the type/category of this algorithm.
        
        This method is required by the TradingAlgorithm base class.
        
        Returns:
            str: Algorithm category (e.g., 'trend_following', 'mean_reversion', 'momentum')
        """
        return 'trend_following'
    
    def calculate_position_size(self, signal, current_price, available_capital):
        """
        Calculate position size based on risk management rules.
        
        This is an optional method that can be used for advanced position sizing.
        
        Args:
            signal (int): Trading signal (1=buy, -1=sell, 0=hold)
            current_price (float): Current market price
            available_capital (float): Available capital for trading
            
        Returns:
            float: Position size (number of shares/contracts)
        """
        if signal == 0:
            return 0
        
        # Calculate maximum position size based on risk threshold
        max_position_value = available_capital * self.risk_threshold
        position_size = max_position_value / current_price
        
        return position_size
    
    def get_strategy_info(self):
        """
        Return detailed information about the strategy.
        
        Returns:
            dict: Strategy information
        """
        return {
            'name': self.name,
            'type': self.get_algorithm_type(),
            'parameters': {
                'short_window': self.short_window,
                'long_window': self.long_window,
                'risk_threshold': self.risk_threshold
            },
            'description': 'Moving average crossover strategy with basic risk management',
            'author': 'Monte Carlo System',
            'version': '1.0'
        }


# Example of how to test the algorithm (optional)
if __name__ == "__main__":
    # This code will run if the file is executed directly
    # It's useful for testing your algorithm before uploading
    
    print("🧪 Testing Custom Algorithm...")
    
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)  # For reproducible results
    
    sample_data = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'High': 100 + np.cumsum(np.random.randn(100) * 0.5) + 1,
        'Low': 100 + np.cumsum(np.random.randn(100) * 0.5) - 1,
        'Close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Ensure High >= Close >= Low
    sample_data['High'] = sample_data[['Open', 'Close', 'High']].max(axis=1)
    sample_data['Low'] = sample_data[['Open', 'Close', 'Low']].min(axis=1)
    
    # Test the algorithm
    strategy = MyCustomStrategy()
    signals = strategy.generate_signals(sample_data)
    
    print(f"✅ Test completed!")
    print(f"📊 Total signals: {len(signals)}")
    print(f"📈 Buy signals: {(signals == 1).sum()}")
    print(f"📉 Sell signals: {(signals == -1).sum()}")
    print(f"🔄 Hold signals: {(signals == 0).sum()}")
    
    # Show strategy info
    info = strategy.get_strategy_info()
    print(f"\n📋 Strategy Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
