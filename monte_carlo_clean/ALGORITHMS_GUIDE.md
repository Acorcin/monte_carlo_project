# 🤖 Trading Algorithms Framework

## 📁 **Project Structure**

```
monte_carlo_project/
├── algorithms/                          # 🎯 Main algorithms folder
│   ├── technical_indicators/            # 📊 Technical analysis algorithms
│   │   ├── moving_average_crossover.py  # MA crossover strategy
│   │   └── __init__.py
│   ├── mean_reversion/                  # 🔄 Mean reversion algorithms  
│   │   ├── rsi_oversold_overbought.py   # RSI-based strategy
│   │   └── __init__.py
│   ├── momentum/                        # 🚀 Momentum algorithms
│   │   ├── price_momentum.py            # Price momentum strategy
│   │   └── __init__.py
│   ├── trend_following/                 # 📈 Trend following algorithms
│   ├── machine_learning/               # 🧠 ML-based algorithms  
│   ├── custom/                         # 🛠️ Your custom algorithms
│   ├── base_algorithm.py               # 🏗️ Base class for all algorithms
│   ├── algorithm_manager.py            # 🎮 Algorithm discovery & management
│   └── __init__.py
├── backtest_algorithms.py              # 🚀 Main backtesting interface
├── test_algorithms.py                  # 🧪 Testing & validation
└── [other Monte Carlo files]
```

## 🎯 **Quick Start**

### **1. Run Complete Backtesting System**
```bash
python backtest_algorithms.py
```

### **2. Test Everything Works**
```bash
python test_algorithms.py
```

### **3. See Available Algorithms**
```bash
python -c "import sys; sys.path.append('algorithms'); from algorithm_manager import algorithm_manager; algorithm_manager.print_available_algorithms()"
```

## 🤖 **Available Algorithms**

### **📊 Technical Indicators**
- **Moving Average Crossover**: Buy when fast MA crosses above slow MA
  - Parameters: `fast_period` (default: 10), `slow_period` (default: 30)

### **🔄 Mean Reversion**  
- **RSI Oversold/Overbought**: Buy when RSI < 30, sell when RSI > 70
  - Parameters: `rsi_period` (default: 14), `oversold_threshold` (default: 30), `overbought_threshold` (default: 70)

### **🚀 Momentum**
- **Price Momentum**: Buy when momentum > threshold
  - Parameters: `momentum_period` (default: 10), `buy_threshold` (default: 0.02), `sell_threshold` (default: -0.01)

## 🛠️ **Creating Your Own Algorithm**

### **Step 1: Choose Algorithm Category**
Put your algorithm in the appropriate folder:
- `technical_indicators/` - Chart-based analysis
- `mean_reversion/` - Buy low, sell high strategies  
- `momentum/` - Follow the trend strategies
- `trend_following/` - Long-term trend strategies
- `machine_learning/` - AI/ML-based strategies
- `custom/` - Your unique strategies

### **Step 2: Create Algorithm File**

```python
# algorithms/custom/my_algorithm.py

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_algorithm import TradingAlgorithm

class MyCustomAlgorithm(TradingAlgorithm):
    """
    Your custom trading algorithm.
    
    Describe your strategy here.
    """
    
    def __init__(self, param1: int = 10, param2: float = 0.05):
        """Initialize your algorithm."""
        parameters = {
            'param1': param1,
            'param2': param2
        }
        
        super().__init__(
            name="My Custom Algorithm",
            description="Description of what your algorithm does",
            parameters=parameters
        )
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            pd.Series: Trading signals (1=buy, -1=sell, 0=hold)
        """
        # Your algorithm logic here
        signals = pd.Series(0, index=data.index)
        
        # Example: Simple price-based signal
        price_change = data['Close'].pct_change()
        
        # Buy when price goes up more than param2
        buy_signals = price_change > self.parameters['param2']
        
        # Sell when price goes down more than param2  
        sell_signals = price_change < -self.parameters['param2']
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals
    
    def get_algorithm_type(self) -> str:
        """Return algorithm category."""
        return "custom"  # or "momentum", "mean_reversion", etc.
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Define parameter configuration."""
        return {
            'param1': {
                'type': 'int',
                'default': 10,
                'min': 1,
                'max': 100,
                'description': 'Description of param1'
            },
            'param2': {
                'type': 'float', 
                'default': 0.05,
                'min': 0.01,
                'max': 0.5,
                'description': 'Description of param2'
            }
        }
```

### **Step 3: Test Your Algorithm**
```bash
python test_algorithms.py
```
Your algorithm will be automatically discovered and tested!

## 📊 **Algorithm Performance Metrics**

Each backtest provides:

- **Total Return**: Overall percentage gain/loss
- **Total Trades**: Number of completed trades  
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: Gross profits / Gross losses

## 🎛️ **Backtesting Features**

### **Data Sources**
- **SPY ETF**: S&P 500 index fund  
- **Custom Tickers**: Any Yahoo Finance symbol
- **Time Periods**: 1mo, 3mo, 6mo, 1y, 2y, 5y
- **Intervals**: 1d (daily), 1h (hourly)

### **Backtesting Options**
- **Individual Algorithm**: Test one algorithm with custom parameters
- **Algorithm Categories**: Test all algorithms in a category
- **All Algorithms**: Test everything at once
- **Parameter Optimization**: Customize algorithm settings

### **Analysis Features**
- **Performance Comparison**: Side-by-side results table
- **Equity Curve Plotting**: Visual performance comparison  
- **Monte Carlo Integration**: Test with return shuffling
- **CSV Export**: Save results for further analysis

## 🔧 **Advanced Usage**

### **Programmatic Usage**
```python
from algorithms.algorithm_manager import algorithm_manager
from data_fetcher import fetch_stock_data

# Get data
data = fetch_stock_data("SPY", period="6mo", interval="1d")

# Create algorithm with custom parameters
algorithm = algorithm_manager.create_algorithm(
    "MovingAverageCrossover", 
    parameters={'fast_period': 5, 'slow_period': 20}
)

# Run backtest
results = algorithm.backtest(data, initial_capital=10000)

# Access results
print(f"Total Return: {results['total_return']:.2f}%")
print(f"Number of Trades: {results['metrics']['total_trades']}")
```

### **Batch Testing Multiple Algorithms**
```python
# Test multiple algorithms
configs = [
    {'name': 'MovingAverageCrossover', 'parameters': {'fast_period': 5}},
    {'name': 'RSIOversoldOverbought', 'parameters': {'rsi_period': 10}},
    {'name': 'PriceMomentum', 'parameters': {'momentum_period': 15}}
]

results = algorithm_manager.backtest_multiple_algorithms(configs, data)
```

## 💡 **Algorithm Ideas to Implement**

### **Technical Indicators**
- Bollinger Bands strategy
- MACD crossover
- Stochastic oscillator
- Williams %R

### **Mean Reversion**
- Pairs trading
- Statistical arbitrage
- Contrarian strategies
- Market maker algorithms

### **Momentum**
- Breakout strategies
- Channel breakouts
- Turtle trading system
- Trend following with stops

### **Machine Learning**
- Linear regression predictions
- Random forest classifier
- Neural network predictions
- Sentiment analysis integration

## ⚠️ **Important Notes**

1. **Paper Trading**: All results are simulated - not real money
2. **Overfitting Risk**: Don't over-optimize on historical data
3. **Transaction Costs**: Consider commissions and slippage
4. **Market Conditions**: Strategies may work differently in different markets
5. **Risk Management**: Always use proper position sizing and stop losses

## 🎯 **Next Steps**

1. **Create your first custom algorithm** in `algorithms/custom/`
2. **Test it** with `python test_algorithms.py` 
3. **Backtest it** with `python backtest_algorithms.py`
4. **Optimize parameters** for better performance
5. **Compare** with existing algorithms
6. **Integrate with Monte Carlo** simulation for risk analysis

The framework is designed to be easily extensible - add your trading ideas and see how they perform!
