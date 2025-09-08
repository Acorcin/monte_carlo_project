# 🎛️ Usage Guide - Changing Number of Simulations

## 📊 **Multiple Ways to Control Simulation Count**

### **Method 1: Interactive Mode (Recommended)**
```bash
python interactive_simulation.py
```

**Features:**
- ✅ User-friendly prompts
- ✅ Input validation
- ✅ Quick menu options (100, 1K, 5K, 10K simulations)
- ✅ Custom data source selection
- ✅ Visualization and saving options

**Example:**
```
📊 Number of simulations (default: 1000): 2500
💰 Initial capital (default: $10,000): 50000
📈 Data Source Options:
  1. Use example trade returns
  2. Fetch real market data (SPY ETF)
  3. Enter custom trade returns
Select option (1, 2, or 3): 2
```

### **Method 2: Command Line Arguments**
```bash
python monte_carlo_trade_simulation.py [NUMBER_OF_SIMULATIONS]
```

**Examples:**
```bash
python monte_carlo_trade_simulation.py 1000    # 1,000 simulations
python monte_carlo_trade_simulation.py 5000    # 5,000 simulations
python monte_carlo_trade_simulation.py 100     # 100 simulations (fast test)
```

### **Method 3: Quick Menu**
```bash
python interactive_simulation.py
# Then select "y" for quick menu
# Choose from: 100, 1000, 5000, 10000 simulations
```

### **Method 4: Programming (Custom Script)**
```python
from monte_carlo_trade_simulation import random_trade_order_simulation

# Your returns
returns = [0.05, -0.02, 0.03, -0.01, 0.04]

# Custom simulation count
results = random_trade_order_simulation(
    returns,
    num_simulations=2500,  # ← Change this number
    initial_capital=10000
)
```

## ⚡ **Quick Reference**

| Simulation Count | Usage | Speed | Purpose |
|-----------------|-------|-------|---------|
| 100 | Testing | ⚡ Very Fast | Quick validation |
| 1,000 | Standard | 🚀 Fast | Normal analysis |
| 5,000 | Comprehensive | ⏱️ Medium | Detailed analysis |
| 10,000+ | Extensive | 🐌 Slow | Research/publication |

## 💡 **Performance Tips**

### **For Large Simulations (10,000+):**
- ⚠️ System will warn you about potential slowness
- 📊 Plotting will automatically sample curves (max 100 shown)
- 💾 Consider saving results to CSV for later analysis
- 🖥️ Close other applications to free up memory

### **Recommended Simulation Counts:**
- **Learning/Testing**: 100-500 simulations
- **Analysis**: 1,000-2,000 simulations  
- **Research**: 5,000-10,000 simulations
- **Academic**: 10,000+ simulations

## 🎯 **Examples by Use Case**

### **Quick Test (Development)**
```bash
python interactive_simulation.py
# Quick menu → 1 (Fast test - 100 simulations)
```

### **Standard Analysis**
```bash
python monte_carlo_trade_simulation.py 1000
```

### **Research Paper**
```bash
python interactive_simulation.py
# Custom → 10000 simulations
# Real market data
# Save results to CSV
```

### **Educational Demo**
```bash
python simple_usage_example.py  # Fixed 100 simulations
```

## 📈 **Data Source Options**

### **1. Example Returns (Built-in)**
- Pre-defined realistic returns
- Good for learning and testing
- Consistent results across runs

### **2. Real Market Data**
- SPY ETF from Yahoo Finance
- Choose timeframe (1mo, 3mo, 6mo, 1y)
- Choose interval (1d, 1h)
- Demonstrates real-world applicability

### **3. Custom Returns**
- Enter your actual trading results
- Interactive input validation
- Supports any number of trades

## 🔧 **Troubleshooting**

### **Slow Performance?**
- Reduce simulation count
- Use quick test mode (100 sims)
- Close other applications

### **Memory Issues?**
- Reduce to < 5,000 simulations
- Don't save large result sets
- Use command line mode instead of interactive

### **Want Different Results?**
- Remember: identical results are mathematically correct
- For true sequence risk, need withdrawal scenarios
- Consider building custom position sizing models

## 📋 **File Summary**

| File | Purpose | Simulation Control |
|------|---------|-------------------|
| `interactive_simulation.py` | ⭐ Full control | User input + validation |
| `monte_carlo_trade_simulation.py` | Command line | Argument: `python file.py 1000` |
| `simple_usage_example.py` | Basic demo | Fixed 100 simulations |
| `debug_simulation.py` | Math verification | Fixed 3 simulations |

## 🚀 **Getting Started**

1. **First time?** Run `python simple_usage_example.py`
2. **Want control?** Run `python interactive_simulation.py`
3. **Command line?** Run `python monte_carlo_trade_simulation.py 1000`
4. **Custom code?** Import and use `random_trade_order_simulation()`

The system is designed to be flexible - use whichever method works best for your needs!
