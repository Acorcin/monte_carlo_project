# Monte Carlo Trade Order Simulation

A Python tool for analyzing the impact of trade order randomization on portfolio performance using Monte Carlo simulation methods.

## Overview

This tool helps quantify **sequence risk** - the risk that the order of returns can significantly impact the final portfolio value. By shuffling the same set of trade returns thousands of times, you can understand the range of possible outcomes based purely on timing.

## Features

- **Monte Carlo Simulation**: Run thousands of simulations with randomized trade orders
- **Enhanced Visualization**: Plot equity curves with percentile bands
- **Comprehensive Analysis**: Statistical analysis including VaR, probability metrics
- **Risk Assessment**: Understand the impact of trade sequence on portfolio outcomes

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
import numpy as np
from monte_carlo_trade_simulation import random_trade_order_simulation, plot_trade_order_simulations

# Example trade returns (in decimal format: 0.05 = 5%)
trade_returns = [0.02, -0.01, 0.03, -0.005, 0.015, 0.008, -0.012]

# Run simulation
results = random_trade_order_simulation(
    trade_returns, 
    num_simulations=1000, 
    initial_capital=10000
)

# Plot results
plot_trade_order_simulations(results)
```

### Advanced Analysis

```python
from monte_carlo_trade_simulation import analyze_simulation_results, print_analysis_report

# Get detailed statistics
analysis = analyze_simulation_results(results)
print_analysis_report(analysis)
```

## Key Metrics

The analysis provides several important metrics:

- **Final Equity Statistics**: Mean, median, standard deviation of final portfolio values
- **Return Statistics**: Total return statistics across all simulations
- **Risk Metrics**: 
  - Probability of loss/gain
  - Value at Risk (VaR) at 1% and 5% levels
  - Volatility of outcomes

## Understanding the Results

- **Spread of Outcomes**: Shows how much the final portfolio value can vary based solely on trade order
- **Percentile Bands**: Visualize the distribution of possible equity curves
- **Risk Assessment**: Understand worst-case scenarios and their probabilities

## Applications

- **Strategy Validation**: Test the robustness of trading strategies
- **Risk Management**: Quantify sequence risk in backtesting
- **Portfolio Analysis**: Understand the impact of trade timing
- **Performance Attribution**: Separate skill from luck in trading results

## Example Output

```
Monte Carlo Trade Order Simulation Example
--------------------------------------------------
Analyzing 252 trades with 500 simulations...

============================================================
MONTE CARLO TRADE ORDER SIMULATION ANALYSIS
============================================================

Initial Capital: $10,000.00

FINAL EQUITY STATISTICS:
  Mean:     $12,693.85
  Median:   $12,601.47
  Std Dev:  $1,247.83
  Min:      $9,778.98
  Max:      $16,421.31

TOTAL RETURN STATISTICS:
  Mean:     26.94%
  Median:   26.01%
  Std Dev:  12.48%
  Min:      -2.21%
  Max:      64.21%

RISK METRICS:
  Probability of Loss:     2.8%
  Probability of Gain:     97.2%
  Value at Risk (5%):      $10,728.45
  Value at Risk (1%):      $10,234.12
============================================================
```

## Notes

- Trade returns should be in decimal format (0.05 for 5%)
- The simulation preserves the original set of returns, only changing their order
- Results are reproducible when using the same random seed
- Extreme negative returns (< -100%) will trigger warnings
