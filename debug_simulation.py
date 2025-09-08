"""
Debug script to understand the simulation structure
"""

import numpy as np
from data_fetcher import fetch_stock_data, calculate_returns_from_ohlcv
from monte_carlo_trade_simulation import random_trade_order_simulation

# Simple test with small dataset
print("=== DEBUGGING MONTE CARLO SIMULATION ===")

# Create simple test data
test_returns = np.array([0.02, -0.015, 0.01, -0.008, 0.025])
print(f"Test returns: {test_returns}")

# Run small simulation
print("\nRunning simulation with 3 simulations...")
results = random_trade_order_simulation(
    test_returns, 
    num_simulations=3,
    initial_capital=1000,
    position_size_mode='compound'
)

print(f"\nResults shape: {results.shape}")
print(f"Results columns: {results.columns.tolist()}")
print(f"Results index: {results.index.tolist()}")

print(f"\nFirst few rows:")
print(results.head())

print(f"\nLast row (final values):")
print(results.iloc[-1])
print(f"Type of final values: {type(results.iloc[-1])}")

print(f"\nFinal values as array:")
final_values = results.iloc[-1].values
print(final_values)
print(f"Type: {type(final_values)}")

print(f"\nBasic statistics:")
print(f"Min: {final_values.min()}")
print(f"Max: {final_values.max()}")
print(f"Mean: {final_values.mean()}")
print(f"Std: {final_values.std()}")

# Test with different orderings manually
print(f"\n=== MANUAL TEST OF DIFFERENT ORDERINGS ===")

orderings = [
    [0.02, -0.015, 0.01, -0.008, 0.025],
    [0.025, 0.02, 0.01, -0.008, -0.015],
    [-0.015, -0.008, 0.01, 0.02, 0.025]
]

for i, order in enumerate(orderings):
    value = 1000
    print(f"\nOrder {i+1}: {order}")
    for ret in order:
        value = value * (1 + ret)
    print(f"Final value: ${value:.2f}")
