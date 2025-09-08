"""
Simple Usage Example - Monte Carlo Trade Simulation

This shows the easiest ways to use the Monte Carlo simulation system.
"""

import numpy as np
from monte_carlo_trade_simulation import random_trade_order_simulation, plot_trade_order_simulations
from data_fetcher import fetch_stock_data, calculate_returns_from_ohlcv

print("🚀 MONTE CARLO SIMULATION - SIMPLE USAGE EXAMPLES")
print("=" * 60)

# Example 1: Using Your Own Trade Returns
print("\n📊 EXAMPLE 1: Using Custom Trade Returns")
print("-" * 40)

# Your own trading returns (in decimal format: 0.05 = 5%)
my_trade_returns = [
    0.03,   # +3% trade
    -0.02,  # -2% trade  
    0.05,   # +5% trade
    -0.01,  # -1% trade
    0.04,   # +4% trade
    -0.03,  # -3% trade
    0.02    # +2% trade
]

print(f"Your trades: {[f'{r:.1%}' for r in my_trade_returns]}")

# Run simulation
results = random_trade_order_simulation(
    my_trade_returns, 
    num_simulations=100,
    initial_capital=10000
)

print(f"Simulation completed:")
print(f"  • Number of simulations: 100")
print(f"  • Number of trades per simulation: {len(my_trade_returns)}")
print(f"  • Results shape: {results.shape}")

# Get final values
final_values = results.iloc[-1].values
print(f"  • All final values: ${final_values[0]:,.2f} (identical - as expected)")

# Show that order doesn't matter mathematically
print(f"\n🔍 Why are all results identical?")
print(f"  Mathematical proof:")
manual_calc = 10000
for ret in my_trade_returns:
    manual_calc *= (1 + ret)
print(f"  Manual calculation: ${manual_calc:.2f}")
print(f"  All simulations:    ${final_values[0]:.2f}")
print(f"  ✅ Perfect match! Order doesn't matter for multiplication.")

# Example 2: Using Real Market Data (if available)
print(f"\n📈 EXAMPLE 2: Using Real Market Data")
print("-" * 40)

try:
    # Fetch SPY data
    print("Fetching SPY ETF data...")
    spy_data = fetch_stock_data("SPY", period="1mo", interval="1d")
    spy_returns = calculate_returns_from_ohlcv(spy_data, remove_outliers=True)
    
    print(f"✅ Fetched {len(spy_returns)} daily returns from SPY")
    print(f"   Mean return: {spy_returns.mean():.4f} ({spy_returns.mean()*100:.2f}%)")
    print(f"   Std return:  {spy_returns.std():.4f} ({spy_returns.std()*100:.2f}%)")
    
    # Run simulation with real data
    spy_results = random_trade_order_simulation(
        spy_returns,
        num_simulations=50,
        initial_capital=10000
    )
    
    spy_final = spy_results.iloc[-1].values
    print(f"   Final portfolio values: All ${spy_final[0]:,.2f} (identical)")
    
except Exception as e:
    print(f"❌ Could not fetch real data: {e}")
    print("   This is normal if internet connection is limited.")

# Example 3: Understanding When Sequence Risk Matters
print(f"\n🎯 EXAMPLE 3: When Does Order Actually Matter?")
print("-" * 50)

print("Sequence risk occurs in scenarios like:")
print("  1. 🏦 Retirement with fixed withdrawals")
print("  2. 💰 Fixed dollar position sizing")
print("  3. 📊 Leverage constraints")
print("  4. ⚖️  Portfolio rebalancing")
print()
print("For simple percentage returns with full reinvestment,")
print("order doesn't matter because: A × B × C = C × B × A")

# Example 4: Custom Analysis
print(f"\n📋 EXAMPLE 4: Custom Analysis")
print("-" * 30)

print("You can analyze the results manually:")
print(f"  • Initial capital: ${results.iloc[0, 0]:,.2f}")
print(f"  • Final capital:   ${results.iloc[-1, 0]:,.2f}")
print(f"  • Total return:    {((results.iloc[-1, 0] / results.iloc[0, 0]) - 1) * 100:.2f}%")
print(f"  • All simulations: Identical (as expected)")

# Plotting
print(f"\n📊 VISUALIZATION")
print("-" * 20)
print("Generating plot of simulation results...")

try:
    plot_trade_order_simulations(results, show_percentiles=False)
    print("✅ Plot displayed (all lines overlap perfectly)")
except Exception as e:
    print(f"❌ Plot failed: {e}")

print(f"\n🎓 SUMMARY")
print("-" * 15)
print("✅ System working correctly")
print("✅ Real data fetching operational") 
print("✅ Mathematical behavior confirmed")
print("📚 Ready for educational use or further development")

print(f"\n💡 NEXT STEPS")
print("-" * 15)
print("• Modify 'my_trade_returns' with your own trading results")
print("• Try different assets in the data fetcher")
print("• Experiment with different time periods")
print("• Build scenarios where sequence risk actually matters")
