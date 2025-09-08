"""
Real Data Monte Carlo Simulation Example

This script demonstrates how to use real market data (stocks/futures) 
with the Monte Carlo trade order simulation.
"""

import numpy as np
from data_fetcher import fetch_stock_data, fetch_futures_data, calculate_returns_from_ohlcv
from monte_carlo_trade_simulation import random_trade_order_simulation, plot_trade_order_simulations, analyze_simulation_results, print_analysis_report

def run_monte_carlo_with_real_data():
    """
    Fetch real market data and run Monte Carlo simulation to analyze sequence risk.
    """
    print("=== MONTE CARLO SIMULATION WITH REAL MARKET DATA ===")
    print("=" * 60)
    
    # Try to fetch real market data
    print("\n1. FETCHING MARKET DATA")
    print("-" * 30)
    
    try:
        # First try with stock data (more reliable)
        print("Attempting to fetch SPY ETF data...")
        stock_data = fetch_stock_data("SPY", period="3mo", interval="1h")
        returns = calculate_returns_from_ohlcv(stock_data, method="close_to_close", remove_outliers=True)
        data_source = "SPY ETF (1-hour intervals, 3 months)"
        
    except Exception as e:
        print(f"Stock data fetch failed: {e}")
        print("\nTrying with simulated realistic data...")
        
        # Fallback to realistic simulated data
        np.random.seed(42)
        returns = np.random.normal(loc=0.0005, scale=0.01, size=500)  # Realistic intraday returns
        data_source = "Simulated realistic market data"
    
    print(f"\n✓ Data source: {data_source}")
    print(f"✓ Number of return observations: {len(returns)}")
    print(f"✓ Data ready for Monte Carlo simulation")
    
    # Run Monte Carlo simulation
    print("\n" + "=" * 60)
    print("\n2. RUNNING MONTE CARLO SIMULATION")
    print("-" * 40)
    
    num_simulations = 1000
    initial_capital = 100000  # $100k starting capital
    
    print(f"Simulations: {num_simulations}")
    print(f"Initial capital: ${initial_capital:,}")
    print(f"Using {len(returns)} real market returns")
    print()
    
    # Run simulation
    simulation_results = random_trade_order_simulation(
        returns, 
        num_simulations=num_simulations,
        initial_capital=initial_capital,
        position_size_mode='compound'
    )
    
    # Analyze results
    print("\n" + "=" * 60)
    print("\n3. ANALYSIS RESULTS")
    print("-" * 25)
    
    analysis = analyze_simulation_results(simulation_results)
    print_analysis_report(analysis)
    
    # Additional insights
    print(f"\n4. KEY INSIGHTS FROM REAL DATA")
    print("-" * 35)
    
    spread = analysis['max_final_equity'] - analysis['min_final_equity']
    spread_pct = (spread / analysis['initial_capital']) * 100
    
    print(f"📊 Data Source: {data_source}")
    print(f"📈 Portfolio Value Range: ${analysis['min_final_equity']:,.2f} - ${analysis['max_final_equity']:,.2f}")
    print(f"💰 Absolute Spread: ${spread:,.2f}")
    print(f"📊 Percentage Spread: {spread_pct:.2f}% of initial capital")
    print(f"🎯 Risk Assessment: {analysis['probability_of_loss']:.1f}% chance of loss")
    
    if spread > 0:
        print(f"\n✅ SEQUENCE RISK DETECTED!")
        print(f"   The order of trades matters - same trades in different")
        print(f"   orders can result in ${spread:,.2f} difference in final value!")
    else:
        print(f"\n⚠️  NO SEQUENCE RISK DETECTED")
        print(f"   This may be due to mathematical properties of the returns")
        print(f"   or the small magnitude of individual trade impacts.")
    
    # Plot results
    print(f"\n5. GENERATING VISUALIZATION")
    print("-" * 30)
    print("Creating plot of simulation results...")
    
    try:
        plot_trade_order_simulations(simulation_results, show_percentiles=True)
        print("✓ Plot displayed successfully")
    except Exception as e:
        print(f"❌ Plot generation failed: {e}")
    
    return simulation_results, analysis


def compare_different_assets():
    """
    Compare sequence risk across different assets/timeframes.
    """
    print("\n" + "=" * 60)
    print("COMPARING SEQUENCE RISK ACROSS DIFFERENT SCENARIOS")
    print("=" * 60)
    
    scenarios = [
        {"name": "Conservative (Low Volatility)", "mean": 0.0002, "std": 0.005, "size": 252},
        {"name": "Moderate (Medium Volatility)", "mean": 0.0005, "std": 0.012, "size": 252},
        {"name": "Aggressive (High Volatility)", "mean": 0.001, "std": 0.025, "size": 252},
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        np.random.seed(42)  # Consistent for comparison
        returns = np.random.normal(scenario['mean'], scenario['std'], scenario['size'])
        
        results = random_trade_order_simulation(
            returns, 
            num_simulations=500,
            initial_capital=10000,
            position_size_mode='compound'
        )
        
        analysis = analyze_simulation_results(results)
        spread = analysis['max_final_equity'] - analysis['min_final_equity']
        
        print(f"  Final value range: ${analysis['min_final_equity']:,.2f} - ${analysis['max_final_equity']:,.2f}")
        print(f"  Spread: ${spread:,.2f}")
        print(f"  Std Dev of outcomes: ${analysis['volatility_of_outcomes']:,.2f}")


if __name__ == "__main__":
    try:
        # Run main simulation with real data
        results, analysis = run_monte_carlo_with_real_data()
        
        # Compare different scenarios
        compare_different_assets()
        
        print(f"\n" + "=" * 60)
        print("SIMULATION COMPLETE!")
        print("=" * 60)
        print("The simulation demonstrates how the order of identical trades")
        print("can affect final portfolio outcomes due to sequence risk.")
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        print("This might be due to network issues or data provider limitations.")
        print("Try running the simulation again or check your internet connection.")
