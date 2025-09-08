"""
Monte Carlo Graphical Visualization Guide

This script shows you exactly how to view graphical interpretations
of your Monte Carlo simulations with comprehensive visualizations.
"""

import sys
sys.path.append('algorithms')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monte_carlo_trade_simulation import random_trade_order_simulation, plot_comprehensive_monte_carlo_analysis
from algorithms.algorithm_manager import AlgorithmManager
from data_fetcher import fetch_stock_data

def demo_monte_carlo_graphs():
    """Demonstrate all Monte Carlo graphical capabilities."""
    print("🎨 MONTE CARLO GRAPHICAL VISUALIZATION DEMO")
    print("="*70)
    print("This demonstrates all the ways to view Monte Carlo graphs")
    print()
    
    # Method 1: Basic Monte Carlo Visualization
    print("📊 METHOD 1: Basic Monte Carlo Simulation Graphs")
    print("-" * 50)
    
    # Create sample returns (simulating a trading strategy)
    np.random.seed(42)
    sample_returns = np.array([
        0.025, -0.015, 0.035, -0.008, 0.020, -0.012, 0.030, -0.005,
        0.018, -0.025, 0.040, -0.010, 0.022, -0.018, 0.028, -0.007
    ])
    
    print(f"   📈 Sample strategy returns: {len(sample_returns)} trades")
    print(f"   💰 Average return: {np.mean(sample_returns):.2%}")
    print(f"   📊 Volatility: {np.std(sample_returns):.2%}")
    
    print("\n   🎲 Running Monte Carlo simulation with automatic graphs...")
    
    # This will automatically display comprehensive graphs
    mc_results = random_trade_order_simulation(
        sample_returns,
        num_simulations=500,
        initial_capital=10000,
        simulation_method='synthetic_returns'
    )
    
    # Manually trigger comprehensive plotting
    plot_comprehensive_monte_carlo_analysis(
        mc_results, 
        simulation_method='synthetic_returns',
        title='Monte Carlo Analysis - Sample Strategy'
    )
    
    print("   ✅ Graphs should have appeared automatically!")
    
    # Method 2: Advanced ML Strategy Visualization
    print(f"\n📊 METHOD 2: Advanced ML Strategy with Full Visualization")
    print("-" * 50)
    
    try:
        # Get real market data
        data = fetch_stock_data('SPY', period='6mo', interval='1d')
        print(f"   📊 Loaded {len(data)} data points for SPY")
        
        # Run Advanced ML Strategy
        manager = AlgorithmManager()
        results = manager.backtest_algorithm('AdvancedMLStrategy', data, initial_capital=10000)
        
        if results and results['returns'] and len(results['returns']) > 5:
            print(f"   🧠 ML Strategy generated {len(results['returns'])} trades")
            
            # Run Monte Carlo on ML strategy returns
            ml_returns = np.array(results['returns'])
            
            print("   🎲 Running Monte Carlo on ML strategy...")
            ml_mc_results = random_trade_order_simulation(
                ml_returns,
                num_simulations=300,
                initial_capital=10000,
                simulation_method='synthetic_returns'
            )
            
            # Create comprehensive visualization
            plot_comprehensive_monte_carlo_analysis(
                ml_mc_results,
                simulation_method='synthetic_returns', 
                title='Advanced ML Strategy - Monte Carlo Analysis'
            )
            
            print("   ✅ Advanced ML graphs displayed!")
            
        else:
            print("   ⚠️  Not enough ML trades for visualization")
            
    except Exception as e:
        print(f"   ❌ Error with ML visualization: {e}")
    
    # Method 3: Market Scenario Visualization
    print(f"\n📊 METHOD 3: Market Scenario Testing with Graphs")
    print("-" * 50)
    
    try:
        from market_scenario_simulation import demo_market_scenario_simulation
        
        print("   🌍 Running market scenario analysis with full visualization...")
        scenarios, scenario_results = demo_market_scenario_simulation()
        print("   ✅ Market scenario graphs should have displayed!")
        
    except Exception as e:
        print(f"   ❌ Error with scenario visualization: {e}")
    
    # Summary
    print(f"\n🎯 SUMMARY: How to View Monte Carlo Graphs")
    print("="*60)
    print("✅ AUTOMATIC DISPLAY:")
    print("  • Graphs automatically pop up when running simulations")
    print("  • No additional commands needed")
    print()
    print("✅ MANUAL METHODS:")
    print("  1. python monte_carlo_trade_simulation.py")
    print("  2. python market_scenario_simulation.py") 
    print("  3. python demo_ml_with_monte_carlo.py")
    print("  4. python view_monte_carlo_graphs.py  # This script")
    print()
    print("✅ IN BACKTESTING SYSTEM:")
    print("  • Run: python backtest_algorithms.py")
    print("  • Answer 'y' to 'Generate equity curve plot?'")
    print("  • Answer 'y' to 'Run Monte Carlo analysis?'")
    print("  • Comprehensive graphs will display automatically")
    print()
    print("📊 GRAPH TYPES INCLUDED:")
    print("  • Portfolio equity curves (multiple simulation paths)")
    print("  • Final value distributions (histogram)")
    print("  • Return distributions")
    print("  • Drawdown analysis")
    print("  • Risk metrics (VaR, percentiles)")
    print("  • Performance statistics")
    print("  • Market scenario comparisons")
    print("  • Efficient frontier plots (portfolio optimization)")


def create_custom_visualization():
    """Create a custom Monte Carlo visualization."""
    print(f"\n🎨 BONUS: Custom Monte Carlo Visualization")
    print("-" * 50)
    
    # Create sample data
    returns = np.random.normal(0.01, 0.03, 20)  # 20 trades
    
    # Run simulation
    results = random_trade_order_simulation(
        returns,
        num_simulations=1000,
        initial_capital=10000,
        simulation_method='synthetic_returns'
    )
    
    # Create custom plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Custom Monte Carlo Visualization', fontsize=16, fontweight='bold')
    
    # 1. Equity curves
    ax = axes[0, 0]
    for i in range(min(50, results.shape[1])):
        ax.plot(results.iloc[:, i], alpha=0.3, linewidth=0.5, color='blue')
    
    mean_curve = results.mean(axis=1)
    ax.plot(mean_curve, color='red', linewidth=2, label='Mean Path')
    ax.set_title('Portfolio Evolution Paths')
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Final value distribution
    ax = axes[0, 1]
    final_values = results.iloc[-1].values
    ax.hist(final_values, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(np.mean(final_values), color='red', linestyle='--', 
               label=f'Mean: ${np.mean(final_values):,.0f}')
    ax.axvline(10000, color='orange', linestyle='-', 
               label='Initial Capital')
    ax.set_title('Final Portfolio Values')
    ax.set_xlabel('Final Value ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Return distribution
    ax = axes[1, 0]
    total_returns = (final_values - 10000) / 10000
    ax.hist(total_returns, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(np.mean(total_returns), color='red', linestyle='--', 
               label=f'Mean: {np.mean(total_returns):.1%}')
    ax.axvline(0, color='black', linestyle='-', alpha=0.5, label='Breakeven')
    ax.set_title('Total Return Distribution')
    ax.set_xlabel('Total Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Risk metrics
    ax = axes[1, 1]
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = [np.percentile(total_returns, p) for p in percentiles]
    
    ax.plot(percentiles, percentile_values, 'o-', linewidth=2, markersize=8)
    ax.axhline(0, color='red', linestyle='--', alpha=0.7, label='Breakeven')
    ax.set_title('Return Percentiles')
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("   ✅ Custom visualization created!")
    
    # Summary statistics
    print(f"\n📊 Simulation Results:")
    print(f"   💰 Portfolio range: ${np.min(final_values):,.0f} to ${np.max(final_values):,.0f}")
    print(f"   📈 Mean final value: ${np.mean(final_values):,.0f}")
    print(f"   📊 Standard deviation: ${np.std(final_values):,.0f}")
    print(f"   🎯 VaR (5%): ${np.percentile(final_values, 5):,.0f}")
    print(f"   ⚠️  Probability of loss: {np.sum(final_values < 10000)/len(final_values):.1%}")


if __name__ == "__main__":
    # Run the demo
    demo_monte_carlo_graphs()
    
    # Create custom visualization
    create_custom_visualization()
    
    print(f"\n🎉 MONTE CARLO VISUALIZATION COMPLETE!")
    print("All graphs should have appeared on your screen.")
    print("\nIf graphs didn't appear, check:")
    print("• Your matplotlib backend (try: python -c 'import matplotlib; print(matplotlib.get_backend())')")
    print("• Your display settings")
    print("• Try running from command line instead of IDE")
