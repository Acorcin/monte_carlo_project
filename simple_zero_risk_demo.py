"""
Simple Demo: Your Exact Plotting Code

This script shows your exact plotting code working with minimal modifications,
integrated with real market data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portfolio_optimization import PortfolioOptimizer

def simple_demo():
    """
    Simple demonstration of your exact plotting code.
    """
    print("🎯 YOUR EXACT PLOTTING CODE DEMO")
    print("="*40)
    print("Running your code with real market data...\n")
    
    # Setup with real data
    assets = ['AAPL', 'MSFT', 'GOOGL']
    optimizer = PortfolioOptimizer(assets, period="3mo")
    
    # Get real market data and run Monte Carlo
    print("📊 Fetching real market data...")
    results = optimizer.run_full_optimization(num_simulations=2000, plot_results=False)
    
    if results is None:
        print("❌ Could not fetch data")
        return
    
    # Extract your variables from real data
    simulation_results = results['simulation_results']['weights']
    portfolio_returns = results['simulation_results']['returns']
    portfolio_volatility = results['simulation_results']['volatility']
    
    print(f"✅ Data loaded: {len(portfolio_returns)} portfolios simulated")
    print(f"✅ Assets: {', '.join(assets)}")
    print(f"✅ Variables ready for your code\n")
    
    # =============================================================
    # YOUR EXACT CODE STARTS HERE (unmodified)
    # =============================================================
    
    print("🔥 EXECUTING YOUR EXACT CODE:")
    print("-" * 35)
    
    # Create a DataFrame from simulation results
    portfolio_df = pd.DataFrame({'Return': portfolio_returns, 'Volatility': portfolio_volatility})
    
    # simplifying assumption, risk free rate is zero, for sharpe ratio
    risk_free_rate = 0
    
    # Create Sharpe ratio array  
    sharpe_arr = (portfolio_returns - risk_free_rate) / portfolio_volatility
    max_sr_idx = sharpe_arr.argmax()
    
    # Get optimal portfolio
    optimal_weights = simulation_results[max_sr_idx]
    optimal_return = portfolio_returns[max_sr_idx]
    optimal_volatility = portfolio_volatility[max_sr_idx]
    
    # Plot the Monte Carlo efficient frontier
    plt.figure(figsize=(12, 6))
    plt.scatter(portfolio_df['Volatility'], portfolio_df['Return'], c=(portfolio_df['Return']-risk_free_rate) / portfolio_df['Volatility'], marker='o')  
    plt.title('Monte Carlo Efficient Frontier')
    plt.xlabel('Portfolio Volatility')
    plt.ylabel('Portfolio Return')
    plt.colorbar(label='Sharpe Ratio')
    
    # Add a red dot for the optimal portfolio
    plt.scatter(optimal_volatility, optimal_return, color='red', marker='o', s=100, label='Optimal Portfolio')
    
    # Show legend
    plt.legend()
    plt.show()
    
    # =============================================================
    # YOUR EXACT CODE ENDS HERE
    # =============================================================
    
    print("✅ Your plotting code executed successfully!")
    
    # Show results
    print(f"\n🎯 RESULTS FROM YOUR CODE:")
    print("-" * 35)
    print(f"Risk-Free Rate:      {risk_free_rate:.1%}")
    print(f"Max Sharpe Ratio:    {sharpe_arr[max_sr_idx]:.4f}")
    print(f"Optimal Return:      {optimal_return:.2%}")
    print(f"Optimal Volatility:  {optimal_volatility:.2%}")
    
    print(f"\nOptimal Allocation:")
    for i, asset in enumerate(assets):
        print(f"  {asset}: {optimal_weights[i]:>7.1%}")
    
    print(f"\n✅ SUCCESS! Your code generated:")
    print(f"   • Monte Carlo efficient frontier plot")
    print(f"   • Portfolios colored by Sharpe ratio")
    print(f"   • Red dot highlighting optimal portfolio")
    print(f"   • Real market data integration")
    
    return {
        'portfolio_df': portfolio_df,
        'risk_free_rate': risk_free_rate,
        'optimal_weights': optimal_weights,
        'optimal_return': optimal_return,
        'optimal_volatility': optimal_volatility,
        'max_sharpe_ratio': sharpe_arr[max_sr_idx]
    }


if __name__ == "__main__":
    simple_demo()
