"""
Your Original Code - Integrated Example

This script demonstrates how your original Monte Carlo portfolio optimization code
has been integrated into our comprehensive trading and portfolio system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portfolio_optimization import PortfolioOptimizer

def demonstrate_your_original_code():
    """
    Demonstrate your original code integration with real market data.
    
    This shows exactly how your code snippet works within our enhanced framework.
    """
    print("🎯 YOUR ORIGINAL CODE DEMONSTRATION")
    print("="*50)
    
    # Setup with real market data
    assets = ['AAPL', 'MSFT', 'GOOGL']
    optimizer = PortfolioOptimizer(assets, period="3mo")
    
    # Fetch data and run optimization
    results = optimizer.run_full_optimization(num_simulations=3000, plot_results=False)
    
    if results is None:
        print("❌ Optimization failed")
        return
    
    # Extract the simulation results (your variables)
    simulation_results = results['simulation_results']['weights']  # Your simulation_results array
    portfolio_returns = results['simulation_results']['returns']   # Your portfolio_returns array  
    portfolio_volatility = results['simulation_results']['volatility']  # Your portfolio_volatility array
    num_simulations = len(portfolio_returns)
    num_assets = len(assets)
    
    print(f"\n🔍 YOUR ORIGINAL CODE VARIABLES:")
    print(f"   simulation_results shape: {simulation_results.shape}")
    print(f"   portfolio_returns shape:  {portfolio_returns.shape}")  
    print(f"   portfolio_volatility shape: {portfolio_volatility.shape}")
    print(f"   num_simulations: {num_simulations}")
    print(f"   num_assets: {num_assets}")
    
    # =============================================================
    # YOUR ORIGINAL CODE STARTS HERE (with minimal modifications)
    # =============================================================
    
    print(f"\n📊 EXECUTING YOUR ORIGINAL CODE:")
    print("-" * 40)
    
    # Create a DataFrame from simulation results
    portfolio_df = pd.DataFrame({'Return': portfolio_returns, 'Volatility': portfolio_volatility})
    print(f"✅ Created portfolio DataFrame: {portfolio_df.shape}")
    
    # Create an array of the Sharpe ratio
    risk_free_rate = 0.03  # Replace with your risk-free rate
    sharpe_arr = (portfolio_returns - risk_free_rate) / portfolio_volatility
    print(f"✅ Calculated Sharpe ratio array: {len(sharpe_arr)} values")
    
    # Find the index of the maximum Sharpe ratio
    max_sr_idx = sharpe_arr.argmax()
    print(f"✅ Found maximum Sharpe ratio index: {max_sr_idx}")
    
    # Retrieve the optimal weights and corresponding return and volatility
    optimal_weights = simulation_results[max_sr_idx]
    optimal_return = portfolio_returns[max_sr_idx]
    optimal_volatility = portfolio_volatility[max_sr_idx]
    print(f"✅ Retrieved optimal portfolio characteristics")
    
    # Calculate the Sharpe ratio at the maximum point
    MC_SR = sharpe_arr[max_sr_idx]
    print(f"✅ Maximum Sharpe ratio: {MC_SR:.4f}")
    
    # Calculate the annualized Sharpe ratio
    SR_annualized = MC_SR * np.sqrt(12)  # Assuming monthly data, annualize by sqrt(12)
    print(f"✅ Annualized Sharpe ratio: {SR_annualized:.4f}")
    
    # =============================================================
    # YOUR ORIGINAL CODE ENDS HERE
    # =============================================================
    
    print(f"\n🎯 YOUR CODE RESULTS:")
    print("="*50)
    print(f"Risk-Free Rate:        {risk_free_rate:.1%}")
    print(f"Maximum Sharpe Ratio:  {MC_SR:.4f}")
    print(f"Annualized Sharpe:     {SR_annualized:.4f}")
    print(f"Optimal Return:        {optimal_return:.2%}")
    print(f"Optimal Volatility:    {optimal_volatility:.2%}")
    print(f"Simulation Index:      {max_sr_idx:,}")
    
    print(f"\nOptimal Asset Allocation:")
    for i, asset in enumerate(assets):
        print(f"  {asset}: {optimal_weights[i]:>8.1%}")
    
    # Additional analysis using your results
    print(f"\n📊 ADDITIONAL ANALYSIS:")
    print("="*50)
    
    # Sharpe ratio statistics
    print(f"Sharpe Ratio Statistics:")
    print(f"  Mean:     {sharpe_arr.mean():.4f}")
    print(f"  Median:   {np.median(sharpe_arr):.4f}")
    print(f"  Std Dev:  {sharpe_arr.std():.4f}")
    print(f"  Min:      {sharpe_arr.min():.4f}")
    print(f"  Max:      {sharpe_arr.max():.4f}")
    
    # Percentile analysis
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\nSharpe Ratio Percentiles:")
    for p in percentiles:
        value = np.percentile(sharpe_arr, p)
        print(f"  {p:2d}th: {value:.4f}")
    
    # Show how many portfolios achieved different Sharpe levels
    excellent_count = (sharpe_arr >= 1.5).sum()
    good_count = ((sharpe_arr >= 1.0) & (sharpe_arr < 1.5)).sum()
    moderate_count = ((sharpe_arr >= 0.5) & (sharpe_arr < 1.0)).sum()
    
    print(f"\nPortfolio Quality Distribution:")
    print(f"  Excellent (≥1.5): {excellent_count:4,} ({excellent_count/len(sharpe_arr)*100:5.1f}%)")
    print(f"  Good (1.0-1.5):   {good_count:4,} ({good_count/len(sharpe_arr)*100:5.1f}%)")
    print(f"  Moderate (0.5-1): {moderate_count:4,} ({moderate_count/len(sharpe_arr)*100:5.1f}%)")
    
    # Create visualization
    print(f"\n📈 CREATING VISUALIZATION...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Efficient frontier with Sharpe coloring
    scatter = ax1.scatter(portfolio_volatility, portfolio_returns, 
                         c=sharpe_arr, cmap='viridis', alpha=0.6, s=30)
    ax1.scatter(optimal_volatility, optimal_return, 
               color='red', marker='*', s=300, label='Your Optimal Portfolio')
    ax1.set_xlabel('Volatility (Risk)')
    ax1.set_ylabel('Expected Return')
    ax1.set_title('Efficient Frontier (Your Code Results)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Sharpe Ratio')
    
    # Plot 2: Sharpe ratio distribution
    ax2.hist(sharpe_arr, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(MC_SR, color='red', linestyle='-', linewidth=3, 
               label=f'Your Max Sharpe: {MC_SR:.3f}')
    ax2.axvline(sharpe_arr.mean(), color='orange', linestyle='--', 
               label=f'Mean: {sharpe_arr.mean():.3f}')
    ax2.set_xlabel('Sharpe Ratio')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Sharpe Ratio Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Return the key results for further use
    return {
        'portfolio_df': portfolio_df,
        'sharpe_arr': sharpe_arr,
        'max_sr_idx': max_sr_idx,
        'optimal_weights': optimal_weights,
        'optimal_return': optimal_return,
        'optimal_volatility': optimal_volatility,
        'MC_SR': MC_SR,
        'SR_annualized': SR_annualized,
        'risk_free_rate': risk_free_rate,
        'assets': assets,
        'simulation_results': simulation_results,
        'portfolio_returns': portfolio_returns,
        'portfolio_volatility': portfolio_volatility
    }


def compare_with_different_risk_free_rates():
    """
    Show how your code performs with different risk-free rate assumptions.
    """
    print(f"\n🔄 SENSITIVITY ANALYSIS: RISK-FREE RATE IMPACT")
    print("="*60)
    
    # Test different risk-free rates
    risk_free_rates = [0.01, 0.02, 0.03, 0.04, 0.05]  # 1% to 5%
    
    assets = ['AAPL', 'MSFT']  # Simpler example
    optimizer = PortfolioOptimizer(assets, period="3mo")
    results = optimizer.run_full_optimization(num_simulations=2000, plot_results=False)
    
    if results is None:
        print("❌ Could not fetch data")
        return
    
    portfolio_returns = results['simulation_results']['returns']
    portfolio_volatility = results['simulation_results']['volatility']
    simulation_results = results['simulation_results']['weights']
    
    print(f"Testing with {len(risk_free_rates)} different risk-free rates:")
    print(f"{'Risk-Free':<12} {'Max Sharpe':<12} {'Optimal Return':<15} {'Optimal Vol':<12} {'Asset Allocation'}")
    print("-" * 80)
    
    for rf_rate in risk_free_rates:
        # Your original code applied to each risk-free rate
        sharpe_arr = (portfolio_returns - rf_rate) / portfolio_volatility
        max_sr_idx = sharpe_arr.argmax()
        optimal_weights = simulation_results[max_sr_idx]
        optimal_return = portfolio_returns[max_sr_idx]
        optimal_volatility = portfolio_volatility[max_sr_idx]
        MC_SR = sharpe_arr[max_sr_idx]
        
        allocation_str = f"{optimal_weights[0]:.1%}/{optimal_weights[1]:.1%}"
        
        print(f"{rf_rate:>10.1%} {MC_SR:>10.3f} {optimal_return:>13.2%} {optimal_volatility:>10.2%} {allocation_str:>15}")
    
    print(f"\n💡 Insight: Higher risk-free rates require higher returns to achieve same Sharpe ratio!")


def main():
    """Main demonstration function."""
    print("🚀 YOUR MONTE CARLO CODE - COMPLETE INTEGRATION")
    print("="*60)
    
    print(f"This demonstration shows how your original Monte Carlo")
    print(f"portfolio optimization code has been enhanced and integrated")
    print(f"into our comprehensive trading and portfolio system.\n")
    
    # Run your original code demonstration
    results = demonstrate_your_original_code()
    
    if results:
        # Show sensitivity analysis
        compare_with_different_risk_free_rates()
        
        print(f"\n🎉 SUCCESS!")
        print("="*60)
        print(f"✅ Your original code variables successfully created")
        print(f"✅ Monte Carlo optimization completed") 
        print(f"✅ Maximum Sharpe ratio portfolio identified")
        print(f"✅ Optimal weights calculated and displayed")
        print(f"✅ Annualized Sharpe ratio computed")
        print(f"✅ Additional analysis and visualization provided")
        print(f"✅ Sensitivity analysis demonstrated")
        
        print(f"\n🎯 Your code is now part of a complete system that includes:")
        print(f"   • Real market data integration")
        print(f"   • Algorithm backtesting framework")
        print(f"   • Multiple optimization approaches")
        print(f"   • Comprehensive visualization")
        print(f"   • Educational and research capabilities")


if __name__ == "__main__":
    main()
