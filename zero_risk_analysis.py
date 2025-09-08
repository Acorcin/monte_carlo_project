"""
Zero Risk-Free Rate Analysis

This script implements your exact plotting code with zero risk-free rate assumption
and provides additional analysis for this simplified scenario.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portfolio_optimization import PortfolioOptimizer

def run_zero_risk_analysis(assets=['AAPL', 'MSFT', 'GOOGL'], num_simulations=5000):
    """
    Run portfolio optimization with zero risk-free rate assumption.
    
    Args:
        assets (list): List of asset tickers
        num_simulations (int): Number of Monte Carlo simulations
        
    Returns:
        dict: Analysis results with zero risk-free rate
    """
    print("🎯 ZERO RISK-FREE RATE ANALYSIS")
    print("="*50)
    print("Simplifying assumption: risk_free_rate = 0")
    print(f"Assets: {', '.join(assets)}")
    print(f"Simulations: {num_simulations:,}\n")
    
    # Setup optimizer
    optimizer = PortfolioOptimizer(assets, period="6mo")
    
    # Fetch data and calculate statistics
    price_data = optimizer.fetch_data()
    optimizer.calculate_returns_statistics(price_data)
    
    # Run Monte Carlo optimization with zero risk-free rate
    results = optimizer.monte_carlo_optimization(num_simulations)
    
    # =============================================================
    # YOUR ORIGINAL CODE STARTS HERE
    # =============================================================
    
    # Simplifying assumption, risk free rate is zero, for sharpe ratio
    risk_free_rate = 0
    
    # Extract your variables
    portfolio_returns = results['returns']
    portfolio_volatility = results['volatility']
    simulation_results = results['weights']
    
    # Create DataFrame from simulation results (your code)
    portfolio_df = pd.DataFrame({'Return': portfolio_returns, 'Volatility': portfolio_volatility})
    
    # Calculate Sharpe ratio with zero risk-free rate
    sharpe_arr = (portfolio_returns - risk_free_rate) / portfolio_volatility
    max_sr_idx = sharpe_arr.argmax()
    
    # Get optimal portfolio characteristics
    optimal_weights = simulation_results[max_sr_idx]
    optimal_return = portfolio_returns[max_sr_idx]
    optimal_volatility = portfolio_volatility[max_sr_idx]
    MC_SR = sharpe_arr[max_sr_idx]
    
    print("📊 ZERO RISK-FREE RATE RESULTS:")
    print("-" * 40)
    print(f"Risk-Free Rate:        {risk_free_rate:.1%}")
    print(f"Maximum Sharpe Ratio:  {MC_SR:.4f}")
    print(f"Optimal Return:        {optimal_return:.2%}")
    print(f"Optimal Volatility:    {optimal_volatility:.2%}")
    print(f"Simulation Index:      {max_sr_idx:,}")
    
    print(f"\nOptimal Asset Allocation:")
    for i, asset in enumerate(assets):
        print(f"  {asset}: {optimal_weights[i]:>8.1%}")
    
    # Plot the Monte Carlo efficient frontier (your exact code)
    plt.figure(figsize=(12, 6))
    plt.scatter(portfolio_df['Volatility'], portfolio_df['Return'], 
               c=(portfolio_df['Return']-risk_free_rate) / portfolio_df['Volatility'], 
               marker='o')  
    plt.title('Monte Carlo Efficient Frontier')
    plt.xlabel('Portfolio Volatility')
    plt.ylabel('Portfolio Return')
    plt.colorbar(label='Sharpe Ratio')
    
    # Add a red dot for the optimal portfolio (your exact code)
    plt.scatter(optimal_volatility, optimal_return, color='red', marker='o', 
               s=100, label='Optimal Portfolio')
    
    # Show legend (your exact code)
    plt.legend()
    plt.show()
    
    # =============================================================
    # YOUR ORIGINAL CODE ENDS HERE  
    # =============================================================
    
    return {
        'portfolio_df': portfolio_df,
        'sharpe_arr': sharpe_arr,
        'max_sr_idx': max_sr_idx,
        'optimal_weights': optimal_weights,
        'optimal_return': optimal_return,
        'optimal_volatility': optimal_volatility,
        'MC_SR': MC_SR,
        'risk_free_rate': risk_free_rate,
        'assets': assets,
        'simulation_results': simulation_results
    }


def compare_risk_free_scenarios(assets=['AAPL', 'MSFT', 'GOOGL'], num_simulations=3000):
    """
    Compare zero risk-free rate vs non-zero scenarios.
    
    Args:
        assets (list): List of asset tickers
        num_simulations (int): Number of simulations
    """
    print("\n🔄 RISK-FREE RATE COMPARISON ANALYSIS")
    print("="*50)
    
    # Setup optimizer
    optimizer = PortfolioOptimizer(assets, period="3mo")
    price_data = optimizer.fetch_data()
    optimizer.calculate_returns_statistics(price_data)
    results = optimizer.monte_carlo_optimization(num_simulations)
    
    # Test different risk-free rates including zero
    risk_free_rates = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    
    print("Comparing different risk-free rate assumptions:")
    print(f"{'Risk-Free':<12} {'Max Sharpe':<12} {'Optimal Return':<15} {'Optimal Vol':<12} {'Return/Vol Ratio':<15}")
    print("-" * 85)
    
    comparison_results = {}
    
    for rf_rate in risk_free_rates:
        # Calculate Sharpe ratio for this risk-free rate
        portfolio_returns = results['returns']
        portfolio_volatility = results['volatility']
        simulation_results = results['weights']
        
        sharpe_arr = (portfolio_returns - rf_rate) / portfolio_volatility
        max_sr_idx = sharpe_arr.argmax()
        
        optimal_weights = simulation_results[max_sr_idx]
        optimal_return = portfolio_returns[max_sr_idx]
        optimal_volatility = portfolio_volatility[max_sr_idx]
        MC_SR = sharpe_arr[max_sr_idx]
        
        # Simple return/volatility ratio for comparison
        return_vol_ratio = optimal_return / optimal_volatility
        
        print(f"{rf_rate:>10.1%} {MC_SR:>10.4f} {optimal_return:>13.2%} {optimal_volatility:>10.2%} {return_vol_ratio:>13.4f}")
        
        comparison_results[rf_rate] = {
            'sharpe_ratio': MC_SR,
            'optimal_return': optimal_return,
            'optimal_volatility': optimal_volatility,
            'optimal_weights': optimal_weights,
            'return_vol_ratio': return_vol_ratio
        }
    
    # Special focus on zero risk-free rate
    zero_rf_results = comparison_results[0.0]
    print(f"\n🎯 ZERO RISK-FREE RATE INSIGHTS:")
    print("-" * 40)
    print(f"• With zero risk-free rate, Sharpe ratio = Return / Volatility")
    print(f"• Maximum ratio achieved: {zero_rf_results['return_vol_ratio']:.4f}")
    print(f"• This represents pure return-to-risk efficiency")
    print(f"• No penalty for 'risk-free' opportunity cost")
    
    return comparison_results


def enhanced_zero_risk_visualization(assets=['AAPL', 'MSFT', 'GOOGL'], num_simulations=4000):
    """
    Create enhanced visualizations for zero risk-free rate analysis.
    
    Args:
        assets (list): List of asset tickers
        num_simulations (int): Number of simulations
    """
    print(f"\n📊 ENHANCED ZERO RISK-FREE RATE VISUALIZATION")
    print("="*55)
    
    # Run analysis
    results = run_zero_risk_analysis(assets, num_simulations)
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    portfolio_df = results['portfolio_df']
    optimal_return = results['optimal_return']
    optimal_volatility = results['optimal_volatility']
    sharpe_arr = results['sharpe_arr']
    
    # 1. Your original plot (enhanced)
    scatter = ax1.scatter(portfolio_df['Volatility'], portfolio_df['Return'], 
                         c=portfolio_df['Return'] / portfolio_df['Volatility'], 
                         marker='o', alpha=0.6, s=20, cmap='viridis')
    ax1.scatter(optimal_volatility, optimal_return, color='red', marker='o', 
               s=200, label='Optimal Portfolio', edgecolors='black', linewidth=2)
    ax1.set_title('Monte Carlo Efficient Frontier (Zero Risk-Free Rate)')
    ax1.set_xlabel('Portfolio Volatility')
    ax1.set_ylabel('Portfolio Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Return/Volatility Ratio')
    
    # 2. Sharpe ratio distribution
    ax2.hist(sharpe_arr, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(results['MC_SR'], color='red', linestyle='-', linewidth=3,
               label=f'Max Sharpe: {results["MC_SR"]:.3f}')
    ax2.axvline(sharpe_arr.mean(), color='orange', linestyle='--',
               label=f'Mean: {sharpe_arr.mean():.3f}')
    ax2.set_xlabel('Sharpe Ratio (Zero Risk-Free Rate)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Return/Volatility Ratios')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Return vs Volatility scatter with size
    sizes = (sharpe_arr - sharpe_arr.min()) * 100 + 10
    ax3.scatter(portfolio_df['Volatility'], portfolio_df['Return'], 
               s=sizes, alpha=0.5, c='blue')
    ax3.scatter(optimal_volatility, optimal_return, color='red', marker='*', 
               s=400, label='Optimal Portfolio')
    ax3.set_xlabel('Volatility')
    ax3.set_ylabel('Return')
    ax3.set_title('Risk-Return (Size = Sharpe Ratio)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Optimal weights pie chart
    weights = results['optimal_weights']
    ax4.pie(weights, labels=assets, autopct='%1.1f%%', startangle=90)
    ax4.set_title(f'Optimal Portfolio Allocation\n(Sharpe: {results["MC_SR"]:.3f})')
    
    plt.tight_layout()
    plt.show()
    
    # Additional statistics
    print(f"\n📈 ZERO RISK-FREE RATE STATISTICS:")
    print("-" * 45)
    print(f"Sharpe Ratio Range:    {sharpe_arr.min():.4f} to {sharpe_arr.max():.4f}")
    print(f"Mean Sharpe Ratio:     {sharpe_arr.mean():.4f}")
    print(f"Sharpe Std Deviation:  {sharpe_arr.std():.4f}")
    print(f"Median Sharpe Ratio:   {np.median(sharpe_arr):.4f}")
    
    # Percentile analysis
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\nSharpe Ratio Percentiles:")
    for p in percentiles:
        value = np.percentile(sharpe_arr, p)
        print(f"  {p:2d}th: {value:.4f}")
    
    return results


def theoretical_analysis():
    """
    Provide theoretical insights about zero risk-free rate assumption.
    """
    print(f"\n🎓 THEORETICAL ANALYSIS: ZERO RISK-FREE RATE")
    print("="*55)
    
    print(f"📚 Mathematical Implications:")
    print(f"• Standard Sharpe Ratio: (Return - Risk_Free_Rate) / Volatility")
    print(f"• Zero Risk-Free Sharpe:  Return / Volatility")
    print(f"• This becomes a pure return-to-risk efficiency measure")
    print(f"• No opportunity cost adjustment for risk-free alternatives")
    
    print(f"\n💡 Practical Interpretations:")
    print(f"• Useful for environments with near-zero interest rates")
    print(f"• Simplifies analysis when risk-free rate is negligible")
    print(f"• Focus purely on return generation per unit of risk")
    print(f"• Commonly used in academic research for simplicity")
    
    print(f"\n⚠️  Considerations:")
    print(f"• Real-world risk-free rates are rarely exactly zero")
    print(f"• May overstate portfolio attractiveness vs bonds/cash")
    print(f"• Should compare with non-zero scenarios for robustness")
    print(f"• Inflation erodes real value of zero-return alternatives")
    
    print(f"\n🎯 When to Use Zero Risk-Free Rate:")
    print(f"• Deflationary environments")
    print(f"• Very low interest rate periods")
    print(f"• Academic modeling and research")
    print(f"• Simplifying assumptions for initial analysis")
    print(f"• Focus on relative portfolio efficiency")


def main():
    """Main demonstration function."""
    print("🚀 ZERO RISK-FREE RATE PORTFOLIO OPTIMIZATION")
    print("="*55)
    print("Implementing your exact plotting code with comprehensive analysis\n")
    
    # Demo with tech stocks
    assets = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    
    try:
        # Step 1: Run your original code
        print("STEP 1: Your Original Code Implementation")
        zero_results = run_zero_risk_analysis(assets, num_simulations=3000)
        
        # Step 2: Comparison analysis
        print("\nSTEP 2: Risk-Free Rate Comparison")
        comparison = compare_risk_free_scenarios(assets[:3], num_simulations=2000)
        
        # Step 3: Enhanced visualization
        print("\nSTEP 3: Enhanced Visualization")
        enhanced_results = enhanced_zero_risk_visualization(assets[:3], num_simulations=3000)
        
        # Step 4: Theoretical insights
        print("\nSTEP 4: Theoretical Analysis")
        theoretical_analysis()
        
        print(f"\n🎉 ZERO RISK-FREE RATE ANALYSIS COMPLETE!")
        print("="*55)
        print(f"✅ Your original plotting code executed successfully")
        print(f"✅ Monte Carlo efficient frontier generated")
        print(f"✅ Optimal portfolio identified and highlighted")
        print(f"✅ Comprehensive comparison with other scenarios")
        print(f"✅ Enhanced visualizations created")
        print(f"✅ Theoretical insights provided")
        
        return {
            'zero_results': zero_results,
            'comparison': comparison,
            'enhanced_results': enhanced_results
        }
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None


if __name__ == "__main__":
    main()
