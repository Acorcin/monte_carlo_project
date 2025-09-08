"""
Portfolio Analysis with Statistical Monte Carlo

This script demonstrates your exact portfolio analysis code integrated
with the statistical Monte Carlo enhancement.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portfolio_optimization import PortfolioOptimizer

def demonstrate_portfolio_analysis():
    """
    Demonstrate your exact portfolio analysis code with statistical Monte Carlo.
    """
    print("🎯 PORTFOLIO ANALYSIS WITH STATISTICAL MONTE CARLO")
    print("="*60)
    print("Your exact analysis code integrated with enhanced sampling")
    print()
    
    # Setup with real market data
    assets = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    optimizer = PortfolioOptimizer(assets, period="6mo")
    
    # Fetch data and calculate statistics
    print("📊 Fetching market data and running statistical Monte Carlo...")
    price_data = optimizer.fetch_data()
    optimizer.calculate_returns_statistics(price_data)
    
    # Run statistical Monte Carlo optimization
    results = optimizer.monte_carlo_optimization(
        num_simulations=5000,
        method="statistical"  # Enhanced with 2σ constraints
    )
    
    # Extract your variables from the results
    simulation_results = results['weights']
    portfolio_returns = results['returns']
    portfolio_volatility = results['volatility']
    
    print(f"✅ Generated {len(portfolio_returns)} statistical portfolio simulations")
    print(f"✅ Assets: {', '.join(assets)}")
    print()
    
    # =============================================================
    # YOUR EXACT CODE STARTS HERE
    # =============================================================
    
    print("🔥 EXECUTING YOUR EXACT ANALYSIS CODE:")
    print("-" * 50)
    
    # Calculate the Sharpe ratio for each portfolio
    risk_free_rate = 0.03  # Replace with your risk-free rate
    sharpe_arr = (portfolio_returns - risk_free_rate) / portfolio_volatility
    
    # Find the index of the maximum Sharpe ratio
    max_sr_idx = sharpe_arr.argmax()
    
    # Retrieve the optimal weights and corresponding return and volatility
    optimal_weights = simulation_results[max_sr_idx]
    optimal_return = portfolio_returns[max_sr_idx]
    optimal_volatility = portfolio_volatility[max_sr_idx]
    
    # Calculate the Sharpe ratio at the maximum point
    MC_SR = sharpe_arr[max_sr_idx]
    
    # Calculate the annualized Sharpe ratio
    SR_annualized = MC_SR * np.sqrt(12)  # Assuming monthly data, annualize by sqrt(12)
    
    print("Optimal Portfolio Weights:", optimal_weights)
    print("Optimal Portfolio Return:", optimal_return)
    print("Optimal Portfolio Volatility:", optimal_volatility)
    print("Max Sharpe Ratio:", MC_SR)
    print("Max Annualized Sharpe Ratio:", SR_annualized)
    
    # =============================================================
    # YOUR EXACT CODE ENDS HERE
    # =============================================================
    
    print("\n✅ Your analysis code executed successfully!")
    
    # Enhanced analysis with the statistical results
    print(f"\n📊 ENHANCED ANALYSIS WITH STATISTICAL SAMPLING:")
    print("=" * 60)
    
    # Detailed asset allocation breakdown
    print(f"🎯 OPTIMAL PORTFOLIO BREAKDOWN:")
    print("-" * 40)
    print(f"Risk-Free Rate:        {risk_free_rate:.1%}")
    print(f"Simulation Method:     Statistical (2σ constraints)")
    print(f"Portfolio Index:       {max_sr_idx:,} out of {len(portfolio_returns):,}")
    print()
    
    print(f"📈 PERFORMANCE METRICS:")
    print("-" * 30)
    print(f"Expected Return:       {optimal_return:.2%}")
    print(f"Portfolio Volatility:  {optimal_volatility:.2%}")
    print(f"Sharpe Ratio:          {MC_SR:.4f}")
    print(f"Annualized Sharpe:     {SR_annualized:.4f}")
    print(f"Excess Return:         {(optimal_return - risk_free_rate):.2%}")
    print()
    
    print(f"💼 ASSET ALLOCATION:")
    print("-" * 25)
    for i, asset in enumerate(assets):
        weight = optimal_weights[i]
        print(f"  {asset:>6}: {weight:>8.1%}")
    
    # Statistical insights
    print(f"\n📊 STATISTICAL MONTE CARLO INSIGHTS:")
    print("-" * 45)
    print(f"Sharpe Ratio Distribution:")
    print(f"  Mean:        {sharpe_arr.mean():.4f}")
    print(f"  Median:      {np.median(sharpe_arr):.4f}")
    print(f"  Std Dev:     {sharpe_arr.std():.4f}")
    print(f"  Min:         {sharpe_arr.min():.4f}")
    print(f"  Max:         {sharpe_arr.max():.4f}")
    print(f"  Range:       {sharpe_arr.max() - sharpe_arr.min():.4f}")
    
    # Percentile analysis
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\nSharpe Ratio Percentiles:")
    for p in percentiles:
        value = np.percentile(sharpe_arr, p)
        print(f"  {p:2d}th: {value:.4f}")
    
    # Portfolio quality analysis
    print(f"\nPortfolio Quality Distribution:")
    excellent_count = (sharpe_arr >= 2.0).sum()
    very_good_count = ((sharpe_arr >= 1.5) & (sharpe_arr < 2.0)).sum()
    good_count = ((sharpe_arr >= 1.0) & (sharpe_arr < 1.5)).sum()
    moderate_count = ((sharpe_arr >= 0.5) & (sharpe_arr < 1.0)).sum()
    poor_count = (sharpe_arr < 0.5).sum()
    
    total_portfolios = len(sharpe_arr)
    print(f"  Excellent (≥2.0):  {excellent_count:5,} ({excellent_count/total_portfolios*100:5.1f}%)")
    print(f"  Very Good (1.5-2): {very_good_count:5,} ({very_good_count/total_portfolios*100:5.1f}%)")
    print(f"  Good (1.0-1.5):    {good_count:5,} ({good_count/total_portfolios*100:5.1f}%)")
    print(f"  Moderate (0.5-1):  {moderate_count:5,} ({moderate_count/total_portfolios*100:5.1f}%)")
    print(f"  Poor (<0.5):       {poor_count:5,} ({poor_count/total_portfolios*100:5.1f}%)")
    
    return {
        'optimal_weights': optimal_weights,
        'optimal_return': optimal_return,
        'optimal_volatility': optimal_volatility,
        'max_sharpe_ratio': MC_SR,
        'annualized_sharpe': SR_annualized,
        'sharpe_array': sharpe_arr,
        'assets': assets,
        'risk_free_rate': risk_free_rate
    }


def create_portfolio_visualization(analysis_results):
    """Create comprehensive portfolio analysis visualization."""
    print(f"\n📊 CREATING PORTFOLIO ANALYSIS VISUALIZATION...")
    
    # Extract results
    optimal_weights = analysis_results['optimal_weights']
    sharpe_arr = analysis_results['sharpe_array']
    assets = analysis_results['assets']
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Optimal Portfolio Allocation Pie Chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(assets)))
    wedges, texts, autotexts = ax1.pie(optimal_weights, labels=assets, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    ax1.set_title(f'Optimal Portfolio Allocation\n(Sharpe Ratio: {analysis_results["max_sharpe_ratio"]:.3f})')
    
    # 2. Sharpe Ratio Distribution Histogram
    ax2.hist(sharpe_arr, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(analysis_results['max_sharpe_ratio'], color='red', linestyle='-', 
               linewidth=3, label=f'Max: {analysis_results["max_sharpe_ratio"]:.3f}')
    ax2.axvline(sharpe_arr.mean(), color='orange', linestyle='--', 
               label=f'Mean: {sharpe_arr.mean():.3f}')
    ax2.set_xlabel('Sharpe Ratio')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Sharpe Ratio Distribution (Statistical Monte Carlo)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Asset Allocation Bar Chart
    x_pos = np.arange(len(assets))
    bars = ax3.bar(x_pos, optimal_weights, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Assets')
    ax3.set_ylabel('Weight')
    ax3.set_title('Optimal Portfolio Weights')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(assets)
    ax3.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, weight in zip(bars, optimal_weights):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{weight:.1%}', ha='center', va='bottom')
    
    # 4. Cumulative Distribution of Sharpe Ratios
    sorted_sharpe = np.sort(sharpe_arr)
    cumulative_prob = np.arange(1, len(sorted_sharpe) + 1) / len(sorted_sharpe)
    ax4.plot(sorted_sharpe, cumulative_prob, linewidth=2, color='green')
    ax4.axvline(analysis_results['max_sharpe_ratio'], color='red', linestyle='--',
               label=f'Max Sharpe: {analysis_results["max_sharpe_ratio"]:.3f}')
    ax4.axhline(0.95, color='gray', linestyle=':', alpha=0.7, label='95th Percentile')
    ax4.set_xlabel('Sharpe Ratio')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Distribution of Sharpe Ratios')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Portfolio analysis visualization completed")


def compare_annualization_methods(analysis_results):
    """Compare different annualization approaches."""
    print(f"\n🔄 ANNUALIZATION METHOD COMPARISON:")
    print("=" * 45)
    
    MC_SR = analysis_results['max_sharpe_ratio']
    optimal_return = analysis_results['optimal_return']
    optimal_volatility = analysis_results['optimal_volatility']
    risk_free_rate = analysis_results['risk_free_rate']
    
    # Different annualization factors
    methods = {
        'Monthly (√12)': np.sqrt(12),
        'Daily (√252)': np.sqrt(252),
        'Weekly (√52)': np.sqrt(52),
        'Quarterly (√4)': np.sqrt(4)
    }
    
    print(f"Base Sharpe Ratio: {MC_SR:.4f}")
    print(f"Risk-Free Rate: {risk_free_rate:.1%}")
    print()
    
    for method_name, factor in methods.items():
        annualized_sharpe = MC_SR * factor
        print(f"{method_name:15}: {annualized_sharpe:8.4f}")
    
    print(f"\n💡 Note: Your code uses √12 (monthly), which gives: {MC_SR * np.sqrt(12):.4f}")
    print(f"   This assumes the input data represents monthly returns.")


def main():
    """Main demonstration function."""
    print("🚀 PORTFOLIO ANALYSIS WITH STATISTICAL MONTE CARLO")
    print("="*60)
    print("Your exact portfolio analysis code enhanced with statistical sampling\n")
    
    # Run the portfolio analysis
    analysis_results = demonstrate_portfolio_analysis()
    
    if analysis_results:
        # Create visualizations
        create_portfolio_visualization(analysis_results)
        
        # Compare annualization methods
        compare_annualization_methods(analysis_results)
        
        print(f"\n🎉 PORTFOLIO ANALYSIS COMPLETE!")
        print("="*50)
        print(f"✅ Your exact analysis code executed successfully")
        print(f"✅ Statistical Monte Carlo with 2σ constraints applied")
        print(f"✅ Enhanced portfolio insights generated")
        print(f"✅ Comprehensive visualization created")
        print(f"✅ Annualization comparison provided")
        
        print(f"\nKey Results:")
        print(f"• Optimal Sharpe Ratio: {analysis_results['max_sharpe_ratio']:.4f}")
        print(f"• Annualized Sharpe: {analysis_results['annualized_sharpe']:.4f}")
        print(f"• Expected Return: {analysis_results['optimal_return']:.2%}")
        print(f"• Portfolio Risk: {analysis_results['optimal_volatility']:.2%}")
        print(f"• Statistical sampling used {len(analysis_results['sharpe_array']):,} simulations")


if __name__ == "__main__":
    main()
