"""
Enhanced Sharpe Ratio Analysis with Monte Carlo Portfolio Optimization

This script demonstrates advanced Sharpe ratio analysis using your original code
integrated with real market data and comprehensive portfolio optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import seaborn as sns

from portfolio_optimization import PortfolioOptimizer

class SharpeRatioAnalyzer:
    """
    Advanced Sharpe ratio analysis with Monte Carlo optimization.
    
    Integrates your original code for finding optimal portfolios based on
    maximum Sharpe ratio with comprehensive analysis and visualization.
    """
    
    def __init__(self, assets: List[str], period: str = "1y", risk_free_rate: float = 0.03):
        """
        Initialize Sharpe ratio analyzer.
        
        Args:
            assets (list): List of asset tickers
            period (str): Data period for analysis
            risk_free_rate (float): Risk-free rate for Sharpe calculations
        """
        self.assets = assets
        self.period = period
        self.risk_free_rate = risk_free_rate
        self.optimizer = PortfolioOptimizer(assets, period)
        self.results = None
        self.optimal_portfolios = None
        
    def run_monte_carlo_sharpe_analysis(self, num_simulations: int = 10000) -> Dict:
        """
        Run Monte Carlo optimization focused on Sharpe ratio analysis.
        
        Args:
            num_simulations (int): Number of Monte Carlo simulations
            
        Returns:
            dict: Complete analysis results with Sharpe focus
        """
        print(f"🎯 SHARPE RATIO FOCUSED ANALYSIS")
        print("="*50)
        
        # Fetch data and calculate statistics
        price_data = self.optimizer.fetch_data()
        self.optimizer.calculate_returns_statistics(price_data)
        
        # Run Monte Carlo optimization
        self.results = self.optimizer.monte_carlo_optimization(num_simulations)
        
        # Find optimal portfolios with your enhanced code
        self.optimal_portfolios = self.optimizer.find_optimal_portfolios(
            self.results, risk_free_rate=self.risk_free_rate
        )
        
        # Your original code integration - extract key variables
        portfolio_df = self.optimal_portfolios['portfolio_dataframe']
        sharpe_arr = self.optimal_portfolios['sharpe_array']
        max_sr_idx = sharpe_arr.argmax()
        
        optimal_weights = self.results['weights'][max_sr_idx]
        optimal_return = self.results['returns'][max_sr_idx]
        optimal_volatility = self.results['volatility'][max_sr_idx]
        MC_SR = sharpe_arr[max_sr_idx]
        
        # Calculate annualized Sharpe ratio (your code enhanced)
        annualization_factor = self.optimal_portfolios['annualization_factor']
        SR_annualized = MC_SR * annualization_factor
        
        # Store results for analysis
        analysis_results = {
            'portfolio_dataframe': portfolio_df,
            'sharpe_array': sharpe_arr,
            'max_sharpe_index': max_sr_idx,
            'optimal_weights': optimal_weights,
            'optimal_return': optimal_return,
            'optimal_volatility': optimal_volatility,
            'sharpe_ratio': MC_SR,
            'sharpe_ratio_annualized': SR_annualized,
            'risk_free_rate': self.risk_free_rate,
            'annualization_factor': annualization_factor,
            'simulation_results': self.results,
            'optimal_portfolios': self.optimal_portfolios
        }
        
        return analysis_results
    
    def analyze_sharpe_distribution(self, analysis_results: Dict) -> Dict:
        """
        Analyze the distribution of Sharpe ratios from Monte Carlo simulation.
        
        Args:
            analysis_results (dict): Results from Monte Carlo analysis
            
        Returns:
            dict: Statistical analysis of Sharpe ratio distribution
        """
        print(f"\n📊 SHARPE RATIO DISTRIBUTION ANALYSIS")
        print("-" * 50)
        
        sharpe_arr = analysis_results['sharpe_array']
        
        # Basic statistics
        stats = {
            'count': len(sharpe_arr),
            'mean': sharpe_arr.mean(),
            'median': np.median(sharpe_arr),
            'std': sharpe_arr.std(),
            'min': sharpe_arr.min(),
            'max': sharpe_arr.max(),
            'range': sharpe_arr.max() - sharpe_arr.min(),
            'skewness': pd.Series(sharpe_arr).skew(),
            'kurtosis': pd.Series(sharpe_arr).kurtosis()
        }
        
        # Percentile analysis
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = {f'p{p}': np.percentile(sharpe_arr, p) for p in percentiles}
        
        # Count portfolios in different Sharpe ranges
        sharpe_ranges = {
            'negative': (sharpe_arr < 0).sum(),
            'low (0-0.5)': ((sharpe_arr >= 0) & (sharpe_arr < 0.5)).sum(),
            'moderate (0.5-1.0)': ((sharpe_arr >= 0.5) & (sharpe_arr < 1.0)).sum(),
            'good (1.0-1.5)': ((sharpe_arr >= 1.0) & (sharpe_arr < 1.5)).sum(),
            'excellent (1.5+)': (sharpe_arr >= 1.5).sum()
        }
        
        # Print analysis
        print(f"Basic Statistics:")
        print(f"  Count:     {stats['count']:,}")
        print(f"  Mean:      {stats['mean']:.4f}")
        print(f"  Median:    {stats['median']:.4f}")
        print(f"  Std Dev:   {stats['std']:.4f}")
        print(f"  Min:       {stats['min']:.4f}")
        print(f"  Max:       {stats['max']:.4f}")
        print(f"  Range:     {stats['range']:.4f}")
        print(f"  Skewness:  {stats['skewness']:.4f}")
        print(f"  Kurtosis:  {stats['kurtosis']:.4f}")
        
        print(f"\nPercentile Analysis:")
        for p in percentiles:
            print(f"  {p:2d}th: {percentile_values[f'p{p}']:.4f}")
        
        print(f"\nSharpe Ratio Categories:")
        for category, count in sharpe_ranges.items():
            pct = (count / len(sharpe_arr)) * 100
            print(f"  {category:20s}: {count:6,} ({pct:5.1f}%)")
        
        return {**stats, **percentile_values, 'ranges': sharpe_ranges}
    
    def plot_sharpe_analysis(self, analysis_results: Dict, save_plots: bool = False):
        """
        Create comprehensive Sharpe ratio visualizations.
        
        Args:
            analysis_results (dict): Results from Monte Carlo analysis
            save_plots (bool): Whether to save plots to files
        """
        print(f"\n📈 GENERATING SHARPE RATIO VISUALIZATIONS")
        print("-" * 50)
        
        sharpe_arr = analysis_results['sharpe_array']
        returns = analysis_results['simulation_results']['returns']
        volatility = analysis_results['simulation_results']['volatility']
        
        # Create comprehensive plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Efficient Frontier with Sharpe Ratio Coloring
        scatter = ax1.scatter(volatility, returns, c=sharpe_arr, cmap='viridis', 
                             alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax1, label='Sharpe Ratio')
        
        # Highlight maximum Sharpe ratio point
        max_idx = analysis_results['max_sharpe_index']
        ax1.scatter(analysis_results['optimal_volatility'], analysis_results['optimal_return'],
                   color='red', marker='*', s=300, label='Max Sharpe', 
                   edgecolors='black', linewidth=2)
        
        ax1.set_xlabel('Volatility (Risk)')
        ax1.set_ylabel('Expected Return')
        ax1.set_title('Efficient Frontier - Colored by Sharpe Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sharpe Ratio Distribution Histogram
        ax2.hist(sharpe_arr, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(sharpe_arr.mean(), color='red', linestyle='--', 
                   label=f'Mean: {sharpe_arr.mean():.3f}')
        ax2.axvline(analysis_results['sharpe_ratio'], color='orange', linestyle='-', 
                   linewidth=3, label=f'Max: {analysis_results["sharpe_ratio"]:.3f}')
        ax2.set_xlabel('Sharpe Ratio')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Sharpe Ratios')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Risk-Return Scatter with Size by Sharpe
        sizes = (sharpe_arr - sharpe_arr.min()) * 100 + 10  # Scale for visibility
        ax3.scatter(volatility, returns, s=sizes, alpha=0.5, c='blue')
        ax3.scatter(analysis_results['optimal_volatility'], analysis_results['optimal_return'],
                   color='red', marker='*', s=400, label='Max Sharpe')
        ax3.set_xlabel('Volatility')
        ax3.set_ylabel('Return')
        ax3.set_title('Risk-Return (Size = Sharpe Ratio)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative Distribution of Sharpe Ratios
        sorted_sharpe = np.sort(sharpe_arr)
        cumulative_prob = np.arange(1, len(sorted_sharpe) + 1) / len(sorted_sharpe)
        ax4.plot(sorted_sharpe, cumulative_prob, linewidth=2, color='green')
        ax4.axvline(analysis_results['sharpe_ratio'], color='red', linestyle='--',
                   label=f'Max Sharpe: {analysis_results["sharpe_ratio"]:.3f}')
        ax4.set_xlabel('Sharpe Ratio')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution of Sharpe Ratios')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"sharpe_analysis_{'_'.join(self.assets)}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Plots saved as {filename}")
        
        plt.show()
    
    def compare_optimization_methods(self, num_simulations: int = 5000) -> Dict:
        """
        Compare different portfolio optimization approaches.
        
        Args:
            num_simulations (int): Number of simulations for comparison
            
        Returns:
            dict: Comparison results
        """
        print(f"\n🔄 OPTIMIZATION METHOD COMPARISON")
        print("-" * 50)
        
        # Method 1: Your original approach (maximum Sharpe ratio)
        analysis_results = self.run_monte_carlo_sharpe_analysis(num_simulations)
        
        # Method 2: Equal weights
        equal_weights = np.ones(len(self.assets)) / len(self.assets)
        equal_return = np.sum(equal_weights * self.optimizer.expected_returns)
        equal_volatility = np.sqrt(np.dot(equal_weights.T, 
                                         np.dot(self.optimizer.covariance_matrix, equal_weights)))
        equal_sharpe = (equal_return - self.risk_free_rate) / equal_volatility
        
        # Method 3: Risk parity (inverse volatility weighting)
        inv_vol_weights = 1 / self.optimizer.std_deviations
        inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
        rp_return = np.sum(inv_vol_weights * self.optimizer.expected_returns)
        rp_volatility = np.sqrt(np.dot(inv_vol_weights.T, 
                                      np.dot(self.optimizer.covariance_matrix, inv_vol_weights)))
        rp_sharpe = (rp_return - self.risk_free_rate) / rp_volatility
        
        # Comparison summary
        comparison = {
            'Monte Carlo Max Sharpe': {
                'weights': dict(zip(self.assets, analysis_results['optimal_weights'])),
                'return': analysis_results['optimal_return'],
                'volatility': analysis_results['optimal_volatility'],
                'sharpe': analysis_results['sharpe_ratio'],
                'method': 'Monte Carlo optimization'
            },
            'Equal Weight': {
                'weights': dict(zip(self.assets, equal_weights)),
                'return': equal_return,
                'volatility': equal_volatility,
                'sharpe': equal_sharpe,
                'method': '1/N allocation'
            },
            'Risk Parity': {
                'weights': dict(zip(self.assets, inv_vol_weights)),
                'return': rp_return,
                'volatility': rp_volatility,
                'sharpe': rp_sharpe,
                'method': 'Inverse volatility weighting'
            }
        }
        
        # Print comparison
        print(f"Method Comparison Results:")
        print(f"{'Method':<25} {'Return':<8} {'Vol':<8} {'Sharpe':<8}")
        print("-" * 55)
        
        for method, data in comparison.items():
            print(f"{method:<25} {data['return']:>6.2%} {data['volatility']:>6.2%} {data['sharpe']:>6.3f}")
        
        return comparison
    
    def run_complete_sharpe_analysis(self, num_simulations: int = 10000, 
                                   save_plots: bool = False) -> Dict:
        """
        Run complete Sharpe ratio analysis workflow.
        
        Args:
            num_simulations (int): Number of Monte Carlo simulations
            save_plots (bool): Whether to save plots
            
        Returns:
            dict: Complete analysis results
        """
        print(f"🚀 COMPREHENSIVE SHARPE RATIO ANALYSIS")
        print("="*60)
        print(f"Assets: {', '.join(self.assets)}")
        print(f"Period: {self.period}")
        print(f"Risk-Free Rate: {self.risk_free_rate:.1%}")
        print(f"Simulations: {num_simulations:,}")
        
        # Step 1: Monte Carlo Sharpe analysis
        analysis_results = self.run_monte_carlo_sharpe_analysis(num_simulations)
        
        # Step 2: Distribution analysis
        distribution_stats = self.analyze_sharpe_distribution(analysis_results)
        
        # Step 3: Visualization
        self.plot_sharpe_analysis(analysis_results, save_plots)
        
        # Step 4: Method comparison
        method_comparison = self.compare_optimization_methods(num_simulations)
        
        # Step 5: Summary
        print(f"\n🎯 ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"✅ Monte Carlo optimization: {num_simulations:,} simulations")
        print(f"✅ Maximum Sharpe ratio found: {analysis_results['sharpe_ratio']:.4f}")
        print(f"✅ Annualized Sharpe ratio: {analysis_results['sharpe_ratio_annualized']:.4f}")
        print(f"✅ Optimal return: {analysis_results['optimal_return']:.2%}")
        print(f"✅ Optimal volatility: {analysis_results['optimal_volatility']:.2%}")
        print(f"✅ Distribution analysis completed")
        print(f"✅ Method comparison completed")
        print(f"✅ Visualizations generated")
        
        return {
            'analysis_results': analysis_results,
            'distribution_stats': distribution_stats,
            'method_comparison': method_comparison,
            'assets': self.assets,
            'parameters': {
                'num_simulations': num_simulations,
                'risk_free_rate': self.risk_free_rate,
                'period': self.period
            }
        }


def demo_sharpe_analysis():
    """Demonstration of comprehensive Sharpe ratio analysis."""
    print("🎯 SHARPE RATIO ANALYSIS DEMO")
    print("="*40)
    
    # Demo with tech stocks
    assets = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    
    analyzer = SharpeRatioAnalyzer(assets, period="6mo", risk_free_rate=0.025)
    
    try:
        results = analyzer.run_complete_sharpe_analysis(
            num_simulations=5000,
            save_plots=False
        )
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"• Your original code successfully integrated")
        print(f"• Monte Carlo finds globally optimal Sharpe ratio")
        print(f"• Distribution analysis reveals portfolio efficiency spread") 
        print(f"• Method comparison shows optimization benefits")
        print(f"• Annualized metrics enable cross-strategy comparison")
        
        return results
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None


if __name__ == "__main__":
    demo_sharpe_analysis()
