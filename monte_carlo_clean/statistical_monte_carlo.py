"""
Statistical Monte Carlo Portfolio Optimization

Enhanced Monte Carlo simulation using statistical sampling based on 
mean and standard deviation within two deviations, instead of purely random weights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import warnings
from scipy import stats

from data_fetcher import fetch_stock_data, calculate_returns_from_ohlcv

class StatisticalMonteCarloOptimizer:
    """
    Enhanced Monte Carlo Portfolio Optimizer using statistical sampling.
    
    Instead of purely random weights, this uses statistical distributions
    based on asset characteristics and samples within two standard deviations.
    """
    
    def __init__(self, assets: List[str], period: str = "6mo", interval: str = "1d"):
        """
        Initialize the statistical Monte Carlo optimizer.
        
        Args:
            assets (list): List of asset tickers
            period (str): Data period for analysis
            interval (str): Data interval
        """
        self.assets = assets
        self.period = period
        self.interval = interval
        self.returns_data = None
        self.expected_returns = None
        self.std_deviations = None
        self.correlation_matrix = None
        self.covariance_matrix = None
        
    def fetch_and_analyze_data(self) -> pd.DataFrame:
        """
        Fetch market data and calculate statistical properties.
        
        Returns:
            pd.DataFrame: Combined price data for all assets
        """
        print(f"📊 Fetching data for {len(self.assets)} assets...")
        
        all_data = {}
        
        for asset in self.assets:
            try:
                print(f"   Fetching {asset}...")
                data = fetch_stock_data(asset, period=self.period, interval=self.interval)
                all_data[asset] = data['Close']
                print(f"   ✅ {asset}: {len(data)} data points")
            except Exception as e:
                print(f"   ❌ {asset}: Failed ({e})")
                
        if not all_data:
            raise ValueError("No data could be fetched for any assets")
        
        # Combine all asset prices
        price_data = pd.DataFrame(all_data)
        print(f"✅ Combined dataset: {len(price_data)} periods for {len(price_data.columns)} assets")
        
        # Calculate statistical properties
        self.returns_data = price_data.pct_change().dropna()
        
        # Calculate expected returns and volatilities (annualized)
        trading_periods = 252 if self.interval == "1d" else 252 * 24
        self.expected_returns = self.returns_data.mean() * trading_periods
        self.std_deviations = self.returns_data.std() * np.sqrt(trading_periods)
        self.correlation_matrix = self.returns_data.corr()
        self.covariance_matrix = self.returns_data.cov() * trading_periods
        
        print("📈 Statistical properties calculated:")
        print(f"   Expected Returns: {self.expected_returns.mean():.2%} avg")
        print(f"   Volatilities: {self.std_deviations.mean():.2%} avg")
        print(f"   Return Sharpe ratios: {(self.expected_returns / self.std_deviations).mean():.3f} avg")
        
        return price_data
    
    def generate_statistical_weights(self, num_simulations: int, method: str = "normal_constrained") -> np.ndarray:
        """
        Generate portfolio weights using statistical sampling within two standard deviations.
        
        Args:
            num_simulations (int): Number of weight combinations to generate
            method (str): Statistical sampling method
            
        Returns:
            np.ndarray: Array of portfolio weights (num_simulations x num_assets)
        """
        print(f"🎲 Generating {num_simulations:,} statistical weight combinations...")
        print(f"   Method: {method}")
        print(f"   Constraint: Within 2 standard deviations of mean")
        
        num_assets = len(self.assets)
        all_weights = np.zeros((num_simulations, num_assets))
        
        if method == "normal_constrained":
            # Calculate mean weight (equal allocation) and std dev based on asset characteristics
            mean_weight = 1.0 / num_assets
            
            # Use asset volatility to determine weight standard deviation
            # Higher volatility assets get more variable weights
            weight_std_devs = self.std_deviations.values / self.std_deviations.values.mean() * 0.15
            
            print(f"   Mean weight per asset: {mean_weight:.3f}")
            print(f"   Weight std devs: {weight_std_devs.mean():.3f} avg")
            
            for i in range(num_simulations):
                # Generate weights from normal distribution around mean
                weights = np.random.normal(mean_weight, weight_std_devs)
                
                # Ensure all weights are positive
                weights = np.abs(weights)
                
                # Constrain to within 2 standard deviations
                for j in range(num_assets):
                    lower_bound = max(0, mean_weight - 2 * weight_std_devs[j])
                    upper_bound = mean_weight + 2 * weight_std_devs[j]
                    weights[j] = np.clip(weights[j], lower_bound, upper_bound)
                
                # Normalize to sum to 1
                weights = weights / weights.sum()
                all_weights[i] = weights
                
        elif method == "return_weighted":
            # Weight sampling based on expected returns within 2 std devs
            normalized_returns = self.expected_returns.values / self.expected_returns.values.sum()
            return_std = normalized_returns.std()
            
            print(f"   Return-based weighting")
            print(f"   Return std dev: {return_std:.3f}")
            
            for i in range(num_simulations):
                # Sample around return-weighted allocation
                weights = np.random.normal(normalized_returns, return_std * 0.5)
                weights = np.abs(weights)
                
                # Constrain within 2 standard deviations
                for j in range(num_assets):
                    lower_bound = max(0, normalized_returns[j] - 2 * return_std)
                    upper_bound = normalized_returns[j] + 2 * return_std
                    weights[j] = np.clip(weights[j], lower_bound, upper_bound)
                
                # Normalize
                weights = weights / weights.sum()
                all_weights[i] = weights
                
        elif method == "volatility_inverse":
            # Inverse volatility weighting with statistical sampling
            inv_vol_weights = 1 / self.std_deviations.values
            inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
            vol_std = inv_vol_weights.std()
            
            print(f"   Inverse volatility weighting")
            print(f"   Vol weight std dev: {vol_std:.3f}")
            
            for i in range(num_simulations):
                # Sample around inverse volatility weights
                weights = np.random.normal(inv_vol_weights, vol_std * 0.3)
                weights = np.abs(weights)
                
                # Constrain within 2 standard deviations
                for j in range(num_assets):
                    lower_bound = max(0, inv_vol_weights[j] - 2 * vol_std)
                    upper_bound = inv_vol_weights[j] + 2 * vol_std
                    weights[j] = np.clip(weights[j], lower_bound, upper_bound)
                
                # Normalize
                weights = weights / weights.sum()
                all_weights[i] = weights
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Validate results
        weight_sums = all_weights.sum(axis=1)
        print(f"✅ Weight validation:")
        print(f"   All weights sum to 1: {np.allclose(weight_sums, 1.0)}")
        print(f"   Weight range: {all_weights.min():.3f} to {all_weights.max():.3f}")
        print(f"   Mean weight per asset: {all_weights.mean(axis=0)}")
        
        return all_weights
    
    def run_statistical_monte_carlo(self, num_simulations: int = 5000, 
                                  method: str = "normal_constrained",
                                  risk_free_rate: float = 0.03) -> Dict:
        """
        Run statistical Monte Carlo optimization.
        
        Args:
            num_simulations (int): Number of simulations
            method (str): Weight generation method
            risk_free_rate (float): Risk-free rate for Sharpe calculation
            
        Returns:
            dict: Simulation results with enhanced statistics
        """
        if self.expected_returns is None:
            raise ValueError("Must fetch and analyze data first")
        
        print(f"🎯 STATISTICAL MONTE CARLO OPTIMIZATION")
        print("="*55)
        print(f"Method: {method}")
        print(f"Simulations: {num_simulations:,}")
        print(f"Sampling: Within 2 standard deviations")
        
        # Generate statistical weight combinations
        simulation_weights = self.generate_statistical_weights(num_simulations, method)
        
        # Calculate portfolio metrics for each weight combination
        portfolio_returns = np.zeros(num_simulations)
        portfolio_volatility = np.zeros(num_simulations)
        sharpe_ratios = np.zeros(num_simulations)
        
        print(f"\n📊 Calculating portfolio metrics...")
        
        for i in range(num_simulations):
            weights = simulation_weights[i]
            
            # Portfolio return
            portfolio_return = np.sum(weights * self.expected_returns.values)
            
            # Portfolio volatility using covariance matrix
            portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix.values, weights))
            portfolio_vol = np.sqrt(portfolio_variance)
            
            # Sharpe ratio
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            portfolio_returns[i] = portfolio_return
            portfolio_volatility[i] = portfolio_vol
            sharpe_ratios[i] = sharpe_ratio
        
        print(f"✅ Portfolio calculations completed!")
        
        # Enhanced statistics due to constrained sampling
        results = {
            'weights': simulation_weights,
            'returns': portfolio_returns,
            'volatility': portfolio_volatility,
            'sharpe_ratios': sharpe_ratios,
            'num_simulations': num_simulations,
            'method': method,
            'risk_free_rate': risk_free_rate,
            'weight_statistics': {
                'mean_weights': simulation_weights.mean(axis=0),
                'std_weights': simulation_weights.std(axis=0),
                'min_weights': simulation_weights.min(axis=0),
                'max_weights': simulation_weights.max(axis=0)
            }
        }
        
        return results
    
    def find_optimal_portfolios_statistical(self, results: Dict) -> Dict:
        """
        Find optimal portfolios from statistical simulation results.
        
        Args:
            results (dict): Results from statistical Monte Carlo
            
        Returns:
            dict: Enhanced optimal portfolio analysis
        """
        print(f"\n🎯 FINDING OPTIMAL PORTFOLIOS (Statistical Method)")
        print("-" * 55)
        
        # Find optimal portfolios
        max_sharpe_idx = np.argmax(results['sharpe_ratios'])
        min_vol_idx = np.argmin(results['volatility'])
        max_return_idx = np.argmax(results['returns'])
        
        # Create enhanced portfolio summaries
        optimal_portfolios = {
            'max_sharpe': {
                'name': 'Maximum Sharpe Ratio',
                'weights': dict(zip(self.assets, results['weights'][max_sharpe_idx])),
                'weights_array': results['weights'][max_sharpe_idx],
                'expected_return': results['returns'][max_sharpe_idx],
                'volatility': results['volatility'][max_sharpe_idx],
                'sharpe_ratio': results['sharpe_ratios'][max_sharpe_idx],
                'index': max_sharpe_idx
            },
            'min_volatility': {
                'name': 'Minimum Volatility',
                'weights': dict(zip(self.assets, results['weights'][min_vol_idx])),
                'weights_array': results['weights'][min_vol_idx],
                'expected_return': results['returns'][min_vol_idx],
                'volatility': results['volatility'][min_vol_idx],
                'sharpe_ratio': results['sharpe_ratios'][min_vol_idx],
                'index': min_vol_idx
            },
            'max_return': {
                'name': 'Maximum Return',
                'weights': dict(zip(self.assets, results['weights'][max_return_idx])),
                'weights_array': results['weights'][max_return_idx],
                'expected_return': results['returns'][max_return_idx],
                'volatility': results['volatility'][max_return_idx],
                'sharpe_ratio': results['sharpe_ratios'][max_return_idx],
                'index': max_return_idx
            }
        }
        
        # Add statistical analysis
        optimal_portfolios['statistics'] = {
            'method': results['method'],
            'sharpe_range': (results['sharpe_ratios'].min(), results['sharpe_ratios'].max()),
            'return_range': (results['returns'].min(), results['returns'].max()),
            'volatility_range': (results['volatility'].min(), results['volatility'].max()),
            'weight_concentration': results['weight_statistics']
        }
        
        print(f"✅ Optimal portfolios identified using {results['method']} method:")
        for key, portfolio in optimal_portfolios.items():
            if isinstance(portfolio, dict) and 'name' in portfolio:
                print(f"   {portfolio['name']}: {portfolio['expected_return']:.2%} return, {portfolio['volatility']:.2%} vol, Sharpe: {portfolio['sharpe_ratio']:.3f}")
        
        return optimal_portfolios
    
    def plot_statistical_efficient_frontier(self, results: Dict, optimal_portfolios: Dict, 
                                          save_plot: bool = False):
        """
        Plot efficient frontier showing statistical sampling effects.
        
        Args:
            results (dict): Statistical Monte Carlo results
            optimal_portfolios (dict): Optimal portfolio configurations
            save_plot (bool): Whether to save the plot
        """
        print(f"\n📊 Generating statistical efficient frontier plot...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Main efficient frontier with statistical sampling
        scatter = ax1.scatter(results['volatility'], results['returns'], 
                            c=results['sharpe_ratios'], cmap='viridis', 
                            alpha=0.7, s=30)
        
        # Highlight optimal portfolios
        colors = {'max_sharpe': 'red', 'min_volatility': 'blue', 'max_return': 'green'}
        for key, portfolio in optimal_portfolios.items():
            if isinstance(portfolio, dict) and 'volatility' in portfolio:
                ax1.scatter(portfolio['volatility'], portfolio['expected_return'],
                          color=colors[key], s=200, marker='*', 
                          label=portfolio['name'], edgecolors='black', linewidth=2)
        
        ax1.set_xlabel('Volatility (Risk)')
        ax1.set_ylabel('Expected Return')
        ax1.set_title(f'Statistical Efficient Frontier ({results["method"]})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Sharpe Ratio')
        
        # 2. Weight distribution analysis
        weight_stats = results['weight_statistics']
        x_pos = np.arange(len(self.assets))
        
        ax2.bar(x_pos - 0.2, weight_stats['mean_weights'], 0.4, 
               label='Mean', alpha=0.7, color='blue')
        ax2.errorbar(x_pos - 0.2, weight_stats['mean_weights'], 
                    yerr=weight_stats['std_weights'], fmt='none', color='black')
        
        ax2.bar(x_pos + 0.2, optimal_portfolios['max_sharpe']['weights_array'], 0.4,
               label='Optimal', alpha=0.7, color='red')
        
        ax2.set_xlabel('Assets')
        ax2.set_ylabel('Weight')
        ax2.set_title('Weight Distribution (Mean ± Std Dev)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.assets)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Sharpe ratio distribution
        ax3.hist(results['sharpe_ratios'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(optimal_portfolios['max_sharpe']['sharpe_ratio'], color='red', 
                   linestyle='-', linewidth=3, label=f'Max: {optimal_portfolios["max_sharpe"]["sharpe_ratio"]:.3f}')
        ax3.axvline(results['sharpe_ratios'].mean(), color='orange', 
                   linestyle='--', label=f'Mean: {results["sharpe_ratios"].mean():.3f}')
        ax3.set_xlabel('Sharpe Ratio')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Sharpe Ratio Distribution (Statistical Sampling)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Return vs Risk scatter with weight concentration
        weight_concentration = np.sum(results['weights']**2, axis=1)  # Herfindahl index
        scatter2 = ax4.scatter(results['volatility'], results['returns'], 
                             c=weight_concentration, cmap='plasma', alpha=0.6, s=20)
        ax4.scatter(optimal_portfolios['max_sharpe']['volatility'], 
                   optimal_portfolios['max_sharpe']['expected_return'],
                   color='red', s=200, marker='*', label='Optimal Portfolio')
        ax4.set_xlabel('Volatility')
        ax4.set_ylabel('Return')
        ax4.set_title('Risk-Return (Color = Weight Concentration)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax4, label='Weight Concentration')
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"statistical_efficient_frontier_{results['method']}_{'_'.join(self.assets)}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved as {filename}")
        
        plt.show()
    
    def run_complete_statistical_analysis(self, num_simulations: int = 5000,
                                        method: str = "normal_constrained",
                                        risk_free_rate: float = 0.03) -> Dict:
        """
        Run complete statistical Monte Carlo analysis.
        
        Args:
            num_simulations (int): Number of simulations
            method (str): Statistical sampling method
            risk_free_rate (float): Risk-free rate
            
        Returns:
            dict: Complete analysis results
        """
        print(f"🚀 STATISTICAL MONTE CARLO ANALYSIS")
        print("="*50)
        print(f"Enhancement: Statistical sampling within 2 standard deviations")
        print(f"Assets: {', '.join(self.assets)}")
        print(f"Method: {method}")
        print(f"Simulations: {num_simulations:,}")
        
        try:
            # Step 1: Fetch and analyze data
            price_data = self.fetch_and_analyze_data()
            
            # Step 2: Run statistical Monte Carlo
            results = self.run_statistical_monte_carlo(num_simulations, method, risk_free_rate)
            
            # Step 3: Find optimal portfolios
            optimal_portfolios = self.find_optimal_portfolios_statistical(results)
            
            # Step 4: Generate visualizations
            self.plot_statistical_efficient_frontier(results, optimal_portfolios)
            
            # Step 5: Print summary
            self.print_statistical_summary(results, optimal_portfolios)
            
            return {
                'price_data': price_data,
                'simulation_results': results,
                'optimal_portfolios': optimal_portfolios,
                'method': method,
                'assets': self.assets
            }
            
        except Exception as e:
            print(f"❌ Statistical analysis failed: {e}")
            return None
    
    def print_statistical_summary(self, results: Dict, optimal_portfolios: Dict):
        """Print detailed summary of statistical Monte Carlo results."""
        print(f"\n" + "="*70)
        print("STATISTICAL MONTE CARLO RESULTS")
        print("="*70)
        
        print(f"\n📊 Sampling Method: {results['method']}")
        print(f"   Constraint: Within 2 standard deviations of mean")
        print(f"   Simulations: {results['num_simulations']:,}")
        print(f"   Risk-Free Rate: {results['risk_free_rate']:.1%}")
        
        # Weight statistics
        weight_stats = results['weight_statistics']
        print(f"\n📈 Weight Distribution Statistics:")
        for i, asset in enumerate(self.assets):
            print(f"   {asset}: {weight_stats['mean_weights'][i]:.3f} ± {weight_stats['std_weights'][i]:.3f} "
                  f"[{weight_stats['min_weights'][i]:.3f}, {weight_stats['max_weights'][i]:.3f}]")
        
        # Performance statistics
        stats = optimal_portfolios['statistics']
        print(f"\n📊 Performance Ranges:")
        print(f"   Sharpe Ratio: {stats['sharpe_range'][0]:.3f} to {stats['sharpe_range'][1]:.3f}")
        print(f"   Returns: {stats['return_range'][0]:.2%} to {stats['return_range'][1]:.2%}")
        print(f"   Volatility: {stats['volatility_range'][0]:.2%} to {stats['volatility_range'][1]:.2%}")
        
        # Optimal portfolios
        for key, portfolio in optimal_portfolios.items():
            if isinstance(portfolio, dict) and 'name' in portfolio:
                print(f"\n🎯 {portfolio['name'].upper()}")
                print("-" * 50)
                print(f"Expected Return: {portfolio['expected_return']:.2%}")
                print(f"Volatility:     {portfolio['volatility']:.2%}")
                print(f"Sharpe Ratio:   {portfolio['sharpe_ratio']:.3f}")
                print(f"Asset Allocation:")
                for asset, weight in portfolio['weights'].items():
                    print(f"  {asset}: {weight:>8.1%}")
        
        print("\n" + "="*70)


def demo_statistical_monte_carlo():
    """Demonstration of statistical Monte Carlo optimization."""
    print("🎯 STATISTICAL MONTE CARLO DEMO")
    print("="*40)
    print("Enhancement: Using statistical sampling within 2 standard deviations")
    print("Instead of purely random weights\n")
    
    # Demo with tech stocks
    assets = ['AAPL', 'MSFT', 'GOOGL']
    
    optimizer = StatisticalMonteCarloOptimizer(assets, period="3mo")
    
    try:
        # Test different methods
        methods = ["normal_constrained", "return_weighted", "volatility_inverse"]
        
        for method in methods:
            print(f"\n{'='*60}")
            print(f"TESTING METHOD: {method.upper()}")
            print('='*60)
            
            results = optimizer.run_complete_statistical_analysis(
                num_simulations=3000,
                method=method,
                risk_free_rate=0.025
            )
            
            if results:
                print(f"✅ {method} method completed successfully")
            else:
                print(f"❌ {method} method failed")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"• Statistical sampling creates more realistic portfolio distributions")
        print(f"• Weights constrained within 2 standard deviations of mean")
        print(f"• Different methods emphasize different asset characteristics")
        print(f"• More concentrated, practical portfolio allocations")
        
        return results
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None


if __name__ == "__main__":
    demo_statistical_monte_carlo()
