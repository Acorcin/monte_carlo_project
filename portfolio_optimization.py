"""
Portfolio Optimization with Monte Carlo Simulation

This module adds portfolio optimization capabilities to the existing Monte Carlo
simulation framework, allowing users to find optimal asset allocations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import warnings

from data_fetcher import fetch_stock_data, calculate_returns_from_ohlcv

class PortfolioOptimizer:
    """
    Monte Carlo Portfolio Optimization using Modern Portfolio Theory.
    
    Finds optimal portfolio weights by simulating thousands of random
    weight combinations and analyzing the risk-return profile.
    """
    
    def __init__(self, assets: List[str], period: str = "1y", interval: str = "1d"):
        """
        Initialize the portfolio optimizer.
        
        Args:
            assets (list): List of asset tickers (e.g., ['AAPL', 'GOOGL', 'MSFT'])
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
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch price data for all assets.
        
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
        
        # Remove any assets with insufficient data
        min_length = price_data.count().min()
        if min_length < 20:
            warnings.warn(f"Limited data available (only {min_length} periods)")
        
        print(f"✅ Combined dataset: {len(price_data)} periods for {len(price_data.columns)} assets")
        return price_data
    
    def calculate_returns_statistics(self, price_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate return statistics for portfolio optimization.
        
        Args:
            price_data (pd.DataFrame): Price data for all assets
            
        Returns:
            tuple: (expected_returns, std_deviations, correlation_matrix)
        """
        print("📈 Calculating return statistics...")
        
        # Calculate returns
        self.returns_data = price_data.pct_change().dropna()
        
        # Calculate expected returns (annualized)
        trading_periods = 252 if self.interval == "1d" else 252 * 24  # Approximate
        self.expected_returns = self.returns_data.mean() * trading_periods
        
        # Calculate standard deviations (annualized)
        self.std_deviations = self.returns_data.std() * np.sqrt(trading_periods)
        
        # Calculate correlation matrix
        self.correlation_matrix = self.returns_data.corr()
        
        # Calculate covariance matrix
        self.covariance_matrix = self.returns_data.cov() * trading_periods
        
        print("✅ Statistics calculated:")
        print(f"   Expected Returns: {self.expected_returns.mean():.2%} avg")
        print(f"   Volatilities: {self.std_deviations.mean():.2%} avg")
        print(f"   Correlation range: {self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)].min():.2f} to {self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)].max():.2f}")
        
        return self.expected_returns.values, self.std_deviations.values, self.correlation_matrix.values
    
    def monte_carlo_optimization(self, num_simulations: int = 10000, 
                                allow_short_selling: bool = False,
                                method: str = "synthetic_prices") -> Dict[str, np.ndarray]:
        """
        Perform Monte Carlo portfolio optimization with enhanced statistical methods.
        
        Args:
            num_simulations (int): Number of random portfolios to simulate
            allow_short_selling (bool): Whether to allow negative weights
            method (str): Simulation method ("random", "statistical", "normal_constrained")
            
        Returns:
            dict: Simulation results with weights, returns, and volatilities
        """
        if self.expected_returns is None:
            raise ValueError("Must fetch data and calculate statistics first")
        
        print(f"🎲 Running Monte Carlo optimization ({num_simulations:,} simulations)...")
        print(f"   Method: {method}")
        
        num_assets = len(self.assets)
        
        # Initialize arrays to save simulation results
        simulation_results = np.zeros((num_simulations, num_assets))
        portfolio_returns = np.zeros(num_simulations)
        portfolio_volatility = np.zeros(num_simulations)
        sharpe_ratios = np.zeros(num_simulations)
        
        # Risk-free rate (approximate)
        risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Generate weights and prices based on method
        if method == "random":
            # Original random method
            print("   Using original random weight generation")
            weights_array = self._generate_random_weights(num_simulations, num_assets, allow_short_selling)
            
        elif method == "statistical" or method == "normal_constrained":
            # Enhanced statistical method with 2 standard deviation constraint
            print("   Using statistical sampling within 2 standard deviations")
            weights_array = self._generate_statistical_weights(num_simulations, num_assets)
            
        elif method == "synthetic_prices":
            # NEW: Synthetic price generation method
            print("   Using synthetic price generation (statistically identical to original)")
            print("   Constraint: Average prices within 2σ of original means")
            return self._generate_synthetic_price_simulations(num_simulations, allow_short_selling)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate portfolio metrics for each weight combination
        print(f"   Calculating portfolio metrics...")
        
        for i in range(num_simulations):
            weights = weights_array[i]
            
            # Calculate portfolio return
            portfolio_return = np.sum(weights * self.expected_returns)
            
            # Calculate portfolio volatility using covariance matrix
            portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
            portfolio_std_dev = np.sqrt(portfolio_variance)
            
            # Calculate Sharpe ratio
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev if portfolio_std_dev > 0 else 0
            
            # Store results
            simulation_results[i, :] = weights
            portfolio_returns[i] = portfolio_return
            portfolio_volatility[i] = portfolio_std_dev
            sharpe_ratios[i] = sharpe_ratio
        
        print("✅ Monte Carlo optimization completed!")
        
        return {
            'weights': simulation_results,
            'returns': portfolio_returns,
            'volatility': portfolio_volatility,
            'sharpe_ratios': sharpe_ratios,
            'num_simulations': num_simulations,
            'method': method
        }
    
    def _generate_random_weights(self, num_simulations: int, num_assets: int, 
                               allow_short_selling: bool) -> np.ndarray:
        """Generate random portfolio weights (original method)."""
        weights_array = np.zeros((num_simulations, num_assets))
        
        for i in range(num_simulations):
            if allow_short_selling:
                # Allow negative weights (short selling)
                weights = np.random.normal(0, 0.3, num_assets)
            else:
                # Long-only portfolio
                weights = np.random.random(num_assets)
            
            # Normalize weights to sum to 1
            weights /= np.sum(np.abs(weights))
            weights_array[i] = weights
        
        return weights_array
    
    def _generate_statistical_weights(self, num_simulations: int, num_assets: int) -> np.ndarray:
        """
        Generate portfolio weights using statistical sampling within 2 standard deviations.
        
        Args:
            num_simulations (int): Number of weight combinations to generate
            num_assets (int): Number of assets
            
        Returns:
            np.ndarray: Array of portfolio weights
        """
        weights_array = np.zeros((num_simulations, num_assets))
        
        # Calculate mean weight (equal allocation) 
        mean_weight = 1.0 / num_assets
        
        # Use asset characteristics to determine weight variability
        # Higher volatility assets get more variable weights
        normalized_vols = self.std_deviations.values / self.std_deviations.values.mean()
        weight_std_devs = normalized_vols * 0.12  # Scale factor for weight variability
        
        print(f"   Statistical parameters:")
        print(f"     Mean weight per asset: {mean_weight:.3f}")
        print(f"     Weight std devs: {weight_std_devs.mean():.3f} avg")
        print(f"     Constraint: Within 2 standard deviations of mean")
        
        for i in range(num_simulations):
            # Generate weights from normal distribution around mean
            weights = np.random.normal(mean_weight, weight_std_devs)
            
            # Ensure all weights are positive
            weights = np.abs(weights)
            
            # Constrain to within 2 standard deviations of mean
            for j in range(num_assets):
                lower_bound = max(0.01, mean_weight - 2 * weight_std_devs[j])  # Min 1%
                upper_bound = min(0.8, mean_weight + 2 * weight_std_devs[j])   # Max 80%
                weights[j] = np.clip(weights[j], lower_bound, upper_bound)
            
            # Normalize to sum to 1
            weights = weights / weights.sum()
            weights_array[i] = weights
        
        # Validate statistical properties
        mean_weights = weights_array.mean(axis=0)
        std_weights = weights_array.std(axis=0)
        
        print(f"   Generated weights validation:")
        print(f"     Actual mean weights: {mean_weights}")
        print(f"     Actual std weights: {std_weights}")
        print(f"     Min weights: {weights_array.min(axis=0)}")
        print(f"     Max weights: {weights_array.max(axis=0)}")
        
        return weights_array
    
    def _generate_synthetic_price_simulations(self, num_simulations: int, allow_short_selling: bool) -> Dict[str, np.ndarray]:
        """
        Generate Monte Carlo simulations using synthetic price generation.
        
        Creates random price points that are statistically identical to original data,
        with average prices within 2 standard deviations of original means.
        
        Args:
            num_simulations (int): Number of simulations to generate
            allow_short_selling (bool): Whether to allow negative weights
            
        Returns:
            Dict with simulation results
        """
        from synthetic_price_simulation import generate_synthetic_prices
        
        print(f"   📊 Generating synthetic prices for each asset...")
        
        num_assets = len(self.assets)
        num_periods = len(self.returns_data)
        
        # Generate synthetic prices for each asset
        all_asset_synthetic_prices = {}
        
        for i, asset in enumerate(self.assets):
            # Get original prices for this asset (reconstruct from returns)
            original_returns = self.returns_data.iloc[:, i]
            
            # Reconstruct price series (assuming starting price of 100)
            original_prices = [100]
            for ret in original_returns:
                original_prices.append(original_prices[-1] * (1 + ret))
            
            original_prices = np.array(original_prices[1:])  # Remove initial value
            
            # Generate synthetic prices
            synthetic_prices = generate_synthetic_prices(
                original_prices,
                num_simulations=num_simulations,
                num_periods=num_periods,
                constraint_method='normal_2sigma'
            )
            
            all_asset_synthetic_prices[asset] = synthetic_prices
        
        print(f"   💰 Generating portfolio weights and calculating performance...")
        
        # Initialize result arrays
        simulation_results = np.zeros((num_simulations, num_assets))
        portfolio_returns = np.zeros(num_simulations)
        portfolio_volatility = np.zeros(num_simulations)
        sharpe_ratios = np.zeros(num_simulations)
        risk_free_rate = 0.02
        
        for sim in range(num_simulations):
            # Generate portfolio weights for this simulation
            if allow_short_selling:
                weights = np.random.normal(0, 0.3, num_assets)
            else:
                weights = np.random.random(num_assets)
            
            # Normalize weights
            weights = weights / np.sum(np.abs(weights))
            
            # Calculate synthetic portfolio returns for this simulation
            synthetic_portfolio_values = []
            
            for period in range(num_periods):
                period_value = 0
                for asset_idx, asset in enumerate(self.assets):
                    asset_price = all_asset_synthetic_prices[asset][sim, period]
                    period_value += weights[asset_idx] * asset_price
                synthetic_portfolio_values.append(period_value)
            
            # Calculate returns from the synthetic portfolio values
            synthetic_portfolio_values = np.array(synthetic_portfolio_values)
            portfolio_period_returns = np.diff(synthetic_portfolio_values) / synthetic_portfolio_values[:-1]
            
            # Calculate performance metrics
            portfolio_return = np.mean(portfolio_period_returns) * 252  # Annualized
            portfolio_vol = np.std(portfolio_period_returns) * np.sqrt(252)  # Annualized
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            # Store results
            simulation_results[sim] = weights
            portfolio_returns[sim] = portfolio_return
            portfolio_volatility[sim] = portfolio_vol
            sharpe_ratios[sim] = sharpe_ratio
        
        print(f"   ✅ Generated {num_simulations} synthetic price simulations")
        print(f"   📈 Portfolio return range: {portfolio_returns.min():.2%} to {portfolio_returns.max():.2%}")
        print(f"   📊 Portfolio volatility range: {portfolio_volatility.min():.2%} to {portfolio_volatility.max():.2%}")
        
        return {
            'weights': simulation_results,
            'returns': portfolio_returns,
            'volatility': portfolio_volatility,
            'sharpe_ratios': sharpe_ratios,
            'num_simulations': num_simulations,
            'method': 'synthetic_prices',
            'synthetic_prices': all_asset_synthetic_prices
        }
    
    def find_optimal_portfolios(self, results: Dict[str, np.ndarray], risk_free_rate: float = 0.03) -> Dict[str, Dict]:
        """
        Find optimal portfolios from simulation results with enhanced Sharpe ratio analysis.
        
        Args:
            results (dict): Results from monte_carlo_optimization
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
            
        Returns:
            dict: Optimal portfolio configurations with detailed analysis
        """
        print("🎯 Finding optimal portfolios...")
        
        # Create a DataFrame from simulation results (your code integration)
        portfolio_df = pd.DataFrame({
            'Return': results['returns'], 
            'Volatility': results['volatility']
        })
        
        # Create an array of the Sharpe ratio (your code integration)
        sharpe_arr = (results['returns'] - risk_free_rate) / results['volatility']
        
        # Find the index of the maximum Sharpe ratio (your code integration)
        max_sr_idx = sharpe_arr.argmax()
        
        # Retrieve the optimal weights and corresponding return and volatility (your code integration)
        optimal_weights = results['weights'][max_sr_idx]
        optimal_return = results['returns'][max_sr_idx]
        optimal_volatility = results['volatility'][max_sr_idx]
        
        # Calculate the Sharpe ratio at the maximum point (your code integration)
        MC_SR = sharpe_arr[max_sr_idx]
        
        # Calculate the annualized Sharpe ratio (enhanced from your code)
        # Determine annualization factor based on data interval
        if self.interval == "1d":
            annualization_factor = np.sqrt(252)  # Daily data
        elif self.interval == "1h":
            annualization_factor = np.sqrt(252 * 24)  # Hourly data
        elif self.interval == "1mo":
            annualization_factor = np.sqrt(12)  # Monthly data (your original assumption)
        else:
            annualization_factor = np.sqrt(252)  # Default to daily
        
        SR_annualized = MC_SR * annualization_factor
        
        # Find other optimal portfolios
        min_vol_idx = np.argmin(results['volatility'])
        max_return_idx = np.argmax(results['returns'])
        
        # Create enhanced portfolio summaries
        optimal_portfolios = {
            'max_sharpe': {
                'name': 'Maximum Sharpe Ratio',
                'weights': dict(zip(self.assets, optimal_weights)),
                'weights_array': optimal_weights,
                'expected_return': optimal_return,
                'volatility': optimal_volatility,
                'sharpe_ratio': MC_SR,
                'sharpe_ratio_annualized': SR_annualized,
                'risk_free_rate': risk_free_rate,
                'excess_return': optimal_return - risk_free_rate,
                'index': max_sr_idx
            },
            'min_volatility': {
                'name': 'Minimum Volatility',
                'weights': dict(zip(self.assets, results['weights'][min_vol_idx])),
                'weights_array': results['weights'][min_vol_idx],
                'expected_return': results['returns'][min_vol_idx],
                'volatility': results['volatility'][min_vol_idx],
                'sharpe_ratio': results['sharpe_ratios'][min_vol_idx],
                'sharpe_ratio_annualized': results['sharpe_ratios'][min_vol_idx] * annualization_factor,
                'risk_free_rate': risk_free_rate,
                'excess_return': results['returns'][min_vol_idx] - risk_free_rate,
                'index': min_vol_idx
            },
            'max_return': {
                'name': 'Maximum Return',
                'weights': dict(zip(self.assets, results['weights'][max_return_idx])),
                'weights_array': results['weights'][max_return_idx],
                'expected_return': results['returns'][max_return_idx],
                'volatility': results['volatility'][max_return_idx],
                'sharpe_ratio': results['sharpe_ratios'][max_return_idx],
                'sharpe_ratio_annualized': results['sharpe_ratios'][max_return_idx] * annualization_factor,
                'risk_free_rate': risk_free_rate,
                'excess_return': results['returns'][max_return_idx] - risk_free_rate,
                'index': max_return_idx
            }
        }
        
        # Add DataFrame to results for further analysis
        optimal_portfolios['portfolio_dataframe'] = portfolio_df
        optimal_portfolios['sharpe_array'] = sharpe_arr
        optimal_portfolios['annualization_factor'] = annualization_factor
        
        print("✅ Optimal portfolios identified:")
        for key, portfolio in optimal_portfolios.items():
            if isinstance(portfolio, dict) and 'name' in portfolio:
                print(f"   {portfolio['name']}: {portfolio['expected_return']:.2%} return, {portfolio['volatility']:.2%} volatility, Sharpe: {portfolio['sharpe_ratio']:.3f}")
        
        return optimal_portfolios
    
    def plot_efficient_frontier(self, results: Dict[str, np.ndarray], 
                               optimal_portfolios: Dict[str, Dict] = None,
                               save_plot: bool = False):
        """
        Plot the efficient frontier and highlight optimal portfolios.
        
        Args:
            results (dict): Monte Carlo simulation results
            optimal_portfolios (dict): Optimal portfolio configurations
            save_plot (bool): Whether to save the plot
        """
        print("📊 Generating efficient frontier plot...")
        
        plt.figure(figsize=(12, 8))
        
        # Plot all simulated portfolios
        scatter = plt.scatter(results['volatility'], results['returns'], 
                            c=results['sharpe_ratios'], cmap='viridis', 
                            alpha=0.6, s=20)
        
        plt.colorbar(scatter, label='Sharpe Ratio')
        
        # Highlight optimal portfolios
        if optimal_portfolios:
            colors = {'max_sharpe': 'red', 'min_volatility': 'blue', 'max_return': 'green'}
            markers = {'max_sharpe': '*', 'min_volatility': 'o', 'max_return': '^'}
            
            for key, portfolio in optimal_portfolios.items():
                if isinstance(portfolio, dict) and 'volatility' in portfolio:
                    plt.scatter(portfolio['volatility'], portfolio['expected_return'],
                              color=colors[key], marker=markers[key], s=200, 
                              label=portfolio['name'], edgecolors='black', linewidth=2)
        
        # Formatting
        plt.xlabel('Volatility (Risk)', fontsize=12)
        plt.ylabel('Expected Return', fontsize=12)
        plt.title('Efficient Frontier - Monte Carlo Portfolio Optimization', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format axes as percentages
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"efficient_frontier_{'_'.join(self.assets)}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved as {filename}")
        
        plt.show()
    
    def plot_zero_risk_efficient_frontier(self, results: Dict[str, np.ndarray], 
                                        save_plot: bool = False):
        """
        Plot efficient frontier with zero risk-free rate assumption (your original code).
        
        Args:
            results (dict): Monte Carlo simulation results
            save_plot (bool): Whether to save the plot
        """
        print("📊 Generating zero risk-free rate efficient frontier (your original plot)...")
        
        # =============================================================
        # YOUR ORIGINAL CODE IMPLEMENTATION
        # =============================================================
        
        # Extract variables from results
        portfolio_returns = results['returns']
        portfolio_volatility = results['volatility']
        simulation_results = results['weights']
        
        # Simplifying assumption, risk free rate is zero, for sharpe ratio
        risk_free_rate = 0
        
        # Create DataFrame from simulation results (your code)
        portfolio_df = pd.DataFrame({'Return': portfolio_returns, 'Volatility': portfolio_volatility})
        
        # Calculate Sharpe with zero risk-free rate
        sharpe_arr = (portfolio_returns - risk_free_rate) / portfolio_volatility
        max_sr_idx = sharpe_arr.argmax()
        
        # Get optimal portfolio
        optimal_weights = simulation_results[max_sr_idx]
        optimal_return = portfolio_returns[max_sr_idx]
        optimal_volatility = portfolio_volatility[max_sr_idx]
        
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
        
        if save_plot:
            filename = f"zero_risk_frontier_{'_'.join(self.assets)}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved as {filename}")
        
        plt.show()
        
        # Print results
        print(f"\n🎯 ZERO RISK-FREE RATE RESULTS:")
        print(f"   Risk-Free Rate: {risk_free_rate:.1%}")
        print(f"   Maximum Sharpe: {sharpe_arr[max_sr_idx]:.4f}")
        print(f"   Optimal Return: {optimal_return:.2%}")
        print(f"   Optimal Volatility: {optimal_volatility:.2%}")
        print(f"   Optimal Allocation: {dict(zip(self.assets, optimal_weights))}")
        
        return {
            'portfolio_df': portfolio_df,
            'sharpe_arr': sharpe_arr,
            'optimal_weights': optimal_weights,
            'optimal_return': optimal_return,
            'optimal_volatility': optimal_volatility,
            'max_sharpe_ratio': sharpe_arr[max_sr_idx]
        }
    
    def print_portfolio_summary(self, optimal_portfolios: Dict[str, Dict]):
        """Print detailed summary of optimal portfolios with enhanced Sharpe analysis."""
        print("\n" + "="*80)
        print("PORTFOLIO OPTIMIZATION RESULTS")
        print("="*80)
        
        # Show annualization info
        if 'annualization_factor' in optimal_portfolios:
            annualization_factor = optimal_portfolios['annualization_factor']
            data_frequency = "Daily" if self.interval == "1d" else "Hourly" if self.interval == "1h" else "Monthly" if self.interval == "1mo" else "Unknown"
            print(f"\n📊 Data Analysis:")
            print(f"   Frequency: {data_frequency} ({self.interval})")
            print(f"   Annualization Factor: {annualization_factor:.2f}")
            print(f"   Risk-Free Rate: {optimal_portfolios.get('max_sharpe', {}).get('risk_free_rate', 0.03):.1%}")
        
        for key, portfolio in optimal_portfolios.items():
            if isinstance(portfolio, dict) and 'name' in portfolio:
                print(f"\n🎯 {portfolio['name'].upper()}")
                print("-" * 60)
                print(f"Expected Annual Return: {portfolio['expected_return']:.2%}")
                print(f"Annual Volatility:      {portfolio['volatility']:.2%}")
                print(f"Sharpe Ratio:          {portfolio['sharpe_ratio']:.3f}")
                
                # Enhanced Sharpe ratio analysis (your code integration)
                if 'sharpe_ratio_annualized' in portfolio:
                    print(f"Annualized Sharpe:     {portfolio['sharpe_ratio_annualized']:.3f}")
                if 'excess_return' in portfolio:
                    print(f"Excess Return:         {portfolio['excess_return']:.2%}")
                if 'index' in portfolio:
                    print(f"Simulation Index:      {portfolio['index']:,}")
                
                print(f"\nAsset Allocation:")
                for asset, weight in portfolio['weights'].items():
                    print(f"  {asset}: {weight:>8.1%}")
        
        # Additional analysis summary
        print(f"\n📈 SHARPE RATIO ANALYSIS")
        print("-" * 60)
        if 'sharpe_array' in optimal_portfolios:
            sharpe_arr = optimal_portfolios['sharpe_array']
            print(f"Portfolio Count:       {len(sharpe_arr):,}")
            print(f"Sharpe Ratio Range:    {sharpe_arr.min():.3f} to {sharpe_arr.max():.3f}")
            print(f"Average Sharpe:        {sharpe_arr.mean():.3f}")
            print(f"Sharpe Std Dev:        {sharpe_arr.std():.3f}")
            
            # Percentile analysis
            percentiles = [10, 25, 50, 75, 90]
            print(f"\nSharpe Ratio Percentiles:")
            for p in percentiles:
                value = np.percentile(sharpe_arr, p)
                print(f"  {p:2d}th percentile: {value:.3f}")
        
        print("\n" + "="*80)
    
    def print_detailed_portfolio_analysis(self, results: Dict, optimal_portfolios: Dict):
        """
        Print detailed portfolio analysis using your exact code structure.
        
        Args:
            results (dict): Monte Carlo simulation results
            optimal_portfolios (dict): Optimal portfolio configurations
        """
        print(f"\n🎯 DETAILED PORTFOLIO ANALYSIS")
        print("="*50)
        
        # Extract variables for your analysis code
        simulation_results = results['weights']
        portfolio_returns = results['returns']
        portfolio_volatility = results['volatility']
        
        # =============================================================
        # YOUR EXACT CODE IMPLEMENTATION
        # =============================================================
        
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
        # END YOUR EXACT CODE
        # =============================================================
        
        # Enhanced breakdown
        print(f"\n📊 ENHANCED ANALYSIS BREAKDOWN:")
        print("-" * 40)
        
        print(f"Risk Analysis:")
        print(f"  Risk-Free Rate:     {risk_free_rate:.1%}")
        print(f"  Excess Return:      {(optimal_return - risk_free_rate):.2%}")
        print(f"  Return/Risk Ratio:  {optimal_return/optimal_volatility:.4f}")
        
        print(f"\nAsset Allocation Details:")
        for i, asset in enumerate(self.assets):
            weight = optimal_weights[i]
            print(f"  {asset:>6}: {weight:>8.1%} (${weight * 10000:>7,.0f} on $10k)")
        
        # Statistical insights
        print(f"\nSharpe Ratio Statistics:")
        print(f"  Simulations:     {len(sharpe_arr):,}")
        print(f"  Mean Sharpe:     {sharpe_arr.mean():.4f}")
        print(f"  Std Dev:         {sharpe_arr.std():.4f}")
        print(f"  Best Portfolio:  #{max_sr_idx:,} out of {len(sharpe_arr):,}")
        
        # Percentile analysis
        percentiles = [50, 75, 90, 95, 99]
        print(f"\nSharpe Percentiles:")
        for p in percentiles:
            value = np.percentile(sharpe_arr, p)
            print(f"  {p:2d}th: {value:.4f}")
        
        # Portfolio quality
        excellent = (sharpe_arr >= 2.0).sum()
        good = ((sharpe_arr >= 1.0) & (sharpe_arr < 2.0)).sum()
        moderate = ((sharpe_arr >= 0.5) & (sharpe_arr < 1.0)).sum()
        poor = (sharpe_arr < 0.5).sum()
        
        total = len(sharpe_arr)
        print(f"\nPortfolio Quality Distribution:")
        print(f"  Excellent (≥2.0): {excellent:5,} ({excellent/total*100:5.1f}%)")
        print(f"  Good (1.0-2.0):   {good:5,} ({good/total*100:5.1f}%)")
        print(f"  Moderate (0.5-1): {moderate:5,} ({moderate/total*100:5.1f}%)")
        print(f"  Poor (<0.5):      {poor:5,} ({poor/total*100:5.1f}%)")
        
        print("\n" + "="*80)
    
    def run_full_optimization(self, num_simulations: int = 10000, 
                            allow_short_selling: bool = False,
                            plot_results: bool = True,
                            save_plot: bool = False,
                            method: str = "synthetic_prices") -> Dict:
        """
        Run complete portfolio optimization workflow.
        
        Args:
            num_simulations (int): Number of Monte Carlo simulations
            allow_short_selling (bool): Allow negative weights
            plot_results (bool): Generate visualization
            save_plot (bool): Save plot to file
            method (str): Simulation method ("random", "statistical")
            
        Returns:
            dict: Complete optimization results
        """
        print(f"🚀 PORTFOLIO OPTIMIZATION - {len(self.assets)} ASSETS")
        print("="*60)
        print(f"Method: {method}")
        
        try:
            # Step 1: Fetch data
            price_data = self.fetch_data()
            
            # Step 2: Calculate statistics
            self.calculate_returns_statistics(price_data)
            
            # Step 3: Run Monte Carlo simulation
            results = self.monte_carlo_optimization(num_simulations, allow_short_selling, method)
            
            # Step 4: Find optimal portfolios
            optimal_portfolios = self.find_optimal_portfolios(results)
            
            # Step 5: Display results
            self.print_portfolio_summary(optimal_portfolios)
            
            # Step 5b: Detailed analysis using your exact code
            self.print_detailed_portfolio_analysis(results, optimal_portfolios)
            
            # Step 6: Plot efficient frontier
            if plot_results:
                try:
                    self.plot_efficient_frontier(results, optimal_portfolios, save_plot)
                except Exception as e:
                    print(f"⚠️  Plotting failed: {e}")
            
            return {
                'price_data': price_data,
                'returns_data': self.returns_data,
                'simulation_results': results,
                'optimal_portfolios': optimal_portfolios,
                'assets': self.assets,
                'expected_returns': self.expected_returns,
                'std_deviations': self.std_deviations,
                'correlation_matrix': self.correlation_matrix
            }
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return None


def demo_portfolio_optimization():
    """Demonstration of portfolio optimization."""
    print("🎯 PORTFOLIO OPTIMIZATION DEMO")
    print("="*40)
    
    # Demo with tech stocks
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    optimizer = PortfolioOptimizer(assets, period="6mo", interval="1d")
    
    try:
        results = optimizer.run_full_optimization(
            num_simulations=5000,
            allow_short_selling=False,
            plot_results=True,
            save_plot=False
        )
        
        if results:
            print("\n💡 INSIGHTS:")
            print("• The efficient frontier shows optimal risk-return combinations")
            print("• Maximum Sharpe ratio portfolio offers best risk-adjusted returns") 
            print("• Minimum volatility portfolio offers lowest risk")
            print("• Diversification reduces risk through correlation effects")
            
        return results
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None


if __name__ == "__main__":
    demo_portfolio_optimization()
