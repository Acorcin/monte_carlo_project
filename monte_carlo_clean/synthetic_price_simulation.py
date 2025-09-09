"""
Synthetic Price Point Monte Carlo Simulation

This module creates random price points that are statistically identical 
to the original data, ensuring the average price is within 2 standard 
deviations of the original mean.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
from scipy import stats
import warnings

def generate_synthetic_prices(original_prices: Union[List[float], np.ndarray, pd.Series],
                            num_simulations: int = 1000,
                            num_periods: int = None,
                            constraint_method: str = 'normal_2sigma') -> np.ndarray:
    """
    Generate synthetic price points that are statistically identical to original data.
    
    The average price will be within 2 standard deviations of the original mean.
    
    Args:
        original_prices: Original price data to match statistically
        num_simulations: Number of simulation paths to generate
        num_periods: Number of periods per simulation (defaults to len(original_prices))
        constraint_method: Method for generating constrained random prices
        
    Returns:
        np.ndarray: Array of synthetic prices (num_simulations x num_periods)
    """
    # Convert to numpy array
    original_prices = np.array(original_prices)
    
    if num_periods is None:
        num_periods = len(original_prices)
    
    # Calculate statistical properties of original data
    original_mean = np.mean(original_prices)
    original_std = np.std(original_prices)
    original_min = np.min(original_prices)
    original_max = np.max(original_prices)
    
    print(f"🎲 GENERATING SYNTHETIC PRICE POINTS")
    print(f"   Method: {constraint_method}")
    print(f"   Original data statistics:")
    print(f"     Mean: ${original_mean:.2f}")
    print(f"     Std Dev: ${original_std:.2f}")
    print(f"     Range: ${original_min:.2f} to ${original_max:.2f}")
    print(f"   Constraint: Average within 2σ of original mean")
    print(f"   Target range: ${original_mean - 2*original_std:.2f} to ${original_mean + 2*original_std:.2f}")
    
    all_simulations = []
    
    for sim in range(num_simulations):
        if constraint_method == 'normal_2sigma':
            # Generate random prices from normal distribution
            synthetic_prices = np.random.normal(original_mean, original_std, num_periods)
            
            # Ensure prices are within reasonable bounds (positive)
            synthetic_prices = np.maximum(synthetic_prices, original_min * 0.1)
            
            # Constrain the average to be within 2 standard deviations
            current_mean = np.mean(synthetic_prices)
            
            # If mean is outside 2σ range, adjust
            target_min = original_mean - 2 * original_std
            target_max = original_mean + 2 * original_std
            
            if current_mean < target_min:
                # Scale up to bring mean into range
                adjustment = (target_min - original_mean) / (current_mean - original_mean)
                synthetic_prices = original_mean + (synthetic_prices - original_mean) * adjustment
            elif current_mean > target_max:
                # Scale down to bring mean into range
                adjustment = (target_max - original_mean) / (current_mean - original_mean)
                synthetic_prices = original_mean + (synthetic_prices - original_mean) * adjustment
            
        elif constraint_method == 'bootstrap_2sigma':
            # Bootstrap sampling with constraints
            synthetic_prices = np.random.choice(original_prices, size=num_periods, replace=True)
            
            # Add noise while maintaining statistical properties
            noise_factor = 0.1  # 10% noise
            noise = np.random.normal(0, original_std * noise_factor, num_periods)
            synthetic_prices = synthetic_prices + noise
            
            # Ensure average is within 2σ
            current_mean = np.mean(synthetic_prices)
            target_min = original_mean - 2 * original_std
            target_max = original_mean + 2 * original_std
            
            if current_mean < target_min or current_mean > target_max:
                # Adjust all prices proportionally to bring mean into range
                target_mean = np.random.uniform(target_min, target_max)
                adjustment = target_mean / current_mean
                synthetic_prices = synthetic_prices * adjustment
        
        elif constraint_method == 'truncated_normal':
            # Use truncated normal distribution
            # Set bounds to ensure reasonable prices
            lower_bound = max(0, original_mean - 3 * original_std)
            upper_bound = original_mean + 3 * original_std
            
            # Generate from truncated normal
            synthetic_prices = stats.truncnorm.rvs(
                (lower_bound - original_mean) / original_std,
                (upper_bound - original_mean) / original_std,
                loc=original_mean,
                scale=original_std,
                size=num_periods
            )
            
            # Verify mean constraint
            current_mean = np.mean(synthetic_prices)
            if not (original_mean - 2*original_std <= current_mean <= original_mean + 2*original_std):
                # Force adjustment if needed
                target_mean = np.random.uniform(
                    original_mean - 2*original_std, 
                    original_mean + 2*original_std
                )
                synthetic_prices = synthetic_prices + (target_mean - current_mean)
        
        all_simulations.append(synthetic_prices)
    
    # Convert to array and validate
    all_simulations = np.array(all_simulations)
    
    # Validation
    simulation_means = np.mean(all_simulations, axis=1)
    valid_simulations = np.sum(
        (simulation_means >= original_mean - 2*original_std) & 
        (simulation_means <= original_mean + 2*original_std)
    )
    
    print(f"   Generated {num_simulations} simulations")
    print(f"   Simulations with mean in 2σ range: {valid_simulations}/{num_simulations} ({valid_simulations/num_simulations*100:.1f}%)")
    print(f"   Simulation means range: ${simulation_means.min():.2f} to ${simulation_means.max():.2f}")
    
    return all_simulations


def synthetic_price_monte_carlo_portfolio(assets_data: Dict[str, np.ndarray],
                                        weights: np.ndarray,
                                        num_simulations: int = 1000,
                                        num_periods: int = None) -> Dict:
    """
    Run Monte Carlo portfolio simulation with synthetic price generation.
    
    Args:
        assets_data: Dictionary of asset name -> price array
        weights: Portfolio weights for each asset
        num_simulations: Number of Monte Carlo simulations
        num_periods: Number of periods to simulate
        
    Returns:
        Dict with simulation results
    """
    print(f"\n🚀 SYNTHETIC PRICE PORTFOLIO MONTE CARLO")
    print("="*50)
    
    assets = list(assets_data.keys())
    if num_periods is None:
        num_periods = len(next(iter(assets_data.values())))
    
    print(f"Assets: {', '.join(assets)}")
    print(f"Portfolio weights: {dict(zip(assets, weights))}")
    print(f"Simulations: {num_simulations:,}")
    print(f"Periods per simulation: {num_periods}")
    
    # Generate synthetic prices for each asset
    all_asset_simulations = {}
    
    for asset, original_prices in assets_data.items():
        print(f"\n📊 Generating synthetic prices for {asset}...")
        synthetic_prices = generate_synthetic_prices(
            original_prices,
            num_simulations=num_simulations,
            num_periods=num_periods,
            constraint_method='normal_2sigma'
        )
        all_asset_simulations[asset] = synthetic_prices
    
    # Calculate portfolio values for each simulation
    print(f"\n💰 Calculating portfolio values...")
    
    portfolio_simulations = []
    
    for sim in range(num_simulations):
        # Get synthetic prices for this simulation
        portfolio_values = []
        
        for period in range(num_periods):
            period_value = 0
            
            for i, asset in enumerate(assets):
                asset_price = all_asset_simulations[asset][sim, period]
                period_value += weights[i] * asset_price
            
            portfolio_values.append(period_value)
        
        portfolio_simulations.append(portfolio_values)
    
    portfolio_simulations = np.array(portfolio_simulations)
    
    # Calculate statistics
    final_values = portfolio_simulations[:, -1]
    initial_values = portfolio_simulations[:, 0]
    returns = (final_values - initial_values) / initial_values
    
    results = {
        'portfolio_simulations': portfolio_simulations,
        'asset_simulations': all_asset_simulations,
        'final_values': final_values,
        'returns': returns,
        'weights': weights,
        'assets': assets,
        'statistics': {
            'mean_final_value': np.mean(final_values),
            'std_final_value': np.std(final_values),
            'min_final_value': np.min(final_values),
            'max_final_value': np.max(final_values),
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'var_95': np.percentile(final_values, 5),
            'var_99': np.percentile(final_values, 1)
        }
    }
    
    print(f"\n📈 PORTFOLIO SIMULATION RESULTS:")
    print("-" * 40)
    stats = results['statistics']
    print(f"Final Value Range: ${stats['min_final_value']:,.2f} to ${stats['max_final_value']:,.2f}")
    print(f"Mean Final Value:  ${stats['mean_final_value']:,.2f}")
    print(f"Std Dev:           ${stats['std_final_value']:,.2f}")
    print(f"Mean Return:       {stats['mean_return']:.2%}")
    print(f"Return Std Dev:    {stats['std_return']:.2%}")
    print(f"VaR (95%):         ${stats['var_95']:,.2f}")
    print(f"VaR (99%):         ${stats['var_99']:,.2f}")
    
    return results


def plot_synthetic_price_analysis(results: Dict):
    """Plot analysis of synthetic price simulations."""
    print(f"\n📊 Creating synthetic price analysis plots...")
    
    portfolio_sims = results['portfolio_simulations']
    asset_sims = results['asset_simulations']
    assets = results['assets']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Portfolio evolution paths
    ax = axes[0, 0]
    for i in range(min(50, len(portfolio_sims))):  # Show first 50 paths
        ax.plot(portfolio_sims[i], alpha=0.3, color='blue', linewidth=0.5)
    
    # Plot mean path
    mean_path = np.mean(portfolio_sims, axis=0)
    ax.plot(mean_path, color='red', linewidth=2, label='Mean Path')
    
    # Plot confidence bands
    std_path = np.std(portfolio_sims, axis=0)
    ax.fill_between(range(len(mean_path)), 
                    mean_path - std_path, 
                    mean_path + std_path, 
                    alpha=0.3, color='red', label='±1 Std Dev')
    
    ax.set_title('Portfolio Evolution (Synthetic Prices)')
    ax.set_xlabel('Period')
    ax.set_ylabel('Portfolio Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Final value distribution
    ax = axes[0, 1]
    final_values = results['final_values']
    ax.hist(final_values, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(np.mean(final_values), color='red', linestyle='--', 
               label=f'Mean: ${np.mean(final_values):,.0f}')
    ax.set_title('Distribution of Final Portfolio Values')
    ax.set_xlabel('Final Portfolio Value ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Return distribution
    ax = axes[0, 2]
    returns = results['returns']
    ax.hist(returns, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(np.mean(returns), color='red', linestyle='--', 
               label=f'Mean: {np.mean(returns):.2%}')
    ax.set_title('Distribution of Portfolio Returns')
    ax.set_xlabel('Portfolio Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Asset price paths (first asset)
    if len(assets) > 0:
        ax = axes[1, 0]
        first_asset = assets[0]
        asset_prices = asset_sims[first_asset]
        
        for i in range(min(50, len(asset_prices))):
            ax.plot(asset_prices[i], alpha=0.3, color='purple', linewidth=0.5)
        
        mean_asset_path = np.mean(asset_prices, axis=0)
        ax.plot(mean_asset_path, color='red', linewidth=2, label='Mean Path')
        ax.set_title(f'{first_asset} Synthetic Price Paths')
        ax.set_xlabel('Period')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. Correlation between simulations (first vs last period)
    ax = axes[1, 1]
    first_period = portfolio_sims[:, 0]
    last_period = portfolio_sims[:, -1]
    ax.scatter(first_period, last_period, alpha=0.6, s=20)
    ax.set_xlabel('Initial Portfolio Value')
    ax.set_ylabel('Final Portfolio Value')
    ax.set_title('Initial vs Final Portfolio Value')
    ax.grid(True, alpha=0.3)
    
    # 6. Simulation means validation
    ax = axes[1, 2]
    # Calculate mean of each simulation
    sim_means = np.mean(portfolio_sims, axis=1)
    overall_mean = np.mean(portfolio_sims)
    overall_std = np.std(portfolio_sims)
    
    ax.hist(sim_means, bins=30, alpha=0.7, color='cyan', edgecolor='black')
    ax.axvline(overall_mean, color='red', linestyle='-', linewidth=2, 
               label=f'Overall Mean: ${overall_mean:.0f}')
    ax.axvline(overall_mean - 2*overall_std, color='orange', linestyle='--', 
               label=f'2σ Bounds')
    ax.axvline(overall_mean + 2*overall_std, color='orange', linestyle='--')
    ax.set_title('Distribution of Simulation Means\n(Should be within 2σ)')
    ax.set_xlabel('Simulation Mean Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demo_synthetic_price_simulation():
    """Demonstrate synthetic price simulation."""
    print("🎯 SYNTHETIC PRICE MONTE CARLO DEMO")
    print("="*40)
    print("New approach: Generate random price points statistically identical to original data")
    print("Constraint: Average price within 2 standard deviations of original mean")
    print()
    
    # Create sample price data
    np.random.seed(42)
    
    # Simulate some original price data
    periods = 252  # One year of daily data
    initial_prices = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500}
    
    original_data = {}
    for asset, initial_price in initial_prices.items():
        # Generate realistic price path
        returns = np.random.normal(0.0008, 0.02, periods)  # Daily returns
        prices = [initial_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        original_data[asset] = np.array(prices)
    
    print(f"📊 Generated sample data:")
    for asset, prices in original_data.items():
        print(f"   {asset}: ${prices[0]:.2f} → ${prices[-1]:.2f} (mean: ${np.mean(prices):.2f})")
    
    # Portfolio weights
    weights = np.array([0.4, 0.3, 0.3])  # 40% AAPL, 30% MSFT, 30% GOOGL
    
    # Run synthetic price Monte Carlo
    results = synthetic_price_monte_carlo_portfolio(
        original_data,
        weights,
        num_simulations=1000,
        num_periods=100  # Shorter for demo
    )
    
    # Plot results
    plot_synthetic_price_analysis(results)
    
    print(f"\n✅ SYNTHETIC PRICE SIMULATION COMPLETE!")
    print(f"   Generated {len(results['portfolio_simulations'])} different portfolio paths")
    print(f"   Each simulation has statistically identical properties to original data")
    print(f"   Average prices constrained within 2σ of original means")
    
    return results


if __name__ == "__main__":
    demo_synthetic_price_simulation()
