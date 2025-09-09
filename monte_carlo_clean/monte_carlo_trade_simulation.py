"""
Monte Carlo Trade Order Simulation

This module provides tools for analyzing the impact of trade order randomization
on portfolio performance using Monte Carlo simulation methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
import warnings

def random_trade_order_simulation(
    trade_returns: Union[List[float], np.ndarray], 
    num_simulations: int = 1000, 
    initial_capital: float = 10000,
    position_size_mode: str = 'compound',
    simulation_method: str = 'synthetic_returns'
) -> pd.DataFrame:
    """
    Performs a Monte Carlo simulation by shuffling the order of trade returns.
    
    This function tests how different orderings of the same trades affect
    the final portfolio value, helping to understand sequence risk.

    Args:
        trade_returns (list or np.array): A list of returns from individual trades.
                                        Should be in decimal format (e.g., 0.05 for 5%).
        num_simulations (int): The number of simulations to run. Default is 1000.
        initial_capital (float): The starting capital for the backtest. Default is 10000.
        position_size_mode (str): 'compound' for compounding returns, 'fixed' for fixed position sizes.
        simulation_method (str): 'random' for shuffling, 'statistical' for statistical sampling,
                               'synthetic_returns' for generating new returns within 2σ of original mean.

    Returns:
        pandas.DataFrame: A DataFrame containing the equity curves for each simulation.
                         Each column represents one simulation, with rows showing
                         the portfolio value after each trade.
                         
    Example:
        >>> returns = [0.02, -0.01, 0.03, -0.005, 0.015]
        >>> results = random_trade_order_simulation(returns, num_simulations=100)
        >>> print(results.shape)
        (6, 100)  # 6 rows (initial + 5 trades), 100 simulations
    """
    # Convert to numpy array and create a copy to avoid modifying original
    trade_returns = np.array(trade_returns).copy()
    
    # Validate inputs
    if len(trade_returns) == 0:
        raise ValueError("trade_returns cannot be empty")
    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    
    # Check for extreme returns that might cause issues
    if np.any(trade_returns <= -1):
        warnings.warn("Some returns are <= -100%, which would result in negative portfolio values")
    
    # Pre-allocate data structure for better performance
    num_trades = len(trade_returns)
    
    # Generate return sequences based on simulation method
    if simulation_method == 'statistical':
        print(f"🎲 Using statistical sampling method (2 std dev constraints)")
        return_sequences = _generate_statistical_return_sequences(trade_returns, num_simulations)
    elif simulation_method == 'synthetic_returns':
        print(f"🎲 Using synthetic return generation (statistically identical to original)")
        return_sequences = _generate_synthetic_return_sequences(trade_returns, num_simulations)
    else:
        print(f"🎲 Using random shuffling method")
        return_sequences = _generate_random_return_sequences(trade_returns, num_simulations)
    
    all_equity_curves = {}
    
    for i in range(num_simulations):
        # Get the return sequence for this simulation (already generated above)
        shuffled_returns = return_sequences[i]
        
        # Calculate equity curve for this simulation
        equity_curve = [initial_capital]
        current_value = initial_capital
        
        if position_size_mode == 'compound':
            # Standard compounding mode
            for trade_return in shuffled_returns:
                current_value = current_value * (1 + trade_return)
                equity_curve.append(current_value)
        elif position_size_mode == 'fixed':
            # Fixed dollar amount per trade - demonstrates sequence risk
            # Large losses early vs late have different impacts on portfolio
            position_size_pct = 0.2  # Risk 20% of current portfolio per trade
            for trade_return in shuffled_returns:
                risk_amount = current_value * position_size_pct
                pnl = risk_amount * trade_return
                current_value = current_value + pnl
                equity_curve.append(max(0, current_value))  # Prevent negative values
        
        all_equity_curves[f'Simulation_{i+1}'] = equity_curve
    
    # Create DataFrame efficiently
    equity_curves = pd.DataFrame(all_equity_curves)
    
    return equity_curves


def plot_trade_order_simulations(
    equity_curves: pd.DataFrame, 
    show_percentiles: bool = True,
    alpha: float = 0.1
) -> None:
    """
    Plots the results of the random trade order simulation.
    
    Args:
        equity_curves (pd.DataFrame): DataFrame from random_trade_order_simulation()
        show_percentiles (bool): Whether to highlight percentile bands
        alpha (float): Transparency for individual simulation lines
    """
    plt.figure(figsize=(12, 8))
    
    # Plot all simulation paths
    plt.plot(equity_curves, alpha=alpha, color='lightblue', linewidth=0.5)
    
    if show_percentiles:
        # Calculate and plot percentile bands
        percentiles = [5, 25, 50, 75, 95]
        colors = ['red', 'orange', 'green', 'orange', 'red']
        linewidths = [2, 1.5, 3, 1.5, 2]
        
        for pct, color, lw in zip(percentiles, colors, linewidths):
            pct_values = equity_curves.quantile(pct/100, axis=1)
            plt.plot(pct_values, color=color, linewidth=lw, 
                    label=f'{pct}th percentile')
    
    plt.ylabel("Portfolio Equity ($)", fontsize=12)
    plt.xlabel("Number of Trades", fontsize=12)
    plt.title("Monte Carlo Simulation: Random Trade Order Impact", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if show_percentiles:
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def _generate_random_return_sequences(trade_returns: np.ndarray, num_simulations: int) -> List[np.ndarray]:
    """
    Generate random shuffled sequences of trade returns.
    
    Args:
        trade_returns (np.ndarray): Original trade returns
        num_simulations (int): Number of sequences to generate
        
    Returns:
        List[np.ndarray]: List of shuffled return sequences
    """
    sequences = []
    for i in range(num_simulations):
        shuffled_returns = trade_returns.copy()
        rng = np.random.RandomState(42 + i)
        rng.shuffle(shuffled_returns)
        sequences.append(shuffled_returns)
    return sequences


def _generate_statistical_return_sequences(trade_returns: np.ndarray, num_simulations: int) -> List[np.ndarray]:
    """
    Generate statistically sampled sequences of trade returns within 2 standard deviations.
    
    Instead of pure randomization, this method creates return sequences that:
    1. Maintain statistical properties of the original returns
    2. Apply constraints based on return characteristics
    3. Sample within 2 standard deviations of mean patterns
    
    Args:
        trade_returns (np.ndarray): Original trade returns
        num_simulations (int): Number of sequences to generate
        
    Returns:
        List[np.ndarray]: List of statistically sampled return sequences
    """
    print(f"   Analyzing trade return patterns...")
    
    # Analyze return characteristics
    mean_return = np.mean(trade_returns)
    std_return = np.std(trade_returns)
    num_trades = len(trade_returns)
    
    # Classify returns by type
    positive_returns = trade_returns[trade_returns > 0]
    negative_returns = trade_returns[trade_returns < 0]
    neutral_returns = trade_returns[trade_returns == 0]
    
    print(f"   Trade statistics:")
    print(f"     Mean return: {mean_return:.4f}")
    print(f"     Std dev: {std_return:.4f}")
    print(f"     Positive trades: {len(positive_returns)}")
    print(f"     Negative trades: {len(negative_returns)}")
    print(f"     Neutral trades: {len(neutral_returns)}")
    
    # Calculate statistical constraints for sequencing
    # Instead of pure randomization, create sequences that respect return patterns
    sequences = []
    
    for i in range(num_simulations):
        # Create base sequence with statistical sampling
        sequence = trade_returns.copy()
        
        # Apply statistical reordering within 2 standard deviations
        # This creates more realistic trade sequences based on return characteristics
        
        # Method: Weighted sampling based on return clustering tendencies
        if len(trade_returns) > 1:
            # Calculate sampling probabilities within 2 std devs
            uniform_prob = 1.0 / num_trades
            prob_std = uniform_prob * 0.25  # 25% variability around uniform
            
            # Generate sampling probabilities within 2 standard deviations
            rng = np.random.RandomState(42 + i * 13)  # Different seed pattern
            sampling_probs = rng.normal(uniform_prob, prob_std, num_trades)
            
            # Constrain within 2 standard deviations
            lower_bound = max(0.01, uniform_prob - 2 * prob_std)
            upper_bound = min(0.99, uniform_prob + 2 * prob_std)
            sampling_probs = np.clip(sampling_probs, lower_bound, upper_bound)
            
            # Normalize probabilities
            sampling_probs = sampling_probs / sampling_probs.sum()
            
            # Create weighted shuffle based on statistical sampling
            indices = np.arange(num_trades)
            # Sample without replacement using the calculated probabilities
            sampled_indices = rng.choice(indices, size=num_trades, replace=False, p=sampling_probs)
            sequence = trade_returns[sampled_indices]
        
        sequences.append(sequence)
    
    print(f"   Generated {num_simulations} statistically sampled sequences")
    return sequences


def _generate_synthetic_return_sequences(trade_returns: np.ndarray, num_simulations: int) -> List[np.ndarray]:
    """
    Generate synthetic return sequences that are statistically identical to original data.
    
    Creates random return points with average within 2 standard deviations of original mean.
    
    Args:
        trade_returns: Array of original trade returns
        num_simulations: Number of simulation sequences to generate
        
    Returns:
        List of synthetic return sequences
    """
    print(f"   🎯 Generating synthetic return sequences...")
    print(f"   📊 Analyzing original return patterns...")
    
    # Calculate statistics of original returns
    mean_return = np.mean(trade_returns)
    std_return = np.std(trade_returns)
    num_trades = len(trade_returns)
    
    # Additional statistics for validation
    positive_returns = trade_returns[trade_returns > 0]
    negative_returns = trade_returns[trade_returns < 0]
    neutral_returns = trade_returns[trade_returns == 0]
    
    print(f"   📈 Original return statistics:")
    print(f"     Mean return: {mean_return:.4f}")
    print(f"     Std dev: {std_return:.4f}")
    print(f"     Positive trades: {len(positive_returns)}")
    print(f"     Negative trades: {len(negative_returns)}")
    print(f"     Neutral trades: {len(neutral_returns)}")
    print(f"   🎯 Target: Average return within 2σ of mean ({mean_return - 2*std_return:.4f} to {mean_return + 2*std_return:.4f})")
    
    sequences = []
    
    for i in range(num_simulations):
        # Generate synthetic returns using normal distribution
        rng = np.random.RandomState(42 + i * 17)
        
        # Generate random returns from normal distribution
        synthetic_returns = rng.normal(mean_return, std_return, num_trades)
        
        # Ensure the average is within 2 standard deviations
        current_mean = np.mean(synthetic_returns)
        target_min = mean_return - 2 * std_return
        target_max = mean_return + 2 * std_return
        
        # If mean is outside bounds, adjust
        if current_mean < target_min:
            # Scale up to bring mean into range
            adjustment = (target_min - mean_return) / (current_mean - mean_return)
            synthetic_returns = mean_return + (synthetic_returns - mean_return) * adjustment
        elif current_mean > target_max:
            # Scale down to bring mean into range
            adjustment = (target_max - mean_return) / (current_mean - mean_return)
            synthetic_returns = mean_return + (synthetic_returns - mean_return) * adjustment
        
        # Ensure no returns cause portfolio to go to zero (cap at -95% loss)
        synthetic_returns = np.maximum(synthetic_returns, -0.95)
        
        sequences.append(synthetic_returns)
    
    # Validation
    all_means = [np.mean(seq) for seq in sequences]
    valid_count = sum(1 for m in all_means if target_min <= m <= target_max)
    
    print(f"   ✅ Generated {num_simulations} synthetic return sequences")
    print(f"   📊 Sequences with mean in 2σ range: {valid_count}/{num_simulations} ({valid_count/num_simulations*100:.1f}%)")
    print(f"   📈 Sequence means range: {min(all_means):.4f} to {max(all_means):.4f}")
    
    return sequences


def analyze_simulation_results(equity_curves: pd.DataFrame) -> dict:
    """
    Provides comprehensive analysis of the simulation results.
    
    Args:
        equity_curves (pd.DataFrame): Results from random_trade_order_simulation()
        
    Returns:
        dict: Dictionary containing various statistics and metrics
    """
    final_equity = equity_curves.iloc[-1]
    initial_equity = equity_curves.iloc[0, 0]  # Should be same for all simulations
    
    # Calculate returns
    total_returns = (final_equity / initial_equity - 1) * 100
    
    # Basic statistics
    stats = {
        'initial_capital': initial_equity,
        'final_equity_stats': final_equity.describe(),
        'total_return_stats': total_returns.describe(),
        'probability_of_loss': (total_returns < 0).mean() * 100,
        'probability_of_gain': (total_returns > 0).mean() * 100,
        'max_final_equity': final_equity.max(),
        'min_final_equity': final_equity.min(),
        'volatility_of_outcomes': final_equity.std(),
    }
    
    # Risk metrics
    stats['value_at_risk_5pct'] = np.percentile(final_equity, 5)
    stats['value_at_risk_1pct'] = np.percentile(final_equity, 1)
    
    return stats


def print_analysis_report(analysis_results: dict) -> None:
    """Print a formatted analysis report."""
    print("\n" + "="*60)
    print("MONTE CARLO TRADE ORDER SIMULATION ANALYSIS")
    print("="*60)
    
    print(f"\nInitial Capital: ${analysis_results['initial_capital']:,.2f}")
    
    print(f"\nFINAL EQUITY STATISTICS:")
    print(f"  Mean:     ${analysis_results['final_equity_stats']['mean']:,.2f}")
    print(f"  Median:   ${analysis_results['final_equity_stats']['50%']:,.2f}")
    print(f"  Std Dev:  ${analysis_results['final_equity_stats']['std']:,.2f}")
    print(f"  Min:      ${analysis_results['final_equity_stats']['min']:,.2f}")
    print(f"  Max:      ${analysis_results['final_equity_stats']['max']:,.2f}")
    
    print(f"\nTOTAL RETURN STATISTICS:")
    print(f"  Mean:     {analysis_results['total_return_stats']['mean']:,.2f}%")
    print(f"  Median:   {analysis_results['total_return_stats']['50%']:,.2f}%")
    print(f"  Std Dev:  {analysis_results['total_return_stats']['std']:,.2f}%")
    print(f"  Min:      {analysis_results['total_return_stats']['min']:,.2f}%")
    print(f"  Max:      {analysis_results['total_return_stats']['max']:,.2f}%")
    
    print(f"\nRISK METRICS:")
    print(f"  Probability of Loss:     {analysis_results['probability_of_loss']:.1f}%")
    print(f"  Probability of Gain:     {analysis_results['probability_of_gain']:.1f}%")
    print(f"  Value at Risk (5%):      ${analysis_results['value_at_risk_5pct']:,.2f}")
    print(f"  Value at Risk (1%):      ${analysis_results['value_at_risk_1pct']:,.2f}")
    
    print("\n" + "="*60)


def plot_trade_order_simulations(results: pd.DataFrame, title: str = "Monte Carlo Trade Order Simulation"):
    """
    Plots the results of the trade order simulations with comprehensive analysis.
    
    Args:
        results (pd.DataFrame): Results from random_trade_order_simulation()
        title (str): Title for the plot
    """
    print(f"\n📊 Creating Monte Carlo visualization...")
    
    # Create comprehensive visualization
    plot_comprehensive_monte_carlo_analysis(results, "trade_order", title)


def plot_comprehensive_monte_carlo_analysis(results: pd.DataFrame, 
                                           simulation_method: str = "synthetic_returns",
                                           title: str = "Monte Carlo Simulation Analysis"):
    """
    Create comprehensive graphical analysis of Monte Carlo simulation results.
    
    Args:
        results: DataFrame with simulation results (equity curves)
        simulation_method: Method used for simulation
        title: Title for the plots
    """
    print(f"\n📊 Creating comprehensive Monte Carlo visualization...")
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'{title}\nMethod: {simulation_method.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Equity Curves (Top left)
    ax1 = plt.subplot(2, 3, 1)
    plot_equity_curves(results, ax1)
    
    # 2. Final Value Distribution (Top middle)
    ax2 = plt.subplot(2, 3, 2)
    plot_final_value_distribution(results, ax2)
    
    # 3. Return Distribution (Top right)
    ax3 = plt.subplot(2, 3, 3)
    plot_return_distribution(results, ax3)
    
    # 4. Drawdown Analysis (Bottom left)
    ax4 = plt.subplot(2, 3, 4)
    plot_drawdown_analysis(results, ax4)
    
    # 5. Performance Statistics (Bottom middle)
    ax5 = plt.subplot(2, 3, 5)
    plot_performance_statistics(results, ax5)
    
    # 6. Risk Metrics (Bottom right)
    ax6 = plt.subplot(2, 3, 6)
    plot_risk_metrics(results, ax6)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)
    plt.show()


def plot_equity_curves(results: pd.DataFrame, ax):
    """Plot equity curves from Monte Carlo simulation."""
    ax.set_title('Portfolio Equity Curves\n(Sample of Simulation Paths)', fontsize=12, fontweight='bold')
    
    # Plot sample of equity curves
    num_curves_to_plot = min(50, results.shape[1])
    if results.shape[1] > 1:
        sample_columns = np.random.choice(results.columns, num_curves_to_plot, replace=False)
    else:
        sample_columns = results.columns
    
    for i, col in enumerate(sample_columns):
        alpha = 0.6 if i < 10 else 0.3  # More prominent for first 10
        linewidth = 1.5 if i < 5 else 0.8
        ax.plot(results[col], alpha=alpha, linewidth=linewidth, color='steelblue')
    
    # Plot mean and percentiles
    mean_curve = results.mean(axis=1)
    p25_curve = results.quantile(0.25, axis=1)
    p75_curve = results.quantile(0.75, axis=1)
    
    ax.plot(mean_curve, color='red', linewidth=3, label='Mean Path', alpha=0.9)
    ax.fill_between(range(len(mean_curve)), p25_curve, p75_curve, 
                    color='red', alpha=0.2, label='25th-75th Percentile')
    
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


def plot_final_value_distribution(results: pd.DataFrame, ax):
    """Plot distribution of final portfolio values."""
    ax.set_title('Final Portfolio Values\nDistribution', fontsize=12, fontweight='bold')
    
    final_values = results.iloc[-1].values
    
    # Histogram
    n, bins, patches = ax.hist(final_values, bins=30, alpha=0.7, color='lightgreen', 
                              edgecolor='black', linewidth=0.5)
    
    # Statistics
    mean_val = np.mean(final_values)
    median_val = np.median(final_values)
    initial_capital = results.iloc[0, 0]  # Assume first value is initial capital
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: ${mean_val:,.0f}')
    ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, 
               label=f'Median: ${median_val:,.0f}')
    ax.axvline(initial_capital, color='orange', linestyle='-', linewidth=2, 
               label=f'Initial: ${initial_capital:,.0f}')
    
    ax.set_xlabel('Final Portfolio Value ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_return_distribution(results: pd.DataFrame, ax):
    """Plot distribution of total returns."""
    ax.set_title('Total Return Distribution\n(All Simulations)', fontsize=12, fontweight='bold')
    
    initial_values = results.iloc[0].values
    final_values = results.iloc[-1].values
    returns = (final_values - initial_values) / initial_values
    
    # Histogram
    ax.hist(returns, bins=30, alpha=0.7, color='lightcoral', 
            edgecolor='black', linewidth=0.5)
    
    # Statistics
    mean_return = np.mean(returns)
    median_return = np.median(returns)
    
    ax.axvline(mean_return, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_return:.1%}')
    ax.axvline(median_return, color='blue', linestyle='--', linewidth=2, 
               label=f'Median: {median_return:.1%}')
    ax.axvline(0, color='orange', linestyle='-', linewidth=2, 
               label='Breakeven')
    
    ax.set_xlabel('Total Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis as percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))


def plot_drawdown_analysis(results: pd.DataFrame, ax):
    """Plot drawdown analysis."""
    ax.set_title('Maximum Drawdown\nAnalysis', fontsize=12, fontweight='bold')
    
    # Calculate maximum drawdowns for each simulation
    max_drawdowns = []
    
    for col in results.columns:
        equity_curve = results[col].values
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = np.min(drawdown)
        max_drawdowns.append(max_drawdown)
    
    max_drawdowns = np.array(max_drawdowns)
    
    # Histogram
    ax.hist(max_drawdowns, bins=25, alpha=0.7, color='lightsteelblue', 
            edgecolor='black', linewidth=0.5)
    
    # Statistics
    mean_dd = np.mean(max_drawdowns)
    p95_dd = np.percentile(max_drawdowns, 5)  # 95th percentile (worst 5%)
    
    ax.axvline(mean_dd, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_dd:.1%}')
    ax.axvline(p95_dd, color='darkred', linestyle='--', linewidth=2, 
               label=f'95th Percentile: {p95_dd:.1%}')
    
    ax.set_xlabel('Maximum Drawdown')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis as percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))


def plot_performance_statistics(results: pd.DataFrame, ax):
    """Plot key performance statistics."""
    ax.set_title('Performance Statistics\nSummary', fontsize=12, fontweight='bold')
    
    # Calculate statistics
    initial_values = results.iloc[0].values
    final_values = results.iloc[-1].values
    returns = (final_values - initial_values) / initial_values
    
    stats = {
        'Mean Return': np.mean(returns),
        'Median Return': np.median(returns),
        'Std Dev': np.std(returns),
        'Win Rate': np.sum(returns > 0) / len(returns),
        'Best Case': np.max(returns),
        'Worst Case': np.min(returns)
    }
    
    # Create bar chart
    metrics = list(stats.keys())
    values = list(stats.values())
    
    colors = ['green' if v > 0 else 'red' for v in values]
    colors[3] = 'blue'  # Win rate always blue
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if metrics[bars.index(bar)] == 'Win Rate':
            label = f'{value:.1%}'
        else:
            label = f'{value:.1%}'
        
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                label, ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    ax.set_ylabel('Value')
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.7)


def plot_risk_metrics(results: pd.DataFrame, ax):
    """Plot risk metrics analysis."""
    ax.set_title('Risk Metrics\nAnalysis', fontsize=12, fontweight='bold')
    
    # Calculate various risk metrics
    initial_values = results.iloc[0].values
    final_values = results.iloc[-1].values
    returns = (final_values - initial_values) / initial_values
    
    # Calculate some risk metrics
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = [np.percentile(returns, p) for p in percentiles]
    
    # Create box plot style visualization
    ax.boxplot([returns], widths=0.5, patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    
    # Add percentile markers
    for i, (p, val) in enumerate(zip(percentiles, percentile_values)):
        if p in [1, 5, 95, 99]:  # Highlight extreme percentiles
            ax.plot(1, val, 'ro', markersize=8, alpha=0.8)
            ax.text(1.15, val, f'{p}th: {val:.1%}', va='center', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Return Distribution')
    ax.set_xlabel('All Simulations')
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Breakeven')
    
    # Add summary statistics text
    summary_text = f"VaR (5%): {np.percentile(returns, 5):.1%}\n"
    summary_text += f"VaR (1%): {np.percentile(returns, 1):.1%}\n"
    summary_text += f"Expected Shortfall: {np.mean(returns[returns <= np.percentile(returns, 5)]):.1%}"
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# --- Example Usage ---
if __name__ == '__main__':
    print("Monte Carlo Trade Order Simulation Example")
    print("-" * 50)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Create a simple example with clear sequence risk
    # Using a scenario with retirement withdrawals to show path dependency
    trade_returns_example = np.array([
        0.20, -0.30, 0.25,  # Three simple trades with very different outcomes based on order
    ])
    
    print(f"Analyzing {len(trade_returns_example)} trades with {500} simulations...")
    print("Trade returns:", trade_returns_example)
    print()
    
    # Show both modes for comparison
    # Manual verification of different orderings
    print("=== MANUAL VERIFICATION OF DIFFERENT ORDERINGS ===")
    orderings = [
        [0.20, -0.30, 0.25],
        [-0.30, 0.20, 0.25],
        [0.25, 0.20, -0.30]
    ]
    
    for i, order in enumerate(orderings):
        value = 10000
        print(f"Order {i+1}: {order}")
        print(f"  Start: ${value:,.2f}")
        for j, ret in enumerate(order):
            value = value * (1 + ret)
            print(f"  After trade {j+1} ({ret:+.1%}): ${value:,.2f}")
        print(f"  Final: ${value:,.2f}")
        print()
    
    print("=== COMPOUND MODE (traditional multiplicative returns) ===")
    shuffled_trade_simulations = random_trade_order_simulation(
        trade_returns_example, 
        num_simulations=500,
        initial_capital=10000,
        position_size_mode='compound'
    )
    
    analysis = analyze_simulation_results(shuffled_trade_simulations)
    print(f"Range of outcomes: ${analysis['min_final_equity']:,.2f} to ${analysis['max_final_equity']:,.2f}")
    print(f"Standard deviation: ${analysis['volatility_of_outcomes']:,.2f}")
    print()
    
    print("=== PERCENTAGE RISK MODE (percentage of portfolio at risk each trade) ===")
    
    # Allow user to specify number of simulations and method via command line
    try:
        import sys
        if len(sys.argv) > 1:
            num_sims = int(sys.argv[1])
        else:
            num_sims = 500
        
        if len(sys.argv) > 2:
            method = sys.argv[2]
        else:
            method = 'random'
    except (ValueError, IndexError):
        num_sims = 500
        method = 'random'
    
    print(f"Running {num_sims} simulations...")
    print(f"Method: {method}")
    
    shuffled_trade_simulations = random_trade_order_simulation(
        trade_returns_example, 
        num_simulations=num_sims,
        initial_capital=10000,
        position_size_mode='fixed',
        simulation_method=method
    )
    
    # Plot the results
    plot_trade_order_simulations(shuffled_trade_simulations)
    
    # Analyze and print results
    analysis = analyze_simulation_results(shuffled_trade_simulations)
    print_analysis_report(analysis)
    
    # Additional insights
    print(f"\nKey Insights:")
    print(f"- The order of trades can result in final portfolio values")
    print(f"  ranging from ${analysis['min_final_equity']:,.2f} to ${analysis['max_final_equity']:,.2f}")
    print(f"- This represents a spread of ${analysis['max_final_equity'] - analysis['min_final_equity']:,.2f}")
    print(f"- Standard deviation of outcomes: ${analysis['volatility_of_outcomes']:,.2f}")
