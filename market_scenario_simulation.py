"""
Market Scenario Monte Carlo Simulation

Generates different possible market paths that could have occurred,
allowing you to see how your trading strategy would have performed
in various market environments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
import warnings

def generate_market_scenarios(
    historical_data: pd.DataFrame,
    num_scenarios: int = 1000,
    scenario_length: int = 252,
    market_regimes: List[str] = None
) -> Dict[str, np.ndarray]:
    """
    Generate different market scenarios based on historical data patterns.
    
    Each scenario represents a different possible market path that could have occurred,
    allowing strategy testing across various market conditions.
    
    Args:
        historical_data: Historical price/return data
        num_scenarios: Number of different market scenarios to generate
        scenario_length: Length of each scenario (trading days)
        market_regimes: List of market types to simulate ['bull', 'bear', 'sideways', 'volatile']
        
    Returns:
        Dict containing different market scenarios
    """
    if market_regimes is None:
        market_regimes = ['bull', 'bear', 'sideways', 'volatile']
    
    print(f"🌍 GENERATING MARKET SCENARIOS")
    print("="*50)
    print(f"Purpose: Create different possible market paths your strategy could encounter")
    print(f"Scenarios: {num_scenarios}")
    print(f"Length: {scenario_length} trading days each")
    print(f"Market types: {', '.join(market_regimes)}")
    
    # Calculate base statistics from historical data
    if 'Close' in historical_data.columns:
        prices = historical_data['Close'].values
    else:
        prices = historical_data.iloc[:, 0].values  # Assume first column is price
    
    returns = np.diff(prices) / prices[:-1]
    
    base_stats = {
        'mean_return': np.mean(returns),
        'volatility': np.std(returns),
        'skewness': calculate_skewness(returns),
        'kurtosis': calculate_kurtosis(returns)
    }
    
    print(f"\n📊 Base market statistics (from historical data):")
    print(f"   Mean daily return: {base_stats['mean_return']:.4f}")
    print(f"   Daily volatility: {base_stats['volatility']:.4f}")
    print(f"   Skewness: {base_stats['skewness']:.3f}")
    print(f"   Kurtosis: {base_stats['kurtosis']:.3f}")
    
    scenarios = {}
    
    # Generate scenarios for each market regime
    scenarios_per_regime = num_scenarios // len(market_regimes)
    
    for regime in market_regimes:
        print(f"\n🎯 Generating {scenarios_per_regime} scenarios for {regime} market...")
        
        regime_scenarios = []
        
        for i in range(scenarios_per_regime):
            if regime == 'bull':
                # Bull market: Higher mean return, moderate volatility
                mean_ret = base_stats['mean_return'] * 2.0  # 2x normal returns
                volatility = base_stats['volatility'] * 0.8  # 20% less volatile
                
            elif regime == 'bear':
                # Bear market: Negative returns, higher volatility
                mean_ret = base_stats['mean_return'] * -1.5  # Negative returns
                volatility = base_stats['volatility'] * 1.5  # 50% more volatile
                
            elif regime == 'sideways':
                # Sideways market: Low returns, low volatility
                mean_ret = base_stats['mean_return'] * 0.1  # Minimal returns
                volatility = base_stats['volatility'] * 0.6  # Lower volatility
                
            elif regime == 'volatile':
                # Volatile market: Normal returns, high volatility
                mean_ret = base_stats['mean_return']
                volatility = base_stats['volatility'] * 2.0  # Double volatility
            
            # Generate returns for this scenario
            np.random.seed(42 + i * len(market_regimes) + hash(regime) % 1000)
            scenario_returns = np.random.normal(mean_ret, volatility, scenario_length)
            
            # Add some realistic market features
            scenario_returns = add_market_features(scenario_returns, regime)
            
            regime_scenarios.append(scenario_returns)
        
        scenarios[regime] = np.array(regime_scenarios)
        
        # Statistics for this regime
        regime_mean_return = np.mean([np.mean(scenario) for scenario in regime_scenarios])
        regime_volatility = np.mean([np.std(scenario) for scenario in regime_scenarios])
        
        print(f"   ✅ {regime.capitalize()} scenarios: mean return {regime_mean_return:.4f}, volatility {regime_volatility:.4f}")
    
    # Add mixed scenarios (random combinations)
    remaining_scenarios = num_scenarios - len(market_regimes) * scenarios_per_regime
    if remaining_scenarios > 0:
        print(f"\n🎯 Generating {remaining_scenarios} mixed market scenarios...")
        mixed_scenarios = []
        
        for i in range(remaining_scenarios):
            # Create scenario with changing regimes
            scenario_returns = generate_mixed_regime_scenario(
                base_stats, scenario_length, market_regimes, seed=42 + i + 10000
            )
            mixed_scenarios.append(scenario_returns)
        
        scenarios['mixed'] = np.array(mixed_scenarios)
    
    print(f"\n✅ Generated {num_scenarios} total market scenarios")
    return scenarios, base_stats


def add_market_features(returns: np.ndarray, regime: str) -> np.ndarray:
    """Add realistic market features to return series."""
    
    # Add some autocorrelation (market momentum/mean reversion)
    if regime == 'bull':
        # Bull markets tend to have momentum
        for i in range(1, len(returns)):
            if returns[i-1] > 0:
                returns[i] += returns[i-1] * 0.1  # Small momentum effect
                
    elif regime == 'bear':
        # Bear markets can have volatility clustering
        for i in range(1, len(returns)):
            if abs(returns[i-1]) > np.std(returns):
                returns[i] *= 1.2  # Increase volatility after big moves
    
    elif regime == 'volatile':
        # Add volatility clustering
        for i in range(2, len(returns)):
            if abs(returns[i-1]) > abs(returns[i-2]):
                returns[i] *= 1.3
    
    # Add occasional market shocks
    shock_probability = 0.02  # 2% chance per day
    shock_indices = np.random.random(len(returns)) < shock_probability
    shock_magnitude = np.random.normal(0, np.std(returns) * 3, np.sum(shock_indices))
    returns[shock_indices] += shock_magnitude
    
    return returns


def generate_mixed_regime_scenario(base_stats: dict, length: int, regimes: List[str], seed: int) -> np.ndarray:
    """Generate a scenario that switches between different market regimes."""
    np.random.seed(seed)
    
    scenario_returns = []
    current_pos = 0
    
    while current_pos < length:
        # Choose random regime and duration
        regime = np.random.choice(regimes)
        duration = np.random.randint(20, 80)  # 20-80 day periods
        
        if current_pos + duration > length:
            duration = length - current_pos
        
        # Generate returns for this regime period
        if regime == 'bull':
            mean_ret = base_stats['mean_return'] * 1.5
            volatility = base_stats['volatility'] * 0.9
        elif regime == 'bear':
            mean_ret = base_stats['mean_return'] * -1.2
            volatility = base_stats['volatility'] * 1.4
        elif regime == 'sideways':
            mean_ret = base_stats['mean_return'] * 0.2
            volatility = base_stats['volatility'] * 0.7
        else:  # volatile
            mean_ret = base_stats['mean_return']
            volatility = base_stats['volatility'] * 1.8
        
        period_returns = np.random.normal(mean_ret, volatility, duration)
        scenario_returns.extend(period_returns)
        current_pos += duration
    
    return np.array(scenario_returns[:length])


def test_strategy_across_scenarios(
    strategy_function,
    market_scenarios: Dict[str, np.ndarray],
    initial_capital: float = 10000
) -> Dict[str, Dict]:
    """
    Test a trading strategy across different market scenarios.
    
    Args:
        strategy_function: Function that takes price data and returns trades
        market_scenarios: Dict of market scenario arrays
        initial_capital: Starting capital
        
    Returns:
        Results for each market regime
    """
    print(f"\n🚀 TESTING STRATEGY ACROSS MARKET SCENARIOS")
    print("="*60)
    print(f"Purpose: See how your strategy performs in different possible market conditions")
    
    all_results = {}
    
    for regime_name, scenarios in market_scenarios.items():
        print(f"\n📊 Testing in {regime_name} market scenarios...")
        
        regime_results = []
        
        for i, scenario_returns in enumerate(scenarios):
            # Convert returns to prices (starting at $100)
            prices = [100]
            for ret in scenario_returns:
                prices.append(prices[-1] * (1 + ret))
            
            # Create complete OHLCV data for ML algorithms
            price_series = pd.Series(prices[1:], index=pd.date_range('2024-01-01', periods=len(prices)-1))
            price_data = create_synthetic_ohlcv_from_prices(price_series)
            
            try:
                # Run strategy on this scenario
                trades = strategy_function(price_data)
                
                # Calculate performance
                if trades:
                    trade_returns = [trade['return'] for trade in trades if 'return' in trade]
                    if trade_returns:
                        final_value = initial_capital
                        for ret in trade_returns:
                            final_value *= (1 + ret)
                        
                        total_return = (final_value - initial_capital) / initial_capital
                        num_trades = len(trade_returns)
                        win_rate = sum(1 for ret in trade_returns if ret > 0) / len(trade_returns)
                        
                        regime_results.append({
                            'scenario_index': i,
                            'final_value': final_value,
                            'total_return': total_return,
                            'num_trades': num_trades,
                            'win_rate': win_rate,
                            'trade_returns': trade_returns
                        })
                
            except Exception as e:
                print(f"   ⚠️  Error in scenario {i}: {e}")
                continue
        
        if regime_results:
            # Calculate statistics for this regime
            final_values = [r['final_value'] for r in regime_results]
            total_returns = [r['total_return'] for r in regime_results]
            
            regime_stats = {
                'num_scenarios': len(regime_results),
                'mean_final_value': np.mean(final_values),
                'std_final_value': np.std(final_values),
                'min_final_value': np.min(final_values),
                'max_final_value': np.max(final_values),
                'mean_return': np.mean(total_returns),
                'std_return': np.std(total_returns),
                'win_rate': np.mean([r['win_rate'] for r in regime_results]),
                'results': regime_results
            }
            
            all_results[regime_name] = regime_stats
            
            print(f"   ✅ {regime_name.capitalize()} market results:")
            print(f"      Scenarios tested: {regime_stats['num_scenarios']}")
            print(f"      Final value range: ${regime_stats['min_final_value']:,.0f} to ${regime_stats['max_final_value']:,.0f}")
            print(f"      Average return: {regime_stats['mean_return']:.2%}")
            print(f"      Return std dev: {regime_stats['std_return']:.2%}")
            print(f"      Average win rate: {regime_stats['win_rate']:.1%}")
    
    return all_results


def create_synthetic_ohlcv_from_prices(prices: pd.Series, base_volume: float = 1000000) -> pd.DataFrame:
    """
    Create synthetic OHLCV data from price series for ML algorithm compatibility.
    
    Args:
        prices: Series of closing prices
        base_volume: Base volume to use for synthetic volume data
        
    Returns:
        DataFrame with OHLCV columns
    """
    df = pd.DataFrame(index=prices.index)
    
    # Use price as Close
    df['Close'] = prices
    
    # Create synthetic OHLC based on price movements
    returns = prices.pct_change().fillna(0)
    volatility = returns.rolling(20, min_periods=1).std().fillna(0.02)
    
    # Generate synthetic intraday ranges
    np.random.seed(42)
    
    # High: Close + random upward movement
    high_factor = np.random.uniform(0.5, 2.0, len(prices))
    df['High'] = prices * (1 + volatility * high_factor)
    
    # Low: Close - random downward movement  
    low_factor = np.random.uniform(0.5, 2.0, len(prices))
    df['Low'] = prices * (1 - volatility * low_factor)
    
    # Open: Previous close + gap
    gap_factor = np.random.normal(0, 0.1, len(prices))
    df['Open'] = prices.shift(1) * (1 + volatility * gap_factor)
    df['Open'].iloc[0] = prices.iloc[0]  # First open = first close
    
    # Ensure OHLC consistency (High >= max(O,C), Low <= min(O,C))
    df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
    df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))
    
    # Synthetic volume based on price volatility
    volume_factor = 1 + np.abs(returns) * 5  # Higher volume on bigger moves
    df['Volume'] = base_volume * volume_factor
    
    return df


def calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness of data."""
    mean_val = np.mean(data)
    std_val = np.std(data)
    return np.mean(((data - mean_val) / std_val) ** 3)


def calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate kurtosis of data."""
    mean_val = np.mean(data)
    std_val = np.std(data)
    return np.mean(((data - mean_val) / std_val) ** 4) - 3


def simple_moving_average_strategy(price_data: pd.DataFrame) -> List[Dict]:
    """
    Simple example strategy for demonstration.
    
    Generates buy/sell signals based on moving average crossover.
    """
    prices = price_data['Close'].values
    
    # Calculate moving averages
    short_window = 10
    long_window = 30
    
    if len(prices) < long_window:
        return []
    
    short_ma = pd.Series(prices).rolling(window=short_window).mean().values
    long_ma = pd.Series(prices).rolling(window=long_window).mean().values
    
    trades = []
    position = None
    entry_price = None
    
    for i in range(long_window, len(prices)):
        # Buy signal: short MA crosses above long MA
        if (short_ma[i] > long_ma[i] and 
            short_ma[i-1] <= long_ma[i-1] and 
            position != 'long'):
            
            if position == 'short' and entry_price is not None:
                # Close short position
                ret = (entry_price - prices[i]) / entry_price
                trades.append({
                    'type': 'short_close',
                    'price': prices[i],
                    'return': ret
                })
            
            # Open long position
            position = 'long'
            entry_price = prices[i]
            trades.append({
                'type': 'long_open',
                'price': prices[i]
            })
        
        # Sell signal: short MA crosses below long MA
        elif (short_ma[i] < long_ma[i] and 
              short_ma[i-1] >= long_ma[i-1] and 
              position != 'short'):
            
            if position == 'long' and entry_price is not None:
                # Close long position
                ret = (prices[i] - entry_price) / entry_price
                trades.append({
                    'type': 'long_close',
                    'price': prices[i],
                    'return': ret
                })
            
            # Open short position
            position = 'short'
            entry_price = prices[i]
            trades.append({
                'type': 'short_open',
                'price': prices[i]
            })
    
    return trades


def plot_market_scenarios_analysis(scenarios: Dict[str, np.ndarray], 
                                 results: Dict[str, Dict], 
                                 base_stats: Dict):
    """
    Create comprehensive graphical analysis of market scenarios and strategy performance.
    
    Args:
        scenarios: Generated market scenarios
        results: Strategy performance results
        base_stats: Base market statistics
    """
    print(f"\n📊 Creating comprehensive graphical analysis...")
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Monte Carlo Market Scenario Analysis\nStrategy Performance Across Different Possible Market Routes', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Define colors for market regimes
    regime_colors = {
        'bull': '#2E8B57',      # Sea Green
        'bear': '#DC143C',      # Crimson
        'sideways': '#4169E1',  # Royal Blue
        'volatile': '#FF8C00',  # Dark Orange
        'mixed': '#9932CC'      # Dark Orchid
    }
    
    # 1. Market Scenario Paths (Top row, left)
    ax1 = plt.subplot(3, 4, 1)
    plot_scenario_paths(scenarios, ax1, regime_colors)
    
    # 2. Market Return Distributions (Top row, middle-left)
    ax2 = plt.subplot(3, 4, 2)
    plot_return_distributions(scenarios, ax2, regime_colors)
    
    # 3. Strategy Performance by Market Type (Top row, middle-right)
    ax3 = plt.subplot(3, 4, 3)
    plot_performance_by_market(results, ax3, regime_colors)
    
    # 4. Performance Distribution (Top row, right)
    ax4 = plt.subplot(3, 4, 4)
    plot_performance_distributions(results, ax4, regime_colors)
    
    # 5. Cumulative Performance Paths (Middle row, spans 2 columns)
    ax5 = plt.subplot(3, 4, (5, 6))
    plot_cumulative_performance_paths(results, scenarios, ax5, regime_colors)
    
    # 6. Risk-Return Scatter (Middle row, right side, spans 2 columns)
    ax6 = plt.subplot(3, 4, (7, 8))
    plot_risk_return_scatter(results, ax6, regime_colors)
    
    # 7. Win Rate Analysis (Bottom row, left)
    ax7 = plt.subplot(3, 4, 9)
    plot_win_rate_analysis(results, ax7, regime_colors)
    
    # 8. Scenario Statistics Comparison (Bottom row, middle-left)
    ax8 = plt.subplot(3, 4, 10)
    plot_scenario_statistics(scenarios, base_stats, ax8, regime_colors)
    
    # 9. Performance Heatmap (Bottom row, middle-right)
    ax9 = plt.subplot(3, 4, 11)
    plot_performance_heatmap(results, ax9)
    
    # 10. Summary Statistics (Bottom row, right)
    ax10 = plt.subplot(3, 4, 12)
    plot_summary_statistics(results, ax10, regime_colors)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)
    plt.show()


def plot_scenario_paths(scenarios: Dict[str, np.ndarray], ax, regime_colors: Dict):
    """Plot sample market scenario paths."""
    ax.set_title('Sample Market Scenario Paths\n(Different Possible Market Routes)', fontsize=12, fontweight='bold')
    
    for regime_name, regime_scenarios in scenarios.items():
        color = regime_colors.get(regime_name, 'gray')
        
        # Plot a few sample paths for each regime
        num_samples = min(3, len(regime_scenarios))
        
        for i in range(num_samples):
            returns = regime_scenarios[i]
            # Convert to cumulative price path
            prices = [100]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            alpha = 0.7 if i == 0 else 0.4  # Make first path more prominent
            linewidth = 2 if i == 0 else 1
            label = regime_name.capitalize() if i == 0 else None
            
            ax.plot(prices[1:], color=color, alpha=alpha, linewidth=linewidth, label=label)
    
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Price Level')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_return_distributions(scenarios: Dict[str, np.ndarray], ax, regime_colors: Dict):
    """Plot return distributions for each market regime."""
    ax.set_title('Daily Return Distributions\nby Market Type', fontsize=12, fontweight='bold')
    
    all_returns_by_regime = {}
    
    for regime_name, regime_scenarios in scenarios.items():
        all_returns = []
        for scenario in regime_scenarios:
            all_returns.extend(scenario)
        all_returns_by_regime[regime_name] = all_returns
    
    # Create violin plot
    data_to_plot = []
    labels = []
    colors = []
    
    for regime_name in ['bull', 'bear', 'sideways', 'volatile']:
        if regime_name in all_returns_by_regime:
            data_to_plot.append(all_returns_by_regime[regime_name])
            labels.append(regime_name.capitalize())
            colors.append(regime_colors[regime_name])
    
    if data_to_plot:
        parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), widths=0.7, showmeans=True)
        
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel('Daily Return')
        ax.grid(True, alpha=0.3)


def plot_performance_by_market(results: Dict[str, Dict], ax, regime_colors: Dict):
    """Plot strategy performance by market type."""
    ax.set_title('Strategy Performance\nby Market Type', fontsize=12, fontweight='bold')
    
    regimes = []
    mean_returns = []
    std_returns = []
    colors = []
    
    for regime_name, stats in results.items():
        regimes.append(regime_name.capitalize())
        mean_returns.append(stats['mean_return'] * 100)  # Convert to percentage
        std_returns.append(stats['std_return'] * 100)
        colors.append(regime_colors.get(regime_name, 'gray'))
    
    bars = ax.bar(regimes, mean_returns, color=colors, alpha=0.7, edgecolor='black')
    
    # Add error bars for standard deviation
    ax.errorbar(regimes, mean_returns, yerr=std_returns, fmt='none', color='black', capsize=5)
    
    ax.set_ylabel('Average Return (%)')
    ax.set_xlabel('Market Type')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_returns):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')


def plot_performance_distributions(results: Dict[str, Dict], ax, regime_colors: Dict):
    """Plot distribution of final portfolio values."""
    ax.set_title('Final Portfolio Value\nDistributions', fontsize=12, fontweight='bold')
    
    for regime_name, stats in results.items():
        if 'results' in stats:
            final_values = [r['final_value'] for r in stats['results']]
            color = regime_colors.get(regime_name, 'gray')
            
            ax.hist(final_values, bins=15, alpha=0.6, color=color, 
                   label=f"{regime_name.capitalize()}", edgecolor='black', linewidth=0.5)
    
    ax.axvline(x=10000, color='red', linestyle='--', linewidth=2, label='Initial Capital')
    ax.set_xlabel('Final Portfolio Value ($)')
    ax.set_ylabel('Frequency')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_cumulative_performance_paths(results: Dict[str, Dict], scenarios: Dict[str, np.ndarray], ax, regime_colors: Dict):
    """Plot cumulative performance paths for each market regime."""
    ax.set_title('Cumulative Strategy Performance Paths\nAcross Different Market Scenarios', fontsize=12, fontweight='bold')
    
    for regime_name, stats in results.items():
        if 'results' not in stats:
            continue
            
        color = regime_colors.get(regime_name, 'gray')
        
        # Plot a sample of performance paths
        sample_size = min(5, len(stats['results']))
        
        for i, result in enumerate(stats['results'][:sample_size]):
            if 'trade_returns' in result:
                # Calculate cumulative portfolio value
                portfolio_values = [10000]  # Starting capital
                for trade_return in result['trade_returns']:
                    portfolio_values.append(portfolio_values[-1] * (1 + trade_return))
                
                alpha = 0.8 if i == 0 else 0.4
                linewidth = 2 if i == 0 else 1
                label = f"{regime_name.capitalize()}" if i == 0 else None
                
                ax.plot(portfolio_values, color=color, alpha=alpha, linewidth=linewidth, label=label)
    
    ax.axhline(y=10000, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_risk_return_scatter(results: Dict[str, Dict], ax, regime_colors: Dict):
    """Plot risk-return scatter plot."""
    ax.set_title('Risk-Return Analysis\nBy Market Type', fontsize=12, fontweight='bold')
    
    for regime_name, stats in results.items():
        color = regime_colors.get(regime_name, 'gray')
        
        # Plot each scenario as a point
        if 'results' in stats:
            returns = [r['total_return'] * 100 for r in stats['results']]  # Convert to percentage
            
            # Calculate "risk" as standard deviation of trade returns for each scenario
            risks = []
            for result in stats['results']:
                if 'trade_returns' in result and result['trade_returns']:
                    risk = np.std(result['trade_returns']) * 100
                    risks.append(risk)
                else:
                    risks.append(0)
            
            if len(returns) == len(risks):
                ax.scatter(risks, returns, color=color, alpha=0.7, s=50, 
                          label=regime_name.capitalize(), edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Risk (Std Dev of Trade Returns, %)')
    ax.set_ylabel('Total Return (%)')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_win_rate_analysis(results: Dict[str, Dict], ax, regime_colors: Dict):
    """Plot win rate analysis by market type."""
    ax.set_title('Win Rate Analysis\nby Market Type', fontsize=12, fontweight='bold')
    
    regimes = []
    win_rates = []
    colors = []
    
    for regime_name, stats in results.items():
        regimes.append(regime_name.capitalize())
        win_rates.append(stats['win_rate'] * 100)  # Convert to percentage
        colors.append(regime_colors.get(regime_name, 'gray'))
    
    bars = ax.bar(regimes, win_rates, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Win Rate (%)')
    ax.set_xlabel('Market Type')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% Breakeven')
    
    # Add value labels on bars
    for bar, value in zip(bars, win_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.legend(loc='best', fontsize=10)


def plot_scenario_statistics(scenarios: Dict[str, np.ndarray], base_stats: Dict, ax, regime_colors: Dict):
    """Plot comparison of scenario statistics."""
    ax.set_title('Market Scenario Statistics\nComparison', fontsize=12, fontweight='bold')
    
    regimes = []
    mean_returns = []
    volatilities = []
    
    for regime_name, regime_scenarios in scenarios.items():
        regimes.append(regime_name.capitalize())
        
        # Calculate aggregate statistics
        all_returns = []
        for scenario in regime_scenarios:
            all_returns.extend(scenario)
        
        mean_returns.append(np.mean(all_returns) * 252 * 100)  # Annualized percentage
        volatilities.append(np.std(all_returns) * np.sqrt(252) * 100)  # Annualized percentage
    
    x = np.arange(len(regimes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mean_returns, width, label='Mean Return (%)', alpha=0.7, color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, volatilities, width, label='Volatility (%)', alpha=0.7, color='lightcoral', edgecolor='black')
    
    ax.set_xlabel('Market Type')
    ax.set_ylabel('Annualized %')
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)


def plot_performance_heatmap(results: Dict[str, Dict], ax):
    """Plot performance heatmap."""
    ax.set_title('Performance Metrics\nHeatmap', fontsize=12, fontweight='bold')
    
    # Prepare data for heatmap
    regimes = list(results.keys())
    metrics = ['Mean Return', 'Std Return', 'Win Rate', 'Min Return', 'Max Return']
    
    data = []
    for regime in regimes:
        stats = results[regime]
        row = [
            stats['mean_return'] * 100,
            stats['std_return'] * 100,
            stats['win_rate'] * 100,
            (stats['min_final_value'] / 10000 - 1) * 100,
            (stats['max_final_value'] / 10000 - 1) * 100
        ]
        data.append(row)
    
    data = np.array(data)
    
    # Create heatmap
    im = ax.imshow(data.T, cmap='RdYlGn', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([r.capitalize() for r in regimes])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(regimes)):
            text = ax.text(j, i, f'{data[j, i]:.1f}%', ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.6)


def plot_summary_statistics(results: Dict[str, Dict], ax, regime_colors: Dict):
    """Plot summary statistics comparison."""
    ax.set_title('Strategy Performance\nSummary', fontsize=12, fontweight='bold')
    
    # Create summary text
    summary_text = "MONTE CARLO SIMULATION RESULTS\n"
    summary_text += "="*40 + "\n\n"
    
    total_scenarios = sum(stats['num_scenarios'] for stats in results.values())
    summary_text += f"Total Scenarios Tested: {total_scenarios}\n\n"
    
    for regime_name, stats in results.items():
        summary_text += f"{regime_name.upper()} MARKET:\n"
        summary_text += f"  Scenarios: {stats['num_scenarios']}\n"
        summary_text += f"  Avg Return: {stats['mean_return']:.1%}\n"
        summary_text += f"  Return Range: {(stats['min_final_value']/10000-1):.1%} to {(stats['max_final_value']/10000-1):.1%}\n"
        summary_text += f"  Win Rate: {stats['win_rate']:.1%}\n\n"
    
    # Find best and worst performing markets
    best_regime = max(results.keys(), key=lambda x: results[x]['mean_return'])
    worst_regime = min(results.keys(), key=lambda x: results[x]['mean_return'])
    
    summary_text += f"BEST MARKET TYPE: {best_regime.upper()}\n"
    summary_text += f"WORST MARKET TYPE: {worst_regime.upper()}\n\n"
    
    summary_text += "KEY INSIGHTS:\n"
    summary_text += "• Each scenario represents a different\n  possible market path\n"
    summary_text += "• Wide range of outcomes shows\n  importance of risk management\n"
    summary_text += "• Strategy performance varies\n  significantly by market type"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def demo_market_scenario_simulation():
    """Demonstrate market scenario simulation."""
    print("🎯 MARKET SCENARIO SIMULATION DEMO")
    print("="*50)
    print("Generate different possible market paths to test strategy performance")
    print()
    
    # Create sample historical data
    np.random.seed(42)
    historical_prices = [100]
    for _ in range(252):  # One year of data
        ret = np.random.normal(0.0008, 0.02)  # Daily return
        historical_prices.append(historical_prices[-1] * (1 + ret))
    
    historical_data = pd.DataFrame({
        'Close': historical_prices[1:],
        'Date': pd.date_range('2023-01-01', periods=len(historical_prices)-1)
    })
    
    print(f"📊 Historical data summary:")
    print(f"   Period: {len(historical_data)} days")
    print(f"   Price range: ${historical_data['Close'].min():.2f} to ${historical_data['Close'].max():.2f}")
    
    # Generate market scenarios
    scenarios, base_stats = generate_market_scenarios(
        historical_data,
        num_scenarios=100,  # 100 scenarios total
        scenario_length=126  # Half year scenarios
    )
    
    # Test strategy across scenarios
    results = test_strategy_across_scenarios(
        simple_moving_average_strategy,
        scenarios,
        initial_capital=10000
    )
    
    # Summary
    print(f"\n📈 STRATEGY PERFORMANCE SUMMARY")
    print("="*50)
    for regime, stats in results.items():
        print(f"{regime.capitalize()} Market:")
        print(f"  Average Return: {stats['mean_return']:>8.1%}")
        print(f"  Return Range:   {stats['min_final_value']/10000-1:>8.1%} to {stats['max_final_value']/10000-1:>7.1%}")
        print(f"  Win Rate:       {stats['win_rate']:>8.1%}")
        print()
    
    print("✅ This shows how your strategy would perform across different market conditions!")
    print("   Each scenario represents a different possible market path that could occur.")
    
    # Create comprehensive visualizations
    plot_market_scenarios_analysis(scenarios, results, base_stats)
    
    return scenarios, results


if __name__ == "__main__":
    demo_market_scenario_simulation()
