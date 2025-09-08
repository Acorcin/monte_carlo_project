"""
Algorithm Backtesting System

Main interface for backtesting trading algorithms with Monte Carlo simulation integration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import sys
import os

# Add algorithms directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))

from algorithms.algorithm_manager import algorithm_manager
from data_fetcher import fetch_stock_data, calculate_returns_from_ohlcv
from monte_carlo_trade_simulation import random_trade_order_simulation

def get_user_algorithm_selection() -> List[Dict[str, Any]]:
    """Get user's algorithm selection and parameters."""
    print("🤖 ALGORITHM SELECTION")
    print("=" * 30)
    
    # Show available algorithms
    algorithm_manager.print_available_algorithms()
    
    available_algos = algorithm_manager.get_available_algorithms()
    algo_names = list(available_algos.keys())
    
    selected_configs = []
    
    while True:
        print(f"\n📋 ALGORITHM SELECTION MENU")
        print("1. Select individual algorithm")
        print("2. Select all algorithms from a category")
        print("3. Select all algorithms")
        print("4. Finish selection")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            # Individual algorithm selection
            print(f"\nAvailable algorithms:")
            for i, name in enumerate(algo_names, 1):
                print(f"  {i}. {name}")
            
            try:
                algo_idx = int(input(f"\nSelect algorithm (1-{len(algo_names)}): ")) - 1
                if 0 <= algo_idx < len(algo_names):
                    selected_algo = algo_names[algo_idx]
                    config = configure_algorithm(selected_algo)
                    if config:
                        selected_configs.append(config)
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == '2':
            # Category selection
            categories = algorithm_manager.get_algorithm_categories()
            cat_names = list(categories.keys())
            
            print(f"\nAvailable categories:")
            for i, cat in enumerate(cat_names, 1):
                print(f"  {i}. {cat} ({len(categories[cat])} algorithms)")
            
            try:
                cat_idx = int(input(f"\nSelect category (1-{len(cat_names)}): ")) - 1
                if 0 <= cat_idx < len(cat_names):
                    selected_category = cat_names[cat_idx]
                    for algo_name in categories[selected_category]:
                        config = {'name': algo_name, 'parameters': {}}
                        selected_configs.append(config)
                    print(f"✅ Added all algorithms from {selected_category}")
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == '3':
            # Select all algorithms
            for algo_name in algo_names:
                config = {'name': algo_name, 'parameters': {}}
                selected_configs.append(config)
            print(f"✅ Added all {len(algo_names)} algorithms")
        
        elif choice == '4':
            break
        else:
            print("❌ Invalid option")
    
    return selected_configs

def configure_algorithm(algorithm_name: str) -> Optional[Dict[str, Any]]:
    """Configure parameters for a specific algorithm."""
    info = algorithm_manager.get_algorithm_info(algorithm_name)
    if not info:
        return None
    
    print(f"\n⚙️  CONFIGURING: {info['name']}")
    print(f"   {info['description']}")
    
    parameters = {}
    param_info = info.get('parameters', {})
    
    if not param_info:
        print("   No configurable parameters")
        return {'name': algorithm_name, 'parameters': {}}
    
    print(f"\n📊 Parameters:")
    for param_name, param_details in param_info.items():
        default_value = param_details['default']
        param_type = param_details['type']
        description = param_details['description']
        
        user_input = input(f"   {param_name} (default: {default_value}): ").strip()
        
        if user_input:
            try:
                if param_type == 'int':
                    parameters[param_name] = int(user_input)
                elif param_type == 'float':
                    parameters[param_name] = float(user_input)
                else:
                    parameters[param_name] = user_input
            except ValueError:
                print(f"   ❌ Invalid input, using default: {default_value}")
                parameters[param_name] = default_value
        else:
            parameters[param_name] = default_value
    
    return {'name': algorithm_name, 'parameters': parameters}

def run_backtests(algorithm_configs: List[Dict[str, Any]], data: pd.DataFrame, 
                 initial_capital: float = 10000) -> Dict[str, Any]:
    """Run backtests for all selected algorithms."""
    print(f"\n🔄 RUNNING BACKTESTS")
    print("=" * 25)
    
    results = algorithm_manager.backtest_multiple_algorithms(
        algorithm_configs=algorithm_configs,
        data=data,
        initial_capital=initial_capital
    )
    
    return results

def display_drawdown_analysis(metrics: Dict[str, Any], algorithm_name: str):
    """Display detailed drawdown analysis for an algorithm."""
    print(f"\n📉 DRAWDOWN ANALYSIS - {algorithm_name}")
    print("=" * 80)
    
    # Main drawdown metrics using your formula
    print(f"🎯 FORMULA: [(Highest Peak - Lowest Trough) / Highest Peak] × 100")
    print("-" * 80)
    print(f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
    print(f"Peak Value: ${metrics.get('drawdown_peak_value', 0):,.2f}")
    print(f"Trough Value: ${metrics.get('drawdown_trough_value', 0):,.2f}")
    print(f"Dollar Loss: ${metrics.get('drawdown_peak_value', 0) - metrics.get('drawdown_trough_value', 0):,.2f}")
    
    # Duration and recovery metrics
    print("\n📊 DRAWDOWN BEHAVIOR:")
    print(f"Longest Drawdown Duration: {metrics.get('drawdown_duration_days', 0)} periods")
    print(f"Average Drawdown: {metrics.get('avg_drawdown', 0):.2f}%")
    print(f"Number of Drawdown Periods: {metrics.get('drawdown_periods', 0)}")
    print(f"Time Underwater: {metrics.get('time_underwater_pct', 0):.1f}% of total time")
    
    # Risk assessment
    if metrics.get('max_drawdown', 0) > 0:
        risk_level = "Low" if metrics['max_drawdown'] < 5 else "Medium" if metrics['max_drawdown'] < 15 else "High"
        print(f"\n⚠️  RISK ASSESSMENT: {risk_level} Risk")
        
        if metrics['max_drawdown'] > 20:
            print("   ⚠️  WARNING: Drawdown exceeds 20% - Consider risk management")
        elif metrics['max_drawdown'] > 10:
            print("   ⚠️  CAUTION: Moderate drawdown - Monitor closely")
        else:
            print("   ✅ ACCEPTABLE: Drawdown within reasonable limits")

def display_individual_trades(trades: List[Dict], algorithm_name: str):
    """Display individual trades for an algorithm."""
    if not trades:
        print(f"   📊 No trades executed for {algorithm_name}")
        return
    
    print(f"\n📋 INDIVIDUAL TRADES - {algorithm_name}")
    print("=" * 80)
    print(f"{'#':<3} {'Entry Date':<12} {'Exit Date':<12} {'Direction':<6} {'Entry $':<8} {'Exit $':<8} {'Return %':<8} {'Duration':<10}")
    print("-" * 80)
    
    for i, trade in enumerate(trades, 1):
        entry_date = trade['entry_time'].strftime('%Y-%m-%d') if hasattr(trade['entry_time'], 'strftime') else str(trade['entry_time'])[:10]
        exit_date = trade['exit_time'].strftime('%Y-%m-%d') if hasattr(trade['exit_time'], 'strftime') else str(trade['exit_time'])[:10]
        duration_days = trade['duration'].days if hasattr(trade['duration'], 'days') else str(trade['duration'])[:8]
        
        print(f"{i:<3} {entry_date:<12} {exit_date:<12} {trade['direction']:<6} "
              f"{trade['entry_price']:<8.2f} {trade['exit_price']:<8.2f} "
              f"{trade['return']*100:<8.2f} {duration_days:<10}")
    
    # Trade summary statistics
    returns = [trade['return'] for trade in trades]
    winning_trades = [r for r in returns if r > 0]
    losing_trades = [r for r in returns if r < 0]
    
    print("-" * 80)
    print(f"📊 TRADE SUMMARY:")
    print(f"   Total Trades: {len(trades)}")
    print(f"   Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
    print(f"   Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
    if winning_trades:
        print(f"   Avg Winning Trade: {np.mean(winning_trades)*100:.2f}%")
    if losing_trades:
        print(f"   Avg Losing Trade: {np.mean(losing_trades)*100:.2f}%")
    print(f"   Best Trade: {max(returns)*100:.2f}%")
    print(f"   Worst Trade: {min(returns)*100:.2f}%")

def display_results(results: Dict[str, Any]):
    """Display backtest results in a formatted table."""
    print(f"\n📊 BACKTEST RESULTS")
    print("=" * 70)

    if not results:
        print("❌ No successful backtests to display")
        return

    # Create results table
    table_data = []
    for algo_name, result in results.items():
        metrics = result['metrics']
        table_data.append({
            'Algorithm': algo_name[:20],  # Truncate for display
            'Total Return': f"{result['total_return']:.2f}%",
            'Trades': metrics['total_trades'],
            'Win Rate': f"{metrics['win_rate']:.1f}%",
            'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
            'Max DD': f"{metrics['max_drawdown']:.1f}%"
        })

    # Print table header
    headers = ['Algorithm', 'Total Return', 'Trades', 'Win Rate', 'Sharpe', 'Max DD']
    header_line = " | ".join(f"{h:>12}" for h in headers)
    print(header_line)
    print("-" * len(header_line))

    # Print table rows
    for row in table_data:
        row_line = " | ".join(f"{str(row[h]):>12}" for h in headers)
        print(row_line)

    # Find best performer
    best_algo = max(results.items(), key=lambda x: x[1]['total_return'])
    print(f"\n🏆 BEST PERFORMER: {best_algo[0]} ({best_algo[1]['total_return']:.2f}% return)")
    
    # Display individual trades for each algorithm
    print(f"\n" + "="*80)
    print("📋 DETAILED TRADE BREAKDOWN")
    print("="*80)
    
    for algo_name, result in results.items():
        display_individual_trades(result.get('trades', []), algo_name)
        display_drawdown_analysis(result.get('metrics', {}), algo_name)

def plot_drawdown_analysis(results: Dict[str, Any]):
    """Plot comprehensive drawdown analysis for all algorithms."""
    print(f"\n📉 Generating drawdown analysis charts...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Equity curves with peaks
    ax1 = axes[0]
    for algo_name, result in results.items():
        if 'drawdown_curve' in result and not result['drawdown_curve'].empty:
            df = result['drawdown_curve']
            
            # Plot equity curve
            ax1.plot(df['timestamp'], df['equity'], label=f'{algo_name} - Equity', linewidth=2, alpha=0.8)
            
            # Plot peak curve
            ax1.plot(df['timestamp'], df['peak'], label=f'{algo_name} - Peak', 
                    linestyle='--', alpha=0.6, linewidth=1)
    
    ax1.set_title("📈 Equity Curves with Running Peaks", fontsize=14)
    ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Drawdown curves
    ax2 = axes[1]
    for algo_name, result in results.items():
        if 'drawdown_curve' in result and not result['drawdown_curve'].empty:
            df = result['drawdown_curve']
            
            # Plot drawdown (negative values for visual effect)
            ax2.fill_between(df['timestamp'], 0, -df['drawdown'], 
                           label=f'{algo_name}', alpha=0.6)
            ax2.plot(df['timestamp'], -df['drawdown'], linewidth=1)
    
    ax2.set_title("📉 Drawdown Analysis [(Peak - Current) / Peak] × 100", fontsize=14)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add statistics text
    stats_text = "📊 Drawdown Statistics:\n"
    for algo_name, result in results.items():
        metrics = result.get('metrics', {})
        max_dd = metrics.get('max_drawdown', 0)
        avg_dd = metrics.get('avg_drawdown', 0)
        time_underwater = metrics.get('time_underwater_pct', 0)
        stats_text += f"{algo_name}: Max {max_dd:.1f}%, Avg {avg_dd:.1f}%, Underwater {time_underwater:.1f}%\n"
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def plot_equity_curves(results: Dict[str, Any]):
    """Plot equity curves for all algorithms."""
    print(f"\n📈 Generating equity curve comparison...")
    
    plt.figure(figsize=(12, 8))
    
    for algo_name, result in results.items():
        equity_curve = result['equity_curve']
        plt.plot(equity_curve['timestamp'], equity_curve['equity'], 
                label=algo_name, linewidth=2, alpha=0.8)
    
    plt.title("Algorithm Performance Comparison", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def monte_carlo_integration(results: Dict[str, Any], num_simulations: int = 1000, 
                          simulation_method: str = 'synthetic_returns'):
    """Integrate with Monte Carlo simulation using algorithm returns."""
    print(f"\n🎲 MONTE CARLO ANALYSIS")
    print("=" * 30)
    print(f"Method: {simulation_method} (synthetic returns with average within 2σ of original)")
    
    for algo_name, result in results.items():
        if not result['returns']:
            continue
            
        print(f"\n📊 {algo_name}:")
        
        # Run Monte Carlo on algorithm returns
        returns_array = np.array(result['returns'])
        
        if len(returns_array) < 2:
            print("   ⚠️  Not enough trades for Monte Carlo analysis")
            continue
        
        mc_results = random_trade_order_simulation(
            returns_array,
            num_simulations=num_simulations,
            initial_capital=result['initial_capital'],
            simulation_method=simulation_method
        )
        
        final_values = mc_results.iloc[-1].values
        
        # Check if outcomes are actually different
        min_value = np.min(final_values)
        max_value = np.max(final_values)
        mean_value = np.mean(final_values)
        std_value = np.std(final_values)
        range_span = max_value - min_value
        
        print(f"   Monte Carlo simulations: {num_simulations}")
        
        if range_span > 0.01:  # If range > 1 cent, outcomes are different
            print(f"   📊 DIFFERENT OUTCOMES ACHIEVED! 🎉")
            print(f"   💰 Portfolio range: ${min_value:,.2f} to ${max_value:,.2f}")
            print(f"   📈 Range span: ${range_span:,.2f}")
            print(f"   📊 Mean: ${mean_value:,.2f} ± ${std_value:,.2f}")
            
            # Risk metrics
            var_95 = np.percentile(final_values, 5)
            var_99 = np.percentile(final_values, 1)
            prob_loss = np.sum(final_values < result['initial_capital']) / len(final_values)
            
            print(f"   🎯 Value at Risk (95%): ${var_95:,.2f}")
            print(f"   ⚠️  Probability of loss: {prob_loss:.1%}")
            
        else:
            print(f"   ⚠️  Identical outcomes: ${final_values[0]:,.2f}")
            print(f"   Note: This suggests the method isn't creating variation")
        
        print(f"   🔬 Method used: {simulation_method}")

def main():
    """Main backtesting interface."""
    print("🚀 TRADING ALGORITHM BACKTESTING SYSTEM")
    print("=" * 50)
    
    # Get data
    print(f"\n📡 DATA SELECTION")
    print("-" * 20)
    
    data_choice = input("Data source (1=SPY ETF, 2=Custom ticker): ").strip()
    
    if data_choice == '2':
        ticker = input("Enter ticker symbol: ").strip().upper()
        period = input("Period (1mo, 3mo, 6mo, 1y, default=3mo): ").strip() or "3mo"
        interval = input("Interval (1d, 1h, default=1d): ").strip() or "1d"
    else:
        ticker = "SPY"
        period = "6mo"
        interval = "1d"
    
    try:
        print(f"📊 Fetching {ticker} data ({period}, {interval})...")
        data = fetch_stock_data(ticker, period=period, interval=interval)
        print(f"✅ Loaded {len(data)} data points")
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
        return
    
    # Get initial capital
    try:
        capital_input = input(f"\n💰 Initial capital (default: $10,000): ").strip()
        initial_capital = float(capital_input.replace('$', '').replace(',', '')) if capital_input else 10000
    except ValueError:
        initial_capital = 10000
    
    # Algorithm selection
    algorithm_configs = get_user_algorithm_selection()
    
    if not algorithm_configs:
        print("❌ No algorithms selected")
        return
    
    print(f"\n✅ Selected {len(algorithm_configs)} algorithm(s) for backtesting")
    
    # Run backtests
    results = run_backtests(algorithm_configs, data, initial_capital)
    
    if not results:
        print("❌ No successful backtests")
        return
    
    # Display results
    display_results(results)
    
    # Optional visualizations
    plot_choice = input(f"\n📈 Generate equity curve plot? (y/n): ").strip().lower()
    if plot_choice == 'y':
        try:
            plot_equity_curves(results)
        except Exception as e:
            print(f"❌ Plotting failed: {e}")
    
    # Optional drawdown analysis plot
    drawdown_choice = input(f"\n📉 Generate drawdown analysis plot? (y/n): ").strip().lower()
    if drawdown_choice == 'y':
        try:
            plot_drawdown_analysis(results)
        except Exception as e:
            print(f"❌ Drawdown plotting failed: {e}")
    
    # Optional Monte Carlo analysis
    mc_choice = input(f"\n🎲 Run Monte Carlo analysis on returns? (y/n): ").strip().lower()
    if mc_choice == 'y':
        try:
            num_sims = input("Number of simulations (default: 1000): ").strip()
            num_sims = int(num_sims) if num_sims else 1000
            
            print("Simulation method:")
            print("1. Synthetic Returns (generates new returns within 2σ of original mean)")
            print("2. Statistical 2σ (statistical sampling within 2 standard deviations)")
            print("3. Random (pure random shuffling)")
            method_choice = input("Enter choice (1, 2, or 3, default=1): ").strip()
            if method_choice == '2':
                simulation_method = 'statistical'
            elif method_choice == '3':
                simulation_method = 'random'
            else:
                simulation_method = 'synthetic_returns'
            
            monte_carlo_integration(results, num_sims, simulation_method)
        except Exception as e:
            print(f"❌ Monte Carlo analysis failed: {e}")
    
    # Save results option
    save_choice = input(f"\n💾 Save results to CSV? (y/n): ").strip().lower()
    if save_choice == 'y':
        try:
            # Save summary
            summary_data = []
            for algo_name, result in results.items():
                metrics = result['metrics']
                summary_data.append({
                    'Algorithm': algo_name,
                    'Total_Return_Pct': result['total_return'],
                    'Total_Trades': metrics['total_trades'],
                    'Win_Rate_Pct': metrics['win_rate'],
                    'Sharpe_Ratio': metrics['sharpe_ratio'],
                    'Max_Drawdown_Pct': metrics['max_drawdown'],
                    'Profit_Factor': metrics['profit_factor']
                })
            
            summary_df = pd.DataFrame(summary_data)
            filename = f"backtest_results_{ticker}_{period}.csv"
            summary_df.to_csv(filename, index=False)
            print(f"✅ Results saved to {filename}")
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
    
    print(f"\n🎯 BACKTESTING COMPLETE!")
    print("Thank you for using the algorithm backtesting system!")

if __name__ == "__main__":
    main()
