"""
Advanced ML Algorithm with Monte Carlo Simulation Demo

This demonstrates how the new Advanced ML trading algorithm works
with our Monte Carlo simulation system for comprehensive strategy testing.
"""

import sys
sys.path.append('algorithms')

import pandas as pd
import numpy as np
from machine_learning.advanced_ml_strategy import AdvancedMLStrategy
from data_fetcher import fetch_stock_data
from monte_carlo_trade_simulation import random_trade_order_simulation
from market_scenario_simulation import generate_market_scenarios, test_strategy_across_scenarios

def demo_ml_algorithm_monte_carlo():
    """Demonstrate ML algorithm with Monte Carlo analysis."""
    print("🚀 ADVANCED ML ALGORITHM + MONTE CARLO DEMO")
    print("="*70)
    print("Combining sophisticated ML predictions with comprehensive risk analysis")
    print()
    
    # 1. Setup and Data
    print("📊 STEP 1: Data Preparation")
    print("-" * 30)
    data = fetch_stock_data('SPY', period='1y', interval='1d')
    print(f"✅ Loaded {len(data)} data points for SPY")
    print(f"📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # 2. Initialize ML Algorithm
    print(f"\n🧠 STEP 2: ML Algorithm Setup")
    print("-" * 30)
    ml_algo = AdvancedMLStrategy()
    
    print(f"Algorithm: {ml_algo.name}")
    print(f"Type: {ml_algo.get_algorithm_type()}")
    
    params = ml_algo.get_parameters()
    print(f"Key Parameters:")
    print(f"  • Target Volatility: {params['target_vol']}")
    print(f"  • Maximum Leverage: {params['max_leverage']}")
    print(f"  • ML Cross-Validation Splits: {params['n_splits']}")
    print(f"  • Hyperparameter Search Iterations: {params['n_iter']}")
    print(f"  • Prediction Threshold: {params['prediction_threshold']}")
    
    # 3. Generate ML Trading Signals
    print(f"\n⚡ STEP 3: ML Signal Generation")
    print("-" * 30)
    print("Running comprehensive ML pipeline...")
    print("• Feature engineering (technical indicators, regimes, volatility)")
    print("• Market regime detection using HMM/GMM")
    print("• Walk-forward ML training with hyperparameter optimization")
    print("• Kelly criterion position sizing with volatility targeting")
    
    trades = ml_algo.generate_signals(data)
    
    if len(trades) == 0:
        print("❌ No trades generated. Algorithm may need parameter tuning.")
        return
    
    print(f"\n📈 ML ALGORITHM RESULTS:")
    print(f"  • Total trades executed: {len(trades)}")
    
    # Calculate basic statistics
    returns = [t['return'] for t in trades if 'return' in t]
    if returns:
        total_return = (1 + pd.Series(returns)).prod() - 1
        avg_return = np.mean(returns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        best_trade = max(returns)
        worst_trade = min(returns)
        
        print(f"  • Total return: {total_return:.2%}")
        print(f"  • Average trade return: {avg_return:.2%}")
        print(f"  • Win rate: {win_rate:.1%}")
        print(f"  • Best trade: {best_trade:.2%}")
        print(f"  • Worst trade: {worst_trade:.2%}")
        
        # Calculate additional metrics
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            sharpe = (avg_return * 252) / (volatility + 1e-9)
            print(f"  • Annualized volatility: {volatility:.2%}")
            print(f"  • Sharpe ratio: {sharpe:.2f}")
    
    # 4. Monte Carlo Analysis
    print(f"\n🎲 STEP 4: Monte Carlo Risk Analysis")
    print("-" * 30)
    print("Testing strategy across different possible market scenarios...")
    
    if len(returns) > 0:
        # Traditional Monte Carlo with synthetic returns
        print("\n🔄 Running synthetic return Monte Carlo simulation...")
        mc_results = random_trade_order_simulation(
            np.array(returns),
            num_simulations=500,
            initial_capital=10000,
            position_size_mode='compound',
            simulation_method='synthetic_returns'
        )
        
        final_values = mc_results.iloc[-1].values
        mc_mean = np.mean(final_values)
        mc_std = np.std(final_values)
        mc_min = np.min(final_values)
        mc_max = np.max(final_values)
        
        print(f"📊 Monte Carlo Results (500 simulations):")
        print(f"  • Final portfolio range: ${mc_min:,.0f} to ${mc_max:,.0f}")
        print(f"  • Mean final value: ${mc_mean:,.0f}")
        print(f"  • Standard deviation: ${mc_std:,.0f}")
        print(f"  • Range span: ${mc_max - mc_min:,.0f}")
        
        # Risk metrics
        var_95 = np.percentile(final_values, 5)
        var_99 = np.percentile(final_values, 1)
        prob_loss = np.sum(final_values < 10000) / len(final_values)
        
        print(f"  • Value at Risk (95%): ${var_95:,.0f}")
        print(f"  • Value at Risk (99%): ${var_99:,.0f}")
        print(f"  • Probability of loss: {prob_loss:.1%}")
    
    # 5. Market Scenario Testing
    print(f"\n🌍 STEP 5: Market Scenario Analysis")
    print("-" * 30)
    print("Testing strategy across different market regimes...")
    
    try:
        # Generate market scenarios
        scenarios, base_stats = generate_market_scenarios(
            data,
            num_scenarios=50,  # Smaller for demo
            scenario_length=126  # 6 months
        )
        
        # Test strategy across scenarios
        def ml_strategy_wrapper(price_data):
            """Wrapper to use ML strategy in scenario testing."""
            try:
                trades = ml_algo.generate_signals(price_data)
                return trades
            except:
                return []
        
        scenario_results = test_strategy_across_scenarios(
            ml_strategy_wrapper,
            scenarios,
            initial_capital=10000
        )
        
        if scenario_results:
            print(f"\n📈 MARKET SCENARIO RESULTS:")
            for regime, stats in scenario_results.items():
                print(f"  {regime.upper()} Market:")
                print(f"    • Scenarios tested: {stats['num_scenarios']}")
                print(f"    • Average return: {stats['mean_return']:.1%}")
                print(f"    • Return range: {stats['min_final_value']/10000-1:.1%} to {stats['max_final_value']/10000-1:.1%}")
                print(f"    • Win rate: {stats['win_rate']:.1%}")
        
    except Exception as e:
        print(f"⚠️ Scenario analysis skipped: {e}")
    
    # 6. Summary and Insights
    print(f"\n🎯 SUMMARY & INSIGHTS")
    print("="*50)
    print("✅ Advanced ML Algorithm Features Demonstrated:")
    print("  • Multi-factor feature engineering")
    print("  • Regime-aware market analysis")
    print("  • Walk-forward ML validation")
    print("  • Kelly criterion position sizing")
    print("  • Volatility targeting")
    print()
    print("✅ Monte Carlo Risk Analysis:")
    print("  • Synthetic return generation")
    print("  • Comprehensive risk metrics")
    print("  • Portfolio value distributions")
    print("  • Multiple market scenario testing")
    print()
    print("🚀 The Advanced ML Algorithm is now ready for:")
    print("  • Integration with backtest_algorithms.py")
    print("  • Full Monte Carlo simulation suite")
    print("  • Graphical analysis and visualization")
    print("  • Risk management and portfolio optimization")
    
    return trades, mc_results if 'mc_results' in locals() else None

if __name__ == "__main__":
    demo_ml_algorithm_monte_carlo()
