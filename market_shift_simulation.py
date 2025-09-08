"""
Market Shift Simulation

This script explains why current simulations have identical outcomes
and demonstrates how to create simulations that show different market scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
import warnings

def explain_identical_outcomes():
    """
    Explain why current Monte Carlo simulations produce identical outcomes.
    """
    print("🎯 WHY ARE ALL OUTCOMES THE SAME?")
    print("="*50)
    print("Current Issue: Monte Carlo simulations produce identical results")
    print("Mathematical Reason: Multiplication is commutative")
    print()
    
    # Simple demonstration
    returns = [0.10, -0.05, 0.08]  # 10%, -5%, 8% returns
    initial_value = 1000
    
    print("📊 MATHEMATICAL DEMONSTRATION:")
    print("-"*40)
    print(f"Starting value: ${initial_value}")
    print(f"Returns to apply: {[f'{r:+.1%}' for r in returns]}")
    print()
    
    # Show different orders give same result
    from itertools import permutations
    all_orders = list(permutations(returns))
    
    print("All possible orders and their final values:")
    for i, order in enumerate(all_orders):
        value = initial_value
        calculation = f"${initial_value}"
        for ret in order:
            value *= (1 + ret)
            calculation += f" × {1+ret:.3f}"
        calculation += f" = ${value:.2f}"
        print(f"  Order {i+1}: {calculation}")
    
    print(f"\n💡 KEY INSIGHT:")
    print(f"   All orders produce the same result: ${value:.2f}")
    print(f"   This is because: A × B × C = C × B × A (commutative property)")
    print(f"   Order doesn't matter for multiplicative returns!")
    
    return value


def demonstrate_market_shift_simulation():
    """
    Demonstrate how to simulate different market scenarios.
    """
    print(f"\n🌊 SIMULATING MARKET SHIFTS")
    print("="*40)
    print("Solution: Instead of reordering the same returns,")
    print("simulate different market conditions!")
    print()
    
    # Base scenario parameters
    base_mean_return = 0.08  # 8% annual return
    base_volatility = 0.15   # 15% volatility
    num_periods = 12         # 12 months
    num_simulations = 1000
    
    print(f"📈 MARKET SCENARIO SIMULATION:")
    print("-"*35)
    print(f"Base Parameters:")
    print(f"  Mean Annual Return: {base_mean_return:.1%}")
    print(f"  Annual Volatility:  {base_volatility:.1%}")
    print(f"  Simulation Periods: {num_periods}")
    print(f"  Number of Simulations: {num_simulations:,}")
    print()
    
    # Create different market scenarios
    scenarios = {
        'Bull Market': {'mean': 0.12, 'vol': 0.12, 'color': 'green'},
        'Normal Market': {'mean': 0.08, 'vol': 0.15, 'color': 'blue'},
        'Bear Market': {'mean': 0.02, 'vol': 0.20, 'color': 'red'},
        'Volatile Market': {'mean': 0.08, 'vol': 0.25, 'color': 'orange'},
        'Low Vol Market': {'mean': 0.06, 'vol': 0.08, 'color': 'purple'}
    }
    
    results = {}
    
    for scenario_name, params in scenarios.items():
        print(f"🎲 Simulating {scenario_name}...")
        
        # Generate random returns for this scenario
        monthly_mean = params['mean'] / 12
        monthly_vol = params['vol'] / np.sqrt(12)
        
        # Create Monte Carlo simulations for this market scenario
        simulation_results = []
        
        for sim in range(num_simulations):
            # Generate random monthly returns
            monthly_returns = np.random.normal(monthly_mean, monthly_vol, num_periods)
            
            # Calculate cumulative portfolio value
            portfolio_value = 10000  # Start with $10,000
            values = [portfolio_value]
            
            for monthly_return in monthly_returns:
                portfolio_value *= (1 + monthly_return)
                values.append(portfolio_value)
            
            simulation_results.append(values)
        
        # Convert to DataFrame
        simulation_df = pd.DataFrame(simulation_results).T
        final_values = simulation_df.iloc[-1]
        
        results[scenario_name] = {
            'simulation_df': simulation_df,
            'final_values': final_values,
            'mean_final': final_values.mean(),
            'std_final': final_values.std(),
            'min_final': final_values.min(),
            'max_final': final_values.max(),
            'params': params
        }
        
        print(f"   Final Value Range: ${final_values.min():,.0f} to ${final_values.max():,.0f}")
        print(f"   Mean Final Value: ${final_values.mean():,.0f}")
        print(f"   Standard Deviation: ${final_values.std():,.0f}")
    
    return results


def plot_market_scenarios(results: Dict):
    """
    Plot the different market scenario results.
    """
    print(f"\n📊 CREATING MARKET SCENARIO VISUALIZATION...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Portfolio Evolution Over Time
    for scenario_name, data in results.items():
        simulation_df = data['simulation_df']
        color = data['params']['color']
        
        # Plot mean path and confidence bands
        mean_path = simulation_df.mean(axis=1)
        std_path = simulation_df.std(axis=1)
        periods = range(len(mean_path))
        
        ax1.plot(periods, mean_path, color=color, linewidth=2, label=scenario_name)
        ax1.fill_between(periods, 
                        mean_path - std_path, 
                        mean_path + std_path, 
                        color=color, alpha=0.2)
    
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Evolution by Market Scenario')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Final Value Distributions
    for scenario_name, data in results.items():
        final_values = data['final_values']
        color = data['params']['color']
        ax2.hist(final_values, bins=30, alpha=0.6, label=scenario_name, 
                color=color, density=True)
    
    ax2.set_xlabel('Final Portfolio Value ($)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Final Portfolio Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Risk-Return Scatter
    for scenario_name, data in results.items():
        mean_return = (data['mean_final'] / 10000 - 1) * 100  # Convert to percentage
        volatility = (data['std_final'] / 10000) * 100
        color = data['params']['color']
        
        ax3.scatter(volatility, mean_return, s=100, color=color, 
                   label=scenario_name, alpha=0.7)
        ax3.annotate(scenario_name, (volatility, mean_return), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('Volatility (%)')
    ax3.set_ylabel('Total Return (%)')
    ax3.set_title('Risk-Return Profile by Scenario')
    ax3.grid(True, alpha=0.3)
    
    # 4. Probability of Loss
    for i, (scenario_name, data) in enumerate(results.items()):
        final_values = data['final_values']
        prob_loss = (final_values < 10000).mean() * 100
        color = data['params']['color']
        
        ax4.bar(i, prob_loss, color=color, alpha=0.7, label=scenario_name)
    
    ax4.set_ylabel('Probability of Loss (%)')
    ax4.set_title('Probability of Portfolio Loss by Scenario')
    ax4.set_xticks(range(len(results)))
    ax4.set_xticklabels(results.keys(), rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_portfolio_weight_simulation():
    """
    Create simulation with varying portfolio weights (different approach).
    """
    print(f"\n🎯 ALTERNATIVE: PORTFOLIO WEIGHT SIMULATION")
    print("="*50)
    print("Another way to get different outcomes: Vary portfolio weights")
    print()
    
    # Fixed assets with known characteristics
    assets = ['Stock', 'Bond']
    stock_return = 0.10  # 10% annual
    bond_return = 0.04   # 4% annual
    stock_vol = 0.20     # 20% volatility
    bond_vol = 0.05      # 5% volatility
    correlation = 0.3    # 30% correlation
    
    num_simulations = 1000
    results = []
    
    print(f"📊 SIMULATION PARAMETERS:")
    print(f"   Stock: {stock_return:.1%} return, {stock_vol:.1%} volatility")
    print(f"   Bond:  {bond_return:.1%} return, {bond_vol:.1%} volatility")
    print(f"   Correlation: {correlation:.1%}")
    print(f"   Simulations: {num_simulations:,}")
    print()
    
    for i in range(num_simulations):
        # Random portfolio weights (statistical sampling within constraints)
        stock_weight = np.random.beta(2, 2)  # Beta distribution for realistic weights
        bond_weight = 1 - stock_weight
        
        # Generate correlated returns
        random_vars = np.random.multivariate_normal(
            [0, 0], 
            [[1, correlation], [correlation, 1]]
        )
        
        # Convert to asset returns
        stock_actual_return = stock_return + stock_vol * random_vars[0]
        bond_actual_return = bond_return + bond_vol * random_vars[1]
        
        # Calculate portfolio return
        portfolio_return = (stock_weight * stock_actual_return + 
                           bond_weight * bond_actual_return)
        
        # Calculate portfolio value
        portfolio_value = 10000 * (1 + portfolio_return)
        
        results.append({
            'stock_weight': stock_weight,
            'bond_weight': bond_weight,
            'stock_return': stock_actual_return,
            'bond_return': bond_actual_return,
            'portfolio_return': portfolio_return,
            'portfolio_value': portfolio_value
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"📈 SIMULATION RESULTS:")
    print("-"*30)
    print(f"Portfolio Value Range: ${results_df['portfolio_value'].min():,.0f} to ${results_df['portfolio_value'].max():,.0f}")
    print(f"Mean Portfolio Value: ${results_df['portfolio_value'].mean():,.0f}")
    print(f"Standard Deviation: ${results_df['portfolio_value'].std():,.0f}")
    print(f"Different outcomes: {len(results_df['portfolio_value'].unique()):,}")
    print()
    
    # Quick visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(results_df['stock_weight'], results_df['portfolio_return'], 
                alpha=0.6, s=20)
    plt.xlabel('Stock Weight')
    plt.ylabel('Portfolio Return')
    plt.title('Portfolio Return vs Stock Allocation')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.hist(results_df['portfolio_value'], bins=50, alpha=0.7, color='blue')
    plt.xlabel('Portfolio Value ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Portfolio Values')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.scatter(results_df['portfolio_return'], results_df['portfolio_value'], 
                alpha=0.6, s=20, c=results_df['stock_weight'], cmap='viridis')
    plt.colorbar(label='Stock Weight')
    plt.xlabel('Portfolio Return')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Return vs Value (Color = Stock Weight)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.boxplot([results_df[results_df['stock_weight'] < 0.3]['portfolio_value'],
                 results_df[(results_df['stock_weight'] >= 0.3) & (results_df['stock_weight'] < 0.7)]['portfolio_value'],
                 results_df[results_df['stock_weight'] >= 0.7]['portfolio_value']],
                labels=['Low Stock', 'Medium Stock', 'High Stock'])
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value by Stock Allocation')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df


def main():
    """Main demonstration function."""
    print("🚀 MARKET SHIFT SIMULATION EXPLANATION")
    print("="*50)
    print("Your Question: 'How are all outcomes the same for simulations?")
    print("              We want to simulate possible shifts of the market'")
    print()
    
    # Step 1: Explain why outcomes are identical
    explain_identical_outcomes()
    
    # Step 2: Demonstrate proper market shift simulation
    market_results = demonstrate_market_shift_simulation()
    
    # Step 3: Visualize the results
    plot_market_scenarios(market_results)
    
    # Step 4: Alternative approach with portfolio weights
    portfolio_results = create_portfolio_weight_simulation()
    
    print(f"\n💡 SOLUTION SUMMARY:")
    print("="*40)
    print(f"❌ Current Approach (Order Shuffling):")
    print(f"   • Shuffles the same returns in different orders")
    print(f"   • All outcomes identical due to commutative property")
    print(f"   • Doesn't simulate market shifts")
    print()
    print(f"✅ Correct Approach (Market Scenario Simulation):")
    print(f"   • Generate different return scenarios")
    print(f"   • Vary market conditions (bull/bear/volatile)")
    print(f"   • Use random return generation")
    print(f"   • Vary portfolio weights")
    print()
    print(f"🎯 RESULTS:")
    print(f"   • Market scenarios show {len(market_results)} different outcomes")
    print(f"   • Portfolio weight simulation shows {len(portfolio_results):,} different outcomes")
    print(f"   • Each simulation represents a different market path")
    print(f"   • Realistic range of portfolio values achieved")


if __name__ == "__main__":
    main()
