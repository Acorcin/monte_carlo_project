"""
Synthetic Returns Monte Carlo Demo

Demonstrates the new synthetic return generation approach that creates 
random return points statistically identical to original data, with 
average returns within 2 standard deviations of original mean.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monte_carlo_trade_simulation import random_trade_order_simulation, plot_trade_order_simulations

def compare_simulation_methods():
    """Compare different Monte Carlo simulation methods."""
    print("🎯 SYNTHETIC RETURNS MONTE CARLO COMPARISON")
    print("="*60)
    print("Demonstrates the new approach: Generate random return points")
    print("that are statistically identical to original data")
    print("Average returns constrained within 2σ of original mean")
    print()
    
    # Create sample trading returns
    np.random.seed(42)
    sample_returns = np.array([
        0.05, -0.02, 0.03, -0.01, 0.04, -0.015, 0.025, -0.008, 0.02, -0.03,
        0.015, -0.005, 0.035, -0.012, 0.018, -0.025, 0.022, -0.007, 0.028, -0.015
    ])
    
    print(f"📊 ORIGINAL TRADE RETURNS:")
    print(f"   Number of trades: {len(sample_returns)}")
    print(f"   Mean return: {np.mean(sample_returns):.4f}")
    print(f"   Std deviation: {np.std(sample_returns):.4f}")
    print(f"   Return range: {np.min(sample_returns):.4f} to {np.max(sample_returns):.4f}")
    print(f"   Positive trades: {np.sum(sample_returns > 0)}")
    print(f"   Negative trades: {np.sum(sample_returns < 0)}")
    
    num_simulations = 100
    initial_capital = 10000
    
    methods = [
        ('random', 'Random Shuffling'),
        ('statistical', 'Statistical Sampling (2σ)'),
        ('synthetic_returns', 'Synthetic Returns (2σ constraint)')
    ]
    
    results = {}
    
    for method_code, method_name in methods:
        print(f"\n{'='*20} {method_name.upper()} {'='*20}")
        
        # Run simulation
        equity_curves = random_trade_order_simulation(
            sample_returns,
            num_simulations=num_simulations,
            initial_capital=initial_capital,
            position_size_mode='compound',
            simulation_method=method_code
        )
        
        # Calculate final values
        final_values = equity_curves.iloc[-1].values
        
        # Statistics
        stats = {
            'mean_final': np.mean(final_values),
            'std_final': np.std(final_values),
            'min_final': np.min(final_values),
            'max_final': np.max(final_values),
            'range': np.max(final_values) - np.min(final_values),
            'cv': np.std(final_values) / np.mean(final_values)  # Coefficient of variation
        }
        
        results[method_code] = {
            'equity_curves': equity_curves,
            'final_values': final_values,
            'stats': stats,
            'method_name': method_name
        }
        
        print(f"📈 RESULTS SUMMARY:")
        print(f"   Final value range: ${stats['min_final']:,.0f} to ${stats['max_final']:,.0f}")
        print(f"   Range span: ${stats['range']:,.0f}")
        print(f"   Mean final value: ${stats['mean_final']:,.0f}")
        print(f"   Standard deviation: ${stats['std_final']:,.0f}")
        print(f"   Coefficient of variation: {stats['cv']:.4f}")
    
    # Comparison analysis
    print(f"\n🔍 COMPARISON ANALYSIS")
    print("="*60)
    
    print(f"{'Method':<25} {'Range ($)':<12} {'Std Dev ($)':<12} {'CV':<8} {'Different?'}")
    print("-" * 70)
    
    for method_code, method_name in methods:
        stats = results[method_code]['stats']
        different = "YES" if stats['range'] > 100 else "NO"
        print(f"{method_name:<25} {stats['range']:<12,.0f} {stats['std_final']:<12,.0f} {stats['cv']:<8.4f} {different}")
    
    # Visual comparison
    plot_method_comparison(results)
    
    return results


def plot_method_comparison(results):
    """Create comparison plots for different methods."""
    print(f"\n📊 Creating comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Monte Carlo Simulation Methods Comparison', fontsize=16, fontweight='bold')
    
    methods = list(results.keys())
    colors = ['blue', 'orange', 'green']
    
    # Top row: Equity curves for each method
    for i, method_code in enumerate(methods):
        ax = axes[0, i]
        equity_curves = results[method_code]['equity_curves']
        method_name = results[method_code]['method_name']
        
        # Plot sample of equity curves
        sample_size = min(20, equity_curves.shape[1])
        sample_cols = np.random.choice(equity_curves.columns, sample_size, replace=False)
        
        for col in sample_cols:
            ax.plot(equity_curves[col], alpha=0.3, color=colors[i], linewidth=0.8)
        
        # Plot mean curve
        mean_curve = equity_curves.mean(axis=1)
        ax.plot(mean_curve, color='red', linewidth=2, label='Mean')
        
        ax.set_title(f'{method_name}\nEquity Curves')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Bottom row: Final value distributions
    for i, method_code in enumerate(methods):
        ax = axes[1, i]
        final_values = results[method_code]['final_values']
        method_name = results[method_code]['method_name']
        stats = results[method_code]['stats']
        
        # Histogram
        ax.hist(final_values, bins=20, alpha=0.7, color=colors[i], edgecolor='black')
        ax.axvline(stats['mean_final'], color='red', linestyle='--', linewidth=2, 
                   label=f"Mean: ${stats['mean_final']:,.0f}")
        
        ax.set_title(f'{method_name}\nFinal Value Distribution')
        ax.set_xlabel('Final Portfolio Value ($)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add statistics text
        ax.text(0.02, 0.98, 
                f"Range: ${stats['range']:,.0f}\nStd: ${stats['std_final']:,.0f}\nCV: {stats['cv']:.3f}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def demonstrate_constraint_validation():
    """Demonstrate that synthetic returns meet the 2σ constraint."""
    print(f"\n🎯 CONSTRAINT VALIDATION DEMO")
    print("="*50)
    print("Validating that synthetic returns have average within 2σ of original mean")
    
    # Sample returns
    original_returns = np.array([0.02, -0.01, 0.03, -0.005, 0.015, -0.02, 0.025, -0.008])
    original_mean = np.mean(original_returns)
    original_std = np.std(original_returns)
    
    print(f"📊 Original data:")
    print(f"   Mean: {original_mean:.4f}")
    print(f"   Std Dev: {original_std:.4f}")
    print(f"   2σ range: {original_mean - 2*original_std:.4f} to {original_mean + 2*original_std:.4f}")
    
    # Generate synthetic returns
    from monte_carlo_trade_simulation import _generate_synthetic_return_sequences
    
    synthetic_sequences = _generate_synthetic_return_sequences(original_returns, num_simulations=500)
    
    # Validate constraint
    sequence_means = [np.mean(seq) for seq in synthetic_sequences]
    target_min = original_mean - 2 * original_std
    target_max = original_mean + 2 * original_std
    
    valid_count = sum(1 for m in sequence_means if target_min <= m <= target_max)
    
    print(f"\n✅ VALIDATION RESULTS:")
    print(f"   Generated sequences: {len(synthetic_sequences)}")
    print(f"   Sequences within 2σ: {valid_count}/{len(synthetic_sequences)} ({valid_count/len(synthetic_sequences)*100:.1f}%)")
    print(f"   Sequence means range: {min(sequence_means):.4f} to {max(sequence_means):.4f}")
    print(f"   Target range: {target_min:.4f} to {target_max:.4f}")
    
    # Plot distribution of means
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_means, bins=30, alpha=0.7, color='cyan', edgecolor='black')
    plt.axvline(original_mean, color='red', linestyle='-', linewidth=2, label=f'Original Mean: {original_mean:.4f}')
    plt.axvline(target_min, color='orange', linestyle='--', linewidth=2, label=f'2σ Lower: {target_min:.4f}')
    plt.axvline(target_max, color='orange', linestyle='--', linewidth=2, label=f'2σ Upper: {target_max:.4f}')
    
    plt.title('Distribution of Synthetic Sequence Means\n(Should be within 2σ bounds)')
    plt.xlabel('Sequence Mean Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("🚀 SYNTHETIC RETURNS MONTE CARLO DEMO")
    print("="*50)
    
    # Run comparison
    results = compare_simulation_methods()
    
    # Validation demo
    demonstrate_constraint_validation()
    
    print("\n✅ DEMO COMPLETE!")
    print("Key insights:")
    print("• Random shuffling: Identical outcomes (commutative property)")
    print("• Statistical sampling: Some variation from weighted sampling")
    print("• Synthetic returns: True variation with statistical constraints")
    print("• All methods maintain statistical properties of original data")
    print("• Synthetic returns enable realistic market simulation")
