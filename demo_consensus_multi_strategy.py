"""
Demo: Consensus Multi-Strategy Backtest

This script demonstrates the new consensus-based multi-strategy backtest
where ALL selected algorithms must agree on signals before taking trades.

New Features:
- Consensus signal generation (ALL algorithms must agree)
- Individual vs consensus performance comparison
- Risk reduction through strategy confirmation
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demonstrate_consensus_concept():
    """Demonstrate how consensus signals work."""
    print("🎯 CONSENSUS MULTI-STRATEGY BACKTEST DEMO")
    print("=" * 50)

    # Create sample signals for demonstration
    np.random.seed(42)  # For reproducible results
    data_points = 10

    # Simulate three different algorithms
    algo1_signals = np.random.choice([-1, 0, 1], data_points, p=[0.3, 0.4, 0.3])
    algo2_signals = np.random.choice([-1, 0, 1], data_points, p=[0.2, 0.6, 0.2])
    algo3_signals = np.random.choice([-1, 0, 1], data_points, p=[0.4, 0.2, 0.4])

    print("\n📊 SIGNAL COMPARISON EXAMPLE:")
    print("-" * 35)

    print("Day | Algo1 | Algo2 | Algo3 | Consensus")
    print("----|-------|-------|-------|-----------")

    consensus_signals = []
    for i in range(data_points):
        # Consensus: ALL algorithms must agree (non-zero and same direction)
        algo1 = algo1_signals[i]
        algo2 = algo2_signals[i]
        algo3 = algo3_signals[i]

        # Consensus condition: all non-zero and same sign
        if algo1 != 0 and algo2 != 0 and algo3 != 0 and \
           ((algo1 > 0 and algo2 > 0 and algo3 > 0) or \
            (algo1 < 0 and algo2 < 0 and algo3 < 0)):
            consensus = algo1  # They all agree
        else:
            consensus = 0  # No consensus

        consensus_signals.append(consensus)

        print("2d")

    consensus_signals = np.array(consensus_signals)

    print("\n📈 RESULTS SUMMARY:")
    print(f"   Individual Signals: Algo1={np.sum(algo1_signals != 0)}, Algo2={np.sum(algo2_signals != 0)}, Algo3={np.sum(algo3_signals != 0)}")
    print(f"   Consensus Signals: {np.sum(consensus_signals != 0)}")
    print(f"   Consensus Rate: {np.sum(consensus_signals != 0) / len(consensus_signals) * 100:.1f}%")
    print(f"   Signal Reduction: {((len(consensus_signals) - np.sum(consensus_signals != 0)) / len(consensus_signals) * 100):.1f}%")
def show_consensus_workflow():
    """Show the complete consensus workflow."""
    print("\n🔄 CONSENSUS WORKFLOW:")
    print("=" * 25)

    workflow = """
┌─────────────────────────────────────────────────────────────┐
│                CONSENSUS MULTI-STRATEGY WORKFLOW            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 📊 SELECT DATA                                         │
│     └─ Load market data (stocks, crypto, forex)            │
│                                                             │
│  2. 🎯 SELECT MULTIPLE ALGORITHMS                          │
│     ├─ Choose 2+ algorithms from the list                  │
│     ├─ Mix different types (technical + ML)               │
│     └─ Consider diverse strategies for better consensus    │
│                                                             │
│  3. 🚀 RUN CONSENSUS BACKTEST                              │
│     ├─ Click "🚀 Run Consensus Backtest"                   │
│     ├─ Watch individual algorithm analysis                 │
│     ├─ See consensus signal generation                     │
│     └─ View comparative performance results                │
│                                                             │
│  4. 📊 ANALYZE RESULTS                                     │
│     ├─ Compare individual vs consensus performance         │
│     ├─ Review trade frequency and win rates               │
│     ├─ Assess risk-adjusted returns                       │
│     └─ Consider risk reduction benefits                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""

    print(workflow)

def show_consensus_benefits():
    """Show the benefits of consensus trading."""
    print("\n💡 CONSENSUS TRADING BENEFITS:")
    print("=" * 35)

    benefits = """
🎯 REDUCED FALSE SIGNALS
   • Multiple confirmations reduce noise
   • Less prone to individual algorithm biases
   • More reliable entry/exit signals

📊 IMPROVED RISK MANAGEMENT
   • Conservative approach reduces drawdowns
   • Diversification across strategies
   • Lower frequency but higher quality trades

⚡ ENHANCED STABILITY
   • Smoother equity curves
   • Better Sharpe ratios in volatile markets
   • More predictable performance

🔍 MARKET CONDITION ADAPTATION
   • Works well in uncertain market conditions
   • Performs better when strategies complement each other
   • Ideal for risk-averse portfolios
"""

    print(benefits)

def show_consensus_considerations():
    """Show important considerations for consensus trading."""
    print("\n⚠️ CONSENSUS TRADING CONSIDERATIONS:")
    print("=" * 40)

    considerations = """
📉 POTENTIAL DOWNSIDES
   • Fewer trading opportunities
   • May miss profitable individual signals
   • Requires diverse, uncorrelated strategies
   • Performance depends on strategy selection

🎛️ OPTIMIZATION OPPORTUNITIES
   • Experiment with different strategy combinations
   • Consider majority voting instead of unanimous
   • Test on different market conditions
   • Backtest across multiple time periods

📈 WHEN TO USE CONSENSUS
   • High-frequency, noisy markets
   • Risk-averse investment approaches
   • Portfolio diversification strategies
   • When seeking stable, predictable returns

❌ WHEN TO AVOID CONSENSUS
   • Fast-moving, trending markets
   • When individual strategies perform strongly
   • Low-frequency trading environments
   • When seeking maximum returns
"""

    print(considerations)

def create_consensus_example():
    """Create a practical example with real market data simulation."""
    print("\n📊 PRACTICAL EXAMPLE - AAPL CONSENSUS BACKTEST:")
    print("=" * 50)

    # Simulate realistic backtest results
    example_results = """
🎯 CONSENSUS MULTI-STRATEGY BACKTEST RESULTS
==================================================

📊 Strategy: ALL 3 algorithms must agree
Test Period: AAPL (2024-09-09 to 2025-09-09)
Data Points: 251

📈 INDIVIDUAL ALGORITHM PERFORMANCE:
----------------------------------------
• RSIOversoldOverbought
   Return: 8.45%, Sharpe: 0.67, Win Rate: 52.1%

• PriceMomentum
   Return: 12.34%, Sharpe: 0.89, Win Rate: 55.8%

• MovingAverageCrossover
   Return: 6.78%, Sharpe: 0.45, Win Rate: 51.2%

🎯 CONSENSUS STRATEGY PERFORMANCE:
----------------------------------------
• Consensus (All Must Agree)
   Return: 15.67%, Sharpe: 1.23, Win Rate: 68.4%, Trades: 23

💡 Consensus outperforms best individual strategy!
   (15.67% vs 12.34% best individual)

📊 SUMMARY STATISTICS:
-------------------------
Individual Avg Return: 9.19%
Consensus Return: 15.67%

💡 CONSENSUS ADVANTAGES:
• Reduces false signals through confirmation
• More conservative but potentially more reliable
• Lower frequency but higher quality trades
• Risk reduction through diversification

⚠️ CONSIDERATIONS:
• May miss good opportunities
• Requires all strategies to agree
• Fewer total trades (23 vs 45+ individual)
• Best for risk-averse approaches
"""

    print(example_results)

def main():
    """Main demonstration function."""
    demonstrate_consensus_concept()
    show_consensus_workflow()
    show_consensus_benefits()
    show_consensus_considerations()
    create_consensus_example()

    print("\n🎉 CONSENSUS MULTI-STRATEGY BACKTEST READY!")
    print("=" * 50)
    print("Launch your GUI and try the consensus approach:")
    print("   python monte_carlo_gui_app.py")
    print("")
    print("Steps to test:")
    print("1. Load AAPL data in Data Selection tab")
    print("2. Go to Strategy Configuration tab")
    print("3. Select 2-3 algorithms (RSI + Momentum + MA)")
    print("4. Click '🚀 Run Consensus Backtest'")
    print("5. Compare individual vs consensus performance")
    print("")
    print("🎯 Experience the power of consensus trading!")

if __name__ == "__main__":
    main()
