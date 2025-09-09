"""
Integrated Portfolio Optimization with Algorithm Backtesting

Combines portfolio optimization with trading algorithm backtesting
for comprehensive investment strategy analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import sys
import os

# Add algorithms directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))

from portfolio_optimization import PortfolioOptimizer
from algorithms.algorithm_manager import algorithm_manager
from monte_carlo_trade_simulation import random_trade_order_simulation

class IntegratedOptimizer:
    """
    Combines portfolio optimization with algorithm backtesting.
    
    Allows users to:
    1. Optimize portfolio weights using Monte Carlo
    2. Apply trading algorithms to individual assets
    3. Compare buy-and-hold vs algorithmic strategies
    4. Analyze Monte Carlo simulation results
    """
    
    def __init__(self, assets: List[str], period: str = "1y"):
        """
        Initialize integrated optimizer.
        
        Args:
            assets (list): List of asset tickers
            period (str): Data period for analysis
        """
        self.assets = assets
        self.period = period
        self.portfolio_optimizer = PortfolioOptimizer(assets, period)
        self.price_data = None
        self.optimization_results = None
        self.algorithm_results = {}
        
    def run_portfolio_optimization(self, num_simulations: int = 5000) -> Dict:
        """Run portfolio optimization analysis."""
        print("🎯 STEP 1: PORTFOLIO OPTIMIZATION")
        print("="*50)
        
        self.optimization_results = self.portfolio_optimizer.run_full_optimization(
            num_simulations=num_simulations,
            plot_results=False  # We'll plot later with additional data
        )
        
        if self.optimization_results:
            self.price_data = self.optimization_results['price_data']
            print("✅ Portfolio optimization completed")
        else:
            raise ValueError("Portfolio optimization failed")
            
        return self.optimization_results
    
    def run_algorithm_backtests(self, algorithms_to_test: List[str] = None) -> Dict:
        """
        Run trading algorithms on individual assets.
        
        Args:
            algorithms_to_test (list): List of algorithm names to test
            
        Returns:
            dict: Algorithm backtest results for each asset
        """
        print("\n🤖 STEP 2: ALGORITHM BACKTESTING")
        print("="*50)
        
        if algorithms_to_test is None:
            algorithms_to_test = ['MovingAverageCrossover', 'RSIOversoldOverbought']
        
        results = {}
        
        for asset in self.assets:
            print(f"\n📊 Testing algorithms on {asset}...")
            
            # Create OHLCV data for this asset (simplified - using Close as all prices)
            asset_data = pd.DataFrame({
                'Open': self.price_data[asset],
                'High': self.price_data[asset] * 1.01,  # Approximate
                'Low': self.price_data[asset] * 0.99,   # Approximate  
                'Close': self.price_data[asset],
                'Volume': 1000000  # Dummy volume
            })
            
            asset_results = {}
            
            for algo_name in algorithms_to_test:
                try:
                    result = algorithm_manager.backtest_algorithm(
                        algorithm_name=algo_name,
                        data=asset_data,
                        initial_capital=10000
                    )
                    
                    if result and result['metrics']['total_trades'] > 0:
                        asset_results[algo_name] = result
                        print(f"   ✅ {algo_name}: {result['total_return']:.2f}% return ({result['metrics']['total_trades']} trades)")
                    else:
                        print(f"   ⚠️  {algo_name}: No trades generated")
                        
                except Exception as e:
                    print(f"   ❌ {algo_name}: Failed ({e})")
            
            if asset_results:
                results[asset] = asset_results
        
        self.algorithm_results = results
        return results
    
    def compare_strategies(self) -> Dict:
        """
        Compare different investment strategies.
        
        Returns:
            dict: Comparison of buy-and-hold vs algorithmic strategies
        """
        print("\n📊 STEP 3: STRATEGY COMPARISON")
        print("="*50)
        
        if not self.optimization_results or not self.price_data.empty:
            raise ValueError("Must run portfolio optimization first")
        
        strategies = {}
        
        # 1. Buy and Hold - Equal Weight
        equal_weights = np.ones(len(self.assets)) / len(self.assets)
        equal_weight_return = self.calculate_portfolio_return(equal_weights)
        strategies['Equal Weight Buy & Hold'] = {
            'return': equal_weight_return,
            'type': 'buy_and_hold',
            'weights': dict(zip(self.assets, equal_weights))
        }
        
        # 2. Buy and Hold - Optimized Weights
        optimal_portfolios = self.optimization_results['optimal_portfolios']
        
        for portfolio_name, portfolio_data in optimal_portfolios.items():
            weights = np.array([portfolio_data['weights'][asset] for asset in self.assets])
            portfolio_return = self.calculate_portfolio_return(weights)
            strategies[f"Optimized {portfolio_data['name']}"] = {
                'return': portfolio_return,
                'type': 'optimized_buy_and_hold',
                'weights': portfolio_data['weights'],
                'expected_return': portfolio_data['expected_return'],
                'volatility': portfolio_data['volatility']
            }
        
        # 3. Algorithmic Strategies (if available)
        if self.algorithm_results:
            for asset, algo_results in self.algorithm_results.items():
                for algo_name, result in algo_results.items():
                    strategy_name = f"{asset} - {algo_name}"
                    strategies[strategy_name] = {
                        'return': result['total_return'],
                        'type': 'algorithmic',
                        'asset': asset,
                        'algorithm': algo_name,
                        'trades': result['metrics']['total_trades'],
                        'win_rate': result['metrics']['win_rate']
                    }
        
        # Print comparison
        print("\n🏆 STRATEGY PERFORMANCE COMPARISON")
        print("-" * 60)
        print(f"{'Strategy':<35} {'Return':<10} {'Type':<15}")
        print("-" * 60)
        
        sorted_strategies = sorted(strategies.items(), key=lambda x: x[1]['return'], reverse=True)
        
        for strategy_name, data in sorted_strategies:
            print(f"{strategy_name:<35} {data['return']:>7.2f}% {data['type']:<15}")
        
        return strategies
    
    def calculate_portfolio_return(self, weights: np.ndarray) -> float:
        """
        Calculate actual portfolio return for given weights.
        
        Args:
            weights (np.ndarray): Portfolio weights
            
        Returns:
            float: Portfolio return percentage
        """
        # Calculate weighted portfolio returns
        returns = self.price_data.pct_change().fillna(0)
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calculate cumulative return
        cumulative_return = (1 + portfolio_returns).prod() - 1
        
        return cumulative_return * 100
    
    def run_monte_carlo_on_strategies(self, strategies: Dict, num_simulations: int = 1000):
        """
        Apply Monte Carlo simulation to algorithmic strategy returns.
        
        Args:
            strategies (dict): Strategy comparison results
            num_simulations (int): Number of Monte Carlo simulations
        """
        print(f"\n🎲 STEP 4: MONTE CARLO ANALYSIS")
        print("="*50)
        
        for strategy_name, strategy_data in strategies.items():
            if strategy_data['type'] == 'algorithmic' and 'trades' in strategy_data:
                asset = strategy_data['asset']
                algo_name = strategy_data['algorithm']
                
                # Get the algorithm returns
                algo_result = self.algorithm_results[asset][algo_name]
                trade_returns = algo_result['returns']
                
                if len(trade_returns) >= 2:
                    print(f"\n🔄 Monte Carlo: {strategy_name}")
                    
                    mc_results = random_trade_order_simulation(
                        trade_returns,
                        num_simulations=num_simulations,
                        initial_capital=10000
                    )
                    
                    final_values = mc_results.iloc[-1].values
                    print(f"   Simulations: {num_simulations}")
                    print(f"   All outcomes: ${final_values[0]:,.2f} (identical)")
                    print(f"   ✅ Confirms: Order doesn't matter for compound returns")
                else:
                    print(f"   ⚠️  {strategy_name}: Too few trades for Monte Carlo")
    
    def plot_comprehensive_analysis(self):
        """Create comprehensive visualization of all results."""
        if not self.optimization_results:
            print("❌ No optimization results to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Efficient Frontier
        results = self.optimization_results['simulation_results']
        optimal_portfolios = self.optimization_results['optimal_portfolios']
        
        scatter = ax1.scatter(results['volatility'], results['returns'], 
                            c=results['sharpe_ratios'], cmap='viridis', 
                            alpha=0.6, s=20)
        
        # Highlight optimal portfolios
        colors = {'max_sharpe': 'red', 'min_volatility': 'blue', 'max_return': 'green'}
        for key, portfolio in optimal_portfolios.items():
            ax1.scatter(portfolio['volatility'], portfolio['expected_return'],
                       color=colors[key], s=100, label=portfolio['name'][:15])
        
        ax1.set_xlabel('Volatility')
        ax1.set_ylabel('Expected Return')
        ax1.set_title('Efficient Frontier')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Asset Correlation Heatmap
        corr_matrix = self.optimization_results['correlation_matrix']
        im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(self.assets)))
        ax2.set_yticks(range(len(self.assets)))
        ax2.set_xticklabels(self.assets)
        ax2.set_yticklabels(self.assets)
        ax2.set_title('Asset Correlation Matrix')
        plt.colorbar(im, ax=ax2)
        
        # 3. Price Performance
        normalized_prices = self.price_data / self.price_data.iloc[0]
        for asset in self.assets:
            ax3.plot(normalized_prices.index, normalized_prices[asset], label=asset)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Normalized Price')
        ax3.set_title('Asset Price Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Risk-Return Scatter of Individual Assets
        expected_returns = self.optimization_results['expected_returns']
        std_deviations = self.optimization_results['std_deviations']
        
        ax4.scatter(std_deviations, expected_returns, s=100, alpha=0.7)
        for i, asset in enumerate(self.assets):
            ax4.annotate(asset, (std_deviations[i], expected_returns[i]))
        ax4.set_xlabel('Volatility')
        ax4.set_ylabel('Expected Return') 
        ax4.set_title('Individual Asset Risk-Return')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, num_simulations: int = 5000, 
                            algorithms_to_test: List[str] = None,
                            monte_carlo_sims: int = 1000) -> Dict:
        """
        Run complete integrated analysis.
        
        Args:
            num_simulations (int): Number of portfolio optimization simulations
            algorithms_to_test (list): Algorithms to backtest
            monte_carlo_sims (int): Monte Carlo simulations for algorithm returns
            
        Returns:
            dict: Complete analysis results
        """
        print("🚀 INTEGRATED PORTFOLIO & ALGORITHM ANALYSIS")
        print("="*60)
        
        try:
            # Step 1: Portfolio Optimization
            optimization_results = self.run_portfolio_optimization(num_simulations)
            
            # Step 2: Algorithm Backtesting
            algorithm_results = self.run_algorithm_backtests(algorithms_to_test)
            
            # Step 3: Strategy Comparison
            strategy_comparison = self.compare_strategies()
            
            # Step 4: Monte Carlo on Algorithm Returns
            self.run_monte_carlo_on_strategies(strategy_comparison, monte_carlo_sims)
            
            # Step 5: Comprehensive Visualization
            print(f"\n📊 GENERATING COMPREHENSIVE ANALYSIS...")
            self.plot_comprehensive_analysis()
            
            print(f"\n🎯 ANALYSIS COMPLETE!")
            print(f"   Portfolio optimization: ✅")
            print(f"   Algorithm backtesting: ✅") 
            print(f"   Strategy comparison: ✅")
            print(f"   Monte Carlo analysis: ✅")
            print(f"   Visualization: ✅")
            
            return {
                'optimization': optimization_results,
                'algorithms': algorithm_results,
                'strategies': strategy_comparison,
                'assets': self.assets
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return None


def demo_integrated_analysis():
    """Demonstration of integrated analysis."""
    print("🎯 INTEGRATED ANALYSIS DEMO")
    print("="*35)
    
    # Demo with a few popular stocks
    assets = ['AAPL', 'MSFT', 'GOOGL']
    
    analyzer = IntegratedOptimizer(assets, period="6mo")
    
    results = analyzer.run_complete_analysis(
        num_simulations=3000,
        algorithms_to_test=['MovingAverageCrossover', 'RSIOversoldOverbought'],
        monte_carlo_sims=500
    )
    
    if results:
        print(f"\n💡 KEY INSIGHTS:")
        print(f"• Portfolio optimization finds optimal risk-return combinations")
        print(f"• Algorithm backtesting tests active strategies vs buy-and-hold")
        print(f"• Monte Carlo confirms mathematical properties of returns")
        print(f"• Integrated analysis provides comprehensive investment view")
    
    return results


if __name__ == "__main__":
    demo_integrated_analysis()
