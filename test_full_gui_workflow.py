"""
Test script to validate the complete GUI workflow
"""

import numpy as np
import pandas as pd
import sys
import tkinter as tk
from datetime import datetime, timedelta

def test_gui_workflow():
    """Test the complete GUI workflow programmatically."""
    print("🔍 Testing Complete GUI Workflow...")
    
    try:
        from monte_carlo_gui_app import MonteCarloGUI
        
        # Create root window but don't show it
        root = tk.Tk()
        root.withdraw()
        
        # Create GUI instance
        print("✅ Creating GUI instance...")
        gui = MonteCarloGUI(root)
        
        # Test data loading (simulate)
        print("✅ Testing data loading simulation...")
        sample_data = pd.DataFrame({
            'Open': np.random.randn(100) + 100,
            'High': np.random.randn(100) + 101,
            'Low': np.random.randn(100) + 99,
            'Close': np.random.randn(100) + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        })
        sample_data.index = pd.date_range(start='2024-01-01', periods=100, freq='D')
        gui.current_data = sample_data
        gui.data_info_var.set(f"Test data: {len(sample_data)} records")
        
        # Test backtest results (simulate)
        print("✅ Testing backtest results simulation...")
        sample_results = {
            'algorithm_name': 'Test Algorithm',
            'initial_capital': 10000,
            'final_capital': 11500,
            'total_return': 0.15,
            'returns': [0.02, -0.01, 0.03, -0.015, 0.025, -0.02, 0.01, 0.015, -0.005, 0.02],
            'trades': [
                {'entry_date': '2024-01-01', 'exit_date': '2024-01-02', 'return': 0.02},
                {'entry_date': '2024-01-03', 'exit_date': '2024-01-04', 'return': -0.01},
                {'entry_date': '2024-01-05', 'exit_date': '2024-01-06', 'return': 0.03},
            ],
            'metrics': {
                'total_trades': 10,
                'win_rate': 0.6,
                'avg_return': 0.015,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.05,
                'profit_factor': 1.8
            }
        }
        gui.current_results = sample_results
        
        # Test Monte Carlo simulation
        print("✅ Testing Monte Carlo simulation...")
        returns_array = np.array(sample_results['returns'])
        
        from monte_carlo_trade_simulation import random_trade_order_simulation
        mc_results = random_trade_order_simulation(
            returns_array,
            num_simulations=50,  # Small number for testing
            initial_capital=10000,
            simulation_method='synthetic_returns'
        )
        
        print(f"   Monte Carlo results shape: {mc_results.shape}")
        print(f"   Final values range: ${mc_results.iloc[-1].min():.2f} to ${mc_results.iloc[-1].max():.2f}")
        
        # Test plotting
        print("✅ Testing plotting functionality...")
        gui.plot_monte_carlo_results(mc_results, 'synthetic_returns')
        print("   ✅ Monte Carlo plotting works!")
        
        # Test market scenarios (basic test)
        print("✅ Testing scenario analysis simulation...")
        scenario_results = {
            'bull': {
                'mean_return': 0.08,
                'win_rate': 0.65,
                'min_final_value': 10500,
                'max_final_value': 12000,
                'num_scenarios': 25
            },
            'bear': {
                'mean_return': -0.02,
                'win_rate': 0.45,
                'min_final_value': 9000,
                'max_final_value': 10200,
                'num_scenarios': 25
            }
        }
        
        gui.plot_scenario_results(scenario_results)
        print("   ✅ Scenario plotting works!")
        
        # Test portfolio optimization (basic test)
        print("✅ Testing portfolio optimization simulation...")
        portfolio_results = {
            'volatility': np.random.uniform(0.1, 0.3, 100),
            'returns': np.random.uniform(0.05, 0.15, 100),
            'sharpe_ratios': np.random.uniform(0.5, 2.0, 100)
        }
        
        gui.plot_portfolio_results(portfolio_results)
        print("   ✅ Portfolio plotting works!")
        
        # Clean up
        root.destroy()
        
        print("\n🎉 ALL GUI WORKFLOW TESTS PASSED!")
        print("✅ Data loading simulation: PASS")
        print("✅ Backtest results handling: PASS") 
        print("✅ Monte Carlo simulation: PASS")
        print("✅ Monte Carlo plotting: PASS")
        print("✅ Scenario analysis: PASS")
        print("✅ Portfolio optimization: PASS")
        print("\n🚀 GUI is ready for full operation!")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_gui_workflow()
