"""
Test script to diagnose Monte Carlo issues in the GUI
"""

import numpy as np
import pandas as pd
import sys
import os

# Test Monte Carlo simulation directly
print("🔍 Testing Monte Carlo simulation...")

try:
    from monte_carlo_trade_simulation import random_trade_order_simulation
    
    # Create sample trade returns
    sample_returns = np.array([0.02, -0.01, 0.03, -0.015, 0.025, -0.02, 0.01, 0.015, -0.005, 0.02])
    print(f"✅ Sample returns: {sample_returns}")
    
    # Test the simulation
    results = random_trade_order_simulation(
        sample_returns,
        num_simulations=10,  # Small number for testing
        initial_capital=10000,
        simulation_method='synthetic_returns'
    )
    
    print(f"✅ Monte Carlo simulation completed!")
    print(f"   Shape: {results.shape}")
    print(f"   Final values range: ${results.iloc[-1].min():.2f} to ${results.iloc[-1].max():.2f}")
    
    # Test GUI imports
    print("\n🔍 Testing GUI imports...")
    
    import tkinter as tk
    print("✅ tkinter imported")
    
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    print("✅ matplotlib GUI backend imported")
    
    # Test the GUI Monte Carlo function specifically
    print("\n🔍 Testing GUI Monte Carlo integration...")
    
    from monte_carlo_gui_app import MonteCarloGUI
    
    # Create a test root window (but don't show it)
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    # Create GUI instance
    gui = MonteCarloGUI(root)
    print("✅ GUI instance created")
    
    # Test the Monte Carlo plotting function
    gui.plot_monte_carlo_results(results, 'synthetic_returns')
    print("✅ Monte Carlo plotting function works")
    
    root.destroy()
    
    print("\n🎉 All tests passed! Monte Carlo should work in GUI.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
