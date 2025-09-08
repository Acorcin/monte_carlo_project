"""
Comprehensive launcher for the fully fixed Monte Carlo GUI Application
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

def check_requirements():
    """Check if all required packages are available."""
    required_packages = {
        'tkinter': 'tkinter',
        'matplotlib': 'matplotlib', 
        'pandas': 'pandas',
        'numpy': 'numpy',
        'yfinance': 'yfinance',
        'scikit-learn': 'sklearn'
    }
    
    missing_packages = []
    available_packages = []
    
    print("🔍 Checking requirements...")
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            available_packages.append(package_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"   ❌ {package_name}")
    
    return missing_packages, available_packages

def test_monte_carlo_core():
    """Test core Monte Carlo functionality."""
    print("\n🧪 Testing core Monte Carlo functionality...")
    
    try:
        from monte_carlo_trade_simulation import random_trade_order_simulation
        import numpy as np
        
        # Quick test
        sample_returns = np.array([0.02, -0.01, 0.03, -0.015, 0.025])
        results = random_trade_order_simulation(
            sample_returns, 
            num_simulations=5,
            initial_capital=10000,
            simulation_method='synthetic_returns'
        )
        
        print(f"   ✅ Monte Carlo core functionality: WORKING")
        print(f"   📊 Test results shape: {results.shape}")
        return True
        
    except Exception as e:
        print(f"   ❌ Monte Carlo core test failed: {e}")
        return False

def test_gui_components():
    """Test GUI components without showing windows."""
    print("\n🎨 Testing GUI components...")
    
    try:
        import tkinter as tk
        from monte_carlo_gui_app import MonteCarloGUI
        
        # Create hidden root
        root = tk.Tk()
        root.withdraw()
        
        # Test GUI creation
        gui = MonteCarloGUI(root)
        print(f"   ✅ GUI creation: WORKING")
        
        # Clean up
        root.destroy()
        return True
        
    except Exception as e:
        print(f"   ❌ GUI components test failed: {e}")
        return False

def main():
    """Launch the comprehensive GUI application."""
    print("🚀 Monte Carlo Trading Strategy Analyzer")
    print("=" * 60)
    print("📱 Comprehensive GUI Application Launcher")
    print("=" * 60)
    
    # Check requirements
    missing, available = check_requirements()
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"Please install with: pip install {' '.join(missing)}")
        
        # Try to show GUI error if tkinter is available
        if 'tkinter' in available:
            try:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror(
                    "Missing Dependencies", 
                    f"Missing packages: {', '.join(missing)}\n\n"
                    f"Please install with:\npip install {' '.join(missing)}"
                )
            except:
                pass
        return False
    
    print(f"✅ All required packages available: {', '.join(available)}")
    
    # Test core functionality
    if not test_monte_carlo_core():
        print("\n❌ Core Monte Carlo functionality test failed!")
        return False
    
    # Test GUI components
    if not test_gui_components():
        print("\n❌ GUI components test failed!")
        return False
    
    print("\n🎉 All systems ready!")
    print("\n🎨 Launching Monte Carlo GUI Application...")
    print("=" * 60)
    
    # Launch GUI
    try:
        from monte_carlo_gui_app import main as run_gui
        run_gui()
        return True
        
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        
        # Show error in GUI if possible
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Launch Error", f"Failed to launch GUI:\n{str(e)}")
        except:
            pass
        
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 Troubleshooting:")
        print("   1. Ensure all packages are installed: pip install matplotlib pandas numpy yfinance scikit-learn")
        print("   2. Try running: python monte_carlo_gui_app.py directly")
        print("   3. Check Python version compatibility (3.7+)")
        input("\nPress Enter to exit...")
