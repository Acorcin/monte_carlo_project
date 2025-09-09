"""
Simple launcher for the Monte Carlo GUI Application
"""

import sys
import os
import importlib.util
import tkinter as tk
from tkinter import messagebox

def check_requirements():
    """Check if all required packages are available without importing them."""
    required_packages = [
        'tkinter', 'matplotlib', 'pandas', 'numpy',
        'yfinance', 'scikit-learn'
    ]

    module_name_overrides = {
        'scikit-learn': 'sklearn',
    }

    missing_packages = []

    for package in required_packages:
        name_to_check = module_name_overrides.get(package, package)
        try:
            # Use importlib to avoid importing heavy modules at startup
            if name_to_check == 'tkinter':
                # Special-case tkinter: some distros split it; try spec
                spec = importlib.util.find_spec('tkinter')
            else:
                spec = importlib.util.find_spec(name_to_check)
            if spec is None:
                missing_packages.append(package)
        except Exception:
            missing_packages.append(package)

    return missing_packages

def main():
    """Launch the GUI application."""
    print("🚀 Monte Carlo Trading Strategy Analyzer")
    print("="*50)
    
    # Check requirements
    print("Checking requirements...")
    missing = check_requirements()
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"Please install with: pip install {' '.join(missing)}")
        return
    
    print("✅ All requirements satisfied")
    
    # Launch GUI
    try:
        print("🎨 Launching GUI application...")
        
        # Import and run the main GUI
        from monte_carlo_gui_app import main as run_gui
        run_gui()
        
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        
        # Show error in simple GUI if possible
        try:
            root = tk.Tk()
            root.withdraw()  # Hide main window
            messagebox.showerror("Error", f"Failed to launch GUI:\n{str(e)}")
        except:
            pass

if __name__ == "__main__":
    main()
