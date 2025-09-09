"""
Monte Carlo Trading Application Launcher
========================================

Quick launcher for the clean Monte Carlo trading application.
"""

import subprocess
import sys
import os

def main():
    print("🚀 LAUNCHING MONTE CARLO TRADING APPLICATION")
    print("=" * 50)
    print("📁 Clean Version - Streamlined & Professional")
    print("🎯 All features included, clutter removed")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("monte_carlo_gui_app.py"):
        print("❌ Error: monte_carlo_gui_app.py not found!")
        print("   Make sure you're in the monte_carlo_clean directory")
        return
    
    print("✅ Starting application...")
    
    try:
        # Launch the main GUI application
        subprocess.run([sys.executable, "monte_carlo_gui_app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching application: {e}")
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("\n👋 Thanks for using Monte Carlo Trading Application!")

if __name__ == "__main__":
    main()
