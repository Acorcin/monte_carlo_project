"""
View Your Monte Carlo Trading Application

This script shows you how to view and interact with your enhanced
Monte Carlo trading application with separated tabs and fixed parameters.

Run this to see your application in action!
"""

import os
import sys

def show_viewing_options():
    """Show different ways to view the application."""

    print("🎯 Your Monte Carlo Trading Application - Viewing Guide")
    print("=" * 60)

    print("\n📋 AVAILABLE VIEWING OPTIONS:")
    print("1️⃣  Launch Main GUI (Recommended)")
    print("2️⃣  Launch Modern GUI (Alternative)")
    print("3️⃣  View Demo Scripts")
    print("4️⃣  Check System Status")
    print("5️⃣  View Documentation")

    print("\n🔥 MAIN GUI FEATURES:")
    print("   ✅ Separated Data & Strategy tabs")
    print("   ✅ Fixed parameter handling")
    print("   ✅ Real-time calculations")
    print("   ✅ Risk management integration")
    print("   ✅ Live market data loading")

    print("\n📊 WHAT YOU CAN VIEW:")
    print("   • 📈 Market data charts and analysis")
    print("   • 🎯 Trading strategy configuration")
    print("   • 🎲 Monte Carlo simulation results")
    print("   • 📊 Portfolio optimization")
    print("   • 💧 Advanced liquidity analysis")
    print("   • ⚙️ Risk management dashboard")

def launch_main_gui():
    """Launch the main improved GUI."""
    print("\n🚀 Launching Main GUI...")
    print("This will open your enhanced Monte Carlo application with:")
    print("   • Separated tabs (Data | Strategy | Monte Carlo | etc.)")
    print("   • Fixed parameter controls")
    print("   • Real-time updates")
    print("   • Professional interface")

    try:
        # Import and run the main GUI
        from monte_carlo_gui_app import main
        main()
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")

def launch_modern_gui():
    """Launch the alternative modern GUI."""
    print("\n🎨 Launching Modern GUI...")
    print("This will open a modern CustomTkinter interface with:")
    print("   • Dark/light mode toggle")
    print("   • Enhanced visual design")
    print("   • Responsive layout")

    try:
        # Import and run the modern GUI
        from modern_gui_app import ModernMonteCarloGUI
        app = ModernMonteCarloGUI()
        app.run()
    except Exception as e:
        print(f"❌ Error launching modern GUI: {e}")
        print("CustomTkinter may not be installed:")
        print("pip install customtkinter")

def show_demo_scripts():
    """Show available demo scripts."""
    print("\n📚 Available Demo Scripts:")
    print("-" * 30)

    scripts = [
        "example_risk_liquidity_integration.py - Risk management demo",
        "test_improved_gui.py - GUI improvement test",
        "view_application.py - This viewing guide",
    ]

    for script in scripts:
        print(f"   • {script}")

    print("\n💡 Run any demo with: python <script_name>")

def check_system_status():
    """Check if all components are working."""
    print("\n🔍 System Status Check:")
    print("-" * 25)

    # Check Python version
    print(f"   ✅ Python: {sys.version.split()[0]}")

    # Check if main GUI file exists
    if os.path.exists("monte_carlo_gui_app.py"):
        print("   ✅ Main GUI file found")
    else:
        print("   ❌ Main GUI file missing")

    # Check if modern GUI file exists
    if os.path.exists("modern_gui_app.py"):
        print("   ✅ Modern GUI file found")
    else:
        print("   ❌ Modern GUI file missing")

    # Check requirements file
    if os.path.exists("requirements.txt"):
        print("   ✅ Requirements file found")
    else:
        print("   ❌ Requirements file missing")

    # Check algorithm directory
    if os.path.exists("algorithms"):
        print("   ✅ Algorithms directory found")
    else:
        print("   ❌ Algorithms directory missing")

    print("\n📦 Key Dependencies Status:")
    try:
        import pandas
        print("   ✅ pandas available")
    except ImportError:
        print("   ❌ pandas not installed")

    try:
        import numpy
        print("   ✅ numpy available")
    except ImportError:
        print("   ❌ numpy not installed")

    try:
        import matplotlib
        print("   ✅ matplotlib available")
    except ImportError:
        print("   ❌ matplotlib not installed")

def show_documentation():
    """Show available documentation."""
    print("\n📖 Available Documentation:")
    print("-" * 30)

    docs = [
        "README.md - Main project documentation",
        "IMPROVEMENT_SUMMARY.md - Enhancement details",
        "USAGE_GUIDE.md - How to use the application",
        "GUI_ENHANCEMENT_GUIDE.md - GUI improvements",
        "LIQUIDITY_ANALYZER_INTEGRATION_GUIDE.md - Liquidity features",
    ]

    for doc in docs:
        if os.path.exists(doc):
            print(f"   ✅ {doc}")
        else:
            print(f"   ❌ {doc} (missing)")

def main():
    """Main viewing interface."""
    while True:
        print("\n" + "=" * 60)
        show_viewing_options()

        try:
            choice = input("\n👆 Choose an option (1-5) or 'q' to quit: ").strip().lower()

            if choice == 'q':
                print("👋 Goodbye! Happy trading! 📈")
                break
            elif choice == '1':
                launch_main_gui()
            elif choice == '2':
                launch_modern_gui()
            elif choice == '3':
                show_demo_scripts()
            elif choice == '4':
                check_system_status()
            elif choice == '5':
                show_documentation()
            else:
                print("❌ Invalid choice. Please enter 1-5 or 'q'.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye! Happy trading! 📈")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

        input("\n⏎ Press Enter to continue...")

if __name__ == "__main__":
    main()
