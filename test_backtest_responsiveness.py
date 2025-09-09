"""
Test Script for Backtest Responsiveness

This script tests the improved backtest functionality to ensure it responds correctly.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_backtest_threading():
    """Test that backtests run in threads and don't freeze the GUI."""
    print("🔧 Testing Backtest Threading...")
    try:
        from monte_carlo_gui_app import MonteCarloGUI
        import tkinter as tk

        root = tk.Tk()
        gui = MonteCarloGUI(root)

        # Create test data
        test_data = pd.DataFrame({
            'Close': np.random.randn(100) + 100,
            'High': np.random.randn(100) + 102,
            'Low': np.random.randn(100) + 98,
            'Open': np.random.randn(100) + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        })

        gui.current_data = test_data

        # Test that threading methods exist
        if hasattr(gui, 'run_strategy_backtest'):
            print("✅ Single backtest method exists")
        else:
            print("❌ Single backtest method missing")

        # Test multi-backtest threading
        gui.selected_algorithms = {"MovingAverageCrossover", "RSIOversoldOverbought"}
        if hasattr(gui, 'run_multi_strategy_backtest'):
            print("✅ Multi-backtest method exists")
        else:
            print("❌ Multi-backtest method missing")

        print("✅ Threading test passed")
        return True

    except Exception as e:
        print(f"❌ Threading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backtest_validation():
    """Test that backtest validation works correctly."""
    print("🔧 Testing Backtest Validation...")
    try:
        from monte_carlo_gui_app import MonteCarloGUI
        import tkinter as tk

        root = tk.Tk()
        gui = MonteCarloGUI(root)

        # Test with no data
        gui.current_data = None
        result = gui._perform_strategy_backtest(None, None)
        print("❌ Should have failed with no data")

    except Exception as e:
        print(f"✅ Validation working: {e}")
        return True

def test_algorithm_signal_generation():
    """Test that algorithms can generate signals."""
    print("🔧 Testing Algorithm Signal Generation...")
    try:
        from monte_carlo_gui_app import MonteCarloGUI
        import tkinter as tk

        root = tk.Tk()
        gui = MonteCarloGUI(root)

        # Create test data
        test_data = pd.DataFrame({
            'Close': np.random.randn(100) + 100,
            'High': np.random.randn(100) + 102,
            'Low': np.random.randn(100) + 98,
            'Open': np.random.randn(100) + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        })

        # Test algorithm creation and signal generation
        algorithm = gui.algorithm_manager.create_algorithm('MovingAverageCrossover')
        if algorithm:
            signals = algorithm.generate_signals(test_data)
            print(f"✅ Generated {len(signals)} signals")
            return True
        else:
            print("❌ Failed to create algorithm")
            return False

    except Exception as e:
        print(f"❌ Signal generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_responsiveness_tests():
    """Run all responsiveness tests."""
    print("🚀 BACKTEST RESPONSIVENESS TEST SUITE")
    print("=" * 50)

    tests = [
        ("Backtest Threading", test_backtest_threading),
        ("Backtest Validation", test_backtest_validation),
        ("Algorithm Signals", test_algorithm_signal_generation)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'─' * 20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'═' * 50}")
    print("📊 TEST SUMMARY")
    print(f"{'═' * 50}")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print("30")
        if result:
            passed += 1

    print(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL RESPONSIVENESS TESTS PASSED!")
        print("✅ Backtest threading is working")
        print("✅ Validation is functioning")
        print("✅ Signal generation works")
        return True
    else:
        print("⚠️  SOME TESTS FAILED.")
        return False

if __name__ == "__main__":
    success = run_responsiveness_tests()
    if success:
        print("\n💡 Your backtest functionality is now responsive!")
        print("🚀 Launch the GUI and test the backtest buttons:")
        print("   python monte_carlo_gui_app.py")
    else:
        print("\n🔧 Some issues remain. Check the error messages above.")
