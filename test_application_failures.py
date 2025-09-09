"""
Comprehensive Test Script for Monte Carlo GUI Application Failures

This script tests all major components of the application to identify
and resolve any testing failures.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gui_initialization():
    """Test GUI initialization and component loading."""
    print("🔧 Testing GUI Initialization...")
    try:
        from monte_carlo_gui_app import MonteCarloGUI
        import tkinter as tk

        root = tk.Tk()
        gui = MonteCarloGUI(root)

        # Test basic attributes exist
        assert hasattr(gui, 'algorithm_manager')
        assert hasattr(gui, 'notebook')
        assert hasattr(gui, 'current_data')

        # Test tabs exist
        tab_names = [gui.notebook.tab(i, "text") for i in range(gui.notebook.index("end"))]
        expected_tabs = ["📊 Data Selection", "🎯 Strategy Configuration",
                        "🎲 Monte Carlo", "🌍 Scenarios",
                        "📈 Portfolio", "📊 Results"]

        for expected_tab in expected_tabs:
            assert expected_tab in tab_names, f"Missing tab: {expected_tab}"

        print("✅ GUI initialization test passed")
        return True

    except Exception as e:
        print(f"❌ GUI initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_algorithm_loading():
    """Test algorithm loading and creation."""
    print("🔧 Testing Algorithm Loading...")
    try:
        from monte_carlo_gui_app import MonteCarloGUI
        import tkinter as tk

        root = tk.Tk()
        gui = MonteCarloGUI(root)

        # Test algorithm manager
        algorithms = list(gui.algorithm_manager.algorithms.keys())
        print(f"   Loaded {len(algorithms)} algorithms: {algorithms}")

        # Test algorithm creation
        for algo_name in ['MovingAverageCrossover', 'RSIOversoldOverbought']:
            if algo_name in algorithms:
                algorithm = gui.algorithm_manager.create_algorithm(algo_name)
                assert algorithm is not None, f"Failed to create {algo_name}"
                print(f"   ✅ Created algorithm: {algo_name}")

        print("✅ Algorithm loading test passed")
        return True

    except Exception as e:
        print(f"❌ Algorithm loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_handling():
    """Test data loading and processing."""
    print("🔧 Testing Data Handling...")
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
        assert gui.current_data is not None
        assert len(gui.current_data) == 100
        assert 'Close' in gui.current_data.columns

        print("✅ Data handling test passed")
        return True

    except Exception as e:
        print(f"❌ Data handling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_backtest():
    """Test single strategy backtest functionality."""
    print("🔧 Testing Single Strategy Backtest...")
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

        # Test algorithm creation and backtest
        algorithm = gui.algorithm_manager.create_algorithm('MovingAverageCrossover')
        assert algorithm is not None

        results = gui._perform_strategy_backtest(algorithm, test_data)

        # Verify results structure
        required_keys = ['total_return', 'max_drawdown', 'sharpe_ratio', 'num_trades', 'win_rate']
        for key in required_keys:
            assert key in results, f"Missing key: {key}"

        print(f"   Backtest results: Total Return: {results['total_return']:.4f}")
        print("✅ Single backtest test passed")
        return True

    except Exception as e:
        print(f"❌ Single backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_backtest():
    """Test multi-strategy backtest functionality."""
    print("🔧 Testing Multi-Strategy Backtest...")
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

        # Select algorithms for testing
        gui.selected_algorithms = {'MovingAverageCrossover', 'RSIOversoldOverbought'}

        # Test multi-backtest
        results = {}

        for algo_name in gui.selected_algorithms:
            algorithm = gui.algorithm_manager.create_algorithm(algo_name)
            if algorithm:
                algo_results = gui._perform_strategy_backtest(algorithm, test_data)
                results[algo_name] = algo_results
                print(f"   ✅ Tested {algo_name}")

        assert len(results) > 0, "No algorithms were successfully tested"

        # Test display function
        gui._display_multi_backtest_results(results)
        print("   ✅ Multi-backtest display working")

        print("✅ Multi-backtest test passed")
        return True

    except Exception as e:
        print(f"❌ Multi-backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_interactions():
    """Test GUI interaction elements."""
    print("🔧 Testing GUI Interactions...")
    try:
        from monte_carlo_gui_app import MonteCarloGUI
        import tkinter as tk

        root = tk.Tk()
        gui = MonteCarloGUI(root)

        # Test strategy tab components
        assert hasattr(gui, 'strategy_frame')
        assert hasattr(gui, 'selected_algorithms')
        assert hasattr(gui, 'algo_checkbuttons')

        # Test parameter handling
        assert hasattr(gui, 'risk_mgmt_var')
        assert hasattr(gui, 'stop_loss_var')
        assert hasattr(gui, 'take_profit_var')

        # Test position size calculation
        gui.capital_var.set("10000")
        gui.risk_mgmt_var.set(0.02)
        gui.update_position_size()

        position_size = gui.position_size_var.get()
        assert "$200" in position_size, f"Unexpected position size: {position_size}"

        print("✅ GUI interactions test passed")
        return True

    except Exception as e:
        print(f"❌ GUI interactions failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_algorithm_descriptions():
    """Test algorithm description functionality."""
    print("🔧 Testing Algorithm Descriptions...")
    try:
        from monte_carlo_gui_app import MonteCarloGUI
        import tkinter as tk

        root = tk.Tk()
        gui = MonteCarloGUI(root)

        # Test description display for different algorithms
        test_algorithms = ['MovingAverageCrossover', 'LSTMTradingStrategy', 'RSIOversoldOverbought']

        for algo_name in test_algorithms:
            if algo_name in gui.algorithm_manager.algorithms:
                gui.show_algorithm_description(algo_name)
                print(f"   ✅ Description displayed for {algo_name}")

        print("✅ Algorithm descriptions test passed")
        return True

    except Exception as e:
        print(f"❌ Algorithm descriptions failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests and report results."""
    print("🚀 COMPREHENSIVE APPLICATION FAILURE TEST")
    print("=" * 50)

    tests = [
        ("GUI Initialization", test_gui_initialization),
        ("Algorithm Loading", test_algorithm_loading),
        ("Data Handling", test_data_handling),
        ("Single Backtest", test_single_backtest),
        ("Multi Backtest", test_multi_backtest),
        ("GUI Interactions", test_gui_interactions),
        ("Algorithm Descriptions", test_algorithm_descriptions)
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
        print("🎉 ALL TESTS PASSED! Application is working correctly.")
        return True
    else:
        print("⚠️  SOME TESTS FAILED. Check the error messages above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    if success:
        print("\n💡 To run the GUI: python monte_carlo_gui_app.py")
    else:
        print("\n🔧 Fix the failed tests before running the GUI.")
