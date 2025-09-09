#!/usr/bin/env python3
"""
Debug script for algorithm loading issues
"""

import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'algorithms'))

print("🔍 DEBUGGING ALGORITHM LOADING")
print("=" * 50)
print(f"Current working directory: {os.getcwd()}")
print(f"Python path entries: {len(sys.path)}")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

print("\n📁 Checking algorithms directory...")
algorithms_dir = os.path.join(os.getcwd(), 'algorithms')
if os.path.exists(algorithms_dir):
    print(f"✅ Algorithms directory exists: {algorithms_dir}")
    contents = os.listdir(algorithms_dir)
    print(f"Contents ({len(contents)} items):")
    for item in sorted(contents):
        print(f"  • {item}")
else:
    print(f"❌ Algorithms directory not found: {algorithms_dir}")
    sys.exit(1)

print("\n🔧 Testing imports...")

# Test 1: Base algorithm
try:
    from algorithms.base_algorithm import TradingAlgorithm
    print("✅ Base algorithm import successful")
except Exception as e:
    print(f"❌ Base algorithm import failed: {e}")
    traceback.print_exc()

# Test 2: Algorithm manager
try:
    from algorithms.algorithm_manager import AlgorithmManager
    print("✅ Algorithm manager import successful")
except Exception as e:
    print(f"❌ Algorithm manager import failed: {e}")
    traceback.print_exc()

# Test 3: Individual algorithm
try:
    from algorithms.technical_indicators.moving_average_crossover import MovingAverageCrossover
    print("✅ Moving Average Crossover import successful")
except Exception as e:
    print(f"❌ Moving Average Crossover import failed: {e}")
    traceback.print_exc()

# Test 4: Algorithm manager instantiation
try:
    manager = AlgorithmManager()
    print(f"✅ Algorithm manager created successfully")
    print(f"📊 Algorithms loaded: {len(manager.algorithms)}")
    if manager.algorithms:
        print("Available algorithms:")
        for name in sorted(manager.algorithms.keys()):
            print(f"  • {name}")
    else:
        print("❌ No algorithms loaded!")

        # Debug the discovery process
        print("\n🔍 Debugging discovery process...")
        from pathlib import Path
        algorithms_path = Path(manager.__file__).parent
        print(f"Algorithm manager path: {algorithms_path}")

        for category_dir in algorithms_path.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('_'):
                print(f"Checking category: {category_dir.name}")
                for file_path in category_dir.glob("*.py"):
                    if file_path.name.startswith('_'):
                        continue
                    print(f"  Found file: {file_path.name}")
                    try:
                        # Try to import this specific file
                        module_name = file_path.stem
                        import importlib.util
                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        print(f"    ✅ Module {module_name} loaded successfully")
                    except Exception as e:
                        print(f"    ❌ Module {module_name} failed: {e}")

except Exception as e:
    print(f"❌ Algorithm manager instantiation failed: {e}")
    traceback.print_exc()

print("\n🏁 Debug complete!")
