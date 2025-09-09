#!/usr/bin/env python3
"""
Test algorithm loading in clean directory
"""

import sys
import os

print("🔍 Testing algorithm loading in clean directory")
print("=" * 50)

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"Current directory: {current_dir}")
print(f"Python path includes current dir: {current_dir in sys.path}")

try:
    from algorithms.algorithm_manager import AlgorithmManager
    print("✅ Algorithm manager import successful")

    manager = AlgorithmManager()
    print("✅ Algorithm manager instantiated")
    print(f"📊 Algorithms loaded: {len(manager.algorithms)}")

    if manager.algorithms:
        print("Available algorithms:")
        for name in sorted(manager.algorithms.keys()):
            print(f"  • {name}")
    else:
        print("❌ No algorithms loaded")
        print("Checking algorithm discovery...")

        # Debug the discovery
        from pathlib import Path
        algorithms_dir = Path(__file__).parent / 'algorithms'
        print(f"Algorithms directory: {algorithms_dir}")
        print(f"Exists: {algorithms_dir.exists()}")

        if algorithms_dir.exists():
            for item in algorithms_dir.iterdir():
                if item.is_dir() and not item.name.startswith('_'):
                    print(f"Found category: {item.name}")
                    for py_file in item.glob('*.py'):
                        if not py_file.name.startswith('_'):
                            print(f"  Found algorithm file: {py_file.name}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("🏁 Test complete!")
