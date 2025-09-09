#!/usr/bin/env python3
"""
Debug Algorithm Discovery

This script helps debug why the liquidity strategy isn't being discovered.
"""

import sys
import os
import importlib.util
import inspect
from pathlib import Path

# Add algorithms to path
sys.path.append('algorithms')

from algorithms.base_algorithm import TradingAlgorithm

def debug_discovery():
    """Debug the algorithm discovery process."""
    print("🔍 Debugging Algorithm Discovery")
    print("=" * 50)
    
    # Check the file path
    file_path = Path("algorithms/technical_indicators/liquidity_structure_strategy.py")
    print(f"📁 File exists: {file_path.exists()}")
    print(f"📁 File path: {file_path.absolute()}")
    
    if not file_path.exists():
        print("❌ File does not exist!")
        return
    
    try:
        # Try to import the module manually
        print("\n🔧 Manual import test...")
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(file_path.parent))
        spec.loader.exec_module(module)
        sys.path.pop(0)
        print("✅ Module imported successfully")
        
        # Find classes in the module
        print("\n🔍 Scanning for classes...")
        classes_found = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            print(f"   Found class: {name}")
            print(f"     Is subclass of TradingAlgorithm: {issubclass(obj, TradingAlgorithm)}")
            print(f"     Is TradingAlgorithm itself: {obj == TradingAlgorithm}")
            print(f"     Is abstract: {inspect.isabstract(obj)}")
            
            if (issubclass(obj, TradingAlgorithm) and 
                obj != TradingAlgorithm and 
                not inspect.isabstract(obj)):
                classes_found.append((name, obj))
                print(f"     ✅ WOULD BE LOADED: {name}")
            else:
                print(f"     ❌ Would NOT be loaded")
            print()
        
        print(f"📊 Total trading algorithm classes found: {len(classes_found)}")
        
        # Test creating instances
        for name, cls in classes_found:
            try:
                print(f"\n🧪 Testing instance creation: {name}")
                instance = cls()
                print(f"   ✅ Instance created successfully")
                print(f"   📝 Name: {instance.name}")
                print(f"   📝 Description: {instance.description[:100]}...")
                print(f"   📝 Type: {instance.get_algorithm_type()}")
            except Exception as e:
                print(f"   ❌ Failed to create instance: {e}")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()

def test_algorithm_manager():
    """Test the algorithm manager discovery."""
    print("\n" + "=" * 50)
    print("🎯 Testing Algorithm Manager Discovery")
    print("=" * 50)
    
    try:
        from algorithms.algorithm_manager import AlgorithmManager
        
        print("📊 Creating fresh algorithm manager...")
        manager = AlgorithmManager()
        
        print(f"🔍 Algorithms discovered: {len(manager.algorithms)}")
        for name in manager.algorithms.keys():
            print(f"   • {name}")
        
        # Check if our algorithm is there
        liquidity_algos = [name for name in manager.algorithms.keys() if 'Liquidity' in name]
        if liquidity_algos:
            print(f"\n✅ Found liquidity algorithms: {liquidity_algos}")
        else:
            print(f"\n❌ No liquidity algorithms found")
            
            # Debug: manually load the directory
            print("\n🔧 Manual directory loading test...")
            from pathlib import Path
            tech_indicators_dir = Path("algorithms/technical_indicators")
            print(f"Directory exists: {tech_indicators_dir.exists()}")
            
            if tech_indicators_dir.exists():
                print("Files in directory:")
                for file in tech_indicators_dir.glob("*.py"):
                    print(f"   📄 {file.name}")
                
                # Try loading this specific directory
                try:
                    manager._load_algorithms_from_directory(tech_indicators_dir)
                    print(f"After manual loading: {len(manager.algorithms)} algorithms")
                    for name in manager.algorithms.keys():
                        print(f"   • {name}")
                except Exception as e:
                    print(f"Manual loading failed: {e}")
        
    except Exception as e:
        print(f"❌ Algorithm manager test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_discovery()
    test_algorithm_manager()

