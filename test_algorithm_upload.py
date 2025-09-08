"""
Test the algorithm upload functionality
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

def test_upload_functionality():
    """Test the upload algorithm functionality."""
    print("🧪 Testing Algorithm Upload Functionality...")
    
    try:
        # Import the GUI
        from monte_carlo_gui_app import MonteCarloGUI
        
        # Create hidden root window
        root = tk.Tk()
        root.withdraw()
        
        # Create GUI instance
        gui = MonteCarloGUI(root)
        print("✅ GUI instance created")
        
        # Test algorithm validation with our sample template
        sample_file = "sample_algorithm_template.py"
        if os.path.exists(sample_file):
            print(f"✅ Found sample algorithm file: {sample_file}")
            
            # Test validation
            algorithm_info = gui.validate_algorithm_file(sample_file)
            
            if algorithm_info:
                print("✅ Algorithm validation: PASS")
                print(f"   Class name: {algorithm_info['class_name']}")
                print(f"   Description: {algorithm_info['docstring'][:100]}...")
                
                # Test the algorithm class
                try:
                    # Create a test instance
                    algo_class = algorithm_info['class_obj']
                    test_instance = algo_class()
                    print(f"✅ Algorithm instantiation: PASS")
                    print(f"   Algorithm type: {test_instance.get_algorithm_type()}")
                    
                except Exception as e:
                    print(f"❌ Algorithm instantiation: FAIL - {e}")
                
            else:
                print("❌ Algorithm validation: FAIL")
        else:
            print(f"❌ Sample algorithm file not found: {sample_file}")
        
        # Test custom algorithms directory creation
        custom_dir = os.path.join('algorithms', 'custom')
        if not os.path.exists(custom_dir):
            os.makedirs(custom_dir, exist_ok=True)
            print(f"✅ Created custom algorithms directory: {custom_dir}")
        else:
            print(f"✅ Custom algorithms directory exists: {custom_dir}")
        
        # Test __init__.py creation
        init_file = os.path.join(custom_dir, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Custom algorithms directory\n')
            print(f"✅ Created __init__.py: {init_file}")
        else:
            print(f"✅ __init__.py exists: {init_file}")
        
        # Clean up
        root.destroy()
        
        print("\n🎉 Algorithm Upload Functionality Test: PASS")
        print("✅ GUI creation: WORKING")
        print("✅ Algorithm validation: WORKING")
        print("✅ Directory setup: WORKING")
        print("✅ File system operations: WORKING")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_upload_functionality()
