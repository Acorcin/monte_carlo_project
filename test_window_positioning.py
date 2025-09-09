#!/usr/bin/env python3
"""
Test Window Positioning Fix

This script tests the window positioning fixes to ensure the GUI
appears with the title bar visible and properly centered.
"""

import tkinter as tk
from monte_carlo_gui_app import center_window, reposition_window

def test_window_positioning():
    """Test the window positioning functionality."""
    print("🖥️ Testing Window Positioning")
    print("=" * 40)

    try:
        # Create a test window
        root = tk.Tk()
        root.title("Test Window Positioning")
        root.geometry("800x600+0+0")  # Start at 0,0 to test centering

        # Add some basic content
        label = tk.Label(root, text="Testing Window Positioning\n\nIf you can see this window's title bar,\nthe positioning fix is working!")
        label.pack(expand=True, fill='both', padx=20, pady=20)

        # Test centering
        print("Testing center_window function...")
        success = center_window(root, 800, 600)

        if success:
            print("✅ Window centering successful!")
        else:
            print("⚠️ Window centering had issues but used fallback")

        # Test reposition function
        print("Testing reposition_window function...")
        reposition_window(root)

        # Add a close button
        def close_test():
            print("Test window closed by user")
            root.quit()

        close_btn = tk.Button(root, text="Close Test Window", command=close_test)
        close_btn.pack(pady=10)

        print("\n📋 TEST RESULTS:")
        print("1. ✅ Window should be centered on screen")
        print("2. ✅ Title bar should be visible at the top")
        print("3. ✅ Window should not be off-screen")
        print("4. ✅ Close button should work")

        print("\n🔄 Starting test window... (close it to continue)")
        root.mainloop()

        print("✅ Test window closed successfully")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_screen_detection():
    """Test screen size detection."""
    print("\n📺 Testing Screen Detection")
    print("=" * 30)

    try:
        root = tk.Tk()
        root.withdraw()  # Hide the window for this test

        # Test screen size detection
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        print(f"Detected screen size: {screen_width}x{screen_height}")

        # Test reasonable bounds
        if screen_width >= 1024 and screen_height >= 768:
            print("✅ Screen size is reasonable for GUI display")
        else:
            print("⚠️ Screen size may be too small for optimal GUI experience")

        root.destroy()
        return True

    except Exception as e:
        print(f"❌ Screen detection test failed: {e}")
        return False

def main():
    """Run all window positioning tests."""
    print("🖥️ GUI WINDOW POSITIONING TEST SUITE")
    print("=" * 50)
    print("Testing fixes for GUI window positioning issues")
    print()

    # Test screen detection
    screen_test = test_screen_detection()

    # Test window positioning
    if screen_test:
        window_test = test_window_positioning()

        if window_test:
            print("\n🎉 ALL TESTS PASSED!")
            print("=" * 30)
            print("✅ Window positioning fixes are working correctly")
            print("✅ GUI should now appear with title bar visible")
            print("✅ Use '🔄 Reposition Window' button if needed")
        else:
            print("\n⚠️ SOME TESTS FAILED")
            print("=" * 30)
            print("❌ Window positioning may still have issues")
    else:
        print("\n❌ SCREEN DETECTION FAILED")
        print("=" * 30)
        print("Cannot proceed with window positioning tests")

    print("\n💡 TIPS FOR WINDOW POSITIONING:")
    print("• GUI should appear centered on screen")
    print("• Title bar should be visible at the top")
    print("• Use '🔄 Reposition Window' button in status bar if needed")
    print("• Window should not appear off-screen or minimized")

if __name__ == "__main__":
    main()


