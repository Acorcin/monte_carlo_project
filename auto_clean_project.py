"""
Auto Clean Project - Creates a clean version without user prompts
"""

import os
import shutil
import sys
from pathlib import Path

def create_clean_project(clean_project_name="monte_carlo_clean"):
    """Create a clean version of the project with only essential files."""

    # Define essential files and directories
    essential_files = [
        # Core application files
        "monte_carlo_gui_app.py",
        "data_fetcher.py", 
        "monte_carlo_trade_simulation.py",
        "risk_management.py",
        "liquidity_analyzer.py",
        "modern_gui_app.py",  # Alternative GUI

        # Dependencies and configuration
        "requirements.txt",

        # Documentation (keep only essential ones)
        "README.md",
        "USAGE_GUIDE.md",
        "GUI_APPLICATION_GUIDE.md",

        # Data files (keep minimal examples)
        "MNQ_F_1h_30days.csv",  # Sample data
    ]

    essential_directories = [
        "algorithms",  # All algorithm implementations
    ]

    # Files to exclude (clutter)
    exclude_patterns = [
        # Demo files
        "demo_",
        # Test files  
        "test_",
        # Debug files
        "debug_",
        "diagnose_",
        # Utility scripts
        "fix_",
        "calibrate_",
        "launch_",
        "view_",
        "save_",
        "integrated_",
        "interactive_",
        "market_",
        "method_",
        "portfolio_",
        "real_data_",
        "sample_",
        "sharpe_",
        "simple_",
        "statistical_",
        "synthetic_",
        "your_code_",
        "zero_risk_",
        # Documentation files (keep only essential)
        "_SUMMARY.md",
        "_GUIDE.md", 
        "_INTEGRATION",
        "_ENHANCEMENT",
        "_FIX",
        "_SUPPORT",
        "_ANALYSIS",
        # Generated files
        ".png",
        "__pycache__",
        ".git",
    ]

    print("🧹 CREATING CLEAN PROJECT DIRECTORY")
    print("=" * 50)
    print(f"Target directory: {clean_project_name}")
    print()

    # Create clean directory
    if os.path.exists(clean_project_name):
        print(f"⚠️  Directory '{clean_project_name}' already exists. Removing...")
        shutil.rmtree(clean_project_name)

    os.makedirs(clean_project_name)
    print(f"✅ Created directory: {clean_project_name}")

    # Copy essential files
    copied_files = []
    skipped_files = []

    # Copy individual files
    for file in essential_files:
        if os.path.exists(file):
            shutil.copy2(file, clean_project_name)
            copied_files.append(file)
            print(f"✅ Copied: {file}")
        else:
            print(f"⚠️  Not found: {file}")

    # Copy essential directories
    for directory in essential_directories:
        if os.path.exists(directory):
            shutil.copytree(directory, os.path.join(clean_project_name, directory))
            copied_files.append(f"{directory}/ (entire directory)")
            print(f"✅ Copied directory: {directory}")
        else:
            print(f"⚠️  Directory not found: {directory}")

    # Copy any additional essential files that might be missed
    current_dir = os.getcwd()
    for file in os.listdir(current_dir):
        if file not in [clean_project_name] and os.path.isfile(file):
            # Check if file should be excluded
            should_exclude = False
            for pattern in exclude_patterns:
                if pattern in file.lower():
                    should_exclude = True
                    break

            # Copy if not excluded and not already copied
            if not should_exclude and file not in essential_files:
                # Additional essential files that might be useful
                additional_essentials = [
                    "LICENSE",
                    ".gitignore",
                    "setup.py",
                    "pyproject.toml",
                    "MANIFEST.in"
                ]
                if file in additional_essentials:
                    shutil.copy2(file, clean_project_name)
                    copied_files.append(file)
                    print(f"✅ Copied additional: {file}")

    # Create a summary file
    create_summary_file(clean_project_name, copied_files, skipped_files)

    print()
    print("🎉 CLEAN PROJECT CREATED SUCCESSFULLY!")
    print("=" * 50)
    print(f"📁 Location: {os.path.abspath(clean_project_name)}")
    print(f"📊 Files copied: {len(copied_files)}")
    print()
    print("🚀 To use the clean version:")
    print(f"   cd {clean_project_name}")
    print("   pip install -r requirements.txt")
    print("   python monte_carlo_gui_app.py")
    print()
    print("📋 See CLEAN_PROJECT_SUMMARY.txt for details")

def create_summary_file(clean_project_name, copied_files, skipped_files):
    """Create a summary file with details about the clean project."""

    summary_path = os.path.join(clean_project_name, "CLEAN_PROJECT_SUMMARY.txt")

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("MONTE CARLO TRADING APPLICATION - CLEAN VERSION\n")
        f.write("=" * 60 + "\n\n")
        f.write("This is a cleaned version of the Monte Carlo trading application\n")
        f.write("containing only the essential files needed for the application to function.\n\n")

        f.write("ESSENTIAL FILES INCLUDED:\n")
        f.write("-" * 30 + "\n")
        for file in copied_files:
            f.write(f"✓ {file}\n")

        f.write("\nFILES EXCLUDED (to reduce clutter):\n")
        f.write("-" * 40 + "\n")
        excluded_categories = [
            "Demo scripts (demo_*.py)",
            "Test files (test_*.py)",
            "Debug scripts (debug_*.py)", 
            "Utility scripts (fix_*, calibrate_*, etc.)",
            "Excessive documentation (*.md files)",
            "Generated images (*.png)",
            "Cache directories (__pycache__)",
            "Backup/alternative versions"
        ]

        for category in excluded_categories:
            f.write(f"• {category}\n")

        f.write("\nUSAGE:\n")
        f.write("-" * 10 + "\n")
        f.write("1. Install dependencies: pip install -r requirements.txt\n")
        f.write("2. Run main application: python monte_carlo_gui_app.py\n")
        f.write("3. Run alternative GUI: python modern_gui_app.py\n\n")

        f.write("FEATURES INCLUDED:\n")
        f.write("-" * 20 + "\n")
        features = [
            "Monte Carlo simulation with multiple methods",
            "Multi-strategy consensus backtesting",
            "Advanced ML algorithms (LSTM, Transformer, etc.)",
            "Risk management and position sizing",
            "Liquidity analysis and market structure",
            "Interactive GUI with real-time updates",
            "Portfolio optimization tools",
            "Comprehensive charting and visualization"
        ]

        for feature in features:
            f.write(f"• {feature}\n")

        f.write("\n\nCreated by: Monte Carlo Project Auto Cleaner\n")
        f.write("Original project maintained in parent directory\n")

    print(f"📋 Created summary file: {summary_path}")

def show_project_stats():
    """Show statistics about the current project vs clean version."""

    print("\n📊 PROJECT STATISTICS:")
    print("=" * 30)

    # Count files in current directory
    current_files = []
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git']]
        for file in files:
            current_files.append(os.path.join(root, file))

    # Count Python files
    python_files = [f for f in current_files if f.endswith('.py')]
    test_files = [f for f in python_files if 'test_' in f.lower() or 'demo_' in f.lower()]
    doc_files = [f for f in current_files if f.endswith('.md') or f.endswith('.txt')]

    print("Current project:")
    print(f"  📁 Total files: {len(current_files)}")
    print(f"  🐍 Python files: {len(python_files)}")
    print(f"  🧪 Test/Demo files: {len(test_files)}")
    print(f"  📖 Documentation: {len(doc_files)}")

    # Estimate clean version
    essential_python = [f for f in python_files if not any(x in f.lower() for x in ['test_', 'demo_', 'debug_', 'fix_', 'calibrate_', 'launch_', 'view_', 'save_'])]
    essential_docs = [f for f in doc_files if any(x in f.lower() for x in ['readme', 'usage', 'guide'])]

    print("\nClean version (estimated):")
    print(f"  📁 Total files: {len(essential_python) + len(essential_docs) + 20}")  # +20 for algorithms and other essentials
    print(f"  🐍 Python files: {len(essential_python)}")
    print(f"  📖 Documentation: {len(essential_docs)}")
    print(f"  📉 Reduction: ~{((len(current_files) - (len(essential_python) + len(essential_docs) + 20)) / len(current_files) * 100):.1f}%")

if __name__ == "__main__":
    print("🧹 Monte Carlo Project Auto Cleaner")
    print("=" * 40)

    # Show current stats
    show_project_stats()

    print("\n" + "=" * 50)
    
    # Create clean project automatically
    project_name = "monte_carlo_clean"
    print(f"Creating clean project: {project_name}")
    create_clean_project(project_name)

    print("\n💡 Tip: The original project remains unchanged in the current directory.")
    print("🎯 Navigate to the clean directory to use the streamlined version!")
