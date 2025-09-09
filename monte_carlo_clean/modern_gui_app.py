"""
Modern Monte Carlo Trading Strategy GUI Application

An enhanced, modern GUI with:
- Dark mode support
- Better visual design
- Improved user experience
- Real-time data integration
- Advanced visualizations

Requirements: pip install customtkinter matplotlib pandas numpy plotly seaborn
"""

import customtkinter as ctk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import threading
import sys
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# Set matplotlib style
plt.style.use('dark_background')
sns.set_palette("husl")

class ModernMonteCarloGUI:
    """Modern GUI Application for Monte Carlo Trading Analysis with enhanced design."""

    def __init__(self):
        """Initialize the modern GUI application."""
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
        ctk.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

        # Create main window
        self.root = ctk.CTk()
        self.root.title("🚀 Modern Monte Carlo Trading Analyzer")
        self.root.geometry("1600x1000")

        # Initialize data and managers
        self.current_data = None
        self.current_results = None
        self.is_dark_mode = True

        # Create main interface
        self.create_modern_interface()

        # Status bar
        self.create_status_bar()

    def create_modern_interface(self):
        """Create the modern main interface with tabs and navigation."""

        # Create main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Create header with title and controls
        self.create_header()

        # Create tabview for main functionality
        self.tabview = ctk.CTkTabview(self.main_container, width=1400, height=800)
        self.tabview.pack(pady=(20, 0), padx=20, fill="both", expand=True)

        # Create tabs
        self.create_data_tab()
        self.create_analysis_tab()
        self.create_visualization_tab()
        self.create_portfolio_tab()
        self.create_settings_tab()

    def create_header(self):
        """Create modern header with title and quick controls."""
        header_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 0))

        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="🚀 Monte Carlo Trading Analyzer",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(side="left")

        # Quick action buttons
        button_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        button_frame.pack(side="right")

        # Theme toggle button
        self.theme_button = ctk.CTkButton(
            button_frame,
            text="🌙",
            width=40,
            height=40,
            command=self.toggle_theme
        )
        self.theme_button.pack(side="right", padx=(0, 10))

        # Quick load button
        quick_load_btn = ctk.CTkButton(
            button_frame,
            text="⚡ Quick Load",
            command=self.quick_load_data,
            fg_color="transparent",
            border_width=2
        )
        quick_load_btn.pack(side="right", padx=(0, 10))

    def create_data_tab(self):
        """Create modern data loading and strategy selection tab."""
        self.data_tab = self.tabview.add("📊 Data & Strategy")

        # Create scrollable frame
        data_frame = ctk.CTkScrollableFrame(self.data_tab)
        data_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Data Source Section
        data_source_frame = ctk.CTkFrame(data_frame)
        data_source_frame.pack(fill="x", pady=(0, 20))

        ctk.CTkLabel(
            data_source_frame,
            text="📈 Data Source",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(20, 10))

        # Asset type selection
        asset_frame = ctk.CTkFrame(data_source_frame, fg_color="transparent")
        asset_frame.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(asset_frame, text="Asset Type:").pack(side="left")
        self.asset_type_var = ctk.StringVar(value="stocks")
        asset_combo = ctk.CTkComboBox(
            asset_frame,
            values=["stocks", "futures", "crypto", "forex"],
            variable=self.asset_type_var,
            width=120
        )
        asset_combo.pack(side="right")

        # Ticker selection
        ticker_frame = ctk.CTkFrame(data_source_frame, fg_color="transparent")
        ticker_frame.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(ticker_frame, text="Ticker Symbol:").pack(side="left")
        self.ticker_var = ctk.StringVar(value="SPY")
        self.ticker_combo = ctk.CTkComboBox(
            ticker_frame,
            values=["SPY", "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
            variable=self.ticker_var,
            width=120
        )
        self.ticker_combo.pack(side="right")

        # Time period
        period_frame = ctk.CTkFrame(data_source_frame, fg_color="transparent")
        period_frame.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(period_frame, text="Time Period:").pack(side="left")
        self.period_var = ctk.StringVar(value="1y")
        period_combo = ctk.CTkComboBox(
            period_frame,
            values=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            variable=self.period_var,
            width=120
        )
        period_combo.pack(side="right")

        # Load data button
        load_btn = ctk.CTkButton(
            data_source_frame,
            text="📥 Load Data",
            command=self.load_data,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        load_btn.pack(pady=20)

        # Data preview section
        self.data_preview_frame = ctk.CTkFrame(data_frame)
        self.data_preview_frame.pack(fill="x", pady=(20, 0))

        ctk.CTkLabel(
            self.data_preview_frame,
            text="📋 Data Preview",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(20, 10))

        # Placeholder for data info
        self.data_info_label = ctk.CTkLabel(
            self.data_preview_frame,
            text="No data loaded yet",
            font=ctk.CTkFont(size=12)
        )
        self.data_info_label.pack(pady=10)

    def create_analysis_tab(self):
        """Create modern analysis tab with Monte Carlo simulations."""
        self.analysis_tab = self.tabview.add("🔬 Analysis")

        analysis_frame = ctk.CTkScrollableFrame(self.analysis_tab)
        analysis_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Monte Carlo Parameters
        mc_params_frame = ctk.CTkFrame(analysis_frame)
        mc_params_frame.pack(fill="x", pady=(0, 20))

        ctk.CTkLabel(
            mc_params_frame,
            text="🎲 Monte Carlo Parameters",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(20, 10))

        # Number of simulations
        sim_frame = ctk.CTkFrame(mc_params_frame, fg_color="transparent")
        sim_frame.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(sim_frame, text="Number of Simulations:").pack(side="left")
        self.simulations_var = ctk.StringVar(value="1000")
        sim_entry = ctk.CTkEntry(
            sim_frame,
            textvariable=self.simulations_var,
            width=120
        )
        sim_entry.pack(side="right")

        # Initial capital
        capital_frame = ctk.CTkFrame(mc_params_frame, fg_color="transparent")
        capital_frame.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(capital_frame, text="Initial Capital ($):").pack(side="left")
        self.capital_var = ctk.StringVar(value="10000")
        capital_entry = ctk.CTkEntry(
            capital_frame,
            textvariable=self.capital_var,
            width=120
        )
        capital_entry.pack(side="right")

        # Run analysis button
        run_btn = ctk.CTkButton(
            mc_params_frame,
            text="🚀 Run Monte Carlo Analysis",
            command=self.run_monte_carlo_analysis,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#1f538d"
        )
        run_btn.pack(pady=20)

        # Results section
        self.results_frame = ctk.CTkFrame(analysis_frame)
        self.results_frame.pack(fill="both", expand=True, pady=(20, 0))

        ctk.CTkLabel(
            self.results_frame,
            text="📊 Analysis Results",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(20, 10))

        # Placeholder for results
        self.results_placeholder = ctk.CTkLabel(
            self.results_frame,
            text="Run analysis to see results here",
            font=ctk.CTkFont(size=12)
        )
        self.results_placeholder.pack(pady=10)

    def create_visualization_tab(self):
        """Create modern visualization tab with interactive charts."""
        self.viz_tab = self.tabview.add("📈 Visualization")

        viz_frame = ctk.CTkFrame(self.viz_tab)
        viz_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Chart type selector
        chart_selector_frame = ctk.CTkFrame(viz_frame, fg_color="transparent")
        chart_selector_frame.pack(fill="x", pady=(0, 20))

        ctk.CTkLabel(
            chart_selector_frame,
            text="Chart Type:",
            font=ctk.CTkFont(size=14)
        ).pack(side="left", padx=(0, 10))

        self.chart_type_var = ctk.StringVar(value="equity_curves")
        chart_combo = ctk.CTkComboBox(
            chart_selector_frame,
            values=["equity_curves", "histogram", "scatter", "heatmap", "interactive"],
            variable=self.chart_type_var,
            command=self.update_visualization
        )
        chart_combo.pack(side="left")

        # Chart display area
        self.chart_frame = ctk.CTkFrame(viz_frame)
        self.chart_frame.pack(fill="both", expand=True)

        # Placeholder
        self.chart_placeholder = ctk.CTkLabel(
            self.chart_frame,
            text="📊 Charts will appear here after running analysis",
            font=ctk.CTkFont(size=16)
        )
        self.chart_placeholder.pack(expand=True)

    def create_portfolio_tab(self):
        """Create portfolio optimization tab."""
        self.portfolio_tab = self.tabview.add("📊 Portfolio")

        portfolio_frame = ctk.CTkScrollableFrame(self.portfolio_tab)
        portfolio_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Portfolio composition
        comp_frame = ctk.CTkFrame(portfolio_frame)
        comp_frame.pack(fill="x", pady=(0, 20))

        ctk.CTkLabel(
            comp_frame,
            text="🏗️ Portfolio Optimization",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(20, 10))

        # Optimization method
        method_frame = ctk.CTkFrame(comp_frame, fg_color="transparent")
        method_frame.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(method_frame, text="Optimization Method:").pack(side="left")
        self.opt_method_var = ctk.StringVar(value="modern_portfolio")
        opt_combo = ctk.CTkComboBox(
            method_frame,
            values=["modern_portfolio", "risk_parity", "equal_weight", "min_variance"],
            variable=self.opt_method_var,
            width=150
        )
        opt_combo.pack(side="right")

        # Optimize button
        opt_btn = ctk.CTkButton(
            comp_frame,
            text="⚡ Optimize Portfolio",
            command=self.optimize_portfolio,
            height=40
        )
        opt_btn.pack(pady=20)

    def create_settings_tab(self):
        """Create settings and configuration tab."""
        self.settings_tab = self.tabview.add("⚙️ Settings")

        settings_frame = ctk.CTkScrollableFrame(self.settings_tab)
        settings_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Appearance settings
        appearance_frame = ctk.CTkFrame(settings_frame)
        appearance_frame.pack(fill="x", pady=(0, 20))

        ctk.CTkLabel(
            appearance_frame,
            text="🎨 Appearance",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(20, 10))

        # Theme selection
        theme_frame = ctk.CTkFrame(appearance_frame, fg_color="transparent")
        theme_frame.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(theme_frame, text="Theme:").pack(side="left")
        self.theme_var = ctk.StringVar(value="dark")
        theme_combo = ctk.CTkComboBox(
            theme_frame,
            values=["dark", "light", "system"],
            variable=self.theme_var,
            command=self.change_theme
        )
        theme_combo.pack(side="right")

        # Export settings
        export_frame = ctk.CTkFrame(settings_frame)
        export_frame.pack(fill="x", pady=(20, 0))

        ctk.CTkLabel(
            export_frame,
            text="💾 Export Options",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(20, 10))

        # Export buttons
        export_btn_frame = ctk.CTkFrame(export_frame, fg_color="transparent")
        export_btn_frame.pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkButton(
            export_btn_frame,
            text="📊 Export Results",
            command=self.export_results
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            export_btn_frame,
            text="📈 Export Charts",
            command=self.export_charts
        ).pack(side="left")

    def create_status_bar(self):
        """Create modern status bar."""
        self.status_frame = ctk.CTkFrame(self.root, height=30)
        self.status_frame.pack(fill="x", side="bottom")

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="✅ Ready to analyze trading strategies",
            font=ctk.CTkFont(size=11)
        )
        self.status_label.pack(side="left", padx=20)

        # Progress bar for long operations
        self.progress_bar = ctk.CTkProgressBar(
            self.status_frame,
            width=200
        )
        self.progress_bar.pack(side="right", padx=20)
        self.progress_bar.set(0)

    def toggle_theme(self):
        """Toggle between dark and light mode."""
        if self.is_dark_mode:
            ctk.set_appearance_mode("light")
            self.theme_button.configure(text="☀️")
            self.is_dark_mode = False
        else:
            ctk.set_appearance_mode("dark")
            self.theme_button.configure(text="🌙")
            self.is_dark_mode = True

    def change_theme(self, theme):
        """Change the color theme."""
        if theme == "dark":
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("dark-blue")
        elif theme == "light":
            ctk.set_appearance_mode("light")
            ctk.set_default_color_theme("blue")
        else:  # system
            ctk.set_appearance_mode("system")
            ctk.set_default_color_theme("blue")

    def update_status(self, message, progress=None):
        """Update status bar with message and optional progress."""
        self.status_label.configure(text=message)
        if progress is not None:
            self.progress_bar.set(progress)

    # Placeholder methods for functionality
    def quick_load_data(self):
        """Quick load popular data."""
        self.update_status("Loading SPY data...")
        # Implementation would go here

    def load_data(self):
        """Load market data."""
        self.update_status("Loading data...")
        # Implementation would go here

    def run_monte_carlo_analysis(self):
        """Run Monte Carlo analysis."""
        self.update_status("Running Monte Carlo simulation...")
        # Implementation would go here

    def update_visualization(self, chart_type):
        """Update visualization based on chart type."""
        self.update_status(f"Updating {chart_type} visualization...")
        # Implementation would go here

    def optimize_portfolio(self):
        """Run portfolio optimization."""
        self.update_status("Optimizing portfolio...")
        # Implementation would go here

    def export_results(self):
        """Export analysis results."""
        self.update_status("Exporting results...")
        # Implementation would go here

    def export_charts(self):
        """Export charts."""
        self.update_status("Exporting charts...")
        # Implementation would go here

    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


if __name__ == "__main__":
    # Check if required packages are available
    try:
        import customtkinter as ctk
        print("✅ CustomTkinter available")

        # Create and run modern GUI
        app = ModernMonteCarloGUI()
        app.run()

    except ImportError:
        print("❌ CustomTkinter not available. Install with: pip install customtkinter")
        print("Falling back to basic Tkinter...")

        # Could fall back to basic tkinter here
        import tkinter as tk
        root = tk.Tk()
        root.title("Install CustomTkinter for Modern GUI")
        tk.Label(root, text="Please install CustomTkinter:\npip install customtkinter",
                font=("Arial", 14)).pack(pady=20)
        root.mainloop()
