"""
Monte Carlo Trading Strategy GUI Application

A comprehensive graphical user interface that integrates:
- Advanced ML Trading Strategy
- Monte Carlo Simulations with Synthetic Returns
- Market Scenario Testing
- Real-time Visualization
- Risk Analysis
- Portfolio Optimization

Requirements: pip install tkinter matplotlib pandas numpy
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import threading
import sys
import os
import shutil
import importlib.util
import inspect
from datetime import datetime, timedelta

# Add algorithms to path
sys.path.append('algorithms')

# Import our modules
from data_fetcher import fetch_stock_data
from monte_carlo_trade_simulation import random_trade_order_simulation
from algorithms.algorithm_manager import AlgorithmManager
from market_scenario_simulation import generate_market_scenarios, test_strategy_across_scenarios
from portfolio_optimization import PortfolioOptimizer

class MonteCarloGUI:
    """Main GUI Application for Monte Carlo Trading Analysis."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("Monte Carlo Trading Strategy Analyzer")
        self.root.geometry("1400x900")
        
        # Initialize data
        self.current_data = None
        self.current_results = None
        self.algorithm_manager = AlgorithmManager()
        
        # Create main interface
        self.create_widgets()
        
        # Status
        self.update_status("Ready - Select data and algorithm to begin")
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Create main notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Data & Strategy Selection
        self.create_data_tab()
        
        # Tab 2: Monte Carlo Analysis
        self.create_monte_carlo_tab()
        
        # Tab 3: Market Scenarios
        self.create_scenarios_tab()
        
        # Tab 4: Portfolio Optimization
        self.create_portfolio_tab()
        
        # Tab 5: Results & Visualization
        self.create_results_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief='sunken')
        self.status_bar.pack(side='bottom', fill='x')
    
    def create_data_tab(self):
        """Create data and strategy selection tab."""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="📊 Data & Strategy")
        
        # Data Selection Section
        data_section = ttk.LabelFrame(self.data_frame, text="Data Selection", padding=10)
        data_section.pack(fill='x', padx=10, pady=5)
        
        # Asset type selection
        ttk.Label(data_section, text="Asset Type:").grid(row=0, column=0, sticky='w', padx=5)
        self.asset_type_var = tk.StringVar(value="stocks")
        asset_type_combo = ttk.Combobox(data_section, textvariable=self.asset_type_var,
                                       values=["stocks", "futures", "crypto", "forex"], width=8)
        asset_type_combo.grid(row=0, column=1, padx=5)
        asset_type_combo.bind('<<ComboboxSelected>>', self.on_asset_type_change)
        
        # Ticker input
        ttk.Label(data_section, text="Ticker Symbol:").grid(row=0, column=2, sticky='w', padx=5)
        self.ticker_var = tk.StringVar(value="SPY")
        self.ticker_combo = ttk.Combobox(data_section, textvariable=self.ticker_var, width=12)
        self.ticker_combo.grid(row=0, column=3, padx=5)
        
        # Initialize ticker options
        self.update_ticker_options()
        
        # Period selection
        ttk.Label(data_section, text="Period:").grid(row=0, column=4, sticky='w', padx=5)
        self.period_var = tk.StringVar(value="1y")
        period_combo = ttk.Combobox(data_section, textvariable=self.period_var, 
                                   values=["1mo", "3mo", "6mo", "1y", "2y"], width=8)
        period_combo.grid(row=0, column=5, padx=5)
        
        # Interval selection
        ttk.Label(data_section, text="Interval:").grid(row=0, column=6, sticky='w', padx=5)
        self.interval_var = tk.StringVar(value="1d")
        interval_combo = ttk.Combobox(data_section, textvariable=self.interval_var,
                                     values=["1d", "1h", "30m", "15m"], width=8)
        interval_combo.grid(row=0, column=7, padx=5)
        
        # Load data button
        load_btn = ttk.Button(data_section, text="Load Data", command=self.load_data)
        load_btn.grid(row=0, column=8, padx=10)
        
        # Data info
        self.data_info_var = tk.StringVar(value="No data loaded")
        ttk.Label(data_section, textvariable=self.data_info_var).grid(row=1, column=0, columnspan=7, pady=5)
        
        # Strategy Selection Section
        strategy_section = ttk.LabelFrame(self.data_frame, text="Trading Strategy", padding=10)
        strategy_section.pack(fill='x', padx=10, pady=5)
        
        # Algorithm selection
        ttk.Label(strategy_section, text="Algorithm:").grid(row=0, column=0, sticky='w', padx=5)
        algorithms = list(self.algorithm_manager.algorithms.keys())
        self.algorithm_var = tk.StringVar(value=algorithms[0] if algorithms else "")
        self.algorithm_combo = ttk.Combobox(strategy_section, textvariable=self.algorithm_var,
                                           values=algorithms, width=20)
        self.algorithm_combo.grid(row=0, column=1, padx=5)
        
        # Upload algorithm button
        upload_btn = ttk.Button(strategy_section, text="Upload Algorithm", command=self.upload_algorithm)
        upload_btn.grid(row=0, column=2, padx=5)
        
        # Help button for algorithm creation
        help_btn = ttk.Button(strategy_section, text="Help", command=self.show_algorithm_help)
        help_btn.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        
        # Initial capital
        ttk.Label(strategy_section, text="Initial Capital:").grid(row=0, column=3, sticky='w', padx=5)
        self.capital_var = tk.StringVar(value="10000")
        capital_entry = ttk.Entry(strategy_section, textvariable=self.capital_var, width=10)
        capital_entry.grid(row=0, column=4, padx=5)
        
        # Run backtest button
        backtest_btn = ttk.Button(strategy_section, text="Run Backtest", command=self.run_backtest)
        backtest_btn.grid(row=0, column=5, padx=10)
        
        # Results display
        self.results_text = tk.Text(self.data_frame, height=15, width=80)
        self.results_text.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(self.data_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
    
    def create_monte_carlo_tab(self):
        """Create Monte Carlo analysis tab."""
        self.mc_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.mc_frame, text="🎲 Monte Carlo")
        
        # Monte Carlo Settings
        mc_settings = ttk.LabelFrame(self.mc_frame, text="Monte Carlo Settings", padding=10)
        mc_settings.pack(fill='x', padx=10, pady=5)
        
        # Number of simulations
        ttk.Label(mc_settings, text="Simulations:").grid(row=0, column=0, sticky='w', padx=5)
        self.num_sims_var = tk.StringVar(value="1000")
        sims_entry = ttk.Entry(mc_settings, textvariable=self.num_sims_var, width=10)
        sims_entry.grid(row=0, column=1, padx=5)
        
        # Simulation method
        ttk.Label(mc_settings, text="Method:").grid(row=0, column=2, sticky='w', padx=5)
        self.sim_method_var = tk.StringVar(value="synthetic_returns")
        method_combo = ttk.Combobox(mc_settings, textvariable=self.sim_method_var,
                                   values=["synthetic_returns", "statistical", "random"], width=15)
        method_combo.grid(row=0, column=3, padx=5)
        
        # Run Monte Carlo button
        self.mc_btn = ttk.Button(mc_settings, text="Run Monte Carlo", command=self.run_monte_carlo, state='disabled')
        self.mc_btn.grid(row=0, column=4, padx=10)
        
        # Results frame with embedded matplotlib
        self.mc_results_frame = ttk.Frame(self.mc_frame)
        self.mc_results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create matplotlib figure
        self.mc_fig = Figure(figsize=(12, 8), dpi=100)
        self.mc_canvas = FigureCanvasTkAgg(self.mc_fig, self.mc_results_frame)
        self.mc_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_scenarios_tab(self):
        """Create market scenarios analysis tab."""
        self.scenarios_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.scenarios_frame, text="🌍 Scenarios")
        
        # Scenario Settings
        scenario_settings = ttk.LabelFrame(self.scenarios_frame, text="Market Scenario Settings", padding=10)
        scenario_settings.pack(fill='x', padx=10, pady=5)
        
        # Number of scenarios
        ttk.Label(scenario_settings, text="Scenarios:").grid(row=0, column=0, sticky='w', padx=5)
        self.num_scenarios_var = tk.StringVar(value="100")
        scenarios_entry = ttk.Entry(scenario_settings, textvariable=self.num_scenarios_var, width=10)
        scenarios_entry.grid(row=0, column=1, padx=5)
        
        # Scenario length
        ttk.Label(scenario_settings, text="Length (days):").grid(row=0, column=2, sticky='w', padx=5)
        self.scenario_length_var = tk.StringVar(value="126")
        length_entry = ttk.Entry(scenario_settings, textvariable=self.scenario_length_var, width=10)
        length_entry.grid(row=0, column=3, padx=5)
        
        # Run scenarios button
        scenarios_btn = ttk.Button(scenario_settings, text="Run Scenario Analysis", command=self.run_scenarios)
        scenarios_btn.grid(row=0, column=4, padx=10)
        
        # Scenarios results frame
        self.scenarios_results_frame = ttk.Frame(self.scenarios_frame)
        self.scenarios_results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create matplotlib figure for scenarios
        self.scenarios_fig = Figure(figsize=(12, 8), dpi=100)
        self.scenarios_canvas = FigureCanvasTkAgg(self.scenarios_fig, self.scenarios_results_frame)
        self.scenarios_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_portfolio_tab(self):
        """Create portfolio optimization tab."""
        self.portfolio_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.portfolio_frame, text="📈 Portfolio")
        
        # Portfolio Settings
        portfolio_settings = ttk.LabelFrame(self.portfolio_frame, text="Portfolio Optimization", padding=10)
        portfolio_settings.pack(fill='x', padx=10, pady=5)
        
        # Assets input
        ttk.Label(portfolio_settings, text="Assets (comma-separated):").grid(row=0, column=0, sticky='w', padx=5)
        self.assets_var = tk.StringVar(value="AAPL,MSFT,GOOGL,TSLA")
        assets_entry = ttk.Entry(portfolio_settings, textvariable=self.assets_var, width=30)
        assets_entry.grid(row=0, column=1, padx=5)
        
        # Optimization method
        ttk.Label(portfolio_settings, text="Method:").grid(row=0, column=2, sticky='w', padx=5)
        self.opt_method_var = tk.StringVar(value="synthetic_prices")
        opt_method_combo = ttk.Combobox(portfolio_settings, textvariable=self.opt_method_var,
                                       values=["synthetic_prices", "statistical", "random"], width=15)
        opt_method_combo.grid(row=0, column=3, padx=5)
        
        # Run optimization button
        opt_btn = ttk.Button(portfolio_settings, text="Optimize Portfolio", command=self.run_portfolio_optimization)
        opt_btn.grid(row=0, column=4, padx=10)
        
        # Portfolio results frame
        self.portfolio_results_frame = ttk.Frame(self.portfolio_frame)
        self.portfolio_results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create matplotlib figure for portfolio
        self.portfolio_fig = Figure(figsize=(12, 8), dpi=100)
        self.portfolio_canvas = FigureCanvasTkAgg(self.portfolio_fig, self.portfolio_results_frame)
        self.portfolio_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_results_tab(self):
        """Create comprehensive results and export tab."""
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="📊 Results")
        
        # Export options
        export_section = ttk.LabelFrame(self.results_frame, text="Export & Save", padding=10)
        export_section.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(export_section, text="Save Results to CSV", command=self.save_results_csv).pack(side='left', padx=5)
        ttk.Button(export_section, text="Save Charts as PNG", command=self.save_charts_png).pack(side='left', padx=5)
        ttk.Button(export_section, text="Generate Report", command=self.generate_report).pack(side='left', padx=5)
        
        # Summary display
        self.summary_text = tk.Text(self.results_frame, height=20, width=80, font=('Courier', 10))
        self.summary_text.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Scrollbar for summary
        summary_scrollbar = ttk.Scrollbar(self.results_frame, orient='vertical', command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)
    
    def update_status(self, message):
        """Update status bar message."""
        self.status_var.set(f"Status: {message}")
        self.root.update_idletasks()
    
    def get_ticker_options(self, asset_type):
        """Get ticker options based on asset type."""
        if asset_type == "stocks":
            return {
                # Major Stocks
                "AAPL": "Apple Inc.",
                "MSFT": "Microsoft Corp.",
                "GOOGL": "Alphabet Inc.",
                "TSLA": "Tesla Inc.",
                "AMZN": "Amazon.com Inc.",
                "NVDA": "NVIDIA Corp.",
                "META": "Meta Platforms Inc.",
                "NFLX": "Netflix Inc.",
                "DIS": "Walt Disney Co.",
                "PYPL": "PayPal Holdings",
                
                # ETFs
                "SPY": "SPDR S&P 500 ETF",
                "QQQ": "Invesco QQQ ETF",
                "IWM": "iShares Russell 2000 ETF",
                "VTI": "Vanguard Total Stock Market ETF",
                "VEA": "Vanguard FTSE Developed Markets ETF",
                "VWO": "Vanguard FTSE Emerging Markets ETF",
                
                # Sectors
                "XLF": "Financial Select Sector SPDR",
                "XLK": "Technology Select Sector SPDR",
                "XLE": "Energy Select Sector SPDR",
                "XLV": "Health Care Select Sector SPDR",
            }
        elif asset_type == "futures":
            return {
                # E-mini Futures
                "ES=F": "E-mini S&P 500 Futures",
                "NQ=F": "E-mini NASDAQ-100 Futures", 
                "YM=F": "E-mini Dow Jones Futures",
                "RTY=F": "E-mini Russell 2000 Futures",
                
                # Micro Futures
                "MES=F": "Micro E-mini S&P 500",
                "MNQ=F": "Micro E-mini NASDAQ-100",
                "MYM=F": "Micro E-mini Dow Jones",
                "M2K=F": "Micro E-mini Russell 2000",
                
                # Treasury Futures
                "ZN=F": "10-Year Treasury Note Futures",
                "ZB=F": "30-Year Treasury Bond Futures",
                "ZF=F": "5-Year Treasury Note Futures",
                
                # Currency Futures
                "6E=F": "Euro FX Futures",
                "6J=F": "Japanese Yen Futures",
                "6B=F": "British Pound Futures",
                
                # Commodity Futures
                "GC=F": "Gold Futures",
                "SI=F": "Silver Futures",
                "CL=F": "Crude Oil Futures",
                "NG=F": "Natural Gas Futures",
                
                # Agricultural Futures
                "ZC=F": "Corn Futures",
                "ZS=F": "Soybean Futures",
                "ZW=F": "Wheat Futures",
            }
        elif asset_type == "crypto":
            return {
                "BTC-USD": "Bitcoin USD",
                "ETH-USD": "Ethereum USD",
                "ADA-USD": "Cardano USD",
                "SOL-USD": "Solana USD",
                "MATIC-USD": "Polygon USD",
                "DOT-USD": "Polkadot USD",
                "AVAX-USD": "Avalanche USD",
                "LINK-USD": "Chainlink USD",
            }
        elif asset_type == "forex":
            return {
                "EURUSD=X": "EUR/USD",
                "GBPUSD=X": "GBP/USD",
                "USDJPY=X": "USD/JPY",
                "USDCHF=X": "USD/CHF",
                "AUDUSD=X": "AUD/USD",
                "USDCAD=X": "USD/CAD",
                "NZDUSD=X": "NZD/USD",
                "EURGBP=X": "EUR/GBP",
            }
        else:
            return {"SPY": "SPDR S&P 500 ETF"}  # Default
    
    def update_ticker_options(self):
        """Update ticker dropdown based on selected asset type."""
        asset_type = self.asset_type_var.get()
        ticker_options = self.get_ticker_options(asset_type)
        
        # Update combobox values with formatted strings
        display_values = [f"{symbol} - {name}" for symbol, name in ticker_options.items()]
        self.ticker_combo['values'] = display_values
        
        # Set default value
        first_symbol = list(ticker_options.keys())[0]
        first_display = f"{first_symbol} - {ticker_options[first_symbol]}"
        self.ticker_var.set(first_display)
    
    def on_asset_type_change(self, event=None):
        """Handle asset type selection change."""
        self.update_ticker_options()
        asset_type = self.asset_type_var.get()
        
        # Update interval options based on asset type
        if asset_type == "futures":
            # Futures often have more granular data available
            self.interval_var.set("1h")  # Good default for futures
        elif asset_type == "crypto":
            # Crypto markets are 24/7
            self.interval_var.set("1d")  # Good default for crypto
        else:
            self.interval_var.set("1d")  # Standard for stocks
        
        self.update_status(f"Changed to {asset_type} - select ticker and load data")
    
    def get_clean_ticker(self):
        """Extract clean ticker symbol from display string."""
        ticker_display = self.ticker_var.get()
        if " - " in ticker_display:
            return ticker_display.split(" - ")[0]
        return ticker_display
    
    def load_data(self):
        """Load market data."""
        try:
            self.update_status("Loading data...")
            ticker = self.get_clean_ticker().upper()
            period = self.period_var.get()
            interval = self.interval_var.get()
            asset_type = self.asset_type_var.get()
            
            # Add helpful info about what we're loading
            print(f"Loading {asset_type} data for {ticker}...")
            
            self.current_data = fetch_stock_data(ticker, period=period, interval=interval)
            
            if self.current_data is not None and not self.current_data.empty:
                # Update data info
                start_date = self.current_data.index[0].strftime("%Y-%m-%d")
                end_date = self.current_data.index[-1].strftime("%Y-%m-%d")
                self.data_info_var.set(f"Loaded {len(self.current_data)} records for {ticker} ({start_date} to {end_date})")
                
                # Show data quality info
                print(f"✅ Data loaded successfully:")
                print(f"   📊 Records: {len(self.current_data)}")
                print(f"   📅 Range: {start_date} to {end_date}")
                print(f"   💰 Latest price: ${self.current_data['Close'].iloc[-1]:.2f}")
                print(f"   📈 Columns: {list(self.current_data.columns)}")
                
                self.update_status("Data loaded successfully")
            else:
                self.data_info_var.set(f"Failed to load data for {ticker}")
                self.current_data = None
                messagebox.showerror("Data Load Failed", 
                                   f"Could not load data for {ticker}.\n\n"
                                   f"Possible reasons:\n"
                                   f"• Ticker symbol may be incorrect\n"
                                   f"• No data available for the selected period\n"
                                   f"• Network connectivity issues\n\n"
                                   f"Try:\n"
                                   f"• Different ticker symbol\n"
                                   f"• Shorter time period\n"
                                   f"• Different interval")
                self.update_status("Data load failed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.update_status("Failed to load data")
    
    def run_backtest(self):
        """Run strategy backtest."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        if not self.algorithm_var.get():
            messagebox.showwarning("Warning", "Please select an algorithm")
            return
        
        try:
            self.update_status("Running backtest...")
            
            algorithm_name = self.algorithm_var.get()
            
            # Validate initial capital
            try:
                initial_capital = float(self.capital_var.get())
                if initial_capital <= 0:
                    raise ValueError("Initial capital must be positive")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid initial capital amount")
                return
            
            # Run backtest
            self.current_results = self.algorithm_manager.backtest_algorithm(
                algorithm_name, self.current_data, initial_capital=initial_capital
            )
            
            # Display results
            if self.current_results:
                self.display_backtest_results()
                self.update_status("Backtest completed successfully")
                
                # Enable Monte Carlo button
                if hasattr(self, 'mc_btn'):
                    self.mc_btn.config(state='normal')
                    
            else:
                messagebox.showerror("Error", "Backtest failed - no results returned")
                self.update_status("Backtest failed")
                
        except Exception as e:
            print(f"Backtest error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Backtest failed: {str(e)}")
            self.update_status("Backtest failed")
    
    def display_backtest_results(self):
        """Display backtest results in text widget."""
        self.results_text.delete(1.0, tk.END)
        
        results = self.current_results
        
        output = f"""
🚀 BACKTEST RESULTS
{'='*50}

Algorithm: {results['algorithm_name']}
Initial Capital: ${results['initial_capital']:,.2f}
Final Capital: ${results['final_capital']:,.2f}
Total Return: {results['total_return']:.2%}

📊 PERFORMANCE METRICS
{'='*50}
Total Trades: {results['metrics']['total_trades']}
Win Rate: {results['metrics']['win_rate']:.1%}
Average Return: {results['metrics']['avg_return']:.2%}
Sharpe Ratio: {results['metrics']['sharpe_ratio']:.3f}
Maximum Drawdown: {results['metrics']['max_drawdown']:.2%}
Profit Factor: {results['metrics']['profit_factor']:.2f}

📈 TRADE SUMMARY
{'='*50}
"""
        
        if results['trades'] and len(results['trades']) > 0:
            # Check if trades have actual data - handle both old and new format
            valid_trades = []
            for trade in results['trades']:
                # Handle both formats: entry_date/exit_date OR entry_time/exit_time
                entry_field = trade.get('entry_date') or trade.get('entry_time')
                exit_field = trade.get('exit_date') or trade.get('exit_time')
                if entry_field and exit_field and entry_field != 'N/A':
                    valid_trades.append(trade)

            if valid_trades:
                output += f"📋 DETAILED TRADES ({len(valid_trades)} total):\n"
                output += f"{'#':<3} {'Entry Date':<12} {'Exit Date':<12} {'Direction':<6} {'Entry $':<8} {'Exit $':<8} {'Return %':<8} {'Duration':<10}\n"
                output += f"{'-'*80}\n"
                
                for i, trade in enumerate(valid_trades):
                    # Handle both date/time formats
                    entry_time = trade.get('entry_date') or trade.get('entry_time')
                    exit_time = trade.get('exit_date') or trade.get('exit_time')
                    
                    # Format dates
                    if hasattr(entry_time, 'strftime'):
                        entry_str = entry_time.strftime('%Y-%m-%d')
                    else:
                        entry_str = str(entry_time)[:10]
                        
                    if hasattr(exit_time, 'strftime'):
                        exit_str = exit_time.strftime('%Y-%m-%d')
                    else:
                        exit_str = str(exit_time)[:10]
                    
                    # Get trade details
                    direction = trade.get('direction', 'N/A')
                    entry_price = trade.get('entry_price', 0)
                    exit_price = trade.get('exit_price', 0)
                    trade_return = trade.get('return', 0)
                    
                    # Calculate duration
                    duration = trade.get('duration', 'N/A')
                    if hasattr(duration, 'days'):
                        duration_str = f"{duration.days}d"
                    else:
                        duration_str = str(duration)[:8]
                    
                    output += f"{i+1:<3} {entry_str:<12} {exit_str:<12} {direction:<6} "
                    output += f"{entry_price:<8.2f} {exit_price:<8.2f} {trade_return*100:<8.2f} {duration_str:<10}\n"
                
                # Add trade statistics
                returns = [trade.get('return', 0) for trade in valid_trades]
                winning_trades = [r for r in returns if r > 0]
                losing_trades = [r for r in returns if r < 0]
                
                output += f"{'-'*80}\n"
                output += f"📊 TRADE BREAKDOWN:\n"
                output += f"   Total Trades: {len(valid_trades)}\n"
                output += f"   Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(valid_trades)*100:.1f}%)\n"
                output += f"   Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(valid_trades)*100:.1f}%)\n"
                if winning_trades:
                    output += f"   Avg Winning Trade: {np.mean(winning_trades)*100:.2f}%\n"
                if losing_trades:
                    output += f"   Avg Losing Trade: {np.mean(losing_trades)*100:.2f}%\n"
                output += f"   Best Trade: {max(returns)*100:.2f}%\n"
                output += f"   Worst Trade: {min(returns)*100:.2f}%\n"
                
                # Add drawdown analysis
                metrics = results.get('metrics', {})
                output += f"\n{'-'*80}\n"
                output += f"📉 DRAWDOWN ANALYSIS:\n"
                output += f"🎯 FORMULA: [(Highest Peak - Lowest Trough) / Highest Peak] × 100\n"
                output += f"   Maximum Drawdown: {metrics.get('max_drawdown', 0):.2f}%\n"
                output += f"   Peak Value: ${metrics.get('drawdown_peak_value', 0):,.2f}\n"
                output += f"   Trough Value: ${metrics.get('drawdown_trough_value', 0):,.2f}\n"
                output += f"   Dollar Loss: ${metrics.get('drawdown_peak_value', 0) - metrics.get('drawdown_trough_value', 0):,.2f}\n"
                output += f"   Longest Drawdown: {metrics.get('drawdown_duration_days', 0)} periods\n"
                output += f"   Average Drawdown: {metrics.get('avg_drawdown', 0):.2f}%\n"
                output += f"   Time Underwater: {metrics.get('time_underwater_pct', 0):.1f}% of total time\n"
                
                # Risk assessment
                max_dd = metrics.get('max_drawdown', 0)
                if max_dd > 0:
                    risk_level = "Low" if max_dd < 5 else "Medium" if max_dd < 15 else "High"
                    output += f"   Risk Level: {risk_level}\n"
                    if max_dd > 20:
                        output += f"   ⚠️  WARNING: Drawdown exceeds 20%\n"
                    elif max_dd > 10:
                        output += f"   ⚠️  CAUTION: Moderate drawdown\n"
                    else:
                        output += f"   ✅ ACCEPTABLE: Drawdown within limits\n"
            else:
                output += "No valid trades generated (algorithm may not have triggered any signals)\n"
        else:
            output += "No trades data available\n"
        
        self.results_text.insert(tk.END, output)
    
    def upload_algorithm(self):
        """Upload and validate a custom algorithm file."""
        try:
            # Open file dialog
            file_path = filedialog.askopenfilename(
                title="Select Algorithm File",
                filetypes=[("Python files", "*.py"), ("All files", "*.*")],
                initialdir=os.path.expanduser("~")
            )
            
            if not file_path:
                return  # User cancelled
            
            self.update_status("Validating uploaded algorithm...")
            
            # Validate the algorithm file
            algorithm_info = self.validate_algorithm_file(file_path)
            
            if algorithm_info:
                # Show preview dialog
                if self.show_algorithm_preview(algorithm_info, file_path):
                    # Copy to algorithms directory and load
                    self.install_algorithm(file_path, algorithm_info)
                else:
                    self.update_status("Algorithm upload cancelled")
            else:
                messagebox.showerror("Invalid Algorithm", 
                                   "The selected file does not contain a valid trading algorithm.\n\n"
                                   "Please ensure your algorithm:\n"
                                   "• Inherits from TradingAlgorithm\n"
                                   "• Implements generate_signals() method\n"
                                   "• Implements get_algorithm_type() method")
                self.update_status("Invalid algorithm file")
                
        except Exception as e:
            messagebox.showerror("Upload Error", f"Failed to upload algorithm: {str(e)}")
            self.update_status("Algorithm upload failed")
    
    def validate_algorithm_file(self, file_path):
        """Validate that the file contains a valid trading algorithm."""
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("temp_algorithm", file_path)
            if spec is None:
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find classes that inherit from TradingAlgorithm
            from algorithms.base_algorithm import TradingAlgorithm
            
            algorithm_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, TradingAlgorithm) and 
                    obj != TradingAlgorithm):
                    
                    # Check if it has required methods
                    has_generate_signals = hasattr(obj, 'generate_signals')
                    has_get_algorithm_type = hasattr(obj, 'get_algorithm_type')
                    
                    if has_generate_signals and has_get_algorithm_type:
                        algorithm_classes.append({
                            'class_name': name,
                            'class_obj': obj,
                            'docstring': obj.__doc__ or "No description available",
                            'module_name': module.__name__
                        })
            
            if algorithm_classes:
                return algorithm_classes[0]  # Return the first valid class found
            else:
                return None
                
        except Exception as e:
            print(f"Validation error: {e}")
            return None
    
    def show_algorithm_preview(self, algorithm_info, file_path):
        """Show a preview dialog for the algorithm before installation."""
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Algorithm Preview")
        preview_window.geometry("600x500")
        preview_window.transient(self.root)
        preview_window.grab_set()
        
        # Center the window
        preview_window.update_idletasks()
        x = (preview_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (preview_window.winfo_screenheight() // 2) - (500 // 2)
        preview_window.geometry(f"600x500+{x}+{y}")
        
        # Title
        title_label = ttk.Label(preview_window, text="Algorithm Preview", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Info frame
        info_frame = ttk.LabelFrame(preview_window, text="Algorithm Information", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        # Algorithm details
        ttk.Label(info_frame, text=f"Class Name: {algorithm_info['class_name']}", font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Label(info_frame, text=f"File: {os.path.basename(file_path)}").pack(anchor='w')
        
        # Description
        desc_frame = ttk.LabelFrame(preview_window, text="Description", padding=10)
        desc_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        desc_text = tk.Text(desc_frame, height=8, wrap='word', font=('Arial', 10))
        desc_text.pack(fill='both', expand=True)
        desc_text.insert('1.0', algorithm_info['docstring'])
        desc_text.config(state='disabled')
        
        # Code preview
        code_frame = ttk.LabelFrame(preview_window, text="Code Preview (First 20 lines)", padding=10)
        code_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        code_text = tk.Text(code_frame, height=10, wrap='none', font=('Courier', 9))
        code_text.pack(fill='both', expand=True)
        
        # Show first 20 lines of code
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:20]
                code_text.insert('1.0', ''.join(lines))
                if len(lines) == 20:
                    code_text.insert('end', '\n... (file continues) ...')
        except:
            code_text.insert('1.0', "Could not preview code")
        
        code_text.config(state='disabled')
        
        # Buttons
        button_frame = ttk.Frame(preview_window)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        result = {'install': False}
        
        def install_clicked():
            result['install'] = True
            preview_window.destroy()
        
        def cancel_clicked():
            result['install'] = False
            preview_window.destroy()
        
        ttk.Button(button_frame, text="Install Algorithm", command=install_clicked).pack(side='right', padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel_clicked).pack(side='right')
        
        # Wait for user decision
        preview_window.wait_window()
        
        return result['install']
    
    def install_algorithm(self, file_path, algorithm_info):
        """Install the algorithm to the algorithms directory."""
        try:
            # Create custom algorithms directory if it doesn't exist
            custom_dir = os.path.join('algorithms', 'custom')
            os.makedirs(custom_dir, exist_ok=True)
            
            # Create __init__.py if it doesn't exist
            init_file = os.path.join(custom_dir, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('# Custom algorithms directory\n')
            
            # Generate a unique filename
            original_name = os.path.basename(file_path)
            name_without_ext = os.path.splitext(original_name)[0]
            
            # Clean the name for use as module name
            clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name_without_ext)
            if clean_name[0].isdigit():
                clean_name = 'algo_' + clean_name
            
            dest_path = os.path.join(custom_dir, f"{clean_name}.py")
            
            # If file already exists, add a number
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(custom_dir, f"{clean_name}_{counter}.py")
                counter += 1
            
            # Copy the file
            shutil.copy2(file_path, dest_path)
            
            # Reload the algorithm manager to pick up the new algorithm
            self.algorithm_manager = AlgorithmManager()
            
            # Update the algorithm combo box
            algorithms = list(self.algorithm_manager.algorithms.keys())
            self.algorithm_combo['values'] = algorithms
            
            # Try to select the newly uploaded algorithm
            for algo_name in algorithms:
                if clean_name.lower() in algo_name.lower() or algorithm_info['class_name'].lower() in algo_name.lower():
                    self.algorithm_var.set(algo_name)
                    break
            
            messagebox.showinfo("Success", 
                               f"Algorithm '{algorithm_info['class_name']}' installed successfully!\n\n"
                               f"Saved as: {os.path.basename(dest_path)}\n"
                               f"You can now select it from the algorithm dropdown.")
            
            self.update_status(f"Algorithm '{algorithm_info['class_name']}' installed successfully")
            
        except Exception as e:
            messagebox.showerror("Installation Error", f"Failed to install algorithm: {str(e)}")
            self.update_status("Algorithm installation failed")
    
    def show_algorithm_help(self):
        """Show help dialog for creating custom algorithms."""
        help_window = tk.Toplevel(self.root)
        help_window.title("Algorithm Creation Help")
        help_window.geometry("800x600")
        help_window.transient(self.root)
        help_window.grab_set()
        
        # Center the window
        help_window.update_idletasks()
        x = (help_window.winfo_screenwidth() // 2) - (400)
        y = (help_window.winfo_screenheight() // 2) - (300)
        help_window.geometry(f"800x600+{x}+{y}")
        
        # Title
        title_label = ttk.Label(help_window, text="How to Create Custom Trading Algorithms", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Create notebook for different help sections
        help_notebook = ttk.Notebook(help_window)
        help_notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Requirements tab
        req_frame = ttk.Frame(help_notebook)
        help_notebook.add(req_frame, text="Requirements")
        
        req_text = tk.Text(req_frame, wrap='word', font=('Arial', 10))
        req_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        requirements_content = """ALGORITHM REQUIREMENTS

To create a custom trading algorithm that works with this system, your Python file must:

✅ INHERITANCE:
• Inherit from the TradingAlgorithm base class
• Import: from algorithms.base_algorithm import TradingAlgorithm

✅ REQUIRED METHODS:
• generate_signals(self, data: pd.DataFrame) -> pd.Series
  - Input: OHLCV data (Open, High, Low, Close, Volume columns)
  - Output: Series with 1=buy, -1=sell, 0=hold signals
  
• get_algorithm_type(self) -> str
  - Return algorithm category (e.g., 'trend_following', 'mean_reversion')

✅ CONSTRUCTOR:
• Call super().__init__(name) in your __init__ method
• Example: super().__init__("My Strategy Name")

✅ IMPORTS:
• import pandas as pd
• import numpy as np (if needed)
• Any other required libraries

✅ CLASS NAMING:
• Use descriptive class names (e.g., MovingAverageCrossover)
• Avoid generic names like Strategy or Algorithm
"""
        
        req_text.insert('1.0', requirements_content)
        req_text.config(state='disabled')
        
        # Template tab
        template_frame = ttk.Frame(help_notebook)
        help_notebook.add(template_frame, text="Template")
        
        template_text = tk.Text(template_frame, wrap='none', font=('Courier', 9))
        template_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        template_content = '''"""
Sample Trading Algorithm Template
Copy this code and modify for your strategy
"""

import pandas as pd
import numpy as np
from algorithms.base_algorithm import TradingAlgorithm

class MyCustomStrategy(TradingAlgorithm):
    """
    Describe your strategy here.
    Include parameters, logic, and any special notes.
    """
    
    def __init__(self, name="My Custom Strategy"):
        super().__init__(name)
        # Initialize your parameters here
        self.parameter1 = 10
        self.parameter2 = 0.05
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.
        
        Args:
            data: OHLCV DataFrame with columns Open, High, Low, Close, Volume
        
        Returns:
            pd.Series: 1=buy, -1=sell, 0=hold
        """
        # Validate data
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            return pd.Series(0, index=data.index)
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # YOUR TRADING LOGIC HERE
        # Example: Simple moving average crossover
        short_ma = data['Close'].rolling(10).mean()
        long_ma = data['Close'].rolling(30).mean()
        
        # Buy when short MA crosses above long MA
        signals[(short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))] = 1
        
        # Sell when short MA crosses below long MA  
        signals[(short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))] = -1
        
        return signals
    
    def get_algorithm_type(self) -> str:
        """Return the algorithm type/category."""
        return 'trend_following'  # or 'mean_reversion', 'momentum', etc.
'''
        
        template_text.insert('1.0', template_content)
        template_text.config(state='disabled')
        
        # Examples tab
        examples_frame = ttk.Frame(help_notebook)
        help_notebook.add(examples_frame, text="Examples")
        
        examples_text = tk.Text(examples_frame, wrap='word', font=('Arial', 10))
        examples_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        examples_content = """ALGORITHM EXAMPLES

📈 TREND FOLLOWING:
• Moving Average Crossover
• MACD Strategy
• Breakout Strategy
• Momentum Strategy

📉 MEAN REVERSION:
• RSI Oversold/Overbought
• Bollinger Band Mean Reversion
• Statistical Arbitrage

🔄 MIXED STRATEGIES:
• Machine Learning Algorithms
• Multi-timeframe Analysis
• Regime Detection

COMMON PATTERNS:

🎯 Moving Average Crossover:
short_ma = data['Close'].rolling(10).mean()
long_ma = data['Close'].rolling(50).mean()
signals[short_ma > long_ma] = 1

📊 RSI Strategy:
rsi = calculate_rsi(data['Close'], 14)
signals[rsi < 30] = 1  # Oversold - buy
signals[rsi > 70] = -1  # Overbought - sell

💹 Bollinger Bands:
sma = data['Close'].rolling(20).mean()
std = data['Close'].rolling(20).std()
upper_band = sma + (2 * std)
lower_band = sma - (2 * std)
signals[data['Close'] < lower_band] = 1  # Buy at support
signals[data['Close'] > upper_band] = -1  # Sell at resistance

📈 Breakout Strategy:
high_20 = data['High'].rolling(20).max()
low_20 = data['Low'].rolling(20).min()
signals[data['Close'] > high_20.shift(1)] = 1  # Breakout buy
signals[data['Close'] < low_20.shift(1)] = -1  # Breakdown sell
"""
        
        examples_text.insert('1.0', examples_content)
        examples_text.config(state='disabled')
        
        # Testing tab
        testing_frame = ttk.Frame(help_notebook)
        help_notebook.add(testing_frame, text="Testing")
        
        testing_text = tk.Text(testing_frame, wrap='word', font=('Arial', 10))
        testing_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        testing_content = """TESTING YOUR ALGORITHM

🧪 BEFORE UPLOADING:

1. Test your algorithm locally:
   python your_algorithm.py

2. Verify it generates signals:
   • Check that signals are 1, -1, or 0
   • Ensure signals align with data index
   • Test with sample data

3. Check for errors:
   • Import errors
   • Missing methods
   • Data type issues

🔍 COMMON ISSUES:

❌ Import Error:
   Solution: Ensure algorithms/base_algorithm.py is accessible

❌ "Class not found":
   Solution: Check class name and inheritance

❌ "Method not implemented":
   Solution: Implement generate_signals() and get_algorithm_type()

❌ Signal format error:
   Solution: Return pd.Series with integer values (1, -1, 0)

🎯 VALIDATION CHECKLIST:

✅ File imports without errors
✅ Class inherits from TradingAlgorithm
✅ generate_signals() method exists
✅ get_algorithm_type() method exists
✅ Constructor calls super().__init__(name)
✅ Returns proper signal format

📁 SAMPLE FILES:

The system includes a sample_algorithm_template.py file that you can:
• Copy as a starting point
• Modify for your strategy
• Test before uploading

💡 TIPS:

• Start simple, then add complexity
• Test with different market conditions
• Include proper error handling
• Add comments to explain your logic
• Use meaningful variable names
"""
        
        testing_text.insert('1.0', testing_content)
        testing_text.config(state='disabled')
        
        # Buttons
        button_frame = ttk.Frame(help_window)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        def create_template():
            """Create a template file for the user."""
            try:
                template_path = filedialog.asksaveasfilename(
                    title="Save Algorithm Template",
                    defaultextension=".py",
                    filetypes=[("Python files", "*.py"), ("All files", "*.*")],
                    initialfilename="my_custom_algorithm.py"
                )
                
                if template_path:
                    # Copy the sample template
                    import shutil
                    if os.path.exists("sample_algorithm_template.py"):
                        shutil.copy2("sample_algorithm_template.py", template_path)
                        messagebox.showinfo("Success", f"Template saved to:\n{template_path}")
                    else:
                        # Create a basic template
                        with open(template_path, 'w') as f:
                            f.write(template_content)
                        messagebox.showinfo("Success", f"Template created at:\n{template_path}")
                        
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create template: {str(e)}")
        
        ttk.Button(button_frame, text="Create Template File", command=create_template).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Close", command=help_window.destroy).pack(side='right', padx=5)
    
    def run_monte_carlo(self):
        """Run Monte Carlo analysis."""
        if not self.current_results:
            messagebox.showwarning("Warning", "Please run backtest first")
            return
        
        def monte_carlo_thread():
            try:
                self.update_status("Running Monte Carlo simulation...")
                
                num_sims = int(self.num_sims_var.get())
                method = self.sim_method_var.get()
                
                # Debug: Check what data we have
                print(f"🔍 Debugging Monte Carlo data:")
                print(f"   Current results keys: {list(self.current_results.keys())}")
                
                # Get returns data - try different possible formats
                returns_data = None
                if 'returns' in self.current_results and self.current_results['returns']:
                    returns_data = [r for r in self.current_results['returns'] if r is not None and not np.isnan(r)]
                    print(f"   Using 'returns' key: {len(returns_data)} values")
                elif 'trades' in self.current_results and self.current_results['trades']:
                    # Extract returns from trades - filter out empty/invalid trades
                    valid_trades = [trade for trade in self.current_results['trades'] 
                                  if trade.get('return') is not None and not np.isnan(trade.get('return', 0))]
                    returns_data = [trade.get('return', 0) for trade in valid_trades]
                    print(f"   Extracted from 'trades': {len(returns_data)} values from {len(valid_trades)} valid trades")
                else:
                    # Generate sample returns based on performance
                    total_return = self.current_results.get('total_return', 0)
                    num_trades = self.current_results.get('metrics', {}).get('total_trades', 10)
                    
                    if num_trades > 0 and total_return != 0:
                        avg_return = total_return / max(num_trades, 1)
                        returns_data = [avg_return + np.random.normal(0, abs(avg_return) * 0.5) for _ in range(max(num_trades, 10))]
                        print(f"   Generated synthetic returns: {len(returns_data)} values based on total return {total_return:.2%}")
                    else:
                        # Last resort: create minimal sample data for demonstration
                        returns_data = [0.02, -0.01, 0.015, -0.005, 0.01]
                        print(f"   Using minimal sample returns: {len(returns_data)} values")
                
                if not returns_data or len(returns_data) < 3:
                    raise ValueError(f"Insufficient return data for Monte Carlo simulation. Need at least 3 returns, got {len(returns_data) if returns_data else 0}.")
                
                print(f"   Returns sample: {returns_data[:5]} (showing first 5)")
                
                # Run Monte Carlo
                returns_array = np.array(returns_data)
                mc_results = random_trade_order_simulation(
                    returns_array,
                    num_simulations=num_sims,
                    initial_capital=self.current_results['initial_capital'],
                    simulation_method=method
                )
                
                print(f"   ✅ Monte Carlo completed: {mc_results.shape}")
                
                # Plot results
                self.plot_monte_carlo_results(mc_results, method)
                
                self.update_status("Monte Carlo analysis completed")
                
            except Exception as e:
                print(f"   ❌ Monte Carlo error: {e}")
                import traceback
                traceback.print_exc()
                messagebox.showerror("Error", f"Monte Carlo failed: {str(e)}")
                self.update_status("Monte Carlo analysis failed")
        
        # Run in separate thread to prevent GUI freezing
        threading.Thread(target=monte_carlo_thread, daemon=True).start()
    
    def plot_monte_carlo_results(self, mc_results, method):
        """Plot Monte Carlo results in the GUI."""
        self.mc_fig.clear()
        
        # Create subplots
        axes = self.mc_fig.subplots(2, 3)
        
        # 1. Equity curves
        ax = axes[0, 0]
        for i in range(min(50, mc_results.shape[1])):
            ax.plot(mc_results.iloc[:, i], alpha=0.3, linewidth=0.5, color='blue')
        
        mean_curve = mc_results.mean(axis=1)
        ax.plot(mean_curve, color='red', linewidth=2, label='Mean Path')
        ax.set_title('Portfolio Evolution')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Final value distribution
        ax = axes[0, 1]
        final_values = mc_results.iloc[-1].values
        ax.hist(final_values, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(np.mean(final_values), color='red', linestyle='--', 
                   label=f'Mean: ${np.mean(final_values):,.0f}')
        ax.set_title('Final Value Distribution')
        ax.set_xlabel('Final Value ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Return distribution
        ax = axes[0, 2]
        # Get initial capital safely
        initial_capital = 10000  # Default
        if self.current_results and 'initial_capital' in self.current_results:
            initial_capital = self.current_results['initial_capital']
        returns = (final_values - initial_capital) / initial_capital
        ax.hist(returns, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax.axvline(np.mean(returns), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(returns):.1%}')
        ax.set_title('Return Distribution')
        ax.set_xlabel('Total Return')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Risk metrics
        ax = axes[1, 0]
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = [np.percentile(returns, p) for p in percentiles]
        ax.plot(percentiles, percentile_values, 'o-', linewidth=2, markersize=6)
        ax.set_title('Return Percentiles')
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Return')
        ax.grid(True, alpha=0.3)
        
        # 5. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        min_val = np.min(final_values)
        max_val = np.max(final_values)
        mean_val = np.mean(final_values)
        std_val = np.std(final_values)
        var_95 = np.percentile(final_values, 5)
        
        # Get initial capital safely
        initial_capital = 10000  # Default
        if self.current_results and 'initial_capital' in self.current_results:
            initial_capital = self.current_results['initial_capital']
        
        prob_loss = np.sum(final_values < initial_capital) / len(final_values)
        
        summary_text = f"""MONTE CARLO RESULTS
Method: {method}
Simulations: {len(final_values):,}

💰 Portfolio Range:
${min_val:,.0f} to ${max_val:,.0f}

📊 Statistics:
Mean: ${mean_val:,.0f}
Std Dev: ${std_val:,.0f}

🎯 Risk Metrics:
VaR (95%): ${var_95:,.0f}
Prob. Loss: {prob_loss:.1%}

📈 Range Span:
${max_val - min_val:,.0f}"""
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 6. Method comparison
        ax = axes[1, 2]
        if max_val - min_val > 1:  # Different outcomes achieved
            ax.bar(['Min', 'Mean', 'Max'], [min_val, mean_val, max_val], 
                   color=['red', 'blue', 'green'], alpha=0.7)
            ax.set_title('Outcome Range')
            ax.set_ylabel('Portfolio Value ($)')
            ax.text(0.5, 0.8, '✅ Different\nOutcomes!', transform=ax.transAxes, 
                   ha='center', fontsize=12, fontweight='bold', color='green')
        else:
            ax.text(0.5, 0.5, '⚠️ Identical\nOutcomes', transform=ax.transAxes, 
                   ha='center', fontsize=12, fontweight='bold', color='red')
        
        ax.grid(True, alpha=0.3)
        
        self.mc_fig.tight_layout()
        self.mc_canvas.draw()
    
    def run_scenarios(self):
        """Run market scenario analysis."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        def scenarios_thread():
            try:
                self.update_status("Running market scenario analysis...")
                
                num_scenarios = int(self.num_scenarios_var.get())
                scenario_length = int(self.scenario_length_var.get())
                
                # Generate scenarios
                scenarios, base_stats = generate_market_scenarios(
                    self.current_data,
                    num_scenarios=num_scenarios,
                    scenario_length=scenario_length
                )
                
                # Create strategy wrapper
                algorithm_name = self.algorithm_var.get()
                
                def strategy_wrapper(price_data):
                    try:
                        algorithm = self.algorithm_manager.create_algorithm(algorithm_name)
                        return algorithm.generate_signals(price_data) if algorithm else []
                    except:
                        return []
                
                # Test strategy across scenarios
                scenario_results = test_strategy_across_scenarios(
                    strategy_wrapper,
                    scenarios,
                    initial_capital=float(self.capital_var.get())
                )
                
                # Plot results
                self.plot_scenario_results(scenario_results)
                
                self.update_status("Market scenario analysis completed")
                
            except Exception as e:
                messagebox.showerror("Error", f"Scenario analysis failed: {str(e)}")
                self.update_status("Scenario analysis failed")
        
        threading.Thread(target=scenarios_thread, daemon=True).start()
    
    def plot_scenario_results(self, results):
        """Plot scenario analysis results."""
        self.scenarios_fig.clear()
        
        if not results:
            ax = self.scenarios_fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No scenario results to display', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            self.scenarios_canvas.draw()
            return
        
        # Create subplots
        axes = self.scenarios_fig.subplots(2, 2)
        
        # Extract data
        regimes = list(results.keys())
        mean_returns = [results[regime]['mean_return'] * 100 for regime in regimes]
        win_rates = [results[regime]['win_rate'] * 100 for regime in regimes]
        
        colors = {'bull': 'green', 'bear': 'red', 'sideways': 'blue', 'volatile': 'orange'}
        regime_colors = [colors.get(regime, 'gray') for regime in regimes]
        
        # 1. Performance by market type
        ax = axes[0, 0]
        bars = ax.bar(regimes, mean_returns, color=regime_colors, alpha=0.7, edgecolor='black')
        ax.set_title('Strategy Performance by Market Type')
        ax.set_ylabel('Average Return (%)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, mean_returns):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 2. Win rates
        ax = axes[0, 1]
        ax.bar(regimes, win_rates, color=regime_colors, alpha=0.7, edgecolor='black')
        ax.set_title('Win Rate by Market Type')
        ax.set_ylabel('Win Rate (%)')
        ax.set_ylim(0, 100)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% Breakeven')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # 3. Return ranges
        ax = axes[1, 0]
        ranges = []
        labels = []
        for regime in regimes:
            min_ret = (results[regime]['min_final_value'] / float(self.capital_var.get()) - 1) * 100
            max_ret = (results[regime]['max_final_value'] / float(self.capital_var.get()) - 1) * 100
            ranges.append([min_ret, max_ret])
            labels.append(regime)
        
        for i, (regime, (min_ret, max_ret)) in enumerate(zip(regimes, ranges)):
            color = regime_colors[i]
            ax.barh(i, max_ret - min_ret, left=min_ret, color=color, alpha=0.7)
            ax.text(min_ret + (max_ret - min_ret)/2, i, f'{max_ret - min_ret:.1f}%', 
                   ha='center', va='center', fontweight='bold')
        
        ax.set_yticks(range(len(regimes)))
        ax.set_yticklabels(regimes)
        ax.set_xlabel('Return Range (%)')
        ax.set_title('Return Range by Market Type')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        total_scenarios = sum(results[regime]['num_scenarios'] for regime in regimes)
        best_regime = max(regimes, key=lambda x: results[x]['mean_return'])
        worst_regime = min(regimes, key=lambda x: results[x]['mean_return'])
        
        summary_text = f"""SCENARIO ANALYSIS SUMMARY

Total Scenarios: {total_scenarios}

🏆 Best Market: {best_regime.upper()}
Return: {results[best_regime]['mean_return']:.1%}
Win Rate: {results[best_regime]['win_rate']:.1%}

📉 Worst Market: {worst_regime.upper()}
Return: {results[worst_regime]['mean_return']:.1%}
Win Rate: {results[worst_regime]['win_rate']:.1%}

📊 Market Performance:"""
        
        for regime in regimes:
            summary_text += f"\n{regime.capitalize()}: {results[regime]['mean_return']:.1%}"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        self.scenarios_fig.tight_layout()
        self.scenarios_canvas.draw()
    
    def run_portfolio_optimization(self):
        """Run portfolio optimization."""
        def optimization_thread():
            try:
                self.update_status("Running portfolio optimization...")
                
                assets = [asset.strip() for asset in self.assets_var.get().split(',')]
                method = self.opt_method_var.get()
                
                # Create portfolio optimizer
                optimizer = PortfolioOptimizer(assets)
                
                # Run optimization
                results = optimizer.run_full_optimization(
                    num_simulations=1000,
                    method=method,
                    plot_results=False
                )
                
                # Plot results in GUI
                self.plot_portfolio_results(results)
                
                self.update_status("Portfolio optimization completed")
                
            except Exception as e:
                messagebox.showerror("Error", f"Portfolio optimization failed: {str(e)}")
                self.update_status("Portfolio optimization failed")
        
        threading.Thread(target=optimization_thread, daemon=True).start()
    
    def plot_portfolio_results(self, results):
        """Plot portfolio optimization results."""
        self.portfolio_fig.clear()
        
        # Create efficient frontier plot
        ax = self.portfolio_fig.add_subplot(111)
        
        if 'volatility' in results and 'returns' in results:
            # Scatter plot of portfolios
            scatter = ax.scatter(results['volatility'], results['returns'], 
                               c=results['sharpe_ratios'], cmap='viridis', alpha=0.6)
            
            # Find and highlight optimal portfolios
            max_sharpe_idx = np.argmax(results['sharpe_ratios'])
            min_vol_idx = np.argmin(results['volatility'])
            
            ax.scatter(results['volatility'][max_sharpe_idx], results['returns'][max_sharpe_idx],
                      marker='*', color='red', s=500, label='Max Sharpe')
            ax.scatter(results['volatility'][min_vol_idx], results['returns'][min_vol_idx],
                      marker='*', color='blue', s=500, label='Min Volatility')
            
            ax.set_xlabel('Volatility')
            ax.set_ylabel('Expected Return')
            ax.set_title('Efficient Frontier - Portfolio Optimization')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = self.portfolio_fig.colorbar(scatter, ax=ax)
            cbar.set_label('Sharpe Ratio')
        else:
            ax.text(0.5, 0.5, 'Portfolio optimization results not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        
        self.portfolio_canvas.draw()
    
    def save_results_csv(self):
        """Save results to CSV file."""
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Create summary DataFrame
                summary_data = {
                    'Metric': ['Algorithm', 'Initial Capital', 'Final Capital', 'Total Return', 
                              'Total Trades', 'Win Rate', 'Sharpe Ratio', 'Max Drawdown'],
                    'Value': [
                        self.current_results['algorithm_name'],
                        self.current_results['initial_capital'],
                        self.current_results['final_capital'],
                        f"{self.current_results['total_return']:.2%}",
                        self.current_results['metrics']['total_trades'],
                        f"{self.current_results['metrics']['win_rate']:.1%}",
                        f"{self.current_results['metrics']['sharpe_ratio']:.3f}",
                        f"{self.current_results['metrics']['max_drawdown']:.2%}"
                    ]
                }
                
                df = pd.DataFrame(summary_data)
                df.to_csv(filename, index=False)
                
                messagebox.showinfo("Success", f"Results saved to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def save_charts_png(self):
        """Save charts as PNG files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save Monte Carlo chart
            if hasattr(self, 'mc_fig'):
                mc_filename = f"monte_carlo_analysis_{timestamp}.png"
                self.mc_fig.savefig(mc_filename, dpi=300, bbox_inches='tight')
            
            # Save Scenarios chart
            if hasattr(self, 'scenarios_fig'):
                scenarios_filename = f"scenario_analysis_{timestamp}.png"
                self.scenarios_fig.savefig(scenarios_filename, dpi=300, bbox_inches='tight')
            
            # Save Portfolio chart
            if hasattr(self, 'portfolio_fig'):
                portfolio_filename = f"portfolio_optimization_{timestamp}.png"
                self.portfolio_fig.savefig(portfolio_filename, dpi=300, bbox_inches='tight')
            
            messagebox.showinfo("Success", f"Charts saved with timestamp {timestamp}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save charts: {str(e)}")
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to generate report")
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
MONTE CARLO TRADING STRATEGY ANALYSIS REPORT
Generated: {timestamp}
{'='*80}

STRATEGY CONFIGURATION
{'='*80}
Algorithm: {self.current_results['algorithm_name']}
Ticker: {self.ticker_var.get()}
Period: {self.period_var.get()}
Interval: {self.interval_var.get()}
Initial Capital: ${self.current_results['initial_capital']:,.2f}

BACKTEST PERFORMANCE
{'='*80}
Final Capital: ${self.current_results['final_capital']:,.2f}
Total Return: {self.current_results['total_return']:.2%}
Total Trades: {self.current_results['metrics']['total_trades']}
Win Rate: {self.current_results['metrics']['win_rate']:.1%}
Average Return per Trade: {self.current_results['metrics']['avg_return']:.2%}
Sharpe Ratio: {self.current_results['metrics']['sharpe_ratio']:.3f}
Maximum Drawdown: {self.current_results['metrics']['max_drawdown']:.2%}
Profit Factor: {self.current_results['metrics']['profit_factor']:.2f}

MONTE CARLO ANALYSIS
{'='*80}
Simulation Method: {self.sim_method_var.get()}
Number of Simulations: {self.num_sims_var.get()}

CONCLUSIONS
{'='*80}
This analysis demonstrates the application of advanced Monte Carlo simulation
techniques to trading strategy evaluation. The synthetic returns method provides
realistic assessment of strategy performance across different market scenarios.

Key insights:
• The strategy shows {'positive' if self.current_results['total_return'] > 0 else 'negative'} performance over the test period
• Win rate of {self.current_results['metrics']['win_rate']:.1%} indicates {'good' if self.current_results['metrics']['win_rate'] > 50 else 'room for improvement in'} trade selection
• Sharpe ratio of {self.current_results['metrics']['sharpe_ratio']:.3f} suggests {'attractive' if self.current_results['metrics']['sharpe_ratio'] > 1 else 'modest'} risk-adjusted returns

DISCLAIMER
{'='*80}
This analysis is for educational and research purposes only. Past performance
does not guarantee future results. All trading involves risk of loss.
"""
        
        # Display in results tab
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, report)
        
        # Switch to results tab
        self.notebook.select(4)  # Results tab is index 4
        
        messagebox.showinfo("Report Generated", "Comprehensive analysis report generated and displayed in Results tab")


def main():
    """Main function to run the GUI application."""
    try:
        root = tk.Tk()
        app = MonteCarloGUI(root)
        
        # Center the window
        root.eval('tk::PlaceWindow . center')
        
        # Start the GUI event loop
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting GUI application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
