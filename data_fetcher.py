"""
Data Fetching Module for Monte Carlo Trade Simulation

This module provides functions to fetch real market data from various sources
and convert it into trade returns for Monte Carlo analysis.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import warnings

def fetch_futures_data(
    ticker: str = "MNQ=F", 
    days_back: int = 365, 
    interval: str = "10m",
    save_csv: bool = True
) -> pd.DataFrame:
    """
    Fetch intraday futures data from Yahoo Finance.
    
    Args:
        ticker (str): Futures ticker symbol (default: "MNQ=F" for Micro E-mini Nasdaq-100)
        days_back (int): Number of days to go back from today
        interval (str): Data interval ("1m", "5m", "10m", "15m", "30m", "1h", "1d")
        save_csv (bool): Whether to save the data to CSV file
        
    Returns:
        pd.DataFrame: OHLCV data with datetime index
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"Fetching {ticker} data for the last {days_back} days...")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Interval: {interval}")
    print("-" * 50)
    
    all_data = []
    current_start = start_date
    
    # Yahoo Finance has limitations on intraday data, so we fetch in chunks
    chunk_days = 59  # Safe chunk size for intraday data
    
    while current_start < end_date:
        current_end = current_start + timedelta(days=chunk_days)
        if current_end > end_date:
            current_end = end_date
            
        print(f"Fetching chunk: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
        
        try:
            data = yf.download(ticker, start=current_start, end=current_end, interval=interval, progress=False)
            if not data.empty:
                all_data.append(data)
                print(f"  ✓ Retrieved {len(data)} records")
            else:
                print(f"  ⚠ No data retrieved for this chunk")
        except Exception as e:
            print(f"  ✗ Error occurred: {e}")
            
        current_start = current_end + timedelta(days=1)
    
    if all_data:
        # Concatenate all dataframes
        full_dataframe = pd.concat(all_data)
        full_dataframe = full_dataframe.sort_index()  # Ensure chronological order
        
        # Remove any duplicate timestamps
        full_dataframe = full_dataframe[~full_dataframe.index.duplicated(keep='first')]
        
        print(f"\n✓ Successfully fetched {len(full_dataframe)} records")
        print(f"Date range: {full_dataframe.index.min()} to {full_dataframe.index.max()}")
        
        if save_csv:
            csv_filename = f"{ticker.replace('=', '_')}_{interval}_{days_back}days.csv"
            full_dataframe.to_csv(csv_filename)
            print(f"✓ Data saved to: {csv_filename}")
            
        return full_dataframe
    else:
        raise ValueError("Could not retrieve any data. Check ticker symbol and date range.")


def calculate_returns_from_ohlcv(
    data: pd.DataFrame, 
    method: str = "close_to_close",
    remove_outliers: bool = True,
    outlier_threshold: float = 0.05
) -> np.ndarray:
    """
    Calculate returns from OHLCV data.
    
    Args:
        data (pd.DataFrame): OHLCV data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        method (str): Method for calculating returns:
            - "close_to_close": (Close[t] - Close[t-1]) / Close[t-1]
            - "open_to_close": (Close[t] - Open[t]) / Open[t]
            - "high_to_low": (High[t] - Low[t]) / Low[t] (intraday range)
        remove_outliers (bool): Whether to remove extreme outliers
        outlier_threshold (float): Threshold for outlier removal (e.g., 0.05 = 5%)
        
    Returns:
        np.ndarray: Array of returns
    """
    if method == "close_to_close":
        returns = data['Close'].pct_change().dropna()
    elif method == "open_to_close":
        returns = (data['Close'] - data['Open']) / data['Open']
        returns = returns.dropna()
    elif method == "high_to_low":
        returns = (data['High'] - data['Low']) / data['Low']
        returns = returns.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Remove outliers if requested
    if remove_outliers:
        initial_count = len(returns)
        returns = returns[abs(returns) <= outlier_threshold]
        removed_count = initial_count - len(returns)
        if removed_count > 0:
            print(f"Removed {removed_count} outliers (>{outlier_threshold:.1%}) from {initial_count} returns")
    
    print(f"Calculated {len(returns)} returns using method: {method}")
    mean_val = float(returns.mean())
    std_val = float(returns.std())
    min_val = float(returns.min())
    max_val = float(returns.max())
    
    print(f"Return statistics:")
    print(f"  Mean: {mean_val:.4f} ({mean_val*100:.2f}%)")
    print(f"  Std:  {std_val:.4f} ({std_val*100:.2f}%)")
    print(f"  Min:  {min_val:.4f} ({min_val*100:.2f}%)")
    print(f"  Max:  {max_val:.4f} ({max_val*100:.2f}%)")
    
    return returns.values


def load_data_from_csv(filename: str) -> pd.DataFrame:
    """
    Load previously saved market data from CSV file.
    
    Args:
        filename (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: OHLCV data
    """
    try:
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(data)} records from {filename}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filename}")
    except Exception as e:
        raise Exception(f"Error loading CSV file: {e}")


def fetch_stock_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch stock data (simpler than futures, good for testing).
    
    Args:
        ticker (str): Stock ticker (e.g., "AAPL", "TSLA", "SPY")
        period (str): Time period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
        interval (str): Data interval ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
        
    Returns:
        pd.DataFrame: OHLCV data
    """
    print(f"Fetching {ticker} data for period: {period}, interval: {interval}")
    
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            raise ValueError(f"No data retrieved for {ticker}")
        
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten multi-level columns by taking the first level (price type)
            data.columns = data.columns.get_level_values(0)
        
        print(f"✓ Retrieved {len(data)} records")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        return data
        
    except Exception as e:
        raise Exception(f"Error fetching stock data: {e}")


# Example usage functions
def demo_fetch_futures():
    """Demo: Fetch Micro E-mini Nasdaq-100 futures data."""
    try:
        # Use 1h interval which is supported for futures
        data = fetch_futures_data("MNQ=F", days_back=30, interval="1h")
        returns = calculate_returns_from_ohlcv(data, method="close_to_close")
        print(f"\n📊 Ready for Monte Carlo simulation with {len(returns)} trade returns")
        return returns
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None


def demo_fetch_stock():
    """Demo: Fetch SPY ETF data (easier to get)."""
    try:
        data = fetch_stock_data("SPY", period="6mo", interval="1h")
        returns = calculate_returns_from_ohlcv(data, method="close_to_close")
        print(f"\n📊 Ready for Monte Carlo simulation with {len(returns)} trade returns")
        return returns
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None


if __name__ == "__main__":
    print("=== Data Fetcher Demo ===")
    print("\n1. Trying to fetch stock data (SPY ETF)...")
    spy_returns = demo_fetch_stock()
    
    print("\n" + "="*60)
    print("\n2. Trying to fetch futures data (Micro E-mini Nasdaq-100)...")
    futures_returns = demo_fetch_futures()
    
    print("\n" + "="*60)
    print("\nDemo complete! Use the returned arrays with monte_carlo_trade_simulation.py")
