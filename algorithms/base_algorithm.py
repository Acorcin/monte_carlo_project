"""
Base Trading Algorithm Class

This module defines the base class that all trading algorithms must inherit from.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class TradingAlgorithm(ABC):
    """
    Abstract base class for all trading algorithms.
    
    All trading algorithms must inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str, description: str = "", parameters: Dict[str, Any] = None):
        """
        Initialize the trading algorithm.

        Args:
            name (str): Name of the algorithm
            description (str): Description of the algorithm
            parameters (dict): Algorithm-specific parameters
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.trades = []  # List to store all trades
        self.equity_curve = []  # Equity curve during backtest
        self.current_position = 0  # Current position (0=flat, 1=long, -1=short)
        self.entry_price = None  # Price when position was entered
        self.entry_time = None  # Time when position was entered

        # Performance optimization cache
        self._signal_cache = {}  # Cache for signal calculations
        self._indicator_cache = {}  # Cache for technical indicators
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on market data.

        Args:
            data (pd.DataFrame): OHLCV data with DatetimeIndex

        Returns:
            pd.Series: Trading signals (1=buy, -1=sell, 0=hold)
        """
        pass

    def _generate_signals_cached(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals with caching for performance optimization.

        Args:
            data (pd.DataFrame): OHLCV data

        Returns:
            pd.Series: Trading signals
        """
        # Create cache key based on data and parameters
        data_hash = hash(data.to_string() + str(self.parameters))
        cache_key = f"{self.name}_{data_hash}"

        if cache_key in self._signal_cache:
            return self._signal_cache[cache_key]

        # Generate signals and cache them
        signals = self.generate_signals(data)
        self._signal_cache[cache_key] = signals

        return signals
    
    @abstractmethod
    def get_algorithm_type(self) -> str:
        """
        Return the type/category of the algorithm.
        
        Returns:
            str: Algorithm type (e.g., 'trend_following', 'mean_reversion')
        """
        pass
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000, 
                 commission: float = 0.001, slippage: float = 0.0001) -> Dict[str, Any]:
        """
        Run a backtest of the algorithm on the provided data.
        
        Args:
            data (pd.DataFrame): OHLCV data with DatetimeIndex
            initial_capital (float): Starting capital
            commission (float): Commission rate (0.001 = 0.1%)
            slippage (float): Slippage rate (0.0001 = 0.01%)
            
        Returns:
            dict: Backtest results including trades, returns, and metrics
        """
        # Reset algorithm state
        self.trades = []
        self.equity_curve = []
        self.current_position = 0
        self.entry_price = None
        self.entry_time = None
        
        # Generate signals with caching
        signals = self._generate_signals_cached(data)
        
        # Initialize tracking variables
        capital = initial_capital
        position_size = 0
        
        # Vectorized backtesting implementation for better performance
        signals = signals.fillna(0).astype(int)  # Ensure signals are integers
        prices = data['Close'].values
        timestamps = data.index

        # Initialize arrays for vectorized calculations
        positions = np.zeros(len(data))
        capital_history = np.zeros(len(data))
        trade_log = []

        current_position = 0
        current_capital = initial_capital
        entry_price = None
        entry_time = None

        # Process signals with optimized loop
        for i in range(len(data)):
            signal = signals.iloc[i]
            current_price = prices[i]
            timestamp = timestamps[i]

            # Process position changes
            if signal != 0 and signal != current_position:
                # Close existing position if any
                if current_position != 0:
                    trade_return = self._close_position(current_price, timestamp, commission, slippage)
                    current_capital *= (1 + trade_return)
                    returns.append(trade_return)
                    trade_log.append({
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'direction': 'long' if current_position == 1 else 'short',
                        'return': trade_return,
                        'duration': timestamp - entry_time if entry_time else pd.Timedelta(0)
                    })

                # Open new position
                if signal != 0:
                    current_position = signal
                    entry_price = current_price
                    entry_time = timestamp

            # Calculate current position value (vectorized calculation)
            if current_position != 0:
                unrealized_return = (current_price / entry_price - 1) * current_position
                current_equity = current_capital * (1 + unrealized_return)
            else:
                current_equity = current_capital

            positions[i] = current_position
            capital_history[i] = current_equity

        # Close any remaining position at the end
        if current_position != 0:
            final_price = prices[-1]
            final_timestamp = timestamps[-1]
            trade_return = self._close_position(final_price, final_timestamp, commission, slippage)
            current_capital *= (1 + trade_return)
            returns.append(trade_return)
            trade_log.append({
                'entry_time': entry_time,
                'exit_time': final_timestamp,
                'entry_price': entry_price,
                'exit_price': final_price,
                'direction': 'long' if current_position == 1 else 'short',
                'return': trade_return,
                'duration': final_timestamp - entry_time if entry_time else pd.Timedelta(0)
            })

        # Store trades in the algorithm instance
        self.trades = trade_log

        # Create equity curve data
        equity_data = [{
            'timestamp': timestamps[i],
            'equity': capital_history[i],
            'position': positions[i],
            'price': prices[i]
        } for i in range(len(data))]
        
        # Calculate performance metrics
        returns = self._calculate_returns()
        metrics = self._calculate_metrics(returns, initial_capital, capital)
        
        # Calculate drawdown curve for visualization
        drawdown_curve = self._calculate_drawdown_curve(equity_data, initial_capital)
        
        return {
            'algorithm_name': self.name,
            'algorithm_type': self.get_algorithm_type(),
            'parameters': self.parameters,
            'trades': self.trades,
            'equity_curve': pd.DataFrame(equity_data),
            'drawdown_curve': drawdown_curve,
            'returns': returns,
            'metrics': metrics,
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': (capital / initial_capital - 1) * 100
        }
    
    def _open_position(self, direction: int, price: float, timestamp: pd.Timestamp):
        """Open a new position."""
        self.current_position = direction
        self.entry_price = price
        self.entry_time = timestamp
    
    def _close_position(self, price: float, timestamp: pd.Timestamp, 
                       commission: float, slippage: float) -> float:
        """Close current position and record trade."""
        if self.current_position == 0:
            return 0.0
        
        # Calculate trade return
        price_return = (price / self.entry_price - 1) * self.current_position
        
        # Apply costs
        total_commission = commission * 2  # Entry + exit
        total_slippage = slippage * 2  # Entry + exit
        trade_return = price_return - total_commission - total_slippage
        
        # Record trade
        trade = {
            'entry_time': self.entry_time,
            'exit_time': timestamp,
            'direction': 'Long' if self.current_position > 0 else 'Short',
            'entry_price': self.entry_price,
            'exit_price': price,
            'return': trade_return,
            'duration': timestamp - self.entry_time
        }
        self.trades.append(trade)
        
        # Reset position
        self.current_position = 0
        self.entry_price = None
        self.entry_time = None
        
        return trade_return
    
    def _calculate_returns(self) -> List[float]:
        """Calculate list of trade returns."""
        return [trade['return'] for trade in self.trades]
    
    def _calculate_metrics(self, returns: List[float], initial_capital: float, 
                          final_capital: float) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not returns:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'gross_profit': 0.0,
                'gross_loss': 0.0,
                # Enhanced drawdown metrics
                'drawdown_peak_value': initial_capital,
                'drawdown_trough_value': initial_capital,
                'drawdown_duration_days': 0,
                'drawdown_start_date': None,
                'drawdown_end_date': None,
                'avg_drawdown': 0.0,
                'drawdown_periods': 0,
                'time_underwater_pct': 0.0
            }
        
        returns_array = np.array(returns)
        
        # Basic metrics
        total_trades = len(returns)
        winning_trades = len([r for r in returns if r > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_return = np.mean(returns_array)
        
        # Sharpe ratio (assuming daily returns, 252 trading days)
        if np.std(returns_array) > 0:
            sharpe_ratio = (avg_return / np.std(returns_array)) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Enhanced drawdown calculation using your formula: [(highest peak - lowest trough) / highest peak] * 100
        drawdown_metrics = self._calculate_enhanced_drawdown(returns_array, initial_capital)
        
        # Profit factor
        winning_returns = [r for r in returns if r > 0]
        losing_returns = [r for r in returns if r < 0]
        gross_profit = sum(winning_returns) if winning_returns else 0
        gross_loss = abs(sum(losing_returns)) if losing_returns else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate * 100,
            'avg_return': avg_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': drawdown_metrics['max_drawdown'],
            'profit_factor': profit_factor,
            'gross_profit': gross_profit * 100,
            'gross_loss': gross_loss * 100,
            # Enhanced drawdown metrics
            'drawdown_peak_value': drawdown_metrics['peak_value'],
            'drawdown_trough_value': drawdown_metrics['trough_value'],
            'drawdown_duration_days': drawdown_metrics['duration_days'],
            'drawdown_start_date': drawdown_metrics['start_date'],
            'drawdown_end_date': drawdown_metrics['end_date'],
            'avg_drawdown': drawdown_metrics['avg_drawdown'],
            'drawdown_periods': drawdown_metrics['drawdown_periods'],
            'time_underwater_pct': drawdown_metrics['time_underwater_pct']
        }
    
    def _calculate_enhanced_drawdown(self, returns_array: np.ndarray, initial_capital: float) -> Dict[str, Any]:
        """
        Calculate enhanced drawdown metrics using your formula:
        [(highest peak value - lowest trough value) / highest peak value] * 100
        """
        if len(returns_array) == 0:
            return {
                'max_drawdown': 0.0,
                'peak_value': initial_capital,
                'trough_value': initial_capital,
                'duration_days': 0,
                'start_date': None,
                'end_date': None,
                'avg_drawdown': 0.0,
                'drawdown_periods': 0,
                'time_underwater_pct': 0.0
            }
        
        # Calculate cumulative portfolio values
        cumulative_returns = np.cumprod(1 + returns_array)
        portfolio_values = initial_capital * cumulative_returns
        
        # Calculate running maximum (peaks)
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdowns using your formula: [(peak - current) / peak] * 100
        drawdowns = ((running_max - portfolio_values) / running_max) * 100
        
        # Find maximum drawdown
        max_drawdown_idx = np.argmax(drawdowns)
        max_drawdown = drawdowns[max_drawdown_idx]
        
        # Find the peak and trough for max drawdown
        peak_value = running_max[max_drawdown_idx]
        trough_value = portfolio_values[max_drawdown_idx]
        
        # Calculate drawdown duration and periods
        drawdown_periods = []
        current_period_start = None
        underwater_days = 0
        
        for i, dd in enumerate(drawdowns):
            if dd > 0.01:  # In drawdown (threshold of 0.01%)
                underwater_days += 1
                if current_period_start is None:
                    current_period_start = i
            else:  # Out of drawdown
                if current_period_start is not None:
                    drawdown_periods.append({
                        'start_idx': current_period_start,
                        'end_idx': i - 1,
                        'duration': i - current_period_start,
                        'max_dd': np.max(drawdowns[current_period_start:i])
                    })
                    current_period_start = None
        
        # Handle case where we end in drawdown
        if current_period_start is not None:
            drawdown_periods.append({
                'start_idx': current_period_start,
                'end_idx': len(drawdowns) - 1,
                'duration': len(drawdowns) - current_period_start,
                'max_dd': np.max(drawdowns[current_period_start:])
            })
        
        # Calculate additional metrics
        avg_drawdown = np.mean(drawdowns[drawdowns > 0]) if np.any(drawdowns > 0) else 0.0
        time_underwater_pct = (underwater_days / len(drawdowns)) * 100 if len(drawdowns) > 0 else 0.0
        
        # Find longest drawdown period
        longest_period = max(drawdown_periods, key=lambda x: x['duration']) if drawdown_periods else None
        
        return {
            'max_drawdown': max_drawdown,
            'peak_value': peak_value,
            'trough_value': trough_value,
            'duration_days': longest_period['duration'] if longest_period else 0,
            'start_date': longest_period['start_idx'] if longest_period else None,
            'end_date': longest_period['end_idx'] if longest_period else None,
            'avg_drawdown': avg_drawdown,
            'drawdown_periods': len(drawdown_periods),
            'time_underwater_pct': time_underwater_pct
        }

    def _calculate_drawdown_curve(self, equity_data: List[Dict], initial_capital: float) -> pd.DataFrame:
        """
        Calculate drawdown curve for visualization.
        Returns a DataFrame with timestamp, equity, peak, and drawdown columns.
        """
        if not equity_data:
            return pd.DataFrame(columns=['timestamp', 'equity', 'peak', 'drawdown'])
        
        # Convert equity data to DataFrame
        df = pd.DataFrame(equity_data)
        
        # Calculate running maximum (peaks)
        df['peak'] = df['equity'].cummax()
        
        # Calculate drawdown using your formula: [(peak - current) / peak] * 100
        df['drawdown'] = ((df['peak'] - df['equity']) / df['peak']) * 100
        
        return df[['timestamp', 'equity', 'peak', 'drawdown']]

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Return information about algorithm parameters.
        
        Returns:
            dict: Parameter information for UI/configuration
        """
        return {}
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Set algorithm parameters.
        
        Args:
            parameters (dict): New parameter values
        """
        self.parameters.update(parameters)
    
    def __str__(self):
        return f"{self.name} ({self.get_algorithm_type()})"
    
    def __repr__(self):
        return f"TradingAlgorithm(name='{self.name}', type='{self.get_algorithm_type()}')"
