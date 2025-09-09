# algorithms/technical_indicators/liquidity_structure_strategy.py

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
import os
# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from algorithms.base_algorithm import TradingAlgorithm
    from liquidity_analyzer import run_liquidity_analyzer, StructureEvent, Zone, LiquidityPocket
except ImportError:
    # Fallback imports
    try:
        from base_algorithm import TradingAlgorithm
        from liquidity_analyzer import run_liquidity_analyzer, StructureEvent, Zone, LiquidityPocket
    except ImportError:
        print("⚠️ Liquidity analyzer imports failed, creating fallback classes")
        # Define dummy classes/functions to prevent crashes
        class StructureEvent:
            pass
        class Zone:
            pass
        class LiquidityPocket:
            pass
        def run_liquidity_analyzer(*args, **kwargs):
            return []

        # Try to import TradingAlgorithm at least
        try:
            from algorithms.base_algorithm import TradingAlgorithm
        except ImportError:
            try:
                from base_algorithm import TradingAlgorithm
            except ImportError:
                print("⚠️ TradingAlgorithm import failed")
                TradingAlgorithm = None

class LiquidityStructureStrategy(TradingAlgorithm):
    """
    Advanced Liquidity Structure Trading Strategy
    
    Uses market structure analysis, supply/demand zones, and liquidity pockets
    to generate high-probability trading signals. This strategy combines:
    
    1. Market Structure (BOS/CHOCH) - Trade with trend changes
    2. Supply/Demand Zones - Enter at institutional levels  
    3. Liquidity Pockets - Target stop hunt areas
    4. Fractality Analysis - Adapt to market regime
    """
    
    def __init__(self, 
                 swing_left: int = 3,
                 swing_right: int = 3,
                 zone_lookback: int = 25,
                 impulse_factor: float = 1.8,
                 liquidity_threshold: float = 30.0,
                 structure_weight: float = 0.4,
                 zone_weight: float = 0.4,
                 pocket_weight: float = 0.2):
        """
        Initialize Liquidity Structure Strategy.
        
        Args:
            swing_left: Lookback for swing detection
            swing_right: Lookforward for swing detection  
            zone_lookback: Periods for zone detection
            impulse_factor: Multiplier for impulse moves
            liquidity_threshold: Minimum liquidity score for signals
            structure_weight: Weight for structure signals
            zone_weight: Weight for zone signals
            pocket_weight: Weight for pocket signals
        """
        parameters = {
            'swing_left': swing_left,
            'swing_right': swing_right,
            'zone_lookback': zone_lookback,
            'impulse_factor': impulse_factor,
            'liquidity_threshold': liquidity_threshold,
            'structure_weight': structure_weight,
            'zone_weight': zone_weight,
            'pocket_weight': pocket_weight
        }
        
        super().__init__(
            name="Liquidity Structure Strategy",
            description="Advanced strategy using market structure, supply/demand zones, and liquidity analysis",
            parameters=parameters
        )
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on liquidity analysis.
        
        Args:
            data (pd.DataFrame): OHLCV data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            
        Returns:
            pd.Series: Trading signals (1=buy, -1=sell, 0=hold)
        """
        # Ensure we have the right column names for the analyzer
        if 'Open' in data.columns:
            df = data.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            })
        else:
            df = data.copy()
        
        # Run liquidity analysis
        try:
            analysis = run_liquidity_analyzer(
                df,
                swing_left=self.parameters['swing_left'],
                swing_right=self.parameters['swing_right'],
                zone_lookback=self.parameters['zone_lookback'],
                impulse_factor=self.parameters['impulse_factor']
            )
        except Exception as e:
            # Fallback to simple signals if analysis fails
            print(f"Liquidity analysis failed: {e}")
            return pd.Series(0, index=data.index)
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Get analysis components
        scored_df = analysis.work
        events = analysis.events
        zones = analysis.zones
        pockets = analysis.pockets
        hurst = analysis.hurst
        
        # Generate structure-based signals
        structure_signals = self._generate_structure_signals(scored_df, events)
        
        # Generate zone-based signals
        zone_signals = self._generate_zone_signals(df, zones)
        
        # Generate liquidity pocket signals
        pocket_signals = self._generate_pocket_signals(df, pockets)
        
        # Combine signals with weights
        combined_signals = (
            structure_signals * self.parameters['structure_weight'] +
            zone_signals * self.parameters['zone_weight'] +
            pocket_signals * self.parameters['pocket_weight']
        )
        
        # Apply liquidity threshold
        liquidity_scores = scored_df['liquidity_score'].reindex(data.index, fill_value=0)
        high_liquidity = liquidity_scores >= self.parameters['liquidity_threshold']
        
        # Generate final signals
        signals[combined_signals > 0.5] = 1   # Buy signal
        signals[combined_signals < -0.5] = -1  # Sell signal
        
        # Only trade in high liquidity areas
        signals = signals * high_liquidity.astype(int)
        
        # Adapt to market regime using Hurst exponent
        if hurst < 0.4:  # Mean-reverting market
            signals *= 0.7  # Reduce signal strength
        elif hurst > 0.6:  # Trending market
            signals *= 1.3  # Increase signal strength
        
        return signals.clip(-1, 1)
    
    def _generate_structure_signals(self, df: pd.DataFrame, events: list) -> pd.Series:
        """Generate signals based on market structure events."""
        signals = pd.Series(0.0, index=df.index)
        
        for event in events:
            if event.ref_idx < len(df):
                timestamp = df.index[event.ref_idx]
                
                # BOS signals - continuation
                if 'BOS_UP' in event.kind:
                    # Bullish continuation after break of structure
                    end_idx = min(event.ref_idx + 10, len(df))
                    signals.iloc[event.ref_idx:end_idx] += 0.8
                    
                elif 'BOS_DOWN' in event.kind:
                    # Bearish continuation after break of structure
                    end_idx = min(event.ref_idx + 10, len(df))
                    signals.iloc[event.ref_idx:end_idx] -= 0.8
                
                # CHOCH signals - reversal
                elif 'CHOCH_UP' in event.kind:
                    # Change of character to uptrend
                    end_idx = min(event.ref_idx + 15, len(df))
                    signals.iloc[event.ref_idx:end_idx] += 1.0
                    
                elif 'CHOCH_DOWN' in event.kind:
                    # Change of character to downtrend
                    end_idx = min(event.ref_idx + 15, len(df))
                    signals.iloc[event.ref_idx:end_idx] -= 1.0
        
        return signals
    
    def _generate_zone_signals(self, df: pd.DataFrame, zones: list) -> pd.Series:
        """Generate signals based on supply/demand zones."""
        signals = pd.Series(0.0, index=df.index)
        close_prices = df['close']
        
        for zone in zones:
            # Find when price is near the zone
            zone_center = (zone.price_min + zone.price_max) / 2
            zone_width = zone.price_max - zone.price_min
            
            # Price near zone
            in_zone = (
                (close_prices >= zone.price_min) & 
                (close_prices <= zone.price_max)
            )
            
            # Approaching zone
            approaching_zone = (
                (close_prices >= zone.price_min - zone_width * 0.2) & 
                (close_prices <= zone.price_max + zone_width * 0.2)
            )
            
            if zone.kind == 'DEMAND':
                # Buy at demand zones (support)
                signals[in_zone] += 1.2 * zone.strength
                signals[approaching_zone] += 0.6 * zone.strength
                
            elif zone.kind == 'SUPPLY':
                # Sell at supply zones (resistance)
                signals[in_zone] -= 1.2 * zone.strength
                signals[approaching_zone] -= 0.6 * zone.strength
        
        return signals
    
    def _generate_pocket_signals(self, df: pd.DataFrame, pockets: list) -> pd.Series:
        """Generate signals based on liquidity pockets."""
        signals = pd.Series(0.0, index=df.index)
        close_prices = df['close']
        
        for pocket in pockets:
            # Calculate distance to pocket
            distance = np.abs(close_prices - pocket.level)
            near_pocket = distance <= pocket.width
            
            if pocket.side == 'ABOVE':
                # Price near stops above - expect rejection/reversal down
                signals[near_pocket] -= 0.5 * pocket.score
                
            elif pocket.side == 'BELOW':
                # Price near stops below - expect rejection/reversal up  
                signals[near_pocket] += 0.5 * pocket.score
        
        return signals
    
    def get_algorithm_type(self) -> str:
        """Return algorithm category."""
        return "technical_indicators"
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Define parameter configuration for optimization."""
        return {
            'swing_left': {
                'type': 'int',
                'default': 3,
                'min': 2,
                'max': 10,
                'description': 'Lookback periods for swing detection'
            },
            'swing_right': {
                'type': 'int',
                'default': 3,
                'min': 2,
                'max': 10,
                'description': 'Lookforward periods for swing detection'
            },
            'zone_lookback': {
                'type': 'int',
                'default': 25,
                'min': 10,
                'max': 50,
                'description': 'Periods for zone detection rolling window'
            },
            'impulse_factor': {
                'type': 'float',
                'default': 1.8,
                'min': 1.2,
                'max': 3.0,
                'description': 'Multiplier for impulse move detection'
            },
            'liquidity_threshold': {
                'type': 'float',
                'default': 30.0,
                'min': 10.0,
                'max': 70.0,
                'description': 'Minimum liquidity score for trading'
            },
            'structure_weight': {
                'type': 'float',
                'default': 0.4,
                'min': 0.1,
                'max': 0.8,
                'description': 'Weight for market structure signals'
            },
            'zone_weight': {
                'type': 'float',
                'default': 0.4,
                'min': 0.1,
                'max': 0.8,
                'description': 'Weight for supply/demand zone signals'
            },
            'pocket_weight': {
                'type': 'float',
                'default': 0.2,
                'min': 0.0,
                'max': 0.5,
                'description': 'Weight for liquidity pocket signals'
            }
        }
    
    def get_strategy_description(self) -> str:
        """Get detailed strategy description."""
        return """
        🔥 LIQUIDITY STRUCTURE STRATEGY 🔥
        
        This advanced strategy combines multiple forms of market analysis:
        
        📊 MARKET STRUCTURE ANALYSIS:
        • BOS (Break of Structure) - Trend continuation signals
        • CHOCH (Change of Character) - Trend reversal signals
        • Swing high/low identification
        
        🎯 SUPPLY & DEMAND ZONES:
        • Institutional demand zones (support levels)
        • Institutional supply zones (resistance levels)
        • Zone strength scoring based on impulse moves
        
        💧 LIQUIDITY ANALYSIS:
        • Latent liquidity pockets (stop clusters)
        • Liquidity score-based trade filtering
        • Stop hunt targeting
        
        🌊 MARKET REGIME ADAPTATION:
        • Hurst exponent for trending vs mean-reverting markets
        • Dynamic signal strength adjustment
        
        ENTRY LOGIC:
        • Buy: CHOCH_UP + Demand Zone + High Liquidity
        • Sell: CHOCH_DOWN + Supply Zone + High Liquidity
        
        This strategy aims to trade like institutional players by:
        ✅ Following smart money structure
        ✅ Entering at institutional levels
        ✅ Targeting liquidity pools
        ✅ Adapting to market conditions
        """

