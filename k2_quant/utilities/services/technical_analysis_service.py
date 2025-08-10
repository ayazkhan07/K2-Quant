"""
Technical Analysis Service for K2 Quant

Provides all technical indicators using TA-Lib library.
Organized alphabetically with multi-pane support.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("WARNING: TA-Lib not installed. Install with: pip install TA-Lib")

from k2_quant.utilities.logger import k2_logger


@dataclass
class IndicatorConfig:
    """Configuration for a technical indicator"""
    name: str
    full_name: str
    category: str  # 'overlay', 'momentum', 'volume', 'volatility', 'trend'
    pane: str  # 'main' or 'separate'
    parameters: Dict[str, Any]
    description: str


class TechnicalAnalysisService:
    """Service for calculating technical indicators"""
    
    def __init__(self):
        self.indicators = self.initialize_indicators()
        self.calculated_cache = {}  # Cache calculated indicators
    
    def initialize_indicators(self) -> Dict[str, IndicatorConfig]:
        """Initialize all available indicators alphabetically"""
        indicators = {
            # A
            'ADX': IndicatorConfig(
                'ADX', 'Average Directional Index', 'trend', 'separate',
                {'timeperiod': 14}, 'Measures trend strength'
            ),
            'AROON': IndicatorConfig(
                'AROON', 'Aroon Indicator', 'trend', 'separate',
                {'timeperiod': 14}, 'Identifies trend changes'
            ),
            'ATR': IndicatorConfig(
                'ATR', 'Average True Range', 'volatility', 'separate',
                {'timeperiod': 14}, 'Measures volatility'
            ),
            
            # B
            'BBANDS': IndicatorConfig(
                'BBANDS', 'Bollinger Bands', 'volatility', 'main',
                {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
                'Price envelope based on standard deviation'
            ),
            'BOP': IndicatorConfig(
                'BOP', 'Balance of Power', 'momentum', 'separate',
                {}, 'Measures buying vs selling pressure'
            ),
            
            # C
            'CCI': IndicatorConfig(
                'CCI', 'Commodity Channel Index', 'momentum', 'separate',
                {'timeperiod': 14}, 'Identifies cyclical trends'
            ),
            'CMO': IndicatorConfig(
                'CMO', 'Chande Momentum Oscillator', 'momentum', 'separate',
                {'timeperiod': 14}, 'Momentum oscillator'
            ),
            
            # D
            'DEMA': IndicatorConfig(
                'DEMA', 'Double Exponential Moving Average', 'overlay', 'main',
                {'timeperiod': 30}, 'Responsive moving average'
            ),
            'DX': IndicatorConfig(
                'DX', 'Directional Movement Index', 'trend', 'separate',
                {'timeperiod': 14}, 'Identifies directional movement'
            ),
            
            # E
            'EMA': IndicatorConfig(
                'EMA', 'Exponential Moving Average', 'overlay', 'main',
                {'timeperiod': 20}, 'Weighted moving average'
            ),
            
            # H
            'HT_TRENDLINE': IndicatorConfig(
                'HT_TRENDLINE', 'Hilbert Transform Trendline', 'overlay', 'main',
                {}, 'Instantaneous trendline'
            ),
            
            # K
            'KAMA': IndicatorConfig(
                'KAMA', 'Kaufman Adaptive Moving Average', 'overlay', 'main',
                {'timeperiod': 30}, 'Adaptive moving average'
            ),
            
            # M
            'MACD': IndicatorConfig(
                'MACD', 'MACD', 'momentum', 'separate',
                {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
                'Moving Average Convergence Divergence'
            ),
            'MFI': IndicatorConfig(
                'MFI', 'Money Flow Index', 'volume', 'separate',
                {'timeperiod': 14}, 'Volume-weighted RSI'
            ),
            'MOM': IndicatorConfig(
                'MOM', 'Momentum', 'momentum', 'separate',
                {'timeperiod': 10}, 'Rate of change'
            ),
            
            # O
            'OBV': IndicatorConfig(
                'OBV', 'On Balance Volume', 'volume', 'separate',
                {}, 'Cumulative volume flow'
            ),
            
            # P
            'PPO': IndicatorConfig(
                'PPO', 'Percentage Price Oscillator', 'momentum', 'separate',
                {'fastperiod': 12, 'slowperiod': 26},
                'Percentage version of MACD'
            ),
            
            # R
            'ROC': IndicatorConfig(
                'ROC', 'Rate of Change', 'momentum', 'separate',
                {'timeperiod': 10}, 'Percentage change over time'
            ),
            'RSI': IndicatorConfig(
                'RSI', 'Relative Strength Index', 'momentum', 'separate',
                {'timeperiod': 14}, 'Overbought/oversold indicator'
            ),
            
            # S
            'SAR': IndicatorConfig(
                'SAR', 'Parabolic SAR', 'trend', 'main',
                {'acceleration': 0.02, 'maximum': 0.2},
                'Stop and reverse indicator'
            ),
            'SMA': IndicatorConfig(
                'SMA', 'Simple Moving Average', 'overlay', 'main',
                {'timeperiod': 20}, 'Simple average price'
            ),
            'STOCH': IndicatorConfig(
                'STOCH', 'Stochastic Oscillator', 'momentum', 'separate',
                {'fastk_period': 5, 'slowk_period': 3, 'slowd_period': 3},
                'Momentum indicator comparing close to range'
            ),
            'STOCHRSI': IndicatorConfig(
                'STOCHRSI', 'Stochastic RSI', 'momentum', 'separate',
                {'timeperiod': 14, 'fastk_period': 5, 'fastd_period': 3},
                'RSI of RSI'
            ),
            
            # T
            'T3': IndicatorConfig(
                'T3', 'Triple Exponential Moving Average', 'overlay', 'main',
                {'timeperiod': 5, 'vfactor': 0.7},
                'Smoother moving average'
            ),
            'TEMA': IndicatorConfig(
                'TEMA', 'Triple Exponential Moving Average', 'overlay', 'main',
                {'timeperiod': 30}, 'Very responsive moving average'
            ),
            'TRIX': IndicatorConfig(
                'TRIX', 'TRIX', 'momentum', 'separate',
                {'timeperiod': 30}, 'Rate of change of triple EMA'
            ),
            
            # U
            'ULTOSC': IndicatorConfig(
                'ULTOSC', 'Ultimate Oscillator', 'momentum', 'separate',
                {'timeperiod1': 7, 'timeperiod2': 14, 'timeperiod3': 28},
                'Multi-timeframe momentum'
            ),
            
            # V
            'VWAP': IndicatorConfig(
                'VWAP', 'Volume Weighted Average Price', 'overlay', 'main',
                {}, 'Average price weighted by volume'
            ),
            
            # W
            'WILLR': IndicatorConfig(
                'WILLR', 'Williams %R', 'momentum', 'separate',
                {'timeperiod': 14}, 'Overbought/oversold oscillator'
            ),
            'WMA': IndicatorConfig(
                'WMA', 'Weighted Moving Average', 'overlay', 'main',
                {'timeperiod': 20}, 'Linearly weighted moving average'
            ),
        }
        
        return indicators
    
    def get_all_indicators(self) -> List[str]:
        """Get list of all available indicators alphabetically"""
        return sorted(self.indicators.keys())
    
    def get_indicators_by_category(self, category: str) -> List[str]:
        """Get indicators by category"""
        return [name for name, config in self.indicators.items() 
                if config.category == category]
    
    def get_indicator_info(self, indicator_name: str) -> Optional[IndicatorConfig]:
        """Get information about an indicator"""
        return self.indicators.get(indicator_name)
    
    def calculate_indicator(self, data: pd.DataFrame, indicator_name: str,
                          custom_params: Dict[str, Any] = None) -> pd.Series:
        """Calculate a technical indicator"""
        if not TALIB_AVAILABLE:
            k2_logger.error("TA-Lib not available", "TA")
            return pd.Series()
        
        indicator_name = indicator_name.upper()
        if indicator_name not in self.indicators:
            k2_logger.error(f"Unknown indicator: {indicator_name}", "TA")
            return pd.Series()
        
        config = self.indicators[indicator_name]
        params = config.parameters.copy()
        
        # Override with custom parameters if provided
        if custom_params:
            params.update(custom_params)
        
        try:
            # Prepare data
            high = data['high'].values if 'high' in data.columns else None
            low = data['low'].values if 'low' in data.columns else None
            close = data['close'].values if 'close' in data.columns else None
            volume = data['volume'].values if 'volume' in data.columns else None
            open_price = data['open'].values if 'open' in data.columns else None
            
            # Calculate based on indicator type
            result = self._calculate_specific_indicator(
                indicator_name, open_price, high, low, close, volume, params
            )
            
            # Convert to Series
            if isinstance(result, tuple):
                # For indicators that return multiple values (like BBANDS, MACD)
                # Return the main line
                result = result[0]
            
            if result is not None:
                series = pd.Series(result, index=data.index)
                k2_logger.info(f"Calculated {indicator_name}", "TA")
                return series
            
        except Exception as e:
            k2_logger.error(f"Failed to calculate {indicator_name}: {str(e)}", "TA")
        
        return pd.Series()
    
    def _calculate_specific_indicator(self, name: str, open_price, high, low, close, volume,
                                     params: Dict) -> Any:
        """Calculate specific indicator using TA-Lib"""
        # Moving Averages
        if name == 'SMA':
            return talib.SMA(close, **params)
        elif name == 'EMA':
            return talib.EMA(close, **params)
        elif name == 'WMA':
            return talib.WMA(close, **params)
        elif name == 'DEMA':
            return talib.DEMA(close, **params)
        elif name == 'TEMA':
            return talib.TEMA(close, **params)
        elif name == 'T3':
            return talib.T3(close, **params)
        elif name == 'KAMA':
            return talib.KAMA(close, **params)
        elif name == 'HT_TRENDLINE':
            return talib.HT_TRENDLINE(close)
        
        # Momentum Indicators
        elif name == 'RSI':
            return talib.RSI(close, **params)
        elif name == 'MACD':
            macd, signal, hist = talib.MACD(close, **params)
            return hist  # Return histogram for separate pane
        elif name == 'STOCH':
            slowk, slowd = talib.STOCH(high, low, close, **params)
            return slowk
        elif name == 'STOCHRSI':
            fastk, fastd = talib.STOCHRSI(close, **params)
            return fastk
        elif name == 'MOM':
            return talib.MOM(close, **params)
        elif name == 'CMO':
            return talib.CMO(close, **params)
        elif name == 'ROC':
            return talib.ROC(close, **params)
        elif name == 'PPO':
            return talib.PPO(close, **params)
        elif name == 'WILLR':
            return talib.WILLR(high, low, close, **params)
        elif name == 'CCI':
            return talib.CCI(high, low, close, **params)
        elif name == 'ULTOSC':
            return talib.ULTOSC(high, low, close, **params)
        elif name == 'TRIX':
            return talib.TRIX(close, **params)
        elif name == 'BOP':
            return talib.BOP(open_price, high, low, close)
        
        # Volatility Indicators
        elif name == 'ATR':
            return talib.ATR(high, low, close, **params)
        elif name == 'BBANDS':
            upper, middle, lower = talib.BBANDS(close, **params)
            return middle  # Return middle band for main display
        
        # Volume Indicators
        elif name == 'OBV':
            return talib.OBV(close, volume)
        elif name == 'MFI':
            return talib.MFI(high, low, close, volume, **params)
        
        # Trend Indicators
        elif name == 'ADX':
            return talib.ADX(high, low, close, **params)
        elif name == 'AROON':
            aroon_down, aroon_up = talib.AROON(high, low, **params)
            return aroon_up
        elif name == 'DX':
            return talib.DX(high, low, close, **params)
        elif name == 'SAR':
            return talib.SAR(high, low, **params)
        
        # Custom VWAP calculation (not in TA-Lib)
        elif name == 'VWAP':
            return self._calculate_vwap(high, low, close, volume)
        
        else:
            k2_logger.warning(f"Indicator {name} not implemented", "TA")
            return None
    
    def _calculate_vwap(self, high, low, close, volume) -> np.ndarray:
        """Calculate Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        cumulative_tpv = np.cumsum(typical_price * volume)
        cumulative_volume = np.cumsum(volume)
        
        # Avoid division by zero
        vwap = np.where(cumulative_volume != 0, 
                       cumulative_tpv / cumulative_volume, 
                       typical_price)
        
        return vwap
    
    def calculate_multiple_indicators(self, data: pd.DataFrame, 
                                    indicator_list: List[str]) -> Dict[str, pd.Series]:
        """Calculate multiple indicators at once"""
        results = {}
        
        for indicator_name in indicator_list:
            result = self.calculate_indicator(data, indicator_name)
            if not result.empty:
                results[indicator_name] = result
        
        return results
    
    def get_indicator_signals(self, data: pd.DataFrame, 
                            indicator_name: str) -> pd.Series:
        """Generate buy/sell signals from an indicator"""
        indicator_data = self.calculate_indicator(data, indicator_name)
        
        if indicator_data.empty:
            return pd.Series()
        
        signals = pd.Series(index=data.index, dtype=int)
        signals[:] = 0  # Initialize with no signal
        
        # Generate signals based on indicator type
        if indicator_name == 'RSI':
            # RSI signals: Buy < 30, Sell > 70
            signals[indicator_data < 30] = 1  # Buy
            signals[indicator_data > 70] = -1  # Sell
            
        elif indicator_name == 'MACD':
            # MACD histogram crossover signals
            signals[indicator_data > 0] = 1
            signals[indicator_data < 0] = -1
            
        elif indicator_name in ['SMA', 'EMA']:
            # Price crossover signals
            close = data['close']
            signals[close > indicator_data] = 1
            signals[close < indicator_data] = -1
        
        return signals
    
    def clear_cache(self):
        """Clear calculated indicator cache"""
        self.calculated_cache.clear()
        k2_logger.info("Indicator cache cleared", "TA")


# Singleton instance
ta_service = TechnicalAnalysisService()