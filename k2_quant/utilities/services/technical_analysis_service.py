"""
Technical Analysis Service for K2 Quant

Provides all technical indicators using TA-Lib library.
Organized alphabetically with multi-pane support.

EXPECTATIONS:
=============
This service provides technical indicator calculations with the following objectives:

1. INDICATOR REGISTRY:
   - MUST maintain complete registry of all available indicators
   - MUST provide mapping from display names to service names
   - MUST provide default parameters for each indicator
   - MUST document expected parameter names for TA-Lib functions

2. PARAMETER HANDLING:
   - MUST accept TA-Lib standard parameter names (timeperiod, fastperiod, etc.)
   - MUST NOT accept user-friendly parameter names (period, fast, etc.)
   - MUST validate parameters before calculation
   - MUST use default parameters if custom parameters not provided

3. CALCULATION:
   - MUST return pandas Series with datetime index aligned to input data
   - MUST handle indicators that return multiple values (return primary value)
   - MUST handle missing data gracefully (NaN values)
   - MUST return empty Series on calculation failure (not None)

4. ERROR HANDLING:
   - MUST log calculation errors clearly
   - MUST return empty Series (not None) when calculation fails
   - MUST validate TA-Lib availability before attempting calculations
   - MUST handle parameter mismatches without crashing

5. INDICATOR MAPPING:
   - MUST provide map_display_name_to_service_name() method
   - MUST map "Bollinger Bands" -> "BBANDS"
   - MUST map "Stochastic" -> "STOCH"
   - MUST handle case-insensitive matching
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

    # Upper bound on the number of cached indicator results. Each entry is
    # typically one Series (or a small dict of Series) sized to the chart
    # window; at ~2M rows that's ~16 MB per entry, so keep this modest.
    _CACHE_MAX_ENTRIES = 64

    def __init__(self):
        self.indicators = self.initialize_indicators()
        self.calculated_cache = {}  # (service_name, params_key, fingerprint) -> Series | Dict[str, Series]
    
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
            'VOLUME': IndicatorConfig(
                'VOLUME', 'Volume', 'volume', 'separate',
                {}, 'Trading volume'
            ),
            'VWAP': IndicatorConfig(
                'VWAP', 'Volume Weighted Average Price', 'overlay', 'main',
                {}, 'Average price weighted by volume (daily reset)'
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
    
    def map_display_name_to_service_name(self, display_name: str) -> Optional[str]:
        """
        Map display name (e.g., 'Bollinger Bands') to TA service name (e.g., 'BBANDS').
        
        EXPECTATION: MUST handle all user-friendly display names and map them to 
        TA-Lib service names. Returns None if no mapping found.
        """
        display_name_upper = display_name.upper().strip()
        
        # First, check if it's already a service name
        if display_name_upper in self.indicators:
            return display_name_upper
        
        # Check if it matches a full_name (case-insensitive)
        for service_name, config in self.indicators.items():
            if config.full_name.upper() == display_name_upper:
                return service_name
        
        # Handle common display name variations
        name_mappings = {
            'BOLLINGER BANDS': 'BBANDS',
            'BOLLINGER': 'BBANDS',
            'STOCHASTIC': 'STOCH',
            'STOCHASTIC OSCILLATOR': 'STOCH',
        }
        
        return name_mappings.get(display_name_upper)
    
    def _make_cache_key(self, service_name: str, params: Dict[str, Any],
                        data: pd.DataFrame):
        """Build a stable, lightweight fingerprint for (indicator, params, data window).

        The fingerprint does NOT hold a reference to `data` so cached entries
        don't prevent garbage collection of old chart frames. We disambiguate
        windows of the same length via the first/last index values and the
        first/last close price.
        """
        try:
            n = len(data)
            if n == 0:
                return None
            idx0 = data.index[0]
            idx1 = data.index[-1]
            close0 = close1 = None
            if 'close' in data.columns:
                try:
                    close0 = float(data['close'].iloc[0])
                    close1 = float(data['close'].iloc[-1])
                except Exception:
                    pass
            params_key = tuple(sorted(params.items()))
            return (service_name, params_key, n, idx0, idx1, close0, close1)
        except Exception:
            return None

    def _cache_get(self, key):
        """Return a defensive copy of a cached result, or None."""
        if key is None:
            return None
        cached = self.calculated_cache.get(key)
        if cached is None:
            return None
        if isinstance(cached, dict):
            return {k: v.copy() for k, v in cached.items()}
        return cached.copy()

    def _cache_put(self, key, value):
        """Store a result with simple FIFO eviction to bound memory."""
        if key is None or value is None:
            return
        if isinstance(value, pd.Series) and value.empty:
            return
        cache = self.calculated_cache
        if key in cache:
            cache.pop(key)
        cache[key] = value
        while len(cache) > self._CACHE_MAX_ENTRIES:
            try:
                cache.pop(next(iter(cache)))
            except StopIteration:
                break

    def calculate_indicator(self, data: pd.DataFrame, indicator_name: str,
                          custom_params: Dict[str, Any] = None) -> pd.Series:
        """
        Calculate a technical indicator.
        
        EXPECTATION: MUST accept either display names or service names.
        MUST map display names to service names automatically.
        MUST return empty Series (not None) on failure.
        """
        if not TALIB_AVAILABLE:
            k2_logger.error("TA-Lib not available", "TA")
            return pd.Series()
        
        # Map display name to service name if needed
        service_name = self.map_display_name_to_service_name(indicator_name)
        if service_name is None:
            # Try uppercase as fallback
            indicator_name_upper = indicator_name.upper()
            if indicator_name_upper in self.indicators:
                service_name = indicator_name_upper
            else:
                k2_logger.error(f"Unknown indicator: {indicator_name}", "TA")
                return pd.Series()
        else:
            indicator_name = service_name
        
        if indicator_name not in self.indicators:
            k2_logger.error(f"Unknown indicator: {indicator_name}", "TA")
            return pd.Series()
        
        config = self.indicators[indicator_name]
        params = config.parameters.copy()
        
        # Override with custom parameters if provided
        if custom_params:
            params.update(custom_params)

        cache_key = self._make_cache_key(indicator_name, params, data)
        cached = self._cache_get(cache_key)
        if cached is not None:
            k2_logger.info(f"Cache hit for {indicator_name}", "TA")
            return cached
        
        try:
            # Prepare data - TA-Lib requires float64 (C double)
            def _f64(arr):
                return np.asarray(arr, dtype=np.float64) if arr is not None else None
            
            high = _f64(data['high'].values) if 'high' in data.columns else None
            low = _f64(data['low'].values) if 'low' in data.columns else None
            close = _f64(data['close'].values) if 'close' in data.columns else None
            volume = _f64(data['volume'].values) if 'volume' in data.columns else None
            open_price = _f64(data['open'].values) if 'open' in data.columns else None
            
            # Special handling for VWAP - needs date grouping for daily reset
            if indicator_name == 'VWAP':
                result = self._calculate_vwap_daily(data)
                if result is not None:
                    series = pd.Series(result, index=data.index)
                    k2_logger.info(f"Calculated {indicator_name} (daily reset)", "TA")
                    self._cache_put(cache_key, series)
                    return series.copy()
                return pd.Series()
            
            # Calculate based on indicator type
            result = self._calculate_specific_indicator(
                indicator_name, open_price, high, low, close, volume, params
            )
            
            # Convert to Series or dict of Series
            if isinstance(result, dict):
                # For indicators that return multiple lines (like BBANDS)
                # Return dict of Series
                series_dict = {}
                for key, values in result.items():
                    series_dict[key] = pd.Series(values, index=data.index)
                k2_logger.info(f"Calculated {indicator_name} (multiple lines)", "TA")
                self._cache_put(cache_key, series_dict)
                return {k: v.copy() for k, v in series_dict.items()}
            elif isinstance(result, tuple):
                # For indicators that return multiple values (like MACD)
                # Return the main line
                result = result[0]
            
            if result is not None:
                series = pd.Series(result, index=data.index)
                k2_logger.info(f"Calculated {indicator_name}", "TA")
                self._cache_put(cache_key, series)
                return series.copy()
            
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
            # Return all three bands as a dict for complete Bollinger Bands display
            return {'upper': upper, 'middle': middle, 'lower': lower}
        
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
            # VWAP needs the full dataframe for date grouping - handled in calculate_indicator
            return None  # Placeholder - actual calculation in calculate_indicator
        
        # Volume (raw)
        elif name == 'VOLUME':
            return volume
        
        else:
            k2_logger.warning(f"Indicator {name} not implemented", "TA")
            return None
    
    def _calculate_vwap_daily(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Calculate Volume Weighted Average Price with daily reset.
        
        VWAP resets at the start of each trading day, providing an intraday
        average price weighted by volume.
        """
        try:
            # Get required columns
            high = data['high'].values if 'high' in data.columns else None
            low = data['low'].values if 'low' in data.columns else None
            close = data['close'].values if 'close' in data.columns else None
            volume = data['volume'].values if 'volume' in data.columns else None
            
            if high is None or low is None or close is None or volume is None:
                k2_logger.warning("Missing required columns for VWAP", "TA")
                return None
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Get date from index (should be datetime index)
            if isinstance(data.index, pd.DatetimeIndex):
                dates = data.index.date
            elif 'timestamp' in data.columns:
                # Convert timestamp (ms) to date
                dates = pd.to_datetime(data['timestamp'], unit='ms').dt.date.values
            else:
                # Fallback: try to extract date from index
                try:
                    dates = pd.to_datetime(data.index).date
                except:
                    # No date info available - use cumulative VWAP as fallback
                    k2_logger.warning("No date info for VWAP daily reset - using cumulative", "TA")
                    cumulative_tpv = np.cumsum(typical_price * volume)
                    cumulative_volume = np.cumsum(volume)
                    return np.where(cumulative_volume != 0, 
                                   cumulative_tpv / cumulative_volume, 
                                   typical_price)
            
            # Create arrays for VWAP calculation
            vwap = np.zeros(len(data), dtype=np.float64)
            
            # Group by date and calculate VWAP per day
            unique_dates = np.unique(dates)
            
            for date in unique_dates:
                mask = dates == date
                day_tp = typical_price[mask]
                day_vol = volume[mask]
                
                # Cumulative sums within the day
                cumulative_tpv = np.cumsum(day_tp * day_vol)
                cumulative_vol = np.cumsum(day_vol)
                
                # Calculate VWAP for the day
                day_vwap = np.where(cumulative_vol != 0,
                                   cumulative_tpv / cumulative_vol,
                                   day_tp)
                
                vwap[mask] = day_vwap
            
            return vwap
            
        except Exception as e:
            k2_logger.error(f"Failed to calculate daily VWAP: {e}", "TA")
            return None
    
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