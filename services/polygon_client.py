"""
Polygon.io API Client for Stock Data Fetching

Real-time and historical stock data from Polygon.io API.
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from config.api_config import api_config


class PolygonAPIError(Exception):
    """Custom exception for Polygon API errors"""
    pass


class PolygonClient:
    """Client for Polygon.io stock data API"""
    
    def __init__(self):
        self.api_key = api_config.polygon_api_key
        self.base_url = 'https://api.polygon.io'
        self.session = requests.Session()
        
        if not self.api_key:
            raise PolygonAPIError("Polygon API key not found. Please check your .env file.")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated request to Polygon API"""
        if params is None:
            params = {}
        
        params['apikey'] = self.api_key
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'ERROR':
                raise PolygonAPIError(f"API Error: {data.get('error', 'Unknown error')}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise PolygonAPIError(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            raise PolygonAPIError(f"Invalid JSON response: {str(e)}")
    
    def get_stock_data(self, symbol: str, timespan: str = 'day', 
                      from_date: str = None, to_date: str = None,
                      limit: int = 5000) -> Dict[str, Any]:
        """
        Get aggregated stock data from Polygon.io
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            timespan: Time span (minute, hour, day, week, month, quarter, year)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Number of results to return (max 50000)
        
        Returns:
            Dict containing stock data and metadata
        """
        symbol = symbol.upper()
        
        # Set default date range if not provided
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        
        if not from_date:
            if timespan == 'day':
                from_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            elif timespan == 'week':
                from_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            elif timespan == 'month':
                from_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
            else:
                from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        endpoint = f"/v2/aggs/ticker/{symbol}/range/1/{timespan}/{from_date}/{to_date}"
        
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': min(limit, 50000)  # Polygon API max limit
        }
        
        try:
            response = self._make_request(endpoint, params)
            
            # Transform data to our standard format
            results = response.get('results', [])
            
            stock_data = {
                'symbol': symbol,
                'timespan': timespan,
                'from_date': from_date,
                'to_date': to_date,
                'results_count': response.get('resultsCount', 0),
                'request_id': response.get('request_id'),
                'records': []
            }
            
            for result in results:
                # Convert timestamp to readable date
                timestamp = result.get('t', 0)
                date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
                time = datetime.fromtimestamp(timestamp / 1000).strftime('%H:%M:%S')
                
                # Calculate VWAP
                open_price = result.get('o', 0)
                high_price = result.get('h', 0)
                low_price = result.get('l', 0)
                close_price = result.get('c', 0)
                volume = result.get('v', 0)
                vwap = result.get('vw', (open_price + high_price + low_price + close_price) / 4)
                
                record = {
                    'date': date,
                    'time': time,
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': volume,
                    'vwap': round(vwap, 2),
                    'timestamp': timestamp,
                    'number_of_transactions': result.get('n', 0)
                }
                
                stock_data['records'].append(record)
            
            return stock_data
            
        except PolygonAPIError:
            raise
        except Exception as e:
            raise PolygonAPIError(f"Unexpected error: {str(e)}")
    
    def get_ticker_details(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about a ticker"""
        symbol = symbol.upper()
        endpoint = f"/v3/reference/tickers/{symbol}"
        
        try:
            response = self._make_request(endpoint)
            return response.get('results', {})
        except PolygonAPIError:
            raise
    
    def search_tickers(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for ticker symbols"""
        endpoint = "/v3/reference/tickers"
        
        params = {
            'search': search_term,
            'active': 'true',
            'limit': limit,
            'market': 'stocks'
        }
        
        try:
            response = self._make_request(endpoint, params)
            return response.get('results', [])
        except PolygonAPIError:
            raise
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol exists using previous day's data"""
        try:
            # Quick validation using previous day's data endpoint
            symbol = symbol.upper()
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
            
            response = self.session.get(url, params={'apikey': self.api_key}, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                # Check if we got valid results
                return data.get('status') == 'OK' and 'results' in data and len(data['results']) > 0
            elif response.status_code == 404:
                # Symbol not found
                return False
            else:
                # For other status codes, try the ticker details endpoint as fallback
                details = self.get_ticker_details(symbol)
                return bool(details)
                
        except requests.exceptions.Timeout:
            # On timeout, assume valid to not block operations
            return True
        except Exception:
            # For any other error, try ticker details as fallback
            try:
                details = self.get_ticker_details(symbol)
                return bool(details)
            except:
                # If all else fails, assume valid to not block operations
                return True
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        endpoint = "/v1/marketstatus/now"
        
        try:
            response = self._make_request(endpoint)
            return response
        except PolygonAPIError:
            raise
    
    def convert_timerange_to_polygon(self, timerange: str) -> tuple:
        """Convert UI timerange to Polygon API parameters"""
        now = datetime.now()
        
        range_mapping = {
            '1D': ('minute', (now - timedelta(days=1)).strftime('%Y-%m-%d')),
            '1W': ('hour', (now - timedelta(days=7)).strftime('%Y-%m-%d')),
            '1M': ('day', (now - timedelta(days=30)).strftime('%Y-%m-%d')),
            '3M': ('day', (now - timedelta(days=90)).strftime('%Y-%m-%d')),
            '6M': ('day', (now - timedelta(days=180)).strftime('%Y-%m-%d')),
            '1Y': ('day', (now - timedelta(days=365)).strftime('%Y-%m-%d')),
            '2Y': ('week', (now - timedelta(days=730)).strftime('%Y-%m-%d')),
            '5Y': ('week', (now - timedelta(days=1825)).strftime('%Y-%m-%d')),
            '10Y': ('month', (now - timedelta(days=3650)).strftime('%Y-%m-%d')),
            '20Y': ('month', (now - timedelta(days=7300)).strftime('%Y-%m-%d'))
        }
        
        timespan, from_date = range_mapping.get(timerange, ('day', (now - timedelta(days=30)).strftime('%Y-%m-%d')))
        to_date = now.strftime('%Y-%m-%d')
        
        return timespan, from_date, to_date
    
    def get_raw_stock_data(self, symbol: str, timespan: str = 'day', 
                          from_date: str = None, to_date: str = None,
                          multiplier: int = 1, limit: int = 50000) -> List[Dict]:
        """
        Get raw stock data from Polygon.io (used by stock service)
        
        Returns raw data format for database storage
        """
        symbol = symbol.upper()
        
        # Set default date range if not provided
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': limit
        }
        
        try:
            response = self._make_request(endpoint, params)
            
            if response.get('status') == 'OK' and 'results' in response:
                return response['results']
            
            return []
            
        except PolygonAPIError:
            raise
        except Exception as e:
            raise PolygonAPIError(f"Unexpected error: {str(e)}")


# Global client instance
polygon_client = PolygonClient()