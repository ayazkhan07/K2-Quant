"""Polygon.io API Client (relocated)"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

from k2_quant.utilities.config.api_config import api_config


class PolygonAPIError(Exception):
    pass


class PolygonClient:
    def __init__(self):
        self.api_key = api_config.polygon_api_key
        self.base_url = 'https://api.polygon.io'
        self.session = requests.Session()
        if not self.api_key:
            raise PolygonAPIError("Polygon API key not found. Please check your .env file.")

    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
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

    def validate_symbol(self, symbol: str) -> bool:
        try:
            symbol = symbol.upper()
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
            response = self.session.get(url, params={'apikey': self.api_key}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('status') == 'OK' and 'results' in data and len(data['results']) > 0
            elif response.status_code == 404:
                return False
            else:
                details = self.get_ticker_details(symbol)
                return bool(details)
        except requests.exceptions.Timeout:
            return True
        except Exception:
            try:
                details = self.get_ticker_details(symbol)
                return bool(details)
            except:
                return True

    def get_ticker_details(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        endpoint = f"/v3/reference/tickers/{symbol}"
        response = self._make_request(endpoint)
        return response.get('results', {})


polygon_client = PolygonClient()


