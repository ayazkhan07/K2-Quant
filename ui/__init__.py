"""
Stock Price Projection System - UI Components

This package contains the user interface components for the Stock Price Projection System.

Components:
- landing_page: Professional landing page with video background and smooth transitions
- stock_fetcher: Stock data fetching interface with real-time market data display
"""

from .landing_page import LandingPageWidget, LandingPageApplication
from .stock_fetcher import StockFetcherWidget, StockFetcherApplication

__version__ = "1.0.0"
__author__ = "Bana Architect"

__all__ = [
    "LandingPageWidget",
    "LandingPageApplication",
    "StockFetcherWidget", 
    "StockFetcherApplication"
]