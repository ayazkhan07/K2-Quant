#!/usr/bin/env python3
"""
Test script for the Stock Fetcher component

This script demonstrates the stock fetcher functionality independently.

Usage:
    python test_stock_fetcher.py
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ui.stock_fetcher import StockFetcherApplication
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure PyQt6 is installed: pip install PyQt6")
    sys.exit(1)


def main():
    """Main function to run the stock fetcher test"""
    print("Stock Price Projection System - Stock Fetcher Test")
    print("=" * 50)
    print("\nStock Fetcher Features:")
    print("- Enter any stock symbol (e.g., AAPL, GOOGL, MSFT)")
    print("- Simulated data fetching with progress indicator")
    print("- Display of stock information and historical data")
    print("- Export functionality for retrieved data")
    print("- Professional interface with clean design")
    print("\nNote: This component uses simulated data for demonstration")
    print("In production, it would connect to real market data APIs")
    print("\nStarting stock fetcher...")
    
    # Create and run the application
    app = StockFetcherApplication(sys.argv)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())