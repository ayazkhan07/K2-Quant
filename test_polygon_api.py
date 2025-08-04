#!/usr/bin/env python3
"""
Polygon.io API Integration Test

Test script to verify that the Polygon.io API integration is working correctly.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from services.polygon_client import polygon_client
    from config.api_config import api_config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you've run 'python setup_env.py' first")
    sys.exit(1)


def test_api_configuration():
    """Test API configuration"""
    print("🔧 Testing API Configuration...")
    
    validation = api_config.validate_keys()
    
    if validation['polygon']:
        print("✅ Polygon.io API key is configured")
    else:
        print("❌ Polygon.io API key is missing")
        return False
    
    return True


def test_market_status():
    """Test market status endpoint"""
    print("\n📊 Testing Market Status...")
    
    try:
        status = polygon_client.get_market_status()
        
        if status:
            print(f"✅ Market Status: {status.get('market', 'Unknown')}")
            exchanges = status.get('exchanges', {})
            
            for exchange, info in exchanges.items():
                print(f"  📈 {exchange.upper()}: {info}")
            
            return True
        else:
            print("❌ No market status data received")
            return False
            
    except Exception as e:
        print(f"❌ Market status test failed: {str(e)}")
        return False


def test_ticker_search():
    """Test ticker search functionality"""
    print("\n🔍 Testing Ticker Search...")
    
    try:
        results = polygon_client.search_tickers("Apple", limit=3)
        
        if results:
            print(f"✅ Found {len(results)} results for 'Apple':")
            for result in results:
                ticker = result.get('ticker', 'N/A')
                name = result.get('name', 'N/A')
                print(f"  📊 {ticker}: {name}")
            return True
        else:
            print("❌ No search results found")
            return False
            
    except Exception as e:
        print(f"❌ Ticker search test failed: {str(e)}")
        return False


def test_stock_data_fetch():
    """Test stock data fetching"""
    print("\n📈 Testing Stock Data Fetch...")
    
    try:
        # Test with AAPL for 1 week of data
        data = polygon_client.get_stock_data('AAPL', 'day', limit=5)
        
        if data and data.get('records'):
            records = data['records']
            print(f"✅ Fetched {len(records)} records for AAPL")
            
            # Show sample record
            if records:
                sample = records[-1]  # Most recent record
                print(f"  📊 Latest: {sample['date']} - Close: ${sample['close']}")
                print(f"  📊 Volume: {sample['volume']:,} - VWAP: ${sample['vwap']}")
            
            return True
        else:
            print("❌ No stock data received")
            return False
            
    except Exception as e:
        print(f"❌ Stock data fetch test failed: {str(e)}")
        return False


def test_symbol_validation():
    """Test symbol validation"""
    print("\n🔍 Testing Symbol Validation...")
    
    test_symbols = ['AAPL', 'INVALID123', 'GOOGL']
    
    for symbol in test_symbols:
        try:
            is_valid = polygon_client.validate_symbol(symbol)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"  {status}: {symbol}")
        except Exception as e:
            print(f"  ❌ Error validating {symbol}: {str(e)}")
    
    return True


def main():
    """Run all API tests"""
    print("🚀 Polygon.io API Integration Tests")
    print("=" * 50)
    
    tests = [
        ("API Configuration", test_api_configuration),
        ("Market Status", test_market_status),
        ("Ticker Search", test_ticker_search),
        ("Stock Data Fetch", test_stock_data_fetch),
        ("Symbol Validation", test_symbol_validation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except KeyboardInterrupt:
            print("\n\n⚠️ Tests interrupted by user")
            break
        except Exception as e:
            print(f"❌ {test_name} failed with error: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Polygon.io integration is ready!")
        print("\n✅ Your stock fetcher can now use real market data")
    else:
        print("⚠️ Some tests failed. Check your API configuration.")
        print("\nTroubleshooting:")
        print("  • Verify your .env file exists and contains POLYGON_API_KEY")
        print("  • Check your internet connection")
        print("  • Confirm your Polygon.io API key is valid")
        print("  • Check if you've exceeded rate limits")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)