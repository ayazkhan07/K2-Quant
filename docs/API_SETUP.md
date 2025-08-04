# API Integration Setup Guide

This guide covers the setup and configuration of external API services for the K2 Quant Stock Price Projection System.

## 🔑 API Services Configured

### Primary Stock Data Source
- **Polygon.io**: Real-time and historical stock market data
- **Purpose**: Primary data source for stock fetcher component
- **Features**: OHLCV data, company details, market status

### Backup & Additional Services  
- **Alpha Vantage**: Backup stock data source
- **FRED**: Economic data from Federal Reserve
- **OpenAI**: AI/ML model integration
- **Anthropic**: Alternative AI model access
- **Grok**: Additional AI capabilities
- **GitHub**: Repository and deployment automation

## 🛠️ Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
python setup_env.py
```

This will:
- Create `.env` file with your API keys
- Verify security settings
- Test API connectivity
- Confirm Polygon.io integration

### 3. Verify Setup
```bash
python -c "from config.api_config import api_config; print(api_config.validate_keys())"
```

## 📊 Polygon.io Integration

### Features Available
- **Historical Data**: Up to 20 years of historical stock data
- **Multiple Timeframes**: 1-minute to monthly intervals
- **Real-time Quotes**: Current market prices and status
- **Company Information**: Ticker details and metadata
- **Market Status**: Trading hours and market state

### Usage in Stock Fetcher
```python
from services.polygon_client import polygon_client

# Fetch 1 year of daily data for AAPL
data = polygon_client.get_stock_data('AAPL', 'day', limit=365)

# Get company details
details = polygon_client.get_ticker_details('AAPL')

# Search for tickers
results = polygon_client.search_tickers('Apple')
```

### Rate Limits
- **Free Tier**: 5 API calls per minute
- **Paid Tiers**: Higher limits based on subscription
- **Caching**: Client implements request caching for efficiency

## 🔒 Security Features

### Environment Variables
- All API keys stored in `.env` file
- `.env` is git-ignored for security
- No hardcoded credentials in source code

### Configuration Management
```python
from config.api_config import api_config

# Secure access to API keys
polygon_key = api_config.polygon_api_key
openai_key = api_config.openai_api_key
```

### Best Practices
- ✅ API keys never committed to version control
- ✅ Secure credential loading from environment
- ✅ Error handling for missing credentials
- ✅ Validation of API key presence

## 🎯 Stock Fetcher Integration

### Real Data Mode
The enhanced stock fetcher now fetches real market data:

1. **Symbol Validation**: Verify ticker exists before fetching
2. **Flexible Timeframes**: 1D to 20Y with appropriate intervals
3. **Rich Data**: OHLCV + VWAP + transaction counts
4. **Error Handling**: Network errors, invalid symbols, rate limits

### Data Format
```json
{
  "symbol": "AAPL",
  "timespan": "day",
  "from_date": "2023-01-01",
  "to_date": "2024-01-01",
  "results_count": 252,
  "records": [
    {
      "date": "2023-01-01",
      "time": "16:00:00",
      "open": 150.00,
      "high": 155.00,
      "low": 148.00,
      "close": 152.00,
      "volume": 50000000,
      "vwap": 151.25,
      "timestamp": 1672531200000,
      "number_of_transactions": 125000
    }
  ]
}
```

## 🚀 Advanced Features

### Automatic Fallback
If Polygon.io is unavailable:
- System can fall back to Alpha Vantage
- Graceful error handling
- User notification of data source

### Caching Strategy
- Recent requests cached locally
- Reduces API calls for repeated queries
- Configurable cache duration

### Market Hours Awareness
- Checks market status before requests
- Handles pre-market and after-hours data
- Weekend and holiday handling

## 🔧 Troubleshooting

### Common Issues

**API Key Not Found**
```bash
Error: Polygon API key not found. Please check your .env file.
```
**Solution**: Run `python setup_env.py` to configure API keys.

**Rate Limit Exceeded**
```bash
Error: API Error: Rate limit exceeded
```
**Solution**: Wait for rate limit reset or upgrade Polygon.io plan.

**Invalid Symbol**
```bash
Error: Ticker not found
```
**Solution**: Verify ticker symbol exists and is actively traded.

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test API connectivity
from services.polygon_client import polygon_client
status = polygon_client.get_market_status()
print(status)
```

## 📈 Future Enhancements

### Planned Features
- **Real-time Streaming**: WebSocket data feeds
- **Options Data**: Options chains and greeks
- **Crypto Integration**: Cryptocurrency data
- **International Markets**: Global stock exchanges
- **Technical Indicators**: Server-side calculations

### API Expansion
- **News Integration**: Financial news and sentiment
- **Economic Calendar**: Earnings, dividends, splits
- **Insider Trading**: SEC filings and insider activity
- **ESG Ratings**: Environmental and social metrics

---

## 📞 Support

For API-related issues:
1. Check `.env` file configuration
2. Verify internet connectivity
3. Confirm API key validity
4. Review Polygon.io account status
5. Check rate limits and usage

**The API integration provides enterprise-grade real market data access for professional stock analysis and projection.**