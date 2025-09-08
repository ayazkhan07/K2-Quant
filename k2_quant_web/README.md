# K2 Quant Web - Stock Price Projection System

A comprehensive web-based stock analysis and projection system with real-time data fetching, technical analysis, AI-powered insights, and multi-agent automation.

## 🚀 Features

### Core Capabilities
- **Stock Data Fetching**: High-performance parallel fetching from Polygon API
- **Technical Analysis**: 10+ indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- **AI Chat Integration**: OpenAI/Anthropic powered analysis assistant
- **Strategy Engine**: Create, backtest, and apply trading strategies
- **Real-time Updates**: WebSocket-based live data streaming
- **Multi-Agent System**: Automated development, testing, and optimization agents

### Architecture
- **Frontend**: React 18 with TypeScript, Material-UI, TradingView Charts
- **Backend**: FastAPI with async support, PostgreSQL, Redis caching
- **Real-time**: WebSocket connections for live updates
- **Background Tasks**: Celery for async processing
- **Deployment**: Docker containers with orchestration

## 📋 Prerequisites

- Docker & Docker Compose
- Node.js 18+ (for local development)
- Python 3.10+ (for local development)
- PostgreSQL 15+
- Redis 7+
- API Keys:
  - Polygon.io API key (for stock data)
  - OpenAI API key (optional, for AI chat)
  - Anthropic API key (optional, for AI chat)

## 🛠️ Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd k2_quant_web

# Create environment file
cp .env.example .env

# Edit .env and add your API keys:
# POLYGON_API_KEY=your_polygon_key
# OPENAI_API_KEY=your_openai_key (optional)
# ANTHROPIC_API_KEY=your_anthropic_key (optional)
```

### 2. Docker Deployment (Recommended)

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Celery Flower (monitoring): http://localhost:5555

### 3. Local Development Setup

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install TA-Lib (required for technical indicators)
# Ubuntu/Debian:
sudo apt-get install ta-lib
# macOS:
brew install ta-lib
# Windows: Download from https://www.ta-lib.org/

# Setup database
createdb k2quant_web
alembic upgrade head

# Run backend
uvicorn main:app --reload --port 8000
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

## 📁 Project Structure

```
k2_quant_web/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core configuration
│   │   ├── models/         # Database models
│   │   ├── schemas/        # Pydantic schemas
│   │   ├── services/       # Business logic
│   │   └── websocket/      # WebSocket handlers
│   ├── agents/             # Multi-agent system
│   ├── main.py            # FastAPI application
│   └── requirements.txt
│
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/    # Reusable components
│   │   ├── pages/         # Page components
│   │   ├── services/      # API services
│   │   ├── store/         # Redux store
│   │   └── utils/         # Utilities
│   └── package.json
│
├── docker/                 # Docker configurations
│   ├── nginx/             # Nginx configuration
│   └── postgres/          # PostgreSQL init scripts
│
└── docker-compose.yml     # Docker orchestration
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Database
DATABASE_URL=postgresql://k2quant:password@localhost/k2quant_web

# Redis
REDIS_URL=redis://localhost:6379/0

# API Keys
POLYGON_API_KEY=your_polygon_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Security
SECRET_KEY=your-secret-key-here

# CORS
CORS_ORIGINS=["http://localhost:3000"]
```

### API Endpoints

#### Stock Data
- `POST /api/v1/stocks/fetch` - Fetch stock data
- `GET /api/v1/stocks/tables` - List available tables
- `GET /api/v1/stocks/data/{table_name}` - Get table data
- `DELETE /api/v1/stocks/table/{table_name}` - Delete table

#### Analysis
- `POST /api/v1/analysis/indicators` - Calculate indicators
- `POST /api/v1/analysis/strategies` - Apply strategies
- `GET /api/v1/analysis/projections` - Get projections

#### WebSocket
- `ws://localhost:8000/ws/{client_id}` - Real-time updates

## 🎯 Usage

### 1. Fetching Stock Data

1. Navigate to Stock Fetcher page
2. Enter stock symbol (e.g., AAPL)
3. Select time range and frequency
4. Click "Fetch Data"

### 2. Analysis

1. Go to Analysis page
2. Select a saved model from left pane
3. Apply indicators and strategies
4. View charts and projections in middle pane
5. Use AI chat in right pane for insights

### 3. Creating Strategies

```python
# Example strategy code
def strategy(df):
    # Calculate indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # Generate signals
    df['signal'] = 0
    df.loc[df['sma_20'] > df['sma_50'], 'signal'] = 1
    df.loc[df['sma_20'] < df['sma_50'], 'signal'] = -1
    
    # Generate projections
    future_days = 30
    last_price = df['close'].iloc[-1]
    trend = df['signal'].iloc[-1]
    
    projections = []
    for i in range(future_days):
        projected_price = last_price * (1 + trend * 0.002)
        projections.append({
            'close': projected_price,
            'date': df.index[-1] + timedelta(days=i+1)
        })
    
    return pd.DataFrame(projections)
```

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest tests/

# Frontend tests
cd frontend
npm test

# E2E tests
npm run test:e2e
```

## 📊 Performance

- Parallel data fetching with 50 workers
- Redis caching for frequently accessed data
- WebSocket for real-time updates
- Optimized PostgreSQL queries with indexing
- React virtualization for large datasets

## 🔒 Security

- JWT-based authentication
- API rate limiting
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CORS configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Create an issue in the repository
- Check the documentation at `/docs`
- Review API documentation at `http://localhost:8000/docs`

## 🚀 Deployment

### Production Deployment

1. Update environment variables for production
2. Configure SSL certificates in Nginx
3. Set up database backups
4. Configure monitoring (Prometheus/Grafana)
5. Set up CI/CD pipeline

### Scaling

- Horizontal scaling with Kubernetes
- Database read replicas
- Redis cluster for caching
- CDN for static assets
- Load balancing with Nginx

## 📈 Roadmap

- [ ] Machine learning models for predictions
- [ ] Advanced backtesting framework
- [ ] Social sentiment analysis
- [ ] Portfolio optimization
- [ ] Mobile application
- [ ] Advanced charting features
- [ ] Integration with more data providers

---

**Note**: This is a refactored web version of the original K2 Quant desktop application, maintaining all core capabilities while adding web-specific enhancements.