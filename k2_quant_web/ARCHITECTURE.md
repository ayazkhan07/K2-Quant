# K2 Quant Web - Architecture Overview

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Load Balancer (Nginx)                   │
└─────────────────────────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼────────┐        ┌───────▼────────┐        ┌────────▼───────┐
│   Frontend     │        │   Backend API   │        │   WebSocket    │
│   (React)      │◄───────┤   (FastAPI)     ├────────►   Server       │
└────────────────┘        └────────┬────────┘        └────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼────────┐        ┌───────▼────────┐        ┌────────▼───────┐
│   PostgreSQL   │        │     Redis      │        │   Celery       │
│   Database     │        │     Cache      │        │   Workers      │
└────────────────┘        └────────────────┘        └────────────────┘
                                                              │
                                                     ┌────────▼───────┐
                                                     │  Agent System  │
                                                     │  (Coordinator) │
                                                     └────────────────┘
```

## 🔄 Data Flow

### 1. Stock Data Fetching Flow
```
User Request → API Endpoint → Polygon Service → Parallel Fetcher
     ↓              ↓                ↓                ↓
   Response ← Database Store ← Data Processing ← API Response
```

### 2. Technical Analysis Flow
```
Select Indicator → Calculate (Backend) → Store in DB → WebSocket Update
       ↓                  ↓                  ↓              ↓
  Chart Update ← Frontend Receive ← Push to Client ← Redis Cache
```

### 3. Real-time Updates Flow
```
Market Data → WebSocket Connection → Redis Pub/Sub → Client Updates
     ↓               ↓                    ↓              ↓
  Polygon API → Data Processing → Broadcast → UI Refresh
```

## 💾 Database Schema

### Core Tables

```sql
-- Stock data table (dynamic creation)
CREATE TABLE stock_{symbol}_{timespan}_{range} (
    id SERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    date_time_market TIMESTAMP WITH TIME ZONE,
    date_market DATE,
    time_market TIME,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    vwap DECIMAL(10, 4),
    transactions INTEGER,
    -- Projection columns
    is_projection BOOLEAN DEFAULT FALSE,
    projection_source VARCHAR(100),
    -- Indicator columns (dynamic)
    sma_20 DECIMAL(10, 2),
    ema_20 DECIMAL(10, 2),
    rsi_14 DECIMAL(5, 2),
    -- Indexes
    INDEX idx_timestamp (timestamp),
    INDEX idx_date (date_market),
    INDEX idx_projection (is_projection)
);

-- Strategies table
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    code TEXT NOT NULL,
    parameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    author VARCHAR(100),
    performance_metrics JSONB
);

-- Models metadata
CREATE TABLE saved_models (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(255) UNIQUE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    timespan VARCHAR(20),
    range VARCHAR(20),
    total_records INTEGER,
    date_range JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP,
    INDEX idx_symbol (symbol),
    INDEX idx_created (created_at)
);
```

## 🔌 API Endpoints

### Stock Data APIs
- `POST /api/v1/stocks/fetch` - Fetch new stock data
- `GET /api/v1/stocks/tables` - List available tables
- `GET /api/v1/stocks/data/{table}` - Get table data
- `POST /api/v1/stocks/chart-data` - Get optimized chart data
- `DELETE /api/v1/stocks/table/{table}` - Delete table

### Analysis APIs
- `POST /api/v1/indicators/calculate` - Calculate indicator
- `GET /api/v1/indicators/list` - List available indicators
- `POST /api/v1/strategies/apply` - Apply strategy
- `GET /api/v1/strategies/list` - List strategies
- `POST /api/v1/strategies/create` - Create new strategy

### AI/Chat APIs
- `POST /api/v1/chat/message` - Send chat message
- `GET /api/v1/chat/history` - Get chat history
- `POST /api/v1/chat/generate-strategy` - Generate strategy from prompt

### Agent APIs
- `POST /api/v1/agents/task` - Submit agent task
- `GET /api/v1/agents/status` - Get agent status
- `GET /api/v1/agents/workflows` - List workflows

### WebSocket Events
- `stock_update` - Real-time stock updates
- `indicator_calculated` - Indicator results
- `strategy_applied` - Strategy execution results
- `chat_response` - AI chat responses
- `agent_status` - Agent task updates

## 🎯 Component Architecture

### Frontend Components

```
src/
├── components/           # Reusable UI components
│   ├── Layout/          # Main app layout
│   ├── Chart/           # TradingView chart wrapper
│   ├── DataGrid/        # Table component
│   └── AIChat/          # Chat interface
├── pages/               # Route-based pages
│   ├── Landing/         # Landing page
│   ├── StockFetcher/    # Data fetching UI
│   └── Analysis/        # Analysis interface
│       ├── LeftPane/    # Models, indicators, strategies
│       ├── MiddlePane/  # Chart and data
│       └── RightPane/   # AI chat
├── services/            # API service layer
│   ├── api/            # REST API calls
│   └── websocket/      # WebSocket client
├── store/              # Redux state management
│   └── slices/         # Feature slices
└── hooks/              # Custom React hooks
```

### Backend Services

```
app/
├── api/                # API endpoints
│   └── v1/
│       └── endpoints/  # Route handlers
├── core/              # Core configuration
│   ├── config.py      # Settings
│   └── database.py    # DB connection
├── models/            # SQLAlchemy models
├── schemas/           # Pydantic schemas
├── services/          # Business logic
│   ├── stock_service.py
│   ├── indicator_service.py
│   ├── strategy_service.py
│   └── ai_service.py
├── tasks/             # Celery tasks
└── websocket/         # WebSocket handlers
```

## 🚀 Performance Optimizations

### 1. Caching Strategy
- **Redis**: API responses, calculated indicators
- **Frontend**: React Query for data caching
- **CDN**: Static assets and bundles

### 2. Database Optimizations
- Indexed columns for fast queries
- Partitioned tables for large datasets
- Connection pooling
- Read replicas for scaling

### 3. Parallel Processing
- 50 concurrent workers for data fetching
- Celery for background tasks
- WebWorkers for frontend calculations

### 4. Network Optimizations
- HTTP/2 support
- Gzip compression
- WebSocket for real-time data
- Request batching

## 🔒 Security Measures

### Authentication & Authorization
- JWT tokens for API access
- Role-based access control (RBAC)
- Session management
- OAuth2 support (optional)

### Data Protection
- HTTPS encryption
- SQL injection prevention
- XSS protection
- CSRF tokens
- Input validation

### API Security
- Rate limiting
- API key management
- Request signing
- IP whitelisting (optional)

## 📊 Monitoring & Logging

### Application Monitoring
- Health check endpoints
- Performance metrics
- Error tracking
- User analytics

### Infrastructure Monitoring
- Docker container health
- Database performance
- Redis memory usage
- Celery queue lengths

### Logging
- Structured JSON logging
- Log aggregation
- Error alerting
- Audit trails

## 🔄 Deployment Pipeline

### Development
```bash
docker-compose -f docker-compose.dev.yml up
```

### Staging
```bash
docker-compose -f docker-compose.staging.yml up
```

### Production
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### CI/CD Pipeline
1. Code push to repository
2. Automated tests run
3. Docker images built
4. Deploy to staging
5. Run integration tests
6. Deploy to production
7. Health checks

## 📈 Scalability Considerations

### Horizontal Scaling
- Load balancer for multiple backend instances
- Database read replicas
- Redis cluster
- Distributed Celery workers

### Vertical Scaling
- Increase container resources
- Database connection pools
- Worker thread counts
- Cache sizes

### Auto-scaling
- Kubernetes HPA for pods
- AWS Auto Scaling Groups
- Database auto-scaling
- CDN edge locations

## 🔧 Technology Stack

### Frontend
- **React 18**: UI framework
- **TypeScript**: Type safety
- **Material-UI**: Component library
- **TradingView Charts**: Financial charts
- **Redux Toolkit**: State management
- **React Query**: Data fetching
- **Socket.io Client**: WebSocket

### Backend
- **FastAPI**: Web framework
- **PostgreSQL**: Primary database
- **Redis**: Cache & message broker
- **Celery**: Task queue
- **SQLAlchemy**: ORM
- **Pandas**: Data processing
- **TA-Lib**: Technical analysis

### Infrastructure
- **Docker**: Containerization
- **Nginx**: Reverse proxy
- **Docker Compose**: Orchestration
- **GitHub Actions**: CI/CD

### External Services
- **Polygon.io**: Stock data
- **OpenAI**: AI chat
- **Anthropic**: Claude AI

---

This architecture ensures:
- ✅ High performance
- ✅ Scalability
- ✅ Maintainability
- ✅ Security
- ✅ Real-time capabilities
- ✅ Full feature parity with desktop version