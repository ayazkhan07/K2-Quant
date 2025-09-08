# Migration Guide: Desktop to Web Version

This guide helps you migrate from the PyQt6 desktop application to the new web-based version.

## 📊 Feature Mapping

| Desktop Feature | Web Implementation | Status |
|----------------|-------------------|---------|
| **Stock Fetcher Page** | `/stock-fetcher` route | ✅ Complete |
| PyQt6 UI | React + Material-UI | ✅ Complete |
| Polygon API Integration | FastAPI backend service | ✅ Complete |
| Market Hours Filter | API parameter | ✅ Complete |
| CSV Export | API endpoint + download | ✅ Complete |
| **Analysis Page** | `/analysis` route | ✅ Complete |
| 3-Pane Layout | Resizable flex layout | ✅ Complete |
| PyQtGraph Charts | TradingView Lightweight Charts | ✅ Complete |
| Technical Indicators | Backend calculation + WebSocket | ✅ Complete |
| Strategy Engine | API + Celery background tasks | ✅ Complete |
| AI Chat Widget | WebSocket + streaming responses | ✅ Complete |
| **Multi-Agent System** | Celery workers + Docker services | ✅ Complete |
| Landing Page Video | HTML5 video element | ✅ Complete |
| Tab Navigation | React Router + tabs | ✅ Complete |
| Real-time Streaming | WebSocket connections | ✅ Complete |
| Database (PostgreSQL) | Same, with migrations | ✅ Complete |

## 🔄 Data Migration

### 1. Export Existing Data

From the desktop application:

```python
# Export script for desktop app
import pandas as pd
from k2_quant.utilities.data.db_manager import db_manager

# Get all tables
tables = db_manager.get_stock_tables()

for table_name, _ in tables:
    # Export to CSV
    df = db_manager.fetch_dataframe(table_name)
    df.to_csv(f"exports/{table_name}.csv", index=False)
    print(f"Exported {table_name}")

# Export strategies
strategies = strategy_service.get_all_strategies()
with open("exports/strategies.json", "w") as f:
    json.dump(strategies, f)
```

### 2. Import to Web Version

```bash
# Run migration script
cd k2_quant_web/backend
python scripts/migrate_data.py --input-dir /path/to/exports
```

## 🎨 UI Component Mapping

### Desktop → Web Components

| PyQt6 Component | Web Equivalent |
|-----------------|----------------|
| `QMainWindow` | React `Layout` component |
| `QWidget` | React functional component |
| `QPushButton` | MUI `Button` |
| `QLineEdit` | MUI `TextField` |
| `QTableWidget` | MUI `DataGrid` |
| `QComboBox` | MUI `Select` |
| `QCheckBox` | MUI `Checkbox` |
| `QSplitter` | Resizable flex containers |
| `pyqtgraph` | TradingView Charts |
| `QMessageBox` | `react-toastify` notifications |
| `QDialog` | MUI `Dialog` |
| `QStatusBar` | Custom status component |

## 🔌 Service Migration

### Desktop Services → Web APIs

1. **StockService**
   - Desktop: Direct database calls
   - Web: REST API endpoints + caching

2. **TechnicalAnalysisService**
   - Desktop: Synchronous calculations
   - Web: Async API + background workers

3. **AIchatService**
   - Desktop: Direct API calls
   - Web: WebSocket streaming + queue

4. **RealtimeStreamService**
   - Desktop: Threading
   - Web: WebSocket + Redis pub/sub

## 🗄️ Database Changes

### Schema Updates

```sql
-- Add new columns for web version
ALTER TABLE stock_data ADD COLUMN IF NOT EXISTS user_id UUID;
ALTER TABLE stock_data ADD COLUMN IF NOT EXISTS session_id VARCHAR(255);
ALTER TABLE stock_data ADD COLUMN IF NOT EXISTS created_via VARCHAR(50) DEFAULT 'web';

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_stock_data_user_id ON stock_data(user_id);
CREATE INDEX IF NOT EXISTS idx_stock_data_session_id ON stock_data(session_id);
```

## 🔐 Authentication (New in Web)

The web version adds authentication:

```typescript
// Frontend authentication
import { useAuth } from './contexts/AuthContext';

const { user, login, logout } = useAuth();

// Protected routes
<PrivateRoute path="/analysis" component={Analysis} />
```

## 📦 Deployment Differences

### Desktop Deployment
- PyInstaller executable
- Local installation
- Direct database connection

### Web Deployment
- Docker containers
- Cloud hosting
- Load balancing
- SSL/TLS encryption
- CDN for static assets

## 🔧 Configuration Changes

### Desktop (.env)
```env
POLYGON_API_KEY=xxx
DB_PATH=local.db
```

### Web (.env)
```env
POLYGON_API_KEY=xxx
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
SECRET_KEY=xxx
CORS_ORIGINS=["http://localhost:3000"]
```

## 🚀 Performance Improvements

### Web Version Optimizations
1. **Caching**: Redis for API responses
2. **Pagination**: Large datasets handled server-side
3. **Lazy Loading**: Components loaded on demand
4. **WebWorkers**: Heavy calculations off main thread
5. **CDN**: Static assets served globally
6. **Connection Pooling**: Database connection reuse

## 🐛 Common Migration Issues

### Issue 1: Large Dataset Performance
**Solution**: Implement server-side pagination and virtualization

### Issue 2: Real-time Updates
**Solution**: WebSocket connections with reconnection logic

### Issue 3: File Exports
**Solution**: Streaming downloads for large files

### Issue 4: Complex Calculations
**Solution**: Offload to Celery background workers

## 📚 API Documentation

Access the auto-generated API docs:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧪 Testing Migration

### Verify Core Functions
```bash
# Run migration tests
cd k2_quant_web
python scripts/test_migration.py

# Expected output:
# ✅ Stock data fetching
# ✅ Technical indicators
# ✅ Strategy execution
# ✅ Chart rendering
# ✅ AI chat
# ✅ Data export
```

## 📱 Browser Compatibility

Tested and supported:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 🆘 Troubleshooting

### Connection Issues
```bash
# Check services
docker-compose ps

# Restart services
docker-compose restart backend
```

### Data Import Errors
```bash
# Validate CSV format
python scripts/validate_import.py data.csv

# Force reimport
python scripts/migrate_data.py --force
```

### Performance Issues
```bash
# Check Redis cache
redis-cli ping

# Monitor Celery workers
celery -A app.celery_app flower
```

## 📞 Support

For migration assistance:
1. Check the [FAQ](./FAQ.md)
2. Review [error logs](./logs/)
3. Open an issue with migration tag

---

**Note**: The web version maintains 100% feature parity with the desktop application while adding cloud-native capabilities.