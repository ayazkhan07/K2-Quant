#!/bin/bash

# K2 Quant Web - Setup Script
# This script sets up the K2 Quant Web application

set -e

echo "================================================"
echo "     K2 Quant Web - Setup & Installation"
echo "================================================"
echo ""

# Check for required tools
echo "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi
echo "✅ Docker found"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi
echo "✅ Docker Compose found"

# Check for .env file
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Please edit the .env file and add your API keys:"
    echo "   - POLYGON_API_KEY (required for stock data)"
    echo "   - OPENAI_API_KEY (optional for AI chat)"
    echo "   - ANTHROPIC_API_KEY (optional for AI chat)"
    echo ""
    read -p "Press Enter after you've added your API keys to .env file..."
fi

# Check if API keys are set
source .env
if [ -z "$POLYGON_API_KEY" ] || [ "$POLYGON_API_KEY" = "your_polygon_api_key_here" ]; then
    echo "❌ POLYGON_API_KEY is not set in .env file"
    echo "   Please add your Polygon API key to continue"
    exit 1
fi
echo "✅ API keys configured"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p docker/postgres
mkdir -p docker/nginx/conf.d
mkdir -p logs
echo "✅ Directories created"

# Create PostgreSQL init script
echo ""
echo "Creating database initialization script..."
cat > docker/postgres/init.sql << 'EOF'
-- K2 Quant Web Database Initialization

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_stock_timestamp ON stock_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_stock_symbol ON stock_data(symbol);

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE k2quant_web TO k2quant;
EOF
echo "✅ Database init script created"

# Create Nginx configuration
echo ""
echo "Creating Nginx configuration..."
cat > docker/nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }

    upstream frontend {
        server frontend:3000;
    }

    server {
        listen 80;
        server_name localhost;

        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
        }

        # Backend API
        location /api {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket
        location /ws {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
EOF
echo "✅ Nginx configuration created"

# Build and start services
echo ""
echo "Building Docker images..."
docker-compose build

echo ""
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo ""
echo "Waiting for services to start..."
sleep 10

# Check service health
echo ""
echo "Checking service status..."
docker-compose ps

# Display access information
echo ""
echo "================================================"
echo "           Setup Complete! 🎉"
echo "================================================"
echo ""
echo "Access the application at:"
echo "  📊 Frontend:     http://localhost:3000"
echo "  🔧 Backend API:  http://localhost:8000"
echo "  📚 API Docs:     http://localhost:8000/docs"
echo "  🌻 Celery:       http://localhost:5555"
echo ""
echo "Default credentials (if auth enabled):"
echo "  Username: admin"
echo "  Password: admin123"
echo ""
echo "Useful commands:"
echo "  View logs:       docker-compose logs -f [service]"
echo "  Stop services:   docker-compose down"
echo "  Restart service: docker-compose restart [service]"
echo ""
echo "For help, see README.md and MIGRATION_GUIDE.md"
echo "================================================"