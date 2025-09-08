"""
K2 Quant Web Backend - Main FastAPI Application
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import logging
from typing import Dict, List, Optional
import uvicorn

from app.core.config import settings
from app.core.database import engine, Base
from app.api.v1 import api_router
from app.websocket.manager import connection_manager
from app.services.background_tasks import start_background_tasks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting K2 Quant Web Backend...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Start background tasks
    task = asyncio.create_task(start_background_tasks())
    
    # Initialize services
    from app.services.stock_service import stock_service
    from app.services.technical_analysis_service import ta_service
    from app.services.ai_chat_service import ai_service
    
    await stock_service.initialize()
    await ta_service.initialize()
    await ai_service.initialize()
    
    logger.info("K2 Quant Web Backend started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down K2 Quant Web Backend...")
    task.cancel()
    
    # Cleanup services
    await stock_service.cleanup()
    await ta_service.cleanup()
    await ai_service.cleanup()
    
    logger.info("K2 Quant Web Backend shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="K2 Quant Web API",
    description="Stock Price Projection System Web API",
    version="8.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(api_router, prefix="/api/v1")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "K2 Quant Web API",
        "version": "8.0.0",
        "status": "operational",
        "endpoints": {
            "api": "/api/v1",
            "docs": "/docs",
            "redoc": "/redoc",
            "websocket": "/ws",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check database connection
    try:
        from app.core.database import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check services
    from app.services.stock_service import stock_service
    services_status = {
        "stock_service": stock_service.get_status(),
        "database": db_status
    }
    
    overall_health = all(
        status == "healthy" or status.get("status") == "operational"
        for status in services_status.values()
    )
    
    return {
        "status": "healthy" if overall_health else "degraded",
        "services": services_status
    }


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates"""
    await connection_manager.connect(websocket, client_id)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Process different message types
            message_type = data.get("type")
            
            if message_type == "subscribe":
                # Subscribe to specific data streams
                await connection_manager.subscribe(client_id, data.get("channels", []))
                
            elif message_type == "unsubscribe":
                # Unsubscribe from data streams
                await connection_manager.unsubscribe(client_id, data.get("channels", []))
                
            elif message_type == "ping":
                # Respond to ping
                await websocket.send_json({"type": "pong"})
                
            elif message_type == "stock_update":
                # Request stock update
                symbol = data.get("symbol")
                if symbol:
                    from app.services.realtime_service import realtime_service
                    update = await realtime_service.get_stock_update(symbol)
                    await websocket.send_json({
                        "type": "stock_data",
                        "data": update
                    })
                    
            elif message_type == "chat":
                # Handle AI chat messages
                from app.services.ai_chat_service import ai_service
                response = await ai_service.process_chat_message(
                    data.get("message"),
                    data.get("context")
                )
                await websocket.send_json({
                    "type": "chat_response",
                    "data": response
                })
                
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        connection_manager.disconnect(client_id)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )