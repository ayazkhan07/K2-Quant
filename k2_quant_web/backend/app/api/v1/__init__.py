"""
API Version 1 Router
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    stocks,
    analysis,
    strategies,
    indicators,
    models,
    agents,
    chat,
    export
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(stocks.router, prefix="/stocks", tags=["stocks"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
api_router.include_router(strategies.router, prefix="/strategies", tags=["strategies"])
api_router.include_router(indicators.router, prefix="/indicators", tags=["indicators"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(export.router, prefix="/export", tags=["export"])