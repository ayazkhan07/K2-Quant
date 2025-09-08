"""
Stock Data API Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.stock_service import stock_service
from app.services.polygon_service import polygon_service
from app.schemas.stock import (
    StockDataRequest,
    StockDataResponse,
    StockInfo,
    TableInfo,
    ChartDataRequest,
    ChartDataResponse
)

router = APIRouter()


@router.post("/fetch", response_model=StockDataResponse)
async def fetch_stock_data(
    request: StockDataRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Fetch stock data from Polygon API and store in database
    """
    try:
        # Validate symbol
        if not await polygon_service.validate_symbol(request.symbol):
            raise HTTPException(status_code=400, detail=f"Invalid symbol: {request.symbol}")
        
        # Start data fetching in background
        result = await stock_service.fetch_and_store_stock_data(
            symbol=request.symbol,
            time_range=request.time_range,
            frequency=request.frequency,
            market_hours_only=request.market_hours_only,
            db=db
        )
        
        return StockDataResponse(
            success=True,
            message=f"Successfully fetched {result['total_records']} records",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate/{symbol}")
async def validate_symbol(symbol: str):
    """
    Validate if a stock symbol exists
    """
    is_valid = await polygon_service.validate_symbol(symbol)
    return {
        "symbol": symbol,
        "valid": is_valid
    }


@router.get("/tables", response_model=List[TableInfo])
async def get_stock_tables(
    symbol: Optional[str] = None,
    timespan: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get list of available stock data tables
    """
    tables = await stock_service.get_stock_tables(
        symbol=symbol,
        timespan=timespan,
        db=db
    )
    
    return [
        TableInfo(
            table_name=table['name'],
            symbol=table['symbol'],
            timespan=table['timespan'],
            range=table['range'],
            total_records=table['total_records'],
            date_range=table['date_range'],
            created_at=table['created_at']
        )
        for table in tables
    ]


@router.get("/data/{table_name}")
async def get_stock_data(
    table_name: str,
    limit: int = Query(500, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    market_hours_only: bool = False,
    db: Session = Depends(get_db)
):
    """
    Get stock data from a specific table
    """
    try:
        data, total_count = await stock_service.get_display_data(
            table_name=table_name,
            limit=limit,
            offset=offset,
            market_hours_only=market_hours_only,
            db=db
        )
        
        return {
            "table_name": table_name,
            "total_records": total_count,
            "limit": limit,
            "offset": offset,
            "data": data
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Table not found: {table_name}")


@router.post("/chart-data", response_model=ChartDataResponse)
async def get_chart_data(
    request: ChartDataRequest,
    db: Session = Depends(get_db)
):
    """
    Get optimized chart data for visualization
    """
    try:
        # Get data with aggregation if needed
        if request.aggregation and request.aggregation != "none":
            data = await stock_service.get_aggregated_chart_data(
                table_name=request.table_name,
                aggregation=request.aggregation,
                start_date=request.start_date,
                end_date=request.end_date,
                db=db
            )
        else:
            data = await stock_service.get_chart_data_chunk(
                table_name=request.table_name,
                start_idx=request.start_idx or 0,
                end_idx=request.end_idx or 10000,
                db=db
            )
        
        return ChartDataResponse(
            success=True,
            data=data,
            total_points=len(data)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/table/{table_name}")
async def delete_table(
    table_name: str,
    db: Session = Depends(get_db)
):
    """
    Delete a stock data table
    """
    try:
        await stock_service.delete_table(table_name, db)
        return {
            "success": True,
            "message": f"Table {table_name} deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tables/all")
async def delete_all_tables(
    confirm: bool = Query(False),
    db: Session = Depends(get_db)
):
    """
    Delete all stock data tables (requires confirmation)
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Set confirm=true to proceed."
        )
    
    try:
        count = await stock_service.delete_all_tables(db)
        return {
            "success": True,
            "message": f"Deleted {count} tables successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/realtime/{symbol}")
async def get_realtime_quote(symbol: str):
    """
    Get real-time quote for a symbol
    """
    try:
        quote = await polygon_service.get_realtime_quote(symbol)
        return quote
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projections/{table_name}")
async def add_projections(
    table_name: str,
    strategy_name: str,
    projections: List[Dict[str, Any]],
    db: Session = Depends(get_db)
):
    """
    Add projection data to a table
    """
    try:
        count = await stock_service.insert_projections(
            table_name=table_name,
            projections=projections,
            strategy_name=strategy_name,
            db=db
        )
        
        return {
            "success": True,
            "message": f"Added {count} projection records",
            "strategy": strategy_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/projections/{table_name}")
async def remove_projections(
    table_name: str,
    strategy_name: str,
    db: Session = Depends(get_db)
):
    """
    Remove projection data from a table
    """
    try:
        count = await stock_service.delete_projections(
            table_name=table_name,
            strategy_name=strategy_name,
            db=db
        )
        
        return {
            "success": True,
            "message": f"Removed {count} projection records",
            "strategy": strategy_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))