"""
Stock Data Service Layer

Orchestrates data fetching from Polygon API and storage in database.
Updated to support range parameter in table naming.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from services.polygon_client import polygon_client  # Fixed import
from data.db_manager import db_manager  # Fixed import
from utils.logger import k2_logger, log_performance
from config.api_config import api_config


class StockService:
    """Service layer for stock data operations"""
    
    # Optimized for unlimited tier
    MAX_WORKERS = 50
    CHUNK_SIZE_DAYS = {
        'minute': 7,      # Week chunks for minute data
        'hour': 90,       # Quarter chunks for hourly
        'day': 365,       # Year chunks for daily
        'week': 1825,     # 5 year chunks for weekly
        'month': 7300     # Full range for monthly
    }
    
    def __init__(self):
        self.polygon = polygon_client
        self.db = db_manager
        self.api_key = api_config.polygon_api_key
    
    def convert_ui_parameters(self, time_range: str, frequency: str) -> Tuple[str, str, str, int]:
        """Convert UI parameters to API format"""
        end_date = datetime.now()
        
        range_days = {
            '1D': 1, '1W': 7, '1M': 30, '3M': 90,
            '6M': 180, '1Y': 365, '2Y': 730, '5Y': 1825,
            '10Y': 3650, '20Y': 7300
        }
        
        days = range_days.get(time_range, 30)
        start_date = end_date - timedelta(days=days)
        
        freq_map = {
            '1min': 'minute', '5min': 'minute', '15min': 'minute',
            '30min': 'minute', '1H': 'hour', 'D': 'day',
            'W': 'week', 'M': 'month'
        }
        
        timespan = freq_map.get(frequency, 'day')
        multiplier = 1
        
        if frequency in ['5min', '15min', '30min']:
            multiplier = int(frequency.replace('min', ''))
        
        return timespan, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), multiplier
    
    def validate_symbol(self, symbol: str) -> bool:
        """Quick symbol validation"""
        return self.polygon.validate_symbol(symbol)
    
    @log_performance
    def fetch_and_store_stock_data(self, symbol: str, time_range: str, frequency: str) -> Dict:
        """Main method to fetch and store stock data"""
        start_time = datetime.now()
        
        # Convert parameters
        k2_logger.step(1, 6, "Converting parameters")
        timespan, start_date, end_date, multiplier = self.convert_ui_parameters(time_range, frequency)
        
        # Validate symbol
        k2_logger.step(2, 6, f"Validating symbol {symbol}")
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        # Fetch data
        k2_logger.step(3, 6, "Fetching data from Polygon API")
        all_results = self._parallel_fetch_data(symbol, timespan, start_date, end_date, multiplier)
        
        if not all_results:
            raise ValueError(f"No data available for {symbol}")
        
        # Store data with range parameter
        k2_logger.step(4, 6, "Storing data in database")
        table_name = self.db.store_stock_data(symbol, timespan, time_range.lower(), all_results)
        
        # Calculate metrics
        execution_time = (datetime.now() - start_time).total_seconds()
        records_per_sec = int(len(all_results) / execution_time) if execution_time > 0 else 0
        
        k2_logger.step(5, 6, "Calculating metrics")
        k2_logger.performance_metric("Total execution time", execution_time, "seconds")
        k2_logger.performance_metric("Records processed", len(all_results), "records")
        k2_logger.performance_metric("Processing speed", records_per_sec, "records/second")
        
        # Prepare result
        result = {
            'symbol': symbol,
            'range': time_range,
            'frequency': frequency,
            'table_name': table_name,
            'total_records': len(all_results),
            'execution_time': execution_time,
            'records_per_second': records_per_sec
        }
        
        k2_logger.step(6, 6, "Process complete")
        return result
    
    def _parallel_fetch_data(self, symbol: str, timespan: str, start_date: str, 
                           end_date: str, multiplier: int) -> List[Dict]:
        """Parallel data fetching with chunking"""
        # Generate date chunks
        chunks = self._generate_date_chunks(timespan, start_date, end_date)
        k2_logger.api_operation("Parallel fetch plan", f"{len(chunks)} chunks with {self.MAX_WORKERS} workers")
        
        all_results = []
        failed_chunks = []
        completed_chunks = 0
        
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            future_to_chunk = {
                executor.submit(
                    self._fetch_chunk, 
                    symbol, timespan, multiplier, chunk[0], chunk[1]
                ): chunk
                for chunk in chunks
            }
            
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                completed_chunks += 1
                
                try:
                    results = future.result()
                    if results:
                        all_results.extend(results)
                        k2_logger.api_operation(
                            "Chunk completed", 
                            f"{chunk[0]} to {chunk[1]}: {len(results)} records ({completed_chunks}/{len(chunks)})"
                        )
                except Exception as e:
                    failed_chunks.append(chunk)
                    k2_logger.error(f"Chunk failed {chunk[0]} to {chunk[1]}: {str(e)}", "API")
        
        # Sort by timestamp
        all_results.sort(key=lambda x: x['t'])
        
        return all_results
    
    def _generate_date_chunks(self, timespan: str, start_date: str, end_date: str) -> List[Tuple[str, str]]:
        """Generate date chunks for parallel fetching"""
        chunk_days = self.CHUNK_SIZE_DAYS.get(timespan, 30)
        
        chunks = []
        current_start = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_start < end_dt:
            chunk_end = min(current_start + timedelta(days=chunk_days), end_dt)
            chunks.append((
                current_start.strftime('%Y-%m-%d'),
                chunk_end.strftime('%Y-%m-%d')
            ))
            current_start = chunk_end + timedelta(days=1)
        
        return chunks
    
    def _fetch_chunk(self, symbol: str, timespan: str, multiplier: int, 
                    start: str, end: str) -> List[Dict]:
        """Fetch a single chunk of data"""
        # Get raw data from polygon API
        try:
            # Direct API call to get raw format
            base_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range"
            url = f"{base_url}/{multiplier}/{timespan}/{start}/{end}"
            
            params = {
                'apiKey': self.polygon.api_key,
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'OK' and 'results' in data:
                return data['results']  # Return raw format
            
            return []
            
        except Exception as e:
            k2_logger.error(f"Chunk fetch failed: {str(e)}", "API")
            return []
    
    def get_display_data(self, table_name: str, limit: int = 1000) -> Tuple[List[Tuple], int]:
        """Get data for display"""
        return self.db.fetch_display_data(table_name, limit)
    
    def get_all_stock_tables(self) -> List[Tuple[str, str]]:
        """Get all stock tables"""
        return self.db.get_stock_tables()
    
    def delete_table(self, table_name: str):
        """Delete a specific table"""
        self.db.drop_table(table_name)
    
    def delete_all_tables(self) -> int:
        """Delete all stock tables"""
        return self.db.drop_all_stock_tables()
    
    def get_tables_for_ticker(self, symbol: str, timespan: str = None, range_val: str = None) -> List[str]:
        """Get all table versions for a specific ticker/timespan/range combination"""
        return self.db.get_tables_for_ticker(symbol, timespan, range_val)


# Global service instance
stock_service = StockService()