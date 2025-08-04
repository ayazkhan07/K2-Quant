"""
PostgreSQL Database Manager for Stock Data

Handles all database operations including table creation, data insertion, and index management.
Enhanced with:
- Table naming: ticker_frequency_range (with auto-versioning)
- All timestamps in EST
- No merging or updating - each save creates a new immutable table
"""

import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
import pytz

from utils.logger import k2_logger, log_exception, log_performance


class DatabaseManager:
    """Manages PostgreSQL database operations for stock data"""
    
    # Configuration constants
    MAX_TABLE_VERSIONS = 100  # Prevent infinite loops
    BULK_INSERT_PAGE_SIZE = 10000  # Reduced from 50000 for memory safety
    
    def __init__(self):
        self.pool = ThreadedConnectionPool(
            5, 50,
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'k2_quant'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres'),
            port=os.getenv('DB_PORT', '5433')  # Fixed port
        )
        # EST timezone - could be made configurable
        self.timezone_str = os.getenv('MARKET_TIMEZONE', 'US/Eastern')
        self.market_tz = pytz.timezone(self.timezone_str)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, conn):
        """Context manager for database cursors"""
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()
    
    def get_next_available_table_name(self, symbol: str, timespan: str, range_val: str) -> str:
        """
        Get the next available table name with versioning.
        Optimized to check existing tables in one query.
        """
        base_name = f"stock_{symbol.lower()}_{timespan.lower()}_{range_val.lower()}"
        
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                # Get all tables matching the pattern in one query
                pattern = f"{base_name}%"
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_name LIKE %s
                    ORDER BY table_name
                """, (pattern,))
                
                existing_tables = {row[0] for row in cur.fetchall()}
                
                # Check base name first
                if base_name not in existing_tables:
                    return base_name
                
                # Find next available version
                for version in range(2, self.MAX_TABLE_VERSIONS + 1):
                    versioned_name = f"{base_name}_{version}"
                    if versioned_name not in existing_tables:
                        return versioned_name
                
                # If we've exceeded max versions, raise an error
                raise ValueError(f"Maximum table versions ({self.MAX_TABLE_VERSIONS}) exceeded for {base_name}")
    
    def create_stock_table(self, symbol: str, timespan: str, range_val: str) -> str:
        """Create a table for stock data with auto-versioning"""
        table_name = self.get_next_available_table_name(symbol, timespan, range_val)
        
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                # Create unlogged table for faster initial writes
                k2_logger.database_operation("Creating table", f"{table_name} (UNLOGGED)")
                cur.execute(f"""
                    CREATE UNLOGGED TABLE IF NOT EXISTS {table_name} (
                        timestamp BIGINT PRIMARY KEY,
                        date_time_market TIMESTAMP,
                        open NUMERIC(12, 4),
                        high NUMERIC(12, 4),
                        low NUMERIC(12, 4),
                        close NUMERIC(12, 4),
                        volume BIGINT,
                        vwap NUMERIC(12, 4),
                        transactions INTEGER
                    )
                """)
                
                # Add comment with timezone info
                cur.execute(f"""
                    COMMENT ON COLUMN {table_name}.date_time_market IS 
                    'Market time in {self.timezone_str} timezone'
                """)
                
                conn.commit()
                
        return table_name
    
    def convert_to_market_time(self, timestamp_ms: int) -> datetime:
        """Convert millisecond timestamp to market timezone datetime"""
        # Direct conversion without double timezone handling
        utc_dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
        market_dt = self.market_tz.fromutc(utc_dt)
        # Return without timezone info for storage
        return market_dt.replace(tzinfo=None)
    
    @log_performance
    def bulk_insert_stock_data(self, table_name: str, data: List[Dict]) -> int:
        """Bulk insert stock data into table with market timezone conversion"""
        # Prepare records - handle both raw and transformed data formats
        records = []
        for item in data:
            # Check if data is in transformed format (from polygon_client)
            if 'timestamp' in item:
                # Transformed format
                market_datetime = self.convert_to_market_time(item['timestamp'])
                records.append((
                    item['timestamp'],
                    market_datetime,
                    item['open'],
                    item['high'],
                    item['low'],
                    item['close'],
                    item['volume'],
                    item.get('vwap', 0),
                    item.get('number_of_transactions', 0)
                ))
            else:
                # Raw format (direct from API)
                market_datetime = self.convert_to_market_time(item['t'])
                records.append((
                    item['t'],
                    market_datetime,
                    item['o'], item['h'], item['l'], item['c'],
                    item['v'],
                    item.get('vw', 0),
                    item.get('n', 0)
                ))
        
        if not records:
            k2_logger.warning("No records to insert", "DATABASE")
            return 0
        
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                k2_logger.database_operation("Bulk insert", f"{len(records):,} records")
                
                execute_values(
                    cur,
                    f"""INSERT INTO {table_name} 
                        (timestamp, date_time_market, open, high, low, close, volume, vwap, transactions)
                        VALUES %s""",
                    records,
                    template="(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    page_size=self.BULK_INSERT_PAGE_SIZE
                )
                
                conn.commit()
                k2_logger.database_operation("Bulk insert completed", f"{len(records):,} records")
        
        return len(records)
    
    def convert_to_logged_table(self, table_name: str):
        """Convert unlogged table to logged for durability"""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                k2_logger.database_operation("Converting to logged table", table_name)
                cur.execute(f"ALTER TABLE {table_name} SET LOGGED")
                
                # Analyze for query optimization
                cur.execute(f"ANALYZE {table_name}")
                conn.commit()
    
    def create_indexes(self, table_name: str):
        """Create indexes on stock table"""
        with self.get_connection() as conn:
            try:
                # Set autocommit for index creation
                conn.autocommit = True
                
                with self.get_cursor(conn) as cur:
                    # Create datetime index
                    k2_logger.database_operation("Creating datetime index", table_name)
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_datetime 
                        ON {table_name}(date_time_market)
                    """)
                    
                    # Create volume index
                    k2_logger.database_operation("Creating volume index", table_name)
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_volume 
                        ON {table_name}(volume) 
                        WHERE volume > 0
                    """)
                    
            except Exception as e:
                k2_logger.warning(f"Index creation failed: {str(e)}", "DATABASE")
            finally:
                # Reset autocommit
                conn.autocommit = False
    
    @log_performance
    def store_stock_data(self, symbol: str, timespan: str, range_val: str, data: List[Dict]) -> str:
        """Complete process to store stock data with new naming convention"""
        # Create table with auto-versioning
        table_name = self.create_stock_table(symbol, timespan, range_val)
        
        # Bulk insert data
        self.bulk_insert_stock_data(table_name, data)
        
        # Convert to logged table
        self.convert_to_logged_table(table_name)
        
        # Create indexes
        self.create_indexes(table_name)
        
        return table_name
    
    def get_record_count(self, table_name: str) -> int:
        """Get total record count for a table"""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                return cur.fetchone()[0]
    
    def fetch_display_data(self, table_name: str, limit: int = 1000) -> Tuple[List[Tuple], int]:
        """Fetch data for display with simple sampling for large datasets"""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                # Get total count
                total_count = self.get_record_count(table_name)
                
                # Simple approach for display
                if total_count <= limit:
                    # Get all records if under limit
                    query = f"""
                        SELECT date_time_market, open, high, low, close, volume, vwap
                        FROM {table_name}
                        ORDER BY timestamp
                        LIMIT {limit}
                    """
                else:
                    # Get first 500 and last 500 records efficiently
                    query = f"""
                        (
                            SELECT date_time_market, open, high, low, close, volume, vwap
                            FROM {table_name}
                            ORDER BY timestamp ASC
                            LIMIT {limit // 2}
                        )
                        UNION ALL
                        (
                            SELECT date_time_market, open, high, low, close, volume, vwap
                            FROM {table_name}
                            ORDER BY timestamp DESC
                            LIMIT {limit // 2}
                        )
                        ORDER BY date_time_market
                    """
                
                cur.execute(query)
                rows = cur.fetchall()
                
                return rows, total_count
    
    def get_stock_tables(self) -> List[Tuple[str, str]]:
        """Get list of all stock tables with sizes"""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute("""
                    SELECT tablename, 
                           pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size
                    FROM pg_tables 
                    WHERE tablename LIKE 'stock_%'
                    ORDER BY tablename
                """)
                return cur.fetchall()
    
    def get_tables_for_ticker(self, symbol: str, timespan: str = None, range_val: str = None) -> List[str]:
        """Get all table versions for a specific ticker/timespan/range combination"""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                # Build pattern based on provided parameters
                pattern = f"stock_{symbol.lower()}"
                if timespan:
                    pattern += f"_{timespan.lower()}"
                    if range_val:
                        pattern += f"_{range_val.lower()}"
                pattern += "%"
                
                cur.execute("""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE tablename LIKE %s
                    ORDER BY tablename
                """, (pattern,))
                
                return [row[0] for row in cur.fetchall()]
    
    def drop_table(self, table_name: str):
        """Drop a specific table"""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.commit()
                k2_logger.database_operation("Table dropped", table_name)
    
    def drop_all_stock_tables(self) -> int:
        """Drop all stock tables and return count"""
        tables = self.get_stock_tables()
        
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                for table_name, _ in tables:
                    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.commit()
        
        k2_logger.database_operation("All tables dropped", f"{len(tables)} tables")
        return len(tables)
    
    def close(self):
        """Close the connection pool"""
        self.pool.closeall()


# Global database manager instance
db_manager = DatabaseManager()