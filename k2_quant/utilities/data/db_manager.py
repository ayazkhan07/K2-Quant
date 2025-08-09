"""PostgreSQL Database Manager (relocated)"""

import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Generator
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
import pytz

from k2_quant.utilities.logger import k2_logger, log_exception, log_performance


class DatabaseManager:
    MAX_TABLE_VERSIONS = 100
    BULK_INSERT_PAGE_SIZE = 10000
    EXPORT_FETCH_SIZE = 10000

    def __init__(self):
        self.pool = ThreadedConnectionPool(
            5, 50,
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'k2_quant'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres'),
            port=os.getenv('DB_PORT', '5433'),
        )
        self.timezone_str = os.getenv('MARKET_TIMEZONE', 'US/Eastern')
        self.market_tz = pytz.timezone(self.timezone_str)

    @contextmanager
    def get_connection(self):
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    @contextmanager
    def get_cursor(self, conn, cursor_factory=None):
        cur = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cur
        finally:
            cur.close()

    def get_next_available_table_name(self, symbol: str, timespan: str, range_val: str) -> str:
        base_name = f"stock_{symbol.lower()}_{timespan.lower()}_{range_val.lower()}"
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                pattern = f"{base_name}%"
                cur.execute(
                    """
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_name LIKE %s ORDER BY table_name
                    """,
                    (pattern,),
                )
                existing = {row[0] for row in cur.fetchall()}
                if base_name not in existing:
                    return base_name
                for version in range(2, self.MAX_TABLE_VERSIONS + 1):
                    name = f"{base_name}_{version}"
                    if name not in existing:
                        return name
                raise ValueError(f"Maximum table versions ({self.MAX_TABLE_VERSIONS}) exceeded for {base_name}")

    def create_stock_table(self, symbol: str, timespan: str, range_val: str) -> str:
        table_name = self.get_next_available_table_name(symbol, timespan, range_val)
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                k2_logger.database_operation("Creating table", f"{table_name} (UNLOGGED)")
                cur.execute(
                    f"""
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
                    """
                )
                cur.execute(
                    f"""
                    COMMENT ON COLUMN {table_name}.date_time_market IS 
                    'Market time in {self.timezone_str} timezone'
                    """
                )
                conn.commit()
        return table_name

    def convert_to_market_time(self, timestamp_ms: int) -> datetime:
        utc_dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
        market_dt = self.market_tz.fromutc(utc_dt)
        return market_dt.replace(tzinfo=None)

    @log_performance
    def bulk_insert_stock_data(self, table_name: str, data: List[Dict]) -> int:
        records = []
        for item in data:
            if 'timestamp' in item:
                market_datetime = self.convert_to_market_time(item['timestamp'])
                records.append((item['timestamp'], market_datetime, item['open'], item['high'], item['low'], item['close'], item['volume'], item.get('vwap', 0), item.get('number_of_transactions', 0)))
            else:
                market_datetime = self.convert_to_market_time(item['t'])
                records.append((item['t'], market_datetime, item['o'], item['h'], item['l'], item['c'], item['v'], item.get('vw', 0), item.get('n', 0)))
        if not records:
            k2_logger.warning("No records to insert", "DATABASE")
            return 0
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                k2_logger.database_operation("Bulk insert", f"{len(records):,} records")
                execute_values(
                    cur,
                    f"""INSERT INTO {table_name} (timestamp, date_time_market, open, high, low, close, volume, vwap, transactions) VALUES %s""",
                    records,
                    template="(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    page_size=self.BULK_INSERT_PAGE_SIZE,
                )
                conn.commit()
                k2_logger.database_operation("Bulk insert completed", f"{len(records):,} records")
        return len(records)

    def convert_to_logged_table(self, table_name: str):
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                k2_logger.database_operation("Converting to logged table", table_name)
                cur.execute(f"ALTER TABLE {table_name} SET LOGGED")
                cur.execute(f"ANALYZE {table_name}")
                conn.commit()

    def create_indexes(self, table_name: str):
        with self.get_connection() as conn:
            try:
                conn.autocommit = True
                with self.get_cursor(conn) as cur:
                    k2_logger.database_operation("Creating datetime index", table_name)
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_datetime 
                        ON {table_name}(date_time_market)
                    """)
                    k2_logger.database_operation("Creating volume index", table_name)
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_volume 
                        ON {table_name}(volume) 
                        WHERE volume > 0
                    """)
            except Exception as e:
                k2_logger.warning(f"Index creation failed: {str(e)}", "DATABASE")
            finally:
                conn.autocommit = False

    @log_performance
    def store_stock_data(self, symbol: str, timespan: str, range_val: str, data: List[Dict]) -> str:
        table_name = self.create_stock_table(symbol, timespan, range_val)
        self.bulk_insert_stock_data(table_name, data)
        self.convert_to_logged_table(table_name)
        self.create_indexes(table_name)
        return table_name

    def _get_market_hours_where_clause(self) -> str:
        return (
            """
            (EXTRACT(hour FROM date_time_market) * 60 + EXTRACT(minute FROM date_time_market)) >= 570
            AND (EXTRACT(hour FROM date_time_market) * 60 + EXTRACT(minute FROM date_time_market)) <= 960
            """
        )

    def get_record_count(self, table_name: str, market_hours_only: bool = False) -> int:
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                if market_hours_only:
                    query = f"""
                        SELECT COUNT(*) FROM {table_name}
                        WHERE {self._get_market_hours_where_clause()}
                    """
                else:
                    query = f"SELECT COUNT(*) FROM {table_name}"
                cur.execute(query)
                return cur.fetchone()[0]

    def fetch_display_data(self, table_name: str, limit: int = 1000, market_hours_only: bool = False) -> Tuple[List[Tuple], int]:
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                total_count = self.get_record_count(table_name, market_hours_only)
                where_clause = f"WHERE {self._get_market_hours_where_clause()}" if market_hours_only else ""
                if total_count <= limit:
                    query = f"""
                        SELECT date_time_market, open, high, low, close, volume, vwap
                        FROM {table_name}
                        {where_clause}
                        ORDER BY timestamp
                        LIMIT {limit}
                    """
                else:
                    if market_hours_only:
                        query = f"""
                            (
                                SELECT date_time_market, open, high, low, close, volume, vwap
                                FROM {table_name}
                                WHERE {self._get_market_hours_where_clause()}
                                ORDER BY timestamp ASC
                                LIMIT {limit // 2}
                            )
                            UNION ALL
                            (
                                SELECT date_time_market, open, high, low, close, volume, vwap
                                FROM {table_name}
                                WHERE {self._get_market_hours_where_clause()}
                                ORDER BY timestamp DESC
                                LIMIT {limit // 2}
                            )
                            ORDER BY date_time_market
                        """
                    else:
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

    def fetch_export_data(self, table_name: str, offset: int, limit: int, market_hours_only: bool = False) -> List[Tuple]:
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                where_clause = f"WHERE {self._get_market_hours_where_clause()}" if market_hours_only else ""
                query = f"""
                    SELECT date_time_market, open, high, low, close, volume, vwap
                    FROM {table_name}
                    {where_clause}
                    ORDER BY timestamp
                    LIMIT %s OFFSET %s
                """
                cur.execute(query, (limit, offset))
                rows = cur.fetchall()
                k2_logger.database_operation("Export fetch from {table_name}", f"Retrieved {len(rows)} records (offset: {offset}, limit: {limit}, market_hours_only: {market_hours_only})")
                return rows

    def get_date_range(self, table_name: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute(f"SELECT MIN(date_time_market), MAX(date_time_market) FROM {table_name}")
                result = cur.fetchone()
                if result and result[0] is not None:
                    return (result[0], result[1])
                return (None, None)

    def get_table_statistics(self, table_name: str) -> Dict:
        with self.get_connection() as conn:
            with self.get_cursor(conn, cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT 
                        COUNT(*) as total_records,
                        MIN(date_time_market) as earliest_date,
                        MAX(date_time_market) as latest_date,
                        AVG(volume) as avg_volume,
                        SUM(volume) as total_volume,
                        MIN(low) as period_low,
                        MAX(high) as period_high,
                        pg_size_pretty(pg_total_relation_size('{table_name}'::regclass)) as table_size
                    FROM {table_name}
                    """
                )
                stats = dict(cur.fetchone())
                cur.execute(
                    f"""
                    WITH time_diffs AS (
                        SELECT 
                            date_time_market,
                            LAG(date_time_market) OVER (ORDER BY timestamp) as prev_time,
                            date_time_market - LAG(date_time_market) OVER (ORDER BY timestamp) as time_diff
                        FROM {table_name}
                    )
                    SELECT COUNT(*) as gaps FROM time_diffs WHERE time_diff > INTERVAL '1 day'
                    """
                )
                stats['data_gaps'] = cur.fetchone()['gaps']
                return stats

    def validate_table_exists(self, table_name: str) -> bool:
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables WHERE table_name = %s
                    )
                    """,
                    (table_name,),
                )
                return cur.fetchone()[0]

    def get_stock_tables(self) -> List[Tuple[str, str]]:
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute(
                    """
                    SELECT tablename, pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size
                    FROM pg_tables 
                    WHERE tablename LIKE 'stock_%' ORDER BY tablename
                    """
                )
                return cur.fetchall()

    def get_tables_for_ticker(self, symbol: str, timespan: str = None, range_val: str = None) -> List[str]:
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                pattern = f"stock_{symbol.lower()}"
                if timespan:
                    pattern += f"_{timespan.lower()}"
                    if range_val:
                        pattern += f"_{range_val.lower()}"
                pattern += "%"
                cur.execute(
                    """
                    SELECT tablename FROM pg_tables WHERE tablename LIKE %s ORDER BY tablename
                    """,
                    (pattern,),
                )
                return [row[0] for row in cur.fetchall()]

    def drop_table(self, table_name: str):
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.commit()
                k2_logger.database_operation("Table dropped", table_name)

    def drop_all_stock_tables(self) -> int:
        tables = self.get_stock_tables()
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                for table_name, _ in tables:
                    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.commit()
        k2_logger.database_operation("All tables dropped", f"{len(tables)} tables")
        return len(tables)

    def close(self):
        self.pool.closeall()


db_manager = DatabaseManager()


