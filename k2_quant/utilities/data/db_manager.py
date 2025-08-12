"""PostgreSQL Database Manager (relocated)"""

import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Generator
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import pandas as pd
from datetime import datetime
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

    # Projection helpers
    def ensure_projection_columns(self, table_name: str) -> None:
        """Ensure projection-related columns exist on the target table."""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                try:
                    cur.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS is_projection BOOLEAN DEFAULT FALSE"
                    )
                    cur.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS projection_source TEXT"
                    )
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    k2_logger.error(f"Failed ensuring projection columns on {table_name}: {str(e)}", "DB")
                    raise

    def bulk_insert_dataframe(self, table_name: str, df) -> int:
        """Bulk insert a pandas DataFrame into the table using execute_values.

        Assumes DataFrame columns map 1:1 to table columns by name.
        """
        try:
            if df is None or len(df) == 0:
                return 0
            columns = [str(c) for c in df.columns]
            values = [tuple(None if pd.isna(v) else v for v in row) for row in df.itertuples(index=False, name=None)]
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cur:
                    execute_values(
                        cur,
                        f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES %s",
                        values,
                        page_size=self.BULK_INSERT_PAGE_SIZE,
                    )
                    affected = cur.rowcount or 0
                    conn.commit()
                    return affected
        except Exception as e:
            k2_logger.error(f"bulk_insert_dataframe failed: {str(e)}", "DB")
            raise

    def delete_where(self, table_name: str, where_sql: str, params: List) -> int:
        """Delete rows from table by predicate and return affected count."""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute(f"DELETE FROM {table_name} WHERE {where_sql}", params)
                affected = cur.rowcount or 0
                conn.commit()
                return affected

    def ensure_indicator_column(self, table_name: str, column_name: str, sql_type: str = "NUMERIC") -> None:
        """Add indicator column if missing."""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {column_name} {sql_type}")
                conn.commit()

    def bulk_update_column_by_timestamp(self, table_name: str, column_name: str, ts_series: pd.Series, val_series: pd.Series) -> int:
        """Efficiently update a numeric indicator column by joining on timestamp."""
        pairs = [(int(ts), None if pd.isna(val) else float(val)) for ts, val in zip(ts_series.values, val_series.values)]
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                # Use VALUES to batch update
                execute_values(
                    cur,
                    f"UPDATE {table_name} AS t SET {column_name} = v.val FROM (VALUES %s) AS v(ts, val) WHERE t.timestamp = v.ts",
                    pairs,
                )
                affected = cur.rowcount or 0
                conn.commit()
                return affected

    def fetch_dataframe(self, table_name: str) -> pd.DataFrame:
        with self.get_connection() as conn:
            return pd.read_sql_query(
                f"SELECT timestamp, date_time_market, open, high, low, close, volume, vwap FROM {table_name} ORDER BY timestamp",
                conn,
            )

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
        """Get date range for a table"""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute(f"SELECT MIN(date_time_market), MAX(date_time_market) FROM {table_name}")
                result = cur.fetchone()
                if result and result[0] is not None:
                    return (result[0], result[1])
                return (None, None)

    def get_table_statistics(self, table_name: str) -> Dict:
        """Get statistics for a table including record count and size"""
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cur:
                    # Get record count
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    total_records = cur.fetchone()[0]
                    
                    # Get table size
                    cur.execute(
                        "SELECT pg_size_pretty(pg_total_relation_size(%s::regclass))",
                        (table_name,)
                    )
                    size = cur.fetchone()[0]
                    
                    return {
                        'total_records': total_records,
                        'size': size
                    }
        except Exception as e:
            k2_logger.error(f"Failed to get table statistics: {str(e)}", "DB")
            return {'total_records': 0, 'size': '0 MB'}

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

# DataFrame fetch helpers for chart windowing
def _to_dt(value):
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))

def _as_read_sql(conn, query: str, params: tuple) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params)

def fetch_time_window_df(self, table_name: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    with self.get_connection() as conn:
        return _as_read_sql(
            conn,
            f"""
            SELECT timestamp, date_time_market, open, high, low, close, volume, vwap
            FROM {table_name}
            WHERE date_time_market BETWEEN %s AND %s
            ORDER BY timestamp
            """,
            (start_dt, end_dt),
        )

def fetch_older_chunk_df(self, table_name: str, before_timestamp: int, limit: int) -> pd.DataFrame:
    with self.get_connection() as conn:
        df = _as_read_sql(
            conn,
            f"""
            SELECT timestamp, date_time_market, open, high, low, close, volume, vwap
            FROM {table_name}
            WHERE timestamp < %s
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (before_timestamp, limit),
        )
        return df.iloc[::-1].reset_index(drop=True)