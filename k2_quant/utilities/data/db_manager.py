"""PostgreSQL Database Manager (relocated)"""

import os
from datetime import datetime, time
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
                        market_date DATE,
                        market_time TIME,
                        open NUMERIC(12, 4),
                        high NUMERIC(12, 4),
                        low NUMERIC(12, 4),
                        close NUMERIC(12, 4),
                        volume BIGINT,
                        vwap NUMERIC(12, 4),
                        transactions INTEGER,
                        open_pct DOUBLE PRECISION,
                        high_pct DOUBLE PRECISION,
                        low_pct DOUBLE PRECISION,
                        close_pct DOUBLE PRECISION,
                        elasticity DOUBLE PRECISION,
                        close_open_pct DOUBLE PRECISION
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
    def bulk_insert_stock_data(self, table_name: str, data: List[Dict], market_hours_only: bool = False) -> int:
        records = []
        for item in data:
            if 'timestamp' in item:
                market_datetime = self.convert_to_market_time(item['timestamp'])
                market_date = market_datetime.date()
                market_time = market_datetime.time()
                # Filter at write time if requested
                if market_hours_only and not (time(9, 30) <= market_time < time(16, 0)):
                    continue
                records.append((
                    item['timestamp'],
                    market_datetime,
                    market_date,
                    market_time,
                    item['open'],
                    item['high'],
                    item['low'],
                    item['close'],
                    item['volume'],
                    item.get('vwap', 0),
                    item.get('number_of_transactions', 0)
                ))
            else:
                market_datetime = self.convert_to_market_time(item['t'])
                market_date = market_datetime.date()
                market_time = market_datetime.time()
                # Filter at write time if requested
                if market_hours_only and not (time(9, 30) <= market_time < time(16, 0)):
                    continue
                records.append((
                    item['t'],
                    market_datetime,
                    market_date,
                    market_time,
                    item['o'],
                    item['h'],
                    item['l'],
                    item['c'],
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
                    f"""INSERT INTO {table_name} (timestamp, date_time_market, market_date, market_time, open, high, low, close, volume, vwap, transactions) VALUES %s""",
                    records,
                    template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
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
                    # New indexes for split date/time columns
                    k2_logger.database_operation("Creating market_date index", table_name)
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_market_date 
                        ON {table_name}(market_date)
                    """)
                    k2_logger.database_operation("Creating market_time index", table_name)
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_market_time 
                        ON {table_name}(market_time)
                    """)
                    k2_logger.database_operation("Creating composite market_date_time index", table_name)
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_market_date_time 
                        ON {table_name}(market_date, market_time)
                    """)
            except Exception as e:
                k2_logger.warning(f"Index creation failed: {str(e)}", "DATABASE")
            finally:
                conn.autocommit = False

    @log_performance
    def store_stock_data(self, symbol: str, timespan: str, range_val: str, data: List[Dict], market_hours_only: bool = False) -> Tuple[str, int]:
        table_name = self.create_stock_table(symbol, timespan, range_val)
        inserted_count = self.bulk_insert_stock_data(table_name, data, market_hours_only=market_hours_only)
        self.convert_to_logged_table(table_name)
        self.create_indexes(table_name)
        self.compute_derived_columns(table_name)
        return table_name, inserted_count

    def compute_derived_columns(self, table_name: str):
        """Populate open_pct, high_pct, low_pct, close_pct, elasticity, close_open_pct.

        Uses window functions against the chronologically ordered rows.
        The first row in each table will have NULL for the four pct-change
        columns (no prior row to compare against).
        """
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute(f"""
                    WITH computed AS (
                        SELECT
                            timestamp,
                            CASE WHEN LAG(open)  OVER w <> 0
                                 THEN ((open  - LAG(open)  OVER w) / LAG(open)  OVER w) * 100
                            END AS open_pct,
                            CASE WHEN LAG(high)  OVER w <> 0
                                 THEN ((high  - LAG(high)  OVER w) / LAG(high)  OVER w) * 100
                            END AS high_pct,
                            CASE WHEN LAG(low)   OVER w <> 0
                                 THEN ((low   - LAG(low)   OVER w) / LAG(low)   OVER w) * 100
                            END AS low_pct,
                            CASE WHEN LAG(close) OVER w <> 0
                                 THEN ((close - LAG(close) OVER w) / LAG(close) OVER w) * 100
                            END AS close_pct,
                            CASE WHEN low <> 0
                                 THEN ((high - low) / low) * 100
                            END AS elasticity,
                            CASE WHEN open <> 0
                                 THEN ((close - open) / open) * 100
                            END AS close_open_pct
                        FROM {table_name}
                        WINDOW w AS (ORDER BY timestamp)
                    )
                    UPDATE {table_name} t
                    SET open_pct      = c.open_pct,
                        high_pct      = c.high_pct,
                        low_pct       = c.low_pct,
                        close_pct     = c.close_pct,
                        elasticity    = c.elasticity,
                        close_open_pct = c.close_open_pct
                    FROM computed c
                    WHERE t.timestamp = c.timestamp
                """)
                conn.commit()
        k2_logger.info(
            f"Derived columns computed for {table_name}", "DATABASE")

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

    def persist_indicator_columns(
        self,
        table_name: str,
        indicators: 'Dict[str, Any]',
        timestamps: 'pd.Series',
    ) -> int:
        """
        Create indicator columns (if missing) and batch-update their values.

        Parameters
        ----------
        table_name : str
            Target PostgreSQL table.
        indicators : dict
            Mapping of column_name -> numpy array of values (aligned to *timestamps*).
        timestamps : pd.Series
            Timestamp series (integer epoch ms) aligned to the value arrays.

        Returns
        -------
        int
            Total number of column-row updates performed.
        """
        import numpy as np

        total_affected = 0
        for col_name, values in indicators.items():
            try:
                # Ensure column exists
                self.ensure_indicator_column(table_name, col_name, "DOUBLE PRECISION")

                # Build (timestamp, value) pairs, converting NaN to NULL
                pairs = [
                    (int(ts), None if (val is None or np.isnan(val)) else float(val))
                    for ts, val in zip(timestamps.values, values)
                ]

                with self.get_connection() as conn:
                    with self.get_cursor(conn) as cur:
                        execute_values(
                            cur,
                            f"UPDATE {table_name} AS t "
                            f"SET {col_name} = v.val "
                            f"FROM (VALUES %s) AS v(ts, val) "
                            f"WHERE t.timestamp = v.ts",
                            pairs,
                        )
                        total_affected += cur.rowcount or 0
                        conn.commit()
            except Exception as e:
                k2_logger.error(
                    f"persist_indicator_columns failed for '{col_name}': {e}", "DB"
                )

        k2_logger.info(
            f"Persisted {len(indicators)} indicator columns on {table_name} "
            f"({total_affected} cell updates)",
            "DB",
        )
        return total_affected

    def add_row_number_column(self, table_name: str) -> int:
        """Add a '#' column with sequential row numbers (1-based) ordered by timestamp."""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute(f'ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS "#" INTEGER')
                cur.execute(f"""
                    WITH numbered AS (
                        SELECT timestamp, ROW_NUMBER() OVER (ORDER BY timestamp) AS rn
                        FROM {table_name}
                    )
                    UPDATE {table_name} t
                    SET "#" = n.rn
                    FROM numbered n
                    WHERE t.timestamp = n.timestamp
                """)
                affected = cur.rowcount or 0
                conn.commit()
                k2_logger.info(
                    f"Row number column added to {table_name}: {affected} rows numbered",
                    "DATABASE",
                )
                return affected

    def fetch_dataframe(self, table_name: str) -> pd.DataFrame:
        """Fetch dataframe including any persisted indicator columns."""
        with self.get_connection() as conn:
            return pd.read_sql_query(
                f"SELECT * FROM {table_name} ORDER BY timestamp",
                conn,
            )

    def _check_column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table"""
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 
                            FROM information_schema.columns 
                            WHERE table_name = %s 
                            AND column_name = %s
                        )
                    """, (table_name, column_name))
                    return cur.fetchone()[0]
        except Exception:
            return False

    def _get_market_hours_where_clause(self, table_name: str = None) -> str:
        """
        Get market hours WHERE clause that works with both old and new table schemas.
        Checks if market_time column exists to determine which approach to use.
        """
        if table_name and self._check_column_exists(table_name, 'market_time'):
            return "market_time BETWEEN TIME '09:30:00' AND TIME '16:00:00'"
        else:
            return "CAST(date_time_market AS TIME) BETWEEN TIME '09:30:00' AND TIME '16:00:00'"

    def _get_time_filter_clause(self, table_name: str, market_hours_only: bool = False,
                                time_start: str = None, time_end: str = None) -> str:
        """Build a WHERE-clause fragment for time-of-day filtering.

        Priority: explicit time_start/time_end > market_hours_only.
        Returns an empty string when no filtering is needed.
        """
        if time_start and time_end:
            ts = time_start if len(time_start) > 5 else f"{time_start}:00"
            te = time_end if len(time_end) > 5 else f"{time_end}:00"
            if self._check_column_exists(table_name, 'market_time'):
                return f"market_time BETWEEN TIME '{ts}' AND TIME '{te}'"
            else:
                return f"CAST(date_time_market AS TIME) BETWEEN TIME '{ts}' AND TIME '{te}'"
        elif market_hours_only:
            return self._get_market_hours_where_clause(table_name)
        return ""

    def get_record_count(self, table_name: str, market_hours_only: bool = False,
                         time_start: str = None, time_end: str = None) -> int:
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                time_filter = self._get_time_filter_clause(
                    table_name, market_hours_only, time_start, time_end)
                if time_filter:
                    query = f"SELECT COUNT(*) FROM {table_name} WHERE {time_filter}"
                else:
                    query = f"SELECT COUNT(*) FROM {table_name}"
                cur.execute(query)
                return cur.fetchone()[0]

    def fetch_display_data(self, table_name: str, limit: int = 1000, market_hours_only: bool = False,
                           time_start: str = None, time_end: str = None) -> Tuple[List[Tuple], int]:
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                total_count = self.get_record_count(table_name, market_hours_only, time_start, time_end)

                has_new_columns = self._check_column_exists(table_name, 'market_date')
                has_derived = self._check_column_exists(table_name, 'open_pct')
                has_row_num = self._check_column_exists(table_name, '#')

                row_num_prefix = '"#", ' if has_row_num else ''
                derived_clause = ""
                if has_derived:
                    derived_clause = ", open_pct, high_pct, low_pct, close_pct, elasticity, close_open_pct"

                if has_new_columns:
                    select_clause = f"""
                        {row_num_prefix}market_date,
                        market_time,
                        open, high, low, close, volume, vwap{derived_clause}
                    """
                else:
                    select_clause = f"""
                        {row_num_prefix}DATE(date_time_market) AS market_date,
                        CAST(date_time_market AS TIME) AS market_time,
                        open, high, low, close, volume, vwap{derived_clause}
                    """

                time_filter = self._get_time_filter_clause(
                    table_name, market_hours_only, time_start, time_end)
                where_clause = f"WHERE {time_filter}" if time_filter else ""

                if total_count <= limit:
                    query = f"""
                        SELECT {select_clause}
                        FROM {table_name}
                        {where_clause}
                        ORDER BY timestamp
                        LIMIT {limit}
                    """
                else:
                    query = f"""
                        (
                            SELECT {select_clause}
                            FROM {table_name}
                            {where_clause}
                            ORDER BY timestamp ASC
                            LIMIT {limit // 2}
                        )
                        UNION ALL
                        (
                            SELECT {select_clause}
                            FROM {table_name}
                            {where_clause}
                            ORDER BY timestamp DESC
                            LIMIT {limit // 2}
                        )
                        ORDER BY 1, 2
                    """
                cur.execute(query)
                rows = cur.fetchall()
                return rows, total_count

    def fetch_export_data(self, table_name: str, offset: int, limit: int, market_hours_only: bool = False,
                          time_start: str = None, time_end: str = None) -> List[Tuple]:
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                has_new_columns = self._check_column_exists(table_name, 'market_date')

                if has_new_columns:
                    select_clause = """
                        market_date,
                        market_time,
                        open, high, low, close, volume, vwap
                    """
                else:
                    select_clause = """
                        DATE(date_time_market) AS market_date,
                        CAST(date_time_market AS TIME) AS market_time,
                        open, high, low, close, volume, vwap
                    """

                time_filter = self._get_time_filter_clause(
                    table_name, market_hours_only, time_start, time_end)
                where_clause = f"WHERE {time_filter}" if time_filter else ""

                query = f"""
                    SELECT {select_clause}
                    FROM {table_name}
                    {where_clause}
                    ORDER BY timestamp
                    LIMIT %s OFFSET %s
                """
                cur.execute(query, (limit, offset))
                rows = cur.fetchall()
                k2_logger.database_operation(f"Export fetch from {table_name}",
                    f"Retrieved {len(rows)} records (offset: {offset}, limit: {limit})")
                return rows

    def fetch_daily_bars(self, table_name: str, market_hours_only: bool = False) -> List[Tuple]:
        """Server-side daily aggregation — returns (date, '00:00:00', O, H, L, C, V, VWAP)."""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                has_new = self._check_column_exists(table_name, 'market_date')
                date_col = 'market_date' if has_new else 'DATE(date_time_market)'
                time_filter = self._get_time_filter_clause(
                    table_name, market_hours_only)
                where_clause = f"WHERE {time_filter}" if time_filter else ""
                query = f"""
                    SELECT {date_col}                       AS "Date",
                           '00:00:00'                       AS "Time",
                           (ARRAY_AGG(open ORDER BY timestamp))[1]  AS "Open",
                           MAX(high)                        AS "High",
                           MIN(low)                         AS "Low",
                           (ARRAY_AGG(close ORDER BY timestamp DESC))[1] AS "Close",
                           SUM(volume)                      AS "Volume",
                           (ARRAY_AGG(vwap ORDER BY timestamp DESC))[1]  AS "VWAP"
                    FROM {table_name}
                    {where_clause}
                    GROUP BY {date_col}
                    ORDER BY {date_col}
                """
                cur.execute(query)
                rows = cur.fetchall()
                k2_logger.database_operation(
                    f"Daily agg from {table_name}",
                    f"Retrieved {len(rows)} daily bars")
                return rows

    def fetch_time_window_df(self, table_name: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Fetch data within a time window"""
        with self.get_connection() as conn:
            # Check if we have the new columns
            has_new_columns = self._check_column_exists(table_name, 'market_date')
            
            if has_new_columns:
                select_clause = "timestamp, market_date, market_time, open, high, low, close, volume, vwap"
            else:
                select_clause = """
                    timestamp,
                    DATE(date_time_market) as market_date,
                    CAST(date_time_market AS TIME) as market_time,
                    open, high, low, close, volume, vwap
                """
            
            return pd.read_sql_query(
                f"""
                SELECT {select_clause}
                FROM {table_name}
                WHERE date_time_market BETWEEN %s AND %s
                ORDER BY timestamp
                """,
                conn,
                params=(start_dt, end_dt)
            )
    
    def fetch_older_chunk_df(self, table_name: str, before_timestamp: int, limit: int) -> pd.DataFrame:
        """Fetch older data chunks for pagination"""
        with self.get_connection() as conn:
            # Check if we have the new columns
            has_new_columns = self._check_column_exists(table_name, 'market_date')
            
            if has_new_columns:
                select_clause = "timestamp, market_date, market_time, open, high, low, close, volume, vwap"
            else:
                select_clause = """
                    timestamp,
                    DATE(date_time_market) as market_date,
                    CAST(date_time_market AS TIME) as market_time,
                    open, high, low, close, volume, vwap
                """
            
            df = pd.read_sql_query(
                f"""
                SELECT {select_clause}
                FROM {table_name}
                WHERE timestamp < %s
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                conn,
                params=(before_timestamp, limit)
            )
            return df.iloc[::-1].reset_index(drop=True)

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

    # ── Tab data persistence (forecast / workspace) ─────────────────

    def _ensure_tab_data_table(self):
        """Create the k2_tab_data table if it doesn't exist."""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS k2_tab_data (
                        id SERIAL PRIMARY KEY,
                        scope VARCHAR(512) NOT NULL UNIQUE,
                        data JSONB NOT NULL DEFAULT '{}',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()

    def save_tab_data(self, scope: str, data: dict):
        """Upsert a JSON blob keyed by *scope* (e.g. 'forecast:table', 'workspace:table')."""
        import json as _json
        self._ensure_tab_data_table()
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute("""
                    INSERT INTO k2_tab_data (scope, data, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (scope) DO UPDATE SET
                        data = EXCLUDED.data,
                        updated_at = CURRENT_TIMESTAMP
                """, (scope, _json.dumps(data)))
                conn.commit()
        k2_logger.info(f"Tab data saved: {scope}", "DB")

    def load_tab_data(self, scope: str) -> Optional[dict]:
        """Load a previously saved JSON blob by *scope* (returns None if absent)."""
        self._ensure_tab_data_table()
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute("SELECT data FROM k2_tab_data WHERE scope = %s", (scope,))
                row = cur.fetchone()
                return row[0] if row else None

    def delete_tab_data(self, scope: str):
        """Remove a previously saved JSON blob by *scope*."""
        self._ensure_tab_data_table()
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cur:
                cur.execute("DELETE FROM k2_tab_data WHERE scope = %s", (scope,))
                conn.commit()
        k2_logger.info(f"Tab data deleted: {scope}", "DB")

    def close(self):
        self.pool.closeall()


# Singleton instance
db_manager = DatabaseManager()

# Helper function (outside class, doesn't use self)
def _to_dt(value):
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))