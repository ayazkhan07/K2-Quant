"""
Stock Data Service Layer

Orchestrates data fetching from Polygon API and storage in database.
Supports range parameter in table naming, CSV export, and market hours filter.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from k2_quant.utilities.services.polygon_client import polygon_client
from k2_quant.utilities.data.db_manager import db_manager
from k2_quant.utilities.logger import k2_logger, log_performance
from k2_quant.utilities.config.api_config import api_config
from datetime import datetime, timedelta
import pandas as pd


class StockService:
    """Service layer for stock data operations"""

    MAX_WORKERS = 50
    CHUNK_SIZE_DAYS = {
        'minute': 7,
        'hour': 90,
        'day': 365,
        'week': 1825,
        'month': 7300,
    }

    EXPORT_BATCH_SIZE = 100000

    def __init__(self):
        self.polygon = polygon_client
        self.db = db_manager
        self.api_key = api_config.polygon_api_key

    def convert_ui_parameters(self, time_range: str, frequency: str) -> Tuple[str, str, str, int]:
        end_date = datetime.now()
        range_days = {
            '1D': 1, '1W': 7, '1M': 30, '3M': 90,
            '6M': 180, '1Y': 365, '2Y': 730, '5Y': 1825,
            '10Y': 3650, '20Y': 7300,
        }
        days = range_days.get(time_range, 30)
        start_date = end_date - timedelta(days=days)
        freq_map = {
            '1min': 'minute', '5min': 'minute', '15min': 'minute',
            '30min': 'minute', '1H': 'hour', 'D': 'day',
            'W': 'week', 'M': 'month',
        }
        timespan = freq_map.get(frequency, 'day')
        multiplier = 1
        if frequency in ['5min', '15min', '30min']:
            multiplier = int(frequency.replace('min', ''))
        return timespan, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), multiplier

    def validate_symbol(self, symbol: str) -> bool:
        return self.polygon.validate_symbol(symbol)

    @log_performance
    def fetch_and_store_stock_data(self, symbol: str, time_range: str, frequency: str, market_hours_only: bool = False) -> Dict:
        start_time = datetime.now()
        k2_logger.step(1, 6, "Converting parameters")
        timespan, start_date, end_date, multiplier = self.convert_ui_parameters(time_range, frequency)
        k2_logger.step(2, 6, f"Validating symbol {symbol}")
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        k2_logger.step(3, 6, "Fetching data from Polygon API")
        all_results = self._parallel_fetch_data(symbol, timespan, start_date, end_date, multiplier)
        if not all_results:
            raise ValueError(f"No data available for {symbol}")
        k2_logger.step(4, 6, "Storing data in database")
        table_name = self.db.store_stock_data(symbol, timespan, time_range.lower(), all_results, market_hours_only=market_hours_only)
        execution_time = (datetime.now() - start_time).total_seconds()
        records_per_sec = int(len(all_results) / execution_time) if execution_time > 0 else 0
        k2_logger.step(5, 6, "Calculating metrics")
        k2_logger.performance_metric("Total execution time", execution_time, "seconds")
        k2_logger.performance_metric("Records processed", len(all_results), "records")
        k2_logger.performance_metric("Processing speed", records_per_sec, "records/second")
        result = {
            'symbol': symbol,
            'range': time_range,
            'frequency': frequency,
            'table_name': table_name,
            'total_records': len(all_results),
            'execution_time': execution_time,
            'records_per_second': records_per_sec,
        }
        k2_logger.step(6, 6, "Process complete")
        return result

    def _parallel_fetch_data(self, symbol: str, timespan: str, start_date: str, end_date: str, multiplier: int) -> List[Dict]:
        chunks = self._generate_date_chunks(timespan, start_date, end_date)
        k2_logger.api_operation("Parallel fetch plan", f"{len(chunks)} chunks with {self.MAX_WORKERS} workers")
        all_results: List[Dict] = []
        failed_chunks = []
        completed_chunks = 0
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            future_to_chunk = {
                executor.submit(self._fetch_chunk, symbol, timespan, multiplier, chunk[0], chunk[1]): chunk
                for chunk in chunks
            }
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                completed_chunks += 1
                try:
                    results = future.result()
                    if results:
                        all_results.extend(results)
                        k2_logger.api_operation("Chunk completed", f"{chunk[0]} to {chunk[1]}: {len(results)} records ({completed_chunks}/{len(chunks)})")
                except Exception as e:
                    failed_chunks.append(chunk)
                    k2_logger.error(f"Chunk failed {chunk[0]} to {chunk[1]}: {str(e)}", "API")
        all_results.sort(key=lambda x: x['t'])
        return all_results

    def _generate_date_chunks(self, timespan: str, start_date: str, end_date: str) -> List[Tuple[str, str]]:
        chunk_days = self.CHUNK_SIZE_DAYS.get(timespan, 30)
        chunks = []
        current_start = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        while current_start < end_dt:
            chunk_end = min(current_start + timedelta(days=chunk_days), end_dt)
            chunks.append((current_start.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')))
            current_start = chunk_end + timedelta(days=1)
        return chunks

    def _fetch_chunk(self, symbol: str, timespan: str, multiplier: int, start: str, end: str) -> List[Dict]:
        try:
            base_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range"
            url = f"{base_url}/{multiplier}/{timespan}/{start}/{end}"
            params = {
                'apiKey': self.polygon.api_key,
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get('status') == 'OK' and 'results' in data:
                return data['results']
            return []
        except Exception as e:
            k2_logger.error(f"Chunk fetch failed: {str(e)}", "API")
            return []

    def get_display_data(self, table_name: str, limit: int = 1000, market_hours_only: bool = False) -> Tuple[List[Tuple], int]:
        # DIAGNOSTIC: display fetch summary (base 8-col path)
        try:
            k2_logger.database_operation("Display fetch", f"table={table_name} limit={limit} market_hours_only={market_hours_only}")
            rows, total = self.db.fetch_display_data(table_name, limit, market_hours_only)
            if rows:
                k2_logger.database_operation("Display fetch sample", f"first_row_cols={len(rows[0])} first_row_preview={rows[0]}")
            return rows, total
        except Exception as ex:
            k2_logger.error(f"get_display_data failed: {ex}", "DB")
            return [], 0

    def get_export_data(self, table_name: str, offset: int, limit: int, market_hours_only: bool = False) -> List[Tuple]:
        try:
            k2_logger.database_operation(
                f"Fetching export data from {table_name}",
                f"Offset: {offset}, Limit: {limit}, Market hours filter: {market_hours_only}")
            rows = self.db.fetch_export_data(table_name, offset, limit, market_hours_only)
            k2_logger.database_operation("Export data fetched", f"Retrieved {len(rows)} records")
            return rows
        except Exception as e:
            k2_logger.error(f"Failed to fetch export data: {str(e)}", "EXPORT")
            raise

    def get_export_data_streaming(self, table_name: str, batch_size: int = None, market_hours_only: bool = False) -> Generator[List[Tuple], None, None]:
        if batch_size is None:
            batch_size = self.EXPORT_BATCH_SIZE
        try:
            _, total_count = self.get_display_data(table_name, limit=1, market_hours_only=market_hours_only)
            offset = 0
            while offset < total_count:
                batch = self.get_export_data(table_name, offset, batch_size, market_hours_only)
                if not batch:
                    break
                yield batch
                offset += len(batch)
                progress_pct = (offset / total_count) * 100
                k2_logger.database_operation("Export progress", f"{offset:,}/{total_count:,} records ({progress_pct:.1f}%)")
        except Exception as e:
            k2_logger.error(f"Streaming export failed: {str(e)}", "EXPORT")
            raise

    def create_filtered_table(self, source_table: str, target_table: str, market_hours_only: bool = True) -> bool:
        """
        Create a new table that contains filtered data from the source table.

        If market_hours_only is True, only rows within 09:30:00-16:00:00 are copied.
        Uses market_time column if present; otherwise falls back to CAST(date_time_market AS TIME).
        """
        try:
            with self.db.get_connection() as conn:
                with self.db.get_cursor(conn) as cur:
                    # Drop target if exists
                    cur.execute(f"DROP TABLE IF EXISTS {target_table}")

                    # Create target with the same structure; keep UNLOGGED for speed (matches ingestion tables)
                    cur.execute(
                        f"""
                        CREATE UNLOGGED TABLE {target_table} (LIKE {source_table} INCLUDING ALL)
                        """
                    )

                    if market_hours_only:
                        # Prefer market_time if available; otherwise fallback to extracting time from date_time_market
                        try:
                            cur.execute(
                                f"""
                                INSERT INTO {target_table}
                                SELECT * FROM {source_table}
                                WHERE market_time BETWEEN TIME '09:30:00' AND TIME '16:00:00'
                                """
                            )
                        except Exception:
                            cur.execute(
                                f"""
                                INSERT INTO {target_table}
                                SELECT * FROM {source_table}
                                WHERE CAST(date_time_market AS TIME) BETWEEN TIME '09:30:00' AND TIME '16:00:00'
                                """
                            )
                    else:
                        cur.execute(f"INSERT INTO {target_table} SELECT * FROM {source_table}")

                    conn.commit()

                    # Log counts
                    cur.execute(f"SELECT COUNT(*) FROM {source_table}")
                    source_count = cur.fetchone()[0]
                    cur.execute(f"SELECT COUNT(*) FROM {target_table}")
                    target_count = cur.fetchone()[0]
                    k2_logger.info(
                        f"Created filtered table: {target_table} with {target_count:,} records (from {source_count:,})",
                        "STOCK_SERVICE",
                    )
                    return True
        except Exception as e:
            k2_logger.error(f"Failed to create filtered table: {e}", "STOCK_SERVICE")
            return False

    # Minimal chart helpers (no DB manager changes needed)
    def get_chart_data_chunk(self, table_name: str, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Return a DataFrame of rows [start_idx, end_idx) in display format."""
        try:
            limit = max(0, end_idx - start_idx)
            if limit <= 0:
                return pd.DataFrame()
            rows = self.db.fetch_export_data(table_name, offset=start_idx, limit=limit, market_hours_only=False)
            if not rows:
                return pd.DataFrame()
            columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
            return pd.DataFrame(rows, columns=columns[:len(rows[0])])
        except Exception as e:
            k2_logger.error(f"get_chart_data_chunk failed: {str(e)}", "CHART_DATA")
            return pd.DataFrame()

    def get_table_info(self, table_name: str) -> Dict:
        try:
            _, total_records = self.get_display_data(table_name, limit=1)
            parts = table_name.split('_')
            symbol = parts[1] if len(parts) > 1 else 'UNKNOWN'
            timespan = parts[2] if len(parts) > 2 else 'unknown'
            range_val = parts[3] if len(parts) > 3 else 'unknown'
            date_range = self.db.get_date_range(table_name)
            return {
                'table_name': table_name,
                'symbol': symbol.upper(),
                'timespan': timespan,
                'range': range_val,
                'total_records': total_records,
                'date_range': date_range,
                'version': parts[-1] if len(parts) > 4 and parts[-1].isdigit() else '1',
            }
        except Exception as e:
            k2_logger.error(f"Failed to get table info: {str(e)}", "INFO")
            return {}

    def validate_export_size(self, table_name: str, max_size_gb: float = 10.0) -> Tuple[bool, str]:
        try:
            _, total_records = self.get_display_data(table_name, limit=1)
            estimated_bytes = total_records * 100
            estimated_gb = estimated_bytes / (1024 ** 3)
            if estimated_gb > max_size_gb:
                return False, f"Export size (~{estimated_gb:.1f} GB) exceeds limit of {max_size_gb} GB"
            return True, f"Export size approximately {estimated_gb:.2f} GB"
        except Exception as e:
            return False, f"Failed to validate export size: {str(e)}"

    def get_all_stock_tables(self) -> List[Tuple[str, str]]:
        return self.db.get_stock_tables()

    def delete_table(self, table_name: str):
        self.db.drop_table(table_name)

    def delete_all_tables(self) -> int:
        return self.db.drop_all_stock_tables()

    def get_tables_for_ticker(self, symbol: str, timespan: str = None, range_val: str = None) -> List[str]:
        return self.db.get_tables_for_ticker(symbol, timespan, range_val)

    # Projections API (unchanged)
    def insert_projections(self, table_name: str, rows_df, strategy_name: str) -> int:
        try:
            import pandas as pd
            if rows_df is None:
                return 0
            if not isinstance(rows_df, pd.DataFrame):
                rows_df = pd.DataFrame(rows_df)
            self.db.ensure_projection_columns(table_name)
            rows = rows_df.copy()
            rows['is_projection'] = True
            rows['projection_source'] = strategy_name
            affected = self.db.bulk_insert_dataframe(table_name, rows)
            k2_logger.database_operation("Projections inserted", f"{affected} rows into {table_name} for {strategy_name}")
            return affected
        except Exception as e:
            k2_logger.error(f"insert_projections failed: {str(e)}", "DB")
            raise

    def delete_projections(self, table_name: str, strategy_name: str) -> int:
        try:
            affected = self.db.delete_where(table_name, "is_projection = TRUE AND projection_source = %s", [strategy_name])
            k2_logger.database_operation("Projections deleted", f"{affected} rows from {table_name} for {strategy_name}")
            return affected
        except Exception as e:
            k2_logger.error(f"delete_projections failed: {str(e)}", "DB")
            raise

    # Indicator persistence and data access (unchanged)
    def get_full_dataframe(self, table_name: str):
        try:
            import pandas as pd
            df = self.db.fetch_dataframe(table_name)
            return df
        except Exception as e:
            k2_logger.error(f"fetch_dataframe failed: {str(e)}", "DB")
            return None

    def ensure_indicator_column(self, table_name: str, column_name: str, sql_type: str = "NUMERIC"):
        try:
            self.db.ensure_indicator_column(table_name, column_name, sql_type)
        except Exception as e:
            k2_logger.error(f"ensure_indicator_column failed: {str(e)}", "DB")
            raise

    def update_indicator_column(self, table_name: str, column_name: str, ts_series, val_series) -> int:
        try:
            affected = self.db.bulk_update_column_by_timestamp(table_name, column_name, ts_series, val_series)
            k2_logger.database_operation("Indicator column updated", f"{column_name} on {table_name}: {affected} rows")
            return affected
        except Exception as e:
            k2_logger.error(f"update_indicator_column failed: {str(e)}", "DB")
            raise

    # ADD: drop indicator column wrapper
    def drop_indicator_column(self, table_name: str, column_name: str) -> bool:
        try:
            return self.db.drop_indicator_column(table_name, column_name)
        except Exception as e:
            k2_logger.error(f"drop_indicator_column failed: {str(e)}", "DB")
            return False

    def get_preset_range_df(self, table_name: str, preset: str) -> pd.DataFrame:
        days_map = {'5D': 5, '1M': 30, '3M': 90, '6M': 180, '1Y': 365, '5Y': 1825}
        if preset == 'All':
            return self.db.fetch_dataframe(table_name)
        days = days_map.get(preset, 7)
        end = datetime.now()
        start = end - timedelta(days=days)
        return self.db.fetch_time_window_df(table_name, start, end)

    def get_time_window_df(self, table_name: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self.db.fetch_time_window_df(table_name, start, end)

    def get_older_chunk_df(self, table_name: str, before_timestamp: int, limit: int = 5000) -> pd.DataFrame:
        return self.db.fetch_older_chunk_df(table_name, before_timestamp, limit)

    # ADD: display fetch with indicators and pretty headers
    def get_display_data_with_indicators(self, table_name: str, limit: int = 1000) -> Tuple[List[Tuple], List[str], int]:
        """
        Return (rows, columns, total) with nicely formatted headers for indicator columns.
        Base headers: ['Date','Time','Open','High','Low','Close','Volume','VWAP'] + formatted indicators.
        """
        try:
            indicator_cols = self.db.get_indicator_columns(table_name)
            rows, total = self.db.fetch_display_data_with_indicators(table_name, limit)

            base_cols = ["Date", "Time", "Open", "High", "Low", "Close", "Volume", "VWAP"]

            # Pretty names for indicator headers
            formatted = []
            for col in indicator_cols:
                if col.startswith('rsi_timeperiod_'):
                    period = col.split('_')[-1]
                    formatted.append(f"RSI({period})")
                elif col.startswith('macd_'):
                    if col.endswith('_line'):
                        formatted.append("MACD Line")
                    elif col.endswith('_signal'):
                        formatted.append("MACD Signal")
                    elif col.endswith('_hist'):
                        formatted.append("MACD Hist")
                    else:
                        formatted.append(col.replace('_', ' ').title())
                elif col.startswith('bbands_') or col.startswith('bollinger'):
                    if col.endswith('_upper'):
                        formatted.append("BB Upper")
                    elif col.endswith('_middle'):
                        formatted.append("BB Middle")
                    elif col.endswith('_lower'):
                        formatted.append("BB Lower")
                    else:
                        formatted.append("Bollinger")
                elif col.startswith('stoch_'):
                    if col.endswith('_k'):
                        formatted.append("Stoch %K")
                    elif col.endswith('_d'):
                        formatted.append("Stoch %D")
                    else:
                        formatted.append("Stoch")
                else:
                    parts = col.split('_')
                    if len(parts) >= 3 and parts[-1].isdigit():
                        name = parts[0].upper()
                        period = parts[-1]
                        formatted.append(f"{name}({period})")
                    else:
                        formatted.append(col.replace('_', ' ').title())

            columns = base_cols + formatted

            k2_logger.database_operation(
                "Display fetch (+indicators)",
                f"table={table_name} cols={len(columns)} indicators={formatted}"
            )
            return rows, columns, total
        except Exception as e:
            k2_logger.error(f"get_display_data_with_indicators failed: {e}", "DB")
            return [], [], 0

    # ADD: expose repair/base helpers
    def ensure_base_columns(self, table_name: str) -> None:
        try:
            self.db.ensure_base_columns(table_name)
        except Exception as e:
            k2_logger.error(f"ensure_base_columns failed: {e}", "DB")

    def has_column(self, table_name: str, column_name: str) -> bool:
        try:
            return self.db._check_column_exists(table_name, column_name)
        except Exception:
            return False


stock_service = StockService()