"""
Saved Models Manager

Authoritative catalog of saved models and per-model UI state for Analysis.
Backed by a lightweight SQLite database at data/saved_models.db.

Exposes a singleton `saved_models_manager` with methods used across the app:
- save_model(model_dict)
- unsave_model(table_name, delete_table: bool = False)
- is_model_saved(table_name)
- get_saved_models()
- get_model_state(table_name)
- set_model_state(table_name, indicators, active_strategy, chart_range)
"""

from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class _SavedModelsManager:
    def __init__(self) -> None:
        self._db_path = Path("data") / "saved_models.db"
        self._db_path.parent.mkdir(exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS saved_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT UNIQUE NOT NULL,
                    display_name TEXT,
                    symbol TEXT,
                    timespan TEXT,
                    range_val TEXT,
                    record_count INTEGER DEFAULT 0,
                    size TEXT,
                    date_start TEXT,
                    date_end TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS model_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT UNIQUE NOT NULL,
                    indicators_json TEXT DEFAULT '{}',
                    active_strategy TEXT,
                    chart_range TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    # Catalog APIs
    def save_model(self, model: Dict[str, Any]) -> bool:
        """Insert or update a model entry in the catalog.

        Expected keys: table_name (required); display_name, symbol, timespan,
        range_val, record_count, size, date_start, date_end (optional).
        """
        with self._conn() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO saved_models
                (table_name, display_name, symbol, timespan, range_val, record_count, size, date_start, date_end, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(
                    (SELECT created_at FROM saved_models WHERE table_name=?), CURRENT_TIMESTAMP
                ))
                """,
                (
                    model["table_name"],
                    model.get("display_name"),
                    model.get("symbol"),
                    model.get("timespan"),
                    model.get("range_val"),
                    int(model.get("record_count", 0)),
                    model.get("size"),
                    str(model.get("date_start") or ""),
                    str(model.get("date_end") or ""),
                    model["table_name"],
                ),
            )
        return True

    def unsave_model(self, table_name: str, delete_table: bool = False) -> bool:
        """Remove model from catalog and optionally drop the underlying table."""
        with self._conn() as c:
            c.execute("DELETE FROM saved_models WHERE table_name=?", (table_name,))
            c.execute("DELETE FROM model_state WHERE table_name=?", (table_name,))

        if delete_table:
            try:
                # Defer import to avoid circular dependencies at module import time
                from k2_quant.utilities.services.stock_data_service import stock_service

                stock_service.delete_table(table_name)
            except Exception:
                # Non-fatal if table drop fails here; UI will surface errors via normal paths
                pass
        return True

    def is_model_saved(self, table_name: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM saved_models WHERE table_name=?", (table_name,)
            ).fetchone()
            return row is not None

    def get_saved_models(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT table_name, display_name, symbol, timespan,
                       range_val as range, record_count, size, date_start, date_end
                FROM saved_models
                ORDER BY created_at DESC
                """
            ).fetchall()
            models: List[Dict[str, Any]] = []
            for r in rows:
                models.append(
                    {
                        "table_name": r["table_name"],
                        "display_name": r["display_name"],
                        "symbol": r["symbol"],
                        "timespan": r["timespan"],
                        "range": r["range"],
                        "record_count": r["record_count"],
                        "size": r["size"],
                        "date_range": (r["date_start"], r["date_end"]),
                    }
                )
            return models

    # Per-model state APIs
    def get_model_state(self, table_name: str) -> Dict[str, Any]:
        with self._conn() as c:
            r = c.execute(
                """
                SELECT indicators_json, active_strategy, chart_range
                FROM model_state WHERE table_name=?
                """,
                (table_name,),
            ).fetchone()
            if not r:
                return {"indicators": {}, "active_strategy": None, "chart_range": None}
            return {
                "indicators": json.loads(r["indicators_json"] or "{}"),
                "active_strategy": r["active_strategy"],
                "chart_range": r["chart_range"],
            }

    def set_model_state(
        self,
        table_name: str,
        indicators: Dict[str, Any],
        active_strategy: Optional[str],
        chart_range: Optional[str],
    ) -> bool:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO model_state (table_name, indicators_json, active_strategy, chart_range, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(table_name) DO UPDATE SET
                  indicators_json=excluded.indicators_json,
                  active_strategy=excluded.active_strategy,
                  chart_range=excluded.chart_range,
                  updated_at=CURRENT_TIMESTAMP
                """,
                (table_name, json.dumps(indicators or {}), active_strategy, chart_range),
            )
        return True


# Singleton instance to match existing import style
saved_models_manager = _SavedModelsManager()

"""
Saved Models Manager for K2 Quant

Manages the saved_models table in PostgreSQL to track which data tables
are marked as "saved models" for use in the Analysis page.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

from k2_quant.utilities.data.db_manager import db_manager
from k2_quant.utilities.logger import k2_logger


class SavedModelsManager:
    """Manager for saved models registry"""
    
    def __init__(self):
        self.db = db_manager
        self.ensure_table_exists()
    
    def ensure_table_exists(self):
        """Create saved_models table if it doesn't exist"""
        try:
            with self.db.get_connection() as conn:
                with self.db.get_cursor(conn) as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS saved_models (
                            id SERIAL PRIMARY KEY,
                            table_name VARCHAR(255) UNIQUE NOT NULL,
                            symbol VARCHAR(10) NOT NULL,
                            timespan VARCHAR(20) NOT NULL,
                            range_val VARCHAR(10) NOT NULL,
                            frequency VARCHAR(10) NOT NULL,
                            market_hours_only BOOLEAN DEFAULT FALSE,
                            record_count INTEGER,
                            date_range_start TIMESTAMP,
                            date_range_end TIMESTAMP,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            notes TEXT,
                            custom_name VARCHAR(255)
                        )
                    """)
                    conn.commit()
                    k2_logger.info("Saved models table ready", "SAVED_MODELS")
        except Exception as e:
            k2_logger.error(f"Failed to create saved_models table: {str(e)}", "SAVED_MODELS")
    
    def save_model(self, model_data: Dict[str, Any]) -> bool:
        """
        Save a model to the registry
        
        Args:
            model_data: Dictionary containing:
                - table_name: Name of the PostgreSQL table
                - symbol: Stock symbol
                - timespan: Time span (minute, hour, day, etc.)
                - range_val: Range value (1M, 1Y, etc.)
                - frequency: Frequency (1min, 5min, D, etc.)
                - market_hours_only: Boolean for market hours filter
                - record_count: Number of records (optional)
                - custom_name: Custom name for the model (optional)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get additional metadata from the table
            stats = self.db.get_table_statistics(model_data['table_name'])
            date_range = self.db.get_date_range(model_data['table_name'])
            
            with self.db.get_connection() as conn:
                with self.db.get_cursor(conn) as cur:
                    cur.execute("""
                        INSERT INTO saved_models 
                        (table_name, symbol, timespan, range_val, frequency, 
                         market_hours_only, record_count, date_range_start, 
                         date_range_end, custom_name)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (table_name) DO UPDATE SET
                            custom_name = EXCLUDED.custom_name,
                            created_at = CURRENT_TIMESTAMP
                        RETURNING id
                    """, (
                        model_data['table_name'],
                        model_data['symbol'],
                        model_data.get('timespan', ''),
                        model_data.get('range_val', ''),
                        model_data.get('frequency', ''),
                        model_data.get('market_hours_only', False),
                        stats.get('total_records', 0),
                        date_range[0] if date_range else None,
                        date_range[1] if date_range else None,
                        model_data.get('custom_name')
                    ))
                    
                    model_id = cur.fetchone()[0]
                    conn.commit()
                    
                    k2_logger.info(
                        f"Model saved: {model_data['table_name']} (ID: {model_id})", 
                        "SAVED_MODELS"
                    )
                    return True
                    
        except Exception as e:
            k2_logger.error(f"Failed to save model: {str(e)}", "SAVED_MODELS")
            return False
    
    def get_saved_models(self) -> List[Dict[str, Any]]:
        """
        Get all saved models with their metadata
        
        Returns:
            List of dictionaries containing model information
        """
        try:
            with self.db.get_connection() as conn:
                with self.db.get_cursor(conn, RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            sm.*,
                            pg_size_pretty(pg_total_relation_size(sm.table_name::regclass)) as size
                        FROM saved_models sm
                        WHERE EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_name = sm.table_name
                        )
                        ORDER BY sm.created_at DESC
                    """)
                    
                    models = cur.fetchall()
                    
                    # Convert to regular dicts and format
                    result = []
                    for model in models:
                        model_dict = dict(model)
                        # Use custom_name if available, otherwise generate display name
                        if not model_dict.get('custom_name'):
                            model_dict['display_name'] = f"{model_dict['symbol']} - {model_dict['range_val'].upper()}"
                        else:
                            model_dict['display_name'] = model_dict['custom_name']
                        result.append(model_dict)
                    
                    k2_logger.info(f"Retrieved {len(result)} saved models", "SAVED_MODELS")
                    return result
                    
        except Exception as e:
            k2_logger.error(f"Failed to get saved models: {str(e)}", "SAVED_MODELS")
            return []
    
    def is_model_saved(self, table_name: str) -> bool:
        """Check if a table is marked as a saved model"""
        try:
            with self.db.get_connection() as conn:
                with self.db.get_cursor(conn) as cur:
                    cur.execute(
                        "SELECT EXISTS(SELECT 1 FROM saved_models WHERE table_name = %s)",
                        (table_name,)
                    )
                    return cur.fetchone()[0]
        except Exception as e:
            k2_logger.error(f"Failed to check if model is saved: {str(e)}", "SAVED_MODELS")
            return False
    
    def unsave_model(self, table_name: str, delete_table: bool = False) -> bool:
        """
        Remove a model from the saved models registry
        
        Args:
            table_name: Name of the table to unsave
            delete_table: If True, also delete the underlying table
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.db.get_connection() as conn:
                with self.db.get_cursor(conn) as cur:
                    # Remove from saved_models
                    cur.execute(
                        "DELETE FROM saved_models WHERE table_name = %s",
                        (table_name,)
                    )
                    
                    # Optionally delete the table
                    if delete_table:
                        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                        k2_logger.info(f"Deleted table: {table_name}", "SAVED_MODELS")
                    
                    conn.commit()
                    k2_logger.info(f"Model unsaved: {table_name}", "SAVED_MODELS")
                    return True
                    
        except Exception as e:
            k2_logger.error(f"Failed to unsave model: {str(e)}", "SAVED_MODELS")
            return False
    
    def update_model_name(self, table_name: str, custom_name: str) -> bool:
        """Update the custom name of a saved model"""
        try:
            with self.db.get_connection() as conn:
                with self.db.get_cursor(conn) as cur:
                    cur.execute("""
                        UPDATE saved_models 
                        SET custom_name = %s
                        WHERE table_name = %s
                    """, (custom_name, table_name))
                    
                    conn.commit()
                    k2_logger.info(f"Updated model name: {table_name} -> {custom_name}", "SAVED_MODELS")
                    return True
                    
        except Exception as e:
            k2_logger.error(f"Failed to update model name: {str(e)}", "SAVED_MODELS")
            return False
    
    def get_model_by_table(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific saved model by table name"""
        try:
            with self.db.get_connection() as conn:
                with self.db.get_cursor(conn, RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            sm.*,
                            pg_size_pretty(pg_total_relation_size(sm.table_name::regclass)) as size
                        FROM saved_models sm
                        WHERE sm.table_name = %s
                    """, (table_name,))
                    
                    result = cur.fetchone()
                    return dict(result) if result else None
                    
        except Exception as e:
            k2_logger.error(f"Failed to get model by table: {str(e)}", "SAVED_MODELS")
            return None
    
    def cleanup_orphaned_entries(self) -> int:
        """Remove saved_models entries where the table no longer exists"""
        try:
            with self.db.get_connection() as conn:
                with self.db.get_cursor(conn) as cur:
                    cur.execute("""
                        DELETE FROM saved_models
                        WHERE NOT EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_name = saved_models.table_name
                        )
                    """)
                    
                    deleted_count = cur.rowcount
                    conn.commit()
                    
                    if deleted_count > 0:
                        k2_logger.info(f"Cleaned up {deleted_count} orphaned entries", "SAVED_MODELS")
                    
                    return deleted_count
                    
        except Exception as e:
            k2_logger.error(f"Failed to cleanup orphaned entries: {str(e)}", "SAVED_MODELS")
            return 0


# Singleton instance
saved_models_manager = SavedModelsManager()