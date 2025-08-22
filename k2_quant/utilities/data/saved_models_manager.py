"""
Saved Models Manager (PostgreSQL-only, compatibility API)

Single source of truth in PostgreSQL. Provides:
- Registry of saved models (saved_models table)
- Per-model UI state (model_state table)
- Compatibility methods used by page/components:
  - get_model_metadata(table_name)
  - get_model_state(table_name)
  - set_model_state(table_name, indicators, active_strategy, chart_range)

Exposes singleton: saved_models_manager
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from psycopg2.extras import RealDictCursor

from k2_quant.utilities.data.db_manager import db_manager
from k2_quant.utilities.logger import k2_logger


class SavedModelsManager:
	"""PostgreSQL-backed manager for saved models and per-model UI state."""

	def __init__(self):
		self.db = db_manager
		self._ensure_tables_exist()

	def _ensure_tables_exist(self) -> None:
		"""Create required tables if they don't exist."""
		try:
			with self.db.get_connection() as conn:
				with self.db.get_cursor(conn) as cur:
					# Registry of saved models
					cur.execute("""
						CREATE TABLE IF NOT EXISTS saved_models (
							id SERIAL PRIMARY KEY,
							table_name VARCHAR(255) UNIQUE NOT NULL,
							symbol VARCHAR(32) NOT NULL,
							timespan VARCHAR(20) NOT NULL,
							range_val VARCHAR(20) NOT NULL,
							frequency VARCHAR(20) NOT NULL,
							market_hours_only BOOLEAN DEFAULT FALSE,
							record_count INTEGER,
							date_range_start TIMESTAMP,
							date_range_end TIMESTAMP,
							custom_name VARCHAR(255),
							created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
						)
					""")
					# Per-model UI state
					cur.execute("""
						CREATE TABLE IF NOT EXISTS model_state (
							id SERIAL PRIMARY KEY,
							table_name VARCHAR(255) UNIQUE NOT NULL,
							indicators_json TEXT DEFAULT '{}',
							active_strategy TEXT,
							chart_range TEXT,
							updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
						)
					""")
					conn.commit()
			k2_logger.info("Saved models tables ready", "SAVED_MODELS")
		except Exception as e:
			k2_logger.error(f"Failed ensuring saved_models/model_state tables: {e}", "SAVED_MODELS")

	# -------- Registry APIs --------

	def save_model(self, model_data: Dict[str, Any]) -> bool:
		"""
		Upsert a saved model entry.

		Required keys:
			- table_name, symbol, timespan, range_val, frequency
		Optional keys:
			- market_hours_only (bool), custom_name (str)
		Other fields (record_count, date_range) are derived.
		"""
		try:
			stats = self.db.get_table_statistics(model_data["table_name"])
			date_start, date_end = self.db.get_date_range(model_data["table_name"])
			with self.db.get_connection() as conn:
				with self.db.get_cursor(conn) as cur:
					cur.execute("""
						INSERT INTO saved_models
						(table_name, symbol, timespan, range_val, frequency,
						 market_hours_only, record_count, date_range_start, date_range_end, custom_name)
						VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
						ON CONFLICT (table_name) DO UPDATE SET
							symbol = EXCLUDED.symbol,
							timespan = EXCLUDED.timespan,
							range_val = EXCLUDED.range_val,
							frequency = EXCLUDED.frequency,
							market_hours_only = EXCLUDED.market_hours_only,
							record_count = EXCLUDED.record_count,
							date_range_start = EXCLUDED.date_range_start,
							date_range_end = EXCLUDED.date_range_end,
							custom_name = EXCLUDED.custom_name,
							created_at = CURRENT_TIMESTAMP
					""", (
						model_data["table_name"],
						model_data["symbol"],
						model_data.get("timespan", ""),
						model_data.get("range_val", ""),
						model_data.get("frequency", ""),
						model_data.get("market_hours_only", False),
						stats.get("total_records", 0),
						date_start, date_end,
						model_data.get("custom_name"),
					))
					conn.commit()
			k2_logger.info(f"Model saved: {model_data['table_name']}", "SAVED_MODELS")
			return True
		except Exception as e:
			k2_logger.error(f"Failed to save model: {e}", "SAVED_MODELS")
			return False

	def get_saved_models(self) -> List[Dict[str, Any]]:
		"""Return all saved models, with computed display_name and size."""
		try:
			with self.db.get_connection() as conn:
				with self.db.get_cursor(conn, RealDictCursor) as cur:
					cur.execute("""
						SELECT
							sm.*,
							pg_size_pretty(pg_total_relation_size(sm.table_name::regclass)) AS size
						FROM saved_models sm
						WHERE EXISTS (
							SELECT 1 FROM information_schema.tables
							WHERE table_name = sm.table_name
						)
						ORDER BY sm.created_at DESC
					""")
					rows = cur.fetchall()

			result: List[Dict[str, Any]] = []
			for r in rows:
				display_name = r.get("custom_name") or f"{r['symbol']} - {str(r['range_val']).upper()}"
				result.append({
					"table_name": r["table_name"],
					"display_name": display_name,
					"symbol": r["symbol"],
					"timespan": r["timespan"],
					"range": r["range_val"],
					"record_count": r.get("record_count"),
					"size": r.get("size"),
					"date_range": (r.get("date_range_start"), r.get("date_range_end")),
				})
			k2_logger.info(f"Retrieved {len(result)} saved models", "SAVED_MODELS")
			return result
		except Exception as e:
			k2_logger.error(f"Failed to get saved models: {e}", "SAVED_MODELS")
			return []

	def is_model_saved(self, table_name: str) -> bool:
		try:
			with self.db.get_connection() as conn:
				with self.db.get_cursor(conn) as cur:
					cur.execute("SELECT EXISTS (SELECT 1 FROM saved_models WHERE table_name=%s)", (table_name,))
					return bool(cur.fetchone()[0])
		except Exception as e:
			k2_logger.error(f"is_model_saved failed: {e}", "SAVED_MODELS")
			return False

	def unsave_model(self, table_name: str, delete_table: bool = False) -> bool:
		"""Remove registry entry; optionally drop the underlying data table."""
		try:
			with self.db.get_connection() as conn:
				with self.db.get_cursor(conn) as cur:
					cur.execute("DELETE FROM saved_models WHERE table_name=%s", (table_name,))
					cur.execute("DELETE FROM model_state WHERE table_name=%s", (table_name,))
					if delete_table:
						cur.execute(f"DROP TABLE IF EXISTS {table_name}")
				conn.commit()
			k2_logger.info(f"Model unsaved: {table_name}", "SAVED_MODELS")
			return True
		except Exception as e:
			k2_logger.error(f"unsave_model failed: {e}", "SAVED_MODELS")
			return False

	def update_model_name(self, table_name: str, custom_name: str) -> bool:
		try:
			with self.db.get_connection() as conn:
				with self.db.get_cursor(conn) as cur:
					cur.execute("UPDATE saved_models SET custom_name=%s WHERE table_name=%s",
						(custom_name, table_name))
				conn.commit()
			k2_logger.info(f"Updated model name: {table_name} -> {custom_name}", "SAVED_MODELS")
			return True
		except Exception as e:
			k2_logger.error(f"update_model_name failed: {e}", "SAVED_MODELS")
			return False

	def get_model_by_table(self, table_name: str) -> Optional[Dict[str, Any]]:
		"""Return one saved model record by table name (or None)."""
		try:
			with self.db.get_connection() as conn:
				with self.db.get_cursor(conn, RealDictCursor) as cur:
					cur.execute("""
						SELECT
							sm.*,
							pg_size_pretty(pg_total_relation_size(sm.table_name::regclass)) AS size
						FROM saved_models sm
						WHERE sm.table_name = %s
					""", (table_name,))
					row = cur.fetchone()
			return dict(row) if row else None
		except Exception as e:
			k2_logger.error(f"get_model_by_table failed: {e}", "SAVED_MODELS")
			return None

	def cleanup_orphaned_entries(self) -> int:
		"""Delete registry rows whose tables no longer exist."""
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
					deleted = cur.rowcount or 0
				conn.commit()
			if deleted:
				k2_logger.info(f"Cleaned up {deleted} orphaned saved_models", "SAVED_MODELS")
			return deleted
		except Exception as e:
			k2_logger.error(f"cleanup_orphaned_entries failed: {e}", "SAVED_MODELS")
			return 0

	# -------- Compatibility APIs used by page/components --------

	def get_model_metadata(self, table_name: str) -> Dict[str, Any]:
		"""
		Return metadata dict expected by page:
		  - symbol (str)
		  - timespan (str)
		  - range (str)
		  - records (int)
		  - size (str)
		  - date_range (Tuple[Optional[datetime], Optional[datetime]])
		  - display_name (str, optional)
		"""
		try:
			row = self.get_model_by_table(table_name)
			if row:
				# If fetched via RealDictCursor, ensure keys match
				symbol = row.get("symbol")
				timespan = row.get("timespan")
				range_val = row.get("range_val") or row.get("range")
				record_count = row.get("record_count")
				size = row.get("size")
				date_start = row.get("date_range_start")
				date_end = row.get("date_range_end")
				display_name = row.get("custom_name") or (f"{symbol} - {str(range_val).upper()}" if symbol else table_name)
				return {
					"symbol": symbol,
					"timespan": timespan,
					"range": range_val,
					"records": record_count,
					"size": size,
					"date_range": (date_start, date_end),
					"display_name": display_name,
				}
			# Fallback: derive from DB
			stats = self.db.get_table_statistics(table_name)
			date_start, date_end = self.db.get_date_range(table_name)
			# Best-effort parse from table name
			parts = table_name.split("_")
			symbol = parts[1].upper() if len(parts) > 1 else "UNKNOWN"
			timespan = parts[2] if len(parts) > 2 else ""
			range_val = parts[3] if len(parts) > 3 else ""
			return {
				"symbol": symbol,
				"timespan": timespan,
				"range": range_val,
				"records": stats.get("total_records", 0),
				"size": stats.get("size"),
				"date_range": (date_start, date_end),
				"display_name": f"{symbol} - {str(range_val).upper()}" if symbol and range_val else table_name,
			}
		except Exception as e:
			k2_logger.error(f"get_model_metadata failed: {e}", "SAVED_MODELS")
			return {
				"symbol": "UNKNOWN",
				"timespan": "",
				"range": "",
				"records": 0,
				"size": "0 MB",
				"date_range": (None, None),
				"display_name": table_name,
			}

	def get_model_state(self, table_name: str) -> Dict[str, Any]:
		"""
		Return per-model UI state:
		  { "indicators": dict, "active_strategy": Optional[str], "chart_range": Optional[str] }
		"""
		try:
			with self.db.get_connection() as conn:
				with self.db.get_cursor(conn, RealDictCursor) as cur:
					cur.execute("""
						SELECT indicators_json, active_strategy, chart_range
						FROM model_state WHERE table_name = %s
					""", (table_name,))
					row = cur.fetchone()
			if not row:
				return {"indicators": {}, "active_strategy": None, "chart_range": None}
			import json
			return {
				"indicators": json.loads(row.get("indicators_json") or "{}"),
				"active_strategy": row.get("active_strategy"),
				"chart_range": row.get("chart_range"),
			}
		except Exception as e:
			k2_logger.error(f"get_model_state failed: {e}", "SAVED_MODELS")
			return {"indicators": {}, "active_strategy": None, "chart_range": None}

	def set_model_state(
		self,
		table_name: str,
		indicators: Optional[Dict[str, Any]],
		active_strategy: Optional[str],
		chart_range: Optional[str],
	) -> bool:
		"""Upsert per-model UI state."""
		try:
			import json
			with self.db.get_connection() as conn:
				with self.db.get_cursor(conn) as cur:
					cur.execute("""
						INSERT INTO model_state (table_name, indicators_json, active_strategy, chart_range, updated_at)
						VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
						ON CONFLICT (table_name) DO UPDATE SET
						  indicators_json = EXCLUDED.indicators_json,
						  active_strategy = EXCLUDED.active_strategy,
						  chart_range = EXCLUDED.chart_range,
						  updated_at = CURRENT_TIMESTAMP
					""", (
						table_name,
						json.dumps(indicators or {}),
						active_strategy,
						chart_range,
					))
				conn.commit()
			return True
		except Exception as e:
			k2_logger.error(f"set_model_state failed: {e}", "SAVED_MODELS")
			return False


# Singleton instance
saved_models_manager = SavedModelsManager()