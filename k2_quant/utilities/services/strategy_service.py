"""
Strategy Service for K2 Quant

Manages custom trading strategies with database persistence.
Stores both code and metadata for complex strategies.

Lifecycle invariants (deletion, outputs sync): see
``k2_quant.utilities.services.strategy_lifecycle_rules``.
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.strategy_validator import validate_strategy


class StrategyService(QObject):
    """Service for managing custom trading strategies.

    Emits ``strategies_changed`` after any save/rename/delete so every
    open ``LeftPaneWidget`` (Analysis or Stream) can repopulate from the
    single source of truth without page-to-page wiring.
    """

    strategies_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.db_path = Path("data/strategies.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self.initialize_database()
    
    def initialize_database(self):
        """Create strategies database and tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create strategies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    code TEXT NOT NULL,
                    parameters TEXT,
                    category TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    execution_count INTEGER DEFAULT 0,
                    last_executed TIMESTAMP,
                    performance_metrics TEXT,
                    is_active BOOLEAN DEFAULT 1
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    model_table TEXT,
                    run_timestamp TEXT NOT NULL,
                    success INTEGER NOT NULL DEFAULT 0,
                    execution_time_ms REAL,
                    stdout_output TEXT,
                    error_output TEXT,
                    tab_writes_json TEXT,
                    metrics_json TEXT,
                    code_snapshot TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_strategy
                ON strategy_runs (strategy_name, run_timestamp DESC)
            """)

            cur2 = conn.cursor()
            cur2.execute("PRAGMA table_info(strategy_runs)")
            run_cols = {row[1] for row in cur2.fetchall()}
            if "report_blocks_json" not in run_cols:
                cursor.execute(
                    "ALTER TABLE strategy_runs ADD COLUMN report_blocks_json TEXT"
                )

            conn.commit()
        
        k2_logger.info("Strategy database initialized", "STRATEGY")
    
    def save_strategy(self, name: str, code: str, description: str = "",
                     parameters: Dict[str, Any] = None,
                     category: str = "custom") -> Tuple[bool, List[str], List[str]]:
        """Save a new strategy or update existing one.

        Validates before persist. Returns (success, errors, warnings).
        errors block save; warnings are advisory (e.g. unknown frame columns).
        """
        passed, val_errors, val_warnings = validate_strategy(code)
        if not passed:
            k2_logger.warning(
                f"Strategy '{name}' save rejected by validator: {val_errors}",
                "STRATEGY",
            )
            return False, val_errors, val_warnings

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if strategy exists
                cursor.execute("SELECT id FROM strategies WHERE name = ?", (name,))
                existing = cursor.fetchone()
                
                params_json = json.dumps(parameters) if parameters else "{}"
                
                if existing:
                    # Update existing strategy
                    cursor.execute("""
                        UPDATE strategies 
                        SET code = ?, description = ?, parameters = ?, 
                            category = ?, updated_at = CURRENT_TIMESTAMP,
                            is_active = 1
                        WHERE name = ?
                    """, (code, description, params_json, category, name))
                    
                    k2_logger.info(f"Strategy updated: {name}", "STRATEGY")
                else:
                    # Insert new strategy
                    cursor.execute("""
                        INSERT INTO strategies (name, description, code, parameters, category)
                        VALUES (?, ?, ?, ?, ?)
                    """, (name, description, code, params_json, category))
                    
                    k2_logger.info(f"Strategy saved: {name}", "STRATEGY")
                
                conn.commit()
                self._emit_changed()
                return True, [], val_warnings
                
        except Exception as e:
            k2_logger.error(f"Failed to save strategy: {str(e)}", "STRATEGY")
            return False, [f"Failed to save strategy: {str(e)}"], val_warnings
    
    def get_strategy(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a strategy by name"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM strategies WHERE name = ? AND is_active = 1
                """, (name,))
                
                row = cursor.fetchone()
                if row:
                    strategy = dict(row)
                    strategy['parameters'] = json.loads(strategy['parameters'])
                    return strategy
                
        except Exception as e:
            k2_logger.error(f"Failed to get strategy: {str(e)}", "STRATEGY")
        
        return None
    
    def get_strategy_code(self, name: str) -> Optional[str]:
        """Get just the code for a strategy"""
        strategy = self.get_strategy(name)
        return strategy['code'] if strategy else None
    
    def get_all_strategies(self) -> List[Dict[str, Any]]:
        """Get all active strategies"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, name, description, category, created_at, 
                           updated_at, execution_count, last_executed
                    FROM strategies 
                    WHERE is_active = 1
                    ORDER BY name
                """)
                
                strategies = []
                for row in cursor.fetchall():
                    strategies.append(dict(row))
                
                return strategies
                
        except Exception as e:
            k2_logger.error(f"Failed to get strategies: {str(e)}", "STRATEGY")
            return []
    
    def rename_strategy(self, old_name: str, new_name: str) -> bool:
        """Rename a strategy, removing any conflicting row first"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM strategies WHERE name = ? AND name != ?",
                    (new_name, old_name),
                )
                cursor.execute(
                    "UPDATE strategies SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE name = ?",
                    (new_name, old_name),
                )
                if cursor.rowcount == 0:
                    return False
                conn.commit()
                k2_logger.info(f"Strategy renamed: {old_name} -> {new_name}", "STRATEGY")
                self._emit_changed()
                return True
        except Exception as e:
            k2_logger.error(f"Failed to rename strategy: {str(e)}", "STRATEGY")
            return False

    def duplicate_strategy(self, source_name: str, new_name: str) -> bool:
        """Copy an existing strategy under a new name"""
        try:
            source = self.get_strategy(source_name)
            if not source:
                return False
            ok, _, _ = self.save_strategy(
                new_name, source['code'],
                description=source.get('description', ''),
                category=source.get('category', 'custom'),
            )
            return ok
        except Exception as e:
            k2_logger.error(f"Failed to duplicate strategy: {str(e)}", "STRATEGY")
            return False

    def delete_strategy(self, name: str) -> bool:
        """Remove strategy and all OUTPUTS history for that name (same transaction).

        See ``strategy_lifecycle_rules.RULE 1``.
        """
        name = (name or "").strip()
        if not name:
            return False
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM strategy_runs WHERE strategy_name = ?", (name,)
                )
                runs_removed = cursor.rowcount or 0
                cursor.execute("DELETE FROM strategies WHERE name = ?", (name,))
                conn.commit()
                k2_logger.info(
                    f"Strategy deleted: {name} (removed {runs_removed} run record(s))",
                    "STRATEGY",
                )
                self._emit_changed()
                return True

        except Exception as e:
            k2_logger.error(f"Failed to delete strategy: {str(e)}", "STRATEGY")
            return False

    def _emit_changed(self) -> None:
        """Notify listeners that the strategy registry mutated.

        Wrapped so a misbehaving slot cannot break a successful DB write.
        """
        try:
            self.strategies_changed.emit()
        except Exception as e:
            k2_logger.error(
                f"strategies_changed emit failed: {e}", "STRATEGY")

    # ------------------------------------------------------------------
    # Strategy Runs
    # ------------------------------------------------------------------

    def save_run(
        self,
        strategy_name: str,
        code_snapshot: str,
        result: Dict[str, Any],
        model_table: str = "",
    ) -> Optional[int]:
        """Persist one strategy execution result.  Returns the new row id."""
        try:
            ts = datetime.now().isoformat(timespec='seconds')
            success = 1 if result.get('success') else 0
            exec_ms = (result.get('execution_time', 0)) * 1000

            tab_writes = result.get('_tab_writes', [])
            tw_safe = []
            for w in tab_writes:
                entry = {k: v for k, v in w.items() if k != 'values'}
                if 'values' in w and w['values']:
                    entry['length'] = len(w['values'])
                    entry['preview'] = w['values'][:5]
                tw_safe.append(entry)

            with sqlite3.connect(self.db_path, timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO strategy_runs
                        (strategy_name, model_table, run_timestamp, success,
                         execution_time_ms, stdout_output, error_output,
                         tab_writes_json, metrics_json, code_snapshot,
                         report_blocks_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy_name,
                    model_table or "",
                    ts,
                    success,
                    exec_ms,
                    result.get('output', ''),
                    result.get('error', ''),
                    json.dumps(tw_safe, default=str),
                    json.dumps(result.get('metrics', {}), default=str),
                    code_snapshot,
                    json.dumps(result.get('_report_blocks', []), default=str),
                ))
                conn.commit()
                run_id = cursor.lastrowid
                self._prune_excess_runs(strategy_name, conn)

            k2_logger.info(
                f"Strategy run saved: {strategy_name} (id={run_id}, ok={success})",
                "STRATEGY",
            )
            return run_id
        except Exception as e:
            k2_logger.error(f"Failed to save strategy run: {e}", "STRATEGY")
            return None

    MAX_RUNS_PER_STRATEGY = 5

    def _prune_excess_runs(self, strategy_name: str, conn: sqlite3.Connection):
        """Keep only the most recent ``MAX_RUNS_PER_STRATEGY`` runs per strategy."""
        try:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM strategy_runs
                WHERE strategy_name = ?
                  AND id NOT IN (
                      SELECT id FROM strategy_runs
                      WHERE strategy_name = ?
                      ORDER BY run_timestamp DESC
                      LIMIT ?
                  )
            """, (strategy_name, strategy_name, self.MAX_RUNS_PER_STRATEGY))
            removed = cursor.rowcount or 0
            if removed:
                conn.commit()
                k2_logger.info(
                    f"Pruned {removed} old run(s) for '{strategy_name}' "
                    f"(kept last {self.MAX_RUNS_PER_STRATEGY})",
                    "STRATEGY",
                )
        except Exception as e:
            k2_logger.error(f"_prune_excess_runs failed: {e}", "STRATEGY")

    def get_runs(
        self,
        strategy_name: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return recent runs, optionally filtered by strategy name."""
        try:
            with sqlite3.connect(self.db_path, timeout=1) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if strategy_name:
                    cursor.execute("""
                        SELECT * FROM strategy_runs
                        WHERE strategy_name = ?
                        ORDER BY run_timestamp DESC LIMIT ?
                    """, (strategy_name, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM strategy_runs
                        ORDER BY run_timestamp DESC LIMIT ?
                    """, (limit,))
                return [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            k2_logger.error(f"Failed to query strategy runs: {e}", "STRATEGY")
            return []

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single run by id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM strategy_runs WHERE id = ?", (run_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            k2_logger.error(f"Failed to get run {run_id}: {e}", "STRATEGY")
            return None

    def prune_orphan_strategy_runs(self) -> int:
        """Delete run rows whose strategy no longer exists (legacy / failed deletes).

        ``strategy_runs`` is not a foreign-key child of ``strategies``; this heals
        orphans so the OUTPUTS tree cannot list removed strategies.

        Uses a very short timeout so this never blocks the UI thread; if the DB
        is busy (e.g. a run is being saved), the prune is silently skipped —
        it will succeed on the next refresh.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=0.3) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM strategy_runs
                    WHERE strategy_name NOT IN (SELECT name FROM strategies)
                """)
                removed = cursor.rowcount or 0
                conn.commit()
                if removed:
                    k2_logger.info(
                        f"Pruned {removed} orphan strategy run row(s)",
                        "STRATEGY",
                    )
                return removed
        except Exception as e:
            k2_logger.debug(f"prune_orphan_strategy_runs skipped (DB busy): {e}", "STRATEGY")
            return 0

    def get_strategy_names_with_runs(self) -> List[str]:
        """Return distinct strategy names that have runs and still exist in ``strategies``."""
        try:
            with sqlite3.connect(self.db_path, timeout=1) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT r.strategy_name
                    FROM strategy_runs r
                    INNER JOIN strategies s ON s.name = r.strategy_name
                    ORDER BY r.strategy_name
                """)
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            k2_logger.error(f"Failed to list strategy names with runs: {e}", "STRATEGY")
            return []

    def get_all_strategy_names(self) -> List[str]:
        """Return all active strategy names (regardless of whether they have runs)."""
        try:
            with sqlite3.connect(self.db_path, timeout=1) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM strategies
                    WHERE is_active = 1
                    ORDER BY name
                """)
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            k2_logger.debug(f"get_all_strategy_names skipped (DB busy): {e}", "STRATEGY")
            return []


# Singleton instance
strategy_service = StrategyService()