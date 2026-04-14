"""
Strategy Service for K2 Quant

Manages custom trading strategies with database persistence.
Stores both code and metadata for complex strategies.
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from k2_quant.utilities.logger import k2_logger


class StrategyService:
    """Service for managing custom trading strategies"""
    
    def __init__(self):
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

            conn.commit()
        
        k2_logger.info("Strategy database initialized", "STRATEGY")
    
    def save_strategy(self, name: str, code: str, description: str = "",
                     parameters: Dict[str, Any] = None,
                     category: str = "custom") -> bool:
        """Save a new strategy or update existing one"""
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
                return True
                
        except Exception as e:
            k2_logger.error(f"Failed to save strategy: {str(e)}", "STRATEGY")
            return False
    
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
            return self.save_strategy(
                new_name, source['code'],
                description=source.get('description', ''),
                category=source.get('category', 'custom'),
            )
        except Exception as e:
            k2_logger.error(f"Failed to duplicate strategy: {str(e)}", "STRATEGY")
            return False

    def delete_strategy(self, name: str) -> bool:
        """Hard delete a strategy from the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "DELETE FROM strategies WHERE name = ?", (name,)
                )
                
                conn.commit()
                k2_logger.info(f"Strategy deleted: {name}", "STRATEGY")
                return True
                
        except Exception as e:
            k2_logger.error(f"Failed to delete strategy: {str(e)}", "STRATEGY")
            return False

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

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO strategy_runs
                        (strategy_name, model_table, run_timestamp, success,
                         execution_time_ms, stdout_output, error_output,
                         tab_writes_json, metrics_json, code_snapshot)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ))
                conn.commit()
                run_id = cursor.lastrowid

            k2_logger.info(
                f"Strategy run saved: {strategy_name} (id={run_id}, ok={success})",
                "STRATEGY",
            )
            return run_id
        except Exception as e:
            k2_logger.error(f"Failed to save strategy run: {e}", "STRATEGY")
            return None

    def get_runs(
        self,
        strategy_name: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return recent runs, optionally filtered by strategy name."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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

    def get_strategy_names_with_runs(self) -> List[str]:
        """Return distinct strategy names that have at least one run."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT strategy_name FROM strategy_runs
                    ORDER BY strategy_name
                """)
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            k2_logger.error(f"Failed to list strategy names with runs: {e}", "STRATEGY")
            return []


# Singleton instance
strategy_service = StrategyService()