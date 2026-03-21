"""
Strategy Service for K2 Quant

Manages custom trading strategies with database persistence.
Stores both code and metadata for complex strategies.
"""

import json
import sqlite3
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


# Singleton instance
strategy_service = StrategyService()