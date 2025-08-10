"""
Strategy Service for K2 Quant

Manages custom trading strategies with database persistence.
Stores both code and metadata for complex strategies.
"""

import json
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime
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
            
            # Create execution history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id INTEGER,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    timeframe TEXT,
                    input_records INTEGER,
                    output_records INTEGER,
                    execution_time REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    metrics TEXT,
                    FOREIGN KEY (strategy_id) REFERENCES strategies (id)
                )
            """)
            
            # Create AI conversations table for strategy development
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id INTEGER,
                    conversation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (strategy_id) REFERENCES strategies (id)
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
                            category = ?, updated_at = CURRENT_TIMESTAMP
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
    
    def get_strategies_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get strategies by category"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM strategies 
                    WHERE category = ? AND is_active = 1
                    ORDER BY name
                """, (category,))
                
                strategies = []
                for row in cursor.fetchall():
                    strategy = dict(row)
                    strategy['parameters'] = json.loads(strategy['parameters'])
                    strategies.append(strategy)
                
                return strategies
                
        except Exception as e:
            k2_logger.error(f"Failed to get strategies by category: {str(e)}", "STRATEGY")
            return []
    
    def record_execution(self, strategy_name: str, symbol: str, timeframe: str,
                        input_records: int, output_records: int,
                        execution_time: float, success: bool,
                        error_message: str = None, metrics: Dict[str, Any] = None):
        """Record strategy execution history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get strategy ID
                cursor.execute("SELECT id FROM strategies WHERE name = ?", (strategy_name,))
                result = cursor.fetchone()
                if not result:
                    return
                
                strategy_id = result[0]
                metrics_json = json.dumps(metrics) if metrics else "{}"
                
                # Insert execution record
                cursor.execute("""
                    INSERT INTO execution_history 
                    (strategy_id, symbol, timeframe, input_records, output_records,
                     execution_time, success, error_message, metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (strategy_id, symbol, timeframe, input_records, output_records,
                      execution_time, success, error_message, metrics_json))
                
                # Update strategy execution count and last executed
                cursor.execute("""
                    UPDATE strategies 
                    SET execution_count = execution_count + 1,
                        last_executed = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (strategy_id,))
                
                conn.commit()
                k2_logger.info(f"Recorded execution for strategy: {strategy_name}", "STRATEGY")
                
        except Exception as e:
            k2_logger.error(f"Failed to record execution: {str(e)}", "STRATEGY")
    
    def save_ai_conversation(self, strategy_name: str, conversation: List[Dict[str, str]]):
        """Save AI conversation history for a strategy"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get strategy ID
                cursor.execute("SELECT id FROM strategies WHERE name = ?", (strategy_name,))
                result = cursor.fetchone()
                if not result:
                    return
                
                strategy_id = result[0]
                conversation_json = json.dumps(conversation)
                
                cursor.execute("""
                    INSERT INTO ai_conversations (strategy_id, conversation)
                    VALUES (?, ?)
                """, (strategy_id, conversation_json))
                
                conn.commit()
                k2_logger.info(f"Saved AI conversation for strategy: {strategy_name}", "STRATEGY")
                
        except Exception as e:
            k2_logger.error(f"Failed to save AI conversation: {str(e)}", "STRATEGY")
    
    def get_execution_history(self, strategy_name: str = None, 
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history for a strategy or all strategies"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if strategy_name:
                    cursor.execute("""
                        SELECT eh.*, s.name as strategy_name
                        FROM execution_history eh
                        JOIN strategies s ON eh.strategy_id = s.id
                        WHERE s.name = ?
                        ORDER BY eh.executed_at DESC
                        LIMIT ?
                    """, (strategy_name, limit))
                else:
                    cursor.execute("""
                        SELECT eh.*, s.name as strategy_name
                        FROM execution_history eh
                        JOIN strategies s ON eh.strategy_id = s.id
                        ORDER BY eh.executed_at DESC
                        LIMIT ?
                    """, (limit,))
                
                history = []
                for row in cursor.fetchall():
                    record = dict(row)
                    record['metrics'] = json.loads(record['metrics'])
                    history.append(record)
                
                return history
                
        except Exception as e:
            k2_logger.error(f"Failed to get execution history: {str(e)}", "STRATEGY")
            return []
    
    def delete_strategy(self, name: str) -> bool:
        """Soft delete a strategy (mark as inactive)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE strategies SET is_active = 0 WHERE name = ?
                """, (name,))
                
                conn.commit()
                k2_logger.info(f"Strategy deleted: {name}", "STRATEGY")
                return True
                
        except Exception as e:
            k2_logger.error(f"Failed to delete strategy: {str(e)}", "STRATEGY")
            return False
    
    def export_strategy(self, name: str, filepath: str):
        """Export strategy to file"""
        strategy = self.get_strategy(name)
        if strategy:
            with open(filepath, 'w') as f:
                json.dump(strategy, f, indent=2, default=str)
            k2_logger.info(f"Strategy exported to {filepath}", "STRATEGY")
    
    def import_strategy(self, filepath: str) -> bool:
        """Import strategy from file"""
        try:
            with open(filepath, 'r') as f:
                strategy = json.load(f)
            
            return self.save_strategy(
                name=strategy['name'],
                code=strategy['code'],
                description=strategy.get('description', ''),
                parameters=strategy.get('parameters', {}),
                category=strategy.get('category', 'imported')
            )
            
        except Exception as e:
            k2_logger.error(f"Failed to import strategy: {str(e)}", "STRATEGY")
            return False
    
    def get_performance_summary(self, strategy_name: str) -> Dict[str, Any]:
        """Get performance summary for a strategy"""
        history = self.get_execution_history(strategy_name, limit=1000)
        
        if not history:
            return {}
        
        total_executions = len(history)
        successful = sum(1 for h in history if h['success'])
        failed = total_executions - successful
        avg_time = sum(h['execution_time'] for h in history) / total_executions
        
        return {
            'total_executions': total_executions,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / total_executions) * 100,
            'average_execution_time': avg_time,
            'last_executed': history[0]['executed_at'] if history else None
        }


# Singleton instance
strategy_service = StrategyService()