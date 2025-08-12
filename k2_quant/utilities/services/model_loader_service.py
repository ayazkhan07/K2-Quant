"""
Model Loader Service for K2 Quant - Fixed with SQLAlchemy

Manages loading and caching of saved models from PostgreSQL database.
Fixed to use SQLAlchemy for pandas.to_sql() operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from k2_quant.utilities.data.db_manager import db_manager
from k2_quant.utilities.logger import k2_logger


class ModelLoaderService:
    """Service for loading and managing saved models"""
    
    def __init__(self):
        self.db = db_manager
        self.loaded_models = {}  # Simple cache for frequently used models
        self.metadata_cache = {}
        self.max_cache_size = 5  # Maximum number of models to keep in memory
        
        # Create SQLAlchemy engine for pandas operations
        self._create_sqlalchemy_engine()
    
    def _create_sqlalchemy_engine(self):
        """Create SQLAlchemy engine for pandas.to_sql operations"""
        try:
            # Get database connection parameters from environment
            db_params = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'database': os.getenv('DB_NAME', 'k2_quant'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'postgres'),
                'port': os.getenv('DB_PORT', '5433'),
            }
            
            # Create connection string
            conn_string = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
            
            # Create engine with NullPool to avoid connection pool conflicts
            self.engine = create_engine(conn_string, poolclass=NullPool)
            
            k2_logger.info("SQLAlchemy engine created for pandas operations", "MODEL_LOADER")
        except Exception as e:
            k2_logger.error(f"Failed to create SQLAlchemy engine: {str(e)}", "MODEL_LOADER")
            self.engine = None
    
    def get_saved_models(self) -> List[Dict[str, Any]]:
        """Get list of all saved models with metadata"""
        try:
            tables = self.db.get_stock_tables()
            models = []
            
            for table_name, table_size in tables:
                # Parse table name to extract metadata
                parts = table_name.split('_')
                if len(parts) >= 4:
                    symbol = parts[1].upper()
                    timespan = parts[2]
                    range_val = parts[3]
                    
                    # Get additional statistics
                    stats = self.get_model_statistics(table_name)
                    
                    models.append({
                        'table_name': table_name,
                        'symbol': symbol,
                        'timespan': timespan,
                        'range': range_val,
                        'size': table_size,
                        'record_count': stats.get('total_records', 0),
                        'date_range': stats.get('date_range', ('', '')),
                        'created_at': stats.get('created_at')
                    })
            
            k2_logger.info(f"Found {len(models)} saved models", "MODEL_LOADER")
            return models
            
        except Exception as e:
            k2_logger.error(f"Failed to get saved models: {str(e)}", "MODEL_LOADER")
            return []
    
    def get_model_metadata(self, table_name: str) -> Dict[str, Any]:
        """Get metadata for a specific model"""
        if table_name in self.metadata_cache:
            return self.metadata_cache[table_name]
        
        try:
            # Parse table name
            parts = table_name.split('_')
            metadata = {
                'table_name': table_name,
                'symbol': parts[1].upper() if len(parts) > 1 else 'UNKNOWN',
                'timespan': parts[2] if len(parts) > 2 else 'unknown',
                'range': parts[3] if len(parts) > 3 else 'unknown'
            }
            
            # Get statistics
            stats = self.db.get_table_statistics(table_name)
            metadata.update(stats)
            
            # Cache metadata
            self.metadata_cache[table_name] = metadata
            
            return metadata
            
        except Exception as e:
            k2_logger.error(f"Failed to get metadata for {table_name}: {str(e)}", "MODEL_LOADER")
            return {}
    
    def get_model_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get detailed statistics for a model"""
        try:
            stats = self.db.get_table_statistics(table_name)
            
            # Add date range
            date_range = self.db.get_date_range(table_name)
            stats['date_range'] = date_range
            
            return stats
            
        except Exception as e:
            k2_logger.error(f"Failed to get statistics for {table_name}: {str(e)}", "MODEL_LOADER")
            return {}
    
    def load_model_data(self, table_name: str, 
                       limit: Optional[int] = None,
                       use_cache: bool = True) -> pd.DataFrame:
        """Load model data into DataFrame"""
        # Check cache first
        if use_cache and table_name in self.loaded_models:
            k2_logger.info(f"Loading {table_name} from cache", "MODEL_LOADER")
            df = self.loaded_models[table_name]
            if limit:
                return df.head(limit)
            return df
        
        try:
            k2_logger.info(f"Loading model {table_name} from database", "MODEL_LOADER")
            
            # Use SQLAlchemy engine for pandas operations
            if self.engine:
                query = f"SELECT * FROM {table_name} ORDER BY timestamp"
                if limit:
                    query += f" LIMIT {limit}"
                
                df = pd.read_sql_query(query, self.engine)
                
                # Add to cache if not limited
                if not limit or limit >= 10000:
                    self._add_to_cache(table_name, df)
                
                return df
            else:
                # Fallback to psycopg2 if SQLAlchemy engine not available
                return self._load_with_psycopg2(table_name, limit)
                
        except Exception as e:
            k2_logger.error(f"Failed to load model data: {str(e)}", "MODEL_LOADER")
            return pd.DataFrame()
    
    def _load_with_psycopg2(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Fallback loading method using psycopg2"""
        try:
            with self.db.get_connection() as conn:
                query = f"SELECT * FROM {table_name} ORDER BY timestamp"
                if limit:
                    query += f" LIMIT {limit}"
                
                # Read in chunks for large datasets
                chunk_size = 50000
                chunks = []
                offset = 0
                
                while True:
                    chunk_query = f"{query} OFFSET {offset} LIMIT {chunk_size}"
                    chunk = pd.read_sql_query(chunk_query, conn)
                    
                    if chunk.empty:
                        break
                    
                    chunks.append(chunk)
                    offset += chunk_size
                    
                    if limit and offset >= limit:
                        break
                    
                    k2_logger.info(f"Loaded chunk: {offset} records", "MODEL_LOADER")
                
                # Combine chunks
                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
                    if limit:
                        df = df.head(limit)
                    return df
                
                return pd.DataFrame()
                
        except Exception as e:
            k2_logger.error(f"Failed to load with psycopg2: {str(e)}", "MODEL_LOADER")
            return pd.DataFrame()
    
    def _add_to_cache(self, table_name: str, data: pd.DataFrame):
        """Add model to cache with size management"""
        # Remove oldest if cache is full
        if len(self.loaded_models) >= self.max_cache_size:
            # Remove first (oldest) item
            oldest = next(iter(self.loaded_models))
            del self.loaded_models[oldest]
            k2_logger.info(f"Removed {oldest} from cache", "MODEL_LOADER")
        
        self.loaded_models[table_name] = data
        k2_logger.info(f"Added {table_name} to cache", "MODEL_LOADER")
    
    def load_model_subset(self, table_name: str, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Load subset of model data based on criteria"""
        try:
            # Build query
            select_cols = ', '.join(columns) if columns else '*'
            query = f"SELECT {select_cols} FROM {table_name}"
            
            conditions = []
            if start_date:
                conditions.append(f"date_time_market >= '{start_date}'")
            if end_date:
                conditions.append(f"date_time_market <= '{end_date}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp"
            
            # Use SQLAlchemy engine
            if self.engine:
                df = pd.read_sql_query(query, self.engine)
            else:
                # Fallback to psycopg2
                with self.db.get_connection() as conn:
                    df = pd.read_sql_query(query, conn)
            
            return df
            
        except Exception as e:
            k2_logger.error(f"Failed to load subset: {str(e)}", "MODEL_LOADER")
            return pd.DataFrame()
    
    def save_model_with_projections(self, original_table: str, 
                                   data_with_projections: pd.DataFrame,
                                   suffix: str = "projected") -> str:
        """Save model with projections as new table using SQLAlchemy"""
        try:
            # Create new table name
            new_table = f"{original_table}_{suffix}"
            
            # Check if SQLAlchemy engine is available
            if not self.engine:
                k2_logger.error("SQLAlchemy engine not available", "MODEL_LOADER")
                return ""
            
            # Drop table if exists
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {new_table}"))
                conn.commit()
            
            # Save DataFrame to SQL using SQLAlchemy engine
            data_with_projections.to_sql(
                new_table, 
                self.engine, 
                if_exists='replace', 
                index=False,
                method='multi',  # Use multi-row insert for better performance
                chunksize=10000  # Insert in chunks
            )
            
            # Add indexes for better query performance
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{new_table}_timestamp 
                    ON {new_table}(timestamp)
                """))
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{new_table}_datetime 
                    ON {new_table}(date_time_market)
                """))
                conn.commit()
            
            k2_logger.info(f"Saved projected model as {new_table}", "MODEL_LOADER")
            return new_table
            
        except Exception as e:
            k2_logger.error(f"Failed to save projected model: {str(e)}", "MODEL_LOADER")
            return ""
    
    def clear_cache(self):
        """Clear all cached models"""
        self.loaded_models.clear()
        self.metadata_cache.clear()
        k2_logger.info("Model cache cleared", "MODEL_LOADER")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models"""
        return {
            'cached_models': list(self.loaded_models.keys()),
            'cache_size': len(self.loaded_models),
            'max_cache_size': self.max_cache_size,
            'total_records_cached': sum(len(df) for df in self.loaded_models.values())
        }
    
    def __del__(self):
        """Cleanup SQLAlchemy engine on deletion"""
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()


# Singleton instance
model_loader_service = ModelLoaderService()