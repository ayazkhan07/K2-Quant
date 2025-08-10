"""
Model Loader Service for K2 Quant

Manages loading and caching of saved models from PostgreSQL database.
Handles efficient data access for models with millions of records.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

from k2_quant.utilities.data.db_manager import db_manager
from k2_quant.utilities.logger import k2_logger


class ModelLoaderService:
    """Service for loading and managing saved models"""
    
    def __init__(self):
        self.db = db_manager
        self.loaded_models = {}  # Simple cache for frequently used models
        self.metadata_cache = {}
        self.max_cache_size = 5  # Maximum number of models to keep in memory
    
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
            return self.loaded_models[table_name]
        
        try:
            k2_logger.info(f"Loading model {table_name} from database", "MODEL_LOADER")
            
            # For large datasets, use chunked loading
            with self.db.get_connection() as conn:
                # First get count
                with self.db.get_cursor(conn) as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    total_count = cur.fetchone()[0]
                
                # Decide loading strategy based on size
                if total_count > 1000000:  # More than 1M records
                    # Load in chunks for very large datasets
                    df = self._load_large_dataset(conn, table_name, limit)
                else:
                    # Load all at once for smaller datasets
                    query = f"""
                        SELECT date_time_market as date_time, 
                               open, high, low, close, volume, vwap
                        FROM {table_name}
                        ORDER BY timestamp
                    """
                    if limit:
                        query += f" LIMIT {limit}"
                    
                    df = pd.read_sql_query(query, conn)
            
            # Convert date_time to datetime if needed
            if 'date_time' in df.columns:
                df['date_time'] = pd.to_datetime(df['date_time'])
            
            # Cache if not too large
            if len(df) < 100000 and use_cache:
                self._add_to_cache(table_name, df)
            
            k2_logger.info(f"Loaded {len(df)} records from {table_name}", "MODEL_LOADER")
            return df
            
        except Exception as e:
            k2_logger.error(f"Failed to load model {table_name}: {str(e)}", "MODEL_LOADER")
            return pd.DataFrame()
    
    def _load_large_dataset(self, conn, table_name: str, 
                          limit: Optional[int] = None) -> pd.DataFrame:
        """Load large dataset in chunks"""
        chunk_size = 100000
        chunks = []
        offset = 0
        
        while True:
            query = f"""
                SELECT date_time_market as date_time, 
                       open, high, low, close, volume, vwap
                FROM {table_name}
                ORDER BY timestamp
                LIMIT {chunk_size} OFFSET {offset}
            """
            
            chunk = pd.read_sql_query(query, conn)
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
            
            with self.db.get_connection() as conn:
                df = pd.read_sql_query(query, conn)
            
            return df
            
        except Exception as e:
            k2_logger.error(f"Failed to load subset: {str(e)}", "MODEL_LOADER")
            return pd.DataFrame()
    
    def save_model_with_projections(self, original_table: str, 
                                   data_with_projections: pd.DataFrame,
                                   suffix: str = "projected") -> str:
        """Save model with projections as new table"""
        try:
            # Create new table name
            new_table = f"{original_table}_{suffix}"
            
            # Store in database
            with self.db.get_connection() as conn:
                # Create table
                with self.db.get_cursor(conn) as cur:
                    cur.execute(f"""
                        CREATE TABLE {new_table} AS 
                        SELECT * FROM {original_table} WHERE 1=0
                    """)
                    
                    # Add projection columns if needed
                    if 'is_projection' in data_with_projections.columns:
                        cur.execute(f"""
                            ALTER TABLE {new_table} 
                            ADD COLUMN is_projection BOOLEAN DEFAULT FALSE,
                            ADD COLUMN projection_day INTEGER
                        """)
                    
                    conn.commit()
            
            # Insert data
            data_with_projections.to_sql(new_table, conn, if_exists='append', index=False)
            
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


# Singleton instance
model_loader_service = ModelLoaderService()