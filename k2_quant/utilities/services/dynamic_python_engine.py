"""
Dynamic Python Engine (DPE) for K2 Quant

Executes custom strategies with full Python access, real-time monitoring,
and backtesting capabilities. Handles complex calculations like elasticity patterns.
"""

import sys
import io
import ast
import time
import traceback
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from contextlib import redirect_stdout, redirect_stderr
import multiprocessing as mp
from queue import Empty

from k2_quant.utilities.logger import k2_logger


class StrategyExecutor:
    """Executes strategy code in isolated context"""
    
    def __init__(self):
        self.execution_context = {}
        self.setup_execution_environment()
    
    def setup_execution_environment(self):
        """Setup safe execution environment with necessary libraries"""
        # Import common libraries into execution context
        self.execution_context = {
            'pd': pd,
            'np': np,
            'datetime': datetime,
            'timedelta': timedelta,
            # Mathematical functions
            'sqrt': np.sqrt,
            'abs': np.abs,
            'min': np.min,
            'max': np.max,
            'mean': np.mean,
            'median': np.median,
            'std': np.std,
            'sum': np.sum,
            # Pandas functions
            'DataFrame': pd.DataFrame,
            'Series': pd.Series,
            'concat': pd.concat,
            'merge': pd.merge,
            # Custom helper functions
            'calculate_elasticity': self.calculate_elasticity,
            'find_similar_patterns': self.find_similar_patterns,
            'project_future_prices': self.project_future_prices,
            'backtest_strategy': self.backtest_strategy
        }
    
    @staticmethod
    def calculate_elasticity(df: pd.DataFrame) -> pd.Series:
        """Calculate price elasticity: (high - low) / low * 100"""
        if 'high' in df.columns and 'low' in df.columns:
            return ((df['high'] - df['low']) / df['low'] * 100)
        else:
            raise ValueError("DataFrame must have 'high' and 'low' columns")
    
    @staticmethod
    def find_similar_patterns(df: pd.DataFrame, target_idx: int, 
                            column: str, tolerance: float = 0.05) -> List[int]:
        """Find indices with values within tolerance range of target"""
        if target_idx >= len(df):
            raise ValueError(f"Target index {target_idx} out of range")
        
        target_value = df.iloc[target_idx][column]
        lower_bound = target_value * (1 - tolerance)
        upper_bound = target_value * (1 + tolerance)
        
        # Find matching indices (excluding target and future indices)
        matches = []
        for idx in range(min(target_idx, len(df))):
            value = df.iloc[idx][column]
            if lower_bound <= value <= upper_bound and idx != target_idx:
                matches.append(idx)
        
        return matches
    
    @staticmethod
    def project_future_prices(df: pd.DataFrame, base_idx: int, 
                            projection_days: int = 10) -> pd.DataFrame:
        """Project future prices based on historical patterns"""
        if base_idx >= len(df):
            raise ValueError(f"Base index {base_idx} out of range")
        
        base_price = df.iloc[base_idx]['close']
        projections = []
        
        for day in range(1, projection_days + 1):
            # Simple projection - can be made more sophisticated
            projected_date = df.iloc[base_idx]['date_time'] + timedelta(days=day)
            projected_price = base_price * (1 + np.random.normal(0.001, 0.02))
            
            projections.append({
                'date_time': projected_date,
                'projected_price': projected_price,
                'projection_day': day,
                'base_index': base_idx
            })
        
        return pd.DataFrame(projections)
    
    @staticmethod
    def backtest_strategy(df: pd.DataFrame, signal_column: str, 
                         initial_capital: float = 100000) -> Dict[str, Any]:
        """Backtest a strategy based on buy/sell signals"""
        if signal_column not in df.columns:
            raise ValueError(f"Signal column '{signal_column}' not found")
        
        capital = initial_capital
        position = 0
        trades = []
        
        for idx in range(len(df)):
            signal = df.iloc[idx][signal_column]
            price = df.iloc[idx]['close']
            
            if signal == 1 and position == 0:  # Buy signal
                position = capital / price
                trades.append({
                    'type': 'buy',
                    'price': price,
                    'quantity': position,
                    'index': idx
                })
                capital = 0
                
            elif signal == -1 and position > 0:  # Sell signal
                capital = position * price
                trades.append({
                    'type': 'sell',
                    'price': price,
                    'quantity': position,
                    'index': idx
                })
                position = 0
        
        # Final value
        if position > 0:
            capital = position * df.iloc[-1]['close']
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'return_pct': ((capital - initial_capital) / initial_capital) * 100,
            'total_trades': len(trades),
            'trades': trades
        }
    
    def execute_code(self, code: str, data: pd.DataFrame, 
                    monitor_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute strategy code with monitoring"""
        start_time = time.time()
        result = {
            'success': False,
            'data': None,
            'output': '',
            'error': None,
            'execution_time': 0,
            'metrics': {}
        }
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Prepare execution context with data
            exec_context = self.execution_context.copy()
            exec_context['data'] = data.copy()  # Work on copy to preserve original
            exec_context['df'] = exec_context['data']  # Alias for convenience
            
            # Notify monitor
            if monitor_callback:
                monitor_callback('start', {'rows': len(data), 'columns': len(data.columns)})
            
            # Execute code with output capture
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_context)
            
            # Get modified data or result
            if 'result' in exec_context:
                result['data'] = exec_context['result']
            elif 'data' in exec_context:
                result['data'] = exec_context['data']
            elif 'df' in exec_context:
                result['data'] = exec_context['df']
            
            # Capture any print output
            result['output'] = stdout_capture.getvalue()
            
            # Calculate metrics
            result['metrics'] = self.calculate_execution_metrics(
                data, result['data'], start_time
            )
            
            result['success'] = True
            
            # Notify completion
            if monitor_callback:
                monitor_callback('complete', result['metrics'])
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            stderr_output = stderr_capture.getvalue()
            if stderr_output:
                result['error'] += f"\nStderr: {stderr_output}"
            
            # Notify error
            if monitor_callback:
                monitor_callback('error', {'error': str(e)})
            
            k2_logger.error(f"Strategy execution failed: {str(e)}", "DPE")
        
        finally:
            result['execution_time'] = time.time() - start_time
        
        return result
    
    def calculate_execution_metrics(self, original_data: pd.DataFrame, 
                                   result_data: Any, start_time: float) -> Dict[str, Any]:
        """Calculate execution metrics"""
        metrics = {
            'execution_time': time.time() - start_time,
            'original_rows': len(original_data),
            'original_columns': len(original_data.columns)
        }
        
        if isinstance(result_data, pd.DataFrame):
            metrics['result_rows'] = len(result_data)
            metrics['result_columns'] = len(result_data.columns)
            metrics['new_columns'] = list(set(result_data.columns) - set(original_data.columns))
            metrics['rows_added'] = max(0, len(result_data) - len(original_data))
        
        return metrics


class DynamicPythonEngine:
    """Main DPE service for strategy execution and management"""
    
    def __init__(self):
        self.executor = StrategyExecutor()
        self.execution_history = []
        self.saved_strategies = {}
    
    def execute_strategy(self, strategy_code: str, data: pd.DataFrame,
                        metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a trading strategy"""
        k2_logger.info("Executing strategy", "DPE")
        
        # Create execution record
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Execute with monitoring
        def monitor(status, info):
            k2_logger.info(f"Strategy execution {status}: {info}", "DPE")
        
        result = self.executor.execute_code(strategy_code, data, monitor)
        
        # Store in history
        self.execution_history.append({
            'id': execution_id,
            'timestamp': datetime.now().isoformat(),
            'code': strategy_code,
            'metadata': metadata,
            'result': {
                'success': result['success'],
                'error': result.get('error'),
                'metrics': result.get('metrics', {})
            }
        })
        
        return result
    
    def execute_elasticity_strategy(self, data: pd.DataFrame, 
                                   reference_id: int = 5000,
                                   tolerance: float = 0.05,
                                   iterations: int = 5,
                                   projection_days: int = 10) -> pd.DataFrame:
        """
        Execute the complex elasticity strategy as described:
        1. Calculate elasticity for all rows
        2. Find similar patterns to reference_id
        3. Iterate to narrow down matches
        4. Project future prices based on patterns
        """
        k2_logger.info(f"Executing elasticity strategy (ref_id={reference_id})", "DPE")
        
        # Step 1: Calculate elasticity
        data['elasticity'] = ((data['high'] - data['low']) / data['low']) * 100
        
        # Step 2: Find initial matches for reference elasticity
        if reference_id >= len(data):
            raise ValueError(f"Reference ID {reference_id} out of range")
        
        ref_elasticity = data.iloc[reference_id]['elasticity']
        matches = []
        
        for idx in range(reference_id):  # Only look at prior IDs
            elasticity = data.iloc[idx]['elasticity']
            if abs(elasticity - ref_elasticity) / ref_elasticity <= tolerance:
                matches.append(idx)
        
        k2_logger.info(f"Initial matches: {len(matches)} rows", "DPE")
        
        # Step 3: Iterate to narrow down matches
        for iteration in range(1, iterations):
            if len(matches) == 0:
                break
            
            # Look at elasticity of (current_id - iteration)
            ref_idx_prev = reference_id - iteration
            if ref_idx_prev < 0:
                break
            
            ref_elasticity_prev = data.iloc[ref_idx_prev]['elasticity']
            new_matches = []
            
            for match_idx in matches:
                match_idx_prev = match_idx - iteration
                if match_idx_prev >= 0:
                    match_elasticity_prev = data.iloc[match_idx_prev]['elasticity']
                    if abs(match_elasticity_prev - ref_elasticity_prev) / ref_elasticity_prev <= tolerance:
                        new_matches.append(match_idx)
            
            matches = new_matches
            k2_logger.info(f"Iteration {iteration}: {len(matches)} matches remaining", "DPE")
        
        # Step 4: Project future prices
        if len(matches) > 0:
            # Calculate average elasticity changes from historical matches
            elasticity_changes = []
            
            for match_idx in matches[:10]:  # Use up to 10 matches
                future_elasticities = []
                for day in range(1, min(projection_days + 1, len(data) - match_idx)):
                    future_idx = match_idx + day
                    if future_idx < len(data):
                        future_elasticities.append(data.iloc[future_idx]['elasticity'])
                
                if future_elasticities:
                    elasticity_changes.append(future_elasticities)
            
            # Average the elasticity patterns
            if elasticity_changes:
                avg_elasticities = []
                for day in range(projection_days):
                    day_elasticities = [changes[day] for changes in elasticity_changes 
                                       if len(changes) > day]
                    if day_elasticities:
                        avg_elasticities.append(np.mean(day_elasticities))
                    else:
                        avg_elasticities.append(0)
                
                # Create projections
                today_open = data.iloc[-1]['open']
                projections = []
                
                for day, elasticity_pct in enumerate(avg_elasticities, 1):
                    # Calculate projected prices using elasticity
                    projected_low = today_open * (1 + day * 0.001)  # Slight trend
                    projected_high = projected_low * (1 + elasticity_pct / 100)
                    projected_close = (projected_high + projected_low) / 2
                    
                    projections.append({
                        'date_time': data.iloc[-1]['date_time'] + timedelta(days=day),
                        'open': projected_low,
                        'high': projected_high,
                        'low': projected_low,
                        'close': projected_close,
                        'volume': 0,  # No volume for projections
                        'elasticity': elasticity_pct,
                        'is_projection': True,
                        'projection_day': day
                    })
                
                # Append projections to data
                projection_df = pd.DataFrame(projections)
                result = pd.concat([data, projection_df], ignore_index=True)
                
                k2_logger.info(f"Added {len(projections)} projection rows", "DPE")
                return result
        
        k2_logger.warning("No matching patterns found for projection", "DPE")
        return data
    
    def validate_strategy_code(self, code: str) -> Dict[str, Any]:
        """Validate strategy code before execution"""
        result = {
            'valid': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Parse the code to check syntax
            ast.parse(code)
            result['valid'] = True
            
            # Check for dangerous operations
            dangerous_keywords = ['exec', 'eval', '__import__', 'open', 'file']
            for keyword in dangerous_keywords:
                if keyword in code:
                    result['warnings'].append(f"Code contains potentially dangerous keyword: {keyword}")
            
            # Check for required structure
            if 'data' not in code and 'df' not in code:
                result['warnings'].append("Code doesn't reference 'data' or 'df' - may not modify the dataset")
            
        except SyntaxError as e:
            result['errors'].append(f"Syntax error: {str(e)}")
        except Exception as e:
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def save_strategy(self, name: str, code: str, description: str = "") -> bool:
        """Save a strategy for later use"""
        try:
            strategy = {
                'name': name,
                'code': code,
                'description': description,
                'created_at': datetime.now().isoformat(),
                'executions': 0
            }
            
            self.saved_strategies[name] = strategy
            k2_logger.info(f"Strategy saved: {name}", "DPE")
            return True
            
        except Exception as e:
            k2_logger.error(f"Failed to save strategy: {str(e)}", "DPE")
            return False
    
    def load_strategy(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a saved strategy"""
        return self.saved_strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """List all saved strategies"""
        return list(self.saved_strategies.keys())
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:]
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
        k2_logger.info("Execution history cleared", "DPE")


# Singleton instance
dpe_service = DynamicPythonEngine()