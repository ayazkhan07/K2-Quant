"""
K2 Quant Analysis Page - Harmonized Orchestrator

Coordinates the three pane components without containing UI logic.
Save as: k2_quant/pages/analysis/page.py
"""

import re
from typing import Dict, Optional, Any, List, Union
from datetime import datetime
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QLabel, QStatusBar)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.services.technical_analysis_service import ta_service
from k2_quant.utilities.services.stock_data_service import stock_service
from k2_quant.utilities.data.saved_models_manager import saved_models_manager
from k2_quant.utilities.services.strategy_service import strategy_service
from k2_quant.utilities.services.dynamic_python_engine import dpe_service

# Import the three pane components
from k2_quant.pages.analysis.components.left_pane import LeftPaneWidget
from k2_quant.pages.analysis.components.middle_pane import MiddlePaneWidget
from k2_quant.pages.analysis.components.right_pane import RightPaneWidget


class AnalysisPageWidget(QWidget):
    """Main Analysis page widget - orchestrates three pane components"""
    
    # Signals
    back_to_stock_fetcher = pyqtSignal()
    
    def __init__(self, tab_id: int = 0, parent=None):
        super().__init__(parent)
        self.tab_id = tab_id
        self.current_model = None
        self.current_data = None
        self.current_metadata = {}
        self.applied_indicators = {}
        
        self.init_ui()
        self.setup_styling()
        self.load_saved_models()
        
        k2_logger.info(f"Analysis page initialized (Tab ID: {tab_id})", "ANALYSIS")
    
    def init_ui(self):
        """Initialize the UI layout and create pane components"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)
        
        # Create header
        self.create_header(main_layout)
        
        # Create splitter for three panes
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #1a1a1a;
            }
        """)
        
        # Create the three pane components
        self.left_pane = LeftPaneWidget()
        self.middle_pane = MiddlePaneWidget()
        self.right_pane = RightPaneWidget()
        
        # Wire up signals
        self.setup_connections()
        
        # Add to splitter
        self.splitter.addWidget(self.left_pane)
        self.splitter.addWidget(self.middle_pane)
        self.splitter.addWidget(self.right_pane)
        
        # Set sizes - left: 280px, middle: flexible, right: 380px
        self.splitter.setSizes([280, 740, 380])
        
        main_layout.addWidget(self.splitter)
        
        # Create status bar
        self.status_widget = self.create_status_bar()
        main_layout.addWidget(self.status_widget)
        
        # Load initial data
        self.refresh_left_pane_data()
    
    def create_header(self, parent_layout):
        """Create header with title"""
        header = QWidget()
        header.setFixedHeight(40)
        header.setObjectName("analysisHeader")
        
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(20, 0, 20, 0)
        header.setLayout(header_layout)
        
        title = QLabel(f"K2 QUANT - ANALYSIS (Tab {self.tab_id})")
        title.setFont(QFont("Arial", 14))
        title.setStyleSheet("color: #999; letter-spacing: 1px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        parent_layout.addWidget(header)
    
    def create_status_bar(self) -> QWidget:
        """Create status bar"""
        widget = QWidget()
        widget.setFixedHeight(32)
        widget.setObjectName("analysisStatusBar")
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        widget.setLayout(layout)
        
        # Status indicator
        indicator = QLabel("●")
        indicator.setStyleSheet("color: #4a4; font-size: 8px;")
        layout.addWidget(indicator)
        
        # Model info
        self.model_label = QLabel("No model loaded")
        self.model_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.model_label)
        
        layout.addStretch()
        
        # System status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.status_label)
        
        return widget
    
    def setup_connections(self):
        """Wire up all signals between components"""
        # Left pane connections
        self.left_pane.model_selected.connect(self.load_model_by_table)
        self.left_pane.strategy_toggled.connect(self.on_strategy_toggled)
        self.left_pane.indicator_toggled.connect(self.on_indicator_toggled)
        
        # Middle pane connections
        self.middle_pane.column_toggled.connect(self.on_column_toggled)
        self.middle_pane.view_mode_changed.connect(self.on_view_mode_changed)
        self.middle_pane.projection_requested.connect(self.on_projection_requested)
        self.middle_pane.indicator_applied.connect(self.on_indicator_applied_from_middle)
        self.middle_pane.data_exported.connect(self.on_data_exported)
        
        # Right pane connections
        self.right_pane.message_sent.connect(self.on_ai_message_sent)
        self.right_pane.strategy_generated.connect(self.on_strategy_generated)
        self.right_pane.projection_requested.connect(self.on_projection_requested_from_ai)
    
    def refresh_left_pane_data(self):
        """Refresh all data in the left pane"""
        try:
            # Load saved models
            models = saved_models_manager.get_saved_models()
            self.left_pane.populate_models(models)
            
            # Load strategies
            try:
                strategies = strategy_service.get_all_strategies()
                self.left_pane.populate_strategies(strategies)
            except Exception:
                self.left_pane.populate_strategies([])
            
            k2_logger.info("Left pane data refreshed", "ANALYSIS")
            
        except Exception as e:
            k2_logger.error(f"Failed to refresh left pane data: {str(e)}", "ANALYSIS")
    
    def load_model_by_table(self, table_name: str):
        """Load model by table name"""
        k2_logger.info(f"Loading model: {table_name}", "ANALYSIS")
        
        try:
            # Get data from table (limit to last 500 rows for performance)
            rows, total_count = stock_service.get_display_data(table_name, limit=500)
            
            if rows:
                # Update current state
                self.current_model = table_name
                self.current_data = rows
                
                # Gather metadata
                base_metadata = saved_models_manager.get_model_metadata(table_name) or {'symbol': table_name}
                table_info = stock_service.get_table_info(table_name) or {}
                date_range = table_info.get('date_range')
                
                # Merge metadata while keeping existing keys stable
                self.current_metadata = dict(base_metadata)
                self.current_metadata.update({
                    'records': total_count,
                    'table_name': table_name,
                    'total_records': total_count,
                    'date_range': date_range
                })
                
                # Update status
                self.model_label.setText(f"Model: {table_name} ({total_count:,} records)")
                self.status_label.setText("Model loaded")
                
                # Load into middle pane (chart uses limited data; table uses limited data)
                self.middle_pane.load_data(rows, self.current_metadata)
                
                # Update AI context
                ctx = {
                    'symbol': self.current_metadata.get('symbol', table_name),
                    'records': total_count,
                    'table_name': table_name
                }
                if date_range:
                    ctx['date_range'] = date_range
                self.right_pane.set_data_context(ctx)
                
                # Clear any existing indicators/strategies when loading new model
                self.left_pane.clear_all_indicators()
                self.left_pane.clear_all_strategies()
                
                # Restore model state if available
                try:
                    state = saved_models_manager.get_model_state(table_name)
                    if state:
                        k2_logger.info(f"Model state available for {table_name}", "ANALYSIS")
                        # Could restore indicators/strategies here if needed
                except Exception as e:
                    k2_logger.debug(f"No model state available: {e}", "ANALYSIS")
                    
        except Exception as e:
            k2_logger.error(f"Failed to load model: {str(e)}", "ANALYSIS")
            self.status_label.setText("Error loading model")
    
    def on_indicator_toggled(self, indicator_name: str, enabled: bool):
        """Handle indicator toggle from left pane"""
        if not self.current_model:
            k2_logger.warning("No model loaded for indicator toggle", "ANALYSIS")
            return
        
        k2_logger.info(f"Indicator '{indicator_name}' toggled to {enabled}", "ANALYSIS")
        
        if enabled:
            # Apply indicator
            params = self.extract_default_indicator_params(indicator_name)
            self.apply_indicator(indicator_name, params)
        else:
            # Remove indicator
            self.remove_indicator(indicator_name)
    
    def extract_default_indicator_params(self, indicator_name: str) -> Dict:
        """Extract default parameters from indicator name"""
        params = {}
        
        # Extract number from parentheses if present
        if "(" in indicator_name and ")" in indicator_name:
            match = re.search(r'\((\d+)\)', indicator_name)
            if match:
                period = int(match.group(1))
                params['period'] = period
        
        # Get base name without parentheses
        base_name = indicator_name.split("(")[0].strip().upper()
        
        # Set defaults for specific indicators
        if base_name == "RSI" and 'period' not in params:
            params['period'] = 14
        elif base_name == "MACD":
            params = {'fast': 12, 'slow': 26, 'signal': 9}
        elif base_name == "BOLLINGER BANDS":
            params = {'period': 20, 'std': 2}
        elif base_name == "STOCHASTIC":
            params = {'k_period': 14, 'd_period': 3}
        elif base_name in ["OBV", "VWAP", "VOLUME"]:
            params = {}
        elif base_name in ["SMA", "EMA"] and 'period' not in params:
            params['period'] = 20
        
        return params
    
    def apply_indicator(self, indicator_name: str, params: Dict):
        """Apply indicator to current data"""
        try:
            # Get full dataframe
            df = stock_service.get_full_dataframe(self.current_model)
            if df is None or df.empty:
                return
            
            # Create datetime index in UTC
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC')
            df.set_index('datetime', inplace=True)
            
            # Get base indicator name
            base_name = indicator_name.split("(")[0].strip()
            
            # Calculate indicator
            indicator_data = ta_service.calculate_indicator(df, base_name, params)
            
            if indicator_data is not None:
                # Determine color
                colors = {
                    'SMA': '#00ffff',
                    'EMA': '#ff00ff',
                    'MACD': '#ffa500',
                    'RSI': '#00ff00',
                    'BOLLINGER': '#ffff00',
                    'STOCHASTIC': '#00ff00',
                    'OBV': '#ff69b4',
                    'VWAP': '#ffd700'
                }
                color = colors.get(base_name.upper(), '#ffff00')
                
                # Ensure it's a Series
                if isinstance(indicator_data, np.ndarray):
                    indicator_data = pd.Series(indicator_data, index=df.index, name=indicator_name)
                
                # Add to chart
                self.middle_pane.add_indicator(indicator_name, indicator_data, color)
                
                # Store in applied indicators
                self.applied_indicators[indicator_name] = params
                
                # Optionally persist to database
                self.persist_indicator_to_db(base_name, params, indicator_data)
                
        except Exception as e:
            k2_logger.error(f"Error applying indicator {indicator_name}: {e}", "ANALYSIS")
    
    def remove_indicator(self, indicator_name: str):
        """Remove indicator from display"""
        try:
            self.middle_pane.remove_indicator(indicator_name)
            
            if indicator_name in self.applied_indicators:
                del self.applied_indicators[indicator_name]
            
            k2_logger.info(f"Removed indicator: {indicator_name}", "ANALYSIS")
            
        except Exception as e:
            k2_logger.error(f"Failed to remove indicator {indicator_name}: {e}", "ANALYSIS")
    
    def persist_indicator_to_db(self, name: str, params: Dict, data: pd.Series):
        """Persist indicator to database (optional)"""
        if not self.current_model:
            return
        
        try:
            # Create column name
            col_name = self._indicator_column_name(name, params)
            
            # Get dataframe with timestamp
            df = stock_service.get_full_dataframe(self.current_model)
            if df is None:
                return
            
            # Ensure column exists
            stock_service.ensure_indicator_column(self.current_model, col_name, sql_type="NUMERIC")
            
            # Update column
            stock_service.update_indicator_column(self.current_model, col_name, df['timestamp'], data)
            
            # Update model state
            state = saved_models_manager.get_model_state(self.current_model)
            indicators = state.get('indicators', {})
            indicators[col_name] = {'name': name, 'params': params}
            saved_models_manager.set_model_state(
                self.current_model,
                indicators=indicators,
                active_strategy=state.get('active_strategy'),
                chart_range=state.get('chart_range')
            )
            
        except Exception as e:
            k2_logger.debug(f"Could not persist indicator to DB: {e}", "ANALYSIS")
    
    def _indicator_column_name(self, name: str, params: Dict) -> str:
        """Generate column name for indicator"""
        items = [f"{k}_{params[k]}" for k in sorted(params.keys())] if params else []
        return "_".join([name.lower()] + items) if items else name.lower()
    
    def on_strategy_toggled(self, strategy_name: str, enabled: bool):
        """Handle strategy toggle from left pane"""
        if not self.current_model:
            k2_logger.warning("No model loaded for strategy toggle", "ANALYSIS")
            return
        
        k2_logger.info(f"Strategy '{strategy_name}' toggled to {enabled}", "ANALYSIS")
        
        try:
            if enabled:
                self.apply_strategy(strategy_name)
            else:
                self.remove_strategy(strategy_name)
        except Exception as e:
            k2_logger.error(f"Strategy toggle failed: {str(e)}", "ANALYSIS")
    
    def apply_strategy(self, strategy_name: str):
        """Apply strategy projections"""
        table_name = self.current_model
        
        # Fetch strategy code
        code = strategy_service.get_strategy_code(strategy_name)
        if not code:
            k2_logger.warning(f"Strategy code not found: {strategy_name}", "ANALYSIS")
            return
        
        # Load full dataset
        rows, _ = stock_service.get_display_data(table_name, limit=10**9)
        df = pd.DataFrame(rows, columns=['Date','Time','Open','High','Low','Close','Volume','VWAP'])
        
        # Convert to strategy format
        df['date_time_market'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume','VWAP':'vwap'})
        
        # Execute strategy
        result = dpe_service.execute_strategy(code, df)
        if not result.get('success'):
            k2_logger.error(f"Strategy execution failed: {result.get('error')}", "ANALYSIS")
            return
        
        result_df = result.get('data') if isinstance(result.get('data'), pd.DataFrame) else df
        
        # Get projection rows
        proj_df = result_df.iloc[len(df):].copy() if len(result_df) > len(df) else pd.DataFrame()
        
        if not proj_df.empty:
            # Insert projections
            stock_service.delete_projections(table_name, strategy_name)
            stock_service.insert_projections(table_name, proj_df, strategy_name)
            
            # Update state
            saved_models_manager.set_model_state(
                table_name,
                indicators=None,
                active_strategy=strategy_name,
                chart_range=None
            )
            
            # Reload view
            rows, total_count = stock_service.get_display_data(table_name, limit=500)
            self.middle_pane.load_data(rows, self.current_metadata)
            self.model_label.setText(f"Model: {table_name} ({total_count:,} records)")
    
    def remove_strategy(self, strategy_name: str):
        """Remove strategy projections"""
        table_name = self.current_model
        
        stock_service.delete_projections(table_name, strategy_name)
        saved_models_manager.set_model_state(
            table_name,
            indicators=None,
            active_strategy=None,
            chart_range=None
        )
        
        # Reload view
        rows, total_count = stock_service.get_display_data(table_name, limit=500)
        self.middle_pane.load_data(rows, self.current_metadata)
        self.model_label.setText(f"Model: {table_name} ({total_count:,} records)")
    
    # Middle pane event handlers
    def on_column_toggled(self, column: str, visible: bool):
        """Handle column visibility toggle"""
        k2_logger.info(f"Column '{column}' toggled to {'visible' if visible else 'hidden'}", "ANALYSIS")
    
    def on_view_mode_changed(self, mode: str):
        """Handle view mode change"""
        k2_logger.info(f"View mode changed to {mode}", "ANALYSIS")
    
    def on_projection_requested(self):
        """Handle projection request from middle pane"""
        k2_logger.info("Projection requested from middle pane", "ANALYSIS")
        # Could trigger AI to generate projections
    
    def on_indicator_applied_from_middle(self, indicator_type: str, params: dict):
        """Handle indicator application from middle pane"""
        k2_logger.info(f"Indicator {indicator_type} applied with params {params}", "ANALYSIS")
        self.apply_indicator(indicator_type, params)
    
    def on_data_exported(self, format: str, data: pd.DataFrame):
        """Handle data export"""
        k2_logger.info(f"Data exported in {format} format", "ANALYSIS")
    
    # Right pane event handlers
    def on_ai_message_sent(self, message: str):
        """Handle AI message"""
        k2_logger.info(f"AI message: {message}", "ANALYSIS")
        # In full implementation, would process with AI service
    
    def on_strategy_generated(self, name: str, code: str):
        """Handle strategy generation from AI"""
        k2_logger.info(f"Strategy generated: {name}", "ANALYSIS")
        # Could save strategy and apply it
    
    def on_projection_requested_from_ai(self, params: dict):
        """Handle projection request from AI"""
        k2_logger.info(f"Projection requested from AI: {params}", "ANALYSIS")
        # Could generate projections based on params
    
    def load_saved_models(self):
        """Load saved models into the left pane"""
        self.left_pane.refresh_models()
    
    def setup_styling(self):
        """Apply consistent styling"""
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0a0a;
                color: #ffffff;
            }
            
            #analysisHeader {
                background-color: #0f0f0f;
                border-bottom: 1px solid #1a1a1a;
            }
            
            #analysisStatusBar {
                background-color: #0f0f0f;
                border-top: 1px solid #1a1a1a;
            }
        """)
    
    def cleanup(self):
        """Clean up resources when closing tab"""
        try:
            # Clear left pane states
            self.left_pane.clear_all_indicators()
            self.left_pane.clear_all_strategies()
            
            # Clear middle pane
            self.middle_pane.cleanup()
            
            # Clear right pane
            self.right_pane.cleanup()
            
            # Clear current data
            self.current_model = None
            self.current_data = None
            
            k2_logger.info(f"Analysis page cleaned up (Tab ID: {self.tab_id})", "ANALYSIS")
        except Exception as e:
            k2_logger.error(f"Error during cleanup: {str(e)}", "ANALYSIS")
    
    def reset_after_database_cleared(self):
        """Reset UI and state after DB deletion"""
        try:
            # Clear all panes
            self.left_pane.populate_models([])
            self.left_pane.clear_all_indicators()
            self.left_pane.clear_all_strategies()
            
            # Clear middle pane
            self.middle_pane.clear_data()
            
            # Reset labels
            self.model_label.setText("No model loaded")
            self.status_label.setText("Ready")
            
            # Clear AI chat
            self.right_pane.clear_chat()
            
            # Reset internal state
            self.current_model = None
            self.current_data = None
            
            k2_logger.info(f"Analysis tab {self.tab_id} reset after DB clear", "ANALYSIS")
        except Exception as e:
            k2_logger.error(f"Analysis reset failed: {str(e)}", "ANALYSIS")