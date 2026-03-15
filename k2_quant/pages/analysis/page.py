"""
K2 Quant Analysis Page - Harmonized Orchestrator

Coordinates the three pane components without containing UI logic.
Save as: k2_quant/pages/analysis/page.py

EXPECTATIONS:
=============
This module orchestrates technical indicator application with the following objectives:

1. DISPLAY NAME MAPPING:
   - MUST translate user-friendly display names (e.g., "Bollinger Bands", "Stochastic") 
     to TA service names (e.g., "BBANDS", "STOCH")
   - MUST handle indicators with spaces or special characters in display names

2. PARAMETER MAPPING:
   - MUST translate user-friendly parameter names to TA-Lib parameter names:
     * "period" -> "timeperiod" (for SMA, EMA, RSI, etc.)
     * "std" -> "nbdevup"/"nbdevdn" (for Bollinger Bands)
     * "k_period"/"d_period" -> "slowk_period"/"slowd_period" (for Stochastic)
     * "fast"/"slow"/"signal" -> "fastperiod"/"slowperiod"/"signalperiod" (for MACD)

3. INDICATOR APPLICATION:
   - MUST validate indicator calculation succeeds before adding to UI
   - MUST ensure indicator data is valid (not empty, has values) before display
   - MUST add indicator to BOTH chart AND table when applied
   - MUST remove indicator from BOTH chart AND table when removed

4. ERROR HANDLING:
   - MUST NOT add indicators to UI if calculation fails
   - MUST log errors clearly when indicator application fails
   - MUST handle parameter mismatches gracefully
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
        self.middle_pane.forecast_apply.connect(self.on_forecast_apply)
        
        # Right pane connections
        self.right_pane.message_sent.connect(self.on_ai_message_sent)
        self.right_pane.strategy_generated.connect(self.on_strategy_generated)
        self.right_pane.projection_requested.connect(self.on_projection_requested_from_ai)
        self.right_pane.data_modified.connect(self._on_ai_data_modified)
        self.right_pane.tab_writes_ready.connect(self._on_tab_writes)
        self.right_pane.workspace_provider = self._get_workspace_state
        self.right_pane.save_chat_callback = self._save_chat
        self.right_pane.load_chat_callback = self._load_chat
    
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
        
        # Persist tab data for the previous model before switching
        if self.current_model:
            self.middle_pane.persist_tab_data(self.current_model)
        
        try:
            rows, total_count = stock_service.get_display_data(table_name, limit=500)
            
            if rows:
                self.current_model = table_name
                self.current_data = rows
                
                base_metadata = saved_models_manager.get_model_metadata(table_name) or {'symbol': table_name}
                parts = table_name.split('_')
                symbol = parts[1].upper() if len(parts) > 1 else 'UNKNOWN'

                self.current_metadata = dict(base_metadata)
                self.current_metadata.update({
                    'records': total_count,
                    'table_name': table_name,
                    'total_records': total_count,
                    'symbol': symbol,
                })
                
                # Update status
                self.model_label.setText(f"Model: {table_name} ({total_count:,} records)")
                self.status_label.setText("Model loaded")
                
                # Load into middle pane (chart uses limited data; table uses limited data)
                self.middle_pane.load_data(rows, self.current_metadata)
                
                # Restore saved forecast / working-data tabs
                self.middle_pane.restore_tab_data(table_name)
                
                ctx = {
                    'symbol': self.current_metadata.get('symbol', table_name),
                    'records': total_count,
                    'table_name': table_name
                }
                self.right_pane.set_data_context(ctx)
                
                # Clear any existing indicators/strategies when loading new model
                self.left_pane.clear_all_indicators()
                self.left_pane.clear_all_strategies()
                
                # Restore model state if available
                try:
                    state = saved_models_manager.get_model_state(table_name)
                    if state and self.middle_pane.chart_widget:
                        k2_logger.info(f"Model state available for {table_name}", "ANALYSIS")
                        cw = self.middle_pane.chart_widget
                        agg = state.get('aggregation') if isinstance(state, dict) else None
                        if agg:
                            cw.change_timeframe(agg)
                except Exception as e:
                    k2_logger.debug(f"No model state available: {e}", "ANALYSIS")

                # Wire persistence of aggregation and view range
                try:
                    if self.middle_pane.chart_widget:
                        cw = self.middle_pane.chart_widget
                        cw.timeframe_changed.connect(
                            lambda tf, tn=table_name: saved_models_manager.set_model_state(
                                tn, indicators=None, active_strategy=None, chart_range=None
                            )
                        )
                except Exception:
                    pass
                    
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
        """
        Extract default parameters from indicator name.
        
        EXPECTATION: Returns user-friendly parameter names that will be 
        mapped to TA-Lib parameter names later.
        """
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
    
    def _map_indicator_params(self, ta_service_name: str, params: Dict) -> Dict:
        """
        Map user-friendly parameter names to TA-Lib parameter names.
        
        EXPECTATION: MUST convert all user-friendly parameter names to 
        TA-Lib standard parameter names. Returns mapped parameters dict.
        """
        mapped = params.copy()
        
        # Map common parameter names
        if 'period' in mapped:
            mapped['timeperiod'] = mapped.pop('period')
        
        # BBANDS specific mapping
        if ta_service_name == 'BBANDS':
            if 'std' in mapped:
                std_value = mapped.pop('std')
                mapped['nbdevup'] = std_value
                mapped['nbdevdn'] = std_value
        
        # MACD specific mapping
        if ta_service_name == 'MACD':
            if 'fast' in mapped:
                mapped['fastperiod'] = mapped.pop('fast')
            if 'slow' in mapped:
                mapped['slowperiod'] = mapped.pop('slow')
            if 'signal' in mapped:
                mapped['signalperiod'] = mapped.pop('signal')
        
        # Stochastic specific mapping
        if ta_service_name == 'STOCH':
            if 'k_period' in mapped:
                mapped['slowk_period'] = mapped.pop('k_period')
            if 'd_period' in mapped:
                mapped['slowd_period'] = mapped.pop('d_period')
        
        return mapped

    def _get_indicator_source_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Return a small, in-memory dataframe suitable for fast indicator calculation.

        PERFORMANCE OBJECTIVE:
        - Indicator toggles MUST NOT fetch the full dataset (millions of rows) on the UI thread.
        - Prefer using the chart widget's currently loaded data window (already chunked).

        Returns a dataframe in "display" format with columns:
        Date, Time, Open, High, Low, Close, Volume, VWAP
        """
        try:
            cw = getattr(self.middle_pane, "chart_widget", None)
            if cw is None:
                return None

            df = getattr(cw, "data", None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Only keep expected display columns to avoid copying large extras
                cols = [c for c in ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'datetime'] if c in df.columns]
                if cols:
                    return df[cols].copy()
                return df.copy()
        except Exception:
            return None

        return None
    
    def apply_indicator(self, indicator_name: str, params: Dict):
        """
        Apply indicator to current data.
        
        EXPECTATION: MUST validate calculation succeeds before adding to UI.
        MUST add indicator to both chart AND table.
        MUST map display names and parameters correctly.
        """
        try:
            # PERFORMANCE: Use the chart widget's currently loaded window.
            # Do NOT fetch full dataset for large models on indicator toggle.
            display_df = self._get_indicator_source_dataframe()
            if display_df is None or display_df.empty:
                k2_logger.warning("No chart window available for indicator calculation", "ANALYSIS")
                return

            # Build a timezone-naive datetime index (Date+Time when available)
            if 'Date' in display_df.columns and 'Time' in display_df.columns:
                dt_index = pd.to_datetime(
                    display_df['Date'].astype(str) + ' ' + display_df['Time'].astype(str),
                    errors='coerce'
                )
            elif 'Date' in display_df.columns:
                dt_index = pd.to_datetime(display_df['Date'], errors='coerce')
            elif 'datetime' in display_df.columns:
                dt_index = pd.to_datetime(display_df['datetime'], errors='coerce')
            else:
                k2_logger.warning("Indicator source dataframe missing Date column", "ANALYSIS")
                return

            if dt_index.isna().all():
                k2_logger.warning("Could not build datetime index for indicator calculation", "ANALYSIS")
                return

            # Prepare TA dataframe in expected lowercase schema
            df = display_df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'VWAP': 'vwap',
            }).copy()
            df['datetime'] = dt_index
            df.set_index('datetime', inplace=True)
            
            # Get base indicator name (display name)
            base_name = indicator_name.split("(")[0].strip()
            
            # Map display name to TA service name
            ta_service_name = ta_service.map_display_name_to_service_name(base_name)
            if ta_service_name is None:
                ta_service_name = base_name.upper()
            
            # Map parameters to TA-Lib format
            mapped_params = self._map_indicator_params(ta_service_name, params)
            
            # Calculate indicator
            indicator_data = ta_service.calculate_indicator(df, ta_service_name, mapped_params)
            
            # Keep a timezone-naive index for alignment
            df_index = df.index
            if isinstance(df_index, pd.DatetimeIndex) and df_index.tz is not None:
                df_index = df_index.tz_localize(None)
            
            # Use white color for all technical indicators for better visibility
            color = '#ffffff'
            
            # Handle multi-line indicators (like Bollinger Bands)
            if isinstance(indicator_data, dict):
                # Multi-line indicator - add each line separately
                for line_name, line_series in indicator_data.items():
                    full_name = f"{indicator_name} ({line_name})"
                    
                    # Remove timezone from index if present
                    if hasattr(line_series.index, 'tz') and line_series.index.tz is not None:
                        line_series.index = line_series.index.tz_localize(None)
                    
                    # Reindex to match dataframe
                    line_series = line_series.reindex(df_index)
                    line_series.name = full_name
                    
                    # Skip if all NaN
                    if line_series.isna().all():
                        continue
                    
                    # Add to chart and table
                    self.middle_pane.add_indicator(full_name, line_series, color)
                    
                # Store in applied indicators
                self.applied_indicators[indicator_name] = params
                k2_logger.info(f"Successfully applied multi-line indicator: {indicator_name}", "ANALYSIS")
                return
            
            # Single-line indicator
            # Validate indicator data before adding to UI
            if indicator_data is None:
                k2_logger.warning(f"Indicator calculation returned no data for {indicator_name}", "ANALYSIS")
                return
            
            if isinstance(indicator_data, pd.Series) and indicator_data.empty:
                k2_logger.warning(f"Indicator calculation returned empty data for {indicator_name}", "ANALYSIS")
                return
            
            # Check if we have valid values (not all NaN)
            if isinstance(indicator_data, pd.Series) and indicator_data.isna().all():
                k2_logger.warning(f"Indicator calculation returned only NaN values for {indicator_name}", "ANALYSIS")
                return
            
            # Ensure indicator data is a Series with datetime index matching the dataframe
            if isinstance(indicator_data, np.ndarray):
                indicator_data = pd.Series(indicator_data, index=df_index, name=indicator_name)
            elif isinstance(indicator_data, pd.Series):
                # Remove timezone from indicator index if present
                if hasattr(indicator_data.index, 'tz') and indicator_data.index.tz is not None:
                    indicator_data.index = indicator_data.index.tz_localize(None)
                
                # Reindex to match dataframe index
                indicator_data = indicator_data.reindex(df_index)
                indicator_data.name = indicator_name
            else:
                k2_logger.error(f"Unexpected indicator data type: {type(indicator_data)}", "ANALYSIS")
                return
            
            # Get indicator config to determine pane type (main overlay vs separate pane)
            indicator_config = ta_service.get_indicator_info(ta_service_name)
            pane_type = 'main'  # Default to overlay
            if indicator_config and hasattr(indicator_config, 'pane'):
                pane_type = indicator_config.pane
            
            # Add to chart based on pane type
            # Oscillators (RSI, Stochastic, MACD) go in separate panes with 0-100 Y-axis
            # Overlays (SMA, EMA, Bollinger) go on the main chart
            if pane_type == 'separate':
                self.middle_pane.add_indicator_pane(indicator_name, indicator_data, color)
            else:
                self.middle_pane.add_indicator(indicator_name, indicator_data, color)
            
            # Store in applied indicators
            self.applied_indicators[indicator_name] = params
            
            # PERFORMANCE: Do not persist indicators to DB on toggle.
            # Persisting requires full-dataset reads and full-column updates which can take 30+ seconds
            # on large (1MIN-20Y) models and blocks the UI thread.
            
            k2_logger.info(f"Successfully applied indicator: {indicator_name} (pane: {pane_type})", "ANALYSIS")
                
        except Exception as e:
            k2_logger.error(f"Error applying indicator {indicator_name}: {e}", "ANALYSIS")
    
    def remove_indicator(self, indicator_name: str):
        """Remove indicator from display (handles multi-line indicators like Bollinger Bands)"""
        try:
            # For multi-line indicators, remove all lines
            # Check if this is a base indicator name (e.g., "Bollinger Bands")
            base_name = indicator_name.split("(")[0].strip()
            
            # Try to remove the exact indicator name first
            self.middle_pane.remove_indicator(indicator_name)
            
            # Also remove any sub-indicators (e.g., "Bollinger Bands (upper)", etc.)
            sub_names = [f"{indicator_name} (upper)", f"{indicator_name} (middle)", f"{indicator_name} (lower)"]
            for sub_name in sub_names:
                try:
                    self.middle_pane.remove_indicator(sub_name)
                except:
                    pass
            
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
    
    def on_forecast_apply(self, forecast_data: dict):
        """Render forecast data as dashed OHLC lines on the chart."""
        k2_logger.info(
            f"Forecast apply received: {len(forecast_data)} set(s)", "ANALYSIS")
        cw = getattr(self.middle_pane, "chart_widget", None)
        if cw is None:
            k2_logger.warning("No chart widget — cannot render forecast lines", "ANALYSIS")
            return
        cw.add_forecast_data(forecast_data)

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

    def _on_ai_data_modified(self):
        """Refresh middle pane when the AI agent modifies table data."""
        if not self.current_model:
            return
        try:
            k2_logger.info("AI modified data — refreshing middle pane", "ANALYSIS")
            rows, total_count = stock_service.get_display_data(self.current_model, limit=500)
            if rows:
                self.current_data = rows
                self.current_metadata['records'] = total_count
                self.current_metadata['total_records'] = total_count
                self.middle_pane.load_data(rows, self.current_metadata)
                self.model_label.setText(f"Model: {self.current_model} ({total_count:,} records)")
        except Exception as e:
            k2_logger.error(f"Failed to refresh after AI data change: {e}", "ANALYSIS")

    def _get_workspace_state(self) -> Optional[Dict]:
        """Return current workspace DataFrames for the AI persistent engine."""
        tabs = getattr(self.middle_pane, "data_tabs", None)
        if tabs is None:
            return None
        result = {}
        for scope in ('model', 'global'):
            ws_df = tabs.get_working_data(scope)
            if ws_df is not None and not ws_df.empty:
                result[scope] = ws_df
        return result if result else None

    def _save_chat(self, table_name: str, html: str, history: list):
        """Persist chat to the database."""
        saved_models_manager.save_chat_history(table_name, html, history)

    def _load_chat(self, table_name: str) -> Optional[Dict]:
        """Load chat from the database."""
        return saved_models_manager.get_chat_history(table_name)

    def _on_tab_writes(self, writes: list):
        """Route AI-generated data to the correct UI tabs."""
        if not writes:
            return
        tabs = getattr(self.middle_pane, "data_tabs", None)
        if tabs is None:
            k2_logger.warning("DataTabsWidget not available for tab writes", "ANALYSIS")
            return

        for w in writes:
            try:
                wtype = w.get("type")
                if wtype == "working":
                    scope = w.get("scope", "model")
                    col = w.get("column_name", "result")
                    vals = w.get("values", [])
                    grid_col = w.get("column")
                    tabs.add_working_column(scope, col, vals, column=grid_col)
                    letter_info = f" col {grid_col}" if grid_col else ""
                    k2_logger.info(
                        f"AI wrote '{col}' ({len(vals)} rows){letter_info}"
                        f" to working tab [{scope}]",
                        "ANALYSIS")
                elif wtype == "delete_working":
                    scope = w.get("scope", "model")
                    col = w.get("column_name", "")
                    if col:
                        tabs.delete_working_column(scope, col)
                        k2_logger.info(
                            f"AI deleted '{col}' from working tab [{scope}]",
                            "ANALYSIS")
                elif wtype == "forecast":
                    set_idx = w.get("set_index", 1)
                    while tabs._forecast_sets < set_idx:
                        tabs.add_forecast_set()
                    for ohlc, key in [("Open", "open_values"), ("High", "high_values"),
                                      ("Low", "low_values"), ("Close", "close_values")]:
                        vals = w.get(key)
                        if vals:
                            col_name = f"{ohlc}_P{set_idx}"
                            tabs.set_forecast_values(col_name, vals)
                    k2_logger.info(
                        f"AI wrote forecast set P{set_idx} to forecast tab",
                        "ANALYSIS")
            except Exception as e:
                k2_logger.error(f"Failed to process tab write: {e}", "ANALYSIS")
    
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