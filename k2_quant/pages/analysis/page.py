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

import json
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
from k2_quant.utilities.data.data_store import data_store
from k2_quant.utilities.services.strategy_service import strategy_service
from k2_quant.utilities.report_helpers import format_blocks_plain
from k2_quant.utilities.numeric_rounding import round_dataframe_numeric_columns
from k2_quant.utilities.services.dynamic_python_engine import dpe_service
from k2_quant.utilities.services.strategy_runner import strategy_runner

# Import the three pane components
from PyQt6.QtWidgets import QTabWidget

from k2_quant.pages.analysis.components.left_pane import LeftPaneWidget
from k2_quant.pages.analysis.components.middle_pane import MiddlePaneWidget
from k2_quant.pages.analysis.components.right_pane import (
    RightPaneWidget,
    THINKSPACE_DEFAULT_WIDTH,
    THINKSPACE_MIN_WIDTH,
    THINKSPACE_MAX_WIDTH,
)
from k2_quant.pages.analysis.components.outputs_panel import OutputsPanel


_INTRADAY_TIMESPANS = {'minute', 'min', 'hour'}


def _should_filter_market_hours(metadata: dict) -> bool:
    """Return True when the model's data is intraday and market-hours filtering applies."""
    if metadata.get('market_hours_only', False):
        return True
    ts = str(metadata.get('timespan', '')).lower()
    return any(ts.startswith(prefix) for prefix in _INTRADAY_TIMESPANS)


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

        # Strategy-runner wiring. Same contract as stream_window: signals
        # are shared across pages, so handlers filter on ``_strategy_jobs``.
        self._strategy_jobs: Dict[str, str] = {}
        strategy_runner.started.connect(self._on_strategy_started)
        strategy_runner.progress.connect(self._on_strategy_progress)
        strategy_runner.finished.connect(self._on_strategy_finished)
        strategy_runner.failed.connect(self._on_strategy_failed)
        strategy_runner.cancelled.connect(self._on_strategy_cancelled)

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
        self.outputs_panel = OutputsPanel()

        # Tabbed right container (Thinkspace + Outputs)
        self.right_tabs = QTabWidget()
        self.right_tabs.setObjectName("rightTabs")
        self.right_tabs.setMinimumWidth(THINKSPACE_MIN_WIDTH)
        self.right_tabs.setMaximumWidth(THINKSPACE_MAX_WIDTH)
        self.right_tabs.addTab(self.right_pane, "THINKSPACE")
        self.right_tabs.addTab(self.outputs_panel, "OUTPUTS")
        self.right_tabs.setStyleSheet("""
            #rightTabs { background: #0a0a0a; border: none; }
            #rightTabs::pane { border: none; background: #0a0a0a; }
            #rightTabs QTabBar::tab {
                background: #1a1a1a; color: #999;
                padding: 6px 18px; border: none;
                border-bottom: 2px solid transparent;
                font-size: 11px; font-weight: 600; letter-spacing: 1px;
            }
            #rightTabs QTabBar::tab:selected {
                color: #4a9eff; border-bottom: 2px solid #4a9eff;
                background: #0a0a0a;
            }
            #rightTabs QTabBar::tab:hover:!selected {
                color: #ccc; background: #151515;
            }
        """)
        self.right_tabs.currentChanged.connect(self._on_right_tab_changed)

        # Wire up signals
        self.setup_connections()
        
        # Add to splitter
        self.splitter.addWidget(self.left_pane)
        self.splitter.addWidget(self.middle_pane)
        self.splitter.addWidget(self.right_tabs)

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)

        self.splitter.setSizes([280, 740, THINKSPACE_DEFAULT_WIDTH])
        
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
        self.left_pane.strategy_deleted.connect(self.on_strategy_deleted)
        self.left_pane.indicator_toggled.connect(self.on_indicator_toggled)
        
        # Middle pane connections
        self.middle_pane.column_toggled.connect(self.on_column_toggled)
        self.middle_pane.view_mode_changed.connect(self.on_view_mode_changed)
        self.middle_pane.indicator_applied.connect(self.on_indicator_applied_from_middle)
        self.middle_pane.data_exported.connect(self.on_data_exported)
        self.middle_pane.forecast_apply.connect(self.on_forecast_apply)
        self.middle_pane.forecast_column_toggled.connect(self.on_forecast_column_toggled)
        
        # Right pane connections
        self.right_pane.message_sent.connect(self.on_ai_message_sent)
        self.right_pane.strategy_generated.connect(self.on_strategy_generated)
        self.right_pane.strategy_removed_remotely.connect(
            self.on_strategy_removed_by_thinkspace)
        self.right_pane.data_modified.connect(self._on_ai_data_modified)
        self.right_pane.tab_writes_ready.connect(self._on_tab_writes)
        self.right_pane.workspace_provider = self._get_workspace_state
        self.right_pane.save_chat_callback = self._save_chat
        self.right_pane.load_chat_callback = self._load_chat

        # Outputs panel connections
        self.outputs_panel.reference_in_chat.connect(self._on_reference_run_in_chat)
    
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
        
        # Persist tab data and active strategies for the previous model
        if self.current_model:
            self.middle_pane.persist_tab_data(self.current_model)
            import json
            saved_models_manager.set_model_state(
                self.current_model,
                indicators=None,
                active_strategy=json.dumps(sorted(self.left_pane.active_strategies)),
                chart_range=None,
            )
        
        # Clear UI before loading new model (signals blocked — no cascading side effects)
        self.left_pane.clear_all_indicators()
        self.left_pane.clear_all_strategies()
        
        try:
            from k2_quant.utilities.data.db_manager import db_manager as _db
            cached_meta = data_store.get_metadata(table_name)
            if cached_meta:
                base_metadata = cached_meta
            else:
                base_metadata = saved_models_manager.get_model_metadata(table_name) or {'symbol': table_name}
                data_store.set_metadata(table_name, base_metadata)
            mkt_hours = _should_filter_market_hours(base_metadata)

            cached_display = data_store.get_display(table_name, mkt_hours)
            if cached_display is not None:
                rows, total_count = cached_display
            else:
                rows, total_count = stock_service.get_display_data(
                    table_name, limit=500, market_hours_only=mkt_hours)
                if rows:
                    data_store.set_display(table_name, mkt_hours, rows, total_count)

            if rows:
                self.current_model = table_name
                self.current_data = rows

                parts = table_name.split('_')
                symbol = parts[1].upper() if len(parts) > 1 else 'UNKNOWN'
                cached_hrn = data_store.get_has_row_number(table_name)
                if cached_hrn is None:
                    has_row_number = _db._check_column_exists(table_name, '#')
                    data_store.set_has_row_number(table_name, has_row_number)
                else:
                    has_row_number = cached_hrn

                self.current_metadata = dict(base_metadata)
                self.current_metadata.update({
                    'records': total_count,
                    'table_name': table_name,
                    'total_records': total_count,
                    'symbol': symbol,
                    'has_row_number': has_row_number,
                    'market_hours_only': mkt_hours,
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
                    'table_name': table_name,
                    'market_hours_only': mkt_hours,
                }
                self.right_pane.set_data_context(ctx)
                
                # Restore model state (chart timeframe + strategy checkboxes)
                try:
                    import json
                    state = saved_models_manager.get_model_state(table_name)
                    if state and isinstance(state, dict):
                        if self.middle_pane.chart_widget:
                            agg = state.get('aggregation')
                            if agg:
                                self.middle_pane.chart_widget.change_timeframe(agg)
                        raw = state.get('active_strategy') or '[]'
                        try:
                            names = set(json.loads(raw))
                        except (json.JSONDecodeError, TypeError):
                            names = {raw} if raw else set()
                        if names:
                            self.left_pane.restore_strategies(names)
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

            # Build a timezone-naive datetime index. Prefer the chart's
            # already-precomputed 'datetime' column (see _precompute_datetime_column
            # in chart/main.py) to avoid reparsing millions of Date+Time strings
            # on every indicator toggle.
            if 'datetime' in display_df.columns:
                dt_index = pd.to_datetime(display_df['datetime'], errors='coerce')
            elif 'Date' in display_df.columns and 'Time' in display_df.columns:
                dt_index = pd.to_datetime(
                    display_df['Date'].astype(str) + ' ' + display_df['Time'].astype(str),
                    errors='coerce'
                )
            elif 'Date' in display_df.columns:
                dt_index = pd.to_datetime(display_df['Date'], errors='coerce')
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
        """Apply strategy - dispatched to ``strategy_runner`` so the GUI stays
        responsive on large minute-bar models. Post-processing happens in
        ``_on_strategy_finished``."""
        table_name = self.current_model
        code = strategy_service.get_strategy_code(strategy_name)
        if not code:
            k2_logger.warning(f"Strategy code not found: {strategy_name}", "ANALYSIS")
            return

        mkt_hours = bool(self.current_metadata.get('market_hours_only', False))
        has_rn = bool(self.current_metadata.get('has_row_number', False))

        def _loader():
            return self._build_strategy_frame(table_name, mkt_hours, has_rn)

        key = strategy_runner.submit(
            table_name=table_name,
            strategy_name=strategy_name,
            code=code,
            loader=_loader,
        )
        if key is None:
            return
        self._strategy_jobs[key] = strategy_name

    @staticmethod
    def _build_strategy_frame(table_name: str, mkt_hours: bool,
                              has_rn: bool) -> pd.DataFrame:
        """Runs on the strategy worker thread - no Qt calls."""
        import time as _time
        t0 = _time.time()
        rows, _total = stock_service.get_display_data(
            table_name, limit=10**9, market_hours_only=mkt_hours)
        t_fetch = (_time.time() - t0) * 1000.0
        k2_logger.info(
            f"[loader] db_fetch took={t_fetch:,.0f} ms rows={len(rows):,} "
            f"table={table_name} mkt_hours={mkt_hours}",
            "ANALYSIS",
        )

        if has_rn:
            all_columns = ['#','Date','Time','Open','High','Low','Close','Volume','VWAP',
                            'Open_%','High_%','Low_%','Close_%','Elasticity','Close-Open_%']
        else:
            all_columns = ['Date','Time','Open','High','Low','Close','Volume','VWAP',
                            'Open_%','High_%','Low_%','Close_%','Elasticity','Close-Open_%']

        t1 = _time.time()
        df = pd.DataFrame(rows, columns=all_columns[:len(rows[0])] if rows else all_columns[:8])
        t_build = (_time.time() - t1) * 1000.0
        k2_logger.info(
            f"[loader] df_build took={t_build:,.0f} ms shape={df.shape}",
            "ANALYSIS",
        )

        t2 = _time.time()
        # ``Date`` arrives as datetime.date, ``Time`` as datetime.time from psycopg2.
        # The previous implementation did ``Date.astype(str) + ' ' + Time.astype(str)``
        # then ``pd.to_datetime`` with format-inference: on 1.9M rows that is
        # ~1s of pure Python string work + C parse. Passing an explicit
        # ``format`` lets pandas take the fast strptime path (~3-5x faster),
        # and we only pay the construction cost once.
        date_str = df['Date'].astype(str)
        time_str = df['Time'].astype(str)
        df['date_time_market'] = pd.to_datetime(
            date_str + ' ' + time_str,
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce',
            cache=True,
        )
        df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume','VWAP':'vwap'})
        df = df.rename(columns={
            'Open_%': 'open_pct',
            'High_%': 'high_pct',
            'Low_%': 'low_pct',
            'Close_%': 'close_pct',
            'Elasticity': 'elasticity',
            'Close-Open_%': 'close_open_pct',
        })
        recomputed = False
        if 'open_pct' not in df.columns and all(c in df.columns for c in ('open', 'high', 'low', 'close')):
            recomputed = True
            df = df.sort_values('date_time_market', kind='mergesort').reset_index(drop=True)
            # Coerce each price column to float once, then reuse. With the
            # NUMERIC->float typecaster in db_manager these are already float
            # and ``astype(float)`` is a no-op fast path; this keeps
            # correctness on legacy connections while avoiding 4x redundant
            # ``astype`` calls.
            o_ = df['open'].astype(float, copy=False)
            hi = df['high'].astype(float, copy=False)
            lo = df['low'].astype(float, copy=False)
            cl = df['close'].astype(float, copy=False)
            for name, cur in (('open', o_), ('high', hi), ('low', lo), ('close', cl)):
                prev = cur.shift(1)
                df[f'{name}_pct'] = np.where(
                    (prev != 0) & prev.notna() & cur.notna(),
                    (cur - prev) / prev * 100.0,
                    np.nan,
                )
            df['elasticity'] = np.where(lo != 0, (hi - lo) / lo * 100.0, np.nan)
            df['close_open_pct'] = np.where(o_ != 0, (cl - o_) / o_ * 100.0, np.nan)
        t_derive = (_time.time() - t2) * 1000.0
        k2_logger.info(
            f"[loader] derive_cols took={t_derive:,.0f} ms recomputed_pct={recomputed}",
            "ANALYSIS",
        )

        t3 = _time.time()
        out = round_dataframe_numeric_columns(df)
        t_round = (_time.time() - t3) * 1000.0
        k2_logger.info(
            f"[loader] round_numeric took={t_round:,.0f} ms",
            "ANALYSIS",
        )
        k2_logger.info(
            f"[loader] total={(t_fetch + t_build + t_derive + t_round):,.0f} ms",
            "ANALYSIS",
        )
        return out

    # ── Strategy runner signal handlers (GUI thread) ─────────────────
    def _own_strategy_key(self, key: str) -> bool:
        return key in self._strategy_jobs

    def _on_strategy_started(self, key: str):
        if not self._own_strategy_key(key):
            return
        name = self._strategy_jobs.get(key, '?')
        k2_logger.info(f"Strategy started: {name}", "ANALYSIS")

    def _on_strategy_progress(self, key: str, message: str):
        if not self._own_strategy_key(key):
            return
        name = self._strategy_jobs.get(key, '?')
        k2_logger.info(f"Strategy '{name}' progress: {message}", "ANALYSIS")

    def _on_strategy_failed(self, key: str, error: str):
        if not self._own_strategy_key(key):
            return
        name = self._strategy_jobs.pop(key, '?')
        k2_logger.error(f"Strategy '{name}' failed: {error}", "ANALYSIS")

    def _on_strategy_cancelled(self, key: str):
        if not self._own_strategy_key(key):
            return
        name = self._strategy_jobs.pop(key, '?')
        k2_logger.info(f"Strategy '{name}' cancelled", "ANALYSIS")

    def _on_strategy_finished(self, key: str, result: object):
        if not self._own_strategy_key(key):
            return
        strategy_name = self._strategy_jobs.pop(key, None)
        if strategy_name is None or not isinstance(result, dict):
            return

        table_name = self.current_model
        code = strategy_service.get_strategy_code(strategy_name) or ''
        strategy_service.save_run(
            strategy_name=strategy_name,
            code_snapshot=code,
            result=result,
            model_table=table_name,
        )
        self.outputs_panel.refresh()

        if not result.get('success'):
            k2_logger.error(f"Strategy execution failed: {result.get('error')}", "ANALYSIS")
            return

        tab_writes = result.get('_tab_writes', [])
        forecast_writes = [w for w in tab_writes if w.get("type") == "forecast"]
        working_writes = [w for w in tab_writes if w.get("type") == "working"]

        if working_writes:
            self._on_tab_writes(working_writes)
            k2_logger.info(
                f"Strategy '{strategy_name}' wrote {len(working_writes)} working column(s)",
                "ANALYSIS",
            )

        if forecast_writes:
            for w in forecast_writes:
                w['_strategy'] = strategy_name
            self._on_tab_writes(forecast_writes)
            saved_models_manager.set_model_state(
                table_name,
                indicators=None,
                active_strategy=strategy_name,
                chart_range=None,
            )
            k2_logger.info(
                f"Strategy '{strategy_name}' wrote {len(forecast_writes)} forecast column(s)",
                "ANALYSIS",
            )
            return

        result_df = result.get('data')
        if not isinstance(result_df, pd.DataFrame):
            return
        metrics = result.get('metrics') or {}
        original_rows = int(metrics.get('original_rows') or 0)
        proj_df = (result_df.iloc[original_rows:].copy()
                   if original_rows and len(result_df) > original_rows
                   else pd.DataFrame())

        if not proj_df.empty:
            stock_service.delete_projections(table_name, strategy_name)
            stock_service.insert_projections(table_name, proj_df, strategy_name)
            saved_models_manager.set_model_state(
                table_name,
                indicators=None,
                active_strategy=strategy_name,
                chart_range=None,
            )
            mkt_hours = self.current_metadata.get('market_hours_only', False)
            rows, total_count = stock_service.get_display_data(
                table_name, limit=500, market_hours_only=mkt_hours)
            self.middle_pane.load_data(rows, self.current_metadata)
            self.model_label.setText(f"Model: {table_name} ({total_count:,} records)")

    def remove_strategy(self, strategy_name: str):
        """Remove strategy projections and clear that strategy's forecast columns."""
        table_name = self.current_model

        key = strategy_runner.make_key(table_name or '', strategy_name)
        if strategy_runner.is_running(key):
            strategy_runner.cancel(key)

        try:
            stock_service.delete_projections(table_name, strategy_name)
        except Exception:
            pass

        tabs = getattr(self.middle_pane, "data_tabs", None)
        cw = getattr(self.middle_pane, "chart_widget", None)

        if tabs is not None:
            col_names = list(tabs._strategy_columns.get(strategy_name, []))
            tabs.clear_strategy_columns(strategy_name)
            if cw is not None:
                for col_name in col_names:
                    cw.remove_forecast_line(col_name)

        saved_models_manager.set_model_state(
            table_name,
            indicators=None,
            active_strategy=None,
            chart_range=None
        )
        
        mkt_hours = self.current_metadata.get('market_hours_only', False)
        rows, total_count = stock_service.get_display_data(
            table_name, limit=500, market_hours_only=mkt_hours)
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
        """Clear all forecast lines when an empty dict arrives."""
        cw = getattr(self.middle_pane, "chart_widget", None)
        if cw is None:
            k2_logger.warning("No chart widget -- cannot render forecast lines", "ANALYSIS")
            return
        if not forecast_data:
            cw.clear_forecast_data()
            k2_logger.info("Forecast lines cleared from chart", "ANALYSIS")

    def on_forecast_column_toggled(self, column_name: str, visible: bool):
        """Show or hide a single named forecast line on the chart."""
        cw = getattr(self.middle_pane, "chart_widget", None)
        if cw is None:
            return
        tabs = getattr(self.middle_pane, "data_tabs", None)
        if visible:
            if tabs is not None:
                values = tabs.get_forecast_column_data(column_name)
                anchor = tabs.get_forecast_column_anchor(column_name)
                if values:
                    cw.add_forecast_line(column_name, values,
                                         anchor_price=anchor)
        else:
            cw.remove_forecast_line(column_name)

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
        """Handle strategy generation from AI — refresh left pane and outputs."""
        k2_logger.info(f"Strategy generated: {name}", "ANALYSIS")
        self.refresh_left_pane_data()
        self.outputs_panel.refresh()

    def on_strategy_removed_by_thinkspace(self, name: str):
        """delete_strategy tool already purged DB + runs; refresh panes and projections."""
        name = (name or "").strip()
        if not name:
            return
        k2_logger.info(f"Strategy removed via Thinkspace: {name}", "ANALYSIS")
        self.left_pane.discard_active_strategy(name)
        if self.current_model:
            self.remove_strategy(name)
        self.refresh_left_pane_data()
        self.outputs_panel.handle_strategy_deleted(name)

    def on_strategy_deleted(self, name: str):
        """Delete strategy from DB, clean up any active projections, refresh UI."""
        name = (name or "").strip()
        if not name:
            return
        k2_logger.info(f"Strategy deleted: {name}", "ANALYSIS")
        ok = strategy_service.delete_strategy(name)
        if not ok:
            k2_logger.error(f"Strategy delete failed for {name!r}", "ANALYSIS")
        if self.current_model:
            self.remove_strategy(name)
        self.refresh_left_pane_data()
        self.outputs_panel.handle_strategy_deleted(name)
    
    def _on_right_tab_changed(self, index: int):
        """Auto-refresh the Outputs panel when its tab is selected."""
        if index == 1:
            self.outputs_panel.refresh()

    def _on_reference_run_in_chat(self, run_id: int, strategy_name: str):
        """User clicked 'Send to Thinkspace' on a run — switch tab and inject context."""
        run = strategy_service.get_run(run_id)
        if not run:
            return
        summary = (
            f"[Strategy Run Report]\n"
            f"Strategy: {run['strategy_name']}\n"
            f"Model: {run.get('model_table', '--')}\n"
            f"Time: {run['run_timestamp']}\n"
            f"Status: {'SUCCESS' if run.get('success') else 'FAILED'}\n"
        )
        rb = run.get("report_blocks_json")
        if rb:
            try:
                blocks = json.loads(rb)
                if isinstance(blocks, list) and blocks:
                    plain = format_blocks_plain(blocks)
                    if plain:
                        summary += f"\nReport:\n{plain}\n"
            except (json.JSONDecodeError, TypeError):
                pass
        stdout = (run.get('stdout_output') or '').strip()
        if stdout:
            summary += f"\nLog (stdout):\n{stdout}\n"
        error = (run.get('error_output') or '').strip()
        if error:
            summary += f"\nError:\n{error}\n"
        code = (run.get('code_snapshot') or '').strip()
        if code:
            summary += f"\nCode:\n{code}\n"

        self.right_pane.conversation_history.append({
            'role': 'user',
            'content': summary,
        })
        self.right_tabs.setCurrentIndex(0)
        k2_logger.info(f"Run {run_id} injected into Thinkspace context", "ANALYSIS")

    def _on_ai_data_modified(self):
        """Refresh middle pane when the AI agent modifies table data."""
        if not self.current_model:
            return
        try:
            k2_logger.info("AI modified data -- refreshing middle pane", "ANALYSIS")
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
        for name in tabs.get_sheet_names():
            ws_df = tabs.get_working_data('model', sheet=name)
            if ws_df is not None and not ws_df.empty:
                result[name] = ws_df
        return {'sheets': result} if result else None

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
                    sheet = w.get("sheet")
                    tabs.add_working_column(scope, col, vals, column=grid_col, sheet=sheet)
                    letter_info = f" col {grid_col}" if grid_col else ""
                    sheet_info = f" sheet '{sheet}'" if sheet else ""
                    k2_logger.info(
                        f"AI wrote '{col}' ({len(vals)} rows){letter_info}{sheet_info}",
                        "ANALYSIS")
                elif wtype == "delete_working":
                    scope = w.get("scope", "model")
                    col = w.get("column_name", "")
                    sheet = w.get("sheet")
                    if col:
                        tabs.delete_working_column(scope, col, sheet=sheet)
                        sheet_info = f" sheet '{sheet}'" if sheet else ""
                        k2_logger.info(
                            f"AI deleted '{col}'{sheet_info}",
                            "ANALYSIS")
                elif wtype == "forecast":
                    if "column_name" in w:
                        col_name = w["column_name"]
                        vals = w.get("values", [])
                        anchor = w.get("anchor_price")
                        strategy = w.get("_strategy", "AI")
                        if vals:
                            tabs.set_forecast_column(strategy, col_name, vals,
                                                     anchor_price=anchor)
                        k2_logger.info(
                            f"Forecast column '{col_name}' written "
                            f"(strategy: {strategy})", "ANALYSIS")
                    else:
                        strategy = w.get("_strategy", "AI")
                        set_idx = w.get("set_index", 1)
                        for ohlc, key in [("Open", "open_values"),
                                          ("High", "high_values"),
                                          ("Low", "low_values"),
                                          ("Close", "close_values")]:
                            vals = w.get(key)
                            if vals:
                                col_name = f"{ohlc}_P{set_idx}"
                                tabs.set_forecast_column(strategy, col_name, vals)
                        k2_logger.info(
                            f"Legacy forecast set P{set_idx} written "
                            f"(strategy: {strategy})", "ANALYSIS")
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
            if self.current_model:
                self.middle_pane.persist_tab_data(self.current_model)
                self._save_chat(
                    self.current_model,
                    json.dumps(self.right_pane._message_records),
                    list(self.right_pane.conversation_history))

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