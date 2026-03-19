"""
Middle Pane Component - Data Visualization

Contains chart and data table with controls.
Save as: k2_quant/pages/analysis/components/middle_pane.py

EXPECTATIONS:
=============
This component manages both chart and table display with the following objectives:

1. INDICATOR DISPLAY:
   - MUST display indicators on the chart as visual overlays/lines
   - MUST display indicator values in the table as additional columns
   - MUST align indicator data with table rows by timestamp/date
   - MUST update table dynamically when indicators are added/removed

2. DATA SYNCHRONIZATION:
   - MUST keep chart and table data synchronized
   - MUST merge indicator Series data into table DataFrame
   - MUST handle indicators with different data lengths or missing values
   - MUST preserve existing table columns when adding indicators

3. TABLE MANAGEMENT:
   - MUST support dynamic column addition/removal for indicators
   - MUST format indicator values appropriately (numeric with proper decimals)
   - MUST maintain table performance with large datasets
   - MUST update table headers when indicators are added/removed

4. INDICATOR STATE:
   - MUST track active indicators in active_indicators dictionary
   - MUST remove indicator data from both chart and table when indicator is removed
   - MUST handle indicator updates when parameters change
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QComboBox, QLabel, QSplitter,
                             QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal

from k2_quant.utilities.logger import k2_logger

# Updated import path for ChartWidget facade
from k2_quant.pages.analysis.widgets.chart import ChartWidget
from k2_quant.pages.analysis.widgets.data_tabs_widget import DataTabsWidget


class MiddlePaneWidget(QFrame):
    """Middle pane with chart and data table"""
    
    # Signals
    data_updated = pyqtSignal()
    column_toggled = pyqtSignal(str, bool)  # column_name, visible
    view_mode_changed = pyqtSignal(str)  # mode
    projection_requested = pyqtSignal()
    indicator_applied = pyqtSignal(str, dict)  # indicator_type, params
    data_exported = pyqtSignal(str, pd.DataFrame)  # format, data
    forecast_apply = pyqtSignal(dict)  # forwarded from DataTabsWidget
    forecast_column_toggled = pyqtSignal(str, bool)  # column_name, visible
    
    def __init__(self):
        super().__init__()
        self.setObjectName("middlePane")
        
        self.current_data = None
        self.current_metadata = None
        self.current_table_name = None
        self.total_records = 0
        self.active_columns = []
        self.active_indicators = {}
        
        # Create widgets
        self.chart_widget = None
        self.data_tabs = None
        self.view_selector = None
        self.controls_bar = None
        self.splitter = None
        self.empty_placeholder = None
        self.status_label = None
        self.loading_bar = None
        
        self.init_ui()
        self.setup_styling()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        
        # Create empty state placeholder (do not add to final layout to avoid occupying space)
        self.empty_placeholder = QLabel("Select a model to view data")
        self.empty_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_placeholder.setObjectName("emptyPlaceholder")
        self.empty_placeholder.hide()
        layout.addWidget(self.empty_placeholder)
    
    def create_controls_bar(self) -> QWidget:
        """Create top controls bar with simplified navigation"""
        controls = QWidget()
        controls.setFixedHeight(40)
        controls.setObjectName("chartControls")
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        controls.setLayout(layout)
        
        layout.addWidget(QLabel("VIEW:"))
        
        self.view_selector = QComboBox()
        self.view_selector.addItems(["Chart", "Data Table", "Both"])
        self.view_selector.setCurrentText("Both")
        self.view_selector.currentTextChanged.connect(self.change_view)
        layout.addWidget(self.view_selector)

        # Viewport/status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Only keep the working Reset and End buttons
        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Reset view to last 5 days")
        reset_btn.clicked.connect(lambda: self.chart_widget.reset_zoom() if self.chart_widget else None)
        layout.addWidget(reset_btn)
        
        end_btn = QPushButton("End")
        end_btn.setToolTip("Jump to latest data")
        end_btn.clicked.connect(self.jump_to_end)
        layout.addWidget(end_btn)
        
        return controls
    
    def create_data_interface(self):
        """Create the chart and table interface (called when data is first loaded)"""
        # Get main layout
        layout = self.layout()
        
        # Hide placeholder if present
        if self.empty_placeholder and self.empty_placeholder.isVisible():
            self.empty_placeholder.hide()
        
        # Controls bar
        if not self.controls_bar:
            self.controls_bar = self.create_controls_bar()
            layout.insertWidget(0, self.controls_bar)
        
        # Loading bar
        if not self.loading_bar:
            self.loading_bar = QProgressBar()
            self.loading_bar.setFixedHeight(3)
            self.loading_bar.setTextVisible(False)
            self.loading_bar.hide()
            layout.insertWidget(1, self.loading_bar)
        
        # Create splitter if not exists
        if not self.splitter:
            self.splitter = QSplitter(Qt.Orientation.Vertical)
            self.splitter.setHandleWidth(2)
            self.splitter.setObjectName("dataSplitter")
            
            # Import and create chart widget
            self.chart_widget = ChartWidget()
            
            # Connect chart signals
            self.chart_widget.data_loading.connect(self.on_data_loading)
            self.chart_widget.data_loaded.connect(self.on_data_loaded)
            self.chart_widget.viewport_changed.connect(self.on_viewport_changed)
            
            
            self.splitter.addWidget(self.chart_widget)
            
            # Create tabbed data widget (Current / Forecast / Working)
            self.data_tabs = DataTabsWidget()
            self.data_tabs.forecast_apply.connect(self.forecast_apply)
            self.data_tabs.forecast_column_toggled.connect(self.forecast_column_toggled)
            self.splitter.addWidget(self.data_tabs)
            
            # Set initial sizes (60/40 split)
            self.splitter.setSizes([360, 240])
            
            layout.addWidget(self.splitter)
    
    def change_view(self, view_type: str):
        """Change the middle pane view"""
        if not self.chart_widget or not self.data_tabs:
            return
            
        if view_type == "Chart":
            self.chart_widget.setVisible(True)
            self.data_tabs.setVisible(False)
        elif view_type == "Data Table":
            self.chart_widget.setVisible(False)
            self.data_tabs.setVisible(True)
        elif view_type == "Both":
            self.chart_widget.setVisible(True)
            self.data_tabs.setVisible(True)
            self.splitter.setSizes([360, 240])
        
        self.view_mode_changed.emit(view_type)
        k2_logger.info(f"View changed to: {view_type}", "MIDDLE_PANE")
    
    def load_data(self, data: Union[List, pd.DataFrame], metadata: Dict = None):
        """Load data into chart and table"""
        k2_logger.info("Loading data into middle pane", "MIDDLE_PANE")
        
        self.current_metadata = metadata or {}
        self.current_table_name = self.current_metadata.get('table_name')
        self.total_records = int(self.current_metadata.get('total_records', 0) or 0)
        
        # Convert to DataFrame if needed (limited data for table)
        if isinstance(data, list) and len(data) > 0:
            has_row_number = self.current_metadata.get('has_row_number', False)
            if has_row_number:
                columns = ['#', 'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP',
                            'Open_%', 'High_%', 'Low_%', 'Close_%', 'Elasticity', 'Close-Open_%']
            else:
                columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP',
                            'Open_%', 'High_%', 'Low_%', 'Close_%', 'Elasticity', 'Close-Open_%']
            df = pd.DataFrame(data, columns=columns[:len(data[0])])
            self.current_data = df
        elif isinstance(data, pd.DataFrame):
            self.current_data = data.copy()
        else:
            k2_logger.error("Invalid data format", "MIDDLE_PANE")
            return
        
        # Ensure the empty placeholder is hidden before (re)using the interface
        layout = self.layout()
        if self.empty_placeholder and self.empty_placeholder.isVisible():
            self.empty_placeholder.hide()

        # Create UI if first time
        if not self.chart_widget:
            self.create_data_interface()
        
        # Chart: require DB source; do not fallback to limited DataFrame
        if self.chart_widget:
            if self.current_table_name and self.total_records > 0:
                self.chart_widget.load_data_from_table(
                    table_name=self.current_table_name,
                    total_records=self.total_records,
                    metadata=self.current_metadata
                )
            else:
                k2_logger.warning("No table_name provided; chart requires DB source", "MIDDLE_PANE")
                if self.status_label:
                    self.status_label.setText("Chart requires DB source")
        
        # Table: limited data
        self.load_data_into_table(self.current_data)
        
        # Setup forecast timestamps for Tab 2
        if self.data_tabs and self.current_data is not None and len(self.current_data) > 0:
            last_row = self.current_data.iloc[-1]
            self.data_tabs.set_model_context(self.current_table_name)
            last_row_number = None
            if '#' in self.current_data.columns:
                last_row_number = self.total_records or int(self.current_data['#'].iloc[-1])
            self.data_tabs.setup_forecast(
                last_date=last_row.get('Date', last_row.iloc[0]),
                last_time=last_row.get('Time', last_row.iloc[1]),
                timespan=self.current_metadata.get('timespan', 'minute'),
                frequency=self.current_metadata.get('frequency', '1'),
                market_hours_only=self.current_metadata.get('market_hours_only', False),
                last_row_number=last_row_number,
            )
        
        # Initial status
        if self.status_label and self.total_records > 0:
            table_rows = len(self.current_data) if self.current_data is not None else 0
            if table_rows < self.total_records:
                self.status_label.setText(
                    f"Table: {table_rows:,} rows | Chart: {self.total_records:,} total"
                )
            else:
                self.status_label.setText(f"Total: {self.total_records:,} records")
        
        self.data_updated.emit()
    
    def load_data_into_table(self, data):
        """
        Load data into the table widget.
        
        EXPECTATION: MUST merge active indicator data into the table DataFrame.
        MUST include indicator columns in the table display.
        """
        if data is None or not self.data_tabs:
            return
        
        # Convert to DataFrame if needed
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, list):
            has_row_number = self.current_metadata.get('has_row_number', False) if self.current_metadata else False
            if has_row_number:
                columns = ['#', 'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP',
                            'Open_%', 'High_%', 'Low_%', 'Close_%', 'Elasticity', 'Close-Open_%']
            else:
                columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP',
                            'Open_%', 'High_%', 'Low_%', 'Close_%', 'Elasticity', 'Close-Open_%']
            df = pd.DataFrame(data, columns=columns[:len(data[0])] if data else columns)
        else:
            return
        
        # Merge active indicators into DataFrame
        # Indicators are stored as Series with datetime index.
        #
        # IMPORTANT:
        # The table is not necessarily a contiguous time slice (for large datasets we load a
        # sample of oldest + newest rows). Therefore, indicator values MUST be aligned to
        # each table row by timestamp (Date+Time), not by positional slicing.
        if self.active_indicators:
            table_len = len(df)

            table_index = None
            try:
                if 'Date' in df.columns and 'Time' in df.columns:
                    dt = pd.to_datetime(
                        df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                        errors='coerce'
                    )
                    if not dt.isna().all():
                        table_index = pd.DatetimeIndex(dt)
            except Exception:
                table_index = None

            for indicator_name, indicator_series in self.active_indicators.items():
                if indicator_series is None:
                    continue

                try:
                    series = indicator_series
                    if not isinstance(series, pd.Series):
                        series = pd.Series(series)
                    if series.empty:
                        continue

                    if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None:
                        series = series.copy()
                        series.index = series.index.tz_localize(None)

                    if table_index is not None and len(table_index) == table_len:
                        aligned = series.reindex(table_index)
                        df[indicator_name] = aligned.values
                    else:
                        indicator_values = series.values
                        if len(indicator_values) >= table_len:
                            df[indicator_name] = indicator_values[:table_len]
                        else:
                            padded_values = np.full(table_len, np.nan)
                            padded_values[:len(indicator_values)] = indicator_values
                            df[indicator_name] = padded_values
                except Exception as e:
                    k2_logger.warning(f"Could not align indicator {indicator_name}: {e}", "MIDDLE_PANE")
        
        # Delegate display to the DataTabsWidget (Tab 1 – Current Data)
        self.data_tabs.load_current_data(df, self.active_indicators)
    
    def add_indicator(self, indicator_name: str, indicator_data: pd.Series, color: str = '#ffff00'):
        """
        Add indicator overlay to main chart and table.
        
        EXPECTATION: MUST add indicator to chart AND update table with indicator column.
        MUST store indicator data for table merging.
        """
        if self.chart_widget:
            self.chart_widget.add_indicator(indicator_name, indicator_data, color=color)
        
        # Store indicator data for table merging
        self.active_indicators[indicator_name] = indicator_data
        
        # Update table to include new indicator column
        if self.current_data is not None:
            self.load_data_into_table(self.current_data)
        
        k2_logger.info(f"Added indicator: {indicator_name} (chart overlay and table)", "MIDDLE_PANE")
    
    def add_indicator_pane(self, indicator_name: str, indicator_data: pd.Series, color: str = '#ffffff'):
        """
        Add indicator in a separate pane below the main chart.
        
        Used for oscillators (RSI, Stochastic, MACD) that have different Y-axis scales (0-100).
        """
        if self.chart_widget:
            self.chart_widget.add_indicator_pane(indicator_name, indicator_data, color=color)
        
        # Store indicator data for table merging
        self.active_indicators[indicator_name] = indicator_data
        
        # Update table to include new indicator column
        if self.current_data is not None:
            self.load_data_into_table(self.current_data)
        
        k2_logger.info(f"Added indicator: {indicator_name} (separate pane and table)", "MIDDLE_PANE")
    
    def remove_indicator(self, indicator_name: str):
        """
        Remove indicator from both chart and table.
        
        EXPECTATION: MUST remove indicator from chart AND remove column from table.
        Handles both overlays and separate panes.
        """
        if self.chart_widget:
            # Try to remove from overlay first
            self.chart_widget.remove_indicator(indicator_name)
            # Also try to remove from panes (for oscillators)
            if hasattr(self.chart_widget, 'remove_indicator_pane'):
                self.chart_widget.remove_indicator_pane(indicator_name)
        
        # Remove from active indicators
        if indicator_name in self.active_indicators:
            del self.active_indicators[indicator_name]
        
        # Update table to remove indicator column
        if self.current_data is not None:
            self.load_data_into_table(self.current_data)
        
        k2_logger.info(f"Removed indicator: {indicator_name} (chart and table)", "MIDDLE_PANE")
    
    def jump_to_end(self):
        """Jump to the latest data in the chart"""
        if self.chart_widget:
            self.chart_widget.jump_to_end()
    
    # Chart signal handlers
    def on_viewport_changed(self, start_idx: int, end_idx: int, total: int):
        """Update status label when viewport changes"""
        if self.status_label and total > 0:
            shown = max(0, end_idx - start_idx)
            percent = (shown / total) * 100 if total else 0
            self.status_label.setText(
                f"Showing {start_idx:,}-{end_idx:,} of {total:,} ({percent:.1f}%)"
            )
    
    def on_data_loading(self):
        """Show loading bar when data is loading"""
        if self.loading_bar:
            self.loading_bar.show()
            self.loading_bar.setRange(0, 0)
    
    def on_data_loaded(self):
        """Hide loading bar when data is loaded"""
        if self.loading_bar:
            self.loading_bar.hide()

    
    
    def apply_quick_indicator(self, indicator_type: str):
        """Apply a quick indicator with default parameters"""
        if not self.current_data:
            return
        
        params = {
            'SMA': {'period': 20},
            'EMA': {'period': 20},
            'RSI': {'period': 14},
            'BB': {'period': 20, 'std': 2}
        }
        
        if indicator_type in params:
            self.indicator_applied.emit(indicator_type, params[indicator_type])
    
    def clear_data(self):
        """Clear all data and reset to empty state"""
        # Get layout before clearing widgets
        layout = self.layout()
        
        # Clean up chart
        if self.chart_widget:
            self.chart_widget.cleanup()
            self.chart_widget.setParent(None)
            self.chart_widget.deleteLater()
            self.chart_widget = None
        
        # Clean up tabbed data widget
        if self.data_tabs:
            self.data_tabs.cleanup()
            self.data_tabs.setParent(None)
            self.data_tabs.deleteLater()
            self.data_tabs = None
        
        # Clean up controls and splitter
        if self.controls_bar:
            layout.removeWidget(self.controls_bar)  # Remove from layout
            self.controls_bar.setParent(None)
            self.controls_bar.deleteLater()
            self.controls_bar = None
        
        if self.loading_bar:
            layout.removeWidget(self.loading_bar)  # Remove from layout
            self.loading_bar.setParent(None)
            self.loading_bar.deleteLater()
            self.loading_bar = None
        
        if self.splitter:
            layout.removeWidget(self.splitter)  # Remove from layout
            self.splitter.setParent(None)
            self.splitter.deleteLater()
            self.splitter = None
        
        # Reset data
        self.current_data = None
        self.current_metadata = None
        self.current_table_name = None
        self.total_records = 0
        self.active_indicators.clear()
        
        # Note: Do not re-add the empty placeholder to avoid blank top region
    
    def setup_styling(self):
        """Apply styling to the pane"""
        self.setStyleSheet("""
            #middlePane {
                background-color: #0a0a0a;
            }
            
            #emptyPlaceholder {
                color: #666;
                font-size: 18px;
                background-color: #0a0a0a;
                padding: 50px;
            }
            
            #chartControls {
                background-color: #0f0f0f;
                border-bottom: 1px solid #1a1a1a;
            }
            
            #chartControls QLabel {
                background-color: transparent;
                color: #999;
                padding: 0px 5px;
            }
            
            QPushButton {
                background-color: #1a1a1a;
                color: #fff;
                border: 1px solid #2a2a2a;
                padding: 5px 15px;
                border-radius: 3px;
            }
            
            QPushButton:hover {
                background-color: #2a2a2a;
            }
            
            QComboBox {
                background-color: #1a1a1a;
                color: #fff;
                border: 1px solid #2a2a2a;
                padding: 5px;
                border-radius: 3px;
                min-width: 100px;
            }
            
            QComboBox::drop-down {
                border: none;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #999;
                margin-right: 5px;
            }
            
            QTableWidget {
                background-color: #0a0a0a;
                gridline-color: #1a1a1a;
                color: #fff;
            }
            
            QTableWidget::item {
                padding: 5px;
            }
            
            QHeaderView::section {
                background-color: #1a1a1a;
                color: #999;
                padding: 8px;
                border: none;
                font-weight: 600;
            }
            
            #dataSplitter::handle {
                background-color: #1a1a1a;
            }
        """)
    
    # ── Tab data persistence (called by analysis page) ──────────────

    def persist_tab_data(self, table_name: str):
        """Save Tab 2 (forecast) and Tab 3 (working sheets) to the DB."""
        if not self.data_tabs:
            return
        from k2_quant.utilities.data.db_manager import db_manager
        try:
            fc = self.data_tabs.serialise_forecast()
            if fc:
                db_manager.save_tab_data(f"forecast:{table_name}", fc)
            else:
                db_manager.delete_tab_data(f"forecast:{table_name}")

            ws = self.data_tabs.serialise_sheets()
            if ws:
                db_manager.save_tab_data(f"workspace:{table_name}", ws)
            else:
                db_manager.delete_tab_data(f"workspace:{table_name}")

            k2_logger.info(f"Tab data persisted for {table_name}", "MIDDLE_PANE")
        except Exception as e:
            k2_logger.error(f"Failed to persist tab data: {e}", "MIDDLE_PANE")

    def restore_tab_data(self, table_name: str):
        """Restore Tab 2 and Tab 3 from the DB."""
        if not self.data_tabs:
            return
        from k2_quant.utilities.data.db_manager import db_manager
        try:
            fc = db_manager.load_tab_data(f"forecast:{table_name}")
            if fc:
                self.data_tabs.restore_forecast(fc)

            ws = db_manager.load_tab_data(f"workspace:{table_name}")
            self.data_tabs.restore_sheets(ws)

            k2_logger.info(f"Tab data restored for {table_name}", "MIDDLE_PANE")
        except Exception as e:
            k2_logger.error(f"Failed to restore tab data: {e}", "MIDDLE_PANE")

    def cleanup(self):
        """Cleanup resources"""
        if self.chart_widget:
            self.chart_widget.cleanup()
        if self.data_tabs:
            self.data_tabs.cleanup()
        self.current_data = None
        self.current_metadata = None
        self.current_table_name = None
        self.total_records = 0
        self.active_indicators.clear()
        k2_logger.info("Middle pane cleaned up", "MIDDLE_PANE")