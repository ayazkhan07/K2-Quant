"""
Middle Pane Component - Data Visualization

Contains chart and data table with controls.
Save as: k2_quant/pages/analysis/components/middle_pane.py
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QComboBox, QLabel, QSplitter,
                             QTableWidget, QTableWidgetItem, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal

from k2_quant.utilities.logger import k2_logger

# Updated import path for ChartWidget facade
from k2_quant.pages.analysis.widgets.chart import ChartWidget


class MiddlePaneWidget(QFrame):
    """Middle pane with chart and data table"""
    
    # Signals
    data_updated = pyqtSignal()
    column_toggled = pyqtSignal(str, bool)  # column_name, visible
    view_mode_changed = pyqtSignal(str)  # mode
    projection_requested = pyqtSignal()
    indicator_applied = pyqtSignal(str, dict)  # indicator_type, params
    data_exported = pyqtSignal(str, pd.DataFrame)  # format, data
    
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
        self.data_table = None
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

        # View Range control
        layout.addWidget(QLabel("RANGE:"))
        self.view_range_combo = QComboBox()
        self.view_range_combo.addItems(["15m", "30m", "1h", "4h", "1D", "5D", "1M", "3M", "YTD", "1Y", "All"])
        self.view_range_combo.setCurrentText("5D")
        self.view_range_combo.currentTextChanged.connect(
            lambda key: (self.chart_widget.set_view_range(key) if self.chart_widget else None)
        )
        layout.addWidget(self.view_range_combo)
        
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
            self.chart_widget.timeframe_changed.connect(lambda tf: self.update_view_range_combo())
            self.chart_widget.allowed_view_ranges_changed.connect(self._apply_allowed_view_ranges)
            
            self.splitter.addWidget(self.chart_widget)
            
            # Create data table widget
            self.data_table = QTableWidget()
            self.data_table.setAlternatingRowColors(True)
            self.data_table.horizontalHeader().setStretchLastSection(True)
            # Ensure table remains visible when panes grow
            self.data_table.setMinimumHeight(160)
            self.splitter.addWidget(self.data_table)
            
            # Set initial sizes (60/40 split)
            self.splitter.setSizes([360, 240])
            
            layout.addWidget(self.splitter)
            self.update_view_range_combo()
    
    def change_view(self, view_type: str):
        """Change the middle pane view"""
        if not self.chart_widget or not self.data_table:
            return
            
        if view_type == "Chart":
            self.chart_widget.setVisible(True)
            self.data_table.setVisible(False)
        elif view_type == "Data Table":
            self.chart_widget.setVisible(False)
            self.data_table.setVisible(True)
        elif view_type == "Both":
            self.chart_widget.setVisible(True)
            self.data_table.setVisible(True)
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
            # DIAGNOSTIC: table build header/row counts
            _base_headers = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
            try:
                row_len = len(data[0])
                k2_logger.debug(f"TABLE_BUILD: base_headers={len(_base_headers)} row_len={row_len}", "MIDDLE_PANE")
                if row_len != len(_base_headers):
                    k2_logger.error(f"TABLE_BUILD_MISMATCH: expected={len(_base_headers)} got={row_len}. First row={data[0]}", "MIDDLE_PANE")
            except Exception as ex:
                k2_logger.error(f"TABLE_BUILD diagnostics failed: {ex}", "MIDDLE_PANE")
            columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
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
        """Load data into the table widget"""
        if data is None or not self.data_table:
            return
        
        # Handle both DataFrame and list formats
        if isinstance(data, pd.DataFrame):
            try:
                k2_logger.debug(f"TABLE_RENDER: rows={len(data)} columns={list(data.columns)}", "MIDDLE_PANE")
            except Exception:
                pass
            rows = data.values.tolist()
            columns = list(data.columns)
        elif isinstance(data, list):
            try:
                k2_logger.debug(f"TABLE_RENDER: rows={len(data)} columns=fixed_base", "MIDDLE_PANE")
            except Exception:
                pass
            rows = data
            # Correct base headers to include Volume and guard mismatches
            columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
        else:
            return
        
        # Set up table
        self.data_table.setRowCount(len(rows))
        self.data_table.setColumnCount(len(columns))
        self.data_table.setHorizontalHeaderLabels(columns)
        
        # Populate table
        for i, row in enumerate(rows):
            # Guard against header/row length mismatch
            max_cols = min(len(columns), len(row))
            for j in range(max_cols):
                value = row[j]
                if columns[j] in ['Open', 'High', 'Low', 'Close', 'VWAP']:
                    try:
                        text = f"{float(value):.2f}"
                    except:
                        text = str(value)
                elif columns[j] == 'Volume':
                    try:
                        text = f"{int(value):,}"
                    except:
                        text = str(value)
                else:
                    text = str(value)
                
                self.data_table.setItem(i, j, QTableWidgetItem(text))
    

    # ---------------- Indicator Routing Bridge (overlays vs sub-panes) ----------------

    def add_indicator_overlay(self, indicator_name: str, series: pd.Series, cid: str | None = None):
        """Add overlay to main pane (dotted white via ChartWidget)."""
        if not self.chart_widget:
            return
        try:
            self.chart_widget.add_indicator_overlay(indicator_name, series, cid=cid)
        except TypeError:
            # Backward compatibility if chart method signature differs
            self.chart_widget.add_indicator_overlay(indicator_name, series)
        self.active_indicators[indicator_name] = {"pane": "main", "outputs": ["line"]}
        k2_logger.info(f"Added overlay indicator: {indicator_name}", "MIDDLE_PANE")
        self.data_updated.emit()

    def add_indicator_to_pane(self, pane_key: str, series_map: Dict[str, pd.Series], family: str, cid: str | None = None):
        """
        Create/update a shared sub-pane keyed by family (e.g., 'RSI', 'MACD').
        Enforces a maximum of 3 sub-panes.
        """
        if not self.chart_widget:
            return

        # Cap at 3 panes (only if creating a new pane)
        if pane_key not in self.chart_widget.indicator_panes and len(self.chart_widget.indicator_panes) >= 3:
            try:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Limit Reached", "Maximum 3 sub-panes. Remove an indicator first.")
            except Exception:
                pass
            return

        fam = (family or pane_key).upper()
        if fam in ("RSI", "STOCH", "ADX", "MFI"):
            y_mode = "bounded_0_100"
        elif fam in ("MACD", "CCI"):
            y_mode = "symmetric_0"
        elif fam in ("ATR", "VOLUME"):
            y_mode = "auto_positive"
        else:
            y_mode = "auto"

        for label, s in series_map.items():
            try:
                self.chart_widget.add_or_update_indicator_pane(
                    pane_key=pane_key,
                    series_label=label,
                    series=s,
                    y_mode=y_mode,
                    cid=cid
                )
            except TypeError:
                self.chart_widget.add_or_update_indicator_pane(
                    pane_key=pane_key,
                    series_label=label,
                    series=s,
                    y_mode=y_mode
                )

        self.active_indicators[pane_key] = {"pane": "separate", "outputs": list(series_map.keys())}
        # Reassert splitter sizes so table doesn't collapse
        try:
            if self.splitter and self.data_table:
                sizes = self.splitter.sizes()
                top = max(sizes[0] if sizes else 360, 360)
                bottom = max(self.data_table.minimumHeight(), 220)
                self.splitter.setStretchFactor(0, 3)
                self.splitter.setStretchFactor(1, 2)
                self.splitter.setSizes([top, bottom])
        except Exception:
            pass
        self.data_updated.emit()

    def remove_indicator(self, indicator_name: str, family: str = None, cid: str | None = None):
        """
        Remove overlay or a single series from a family pane. If a pane becomes empty
        after removal, it is auto-removed by the chart.
        """
        if not self.chart_widget:
            return
        # Remove overlay if present
        self.chart_widget.remove_indicator(indicator_name)
        # Remove the series from its shared pane
        if family:
            pane_key = family.upper()
            try:
                self.chart_widget.remove_series_from_pane(pane_key, indicator_name, cid=cid)
            except TypeError:
                self.chart_widget.remove_series_from_pane(pane_key, indicator_name)
        if indicator_name in self.active_indicators:
            del self.active_indicators[indicator_name]
        k2_logger.info(f"Removed indicator: {indicator_name}", "MIDDLE_PANE")
        self.data_updated.emit()
    
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
        # Re-apply allowed ranges after (re)load
        self.update_view_range_combo()

    def update_view_range_combo(self):
        """Refresh enabled/disabled state of View Range options based on current aggregation."""
        if not self.chart_widget or not hasattr(self, 'view_range_combo') or not self.view_range_combo:
            return
        try:
            allowed = [vr.value for vr in self.chart_widget._compute_allowed_view_ranges()]
            self._apply_allowed_view_ranges(allowed)
        except Exception:
            pass

    def _apply_allowed_view_ranges(self, allowed_keys: list):
        if not hasattr(self, 'view_range_combo') or not self.view_range_combo:
            return
        current = self.view_range_combo.currentText()
        first_allowed = None
        for i in range(self.view_range_combo.count()):
            key = self.view_range_combo.itemText(i)
            is_allowed = key in allowed_keys
            item = self.view_range_combo.model().item(i)
            if item is not None:
                item.setEnabled(is_allowed)
            if is_allowed and first_allowed is None:
                first_allowed = key
        if current not in allowed_keys and first_allowed:
            self.view_range_combo.setCurrentText(first_allowed)
    
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
        
        # Clean up table
        if self.data_table:
            self.data_table.clear()
            self.data_table.setParent(None)
            self.data_table.deleteLater()
            self.data_table = None
        
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
    
    def cleanup(self):
        """Cleanup resources"""
        if self.chart_widget:
            self.chart_widget.cleanup()
        self.current_data = None
        self.current_metadata = None
        self.current_table_name = None
        self.total_records = 0
        self.active_indicators.clear()
        k2_logger.info("Middle pane cleaned up", "MIDDLE_PANE")