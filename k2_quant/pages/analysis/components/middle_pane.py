"""
Middle Pane Component - Data Visualization

Contains interactive chart and data table with controls.
"""

from typing import Dict, List, Any, Optional
import pandas as pd

from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QComboBox, QLabel, QSplitter,
                             QButtonGroup)
from PyQt6.QtCore import Qt, pyqtSignal

from k2_quant.pages.analysis.widgets.chart_widget import ChartWidget
from k2_quant.pages.analysis.widgets.data_table_widget import DataTableWidget
from k2_quant.utilities.services.technical_analysis_service import ta_service
from k2_quant.utilities.logger import k2_logger


class MiddlePaneWidget(QFrame):
    """Middle pane with chart and data table"""
    
    # Signals
    data_updated = pyqtSignal()
    column_selected = pyqtSignal(str)  # column_name
    projection_requested = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setObjectName("middlePane")
        
        self.current_data = None
        self.current_metadata = None
        self.active_columns = []
        self.active_indicators = {}
        
        self.init_ui()
        self.setup_styling()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        
        # Top controls bar
        self.controls_bar = self.create_controls_bar()
        layout.addWidget(self.controls_bar)
        
        # Column selector bar
        self.column_bar = self.create_column_bar()
        layout.addWidget(self.column_bar)
        
        # Create vertical splitter for chart and table
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setHandleWidth(2)
        self.splitter.setObjectName("dataSplitter")
        
        # Chart widget (60% height)
        self.chart_widget = ChartWidget()
        self.splitter.addWidget(self.chart_widget)
        
        # Data table widget (40% height)
        self.table_widget = DataTableWidget()
        self.splitter.addWidget(self.table_widget)
        
        # Set initial sizes (60/40 split)
        self.splitter.setSizes([360, 240])
        
        layout.addWidget(self.splitter)
    
    def create_controls_bar(self) -> QWidget:
        """Create top controls bar"""
        controls = QWidget()
        controls.setFixedHeight(40)
        controls.setObjectName("controlsBar")
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        controls.setLayout(layout)
        
        # View mode buttons
        self.chart_only_btn = QPushButton("Chart")
        self.chart_only_btn.setCheckable(True)
        self.chart_only_btn.setChecked(False)
        self.chart_only_btn.clicked.connect(lambda: self.set_view_mode('chart'))
        layout.addWidget(self.chart_only_btn)
        
        self.table_only_btn = QPushButton("Table")
        self.table_only_btn.setCheckable(True)
        self.table_only_btn.clicked.connect(lambda: self.set_view_mode('table'))
        layout.addWidget(self.table_only_btn)
        
        self.both_btn = QPushButton("Both")
        self.both_btn.setCheckable(True)
        self.both_btn.setChecked(True)
        self.both_btn.clicked.connect(lambda: self.set_view_mode('both'))
        layout.addWidget(self.both_btn)
        
        # Button group for exclusive selection
        self.view_group = QButtonGroup()
        self.view_group.addButton(self.chart_only_btn)
        self.view_group.addButton(self.table_only_btn)
        self.view_group.addButton(self.both_btn)
        
        layout.addSpacing(20)
        
        # Projection button
        self.projection_btn = QPushButton("Generate Projections")
        self.projection_btn.clicked.connect(self.projection_requested.emit)
        self.projection_btn.setEnabled(False)
        self.projection_btn.setObjectName("projectionBtn")
        layout.addWidget(self.projection_btn)
        
        layout.addStretch()
        
        # Data info label
        self.data_info_label = QLabel("No data loaded")
        self.data_info_label.setObjectName("dataInfoLabel")
        layout.addWidget(self.data_info_label)
        
        return controls
    
    def create_column_bar(self) -> QWidget:
        """Create column selector bar"""
        column_bar = QWidget()
        column_bar.setFixedHeight(35)
        column_bar.setObjectName("columnBar")
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        column_bar.setLayout(layout)
        
        layout.addWidget(QLabel("COLUMNS:"))
        
        # Column buttons - will be populated when data is loaded
        self.column_buttons = {}
        columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
        
        for col in columns:
            btn = QPushButton(col)
            btn.setCheckable(True)
            btn.setObjectName("columnBtn")
            btn.clicked.connect(lambda checked, c=col: self.toggle_column(c, checked))
            self.column_buttons[col] = btn
            layout.addWidget(btn)
        
        layout.addSpacing(20)
        
        # Clear all button
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self.clear_all_lines)
        self.clear_all_btn.setObjectName("clearBtn")
        layout.addWidget(self.clear_all_btn)
        
        layout.addStretch()
        
        return column_bar
    
    def set_view_mode(self, mode: str):
        """Set the view mode (chart, table, or both)"""
        if mode == 'chart':
            self.chart_widget.setVisible(True)
            self.table_widget.setVisible(False)
        elif mode == 'table':
            self.chart_widget.setVisible(False)
            self.table_widget.setVisible(True)
        elif mode == 'both':
            self.chart_widget.setVisible(True)
            self.table_widget.setVisible(True)
            self.splitter.setSizes([360, 240])
        
        k2_logger.info(f"View mode set to: {mode}", "MIDDLE_PANE")
    
    def load_data(self, data: Any, metadata: Dict[str, Any]):
        """Load data into chart and table"""
        k2_logger.info(f"Loading data into visualization", "MIDDLE_PANE")
        
        self.current_data = data
        self.current_metadata = metadata
        
        # Convert to DataFrame if needed
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list) and len(data) > 0:
                # Assuming list of tuples from database
                columns = ['date_time', 'open', 'high', 'low', 'close', 'volume', 'vwap']
                df = pd.DataFrame(data, columns=columns[:len(data[0])])
                self.current_data = df
            else:
                k2_logger.error("Invalid data format", "MIDDLE_PANE")
                return
        
        # Load into chart
        self.chart_widget.load_data(self.current_data)
        
        # Load into table
        self.table_widget.load_data(self.current_data)
        
        # Update info
        self.update_data_info()
        
        # Enable projection button
        self.projection_btn.setEnabled(True)
        
        # Auto-select Close column
        if 'Close' in self.column_buttons:
            self.column_buttons['Close'].setChecked(True)
            self.toggle_column('Close', True)
        
        # Emit update signal
        self.data_updated.emit()
    
    def update_data_info(self):
        """Update data information label"""
        if self.current_metadata:
            symbol = self.current_metadata.get('symbol', 'Unknown')
            records = len(self.current_data) if self.current_data is not None else 0
            self.data_info_label.setText(f"{symbol}: {records:,} records")
        else:
            self.data_info_label.setText("Data loaded")
    
    def toggle_column(self, column_name: str, checked: bool):
        """Toggle a data column on/off in the chart"""
        if checked:
            if column_name not in self.active_columns:
                # Add to chart
                col_lower = column_name.lower()
                if self.current_data is not None and col_lower in self.current_data.columns:
                    self.chart_widget.add_line(column_name, self.current_data[col_lower])
                    self.active_columns.append(column_name)
                    k2_logger.info(f"Added {column_name} to chart", "MIDDLE_PANE")
        else:
            if column_name in self.active_columns:
                # Remove from chart
                self.chart_widget.remove_line(column_name)
                self.active_columns.remove(column_name)
                k2_logger.info(f"Removed {column_name} from chart", "MIDDLE_PANE")
    
    def clear_all_lines(self):
        """Clear all lines from chart"""
        self.chart_widget.clear_all()
        self.active_columns.clear()
        
        # Uncheck all column buttons
        for btn in self.column_buttons.values():
            btn.setChecked(False)
        
        k2_logger.info("Cleared all chart lines", "MIDDLE_PANE")
    
    def add_indicator(self, indicator_name: str, indicator_data: pd.Series):
        """Add technical indicator to chart"""
        # Determine if it needs a separate pane
        if ta_service:
            info = ta_service.get_indicator_info(indicator_name)
            if info and info.pane == 'separate':
                # Add to separate pane
                self.chart_widget.add_indicator_pane(indicator_name, indicator_data)
            else:
                # Add to main chart
                self.chart_widget.add_line(indicator_name, indicator_data, color='#aa4')
        
        self.active_indicators[indicator_name] = indicator_data
        k2_logger.info(f"Added indicator: {indicator_name}", "MIDDLE_PANE")
    
    def remove_indicator(self, indicator_name: str):
        """Remove technical indicator from chart"""
        if indicator_name in self.active_indicators:
            # Check if it's in a separate pane
            if ta_service:
                info = ta_service.get_indicator_info(indicator_name)
                if info and info.pane == 'separate':
                    self.chart_widget.remove_indicator_pane(indicator_name)
                else:
                    self.chart_widget.remove_line(indicator_name)
            
            del self.active_indicators[indicator_name]
            k2_logger.info(f"Removed indicator: {indicator_name}", "MIDDLE_PANE")
    
    def add_strategy_overlay(self, strategy_name: str, data: pd.DataFrame):
        """Add strategy overlay to chart"""
        # Look for projection columns in the data
        for col in data.columns:
            if 'projection' in col.lower() or 'signal' in col.lower():
                self.chart_widget.add_projection(strategy_name, data[col])
                k2_logger.info(f"Added strategy overlay: {strategy_name}", "MIDDLE_PANE")
    
    def remove_strategy_overlay(self, strategy_name: str):
        """Remove strategy overlay from chart"""
        self.chart_widget.remove_projection(strategy_name)
        k2_logger.info(f"Removed strategy overlay: {strategy_name}", "MIDDLE_PANE")
    
    def update_with_projections(self, data: pd.DataFrame):
        """Update chart with projection data"""
        # Update current data
        self.current_data = data
        
        # Reload chart with new data
        self.chart_widget.load_data(data)
        
        # Look for projection columns and highlight them
        for col in data.columns:
            if 'projection' in col.lower() or 'is_projection' in col:
                # Find projection rows
                projection_mask = data['is_projection'] if 'is_projection' in data.columns else pd.Series([False] * len(data))
                if projection_mask.any():
                    projection_data = data.loc[projection_mask, 'close'] if 'close' in data.columns else data.loc[projection_mask].iloc[:, 0]
                    self.chart_widget.add_projection('Generated', projection_data)
        
        # Update table
        self.table_widget.load_data(data)
        
        # Update info
        self.update_data_info()
        
        # Emit update
        self.data_updated.emit()
    
    def setup_styling(self):
        """Apply styling to the pane"""
        self.setStyleSheet("""
            #middlePane {
                background-color: #0a0a0a;
            }
            
            #controlsBar, #columnBar {
                background-color: #0f0f0f;
                border-bottom: 1px solid #1a1a1a;
            }
            
            #dataInfoLabel {
                color: #666;
                font-size: 11px;
            }
            
            QPushButton {
                background-color: #1a1a1a;
                color: #999;
                border: 1px solid #2a2a2a;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 11px;
            }
            
            QPushButton:hover {
                background-color: #2a2a2a;
                color: #fff;
            }
            
            QPushButton:checked {
                background-color: #2a2a2a;
                color: #4a4;
                border-color: #3a3a3a;
            }
            
            #columnBtn:checked {
                background-color: #2a2a2a;
                color: #4a4;
            }
            
            #projectionBtn {
                color: #4aa;
                border-color: #2a4a4a;
            }
            
            #projectionBtn:hover:enabled {
                background-color: #1a2a2a;
                color: #6cc;
            }
            
            #clearBtn {
                color: #a44;
            }
            
            #dataSplitter::handle {
                background-color: #1a1a1a;
            }
            
            QLabel {
                color: #999;
                font-size: 11px;
                font-weight: 600;
            }
        """)
    
    def cleanup(self):
        """Cleanup resources"""
        self.chart_widget.cleanup()
        self.table_widget.cleanup()