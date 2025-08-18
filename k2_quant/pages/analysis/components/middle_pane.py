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
                             QTableWidget, QTableWidgetItem)
from PyQt6.QtCore import Qt, pyqtSignal

from k2_quant.pages.analysis.widgets.chart_widget import ChartWidget
from k2_quant.utilities.logger import k2_logger


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
        self.active_columns = []
        self.active_indicators = {}
        
        # Create widgets
        self.chart_widget = None
        self.data_table = None
        self.view_selector = None
        
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
        
        # Create vertical splitter for chart and table
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setHandleWidth(2)
        self.splitter.setObjectName("dataSplitter")
        
        # Chart widget (60% height)
        self.chart_widget = ChartWidget()
        self.splitter.addWidget(self.chart_widget)
        
        # Data table widget (40% height)
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.horizontalHeader().setStretchLastSection(True)
        self.splitter.addWidget(self.data_table)
        
        # Set initial sizes (60/40 split)
        self.splitter.setSizes([360, 240])
        
        layout.addWidget(self.splitter)
    
    def create_controls_bar(self) -> QWidget:
        """Create top controls bar"""
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
        
        layout.addStretch()
        
        # Zoom controls
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedSize(30, 30)
        zoom_in_btn.clicked.connect(lambda: self.chart_widget.zoom(0.8) if self.chart_widget else None)
        layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setFixedSize(30, 30)
        zoom_out_btn.clicked.connect(lambda: self.chart_widget.zoom(1.25) if self.chart_widget else None)
        layout.addWidget(zoom_out_btn)
        
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(lambda: self.chart_widget.reset_zoom() if self.chart_widget else None)
        layout.addWidget(reset_btn)
        
        return controls
    
    def change_view(self, view_type: str):
        """Change the middle pane view"""
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
        
        # Convert to DataFrame if needed
        if isinstance(data, list) and len(data) > 0:
            # Standardized 8-column format
            columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
            df = pd.DataFrame(data, columns=columns[:len(data[0])])
            self.current_data = df
        elif isinstance(data, pd.DataFrame):
            self.current_data = data.copy()
        else:
            k2_logger.error("Invalid data format", "MIDDLE_PANE")
            return
        
        # Load into chart
        if self.chart_widget:
            self.chart_widget.load_data(self.current_data)
        
        # Load into table
        self.load_data_into_table(self.current_data)
        
        # Emit update signal
        self.data_updated.emit()
    
    def load_data_into_table(self, data):
        """Load data into the table widget"""
        if data is None:
            return
        
        # Handle both DataFrame and list formats
        if isinstance(data, pd.DataFrame):
            rows = data.values.tolist()
            columns = list(data.columns)
        elif isinstance(data, list):
            rows = data
            columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
        else:
            return
        
        # Set up table
        self.data_table.setRowCount(len(rows))
        self.data_table.setColumnCount(len(columns))
        self.data_table.setHorizontalHeaderLabels(columns)
        
        # Populate table
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                # Format based on column
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
    
    def add_indicator(self, indicator_name: str, indicator_data: pd.Series, color: str = '#ffff00'):
        """Add indicator to chart"""
        if self.chart_widget:
            self.chart_widget.add_indicator(indicator_name, indicator_data, color=color)
            self.active_indicators[indicator_name] = indicator_data
            k2_logger.info(f"Added indicator: {indicator_name}", "MIDDLE_PANE")
    
    def remove_indicator(self, indicator_name: str):
        """Remove indicator from chart"""
        if self.chart_widget:
            self.chart_widget.remove_indicator(indicator_name)
            if indicator_name in self.active_indicators:
                del self.active_indicators[indicator_name]
            k2_logger.info(f"Removed indicator: {indicator_name}", "MIDDLE_PANE")
    
    def apply_quick_indicator(self, indicator_type: str):
        """Apply a quick indicator with default parameters"""
        if not self.current_data:
            return
        
        # Define default parameters
        params = {
            'SMA': {'period': 20},
            'EMA': {'period': 20},
            'RSI': {'period': 14},
            'BB': {'period': 20, 'std': 2}
        }
        
        if indicator_type in params:
            self.indicator_applied.emit(indicator_type, params[indicator_type])
    
    def setup_styling(self):
        """Apply styling to the pane"""
        self.setStyleSheet("""
            #middlePane {
                background-color: #0a0a0a;
            }
            
            #chartControls {
                background-color: #0f0f0f;
                border-bottom: 1px solid #1a1a1a;
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
        self.active_indicators.clear()