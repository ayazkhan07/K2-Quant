"""
Advanced Multi-Pane Chart Widget for K2 Quant Analysis

Supports line charts with column selection, multi-pane indicators,
and drawing tools for trend analysis.
Save as: k2_quant/pages/analysis/widgets/chart_widget.py
"""

import pyqtgraph as pg
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QPen

from k2_quant.utilities.logger import k2_logger


class ChartWidget(QWidget):
    """Multi-pane chart widget with drawing tools"""
    
    # Signals
    drawing_added = pyqtSignal(dict)
    time_range_changed = pyqtSignal(str, str)
    time_range_selected = pyqtSignal(str)
    fetch_older_requested = pyqtSignal(object)
    timeframe_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.data = None
        self.date_column = None
        self.active_lines = {}
        self.indicator_panes = {}
        self.indicator_overlays = {}
        self.drawing_mode = None
        self.drawings = []
        self.x_values = None
        
        self.init_ui()
        self.setup_chart_style()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        
        # Drawing tools toolbar
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)
        
        # Main chart
        self.main_plot = pg.PlotWidget()
        self.main_plot.showGrid(x=True, y=True, alpha=0.3)
        self.main_plot.setLabel('left', 'Price', units='$')
        self.main_plot.setLabel('bottom', 'Date/Time')
        
        # Add crosshair
        self.add_crosshair()
        
        layout.addWidget(self.main_plot, stretch=3)
        
        # Container for indicator panes
        self.indicator_container = QVBoxLayout()
        self.indicator_container.setSpacing(2)
        layout.addLayout(self.indicator_container, stretch=1)
        
        # Connect mouse events for drawing
        self.main_plot.scene().sigMouseClicked.connect(self.on_mouse_clicked)
        self.main_plot.scene().sigMouseMoved.connect(self.on_mouse_moved)
    
    def create_toolbar(self):
        """Create drawing tools toolbar and range buttons"""
        toolbar = QWidget()
        toolbar.setFixedHeight(35)
        toolbar.setStyleSheet("""
            QWidget {
                background-color: #0f0f0f;
                border-bottom: 1px solid #1a1a1a;
            }
            QPushButton {
                background-color: #1a1a1a;
                color: #999;
                border: 1px solid #2a2a2a;
                padding: 5px 10px;
                margin: 2px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #2a2a2a;
                color: #fff;
            }
            QPushButton:checked {
                background-color: #3a3a3a;
                color: #4a4;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 0, 5, 0)
        toolbar.setLayout(layout)
        
        # Time range buttons
        for label in ['5D', '1M', '3M', '6M', '1Y', '5Y', 'All']:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, l=label: self.time_range_selected.emit(l))
            layout.addWidget(btn)

        layout.addSpacing(10)

        # Drawing tools
        self.trend_btn = QPushButton("📈 Trend Line")
        self.trend_btn.setCheckable(True)
        self.trend_btn.clicked.connect(lambda: self.set_drawing_mode('trend'))
        layout.addWidget(self.trend_btn)
        
        self.hline_btn = QPushButton("─ H-Line")
        self.hline_btn.setCheckable(True)
        self.hline_btn.clicked.connect(lambda: self.set_drawing_mode('horizontal'))
        layout.addWidget(self.hline_btn)
        
        self.vline_btn = QPushButton("│ V-Line")
        self.vline_btn.setCheckable(True)
        self.vline_btn.clicked.connect(lambda: self.set_drawing_mode('vertical'))
        layout.addWidget(self.vline_btn)
        
        self.text_btn = QPushButton("T Text")
        self.text_btn.setCheckable(True)
        self.text_btn.clicked.connect(lambda: self.set_drawing_mode('text'))
        layout.addWidget(self.text_btn)
        
        layout.addSpacing(20)
        
        self.clear_drawings_btn = QPushButton("Clear Drawings")
        self.clear_drawings_btn.clicked.connect(self.clear_all_drawings)
        layout.addWidget(self.clear_drawings_btn)
        
        layout.addStretch()
        
        # Zoom controls
        self.zoom_fit_btn = QPushButton("Fit All")
        self.zoom_fit_btn.clicked.connect(self.auto_range)
        layout.addWidget(self.zoom_fit_btn)
        
        return toolbar
    
    def setup_chart_style(self):
        """Setup chart appearance"""
        self.main_plot.setBackground('#0a0a0a')
        self.main_plot.getAxis('left').setPen(pg.mkPen(color='#666'))
        self.main_plot.getAxis('left').setTextPen(pg.mkPen(color='#999'))
        self.main_plot.getAxis('bottom').setPen(pg.mkPen(color='#666'))
        self.main_plot.getAxis('bottom').setTextPen(pg.mkPen(color='#999'))
    
    def add_crosshair(self):
        """Add crosshair to main chart"""
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#666', width=1))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#666', width=1))
        self.main_plot.addItem(self.vLine, ignoreBounds=True)
        self.main_plot.addItem(self.hLine, ignoreBounds=True)
        
        # Value label
        self.value_label = pg.TextItem(color='#fff', anchor=(0, 1))
        self.main_plot.addItem(self.value_label)
        
        # Connect mouse movement
        self.proxy = pg.SignalProxy(self.main_plot.scene().sigMouseMoved, 
                                   rateLimit=60, slot=self.update_crosshair)
    
    def update_crosshair(self, evt):
        """Update crosshair position"""
        pos = evt[0]
        if self.main_plot.sceneBoundingRect().contains(pos):
            mousePoint = self.main_plot.getViewBox().mapSceneToView(pos)
            
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
            
            # Update value label
            if self.data is not None and self.date_column is not None:
                x_val = int(mousePoint.x())
                if 0 <= x_val < len(self.data):
                    date_val = self.data.iloc[x_val][self.date_column]
                    date_str = str(date_val)[:16] if pd.notna(date_val) else ""
                    self.value_label.setText(f"{date_str}\nPrice: ${mousePoint.y():.2f}")
                    self.value_label.setPos(mousePoint.x(), mousePoint.y())
    
    def load_data(self, data):
        """Load data into chart"""
        if isinstance(data, pd.DataFrame):
            self.data = data.reset_index(drop=True)
        else:
            # Convert list to DataFrame
            columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
            self.data = pd.DataFrame(data, columns=columns[:len(data[0]) if data else 0])
        
        # Identify date column
        self.date_column = None
        if 'date_time_market' in self.data.columns:
            self.date_column = 'date_time_market'
        elif 'Date' in self.data.columns and 'Time' in self.data.columns:
            try:
                self.data['date_time'] = pd.to_datetime(
                    self.data['Date'].astype(str) + ' ' + self.data['Time'].astype(str)
                )
                self.date_column = 'date_time'
            except Exception:
                self.date_column = 'Date'
        else:
            for col in ['date', 'datetime', 'date_time', 'timestamp', 'Date']:
                if col in self.data.columns:
                    self.date_column = col
                    break
        
        # Clear existing plots
        self.clear_all()
        
        # Create x-axis values
        self.x_values = np.arange(len(self.data))
        
        # Setup x-axis formatting
        self.setup_x_axis()
        
        # Auto-plot OHLC if available
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in self.data.columns:
                self.add_line(col, self.data[col])
        
        k2_logger.info(f"Data loaded: {len(self.data)} records", "CHART")
    
    def setup_x_axis(self):
        """Setup x-axis with intelligent date/time formatting"""
        if self.data is None or self.date_column is None:
            return
        
        # Create custom axis labels
        axis = self.main_plot.getAxis('bottom')
        
        # Determine if we have time component
        first_date = str(self.data.iloc[0][self.date_column]) if len(self.data) > 0 else ""
        has_time = len(first_date) > 10 and ':' in first_date
        
        # Create tick labels
        num_ticks = min(10, len(self.data))
        tick_spacing = max(1, len(self.data) // num_ticks) if num_ticks > 0 else 1
        
        ticks = []
        for i in range(0, len(self.data), tick_spacing):
            if i < len(self.data):
                date_val = self.data.iloc[i][self.date_column]
                if pd.notna(date_val):
                    date_str = str(date_val)
                    if has_time:
                        if i == 0 or str(self.data.iloc[i-1][self.date_column])[:10] != date_str[:10]:
                            label = date_str[:10]
                        else:
                            label = date_str[11:16]
                    else:
                        label = date_str[:10]
                    
                    ticks.append((i, label))
        
        if ticks:
            axis.setTicks([ticks])
    
    def add_line(self, column_name, data=None, color=None):
        """Add a line to the main chart"""
        if column_name in self.active_lines:
            return
        
        if data is None and self.data is not None:
            for col in self.data.columns:
                if col.lower() == column_name.lower():
                    data = self.data[col]
                    break
        
        if data is None:
            return
        
        # Generate color if not provided
        if color is None:
            colors = ['#4a4', '#44a', '#a44', '#aa4', '#a4a', '#4aa']
            color = colors[len(self.active_lines) % len(colors)]
        
        # Plot the line
        pen = pg.mkPen(color=color, width=2)
        if isinstance(data, pd.Series):
            y_values = data.values
        else:
            y_values = data
        
        plot_item = self.main_plot.plot(
            self.x_values[:len(y_values)], 
            y_values,
            pen=pen, 
            name=column_name
        )
        
        self.active_lines[column_name] = plot_item
        k2_logger.info(f"Added line: {column_name}", "CHART")
    
    def add_indicator(self, indicator_name, indicator_data, color='#ffff00'):
        """Add indicator as overlay on main chart"""
        if indicator_name in self.indicator_overlays:
            self.remove_indicator(indicator_name)
        
        if self.x_values is None:
            self.x_values = np.arange(len(indicator_data))
        
        # Plot the indicator
        pen = pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine)
        
        if isinstance(indicator_data, pd.Series):
            y_values = indicator_data.values
        else:
            y_values = indicator_data
            
        plot_item = self.main_plot.plot(
            self.x_values[:len(y_values)], 
            y_values,
            pen=pen, 
            name=indicator_name
        )
        
        self.indicator_overlays[indicator_name] = plot_item
        k2_logger.info(f"Added indicator overlay: {indicator_name}", "CHART")
    
    def remove_indicator(self, indicator_name):
        """Remove indicator from chart"""
        if indicator_name in self.indicator_overlays:
            self.main_plot.removeItem(self.indicator_overlays[indicator_name])
            del self.indicator_overlays[indicator_name]
            k2_logger.info(f"Removed indicator: {indicator_name}", "CHART")
    
    def remove_line(self, column_name):
        """Remove a line from the chart"""
        if column_name in self.active_lines:
            self.main_plot.removeItem(self.active_lines[column_name])
            del self.active_lines[column_name]
            k2_logger.info(f"Removed line: {column_name}", "CHART")
    
    def add_indicator_pane(self, indicator_name, data, chart_type='line'):
        """Add a new pane for an indicator"""
        if indicator_name in self.indicator_panes:
            self.remove_indicator_pane(indicator_name)
        
        # Create new plot widget for indicator
        indicator_plot = pg.PlotWidget()
        indicator_plot.setMaximumHeight(150)
        indicator_plot.showGrid(x=True, y=True, alpha=0.3)
        indicator_plot.setLabel('left', indicator_name)
        indicator_plot.setBackground('#0a0a0a')
        
        # Style the axes
        indicator_plot.getAxis('left').setPen(pg.mkPen(color='#666'))
        indicator_plot.getAxis('left').setTextPen(pg.mkPen(color='#999'))
        indicator_plot.getAxis('bottom').setPen(pg.mkPen(color='#666'))
        indicator_plot.getAxis('bottom').setTextPen(pg.mkPen(color='#999'))
        
        # Link x-axis to main plot
        indicator_plot.setXLink(self.main_plot)
        
        # Plot the indicator
        if chart_type == 'line':
            pen = pg.mkPen(color='#4a4', width=2)
            indicator_plot.plot(self.x_values[:len(data)], data.values, pen=pen)
        elif chart_type == 'bar':
            bargraph = pg.BarGraphItem(
                x=self.x_values[:len(data)], 
                height=data.values, 
                width=0.8, 
                brush='#4a4'
            )
            indicator_plot.addItem(bargraph)
        
        # Add to layout
        self.indicator_container.addWidget(indicator_plot)
        self.indicator_panes[indicator_name] = indicator_plot
        
        k2_logger.info(f"Added indicator pane: {indicator_name}", "CHART")
    
    def remove_indicator_pane(self, indicator_name):
        """Remove an indicator pane"""
        if indicator_name in self.indicator_panes:
            widget = self.indicator_panes[indicator_name]
            self.indicator_container.removeWidget(widget)
            widget.deleteLater()
            del self.indicator_panes[indicator_name]
            k2_logger.info(f"Removed indicator pane: {indicator_name}", "CHART")
    
    def set_drawing_mode(self, mode):
        """Set drawing mode"""
        self.drawing_mode = mode if self.drawing_mode != mode else None
        
        # Update button states
        buttons = [self.trend_btn, self.hline_btn, self.vline_btn, self.text_btn]
        for btn in buttons:
            btn.setChecked(False)
        
        if self.drawing_mode == 'trend':
            self.trend_btn.setChecked(True)
        elif self.drawing_mode == 'horizontal':
            self.hline_btn.setChecked(True)
        elif self.drawing_mode == 'vertical':
            self.vline_btn.setChecked(True)
        elif self.drawing_mode == 'text':
            self.text_btn.setChecked(True)
        
        k2_logger.info(f"Drawing mode: {self.drawing_mode}", "CHART")
    
    def on_mouse_clicked(self, evt):
        """Handle mouse click for drawing"""
        if self.drawing_mode is None:
            return
        
        pos = evt.scenePos()
        if self.main_plot.sceneBoundingRect().contains(pos):
            vb = self.main_plot.getViewBox()
            mousePoint = vb.mapSceneToView(pos)
            
            if self.drawing_mode == 'horizontal':
                line = pg.InfiniteLine(
                    pos=mousePoint.y(), 
                    angle=0,
                    pen=pg.mkPen('#aa4', width=2)
                )
                self.main_plot.addItem(line)
                self.drawings.append(('hline', line))
                
            elif self.drawing_mode == 'vertical':
                line = pg.InfiniteLine(
                    pos=mousePoint.x(), 
                    angle=90,
                    pen=pg.mkPen('#4aa', width=2)
                )
                self.main_plot.addItem(line)
                self.drawings.append(('vline', line))
    
    def on_mouse_moved(self, pos):
        """Handle mouse movement for drawing preview"""
        pass
    
    def clear_all_drawings(self):
        """Clear all drawings"""
        for drawing_type, item in self.drawings:
            self.main_plot.removeItem(item)
        self.drawings.clear()
        k2_logger.info("Cleared all drawings", "CHART")
    
    def clear_all(self):
        """Clear all lines and indicators"""
        # Clear main chart lines
        for line in self.active_lines.values():
            self.main_plot.removeItem(line)
        self.active_lines.clear()
        
        # Clear indicator overlays
        for overlay in self.indicator_overlays.values():
            self.main_plot.removeItem(overlay)
        self.indicator_overlays.clear()
        
        # Clear indicator panes
        for indicator_name in list(self.indicator_panes.keys()):
            self.remove_indicator_pane(indicator_name)
        
        # Clear drawings
        self.clear_all_drawings()
    
    def auto_range(self):
        """Auto-range all plots"""
        self.main_plot.autoRange()
        for plot in self.indicator_panes.values():
            plot.autoRange()
    
    def zoom(self, factor):
        """Zoom in/out"""
        self.main_plot.getViewBox().scaleBy((factor, 1.0))
    
    def reset_zoom(self):
        """Reset zoom to fit all data"""
        self.auto_range()
    
    def add_projection(self, name, projection_data):
        """Add projection data as dashed line"""
        pen = pg.mkPen(color='#f44', width=2, style=Qt.PenStyle.DashLine)
        x_values = np.arange(len(self.data), len(self.data) + len(projection_data))
        plot_item = self.main_plot.plot(
            x_values, 
            projection_data.values,
            pen=pen, 
            name=f"{name} (Projection)"
        )
        self.active_lines[f"{name}_projection"] = plot_item
    
    def remove_projection(self, name):
        """Remove projection from chart"""
        key = f"{name}_projection"
        if key in self.active_lines:
            self.remove_line(key)
    
    def cleanup(self):
        """Cleanup resources"""
        self.clear_all()