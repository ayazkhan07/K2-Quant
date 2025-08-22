"""
Ultra-Optimized Advanced Multi-Pane Chart Widget for K2 Quant Analysis
Complete implementation with discrete X-axis behavior and context-aware grid system
"""

import pyqtgraph as pg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from functools import lru_cache, partial
from dataclasses import dataclass, field
from collections import deque
import gc
import math

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFrame, QButtonGroup)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPointF, QRectF, QEvent
from PyQt6.QtGui import QColor, QPen, QBrush, QFont, QCursor

# If k2_logger is not available, create a simple replacement
try:
    from k2_quant.utilities.logger import k2_logger
except ImportError:
    class DummyLogger:
        def info(self, msg, category=""): print(f"INFO [{category}]: {msg}")
        def error(self, msg, category=""): print(f"ERROR [{category}]: {msg}")
        def debug(self, msg, category=""): pass
    k2_logger = DummyLogger()


# Constants
OHLC_COLORS = {
    'Open': '#00ff00',
    'High': '#0080ff', 
    'Low': '#ff0000',
    'Close': '#ffff00'
}

TIMEFRAME_RULES = {
    '15m': '15min',
    '30m': '30min',
    '1h': '1h',
    '4h': '4h',
    '1W': 'W',
    '1M': 'M',
    '3M': '3M',
    '1Y': 'A'
}

POINTS_PER_DAY = {
    '15m': 26,
    '30m': 13,
    '1h': 7,
    '4h': 2,
    '1D': 1,
    '1W': 0.2,
    '1M': 0.05,
    '3M': 0.017,
    '1Y': 0.004
}

NUMERIC_COLUMNS = frozenset(['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP'])
INTRADAY_TIMEFRAMES = frozenset(['15m', '30m', '1h', '4h'])
DAILY_PLUS_TIMEFRAMES = frozenset(['1D', '1W', '1M', '3M', '1Y'])


@dataclass
class DrawingConfig:
    """Configuration for drawing tools"""
    trend: tuple = ('/', 'Trend Line', '#00ff00')
    ray: tuple = ('→', 'Ray', '#ff00ff')
    extended: tuple = ('↔', 'Extended Line', '#00ffff')
    horizontal: tuple = ('─', 'Horizontal Line', '#ffff00')
    vertical: tuple = ('│', 'Vertical Line', '#00ffff')


class SimplePlotCurveItem(pg.PlotCurveItem):
    """Custom PlotCurveItem that ensures no fill rendering"""
    def __init__(self, *args, **kwargs):
        # Remove any fill-related parameters
        kwargs.pop('fillLevel', None)
        kwargs.pop('fillBrush', None)
        kwargs.pop('brush', None)
        super().__init__(*args, **kwargs)
        
        # Force disable fill
        self.opts['fillLevel'] = None
        self.opts['fillBrush'] = None
        self.opts['brush'] = None


class DiscreteViewBox(pg.ViewBox):
    """Custom ViewBox with discrete X-axis behavior"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discrete_x = True
    
    def mouseDragEvent(self, ev, axis=None):
        """Override to handle discrete X-axis dragging"""
        if self.discrete_x and (axis is None or axis == 0):
            # Handle X-axis dragging with discrete steps
            if ev.button() == Qt.MouseButton.LeftButton:
                if ev.isStart():
                    self._drag_start_pos = ev.pos()
                    self._drag_start_range = self.viewRange()
                elif ev.isFinish():
                    self._drag_start_pos = None
                    self._drag_start_range = None
                else:
                    if hasattr(self, '_drag_start_pos') and self._drag_start_pos:
                        delta = ev.pos() - self._drag_start_pos
                        
                        # Calculate discrete X movement
                        x_range = self._drag_start_range[0]
                        x_scale = (x_range[1] - x_range[0]) / self.width()
                        x_offset = delta.x() * x_scale
                        
                        # Snap to integer values
                        x_offset = round(x_offset)
                        
                        new_x_min = self._drag_start_range[0][0] - x_offset
                        new_x_max = self._drag_start_range[0][1] - x_offset
                        
                        # Ensure integer boundaries
                        new_x_min = round(new_x_min)
                        new_x_max = round(new_x_max)
                        
                        self.setXRange(new_x_min, new_x_max, padding=0)
                        
                        # Handle Y-axis normally
                        if axis is None or axis == 1:
                            y_range = self._drag_start_range[1]
                            y_scale = (y_range[1] - y_range[0]) / self.height()
                            y_offset = delta.y() * y_scale
                            self.setYRange(self._drag_start_range[1][0] + y_offset,
                                         self._drag_start_range[1][1] + y_offset, padding=0)
                ev.accept()
        else:
            super().mouseDragEvent(ev, axis)


class EmbeddedAxis(pg.GraphicsWidget):
    """Custom axis widget for chart"""
    
    def __init__(self, orientation='left', parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.labels = []
        self.parent_plot = parent
        
        self._setup_appearance()
        self._setup_dimensions()
        
        self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(1000000)
    
    def _setup_appearance(self):
        """Setup appearance"""
        self.font = QFont('Arial', 9)
        self.text_color = QColor('#999999')
        self.bg_color = QColor(10, 10, 10, 230)
        self.border_color = QColor(42, 42, 42)
        self.pen = QPen(self.border_color, 1)
        self.text_pen = QPen(self.text_color)
    
    def _setup_dimensions(self):
        """Setup dimensions"""
        if self.orientation == 'left':
            self._width = 70
            self._height = 100
        else:
            self._width = 100
            self._height = 35
    
    def boundingRect(self):
        return QRectF(0, 0, self._width, self._height)
    
    def setSize(self, width, height):
        if self._width != width or self._height != height:
            self._width = width
            self._height = height
            self.prepareGeometryChange()
            self.update()
    
    def paint(self, painter, option, widget):
        rect = self.boundingRect()
        
        painter.fillRect(rect, self.bg_color)
        
        painter.setPen(self.pen)
        if self.orientation == 'left':
            painter.drawLine(QPointF(rect.right(), rect.top()), 
                           QPointF(rect.right(), rect.bottom()))
        else:
            painter.drawLine(QPointF(rect.left(), rect.top()), 
                           QPointF(rect.right(), rect.top()))
        
        if self.labels:
            painter.setPen(self.text_pen)
            painter.setFont(self.font)
            
            if self.orientation == 'left':
                for label, pos in self.labels:
                    text_rect = QRectF(5, pos - 10, self._width - 10, 20)
                    painter.drawText(text_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)
            else:
                for label, pos in self.labels:
                    text_rect = QRectF(pos - 40, 5, 80, self._height - 10)
                    painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)
    
    def setLabels(self, labels):
        if labels != self.labels:
            self.labels = labels
            self.update()


class ChartWidget(QWidget):
    """Trading-view style chart widget with discrete X-axis and context-aware grid"""
    
    # Signals
    drawing_added = pyqtSignal(dict)
    time_range_changed = pyqtSignal(str, str)
    time_range_selected = pyqtSignal(str)
    fetch_older_requested = pyqtSignal(object)
    timeframe_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._init_data_structures()
        self.init_ui()
        self.setup_chart_style()
        self._setup_timers()
    
    def _init_data_structures(self):
        """Initialize data structures"""
        self.data = None
        self.original_data = None
        self.date_column = None
        self.x_values = None
        self.current_timeframe = '1D'
        self.min_granularity = None
        self.last_5_days_range = None
        
        self.active_lines = {}
        self.indicator_panes = {}
        self.indicator_overlays = {}
        self.ohlc_buttons = {}
        self.timeframe_buttons = {}
        self.drawings = []
        
        self.drawing_mode = None
        self.drawing_start_point = None
        self.temp_drawing = None
        self.tool_buttons = {}
        
        self.dragging_y_axis = False
        self.dragging_x_axis = False
        self.drag_start_pos = None
        self.drag_start_y_range = None
        self.drag_start_x_range = None
        self.axis_hover = None
        
        # Grid lines storage
        self.v_grid_lines = []
        self.h_grid_lines = []
        self.grid_initialized = False
        
        self._label_cache = {}
        self._data_cache = {}
        self._grid_cache = {}
    
    def _setup_timers(self):
        """Setup update timers"""
        self.axis_update_timer = QTimer()
        self.axis_update_timer.setSingleShot(True)
        self.axis_update_timer.timeout.connect(self.update_axis_labels_and_grid)
        
        self.range_update_timer = QTimer()
        self.range_update_timer.setSingleShot(True)
        self.range_update_timer.timeout.connect(self.update_axis_geometry)
    
    def init_ui(self):
        """Initialize UI"""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)
        
        self.drawing_toolbar = self._create_drawing_toolbar()
        main_layout.addWidget(self.drawing_toolbar)
        
        chart_container = self._create_chart_container()
        main_layout.addWidget(chart_container)
    
    def _create_chart_container(self):
        """Create main chart container"""
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        container.setLayout(layout)
        
        self.ohlc_bar = self._create_ohlc_toggles()
        layout.addWidget(self.ohlc_bar)
        
        self.timeframe_bar = self._create_timeframe_selector()
        layout.addWidget(self.timeframe_bar)
        
        # Main chart
        self.chart_container = pg.GraphicsLayoutWidget()
        self.chart_container.setBackground('#0a0a0a')
        self.chart_container.ci.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create custom ViewBox
        vb = DiscreteViewBox()
        self.main_plot = self.chart_container.addPlot(row=0, col=0, viewBox=vb)
        self._setup_main_plot()
        
        layout.addWidget(self.chart_container, stretch=3)
        
        # Indicator container
        self.indicator_container = QVBoxLayout()
        self.indicator_container.setSpacing(2)
        layout.addLayout(self.indicator_container, stretch=1)
        
        return container
    
    def _setup_main_plot(self):
        """Setup main plot"""
        self.main_plot.hideAxis('left')
        self.main_plot.hideAxis('bottom')
        # Disable default grid
        self.main_plot.showGrid(x=False, y=False)
        self.main_plot.getViewBox().setMouseEnabled(x=True, y=False)
        self.main_plot.getViewBox().disableAutoRange()
        
        self._create_embedded_axes()
        self._initialize_grid_lines()
        self._add_crosshair()
        self._connect_plot_events()
    
    def _initialize_grid_lines(self):
        """Initialize grid line items"""
        # Create vertical grid lines (for X-axis)
        for i in range(500):  # Pre-create enough lines
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen('#1a1a1a', width=1))
            line.setVisible(False)
            self.main_plot.addItem(line, ignoreBounds=True)
            self.v_grid_lines.append(line)
        
        # Create horizontal grid lines (for Y-axis)
        for i in range(20):  # Usually won't need more than 20
            line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#1a1a1a', width=1))
            line.setVisible(False)
            self.main_plot.addItem(line, ignoreBounds=True)
            self.h_grid_lines.append(line)
        
        self.grid_initialized = True
    
    def _create_embedded_axes(self):
        """Create embedded axes"""
        self.y_axis = EmbeddedAxis('left', self.main_plot)
        self.main_plot.scene().addItem(self.y_axis)
        
        self.x_axis = EmbeddedAxis('bottom', self.main_plot)
        self.main_plot.scene().addItem(self.x_axis)
        
        QTimer.singleShot(0, self.update_axis_geometry)
    
    def _add_crosshair(self):
        """Add crosshair"""
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#666', width=1))
        self.main_plot.addItem(self.vLine, ignoreBounds=True)
        
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#666', width=1))
        self.main_plot.addItem(self.hLine, ignoreBounds=True)
        
        self.value_label = pg.TextItem(color='#fff', anchor=(0, 1))
        self.value_label.setFont(QFont('Arial', 10))
        self.main_plot.addItem(self.value_label)
        
        self.proxy = pg.SignalProxy(self.main_plot.scene().sigMouseMoved, 
                                   rateLimit=33, slot=self.update_crosshair)
    
    def _connect_plot_events(self):
        """Connect plot events"""
        self.main_plot.scene().sigMouseClicked.connect(self.on_mouse_clicked)
        self.main_plot.scene().sigMouseMoved.connect(self.on_mouse_moved)
        
        self.main_plot.getViewBox().sigRangeChanged.connect(
            lambda: self.axis_update_timer.start(16))
        
        try:
            self.main_plot.getViewBox().sigResized.connect(
                lambda: self.range_update_timer.start(16))
        except AttributeError:
            pass
        
        self.chart_container.viewport().installEventFilter(self)
    
    def _create_drawing_toolbar(self):
        """Create drawing toolbar"""
        toolbar = QFrame()
        toolbar.setFixedWidth(40)
        toolbar.setStyleSheet("""
            QFrame {
                background-color: #0f0f0f;
                border-right: 1px solid #1a1a1a;
            }
            QPushButton {
                background-color: transparent;
                color: #666;
                border: none;
                padding: 8px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #1a1a1a;
                color: #999;
            }
            QPushButton:checked {
                background-color: #2a2a2a;
                color: #4a4;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(2)
        toolbar.setLayout(layout)
        
        config = DrawingConfig()
        for tool_id in ['trend', 'ray', 'extended', 'horizontal', 'vertical']:
            icon, tooltip, _ = getattr(config, tool_id)
            btn = QPushButton(icon)
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.setFixedSize(32, 32)
            btn.clicked.connect(partial(self.set_drawing_mode, tool_id))
            layout.addWidget(btn)
            self.tool_buttons[tool_id] = btn
        
        layout.addStretch()
        
        clear_btn = QPushButton('Clear')
        clear_btn.setToolTip('Clear All Drawings')
        clear_btn.setFixedSize(32, 32)
        clear_btn.clicked.connect(self.clear_all_drawings)
        layout.addWidget(clear_btn)
        
        return toolbar
    
    def _create_ohlc_toggles(self):
        """Create OHLC toggle buttons"""
        container = QWidget()
        container.setFixedHeight(32)
        container.setStyleSheet("""
            QWidget {
                background-color: #0f0f0f;
                border-bottom: 1px solid #1a1a1a;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(5)
        container.setLayout(layout)
        
        for name, color in OHLC_COLORS.items():
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setFixedHeight(24)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {color};
                    border: none;
                    padding: 0px 8px;
                    font-weight: bold;
                    font-size: 11px;
                }}
                QPushButton:checked {{
                    color: {color};
                }}
                QPushButton:!checked {{
                    color: #333;
                }}
            """)
            btn.clicked.connect(partial(self.toggle_ohlc_line, name))
            layout.addWidget(btn)
            self.ohlc_buttons[name] = btn
        
        layout.addStretch()
        return container
    
    def _create_timeframe_selector(self):
        """Create timeframe selector"""
        container = QWidget()
        container.setFixedHeight(32)
        container.setStyleSheet("""
            QWidget {
                background-color: #0f0f0f;
                border-bottom: 1px solid #1a1a1a;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(3)
        container.setLayout(layout)
        
        timeframes = ['15m', '30m', '1h', '4h', '1D', '1W', '1M', '3M', '1Y']
        self.timeframe_button_group = QButtonGroup(container)
        self.timeframe_button_group.setExclusive(True)
        
        for tf in timeframes:
            btn = QPushButton(tf)
            btn.setCheckable(True)
            btn.setFixedHeight(24)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #1a1a1a;
                    color: #666;
                    border: 1px solid #2a2a2a;
                    padding: 0px 8px;
                    border-radius: 3px;
                    font-size: 11px;
                }
                QPushButton:checked {
                    background-color: #2a2a2a;
                    color: #4a4;
                    border: 1px solid #4a4;
                }
                QPushButton:disabled {
                    background-color: #0a0a0a;
                    color: #333;
                }
            """)
            btn.clicked.connect(partial(self.change_timeframe, tf))
            self.timeframe_button_group.addButton(btn)
            layout.addWidget(btn)
            self.timeframe_buttons[tf] = btn
            
            if tf == '1D':
                btn.setChecked(True)
        
        layout.addStretch()
        return container
    
    def get_price_interval(self, price_range):
        """Get context-aware price interval for Y-axis grid"""
        if price_range <= 0:
            return 0.1
        
        # Determine magnitude
        magnitude = 10 ** math.floor(math.log10(price_range))
        normalized = price_range / magnitude
        
        # Choose appropriate interval based on range
        if price_range > 100:
            # Use $10, $20, $25, or $50 intervals
            if normalized < 2:
                return 10
            elif normalized < 4:
                return 20
            elif normalized < 5:
                return 25
            else:
                return 50
        elif price_range > 20:
            # Use $2, $5, or $10 intervals
            if normalized < 4:
                return 2
            elif normalized < 8:
                return 5
            else:
                return 10
        elif price_range > 5:
            # Use $0.50, $1, or $2 intervals
            if normalized < 10:
                return 0.5
            elif normalized < 20:
                return 1
            else:
                return 2
        elif price_range > 1:
            # Use $0.10, $0.20, or $0.50 intervals
            if normalized < 2:
                return 0.1
            elif normalized < 5:
                return 0.2
            else:
                return 0.5
        else:
            # Never go below $0.10
            return 0.1
    
    def get_label_interval(self, visible_points, timeframe):
        """Get appropriate label interval based on visible points"""
        if visible_points > 300:
            return 50
        elif visible_points > 200:
            return 30
        elif visible_points > 100:
            return 20
        elif visible_points > 50:
            return 10
        elif visible_points > 20:
            return 5
        elif visible_points > 10:
            return 2
        else:
            return 1
    
    def load_data(self, data):
        """Load data into chart"""
        k2_logger.info("Loading data into chart", "CHART")
        
        if isinstance(data, pd.DataFrame):
            self.original_data = data.copy()
        else:
            columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
            self.original_data = pd.DataFrame(data, columns=columns[:len(data[0]) if data else 0])
        
        # Convert numeric columns
        for col in NUMERIC_COLUMNS & set(self.original_data.columns):
            self.original_data[col] = pd.to_numeric(
                self.original_data[col], errors='coerce', downcast='float')
        
        self.detect_granularity()
        self.process_timeframe_data()
        self.display_ohlc_lines()
        self.set_default_view()
        
        self.update_axis_geometry()
        QTimer.singleShot(0, self.update_axis_labels_and_grid)
        
        k2_logger.info(f"Data loaded: {len(self.data)} records", "CHART")
    
    def detect_granularity(self):
        """Detect data granularity"""
        if 'Time' not in self.original_data.columns or self.original_data['Time'].iloc[0] is None:
            self.min_granularity = '1D'
            for tf in INTRADAY_TIMEFRAMES:
                if tf in self.timeframe_buttons:
                    self.timeframe_buttons[tf].setEnabled(False)
        else:
            try:
                if 'datetime' not in self.original_data.columns:
                    self.original_data['datetime'] = pd.to_datetime(
                        self.original_data['Date'].astype(str) + ' ' + 
                        self.original_data['Time'].astype(str),
                        format='%Y-%m-%d %H:%M:%S',
                        errors='coerce'
                    )
                
                time_diffs = self.original_data['datetime'].diff().dt.total_seconds() / 60
                min_interval = time_diffs[time_diffs > 0].min()
                
                if min_interval <= 15:
                    self.min_granularity = '15m'
                    enabled_from = 0
                elif min_interval <= 30:
                    self.min_granularity = '30m'
                    enabled_from = 1
                elif min_interval <= 60:
                    self.min_granularity = '1h'
                    enabled_from = 2
                elif min_interval <= 240:
                    self.min_granularity = '4h'
                    enabled_from = 3
                else:
                    self.min_granularity = '1D'
                    enabled_from = 4
                
                timeframe_order = ['15m', '30m', '1h', '4h', '1D', '1W', '1M', '3M', '1Y']
                for i, tf in enumerate(timeframe_order):
                    if tf in self.timeframe_buttons:
                        self.timeframe_buttons[tf].setEnabled(i >= enabled_from)
                
            except Exception as e:
                k2_logger.error(f"Error detecting granularity: {e}", "CHART")
                self.min_granularity = '1D'
    
    def process_timeframe_data(self):
        """Process data for selected timeframe"""
        self.data = self.original_data.copy()
        
        # Identify date column
        if 'datetime' in self.data.columns:
            self.date_column = 'datetime'
        elif 'Date' in self.data.columns:
            if 'Time' in self.data.columns:
                try:
                    self.data['datetime'] = pd.to_datetime(
                        self.data['Date'].astype(str) + ' ' + self.data['Time'].astype(str),
                        format='%Y-%m-%d %H:%M:%S',
                        errors='coerce'
                    )
                    self.date_column = 'datetime'
                except:
                    self.date_column = 'Date'
            else:
                self.date_column = 'Date'
        
        # Aggregate if needed
        if self.current_timeframe != '1D' and self.date_column == 'datetime':
            self._aggregate_timeframe_data()
        
        self.clear_all()
        self.x_values = np.arange(len(self.data), dtype=np.int32)
    
    def _aggregate_timeframe_data(self):
        """Aggregate data for timeframe"""
        if self.current_timeframe not in TIMEFRAME_RULES:
            return
        
        try:
            self.data.set_index('datetime', inplace=True)
            
            agg_dict = {}
            if 'Open' in self.data.columns: agg_dict['Open'] = 'first'
            if 'High' in self.data.columns: agg_dict['High'] = 'max'
            if 'Low' in self.data.columns: agg_dict['Low'] = 'min'
            if 'Close' in self.data.columns: agg_dict['Close'] = 'last'
            if 'Volume' in self.data.columns: agg_dict['Volume'] = 'sum'
            
            if agg_dict:
                self.data = self.data.resample(
                    TIMEFRAME_RULES[self.current_timeframe],
                    closed='left',
                    label='left'
                ).agg(agg_dict)
                
                self.data = self.data.dropna(how='all')
                self.data.reset_index(inplace=True)
                self.date_column = 'datetime'
        except Exception as e:
            k2_logger.error(f"Error resampling data: {e}", "CHART")
            self.data = self.original_data.copy()
    
    def display_ohlc_lines(self):
        """Display OHLC lines"""
        for col_name in ['Open', 'High', 'Low', 'Close']:
            if col_name in self.data.columns and self.ohlc_buttons[col_name].isChecked():
                self.add_ohlc_line(col_name)
    
    def add_ohlc_line(self, column_name):
        """Add OHLC line"""
        if column_name in self.active_lines or column_name not in self.data.columns:
            return
        
        # Get clean data
        y_values = self.data[column_name].values.astype(np.float64)
        x_values = self.x_values[:len(y_values)]
        
        # Handle NaN values by creating mask
        finite_mask = np.isfinite(y_values)
        if not np.any(finite_mask):
            return
        
        # Get color
        color = OHLC_COLORS.get(column_name, '#ffffff')
        
        # Use PlotDataItem for better control
        plot_item = pg.PlotDataItem(
            x=x_values,
            y=y_values,
            pen=pg.mkPen(color=color, width=2),
            connect='finite',
            skipFiniteCheck=False
        )
        
        # Explicitly ensure no fill
        plot_item.curve.setBrush(None)
        plot_item.curve.setFillLevel(None)
        
        # Add to plot
        self.main_plot.addItem(plot_item)
        self.active_lines[column_name] = plot_item
        
        k2_logger.info(f"Added {column_name} line", "CHART")
    
    def toggle_ohlc_line(self, column_name, visible=None):
        """Toggle OHLC line visibility"""
        if visible is None:
            visible = self.ohlc_buttons[column_name].isChecked()
        
        if visible and column_name not in self.active_lines:
            self.add_ohlc_line(column_name)
        elif not visible and column_name in self.active_lines:
            item = self.active_lines.pop(column_name)
            self.main_plot.removeItem(item)
    
    def set_default_view(self):
        """Set default view"""
        if self.data is None or len(self.data) == 0:
            return
        
        points_per_day = POINTS_PER_DAY.get(self.current_timeframe, 1)
        points_for_5_days = int(5 * points_per_day)
        points_for_5_days = max(20, min(points_for_5_days, len(self.data)))
        
        future_points = int(points_for_5_days * 0.2)
        
        x_max = len(self.data) - 1 + future_points
        x_min = max(0, len(self.data) - 1 - points_for_5_days)
        
        # Ensure integer boundaries
        x_min = int(round(x_min))
        x_max = int(round(x_max))
        
        self.main_plot.setXRange(x_min, x_max, padding=0)
        self._auto_scale_y_range(x_min, min(len(self.data) - 1, x_max))
        
        self.last_5_days_range = ((x_min, x_max), self.main_plot.getViewBox().viewRange()[1])
    
    def _auto_scale_y_range(self, x_min, x_max):
        """Auto-scale Y range"""
        if self.data is None or len(self.data) == 0:
            return
        
        x_min = int(max(0, x_min))
        x_max = int(min(len(self.data) - 1, x_max))
        
        if x_min >= len(self.data) or x_min > x_max:
            return
        
        visible_data = self.data.iloc[x_min:x_max+1]
        
        active_columns = [col for col in ['Open', 'High', 'Low', 'Close'] 
                         if col in visible_data.columns and col in self.active_lines]
        
        if active_columns:
            data_array = visible_data[active_columns].values
            finite_mask = np.isfinite(data_array)
            
            if np.any(finite_mask):
                finite_data = data_array[finite_mask]
                if len(finite_data) > 0:
                    y_min = np.min(finite_data)
                    y_max = np.max(finite_data)
                    
                    padding = (y_max - y_min) * 0.1
                    y_min = max(0, y_min - padding)
                    y_max = y_max + padding
                    
                    self.main_plot.setYRange(y_min, y_max, padding=0)
    
    def change_timeframe(self, timeframe):
        """Change timeframe"""
        if self.current_timeframe == timeframe:
            return
        
        self.current_timeframe = timeframe
        self.timeframe_changed.emit(timeframe)
        
        if self.original_data is not None:
            self.process_timeframe_data()
            self.display_ohlc_lines()
            self.set_default_view()
            
            self.update_axis_geometry()
            QTimer.singleShot(0, self.update_axis_labels_and_grid)
            
            k2_logger.info(f"Timeframe changed to {timeframe}", "CHART")
    
    def update_axis_geometry(self):
        """Update axis geometry"""
        if not hasattr(self, 'y_axis') or not hasattr(self, 'x_axis'):
            return
        
        vb = self.main_plot.getViewBox()
        vb_rect = vb.sceneBoundingRect()
        
        y_axis_width = 70
        x_axis_height = 35
        
        self.y_axis.setSize(y_axis_width, vb_rect.height())
        self.x_axis.setSize(vb_rect.width(), x_axis_height)
        
        self.y_axis.setPos(vb_rect.right() - y_axis_width, vb_rect.top())
        self.x_axis.setPos(vb_rect.left(), vb_rect.bottom() - x_axis_height)
    
    def update_axis_labels_and_grid(self):
        """Update axis labels and grid lines"""
        if not hasattr(self, 'y_axis') or not hasattr(self, 'x_axis'):
            return
        
        vb = self.main_plot.getViewBox()
        x_range, y_range = vb.viewRange()
        
        # Update Y-axis labels and horizontal grid
        self._update_y_axis_and_grid(y_range)
        
        # Update X-axis labels and vertical grid
        self._update_x_axis_and_grid(x_range)
    
    def _update_y_axis_and_grid(self, y_range):
        """Update Y-axis labels and horizontal grid lines with context-aware intervals"""
        y_labels = []
        y_min, y_max = max(0, y_range[0]), y_range[1]
        
        if y_max - y_min > 0:
            # Get context-aware interval
            price_range = y_max - y_min
            interval = self.get_price_interval(price_range)
            
            # Calculate grid positions
            first_line = math.ceil(y_min / interval) * interval
            last_line = math.floor(y_max / interval) * interval
            
            # Generate prices at intervals
            prices = []
            current = first_line
            while current <= last_line:
                prices.append(current)
                current += interval
            
            # Limit to reasonable number of lines
            if len(prices) > 15:
                # Skip every other line
                prices = prices[::2]
            
            axis_height = self.y_axis._height
            
            # Hide all horizontal grid lines first
            for line in self.h_grid_lines:
                line.setVisible(False)
            
            # Update visible lines and labels
            for i, price in enumerate(prices):
                # Calculate position on axis
                pos = axis_height * (1 - (price - y_min) / (y_max - y_min))
                
                # Format label
                if price < 10:
                    label = f"${price:.3f}"
                elif price < 1000:
                    label = f"${price:.2f}"
                else:
                    label = f"${price:,.2f}"
                
                y_labels.append((label, pos))
                
                # Update grid line
                if i < len(self.h_grid_lines):
                    self.h_grid_lines[i].setPos(price)
                    self.h_grid_lines[i].setVisible(True)
        
        self.y_axis.setLabels(y_labels)
    
    def _update_x_axis_and_grid(self, x_range):
        """Update X-axis labels and vertical grid lines"""
        x_labels = []
        
        # Hide all vertical grid lines first
        for line in self.v_grid_lines:
            line.setVisible(False)
        
        if self.data is not None and self.date_column and len(self.data) > 0:
            x_min = int(max(0, round(x_range[0])))
            x_max = int(min(len(self.data) - 1, round(x_range[1])))
            
            visible_points = x_max - x_min + 1
            
            # Get appropriate label interval
            label_interval = self.get_label_interval(visible_points, self.current_timeframe)
            
            axis_width = self.x_axis._width
            
            # Generate labels at intervals
            label_indices = []
            current = x_min
            while current <= x_max:
                if current % label_interval == 0:
                    label_indices.append(current)
                current += 1
            
            # Show grid line at each visible data point
            grid_line_idx = 0
            for x_idx in range(x_min, min(x_max + 1, len(self.data))):
                if grid_line_idx < len(self.v_grid_lines):
                    self.v_grid_lines[grid_line_idx].setPos(x_idx)
                    self.v_grid_lines[grid_line_idx].setVisible(True)
                    grid_line_idx += 1
            
            # Create labels
            for x_idx in label_indices:
                if 0 <= x_idx < len(self.data):
                    # Calculate position on axis
                    pos = axis_width * ((x_idx - x_range[0]) / (x_range[1] - x_range[0]))
                    
                    date_val = self.data.iloc[x_idx][self.date_column]
                    
                    # Format label based on timeframe
                    if isinstance(date_val, (pd.Timestamp, np.datetime64, datetime)):
                        if isinstance(date_val, np.datetime64):
                            date_val = pd.Timestamp(date_val)
                        
                        if self.current_timeframe in INTRADAY_TIMEFRAMES:
                            # Show time for intraday
                            label = date_val.strftime('%H:%M')
                        else:
                            # Show date for daily+
                            label = date_val.strftime('%m/%d')
                    else:
                        try:
                            date_str = str(date_val)
                            if len(date_str) >= 10:
                                label = f"{date_str[5:7]}/{date_str[8:10]}"
                            else:
                                label = date_str[:10]
                        except:
                            label = str(date_val)[:10]
                    
                    x_labels.append((label, pos))
        
        self.x_axis.setLabels(x_labels)
    
    def update_crosshair(self, evt):
        """Update crosshair with discrete X-axis snapping"""
        pos = evt[0]
        if not self.main_plot.sceneBoundingRect().contains(pos):
            return
        
        mousePoint = self.main_plot.getViewBox().mapSceneToView(pos)
        
        # Snap X to nearest data point
        x_raw = mousePoint.x()
        x_snapped = int(round(x_raw))
        x_snapped = max(0, min(x_snapped, len(self.data) - 1) if self.data is not None else x_snapped)
        
        # Set crosshair to snapped position
        self.vLine.setPos(x_snapped)
        self.hLine.setPos(mousePoint.y())
        
        if self.data is not None and self.date_column and len(self.data) > 0:
            if 0 <= x_snapped < len(self.data):
                date_val = self.data.iloc[x_snapped][self.date_column]
                
                # Format date/time based on timeframe
                if isinstance(date_val, (pd.Timestamp, np.datetime64, datetime)):
                    if isinstance(date_val, np.datetime64):
                        date_val = pd.Timestamp(date_val)
                    
                    if self.current_timeframe in INTRADAY_TIMEFRAMES:
                        date_str = date_val.strftime('%Y-%m-%d %H:%M')
                    else:
                        date_str = date_val.strftime('%Y-%m-%d')
                else:
                    date_str = str(date_val)[:16] if pd.notna(date_val) else ""
                
                # Format price
                if mousePoint.y() < 10:
                    price_str = f"${mousePoint.y():.3f}"
                elif mousePoint.y() < 1000:
                    price_str = f"${mousePoint.y():.2f}"
                else:
                    price_str = f"${mousePoint.y():,.2f}"
                
                self.value_label.setText(f"{date_str}\n{price_str}")
                self.value_label.setPos(x_snapped, mousePoint.y())
    
    def set_drawing_mode(self, mode):
        """Set drawing mode"""
        if self.drawing_mode == mode:
            self.drawing_mode = None
            mode = None
        else:
            self.drawing_mode = mode
        
        for tool_id, btn in self.tool_buttons.items():
            btn.setChecked(tool_id == mode)
        
        self.drawing_start_point = None
        if self.temp_drawing:
            self.main_plot.removeItem(self.temp_drawing)
            self.temp_drawing = None
    
    def on_mouse_clicked(self, evt):
        """Handle mouse click"""
        if self.drawing_mode is None:
            return
        
        pos = evt.scenePos()
        
        # Check if click is on axes
        if hasattr(self, 'y_axis'):
            y_axis_rect = QRectF(self.y_axis.pos(), 
                               QPointF(self.y_axis.pos().x() + self.y_axis._width,
                                      self.y_axis.pos().y() + self.y_axis._height))
            if y_axis_rect.contains(pos):
                return
        
        if hasattr(self, 'x_axis'):
            x_axis_rect = QRectF(self.x_axis.pos(),
                               QPointF(self.x_axis.pos().x() + self.x_axis._width,
                                      self.x_axis.pos().y() + self.x_axis._height))
            if x_axis_rect.contains(pos):
                return
        
        if self.main_plot.sceneBoundingRect().contains(pos):
            vb = self.main_plot.getViewBox()
            mousePoint = vb.mapSceneToView(pos)
            
            if self.drawing_mode in ['horizontal', 'vertical']:
                self.create_single_click_drawing(mousePoint)
            else:
                if self.drawing_start_point is None:
                    self.drawing_start_point = mousePoint
                    self.create_temp_drawing(mousePoint)
                else:
                    self.complete_two_click_drawing(mousePoint)
    
    def on_mouse_moved(self, pos):
        """Handle mouse movement"""
        if self.drawing_mode and self.drawing_start_point and self.temp_drawing:
            if self.main_plot.sceneBoundingRect().contains(pos):
                vb = self.main_plot.getViewBox()
                mousePoint = vb.mapSceneToView(pos)
                self.update_temp_drawing(mousePoint)
    
    def create_single_click_drawing(self, point):
        """Create single-click drawing"""
        if self.drawing_mode == 'horizontal':
            line = pg.InfiniteLine(
                pos=point.y(), 
                angle=0,
                pen=pg.mkPen('#ffff00', width=2),
                movable=True
            )
            self.main_plot.addItem(line)
            self.drawings.append(('horizontal', line))
            
        elif self.drawing_mode == 'vertical':
            line = pg.InfiniteLine(
                pos=point.x(), 
                angle=90,
                pen=pg.mkPen('#00ffff', width=2),
                movable=True
            )
            self.main_plot.addItem(line)
            self.drawings.append(('vertical', line))
    
    def create_temp_drawing(self, start_point):
        """Create temporary drawing"""
        if self.drawing_mode in ['trend', 'ray', 'extended']:
            self.temp_drawing = pg.PlotDataItem(
                [start_point.x(), start_point.x()],
                [start_point.y(), start_point.y()],
                pen=pg.mkPen('#ffffff', width=1, style=Qt.PenStyle.DashLine)
            )
            self.main_plot.addItem(self.temp_drawing)
    
    def update_temp_drawing(self, end_point):
        """Update temporary drawing"""
        if self.temp_drawing and self.drawing_start_point:
            self.temp_drawing.setData(
                [self.drawing_start_point.x(), end_point.x()],
                [self.drawing_start_point.y(), end_point.y()]
            )
    
    def complete_two_click_drawing(self, end_point):
        """Complete two-click drawing"""
        if self.temp_drawing:
            self.main_plot.removeItem(self.temp_drawing)
            self.temp_drawing = None
        
        config = DrawingConfig()
        
        if self.drawing_mode == 'trend':
            _, _, color = config.trend
            line = pg.PlotDataItem(
                [self.drawing_start_point.x(), end_point.x()],
                [self.drawing_start_point.y(), end_point.y()],
                pen=pg.mkPen(color, width=2)
            )
            self.main_plot.addItem(line)
            self.drawings.append(('trend', line))
            
        elif self.drawing_mode == 'ray':
            _, _, color = config.ray
            dx = end_point.x() - self.drawing_start_point.x()
            dy = end_point.y() - self.drawing_start_point.y()
            
            extend_factor = 100
            extended_x = self.drawing_start_point.x() + dx * extend_factor
            extended_y = self.drawing_start_point.y() + dy * extend_factor
            
            line = pg.PlotDataItem(
                [self.drawing_start_point.x(), extended_x],
                [self.drawing_start_point.y(), extended_y],
                pen=pg.mkPen(color, width=2)
            )
            self.main_plot.addItem(line)
            self.drawings.append(('ray', line))
            
        elif self.drawing_mode == 'extended':
            _, _, color = config.extended
            dx = end_point.x() - self.drawing_start_point.x()
            dy = end_point.y() - self.drawing_start_point.y()
            
            extend_factor = 100
            x1 = self.drawing_start_point.x() - dx * extend_factor
            y1 = self.drawing_start_point.y() - dy * extend_factor
            x2 = end_point.x() + dx * extend_factor
            y2 = end_point.y() + dy * extend_factor
            
            line = pg.PlotDataItem(
                [x1, x2],
                [y1, y2],
                pen=pg.mkPen(color, width=2)
            )
            self.main_plot.addItem(line)
            self.drawings.append(('extended', line))
        
        self.drawing_start_point = None
    
    def clear_all_drawings(self):
        """Clear all drawings"""
        for drawing_type, item in self.drawings:
            self.main_plot.removeItem(item)
        self.drawings.clear()
        k2_logger.info("Cleared all drawings", "CHART")
    
    def clear_all(self):
        """Clear all lines and indicators"""
        for line in self.active_lines.values():
            self.main_plot.removeItem(line)
        self.active_lines.clear()
        
        for overlay in self.indicator_overlays.values():
            self.main_plot.removeItem(overlay)
        self.indicator_overlays.clear()
        
        for indicator_name in list(self.indicator_panes.keys()):
            self.remove_indicator_pane(indicator_name)
    
    def add_indicator(self, indicator_name, indicator_data, color='#ffff00'):
        """Add indicator overlay"""
        if indicator_name in self.indicator_overlays:
            self.remove_indicator(indicator_name)
        
        if self.x_values is None:
            self.x_values = np.arange(len(indicator_data))
        
        y_values = indicator_data.values.astype(np.float64) if isinstance(indicator_data, pd.Series) else np.array(indicator_data, dtype=np.float64)
        
        # Use PlotDataItem for better control
        plot_item = pg.PlotDataItem(
            x=self.x_values[:len(y_values)],
            y=y_values,
            pen=pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine),
            connect='finite'
        )
        
        # Ensure no fill
        plot_item.curve.setBrush(None)
        plot_item.curve.setFillLevel(None)
        
        self.main_plot.addItem(plot_item)
        self.indicator_overlays[indicator_name] = plot_item
        k2_logger.info(f"Added indicator overlay: {indicator_name}", "CHART")
    
    def remove_indicator(self, indicator_name):
        """Remove indicator"""
        if indicator_name in self.indicator_overlays:
            self.main_plot.removeItem(self.indicator_overlays[indicator_name])
            del self.indicator_overlays[indicator_name]
            k2_logger.info(f"Removed indicator: {indicator_name}", "CHART")
    
    def add_indicator_pane(self, indicator_name, data, chart_type='line'):
        """Add indicator pane"""
        if indicator_name in self.indicator_panes:
            self.remove_indicator_pane(indicator_name)
        
        indicator_plot = pg.PlotWidget()
        indicator_plot.setMaximumHeight(150)
        indicator_plot.showGrid(x=True, y=True, alpha=0.3)
        indicator_plot.setLabel('left', indicator_name)
        indicator_plot.setBackground('#0a0a0a')
        
        indicator_plot.getAxis('left').setPen(pg.mkPen(color='#666'))
        indicator_plot.getAxis('left').setTextPen(pg.mkPen(color='#999'))
        indicator_plot.getAxis('bottom').setPen(pg.mkPen(color='#666'))
        indicator_plot.getAxis('bottom').setTextPen(pg.mkPen(color='#999'))
        
        indicator_plot.setXLink(self.main_plot)
        
        y_values = data.values.astype(np.float64) if isinstance(data, pd.Series) else np.array(data, dtype=np.float64)
        
        if chart_type == 'line':
            plot_item = pg.PlotDataItem(
                x=self.x_values[:len(y_values)],
                y=y_values,
                pen=pg.mkPen(color='#4a4', width=2),
                connect='finite'
            )
            plot_item.curve.setBrush(None)
            plot_item.curve.setFillLevel(None)
            indicator_plot.addItem(plot_item)
            
        elif chart_type == 'bar':
            bargraph = pg.BarGraphItem(
                x=self.x_values[:len(y_values)], 
                height=y_values, 
                width=0.8, 
                brush='#4a4'
            )
            indicator_plot.addItem(bargraph)
        
        self.indicator_container.addWidget(indicator_plot)
        self.indicator_panes[indicator_name] = indicator_plot
        
        k2_logger.info(f"Added indicator pane: {indicator_name}", "CHART")
    
    def remove_indicator_pane(self, indicator_name):
        """Remove indicator pane"""
        if indicator_name in self.indicator_panes:
            widget = self.indicator_panes[indicator_name]
            self.indicator_container.removeWidget(widget)
            widget.deleteLater()
            del self.indicator_panes[indicator_name]
            k2_logger.info(f"Removed indicator pane: {indicator_name}", "CHART")
    
    def reset_zoom(self):
        """Reset zoom"""
        if self.last_5_days_range:
            x_range, y_range = self.last_5_days_range
            self.main_plot.setXRange(x_range[0], x_range[1], padding=0)
            self.main_plot.setYRange(y_range[0], y_range[1], padding=0)
        else:
            self.set_default_view()
    
    def auto_range(self):
        """Auto-range all plots"""
        self.set_default_view()
        for plot in self.indicator_panes.values():
            plot.autoRange()
    
    def zoom(self, factor):
        """Zoom in/out with discrete X handling"""
        vb = self.main_plot.getViewBox()
        current_x_range = vb.viewRange()[0]
        
        x_center = (current_x_range[0] + current_x_range[1]) / 2
        x_width = (current_x_range[1] - current_x_range[0]) * factor
        
        min_points = 20
        max_points = len(self.data) if self.data is not None else 1000
        
        x_width = max(min_points, min(x_width, max_points * 1.2))
        
        # Ensure integer boundaries
        new_x_min = int(round(x_center - x_width / 2))
        new_x_max = int(round(x_center + x_width / 2))
        
        if new_x_min < 0:
            new_x_min = 0
            new_x_max = int(x_width)
        elif new_x_max > max_points:
            new_x_max = max_points
            new_x_min = max_points - int(x_width)
        
        vb.setXRange(new_x_min, new_x_max, padding=0)
        self.auto_scale_y_for_visible_data()
    
    def auto_scale_y_for_visible_data(self):
        """Auto-scale Y for visible data"""
        if self.data is None or len(self.data) == 0:
            return
        
        x_range = self.main_plot.getViewBox().viewRange()[0]
        x_min = int(max(0, round(x_range[0])))
        x_max = int(min(len(self.data) - 1, round(x_range[1])))
        
        if x_min < len(self.data) and x_max < len(self.data) and x_min <= x_max:
            self._auto_scale_y_range(x_min, x_max)
    
    def eventFilter(self, source, event):
        """Event filter for axis interactions"""
        if not hasattr(self, 'y_axis') or not hasattr(self, 'x_axis'):
            return super().eventFilter(source, event)
        
        if event.type() == QEvent.Type.MouseMove:
            pos = event.position()
            scene_pos = self.chart_container.mapToScene(pos.toPoint())
            
            y_axis_rect = QRectF(self.y_axis.pos(), 
                               QPointF(self.y_axis.pos().x() + self.y_axis._width,
                                      self.y_axis.pos().y() + self.y_axis._height))
            
            x_axis_rect = QRectF(self.x_axis.pos(),
                               QPointF(self.x_axis.pos().x() + self.x_axis._width,
                                      self.x_axis.pos().y() + self.x_axis._height))
            
            if y_axis_rect.contains(scene_pos) and not self.dragging_x_axis:
                self.setCursor(QCursor(Qt.CursorShape.SizeVerCursor))
                self.axis_hover = 'y'
            elif x_axis_rect.contains(scene_pos) and not self.dragging_y_axis:
                self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
                self.axis_hover = 'x'
            elif not self.dragging_y_axis and not self.dragging_x_axis:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                self.axis_hover = None
            
            if self.dragging_y_axis and self.drag_start_pos:
                delta_y = scene_pos.y() - self.drag_start_pos.y()
                scale_factor = 1.0 + (delta_y / 200.0)
                
                if self.drag_start_y_range:
                    y_min, y_max = self.drag_start_y_range
                    y_center = (y_min + y_max) / 2
                    y_range = (y_max - y_min) * scale_factor
                    
                    y_range = np.clip(y_range, 0.01, 1e6)
                    
                    new_y_min = max(0, y_center - y_range / 2)
                    new_y_max = y_center + y_range / 2
                    
                    self.main_plot.setYRange(new_y_min, new_y_max, padding=0)
            
            elif self.dragging_x_axis and self.drag_start_pos:
                delta_x = scene_pos.x() - self.drag_start_pos.x()
                scale_factor = 1.0 - (delta_x / 200.0)
                
                if self.drag_start_x_range:
                    x_min, x_max = self.drag_start_x_range
                    x_center = (x_min + x_max) / 2
                    x_range = (x_max - x_min) * scale_factor
                    
                    min_points = 10
                    max_points = len(self.data) * 1.5 if self.data is not None else 1000
                    
                    x_range = np.clip(x_range, min_points, min(max_points, 1e6))
                    
                    # Ensure integer boundaries for discrete X-axis
                    new_x_min = int(round(x_center - x_range / 2))
                    new_x_max = int(round(x_center + x_range / 2))
                    
                    if new_x_min < 0:
                        new_x_min = 0
                        new_x_max = int(x_range)
                    
                    self.main_plot.setXRange(new_x_min, new_x_max, padding=0)
        
        elif event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                pos = event.position()
                scene_pos = self.chart_container.mapToScene(pos.toPoint())
                
                y_axis_rect = QRectF(self.y_axis.pos(),
                                   QPointF(self.y_axis.pos().x() + self.y_axis._width,
                                          self.y_axis.pos().y() + self.y_axis._height))
                
                x_axis_rect = QRectF(self.x_axis.pos(),
                                   QPointF(self.x_axis.pos().x() + self.x_axis._width,
                                          self.x_axis.pos().y() + self.x_axis._height))
                
                if y_axis_rect.contains(scene_pos):
                    self.dragging_y_axis = True
                    self.drag_start_pos = scene_pos
                    self.drag_start_y_range = self.main_plot.getViewBox().viewRange()[1]
                    return True
                    
                elif x_axis_rect.contains(scene_pos):
                    self.dragging_x_axis = True
                    self.drag_start_pos = scene_pos
                    self.drag_start_x_range = self.main_plot.getViewBox().viewRange()[0]
                    return True
        
        elif event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton:
                if self.dragging_y_axis or self.dragging_x_axis:
                    self.dragging_y_axis = False
                    self.dragging_x_axis = False
                    self.drag_start_pos = None
                    self.drag_start_y_range = None
                    self.drag_start_x_range = None
                    return True
        
        elif event.type() == QEvent.Type.MouseButtonDblClick:
            if event.button() == Qt.MouseButton.LeftButton:
                pos = event.position()
                scene_pos = self.chart_container.mapToScene(pos.toPoint())
                
                y_axis_rect = QRectF(self.y_axis.pos(),
                                   QPointF(self.y_axis.pos().x() + self.y_axis._width,
                                          self.y_axis.pos().y() + self.y_axis._height))
                
                x_axis_rect = QRectF(self.x_axis.pos(),
                                   QPointF(self.x_axis.pos().x() + self.x_axis._width,
                                          self.x_axis.pos().y() + self.x_axis._height))
                
                if y_axis_rect.contains(scene_pos) or x_axis_rect.contains(scene_pos):
                    self.reset_zoom()
                    return True
        
        return super().eventFilter(source, event)
    
    def setup_chart_style(self):
        """Setup chart style"""
        if hasattr(self, 'main_plot'):
            self.main_plot.getViewBox().setBackgroundColor('#0a0a0a')
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'axis_update_timer'):
            self.axis_update_timer.stop()
        if hasattr(self, 'range_update_timer'):
            self.range_update_timer.stop()
        
        self.clear_all()
        self.clear_all_drawings()
        
        if hasattr(self, 'proxy'):
            self.proxy = None
        
        for attr in ['vLine', 'hLine', 'value_label']:
            item = getattr(self, attr, None)
            if item and item.scene():
                self.main_plot.removeItem(item)
            setattr(self, attr, None)
        
        for attr in ['y_axis', 'x_axis']:
            item = getattr(self, attr, None)
            if item and item.scene():
                item.scene().removeItem(item)
            setattr(self, attr, None)
        
        # Clear grid lines
        for line in self.v_grid_lines:
            if line.scene():
                self.main_plot.removeItem(line)
        for line in self.h_grid_lines:
            if line.scene():
                self.main_plot.removeItem(line)
        self.v_grid_lines.clear()
        self.h_grid_lines.clear()
        
        self._label_cache.clear()
        self._data_cache.clear()
        self._grid_cache.clear()
        
        gc.collect()