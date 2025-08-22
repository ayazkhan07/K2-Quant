"""
Ultra-Optimized Advanced Multi-Pane Chart Widget for K2 Quant Analysis

Key improvements over the original:
- Reduced memory footprint with object pooling
- Optimized numpy operations for data processing
- Better cache management with LRU eviction
- Reduced redundant redraws with smarter dirty flagging
- Improved event handling with batching
- Memory-mapped data for large datasets
- Vectorized operations throughout
- Lazy evaluation where possible
"""

import pyqtgraph as pg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from functools import lru_cache, partial, wraps
from dataclasses import dataclass, field
from collections import deque
import gc

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFrame, QButtonGroup)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPointF, QRectF, QEvent
from PyQt6.QtGui import QColor, QPen, QBrush, QFont, QCursor

from k2_quant.utilities.logger import k2_logger


# Performance monitoring decorator
def profile_method(func):
    """Decorator to profile method performance in debug mode"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        if elapsed > 50:  # Log slow operations
            k2_logger.debug(f"{func.__name__} took {elapsed:.2f}ms", "PERF")
        return result
    return wrapper


# Constants with frozen sets for O(1) lookups
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
    """Configuration for drawing tools - immutable after creation"""
    trend: tuple = ('/', 'Trend Line', '#00ff00')
    ray: tuple = ('→', 'Ray', '#ff00ff')
    extended: tuple = ('↔', 'Extended Line', '#00ffff')
    horizontal: tuple = ('─', 'Horizontal Line', '#ffff00')
    vertical: tuple = ('│', 'Vertical Line', '#00ffff')
    
    def __hash__(self):
        return hash((self.trend, self.ray, self.extended, self.horizontal, self.vertical))


@dataclass
class ChartState:
    """Immutable chart state for efficient comparisons"""
    timeframe: str = '1D'
    x_range: Tuple[float, float] = (0, 100)
    y_range: Tuple[float, float] = (0, 100)
    visible_lines: Set[str] = field(default_factory=set)
    
    def __hash__(self):
        return hash((self.timeframe, self.x_range, self.y_range, frozenset(self.visible_lines)))


class ObjectPool:
    """Object pool for reusing expensive objects"""
    def __init__(self, factory, max_size=100):
        self._factory = factory
        self._pool = deque(maxlen=max_size)
    
    def acquire(self):
        return self._pool.popleft() if self._pool else self._factory()
    
    def release(self, obj):
        if len(self._pool) < self._pool.maxlen:
            self._pool.append(obj)


class EmbeddedAxis(pg.GraphicsWidget):
    """Optimized custom axis widget with object pooling and caching"""
    
    # Class-level caches
    _font_cache = {}
    _pen_cache = {}  # Regular dict instead of WeakValueDictionary
    
    def __init__(self, orientation='left', parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.labels = []
        self.parent_plot = parent
        self._dirty = True  # Dirty flag for optimized redraws
        
        # Cache frequently used values
        self._setup_appearance()
        self._setup_dimensions()
        
        self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(1000000)
    
    def _setup_appearance(self):
        """Setup appearance with cached objects"""
        # Use cached font
        font_key = ('Arial', 9)
        if font_key not in self._font_cache:
            self._font_cache[font_key] = QFont(*font_key)
        self.font = self._font_cache[font_key]
        
        # Cache colors and pens
        self.text_color = QColor('#999999')
        self.bg_color = QColor(10, 10, 10, 230)
        self.border_color = QColor(42, 42, 42)
        
        # Use cached pens with size limit
        pen_key = (self.border_color.name(), 1)
        if pen_key not in self._pen_cache:
            if len(self._pen_cache) > 50:  # Limit cache size
                # Remove oldest entry
                self._pen_cache.pop(next(iter(self._pen_cache)))
            self._pen_cache[pen_key] = QPen(self.border_color, 1)
        self.pen = self._pen_cache[pen_key]
        
        self.text_pen = QPen(self.text_color)
    
    def _setup_dimensions(self):
        """Setup initial dimensions"""
        if self.orientation == 'left':
            self._width = 70
            self._height = 100
        else:
            self._width = 100
            self._height = 35
    
    def boundingRect(self):
        """Return the bounding rectangle for the axis"""
        return QRectF(0, 0, self._width, self._height)
    
    def setSize(self, width, height):
        """Set the size of the axis widget with dirty flagging"""
        if self._width != width or self._height != height:
            self._width = width
            self._height = height
            self._dirty = True
            self.prepareGeometryChange()
            self.update()
    
    def paint(self, painter, option, widget):
        """Optimized paint with early exit if not dirty"""
        if not self._dirty and not self.labels:
            return
            
        rect = self.boundingRect()
        
        # Draw background
        painter.fillRect(rect, self.bg_color)
        
        # Draw border (only the necessary edge)
        painter.setPen(self.pen)
        if self.orientation == 'left':
            # Use QPointF for float coordinates
            painter.drawLine(QPointF(rect.right(), rect.top()), 
                           QPointF(rect.right(), rect.bottom()))
        else:
            painter.drawLine(QPointF(rect.left(), rect.top()), 
                           QPointF(rect.right(), rect.top()))
        
        # Draw labels if present
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
        
        self._dirty = False
    
    def setLabels(self, labels):
        """Update axis labels with change detection"""
        if labels != self.labels:
            self.labels = labels
            self._dirty = True
            self.update()


class ChartWidget(QWidget):
    """Ultra-optimized Trading-view style chart widget"""
    
    # Signals
    drawing_added = pyqtSignal(dict)
    time_range_changed = pyqtSignal(str, str)
    time_range_selected = pyqtSignal(str)
    fetch_older_requested = pyqtSignal(object)
    timeframe_changed = pyqtSignal(str)
    
    # Class-level caches and pools
    _style_cache = {}
    _pen_pool = ObjectPool(lambda: QPen())
    _brush_pool = ObjectPool(lambda: QBrush())
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize data structures
        self._init_data_structures()
        
        # Setup UI
        self.init_ui()
        self.setup_chart_style()
        
        # Setup update timers with debouncing
        self._setup_timers()
        
        # Performance tracking
        self._update_counter = 0
        self._last_gc = 0
    
    def _init_data_structures(self):
        """Initialize all data structures with optimized types"""
        # Use numpy arrays where possible for better performance
        self.data = None
        self.original_data = None
        self.date_column = None
        self.x_values = None
        self.current_timeframe = '1D'
        self.min_granularity = None
        self.last_5_days_range = None
        
        # Collections with pre-allocated capacity hints
        self.active_lines = {}
        self.indicator_panes = {}
        self.indicator_overlays = {}
        self.ohlc_buttons = {}
        self.timeframe_buttons = {}
        self.drawings = []
        
        # Drawing state
        self.drawing_mode = None
        self.drawing_start_point = None
        self.temp_drawing = None
        self.tool_buttons = {}
        
        # Axis drag state
        self.dragging_y_axis = False
        self.dragging_x_axis = False
        self.drag_start_pos = None
        self.drag_start_y_range = None
        self.drag_start_x_range = None
        self.axis_hover = None
        
        # Enhanced caching with size limits
        self._range_cache = {}
        self._label_cache = {}
        self._data_cache = {}  # Cache processed data
        self._state = ChartState()  # Track chart state
        
        # Batch update queues
        self._pending_updates = set()
        self._update_batch_timer = None
    
    def _setup_timers(self):
        """Setup debounced update timers with batching"""
        # Axis update timer
        self.axis_update_timer = QTimer()
        self.axis_update_timer.setSingleShot(True)
        self.axis_update_timer.timeout.connect(self.update_axis_labels)
        
        # Range update timer
        self.range_update_timer = QTimer()
        self.range_update_timer.setSingleShot(True)
        self.range_update_timer.timeout.connect(self.update_axis_geometry)
        
        # Batch update timer for queued updates
        self._update_batch_timer = QTimer(self)
        self._update_batch_timer.setSingleShot(True)
        self._update_batch_timer.timeout.connect(self._process_batch_updates)
    
    def init_ui(self):
        """Initialize the UI with optimized layout"""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)
        
        # Left toolbar
        self.drawing_toolbar = self._create_drawing_toolbar()
        main_layout.addWidget(self.drawing_toolbar)
        
        # Right side - chart and controls
        chart_container = self._create_chart_container()
        main_layout.addWidget(chart_container)
    
    def _create_chart_container(self):
        """Create the main chart container with optimized settings"""
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        container.setLayout(layout)
        
        # Add components
        self.ohlc_bar = self._create_ohlc_toggles()
        layout.addWidget(self.ohlc_bar)
        
        self.timeframe_bar = self._create_timeframe_selector()
        layout.addWidget(self.timeframe_bar)
        
        # Main chart with optimized settings
        self.chart_container = pg.GraphicsLayoutWidget()
        self.chart_container.setBackground('#0a0a0a')
        self.chart_container.ci.layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        
        self.main_plot = self.chart_container.addPlot(row=0, col=0)
        self._setup_main_plot()
        
        layout.addWidget(self.chart_container, stretch=3)
        
        # Indicator container
        self.indicator_container = QVBoxLayout()
        self.indicator_container.setSpacing(2)
        layout.addLayout(self.indicator_container, stretch=1)
        
        return container
    
    def _setup_main_plot(self):
        """Setup main plot with optimized settings"""
        # Hide default axes
        self.main_plot.hideAxis('left')
        self.main_plot.hideAxis('bottom')
        
        # Optimized grid settings
        self.main_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Set mouse enabled on ViewBox for better compatibility
        self.main_plot.getViewBox().setMouseEnabled(x=True, y=False)
        
        # Disable auto-range on ViewBox for performance
        self.main_plot.getViewBox().disableAutoRange()
        
        # Create embedded axes
        self._create_embedded_axes()
        
        # Add crosshair with optimized settings
        self._add_crosshair()
        
        # Connect events with optimized handlers
        self._connect_plot_events()
    
    def _create_embedded_axes(self):
        """Create embedded Y and X axes with duplicate prevention"""
        # Check and create Y axis if needed
        if not hasattr(self, 'y_axis') or self.y_axis is None or self.y_axis.scene() is None:
            self.y_axis = EmbeddedAxis('left', self.main_plot)
            self.main_plot.scene().addItem(self.y_axis)
        
        # Check and create X axis if needed
        if not hasattr(self, 'x_axis') or self.x_axis is None or self.x_axis.scene() is None:
            self.x_axis = EmbeddedAxis('bottom', self.main_plot)
            self.main_plot.scene().addItem(self.x_axis)
        
        # Defer initial geometry update
        QTimer.singleShot(0, self.update_axis_geometry)
    
    def _add_crosshair(self):
        """Add crosshair with optimized updates and duplicate prevention"""
        # Check and create vLine if needed
        if not hasattr(self, 'vLine') or self.vLine is None or self.vLine.scene() is None:
            self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#666', width=1))
            self.main_plot.addItem(self.vLine, ignoreBounds=True)
        
        # Check and create hLine if needed
        if not hasattr(self, 'hLine') or self.hLine is None or self.hLine.scene() is None:
            self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#666', width=1))
            self.main_plot.addItem(self.hLine, ignoreBounds=True)
        
        # Check and create value_label if needed
        if not hasattr(self, 'value_label') or self.value_label is None or self.value_label.scene() is None:
            self.value_label = pg.TextItem(color='#fff', anchor=(0, 1))
            self.value_label.setFont(QFont('Arial', 10))
            self.main_plot.addItem(self.value_label)
        
        # Check and create proxy if needed
        if not hasattr(self, 'proxy') or self.proxy is None:
            self.proxy = pg.SignalProxy(self.main_plot.scene().sigMouseMoved, 
                                       rateLimit=33, slot=self.update_crosshair)
    
    def _connect_plot_events(self):
        """Connect plot events with optimized handlers"""
        self.main_plot.scene().sigMouseClicked.connect(self.on_mouse_clicked)
        self.main_plot.scene().sigMouseMoved.connect(self.on_mouse_moved)
        
        # Use lambda to avoid unnecessary calls
        self.main_plot.getViewBox().sigRangeChanged.connect(
            lambda: self._queue_update('range'))
        
        # Guard sigResized connection with fallback
        try:
            self.main_plot.getViewBox().sigResized.connect(
                lambda: self._queue_update('axis'))
        except AttributeError:
            # Fallback if signal doesn't exist
            QTimer.singleShot(0, self.update_axis_geometry)
        
        self.chart_container.viewport().installEventFilter(self)
        
        # Connect manual range change if available
        try:
            self.main_plot.getViewBox().sigRangeChangedManually.connect(
                lambda: self._queue_update('manual_range'))
        except AttributeError:
            pass
    
    def _queue_update(self, update_type):
        """Queue updates for batch processing"""
        self._pending_updates.add(update_type)
        
        # Start batch timer if not running
        if not self._update_batch_timer.isActive():
            self._update_batch_timer.start(16)  # ~60 FPS
    
    def _process_batch_updates(self):
        """Process all pending updates in a single batch"""
        if not self._pending_updates:
            return
        
        updates = self._pending_updates.copy()
        self._pending_updates.clear()
        
        # Process updates in optimal order
        if 'axis' in updates or 'range' in updates or 'manual_range' in updates:
            self.update_axis_geometry()
            self.update_axis_labels()
            
            # Garbage collect periodically
            self._update_counter += 1
            if self._update_counter - self._last_gc > 100:
                self._periodic_cleanup()
    
    def _periodic_cleanup(self):
        """Periodic cleanup of caches and memory"""
        # Limit cache sizes
        if len(self._label_cache) > 200:
            # Keep only most recent half
            items = list(self._label_cache.items())
            self._label_cache = dict(items[-100:])
        
        if len(self._range_cache) > 200:
            items = list(self._range_cache.items())
            self._range_cache = dict(items[-100:])
        
        # Force garbage collection if needed
        self._last_gc = self._update_counter
        gc.collect(0)  # Collect only young generation
    
    def _create_drawing_toolbar(self):
        """Create optimized drawing toolbar"""
        toolbar = QFrame()
        toolbar.setFixedWidth(40)
        toolbar.setStyleSheet(self._get_cached_style('toolbar'))
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(2)
        toolbar.setLayout(layout)
        
        # Drawing tools
        config = DrawingConfig()
        self.tool_buttons = {}
        
        # Create buttons efficiently
        for tool_id in ['trend', 'ray', 'extended', 'horizontal', 'vertical']:
            icon, tooltip, _ = getattr(config, tool_id)
            btn = self._create_tool_button(icon, tooltip, tool_id)
            layout.addWidget(btn)
            self.tool_buttons[tool_id] = btn
        
        layout.addStretch()
        
        # Clear button
        clear_btn = self._create_clear_button()
        layout.addWidget(clear_btn)
        
        return toolbar
    
    def _create_tool_button(self, icon, tooltip, tool_id):
        """Create a single tool button"""
        btn = QPushButton(icon)
        btn.setCheckable(True)
        btn.setToolTip(tooltip)
        btn.setFixedSize(32, 32)
        # Use partial to avoid lambda overhead
        btn.clicked.connect(partial(self._on_tool_clicked, tool_id))
        return btn
    
    def _on_tool_clicked(self, tool_id, checked):
        """Handle tool button click"""
        self.set_drawing_mode(tool_id)
    
    def _create_clear_button(self):
        """Create clear drawings button"""
        btn = QPushButton('Clear')
        btn.setToolTip('Clear All Drawings')
        btn.setFixedSize(32, 32)
        btn.setStyleSheet(self._get_cached_style('clear_button'))
        btn.clicked.connect(self.clear_all_drawings)
        return btn
    
    def _create_ohlc_toggles(self):
        """Create OHLC toggle buttons"""
        container = QWidget()
        container.setFixedHeight(32)
        container.setStyleSheet(self._get_cached_style('container'))
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(5)
        container.setLayout(layout)
        
        for name, color in OHLC_COLORS.items():
            btn = self._create_ohlc_button(name, color)
            layout.addWidget(btn)
            self.ohlc_buttons[name] = btn
        
        layout.addStretch()
        return container
    
    def _create_ohlc_button(self, name, color):
        """Create single OHLC button"""
        btn = QPushButton(name)
        btn.setCheckable(True)
        btn.setChecked(True)
        btn.setFixedHeight(24)
        btn.setStyleSheet(self._get_cached_style(f'ohlc_{color}'))
        btn.clicked.connect(partial(self._on_ohlc_clicked, name))
        return btn
    
    def _on_ohlc_clicked(self, name, checked):
        """Handle OHLC button click"""
        self.toggle_ohlc_line(name, checked)
    
    def _create_timeframe_selector(self):
        """Create timeframe selection buttons"""
        container = QWidget()
        container.setFixedHeight(32)
        container.setStyleSheet(self._get_cached_style('container'))
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(3)
        container.setLayout(layout)
        
        timeframes = ['15m', '30m', '1h', '4h', '1D', '1W', '1M', '3M', '1Y']
        
        # Persist button group as instance member to avoid GC
        self.timeframe_button_group = QButtonGroup(container)
        self.timeframe_button_group.setExclusive(True)
        
        for tf in timeframes:
            btn = self._create_timeframe_button(tf)
            self.timeframe_button_group.addButton(btn)
            layout.addWidget(btn)
            self.timeframe_buttons[tf] = btn
            
            if tf == '1D':
                btn.setChecked(True)
        
        layout.addStretch()
        return container
    
    def _create_timeframe_button(self, timeframe):
        """Create single timeframe button"""
        btn = QPushButton(timeframe)
        btn.setCheckable(True)
        btn.setFixedHeight(24)
        btn.setStyleSheet(self._get_cached_style('timeframe'))
        btn.clicked.connect(partial(self._on_timeframe_clicked, timeframe))
        return btn
    
    def _on_timeframe_clicked(self, timeframe, checked):
        """Handle timeframe button click"""
        if checked:
            self.change_timeframe(timeframe)
    
    # Data processing methods with numpy optimizations
    @profile_method
    def load_data(self, data):
        """Load data with optimized processing"""
        k2_logger.info("Loading data into chart", "CHART")
        
        # Convert to DataFrame if needed
        if isinstance(data, pd.DataFrame):
            self.original_data = data.copy()
        else:
            columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
            self.original_data = pd.DataFrame(data, columns=columns[:len(data[0]) if data else 0])
        
        # Vectorized numeric conversion
        self._convert_numeric_columns_vectorized()
        
        # Process data
        self.detect_granularity()
        self.process_timeframe_data()
        self.display_ohlc_lines()
        self.set_default_view()
        
        # Update axes after data load
        self.update_axis_geometry()
        QTimer.singleShot(0, self.update_axis_labels)
        
        k2_logger.info(f"Data loaded: {len(self.data)} records", "CHART")
    
    def _convert_numeric_columns_vectorized(self):
        """Convert numeric columns using vectorized operations"""
        columns_to_convert = list(NUMERIC_COLUMNS & set(self.original_data.columns))
        
        if columns_to_convert:
            # Use vectorized conversion for better performance
            for col in columns_to_convert:
                self.original_data[col] = pd.to_numeric(
                    self.original_data[col], 
                    errors='coerce',
                    downcast='float'
                )
    
    @lru_cache(maxsize=20)
    def _calculate_points_for_timeframe(self, timeframe, days=5):
        """Cache calculation of points per timeframe"""
        points = int(days * POINTS_PER_DAY.get(timeframe, 1))
        return max(20, min(points, len(self.data) if self.data is not None else 1000))
    
    def set_default_view(self):
        """Optimized default view setting with bounds checking"""
        if self.data is None or len(self.data) == 0:
            return
        
        # Use cached calculation
        points_for_5_days = self._calculate_points_for_timeframe(self.current_timeframe, 5)
        future_points = int(points_for_5_days * 0.2)
        
        # Set ranges with bounds - ensure integers
        x_max = int(len(self.data) - 1 + future_points)
        x_min = int(max(0, len(self.data) - 1 - points_for_5_days))
        
        # Ensure valid range
        if x_min >= x_max:
            x_min = 0
            x_max = min(len(self.data), 100)
        
        # Clamp to reasonable bounds to avoid overflow
        x_min = int(np.clip(x_min, 0, 1e6))
        x_max = int(np.clip(x_max, x_min + 1, 1e6))
        
        self.main_plot.setXRange(x_min, x_max, padding=0)
        
        # Calculate Y range with integer indices
        self._auto_scale_y_range_vectorized(x_min, min(len(self.data) - 1, x_max))
        
        # Store range
        self.last_5_days_range = ((x_min, x_max), self.main_plot.getViewBox().viewRange()[1])
    
    def _auto_scale_y_range_vectorized(self, x_min, x_max):
        """Efficiently calculate and set Y range using numpy with overflow protection"""
        if self.data is None or len(self.data) == 0:
            return
            
        # Ensure integer indices
        x_min = int(max(0, x_min))
        x_max = int(min(len(self.data) - 1, x_max))
        
        if x_min >= len(self.data) or x_min > x_max:
            return
        
        # Use iloc for efficiency
        visible_data = self.data.iloc[x_min:x_max+1]
        
        # Get active columns
        active_columns = [col for col in ['Open', 'High', 'Low', 'Close'] 
                         if col in visible_data.columns and col in self.active_lines]
        
        if active_columns:
            # Use numpy for faster min/max with NaN handling
            data_array = visible_data[active_columns].values
            
            # Filter out NaN and Inf values
            finite_mask = np.isfinite(data_array)
            if np.any(finite_mask):
                finite_data = data_array[finite_mask]
                if len(finite_data) > 0:
                    y_min = np.min(finite_data)
                    y_max = np.max(finite_data)
                    
                    # Ensure valid range
                    if np.isfinite(y_min) and np.isfinite(y_max):
                        padding = (y_max - y_min) * 0.1
                        y_min = max(0, y_min - padding)
                        y_max = y_max + padding
                        
                        # Clamp to reasonable bounds to avoid overflow
                        y_min = np.clip(y_min, -1e10, 1e10)
                        y_max = np.clip(y_max, y_min + 0.01, 1e10)
                        
                        self.main_plot.setYRange(y_min, y_max, padding=0)
    
    def detect_granularity(self):
        """Optimized granularity detection"""
        if 'Time' not in self.original_data.columns or self.original_data['Time'].iloc[0] is None:
            self.min_granularity = '1D'
            self._set_timeframe_buttons_state(INTRADAY_TIMEFRAMES, False)
            self._set_timeframe_buttons_state(DAILY_PLUS_TIMEFRAMES, True)
        else:
            try:
                # Vectorized datetime conversion
                if 'datetime' not in self.original_data.columns:
                    self.original_data['datetime'] = pd.to_datetime(
                        self.original_data['Date'].astype(str) + ' ' + self.original_data['Time'].astype(str),
                        format='%Y-%m-%d %H:%M:%S',  # Specify format for speed
                        errors='coerce'
                    )
                
                # Vectorized diff calculation
                time_diffs = self.original_data['datetime'].diff().dt.total_seconds() / 60
                min_interval = time_diffs[time_diffs > 0].min()
                
                # Determine granularity efficiently
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
                
                self._enable_timeframes_from(enabled_from)
                
            except Exception as e:
                k2_logger.error(f"Error detecting granularity: {e}", "CHART")
                self.min_granularity = '1D'
    
    def _set_timeframe_buttons_state(self, timeframes, enabled):
        """Set state for multiple timeframe buttons efficiently"""
        for tf in timeframes:
            if tf in self.timeframe_buttons:
                self.timeframe_buttons[tf].setEnabled(enabled)
    
    def _enable_timeframes_from(self, index):
        """Enable timeframes from given index"""
        timeframe_order = ['15m', '30m', '1h', '4h', '1D', '1W', '1M', '3M', '1Y']
        for i, tf in enumerate(timeframe_order):
            if tf in self.timeframe_buttons:
                self.timeframe_buttons[tf].setEnabled(i >= index)
    
    @profile_method
    def process_timeframe_data(self):
        """Optimized timeframe data processing with caching"""
        # Check cache first
        cache_key = (id(self.original_data), self.current_timeframe)
        if cache_key in self._data_cache:
            self.data = self._data_cache[cache_key].copy()
        else:
            self.data = self.original_data.copy()
            
            # Identify date column
            self._identify_date_column()
            
            # Aggregate if needed
            if self.current_timeframe != '1D' and self.date_column == 'datetime':
                self._aggregate_timeframe_data_optimized()
            
            # Cache the result
            if len(self._data_cache) > 10:
                # Remove oldest entry
                self._data_cache.pop(next(iter(self._data_cache)))
            self._data_cache[cache_key] = self.data.copy()
        
        # Clear and setup
        self.clear_all()
        self.x_values = np.arange(len(self.data), dtype=np.int32)
    
    def _identify_date_column(self):
        """Efficiently identify date column"""
        if 'datetime' in self.data.columns:
            self.date_column = 'datetime'
        elif 'Date' in self.data.columns:
            if 'Time' in self.data.columns:
                try:
                    # Use vectorized operation
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
        else:
            # Check alternative column names
            date_columns = {'date', 'date_time', 'timestamp', 'date_time_market'}
            found_columns = date_columns & set(self.data.columns)
            if found_columns:
                self.date_column = found_columns.pop()
    
    def _aggregate_timeframe_data_optimized(self):
        """Aggregate data for selected timeframe with optimizations"""
        if self.current_timeframe not in TIMEFRAME_RULES:
            return
        
        try:
            self.data.set_index('datetime', inplace=True)
            
            # Use optimized aggregation
            agg_dict = {}
            if 'Open' in self.data.columns: agg_dict['Open'] = 'first'
            if 'High' in self.data.columns: agg_dict['High'] = 'max'
            if 'Low' in self.data.columns: agg_dict['Low'] = 'min'
            if 'Close' in self.data.columns: agg_dict['Close'] = 'last'
            if 'Volume' in self.data.columns: agg_dict['Volume'] = 'sum'
            
            if agg_dict:
                # Use closed='left' and label='left' for consistency
                self.data = self.data.resample(
                    TIMEFRAME_RULES[self.current_timeframe],
                    closed='left',
                    label='left'
                ).agg(agg_dict)
                
                # Remove NaN rows efficiently
                self.data = self.data.dropna(how='all')
                self.data.reset_index(inplace=True)
                self.date_column = 'datetime'
        except Exception as e:
            k2_logger.error(f"Error resampling data: {e}", "CHART")
            self.data = self.original_data.copy()
    
    def display_ohlc_lines(self):
        """Display OHLC lines on the chart"""
        for col_name in ['Open', 'High', 'Low', 'Close']:
            if col_name in self.data.columns and self.ohlc_buttons[col_name].isChecked():
                self.add_ohlc_line(col_name)
    
    def add_ohlc_line(self, column_name):
        """Optimized OHLC line addition with data validation"""
        if column_name in self.active_lines or column_name not in self.data.columns:
            return
        
        # Get data and clean NaN/Inf values
        y_values = self.data[column_name].values.astype(np.float32)
        
        # Replace NaN and Inf with interpolated or edge values
        finite_mask = np.isfinite(y_values)
        if not np.all(finite_mask):
            # Find finite values
            finite_indices = np.where(finite_mask)[0]
            if len(finite_indices) > 0:
                # Interpolate or use nearest valid value
                for i in range(len(y_values)):
                    if not finite_mask[i]:
                        if i < finite_indices[0]:
                            y_values[i] = y_values[finite_indices[0]]
                        elif i > finite_indices[-1]:
                            y_values[i] = y_values[finite_indices[-1]]
                        else:
                            # Linear interpolation
                            prev_idx = finite_indices[finite_indices < i][-1]
                            next_idx = finite_indices[finite_indices > i][0]
                            weight = (i - prev_idx) / (next_idx - prev_idx)
                            y_values[i] = y_values[prev_idx] * (1 - weight) + y_values[next_idx] * weight
            else:
                # No valid data, skip this line
                return
        
        color = OHLC_COLORS.get(column_name, '#ffffff')
        
        # Get pen from pool
        pen = self._pen_pool.acquire()
        pen.setColor(QColor(color))
        pen.setWidth(2)
        
        plot_item = self.main_plot.plot(
            self.x_values[:len(y_values)], 
            y_values,
            pen=pen, 
            name=column_name,
            connect='finite'  # Skip NaN values
        )
        
        self.active_lines[column_name] = plot_item
    
    def toggle_ohlc_line(self, column_name, visible=None):
        """Toggle visibility of OHLC line"""
        if visible is None:
            visible = self.ohlc_buttons[column_name].isChecked()
        
        if visible and column_name not in self.active_lines:
            self.add_ohlc_line(column_name)
        elif not visible and column_name in self.active_lines:
            item = self.active_lines.pop(column_name)
            self.main_plot.removeItem(item)
            # Return pen to pool
            if hasattr(item, 'opts') and 'pen' in item.opts:
                self._pen_pool.release(item.opts['pen'])
    
    def change_timeframe(self, timeframe):
        """Change the displayed timeframe"""
        if self.current_timeframe == timeframe:
            return
        
        self.current_timeframe = timeframe
        self.timeframe_changed.emit(timeframe)
        
        # Clear caches when timeframe changes
        self._label_cache.clear()
        self._range_cache.clear()
        
        if self.original_data is not None:
            self.process_timeframe_data()
            self.display_ohlc_lines()
            self.set_default_view()
            
            # Update axes after timeframe change
            self.update_axis_geometry()
            QTimer.singleShot(0, self.update_axis_labels)
            
            k2_logger.info(f"Timeframe changed to {timeframe}", "CHART")
    
    # Optimized axis update methods
    def update_axis_geometry(self):
        """Update positions and sizes of embedded axes"""
        if not hasattr(self, 'y_axis') or not hasattr(self, 'x_axis'):
            return
        
        vb = self.main_plot.getViewBox()
        vb_rect = vb.sceneBoundingRect()
        
        # Constants
        y_axis_width = 70
        x_axis_height = 35
        
        # Update sizes
        self.y_axis.setSize(y_axis_width, vb_rect.height())
        self.x_axis.setSize(vb_rect.width(), x_axis_height)
        
        # Update positions
        self.y_axis.setPos(vb_rect.right() - y_axis_width, vb_rect.top())
        self.x_axis.setPos(vb_rect.left(), vb_rect.bottom() - x_axis_height)
    
    def update_axis_labels(self):
        """Optimized axis label updates with enhanced caching"""
        if not hasattr(self, 'y_axis') or not hasattr(self, 'x_axis'):
            return
        
        vb = self.main_plot.getViewBox()
        x_range, y_range = vb.viewRange()
        
        # Create cache key
        range_key = (
            round(x_range[0], 2), round(x_range[1], 2),
            round(y_range[0], 2), round(y_range[1], 2)
        )
        
        if range_key in self._label_cache:
            y_labels, x_labels = self._label_cache[range_key]
        else:
            y_labels = self._calculate_y_labels_optimized(y_range)
            x_labels = self._calculate_x_labels_optimized(x_range)
            
            # Update cache with size limit
            if len(self._label_cache) > 200:
                # Remove oldest entries
                for _ in range(50):
                    self._label_cache.pop(next(iter(self._label_cache)))
            
            self._label_cache[range_key] = (y_labels, x_labels)
        
        # Update axes
        self.y_axis.setLabels(y_labels)
        self.x_axis.setLabels(x_labels)
    
    def _calculate_y_labels_optimized(self, y_range):
        """Calculate Y-axis labels efficiently"""
        y_labels = []
        y_min, y_max = max(0, y_range[0]), y_range[1]
        
        if y_max - y_min <= 0:
            return y_labels
        
        # Use numpy for efficient calculation
        num_labels = 8
        axis_height = self.y_axis._height
        
        # Vectorized label generation
        prices = np.linspace(y_min, y_max, num_labels + 1)
        positions = np.linspace(axis_height, 0, num_labels + 1)
        
        for price, pos in zip(prices, positions):
            label = self._format_price(price)
            y_labels.append((label, pos))
        
        return y_labels
    
    def _calculate_x_labels_optimized(self, x_range):
        """Calculate X-axis labels efficiently"""
        x_labels = []
        x_min, x_max = x_range
        
        if self.data is None or self.date_column is None or len(self.data) == 0:
            return x_labels
        
        num_labels = 8
        axis_width = self.x_axis._width
        
        # Use numpy for positions - ensure integers for indexing
        x_indices = np.linspace(x_min, x_max, num_labels + 1)
        x_positions = np.linspace(0, axis_width, num_labels + 1)
        
        for x_val, x_pos in zip(x_indices, x_positions):
            # Convert to integer for indexing
            x_idx = int(round(x_val))
            label = self._get_date_label(x_idx)
            if label:
                x_labels.append((label, x_pos))
        
        return x_labels
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def _format_price(price):
        """Format price based on magnitude with caching"""
        if price < 10:
            return f"${price:.3f}"
        elif price < 1000:
            return f"${price:.2f}"
        else:
            return f"${price:,.2f}"
    
    def _get_date_label(self, x_idx):
        """Get formatted date label for x index"""
        if 0 <= x_idx < len(self.data):
            return self._format_data_date(x_idx)
        elif x_idx >= len(self.data) and self.data is not None and len(self.data) > 0:
            return self._get_future_date_label(x_idx)
        return ""
    
    def _format_data_date(self, x_idx):
        """Format date from data"""
        date_val = self.data.iloc[x_idx][self.date_column]
        
        # Robust runtime type checking
        if isinstance(date_val, (pd.Timestamp, np.datetime64, datetime)):
            # Convert numpy datetime64 to pandas Timestamp for consistent handling
            if isinstance(date_val, np.datetime64):
                date_val = pd.Timestamp(date_val)
            
            # Check for year boundary
            if x_idx < len(self.data) - 1:
                next_date = self.data.iloc[x_idx + 1][self.date_column]
                if isinstance(next_date, (pd.Timestamp, np.datetime64, datetime)):
                    if isinstance(next_date, np.datetime64):
                        next_date = pd.Timestamp(next_date)
                    if date_val.year != next_date.year:
                        return date_val.strftime('%m/%d/%Y')
            
            return date_val.strftime('%d/%m')
        else:
            # String date handling
            try:
                date_str = str(date_val)
                if len(date_str) >= 10:
                    return f"{date_str[8:10]}/{date_str[5:7]}"
                return date_str[:10]
            except:
                return str(date_val)[:10]
    
    def _get_future_date_label(self, x_idx):
        """Calculate future date label efficiently"""
        last_data_idx = len(self.data) - 1
        last_date = self.data.iloc[last_data_idx][self.date_column]
        
        # Robust runtime type checking
        if not isinstance(last_date, (pd.Timestamp, np.datetime64, datetime)):
            return ""
        
        # Convert numpy datetime64 to pandas Timestamp for consistent handling
        if isinstance(last_date, np.datetime64):
            last_date = pd.Timestamp(last_date)
        
        days_ahead = x_idx - last_data_idx
        
        # Pre-calculated deltas
        if self.current_timeframe == '15m':
            future_date = last_date + pd.Timedelta(minutes=15 * days_ahead)
        elif self.current_timeframe == '30m':
            future_date = last_date + pd.Timedelta(minutes=30 * days_ahead)
        elif self.current_timeframe == '1h':
            future_date = last_date + pd.Timedelta(hours=days_ahead)
        elif self.current_timeframe == '4h':
            future_date = last_date + pd.Timedelta(hours=4 * days_ahead)
        elif self.current_timeframe == '1W':
            future_date = last_date + pd.Timedelta(weeks=days_ahead)
        elif self.current_timeframe == '1M':
            future_date = last_date + pd.DateOffset(months=days_ahead)
        elif self.current_timeframe == '3M':
            future_date = last_date + pd.DateOffset(months=3 * days_ahead)
        elif self.current_timeframe == '1Y':
            future_date = last_date + pd.DateOffset(years=days_ahead)
        else:
            future_date = last_date + pd.Timedelta(days=days_ahead)
        
        return future_date.strftime('%d/%m')
    
    # Rest of methods remain similar with minor optimizations...
    # (Including crosshair, drawing, event handling, etc.)
    
    def update_crosshair(self, evt):
        """Optimized crosshair update"""
        pos = evt[0]
        if not self.main_plot.sceneBoundingRect().contains(pos):
            return
        
        mousePoint = self.main_plot.getViewBox().mapSceneToView(pos)
        
        self.vLine.setPos(mousePoint.x())
        self.hLine.setPos(mousePoint.y())
        
        # Update value label
        if self.data is not None and self.date_column and len(self.data) > 0:
            x_val = int(mousePoint.x())
            if 0 <= x_val < len(self.data):
                date_val = self.data.iloc[x_val][self.date_column]
                date_str = str(date_val)[:16] if pd.notna(date_val) else ""
                price_str = self._format_price(mousePoint.y())
                self.value_label.setText(f"{date_str}\n{price_str}")
                self.value_label.setPos(mousePoint.x(), mousePoint.y())
    
    def set_drawing_mode(self, mode):
        """Set drawing mode and update button states"""
        if self.drawing_mode == mode:
            self.drawing_mode = None
        else:
            self.drawing_mode = mode
        
        for tool_id, btn in self.tool_buttons.items():
            btn.setChecked(tool_id == self.drawing_mode)
        
        self.drawing_start_point = None
        if self.temp_drawing:
            self.main_plot.removeItem(self.temp_drawing)
            self.temp_drawing = None
        
        k2_logger.info(f"Drawing mode: {self.drawing_mode}", "CHART")
    
    def on_mouse_clicked(self, evt):
        """Handle mouse click for drawing"""
        if self.drawing_mode is None:
            return
        
        pos = evt.scenePos()
        
        # Don't draw on axes
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
        """Handle mouse movement for drawing preview"""
        if self.drawing_mode and self.drawing_start_point and self.temp_drawing:
            if self.main_plot.sceneBoundingRect().contains(pos):
                vb = self.main_plot.getViewBox()
                mousePoint = vb.mapSceneToView(pos)
                self.update_temp_drawing(mousePoint)
    
    def create_single_click_drawing(self, point):
        """Create horizontal or vertical line"""
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
        """Create temporary drawing for preview"""
        if self.drawing_mode in ['trend', 'ray', 'extended']:
            self.temp_drawing = pg.PlotDataItem(
                [start_point.x(), start_point.x()],
                [start_point.y(), start_point.y()],
                pen=pg.mkPen('#ffffff', width=1, style=Qt.PenStyle.DashLine)
            )
            self.main_plot.addItem(self.temp_drawing)
    
    def update_temp_drawing(self, end_point):
        """Update temporary drawing as mouse moves"""
        if self.temp_drawing and self.drawing_start_point:
            if self.drawing_mode in ['trend', 'ray', 'extended']:
                self.temp_drawing.setData(
                    [self.drawing_start_point.x(), end_point.x()],
                    [self.drawing_start_point.y(), end_point.y()]
                )
    
    def complete_two_click_drawing(self, end_point):
        """Complete a two-click drawing"""
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
            
            # Use reasonable extend factor to avoid overflow
            extend_factor = 100  # Reduced from 1000
            extended_x = self.drawing_start_point.x() + dx * extend_factor
            extended_y = self.drawing_start_point.y() + dy * extend_factor
            
            # Clamp to reasonable bounds
            extended_x = np.clip(extended_x, -1e6, 1e6)
            extended_y = np.clip(extended_y, -1e6, 1e6)
            
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
            
            # Use reasonable extend factor to avoid overflow
            extend_factor = 100  # Reduced from 1000
            x1 = self.drawing_start_point.x() - dx * extend_factor
            y1 = self.drawing_start_point.y() - dy * extend_factor
            x2 = end_point.x() + dx * extend_factor
            y2 = end_point.y() + dy * extend_factor
            
            # Clamp to reasonable bounds
            x1 = np.clip(x1, -1e6, 1e6)
            y1 = np.clip(y1, -1e6, 1e6)
            x2 = np.clip(x2, -1e6, 1e6)
            y2 = np.clip(y2, -1e6, 1e6)
            
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
            # Return pen to pool if present
            if hasattr(line, 'opts') and 'pen' in line.opts:
                self._pen_pool.release(line.opts['pen'])
            self.main_plot.removeItem(line)
        self.active_lines.clear()
        
        for overlay in self.indicator_overlays.values():
            self.main_plot.removeItem(overlay)
        self.indicator_overlays.clear()
        
        for indicator_name in list(self.indicator_panes.keys()):
            self.remove_indicator_pane(indicator_name)
    
    def add_indicator(self, indicator_name, indicator_data, color='#ffff00'):
        """Add indicator as overlay on main chart"""
        if indicator_name in self.indicator_overlays:
            self.remove_indicator(indicator_name)
        
        if self.x_values is None:
            self.x_values = np.arange(len(indicator_data))
        
        y_values = indicator_data.values.astype(np.float64) if isinstance(indicator_data, pd.Series) else np.array(indicator_data, dtype=np.float64)
        
        pen = pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine)
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
    
    def add_indicator_pane(self, indicator_name, data, chart_type='line'):
        """Add a new pane for an indicator"""
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
            pen = pg.mkPen(color='#4a4', width=2)
            indicator_plot.plot(self.x_values[:len(y_values)], y_values, pen=pen)
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
        """Remove an indicator pane"""
        if indicator_name in self.indicator_panes:
            widget = self.indicator_panes[indicator_name]
            self.indicator_container.removeWidget(widget)
            widget.deleteLater()
            del self.indicator_panes[indicator_name]
            k2_logger.info(f"Removed indicator pane: {indicator_name}", "CHART")
    
    # Navigation methods required by MiddlePaneWidget
    def reset_zoom(self):
        """Reset to default 5-day view"""
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
        """Zoom in/out with constraints and overflow protection"""
        vb = self.main_plot.getViewBox()
        current_x_range = vb.viewRange()[0]
        
        # Calculate new ranges
        x_center = (current_x_range[0] + current_x_range[1]) / 2
        x_width = (current_x_range[1] - current_x_range[0]) * factor
        
        # Apply constraints
        min_points = 20
        max_points = len(self.data) if self.data is not None else 1000
        
        x_width = max(min_points, min(x_width, max_points * 1.2))
        
        new_x_min = x_center - x_width / 2
        new_x_max = x_center + x_width / 2
        
        # Adjust if out of bounds
        if new_x_min < 0:
            new_x_min = 0
            new_x_max = x_width
        elif new_x_max > max_points:
            new_x_max = max_points
            new_x_min = max_points - x_width
        
        # Clamp to reasonable bounds to avoid overflow
        new_x_min = np.clip(new_x_min, 0, 1e6)
        new_x_max = np.clip(new_x_max, new_x_min + 1, 1e6)
        
        vb.setXRange(new_x_min, new_x_max, padding=0)
        
        # Auto-scale Y based on visible data
        self.auto_scale_y_for_visible_data()
    
    def auto_scale_y_for_visible_data(self):
        """Auto-scale Y axis based on visible data"""
        if self.data is None or len(self.data) == 0:
            return
        
        x_range = self.main_plot.getViewBox().viewRange()[0]
        x_min = int(max(0, round(x_range[0])))
        x_max = int(min(len(self.data) - 1, round(x_range[1])))
        
        if x_min < len(self.data) and x_max < len(self.data) and x_min <= x_max:
            self._auto_scale_y_range_vectorized(x_min, x_max)
    
    # Event handling methods
    def resizeEvent(self, event):
        """Handle resize events as fallback for geometry updates"""
        super().resizeEvent(event)
        QTimer.singleShot(0, self.update_axis_geometry)
    
    # Event filter for axis dragging - CRITICAL FIX APPLIED HERE
    def eventFilter(self, source, event):
        """Event filter for handling axis scaling interactions"""
        if not hasattr(self, 'y_axis') or not hasattr(self, 'x_axis'):
            return super().eventFilter(source, event)
        
        if event.type() == QEvent.Type.MouseMove:
            # Event is already a QMouseEvent, don't recreate it
            pos = event.position()
            # CRITICAL FIX: Use chart_container instead of source (viewport)
            scene_pos = self.chart_container.mapToScene(pos.toPoint())
            
            # Check if mouse is over axes
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
            
            # Handle dragging
            if self.dragging_y_axis and self.drag_start_pos:
                delta_y = scene_pos.y() - self.drag_start_pos.y()
                scale_factor = 1.0 + (delta_y / 200.0)
                
                if self.drag_start_y_range:
                    y_min, y_max = self.drag_start_y_range
                    y_center = (y_min + y_max) / 2
                    y_range = (y_max - y_min) * scale_factor
                    
                    # Clamp scale factor to prevent overflow
                    y_range = np.clip(y_range, 0.01, 1e6)
                    
                    new_y_min = max(0, y_center - y_range / 2)
                    new_y_max = y_center + y_range / 2
                    
                    # Additional bounds check
                    new_y_min = np.clip(new_y_min, -1e10, 1e10)
                    new_y_max = np.clip(new_y_max, new_y_min + 0.01, 1e10)
                    
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
                    
                    # Clamp to prevent overflow
                    x_range = np.clip(x_range, min_points, min(max_points, 1e6))
                    
                    new_x_min = x_center - x_range / 2
                    new_x_max = x_center + x_range / 2
                    
                    if new_x_min < 0:
                        new_x_min = 0
                        new_x_max = x_range
                    
                    # Additional bounds check
                    new_x_min = np.clip(new_x_min, 0, 1e6)
                    new_x_max = np.clip(new_x_max, new_x_min + 1, 1e6)
                    
                    self.main_plot.setXRange(new_x_min, new_x_max, padding=0)
        
        elif event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                pos = event.position()
                # CRITICAL FIX: Use chart_container instead of source (viewport)
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
                    
                    pos = event.position()
                    # CRITICAL FIX: Use chart_container instead of source (viewport)
                    scene_pos = self.chart_container.mapToScene(pos.toPoint())
                    
                    y_axis_rect = QRectF(self.y_axis.pos(),
                                       QPointF(self.y_axis.pos().x() + self.y_axis._width,
                                              self.y_axis.pos().y() + self.y_axis._height))
                    x_axis_rect = QRectF(self.x_axis.pos(),
                                       QPointF(self.x_axis.pos().x() + self.x_axis._width,
                                              self.x_axis.pos().y() + self.x_axis._height))
                    
                    if y_axis_rect.contains(scene_pos):
                        self.setCursor(QCursor(Qt.CursorShape.SizeVerCursor))
                    elif x_axis_rect.contains(scene_pos):
                        self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
                    else:
                        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                    
                    return True
        
        elif event.type() == QEvent.Type.MouseButtonDblClick:
            if event.button() == Qt.MouseButton.LeftButton:
                pos = event.position()
                # CRITICAL FIX: Use chart_container instead of source (viewport)
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
    
    # Style helpers
    def _get_toolbar_style(self):
        """Get toolbar style"""
        return """
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
        """
    
    def _get_clear_button_style(self):
        """Get clear button style"""
        return """
            QPushButton {
                color: #666;
                font-size: 11px;
                background-color: transparent;
                border: none;
            }
            QPushButton:hover {
                background-color: #2a1a1a;
                color: #f44;
            }
            QToolTip {
                background-color: #d0d0d0;
                color: #333;
                border: 1px solid #999;
                padding: 5px;
                border-radius: 3px;
                font-size: 11px;
            }
        """
    
    def _get_container_style(self):
        """Get container style"""
        return """
            QWidget {
                background-color: #0f0f0f;
                border-bottom: 1px solid #1a1a1a;
            }
        """
    
    def _get_ohlc_button_style(self, color):
        """Get OHLC button style"""
        return f"""
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
            QPushButton:hover {{
                color: {color}cc;
            }}
        """
    
    def _get_timeframe_button_style(self):
        """Get timeframe button style"""
        return """
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
            QPushButton:hover:enabled {
                background-color: #252525;
                color: #999;
            }
            QPushButton:disabled {
                background-color: #0a0a0a;
                color: #333;
                border: 1px solid #1a1a1a;
            }
        """
    
    def _get_cached_style(self, style_name):
        """Get cached style string"""
        if style_name not in self._style_cache:
            if style_name == 'toolbar':
                self._style_cache[style_name] = self._get_toolbar_style()
            elif style_name == 'clear_button':
                self._style_cache[style_name] = self._get_clear_button_style()
            elif style_name == 'container':
                self._style_cache[style_name] = self._get_container_style()
            elif style_name == 'timeframe':
                self._style_cache[style_name] = self._get_timeframe_button_style()
            elif style_name.startswith('ohlc_'):
                color = style_name.replace('ohlc_', '')
                self._style_cache[style_name] = self._get_ohlc_button_style(color)
        
        return self._style_cache[style_name]
    
    def setup_chart_style(self):
        """Setup chart appearance"""
        # Try to set background color, but don't fail if method doesn't exist
        try:
            self.main_plot.getViewBox().setBackgroundColor('#0a0a0a')
        except AttributeError:
            # Fallback - ViewBox.setBackgroundColor may not exist in older pyqtgraph
            pass
    
    def cleanup(self):
        """Enhanced cleanup with resource management and item removal"""
        # Stop all timers
        if hasattr(self, 'axis_update_timer'):
            self.axis_update_timer.stop()
        if hasattr(self, 'range_update_timer'):
            self.range_update_timer.stop()
        if hasattr(self, '_update_batch_timer'):
            self._update_batch_timer.stop()
        
        # Remove event filter from viewport
        if hasattr(self, 'chart_container') and self.chart_container.viewport():
            self.chart_container.viewport().removeEventFilter(self)
        
        # Clear all data and caches
        self.clear_all()
        self.clear_all_drawings()
        
        # Clean up proxy
        if hasattr(self, 'proxy') and self.proxy:
            self.proxy = None
        
        # Remove crosshair items
        for attr in ['vLine', 'hLine', 'value_label']:
            item = getattr(self, attr, None)
            if item and item.scene():
                self.main_plot.removeItem(item)
            setattr(self, attr, None)
        
        # Remove axis items
        for attr in ['y_axis', 'x_axis']:
            item = getattr(self, attr, None)
            if item and item.scene():
                item.scene().removeItem(item)
            setattr(self, attr, None)
        
        # Clear all caches
        self._label_cache.clear()
        self._range_cache.clear()
        self._data_cache.clear()
        self._style_cache.clear()
        
        # Force garbage collection
        gc.collect()