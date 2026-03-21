"""
Enhanced Chart Widget for K2 Quant Analysis - Fully Fixed Version
All issues resolved:
- Fixed duplicate OHLC lines when fetching data
- Fixed type errors with inf checking
- Fixed Y-axis scaling to show actual price range
- Fixed X-axis labels disappearing
- Fixed viewport positioning to show actual data
- Fixed boundary constraints properly
"""

import pyqtgraph as pg
import pandas as pd
import numpy as np
import time as _time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from functools import lru_cache, partial, wraps
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import gc
import math

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFrame, QButtonGroup)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPointF, QRectF, QEvent, QThread
from PyQt6.QtGui import QColor, QPen, QBrush, QFont, QCursor

# Logger setup
try:
    from k2_quant.utilities.logger import k2_logger
except ImportError:
    class DummyLogger:
        def info(self, msg, category=""): print(f"INFO [{category}]: {msg}")
        def error(self, msg, category=""): print(f"ERROR [{category}]: {msg}")
        def debug(self, msg, category=""): pass
        def warning(self, msg, category=""): print(f"WARNING [{category}]: {msg}")
    k2_logger = DummyLogger()

# Service import
try:
    from k2_quant.utilities.services.stock_data_service import stock_service
except ImportError:
    stock_service = None


# Time span enumeration
class TimeSpan(Enum):
    """Time span categories for adaptive formatting"""
    INTRADAY_MINUTES = "minutes"     # < 1 hour
    INTRADAY_HOURS = "hours"         # 1 hour - 1 day
    DAILY = "daily"                  # 1-7 days
    WEEKLY = "weekly"                # 1-4 weeks
    MONTHLY = "monthly"              # 1-3 months
    QUARTERLY = "quarterly"          # 3-12 months
    YEARLY = "yearly"                # 1-5 years
    MULTI_YEAR = "multi_year"        # > 5 years


# Constants
OHLC_COLORS = {
    'Open': '#00ff00',
    'High': '#0080ff', 
    'Low': '#ff0000',
    'Close': '#ffff00'
}

TIMEFRAME_CONFIG = {
    '1m': {'rule': '1min',  'points_per_day': 390, 'interval_minutes': 1},
    '5m': {'rule': '5min',  'points_per_day': 78,  'interval_minutes': 5},
    '15m': {'rule': '15min', 'points_per_day': 26, 'interval_minutes': 15},
    '30m': {'rule': '30min', 'points_per_day': 13, 'interval_minutes': 30},
    '1h': {'rule': '1h', 'points_per_day': 7, 'interval_minutes': 60},
    '4h': {'rule': '4h', 'points_per_day': 2, 'interval_minutes': 240},
    '1D': {'rule': 'D', 'points_per_day': 1, 'interval_minutes': 1440}
}

LOADED_BARS = 500
DEFAULT_VISIBLE_BARS = 200

# Time format configurations
# Intraday: HH:MM normally | day number at date change | MMM-YY at month change
# Daily+:   day number normally | MMM-YY at month change
TIME_FORMATS = {
    TimeSpan.INTRADAY_MINUTES: {
        'major': '%H:%M',
        'date_change': '%-d',
        'month_change': '%b-%y',
    },
    TimeSpan.INTRADAY_HOURS: {
        'major': '%H:%M',
        'date_change': '%-d',
        'month_change': '%b-%y',
    },
    TimeSpan.DAILY: {
        'major': '%-d',
        'date_change': None,
        'month_change': '%b-%y',
    },
    TimeSpan.WEEKLY: {
        'major': '%-d',
        'date_change': None,
        'month_change': '%b-%y',
    },
    TimeSpan.MONTHLY: {
        'major': '%-d-%b',
        'date_change': None,
        'month_change': '%b-%y',
    },
    TimeSpan.QUARTERLY: {
        'major': '%b-%y',
        'date_change': None,
        'month_change': None,
    },
    TimeSpan.YEARLY: {
        'major': '%b-%y',
        'date_change': None,
        'month_change': None,
    },
    TimeSpan.MULTI_YEAR: {
        'major': '%Y',
        'date_change': None,
        'month_change': None,
    },
}

NUMERIC_COLUMNS = frozenset(['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP'])
INTRADAY_TIMEFRAMES = frozenset(['1m', '5m', '15m', '30m', '1h', '4h'])
DAILY_PLUS_TIMEFRAMES = frozenset(['1D'])


# Utility functions
def safe_strftime(date_val, format_string, default=""):
    """Safely format a date, handling NaT and Windows strftime quirks."""
    try:
        if pd.isna(date_val):
            return default
        if isinstance(date_val, np.datetime64):
            date_val = pd.Timestamp(date_val)
            if pd.isna(date_val):
                return default
        if not hasattr(date_val, 'strftime'):
            return str(date_val)

        import sys
        fmt = format_string
        if sys.platform == 'win32':
            fmt = fmt.replace('%-', '%#')

        formatted = date_val.strftime(fmt)
        if isinstance(formatted, str) and formatted.startswith("0"):
            formatted = formatted[1:]
        return formatted
    except (ValueError, AttributeError, TypeError):
        return default


def get_time_span(start_date, end_date) -> Tuple[TimeSpan, float]:
    """
    Determine the time span category and duration in seconds
    """
    if pd.isna(start_date) or pd.isna(end_date):
        return TimeSpan.DAILY, 86400  # Default to daily
    
    duration = (end_date - start_date).total_seconds()
    
    if duration < 3600:  # < 1 hour
        return TimeSpan.INTRADAY_MINUTES, duration
    elif duration < 86400:  # < 1 day
        return TimeSpan.INTRADAY_HOURS, duration
    elif duration < 604800:  # < 1 week
        return TimeSpan.DAILY, duration
    elif duration < 2592000:  # < 30 days
        return TimeSpan.WEEKLY, duration
    elif duration < 7776000:  # < 90 days
        return TimeSpan.MONTHLY, duration
    elif duration < 31536000:  # < 1 year
        return TimeSpan.QUARTERLY, duration
    elif duration < 157680000:  # < 5 years
        return TimeSpan.YEARLY, duration
    else:
        return TimeSpan.MULTI_YEAR, duration


def snap_to_time_boundary(dt, interval_type: TimeSpan):
    """
    Snap a datetime to the nearest meaningful boundary
    """
    if pd.isna(dt):
        return dt
    
    if interval_type == TimeSpan.INTRADAY_MINUTES:
        # Snap to 5, 15, or 30 minute boundaries
        minute = dt.minute
        if minute % 30 == 0:
            return dt.replace(second=0, microsecond=0)
        elif minute % 15 == 0:
            return dt.replace(second=0, microsecond=0)
        else:
            snap_minute = (minute // 5) * 5
            return dt.replace(minute=snap_minute, second=0, microsecond=0)
    
    elif interval_type == TimeSpan.INTRADAY_HOURS:
        # Snap to hour boundaries
        return dt.replace(minute=0, second=0, microsecond=0)
    
    elif interval_type in [TimeSpan.DAILY, TimeSpan.WEEKLY]:
        # Snap to day boundaries
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    elif interval_type == TimeSpan.MONTHLY:
        # Snap to week boundaries (Monday)
        days_since_monday = dt.weekday()
        if days_since_monday > 0:
            dt = dt - timedelta(days=days_since_monday)
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    elif interval_type == TimeSpan.QUARTERLY:
        # Snap to month boundaries
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    elif interval_type == TimeSpan.YEARLY:
        # Snap to quarter boundaries
        month = dt.month
        quarter_month = ((month - 1) // 3) * 3 + 1
        return dt.replace(month=quarter_month, day=1, hour=0, minute=0, second=0, microsecond=0)
    
    else:  # MULTI_YEAR
        # Snap to year boundaries
        return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def debounce(wait_ms):
    """Debounce decorator for methods"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, '_debounce_timers'):
                self._debounce_timers = {}
            
            timer_name = func.__name__
            if timer_name in self._debounce_timers:
                self._debounce_timers[timer_name].stop()
            
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: func(self, *args, **kwargs))
            timer.start(wait_ms)
            self._debounce_timers[timer_name] = timer
        return wrapper
    return decorator


@dataclass
class DrawingConfig:
    """Configuration for drawing tools"""
    trend: tuple = ('/', 'Trend Line', '#00ff00')
    ray: tuple = ('→', 'Ray', '#ff00ff')
    extended: tuple = ('↔', 'Extended Line', '#00ffff')
    horizontal: tuple = ('─', 'Horizontal Line', '#ffff00')
    vertical: tuple = ('│', 'Vertical Line', '#00ffff')


@dataclass
class ViewportState:
    """Immutable viewport state"""
    x_min: int
    x_max: int
    y_min: float
    y_max: float
    
    def __hash__(self):
        return hash((self.x_min, self.x_max, self.y_min, self.y_max))


class TimeAxisManager:
    """Manages intelligent time axis labeling and formatting.
    Uses a cached numpy datetime array + binary search for O(log N) lookups."""

    def __init__(self):
        self.max_labels = 20
        self.min_label_spacing = 50

    def calculate_time_labels(self, dt_cache, x_range, axis_width):
        """Place a label at every Nth visible data point.
        Intraday: HH:MM | day number at date change | MMM-YY at month change.
        Daily:    day number | MMM-YY at month change."""
        if dt_cache is None or len(dt_cache) == 0:
            return []

        x_min = int(max(0, round(x_range[0])))
        x_max = int(min(len(dt_cache) - 1, round(x_range[1])))
        if x_min > x_max:
            return []

        visible = x_max - x_min + 1
        if visible < 1:
            return []

        start_date = pd.Timestamp(dt_cache[x_min])
        end_date = pd.Timestamp(dt_cache[x_max])
        if pd.isna(start_date) or pd.isna(end_date):
            return []

        time_span, _ = get_time_span(start_date, end_date)
        fmt = TIME_FORMATS[time_span]
        major_fmt = fmt['major']
        date_fmt = fmt.get('date_change')
        month_fmt = fmt.get('month_change')
        x_span = x_range[1] - x_range[0]
        if x_span <= 0:
            return []

        max_labels = max(1, int(axis_width / self.min_label_spacing))
        step = max(1, visible // max_labels)

        labels = []
        last_day = None
        last_month = None
        last_txt = None
        for idx in range(x_min, x_max + 1, step):
            dt = pd.Timestamp(dt_cache[idx])
            if pd.isna(dt):
                continue
            x_pos = axis_width * ((idx - x_range[0]) / x_span)

            if month_fmt and last_month is not None and dt.month != last_month:
                txt = safe_strftime(dt, month_fmt, '')
            elif date_fmt and last_day is not None and dt.day != last_day:
                txt = safe_strftime(dt, date_fmt, '')
            else:
                txt = safe_strftime(dt, major_fmt, '')

            if txt and txt != last_txt:
                labels.append((txt, x_pos))
                last_txt = txt
            last_day = dt.day
            last_month = dt.month

        return labels
    
    def calculate_grid_positions(self, dt_cache, x_range):
        """Calculate grid positions using binary search on cached datetimes."""
        if dt_cache is None or len(dt_cache) == 0:
            return []

        x_min = int(max(0, round(x_range[0])))
        x_max = int(min(len(dt_cache) - 1, round(x_range[1])))
        if x_min >= x_max:
            return []

        visible = x_max - x_min + 1
        if visible <= 50:
            return list(range(x_min, x_max + 1))

        start_date = pd.Timestamp(dt_cache[x_min])
        end_date = pd.Timestamp(dt_cache[x_max])
        if pd.isna(start_date) or pd.isna(end_date):
            step = max(1, visible // 50)
            return list(range(x_min, x_max + 1, step))

        time_span, _ = get_time_span(start_date, end_date)
        _GRID = {
            TimeSpan.INTRADAY_MINUTES: timedelta(minutes=1),
            TimeSpan.INTRADAY_HOURS:   timedelta(minutes=30),
            TimeSpan.DAILY:            timedelta(hours=6),
            TimeSpan.WEEKLY:           timedelta(days=1),
            TimeSpan.MONTHLY:          timedelta(days=1),
            TimeSpan.QUARTERLY:        timedelta(days=7),
            TimeSpan.YEARLY:           timedelta(days=30),
            TimeSpan.MULTI_YEAR:       timedelta(days=90),
        }
        grid_interval = _GRID.get(time_span, timedelta(days=1))

        target_ns = dt_cache.astype('int64')
        positions = []
        cur = snap_to_time_boundary(start_date, time_span)
        while cur <= end_date and len(positions) < 100:
            idx = int(np.searchsorted(target_ns, np.int64(pd.Timestamp(cur).value)))
            idx = min(idx, len(dt_cache) - 1)
            if x_min <= idx <= x_max:
                positions.append(idx)
            cur += grid_interval
        return positions


def _precompute_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute 'datetime' column and fast market-hours mask columns.

    Done once (ideally off the UI thread) so that every subsequent
    timeframe switch reuses the cached result instead of re-parsing
    millions of Date/Time strings.
    """
    if 'datetime' in df.columns:
        return df

    if 'Date' not in df.columns:
        return df

    df = df.copy()
    dates = pd.to_datetime(df['Date'], errors='coerce')
    if 'Time' in df.columns:
        td = pd.to_timedelta(df['Time'].astype(str), errors='coerce')
        df['datetime'] = dates + td.fillna(pd.Timedelta(0))
    else:
        df['datetime'] = dates

    # Pre-compute fast integer columns for market-hours filtering so we
    # never need the extremely slow `.dt.time` accessor later.
    dt = df['datetime']
    df['_mkt_minutes'] = dt.dt.hour * 60 + dt.dt.minute
    df['_weekday'] = dt.dt.weekday

    return df


class _RawDataWorker(QThread):
    """Background worker that fetches all raw rows from Postgres."""
    finished = pyqtSignal(str, object)  # (table_name, DataFrame or None)

    def __init__(self, table_name: str, total_records: int, parent=None):
        super().__init__(parent)
        self._table_name = table_name
        self._total = total_records

    def run(self):
        try:
            df = stock_service.get_chart_data_chunk(
                self._table_name, 0, self._total)
            if isinstance(df, pd.DataFrame) and not df.empty:
                for col in list(NUMERIC_COLUMNS & set(df.columns)):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = _precompute_datetime_column(df)
                self.finished.emit(self._table_name, df)
                return
        except Exception as e:
            k2_logger.error(f"Background fetch failed: {e}", "CHART")
        self.finished.emit(self._table_name, None)


class OptimizedPlotDataItem(pg.PlotDataItem):
    """Optimized PlotDataItem with better memory management"""
    def __init__(self, *args, **kwargs):
        kwargs.pop('fillLevel', None)
        kwargs.pop('fillBrush', None)
        kwargs.pop('brush', None)
        super().__init__(*args, **kwargs)
        self.opts['fillLevel'] = None
        self.opts['fillBrush'] = None
        
    def setData(self, *args, **kwargs):
        if 'x' in kwargs and len(kwargs.get('x', [])) > 5000:
            kwargs['downsample'] = 10
            kwargs['downsampleMethod'] = 'peak'
        super().setData(*args, **kwargs)


class DiscreteViewBox(pg.ViewBox):
    """TradingView-style ViewBox:
    - Left-drag pans freely in X and Y (2D pan)
    - Mouse wheel scrolls horizontally through time
    - Ctrl+wheel zooms at cursor position
    - Drag release triggers momentum / inertia
    - Manual Y mode disables auto-fit until double-click on Y axis
    """

    _MOMENTUM_INTERVAL_MS = 16
    _MOMENTUM_FRICTION = 0.92
    _MOMENTUM_MIN_VELOCITY = 0.3
    _SCROLL_FRACTION = 0.05
    _ZOOM_BASE = 1.15
    _VELOCITY_WINDOW = 6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLimits(xMin=0, xMax=1e6, yMin=0, yMax=1e6)
        self.data_x_max = 0
        self.data_y_min = 0.0
        self.data_y_max = 1e6

        self._chart_widget = None
        self._manual_y_mode = False

        self._pan_origin_x = None
        self._pan_origin_y = None
        self._pan_origin_range = None
        self._pan_origin_y_range = None
        self._velocity_samples: deque = deque(maxlen=self._VELOCITY_WINDOW)

        self._momentum_vx = 0.0
        self._momentum_vy = 0.0
        self._momentum_timer = QTimer()
        self._momentum_timer.setInterval(self._MOMENTUM_INTERVAL_MS)
        self._momentum_timer.timeout.connect(self._tick_momentum)

    def set_chart_widget(self, widget):
        self._chart_widget = widget

    # -- helpers --------------------------------------------------------

    _OVERSCROLL = 0.25  # allow 25% of viewport beyond data edges

    def _clamp_x(self, x_min, x_max):
        """Clamp X viewport allowing 1/4 viewport of overscroll past data edges."""
        span = x_max - x_min
        overshoot = span * self._OVERSCROLL
        left_wall = 0.0 - overshoot
        right_wall = (float(self.data_x_max) if self.data_x_max > 0 else span) + overshoot
        if x_min < left_wall:
            x_min = left_wall
            x_max = x_min + span
        if x_max > right_wall:
            x_max = right_wall
            x_min = x_max - span
        return x_min, x_max

    def _clamp_y(self, y_min, y_max):
        """Clamp Y viewport allowing 1/4 viewport of overscroll past data edges."""
        span = y_max - y_min
        overshoot = span * self._OVERSCROLL
        floor = max(0.0, self.data_y_min - overshoot)
        ceiling = self.data_y_max + overshoot
        if y_min < floor:
            y_min = floor
            y_max = y_min + span
        if y_max > ceiling:
            y_max = ceiling
            y_min = max(floor, y_max - span)
        return y_min, y_max

    def _auto_fit_y(self):
        if self._manual_y_mode:
            return
        if self._chart_widget is not None:
            self._chart_widget.auto_scale_y_for_visible_data()

    # -- drag (2D pan + momentum) ---------------------------------------

    def mouseDragEvent(self, ev, axis=None):
        if axis == 1:
            super().mouseDragEvent(ev, axis)
            return

        if ev.button() != Qt.MouseButton.LeftButton:
            return

        ev.accept()

        if ev.isStart():
            self._momentum_timer.stop()
            self._momentum_vx = 0.0
            self._momentum_vy = 0.0
            self._pan_origin_x = ev.pos().x()
            self._pan_origin_y = ev.pos().y()
            self._pan_origin_range = self.viewRange()[0]
            self._pan_origin_y_range = self.viewRange()[1]
            self._velocity_samples.clear()
            self._manual_y_mode = True
            if self._chart_widget is not None:
                self._chart_widget._is_panning = True
                self._chart_widget.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

        elif ev.isFinish():
            self._pan_origin_x = None
            self._pan_origin_y = None
            self._pan_origin_range = None
            self._pan_origin_y_range = None
            if self._chart_widget is not None:
                self._chart_widget._is_panning = False
                self._chart_widget.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            self._start_momentum()

        elif self._pan_origin_x is not None:
            width = self.width()
            height = self.height()
            if width == 0 or height == 0:
                return

            rx = self._pan_origin_range
            x_scale = (rx[1] - rx[0]) / width
            dx = (ev.pos().x() - self._pan_origin_x) * x_scale
            new_x_min, new_x_max = self._clamp_x(rx[0] - dx, rx[1] - dx)
            self.setXRange(new_x_min, new_x_max, padding=0)

            ry = self._pan_origin_y_range
            y_scale = (ry[1] - ry[0]) / height
            dy = (ev.pos().y() - self._pan_origin_y) * y_scale
            new_y_min, new_y_max = self._clamp_y(ry[0] + dy, ry[1] + dy)
            self.setYRange(new_y_min, new_y_max, padding=0)

            self._velocity_samples.append((
                _time.perf_counter(), ev.pos().x(), ev.pos().y()
            ))

    # -- momentum -------------------------------------------------------

    def _start_momentum(self):
        samples = self._velocity_samples
        if len(samples) < 2:
            return
        t0, x0, y0 = samples[0]
        t1, x1, y1 = samples[-1]
        dt = t1 - t0
        if dt <= 0 or dt > 0.25:
            return

        width = self.width()
        height = self.height()
        if width == 0 or height == 0:
            return

        tick = self._MOMENTUM_INTERVAL_MS / 1000.0
        x_range = self.viewRange()[0]
        y_range = self.viewRange()[1]
        x_scale = (x_range[1] - x_range[0]) / width
        y_scale = (y_range[1] - y_range[0]) / height

        self._momentum_vx = ((x0 - x1) / dt) * x_scale * tick
        self._momentum_vy = ((y0 - y1) / dt) * y_scale * tick

        has_vx = abs(self._momentum_vx) > self._MOMENTUM_MIN_VELOCITY
        has_vy = abs(self._momentum_vy) > self._MOMENTUM_MIN_VELOCITY * (y_scale / x_scale if x_scale > 0 else 1.0)
        if has_vx or has_vy:
            self._momentum_timer.start()

    def _tick_momentum(self):
        vx_alive = abs(self._momentum_vx) >= self._MOMENTUM_MIN_VELOCITY
        vy_alive = abs(self._momentum_vy) >= 0.001

        if not vx_alive and not vy_alive:
            self._momentum_timer.stop()
            self._momentum_vx = 0.0
            self._momentum_vy = 0.0
            return

        x_changed = False
        y_changed = False

        if vx_alive:
            cur_x = self.viewRange()[0]
            new_x_min, new_x_max = self._clamp_x(cur_x[0] + self._momentum_vx,
                                                   cur_x[1] + self._momentum_vx)
            if new_x_min != cur_x[0] or new_x_max != cur_x[1]:
                self.setXRange(new_x_min, new_x_max, padding=0)
                x_changed = True
            self._momentum_vx *= self._MOMENTUM_FRICTION

        if vy_alive:
            cur_y = self.viewRange()[1]
            new_y_min, new_y_max = self._clamp_y(cur_y[0] - self._momentum_vy,
                                                   cur_y[1] - self._momentum_vy)
            if new_y_min != cur_y[0] or new_y_max != cur_y[1]:
                self.setYRange(new_y_min, new_y_max, padding=0)
                y_changed = True
            self._momentum_vy *= self._MOMENTUM_FRICTION

        if not x_changed and not y_changed:
            self._momentum_timer.stop()
            self._momentum_vx = 0.0
            self._momentum_vy = 0.0

    # -- wheel (scroll / zoom) ------------------------------------------

    def wheelEvent(self, ev, axis=None):
        self._momentum_timer.stop()
        delta = ev.delta()
        if delta == 0:
            ev.accept()
            return

        mods = ev.modifiers() if hasattr(ev, 'modifiers') else Qt.KeyboardModifier.NoModifier

        if mods & Qt.KeyboardModifier.ControlModifier:
            factor = self._ZOOM_BASE if delta > 0 else (1.0 / self._ZOOM_BASE)
            cursor_x = self.mapToView(ev.pos()).x()
            x_lo, x_hi = self.viewRange()[0]
            span = x_hi - x_lo
            new_span = span / factor
            if new_span < 3:
                ev.accept()
                return
            frac = (cursor_x - x_lo) / span if span > 0 else 0.5
            new_min = cursor_x - frac * new_span
            new_max = cursor_x + (1.0 - frac) * new_span
            new_min, new_max = self._clamp_x(new_min, new_max)
            self.setXRange(new_min, new_max, padding=0)
        else:
            x_lo, x_hi = self.viewRange()[0]
            step = (x_hi - x_lo) * self._SCROLL_FRACTION
            offset = -step if delta > 0 else step
            new_min, new_max = self._clamp_x(x_lo + offset, x_hi + offset)
            self.setXRange(new_min, new_max, padding=0)

        if not self._manual_y_mode:
            if self._chart_widget is not None and hasattr(self._chart_widget, '_y_fit_timer'):
                self._chart_widget._y_fit_timer.start()
        ev.accept()


class EmbeddedAxis(pg.GraphicsWidget):
    """Custom axis widget for chart with enhanced time display"""

    def __init__(self, orientation='left', parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.labels = []
        self.sublabels = []  # For multi-level time display
        self.parent_plot = parent

        self._setup_appearance()
        self._setup_dimensions()

        self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(1000000)

    def _setup_appearance(self):
        self.font = QFont('Arial', 9)
        self.small_font = QFont('Arial', 8)
        self.text_color = QColor('#999999')
        self.subtext_color = QColor('#666666')
        self.bg_color = QColor(10, 10, 10, 230)
        self.border_color = QColor(42, 42, 42)
        self.pen = QPen(self.border_color, 1)
        self.text_pen = QPen(self.text_color)
        self.subtext_pen = QPen(self.subtext_color)

    def _setup_dimensions(self):
        if self.orientation == 'left':
            self._width = 70
            self._height = 100
        else:
            self._width = 100
            self._height = 50  # Height for stacked date/time labels

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
                # Draw main labels - stacked: date on top, time on bottom
                for label, pos in self.labels:
                    if '\n' in label:
                        parts = label.split('\n')
                        # Date on top (smaller, subdued)
                        painter.setFont(self.small_font)
                        painter.setPen(self.subtext_pen)
                        text_rect = QRectF(pos - 45, 4, 90, 16)
                        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, parts[0])
                        # Time on bottom (normal, brighter)
                        painter.setFont(self.font)
                        painter.setPen(self.text_pen)
                        text_rect = QRectF(pos - 45, 22, 90, 20)
                        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, parts[1])
                    else:
                        text_rect = QRectF(pos - 45, 5, 90, self._height - 10)
                        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)

    def setLabels(self, labels):
        if labels != self.labels:
            self.labels = labels
            self.update()


class ChartWidget(QWidget):
    """Enhanced Trading-view style chart widget with intelligent time axis"""
    
    # Signals
    drawing_added = pyqtSignal(dict)
    timeframe_changed = pyqtSignal(str)
    data_loading = pyqtSignal()
    data_loaded = pyqtSignal()
    viewport_changed = pyqtSignal(int, int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_data_structures()
        self.init_ui()
        self._setup_timers()
        self.setup_chart_style()
        
    def _init_data_structures(self):
        """Initialize data structures with optimization"""
        # Core data
        self.data = None
        self.original_data = None
        self.date_column = None
        self.x_values = None
        self._dt_cache = None
        self.current_timeframe = '1D'
        self.min_visible_points = 3
        self.min_granularity = None
        self.last_default_x_range = None
        
        # Time axis manager
        self.time_axis_manager = TimeAxisManager()
        
        # Collections
        self.active_lines = {}
        self.indicator_panes = {}
        self.indicator_overlays = {}
        self._indicator_full_data = {}
        self._forecast_lines = {}
        self.drawings = []
        
        # UI element references
        self.ohlc_buttons = {}
        self.timeframe_buttons = {}
        self.tool_buttons = {}
        
        # Drawing state
        self.drawing_mode = None
        self.drawing_start_point = None
        self.temp_drawing = None
        
        # Axis interaction state
        self.dragging_y_axis = False
        self.dragging_x_axis = False
        self.drag_start_pos = None
        self.drag_start_y_range = None
        self.drag_start_x_range = None
        self.drag_anchor_y = None
        self.axis_hover = None
        self._is_panning = False
        
        # Grid lines pool (reusable)
        self._grid_pool = {'v': [], 'h': []}
        self._active_grids = {'v': 0, 'h': 0}
        
        # Caching
        self._viewport_cache = None
        self._label_cache = {}
        self._format_cache = {}
        self._method_cache = {}
        self._model_cache: dict = {}
        self._raw_ready: set = set()
        self._bg_worker: Optional[_RawDataWorker] = None
        
        # DB context
        self.current_table_name = None
        self.total_records = 0
        self._global_start_index = 0
        self.is_fetching = False
        
        # Debounce timers dict
        self._debounce_timers = {}
        
    def _setup_timers(self):
        """Setup optimized timers with single timer reuse"""
        # Axis update timer (100ms is fast enough for text labels)
        self.axis_update_timer = QTimer()
        self.axis_update_timer.setSingleShot(True)
        self.axis_update_timer.setInterval(100)
        self.axis_update_timer.timeout.connect(self.update_axis_labels_and_grid)
        
        # Axis geometry update timer
        self.range_update_timer = QTimer()
        self.range_update_timer.setSingleShot(True)
        self.range_update_timer.setInterval(16)
        self.range_update_timer.timeout.connect(self.update_axis_geometry)
        
        # Viewport signal debounce timer
        self._viewport_signal_timer = QTimer()
        self._viewport_signal_timer.setSingleShot(True)
        self._viewport_signal_timer.setInterval(60)
        self._viewport_signal_timer.timeout.connect(self._emit_viewport_changed)

        # Y auto-fit debounce timer (used by wheel events)
        self._y_fit_timer = QTimer()
        self._y_fit_timer.setSingleShot(True)
        self._y_fit_timer.setInterval(50)
        self._y_fit_timer.timeout.connect(self.auto_scale_y_for_visible_data)
        
    def init_ui(self):
        """Initialize UI with optimized layout"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Drawing toolbar
        self.drawing_toolbar = self._create_drawing_toolbar()
        main_layout.addWidget(self.drawing_toolbar)
        
        # Chart container
        chart_widget = self._create_chart_container()
        main_layout.addWidget(chart_widget)
        
    def _create_chart_container(self):
        """Create optimized chart container"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # OHLC toggles
        self.ohlc_bar = self._create_ohlc_toggles()
        layout.addWidget(self.ohlc_bar)
        
        # Timeframe selector
        self.timeframe_bar = self._create_timeframe_selector()
        layout.addWidget(self.timeframe_bar)
        
        # Main chart
        self.chart_container = pg.GraphicsLayoutWidget()
        self.chart_container.setBackground('#0a0a0a')
        self.chart_container.ci.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot with custom ViewBox
        vb = DiscreteViewBox()
        self.main_plot = self.chart_container.addPlot(row=0, col=0, viewBox=vb)
        self._setup_main_plot()
        
        layout.addWidget(self.chart_container, stretch=3)
        
        # Indicator container (QWidget with layout for separate pane indicators)
        # Hidden by default - only shown when oscillator indicators are active
        self.indicator_widget = QWidget()
        self.indicator_container = QVBoxLayout(self.indicator_widget)
        self.indicator_container.setContentsMargins(0, 0, 0, 0)
        self.indicator_container.setSpacing(2)
        self.indicator_widget.hide()  # Start hidden - show only when panes are added
        layout.addWidget(self.indicator_widget, stretch=1)
        
        return container
        
    def _setup_main_plot(self):
        """Setup main plot with optimizations"""
        self.main_plot.hideAxis('left')
        self.main_plot.hideAxis('bottom')
        self.main_plot.showGrid(x=False, y=False)
        vb = self.main_plot.getViewBox()
        vb.setMouseEnabled(x=True, y=False)
        vb.disableAutoRange()
        if isinstance(vb, DiscreteViewBox):
            vb.set_chart_widget(self)
        
        # Create embedded axes
        self._create_embedded_axes()
        
        # Initialize grid pool
        self._init_grid_pool()
        
        # Add crosshair
        self._add_crosshair()
        
        # Connect events
        self._connect_plot_events()
        
    def _create_embedded_axes(self):
        """Create embedded axes"""
        self.y_axis = EmbeddedAxis('left', self.main_plot)
        self.main_plot.scene().addItem(self.y_axis)

        self.x_axis = EmbeddedAxis('bottom', self.main_plot)
        self.main_plot.scene().addItem(self.x_axis)

        QTimer.singleShot(0, self.update_axis_geometry)
        
    def _init_grid_pool(self):
        """Initialize reusable grid line pool"""
        # Create pool of vertical lines
        for _ in range(200):
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen('#1a1a1a', width=1))
            line.setVisible(False)
            self.main_plot.addItem(line, ignoreBounds=True)
            self._grid_pool['v'].append(line)
            
        # Create pool of horizontal lines  
        for _ in range(15):
            line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#1a1a1a', width=1))
            line.setVisible(False)
            self.main_plot.addItem(line, ignoreBounds=True)
            self._grid_pool['h'].append(line)
            
    def _add_crosshair(self):
        """Add crosshair: V-line snaps to data; shows Date, Time, OHLC at intersections.
        H-line is fluid (visual aid only)."""
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#666', width=1))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#666', width=1))
        self.main_plot.addItem(self.vLine, ignoreBounds=True)
        self.main_plot.addItem(self.hLine, ignoreBounds=True)
        
        font = QFont('Arial', 10)
        self.crosshair_date_label = pg.TextItem(color='#fff', anchor=(0, 0))
        self.crosshair_date_label.setFont(font)
        self.main_plot.addItem(self.crosshair_date_label)
        
        self.crosshair_time_label = pg.TextItem(color='#fff', anchor=(0, 0))
        self.crosshair_time_label.setFont(font)
        self.main_plot.addItem(self.crosshair_time_label)
        
        self.crosshair_ohlc_labels = {}
        for col in ['High', 'Open', 'Close', 'Low']:
            color = OHLC_COLORS.get(col, '#fff')
            label = pg.TextItem(color=color, anchor=(0, 0.5))
            label.setFont(font)
            self.main_plot.addItem(label)
            self.crosshair_ohlc_labels[col] = label
        
        self.proxy = pg.SignalProxy(
            self.main_plot.scene().sigMouseMoved,
            rateLimit=33,
            slot=self.update_crosshair
        )
        
    def _connect_plot_events(self):
        """Connect plot events"""
        self.main_plot.scene().sigMouseClicked.connect(self.on_mouse_clicked)
        self.main_plot.scene().sigMouseMoved.connect(self.on_mouse_moved)

        # Range change handlers
        vb = self.main_plot.getViewBox()
        vb.sigRangeChanged.connect(lambda: self.axis_update_timer.start())
        vb.sigRangeChanged.connect(lambda: self._viewport_signal_timer.start())

        try:
            vb.sigResized.connect(lambda: self.range_update_timer.start())
        except AttributeError:
            pass

        # Setup event filter
        self.chart_container.viewport().installEventFilter(self)
        self.chart_container.viewport().setMouseTracking(True)
        
    def _create_drawing_toolbar(self):
        """Create drawing toolbar with full implementation"""
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
        
        clear_btn = QPushButton('×')
        clear_btn.setToolTip('Clear All Drawings')
        clear_btn.setFixedSize(32, 32)
        clear_btn.clicked.connect(self.clear_all_drawings)
        layout.addWidget(clear_btn)
        
        return toolbar
        
    def _create_ohlc_toggles(self):
        """Create OHLC toggles with full implementation"""
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
        """Create timeframe selector with full implementation"""
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
        
        self.timeframe_button_group = QButtonGroup(container)
        self.timeframe_button_group.setExclusive(True)
        
        for tf in TIMEFRAME_CONFIG.keys():
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
        
    def update_axis_labels_and_grid(self):
        """Update axis labels and grid lines with intelligent time handling"""
        if not hasattr(self, 'y_axis') or not hasattr(self, 'x_axis'):
            return

        vb = self.main_plot.getViewBox()
        x_range, y_range = vb.viewRange()

        self._update_y_axis_and_grid(y_range)
        self._update_x_axis_and_grid_intelligent(x_range)
        
    def _update_y_axis_and_grid(self, y_range):
        """Update Y-axis labels and horizontal grid lines"""
        y_labels = []
        y_min, y_max = max(0, y_range[0]), y_range[1]

        if y_max - y_min > 0:
            price_range = y_max - y_min
            interval = self._get_price_interval(price_range)

            first_line = math.ceil(y_min / interval) * interval
            last_line = math.floor(y_max / interval) * interval

            prices = []
            current = first_line
            while current <= last_line:
                prices.append(current)
                current += interval

            if len(prices) > 15:
                prices = prices[::2]

            axis_height = self.y_axis._height

            # Hide all then show needed
            for i in range(self._active_grids['h']):
                if i < len(self._grid_pool['h']):
                    self._grid_pool['h'][i].setVisible(False)

            for i, price in enumerate(prices):
                pos = axis_height * (1 - (price - y_min) / (y_max - y_min))

                if price < 10:
                    label = f"${price:.3f}"
                elif price < 1000:
                    label = f"${price:.2f}"
                else:
                    label = f"${price:,.2f}"

                y_labels.append((label, pos))

                if i < len(self._grid_pool['h']):
                    self._grid_pool['h'][i].setPos(price)
                    self._grid_pool['h'][i].setVisible(True)
            
            # Cap active grids to pool size
            self._active_grids['h'] = min(len(prices), len(self._grid_pool['h']))

        self.y_axis.setLabels(y_labels)
        
    def _update_x_axis_and_grid_intelligent(self, x_range):
        """Update X-axis labels and vertical grid using cached datetime array."""
        for i in range(self._active_grids['v']):
            if i < len(self._grid_pool['v']):
                self._grid_pool['v'][i].setVisible(False)

        if self._dt_cache is None or len(self._dt_cache) == 0:
            self.x_axis.setLabels([])
            return

        axis_width = self.x_axis._width
        labels = self.time_axis_manager.calculate_time_labels(
            self._dt_cache, x_range, axis_width)
        grid_positions = self.time_axis_manager.calculate_grid_positions(
            self._dt_cache, x_range)

        for i, pos in enumerate(grid_positions[:len(self._grid_pool['v'])]):
            self._grid_pool['v'][i].setPos(pos)
            self._grid_pool['v'][i].setVisible(True)
        self._active_grids['v'] = min(len(grid_positions), len(self._grid_pool['v']))

        self.x_axis.setLabels(labels)
        
    def update_axis_geometry(self):
        """Update axis geometry"""
        if not hasattr(self, 'y_axis') or not hasattr(self, 'x_axis'):
            return

        vb = self.main_plot.getViewBox()
        vb_rect = vb.sceneBoundingRect()

        y_axis_width = 70
        x_axis_height = 50  # Height for stacked date/time labels

        self.y_axis.setSize(y_axis_width, vb_rect.height())
        self.x_axis.setSize(vb_rect.width(), x_axis_height)

        self.y_axis.setPos(vb_rect.right() - y_axis_width, vb_rect.top())
        self.x_axis.setPos(vb_rect.left(), vb_rect.bottom() - x_axis_height)
        
    def _get_price_interval(self, price_range):
        """Get appropriate price interval"""
        if price_range <= 0:
            return 0.1
            
        magnitude = 10 ** math.floor(math.log10(price_range))
        normalized = price_range / magnitude
        
        if price_range > 100:
            return 10 if normalized < 2 else 20 if normalized < 4 else 25 if normalized < 5 else 50
        elif price_range > 20:
            return 2 if normalized < 4 else 5 if normalized < 8 else 10
        elif price_range > 5:
            return 0.5 if normalized < 10 else 1 if normalized < 20 else 2
        elif price_range > 1:
            return 0.1 if normalized < 2 else 0.2 if normalized < 5 else 0.5
        else:
            return 0.1
            
    def load_data_from_table(self, table_name: str, total_records: Optional[int] = None,
                            metadata: Optional[Dict] = None):
        """Two-phase load:
        Phase 1 — server-side daily bars (fast, ~200ms).  Chart is usable.
        Phase 2 — background thread fetches all raw rows for intraday."""
        if self.is_fetching:
            return

        self.clear_all()
        self.data = None
        self.original_data = None
        self._dt_cache = None
        vb = self.main_plot.getViewBox()
        if isinstance(vb, DiscreteViewBox):
            vb._manual_y_mode = False
        self.current_table_name = table_name
        self.total_records = total_records or 0

        if self.total_records <= 0:
            k2_logger.warning("No records to load for chart", "CHART")
            return

        # --- Phase 1: daily bars (instant) ---
        if table_name in self._raw_ready:
            self.original_data = self._model_cache.get(table_name)
        else:
            daily_df = None
            if stock_service:
                daily_df = stock_service.get_daily_bars(table_name)
            if daily_df is None or daily_df.empty:
                return
            for col in list(NUMERIC_COLUMNS & set(daily_df.columns)):
                daily_df[col] = pd.to_numeric(daily_df[col], errors='coerce')
            self.original_data = daily_df
            self._model_cache[table_name] = daily_df

        self._detect_granularity_optimized()

        self.current_timeframe = '1D'
        self._update_timeframe_button_checked('1D')

        self._process_timeframe_optimized()
        self._global_start_index = 0
        self._update_viewport_limits()
        self._display_ohlc_optimized()
        self._show_last_n_bars(DEFAULT_VISIBLE_BARS)
        self.update_axis_geometry()
        QTimer.singleShot(0, self.update_axis_labels_and_grid)
        k2_logger.info(
            f"Loaded {len(self.data) if self.data is not None else 0} "
            f"{self.current_timeframe} bars for {table_name}", "CHART")

        # --- Phase 2: background fetch of raw rows for intraday ---
        if table_name not in self._raw_ready:
            self._start_background_fetch(table_name)

    def _start_background_fetch(self, table_name: str):
        """Kick off a background thread to fetch all raw rows."""
        if self._bg_worker is not None and self._bg_worker.isRunning():
            self._bg_worker.finished.disconnect()
            self._bg_worker.quit()
            self._bg_worker.wait(2000)

        self._bg_worker = _RawDataWorker(table_name, self.total_records, self)
        self._bg_worker.finished.connect(self._on_raw_data_ready)
        self._bg_worker.start()
        k2_logger.info(f"Background fetch started for {table_name}", "CHART")

    def _on_raw_data_ready(self, table_name: str, df):
        """Called on main thread when background fetch completes."""
        if df is not None and not df.empty:
            self._model_cache[table_name] = df
            self._raw_ready.add(table_name)
            if self.current_table_name == table_name:
                self.original_data = df
            k2_logger.info(
                f"Background fetch done: {len(df)} raw rows for {table_name}",
                "CHART")
        else:
            k2_logger.warning(
                f"Background fetch returned empty for {table_name}", "CHART")
        
    def _update_viewport_limits(self):
        """Update ViewBox limits based on actual data - FIXED VERSION"""
        if self.data is None or len(self.data) == 0:
            return
            
        vb = self.main_plot.getViewBox()
        
        # Calculate Y limits from data using fast numpy (already numeric)
        y_min = float('inf')
        y_max = float('-inf')
        
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in self.data.columns:
                arr = np.asarray(self.data[col].values, dtype=np.float64)
                finite = arr[np.isfinite(arr)]
                if len(finite) > 0:
                    y_min = min(y_min, finite.min())
                    y_max = max(y_max, finite.max())
        
        # Ensure valid Y range
        if y_min == float('inf') or y_max == float('-inf'):
            y_min, y_max = 0, 100
        
        # Set limits with appropriate padding
        x_max = len(self.data) - 1
        forecast_extra = 1500
        x_max_with_forecast = x_max + forecast_extra
        y_min = max(0, y_min * 0.9)  # 10% padding below (but never negative)
        y_max = y_max * 1.2  # 20% padding above (not 2x!)
        
        # Set pyqtgraph hard limits generously; our _clamp_x/_clamp_y do the real work
        overscroll_x = x_max_with_forecast * 0.25
        overscroll_y = (y_max - y_min) * 0.5
        vb.setLimits(
            xMin=-overscroll_x,
            xMax=x_max_with_forecast + overscroll_x,
            yMin=max(0, y_min - overscroll_y),
            yMax=y_max + overscroll_y
        )
        
        # Store data boundaries in ViewBox for clamping (includes forecast space)
        if isinstance(vb, DiscreteViewBox):
            vb.data_x_max = x_max_with_forecast
            vb.data_y_min = y_min
            vb.data_y_max = y_max
            
    def _detect_granularity_optimized(self):
        """Detect native data granularity from a tiny DB sample (10 rows)."""
        self.min_granularity = '1D'
        if not self.current_table_name or not stock_service:
            self._update_timeframe_buttons('1D')
            return

        try:
            sample_df = stock_service.get_chart_data_chunk(
                self.current_table_name, 0, 100)
            if sample_df is None or sample_df.empty or 'Time' not in sample_df.columns:
                self._update_timeframe_buttons('1D')
                return

            dates = pd.to_datetime(sample_df['Date'], errors='coerce')
            td = pd.to_timedelta(sample_df['Time'].astype(str), errors='coerce')
            dt = dates + td.fillna(pd.Timedelta(0))

            diffs = dt.diff().dt.total_seconds() / 60
            diffs = diffs[diffs > 0]

            if len(diffs) > 0 and not diffs.isna().all():
                min_interval = diffs[~diffs.isna()].min()
                for tf, config in TIMEFRAME_CONFIG.items():
                    if min_interval <= config['interval_minutes']:
                        self.min_granularity = tf
                        break

            self._update_timeframe_buttons(self.min_granularity)

        except Exception as e:
            k2_logger.error(f"Granularity detection failed: {e}", "CHART")
            self._update_timeframe_buttons('1D')
            
    def _update_timeframe_buttons(self, min_tf):
        """Disable timeframes finer than data granularity (always enabled otherwise)."""
        timeframes = list(TIMEFRAME_CONFIG.keys())
        min_idx = timeframes.index(min_tf) if min_tf in timeframes else 4
        for i, tf in enumerate(timeframes):
            if tf in self.timeframe_buttons:
                self.timeframe_buttons[tf].setEnabled(i >= min_idx)

    def _update_timeframe_button_checked(self, tf: str):
        """Visually check the given timeframe button (uncheck others)."""
        for key, btn in self.timeframe_buttons.items():
            btn.setChecked(key == tf)
                
    def _process_timeframe_optimized(self):
        """Build self.data from original_data: create datetime, filter market
        hours (intraday only), resample if needed.  Numerics are already
        coerced at ingestion so we skip redundant conversions."""
        df = self.original_data

        # --- 1. Ensure datetime column exists (fast: usually pre-computed) ---
        if 'datetime' not in df.columns:
            df = _precompute_datetime_column(df)
            self.original_data = df

        if 'datetime' not in df.columns:
            self.data = df
            self.date_column = None
            self.x_values = np.arange(len(df), dtype=np.float32)
            self._dt_cache = None
            return

        self.date_column = 'datetime'

        # --- 2. Filter to market hours for intraday timeframes ---
        #     Uses pre-computed integer columns (_mkt_minutes, _weekday)
        #     instead of the extremely slow .dt.time accessor.
        if 'Time' in df.columns and self.current_timeframe in INTRADAY_TIMEFRAMES:
            try:
                MKT_OPEN_MIN = 9 * 60 + 30   # 09:30
                MKT_CLOSE_MIN = 16 * 60       # 16:00
                weekday = df['_weekday'] if '_weekday' in df.columns else df['datetime'].dt.weekday
                minutes = df['_mkt_minutes'] if '_mkt_minutes' in df.columns else (df['datetime'].dt.hour * 60 + df['datetime'].dt.minute)
                mask = (weekday < 5) & (minutes >= MKT_OPEN_MIN) & (minutes < MKT_CLOSE_MIN)
                df = df[mask].reset_index(drop=True)
            except Exception:
                pass

        # --- 3. Resample if timeframe is coarser than native ---
        if self.min_granularity and self.current_timeframe != self.min_granularity:
            df = self._resample_df(df)

        # --- 4. Store and build index / cache ---
        self.data = df.reset_index(drop=True)
        self.x_values = np.arange(len(self.data), dtype=np.float32)
        if self.date_column in self.data.columns:
            self._dt_cache = self.data[self.date_column].values.astype('datetime64[ns]')
            self._extend_dt_cache()
        else:
            self._dt_cache = None
        
    def _extend_dt_cache(self, extra_points: int = 1500):
        """Extrapolate future dates so the x-axis shows labels beyond the last bar."""
        cache = self._dt_cache
        if cache is None or len(cache) < 2:
            return
        n = min(50, len(cache) - 1)
        total_span = cache[-1] - cache[-1 - n]
        avg_delta = total_span / n
        if avg_delta <= np.timedelta64(0):
            return
        base = cache[-1]
        extension = np.array(
            [base + avg_delta * (i + 1) for i in range(extra_points)],
            dtype='datetime64[ns]',
        )
        self._dt_cache = np.concatenate([cache, extension])

    def _resample_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate *df* into coarser bars using dt.floor() for grouping.
        Preserves real trading timestamps (last in each bar) so the X-axis
        has no calendar gaps."""
        config = TIMEFRAME_CONFIG.get(self.current_timeframe)
        if not config or 'datetime' not in df.columns:
            return df

        rule = config['rule']
        try:
            bar_key = df['datetime'].dt.floor(rule)

            agg = {'datetime': 'last'}
            for col, func in [('Open', 'first'), ('High', 'max'),
                               ('Low', 'min'), ('Close', 'last'),
                               ('Volume', 'sum'), ('VWAP', 'last')]:
                if col in df.columns:
                    agg[col] = func

            out = df.groupby(bar_key, sort=True).agg(agg)
            out = out.dropna(subset=['datetime']).reset_index(drop=True)
            k2_logger.info(
                f"Resampled to {self.current_timeframe}: "
                f"{len(df)} → {len(out)} bars", "CHART")
            return out

        except Exception as e:
            k2_logger.error(f"Resample failed ({rule}): {e}", "CHART")
            return df
            
    # --- Viewport-windowed OHLC rendering ---
    # Only push the visible slice (+ buffer) to pyqtgraph so we never ask
    # it to render hundreds of thousands of points.

    _OHLC_VIEW_BUFFER = 500  # extra bars each side of viewport

    def _display_ohlc_optimized(self):
        """Create empty PlotDataItems, then fill them with the visible window."""
        for line in list(self.active_lines.values()):
            if line.scene():
                self.main_plot.removeItem(line)
        self.active_lines.clear()

        for col in ['Open', 'High', 'Low', 'Close']:
            if col in self.data.columns and col in self.ohlc_buttons:
                if self.ohlc_buttons[col].isChecked():
                    self._create_ohlc_plot_item(col)

        self._refresh_visible_ohlc()

    def _create_ohlc_plot_item(self, column_name):
        """Create an empty PlotDataItem and add it to the scene."""
        if column_name in self.active_lines:
            return
        if column_name not in self.data.columns:
            return

        color = OHLC_COLORS.get(column_name, '#ffffff')
        plot_item = pg.PlotDataItem(
            pen=pg.mkPen(color=color, width=2),
            connect='finite'
        )
        self.main_plot.addItem(plot_item)
        self.active_lines[column_name] = plot_item

    def _visible_ohlc_range(self):
        """Return (lo, hi) index range that should be rendered."""
        if self.data is None or len(self.data) == 0:
            return 0, 0
        vb = self.main_plot.getViewBox()
        x_lo, x_hi = vb.viewRange()[0]
        buf = self._OHLC_VIEW_BUFFER
        lo = int(max(0, x_lo - buf))
        hi = int(min(len(self.data), x_hi + buf + 1))
        return lo, hi

    def _refresh_visible_ohlc(self):
        """Push only the visible window of data into each OHLC PlotDataItem."""
        if self.data is None or len(self.data) == 0:
            return

        lo, hi = self._visible_ohlc_range()
        if lo >= hi:
            return

        x_slice = self.x_values[lo:hi]
        for col, item in list(self.active_lines.items()):
            if col not in self.data.columns:
                continue
            y = np.asarray(self.data[col].values[lo:hi], dtype=np.float32)
            y[np.isinf(y)] = np.nan
            item.setData(x=x_slice, y=y)

    def _refresh_visible_indicators(self):
        """Push only the visible window of data into each indicator overlay PlotDataItem."""
        if self.x_values is None or len(self.x_values) == 0:
            return
        if not self._indicator_full_data:
            return

        lo, hi = self._visible_ohlc_range()
        if lo >= hi:
            return

        x_slice = self.x_values[lo:hi]
        for name, item in list(self.indicator_overlays.items()):
            full_y = self._indicator_full_data.get(name)
            if full_y is None:
                continue
            item.setData(x=x_slice, y=full_y[lo:hi])

    def _add_ohlc_line_optimized(self, column_name):
        """Add a single OHLC line and immediately populate it with visible data."""
        self._create_ohlc_plot_item(column_name)
        self._refresh_visible_ohlc()
        
    def set_default_view(self):
        """Show the last DEFAULT_VISIBLE_BARS bars."""
        vb = self.main_plot.getViewBox()
        if isinstance(vb, DiscreteViewBox):
            vb._manual_y_mode = False
        self._show_last_n_bars(DEFAULT_VISIBLE_BARS)

    def _show_last_n_bars(self, n: int):
        """Position the viewport to show the last *n* bars (or all bars if
        fewer than *n* exist)."""
        if self.data is None or len(self.data) == 0:
            return
        total = len(self.data)
        visible = min(n, total)
        end_idx = total - 1
        start_idx = max(0, end_idx - visible)
        buf = max(2, visible // 20)
        self.main_plot.setXRange(start_idx, end_idx + buf, padding=0)
        self._auto_scale_y_range(start_idx, end_idx)
        self.last_default_x_range = (start_idx, end_idx + buf)
        
    def _auto_scale_y_range(self, x_min, x_max):
        """Auto-scale Y range with numpy optimization"""
        if self.data is None or len(self.data) == 0:
            return
            
        x_min = int(max(0, x_min))
        x_max = int(min(len(self.data) - 1, x_max))
        
        if x_min >= len(self.data) or x_min > x_max:
            return
            
        # Get visible data slice
        visible_data = self.data.iloc[x_min:x_max + 1]
        active_cols = [col for col in ['Open', 'High', 'Low', 'Close']
                      if col in visible_data.columns and col in self.active_lines]
        
        if not active_cols:
            return
            
        # Vectorized min/max calculation
        data_array = visible_data[active_cols].values
        
        # Convert to numeric and filter
        data_array = pd.to_numeric(data_array.flatten(), errors='coerce')
        data_array = data_array[~np.isnan(data_array)]
        data_array = data_array[~np.isinf(data_array)]
        
        if len(data_array) > 0:
            y_min = np.min(data_array)
            y_max = np.max(data_array)
            
            padding = (y_max - y_min) * 0.1
            y_min = max(0, y_min - padding)
            y_max = y_max + padding
            
            self.main_plot.setYRange(y_min, y_max, padding=0)
            
    def update_crosshair(self, evt):
        """Crosshair: V-line snaps to data; Date/Time at top; OHLC labels at intersections.
        H-line stays fluid (visual aid only)."""
        if self._is_panning:
            return
        pos = evt[0]
        if not self.main_plot.sceneBoundingRect().contains(pos):
            self._hide_crosshair_labels()
            return
            
        mousePoint = self.main_plot.getViewBox().mapSceneToView(pos)
        
        x_raw = mousePoint.x()
        x_snapped = int(round(x_raw))
        x_snapped = max(0, min(x_snapped, len(self.data) - 1) if self.data is not None else x_snapped)
        
        self.vLine.setPos(x_snapped)
        self.hLine.setPos(mousePoint.y())
        
        if self.data is None or not self.date_column or len(self.data) == 0 or not (0 <= x_snapped < len(self.data)):
            self._hide_crosshair_labels()
            return
            
        vb = self.main_plot.getViewBox()
        x_range, (y_min, y_max) = vb.viewRange()
        
        date_val = self.data.iloc[x_snapped][self.date_column]
        if pd.isna(date_val):
            self._hide_crosshair_labels()
            return
            
        date_str = safe_strftime(date_val, '%d-%b-%y', '')
        time_str = safe_strftime(date_val, '%H:%M:%S', '') if self.current_timeframe in INTRADAY_TIMEFRAMES else ''
        
        x_label = x_snapped + (x_range[1] - x_range[0]) * 0.008 if (x_range[1] - x_range[0]) > 0 else x_snapped + 0.5
        
        self.crosshair_date_label.setText(date_str or '—')
        self.crosshair_date_label.setPos(x_label, y_max)
        self.crosshair_date_label.setVisible(True)
        
        if time_str:
            self.crosshair_time_label.setText(time_str)
            self.crosshair_time_label.setPos(x_label, y_max - (y_max - y_min) * 0.03)
            self.crosshair_time_label.setVisible(True)
        else:
            self.crosshair_time_label.setVisible(False)
        
        for col in ['High', 'Open', 'Close', 'Low']:
            lbl = self.crosshair_ohlc_labels[col]
            if col not in self.data.columns or col not in self.active_lines:
                lbl.setVisible(False)
                continue
            val_raw = self.data.iloc[x_snapped][col]
            val_num = pd.to_numeric(val_raw, errors='coerce')
            if pd.isna(val_num) or not np.isfinite(val_num):
                lbl.setVisible(False)
                continue
            price = float(val_num)
            short = col[0]
            price_str = f"${price:.3f}" if price < 10 else (f"${price:.2f}" if price < 1000 else f"${price:,.2f}")
            lbl.setText(f"{short} - {price_str}")
            lbl.setPos(x_label, price)
            lbl.setVisible(True)
    
    def _hide_crosshair_labels(self):
        """Hide crosshair labels when mouse outside chart or no data."""
        for attr in ['crosshair_date_label', 'crosshair_time_label']:
            item = getattr(self, attr, None)
            if item is not None:
                item.setVisible(False)
        for lbl in getattr(self, 'crosshair_ohlc_labels', {}).values():
            if lbl is not None:
                lbl.setVisible(False)
            
    def _emit_viewport_changed(self):
        """Emit viewport changed signal and refresh visible data windows."""
        if self.data is None:
            return
        self._refresh_visible_ohlc()
        self._refresh_visible_indicators()
        vb = self.main_plot.getViewBox()
        x_min, x_max = vb.viewRange()[0]
        start = int(max(0, round(x_min)))
        end = int(min(len(self.data), round(x_max)))
        self.viewport_changed.emit(start, end, len(self.data))
        
    
    def _reprocess_for_timeframe(self, timeframe: str):
        """Re-resample already-loaded original_data for a new timeframe.
        No DB fetch — data is already in memory."""
        if self.original_data is None or self.original_data.empty:
            return
        self.current_timeframe = timeframe
        self._process_timeframe_optimized()
        k2_logger.info(
            f"Resampled to {len(self.data)} {timeframe} bars", "CHART")

    # Drawing methods
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
        """Handle mouse click for drawing"""
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
                self._create_single_click_drawing(mousePoint)
            else:
                if self.drawing_start_point is None:
                    self.drawing_start_point = mousePoint
                    self._create_temp_drawing(mousePoint)
                else:
                    self._complete_two_click_drawing(mousePoint)
                    
    def on_mouse_moved(self, pos):
        """Handle mouse movement for drawing"""
        if self.drawing_mode and self.drawing_start_point and self.temp_drawing:
            if self.main_plot.sceneBoundingRect().contains(pos):
                vb = self.main_plot.getViewBox()
                mousePoint = vb.mapSceneToView(pos)
                self._update_temp_drawing(mousePoint)
                
    def _create_single_click_drawing(self, point):
        """Create single-click drawing"""
        config = DrawingConfig()
        
        if self.drawing_mode == 'horizontal':
            _, _, color = config.horizontal
            line = pg.InfiniteLine(
                pos=point.y(),
                angle=0,
                pen=pg.mkPen(color, width=2),
                movable=True
            )
            self.main_plot.addItem(line)
            self.drawings.append(('horizontal', line))

        elif self.drawing_mode == 'vertical':
            _, _, color = config.vertical
            line = pg.InfiniteLine(
                pos=point.x(),
                angle=90,
                pen=pg.mkPen(color, width=2),
                movable=True
            )
            self.main_plot.addItem(line)
            self.drawings.append(('vertical', line))
            
    def _create_temp_drawing(self, start_point):
        """Create temporary drawing"""
        if self.drawing_mode in ['trend', 'ray', 'extended']:
            self.temp_drawing = pg.PlotDataItem(
                [start_point.x(), start_point.x()],
                [start_point.y(), start_point.y()],
                pen=pg.mkPen('#ffffff', width=1, style=Qt.PenStyle.DashLine)
            )
            self.main_plot.addItem(self.temp_drawing)
            
    def _update_temp_drawing(self, end_point):
        """Update temporary drawing"""
        if self.temp_drawing and self.drawing_start_point:
            self.temp_drawing.setData(
                [self.drawing_start_point.x(), end_point.x()],
                [self.drawing_start_point.y(), end_point.y()]
            )
            
    def _complete_two_click_drawing(self, end_point):
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
        
    # OHLC and UI methods
    def toggle_ohlc_line(self, column_name, visible=None):
        """Toggle OHLC line visibility"""
        if visible is None:
            visible = self.ohlc_buttons[column_name].isChecked()
            
        if visible and column_name not in self.active_lines:
            self._add_ohlc_line_optimized(column_name)
        elif not visible and column_name in self.active_lines:
            item = self.active_lines.pop(column_name)
            self.main_plot.removeItem(item)
            
    def change_timeframe(self, timeframe):
        """Switch timeframe: resample in-memory data, display, fit viewport.
        If intraday is requested but raw data hasn't arrived yet, block
        briefly until the background thread finishes."""
        if self.current_timeframe == timeframe:
            return

        needs_raw = timeframe in INTRADAY_TIMEFRAMES
        table = self.current_table_name

        if needs_raw and table and table not in self._raw_ready:
            if self._bg_worker is not None and self._bg_worker.isRunning():
                self.data_loading.emit()
                self._bg_worker.wait()
                self.data_loaded.emit()
            if table in self._raw_ready:
                self.original_data = self._model_cache.get(table)

        self._reprocess_for_timeframe(timeframe)
        self.timeframe_changed.emit(timeframe)
        self._update_viewport_limits()
        self._display_ohlc_optimized()
        self._show_last_n_bars(DEFAULT_VISIBLE_BARS)
        self.update_axis_geometry()
        QTimer.singleShot(0, self.update_axis_labels_and_grid)
            
    # Navigation helpers
    def pan_left(self, points: int = 200):
        """Pan left (toward older data) with Y auto-fit."""
        if self.data is None:
            return
        vb = self.main_plot.getViewBox()
        x_lo, x_hi = vb.viewRange()[0]
        new_lo, new_hi = x_lo - points, x_hi - points
        if isinstance(vb, DiscreteViewBox):
            new_lo, new_hi = vb._clamp_x(new_lo, new_hi)
        else:
            new_lo = max(0, new_lo)
        vb.setXRange(new_lo, new_hi, padding=0)
        self.auto_scale_y_for_visible_data()
        self._emit_viewport_changed()

    def pan_right(self, points: int = 200):
        """Pan right (toward newer data) with Y auto-fit."""
        if self.data is None:
            return
        vb = self.main_plot.getViewBox()
        x_lo, x_hi = vb.viewRange()[0]
        new_lo, new_hi = x_lo + points, x_hi + points
        if isinstance(vb, DiscreteViewBox):
            new_lo, new_hi = vb._clamp_x(new_lo, new_hi)
        else:
            new_hi = min(len(self.data) + 50, new_hi)
        vb.setXRange(new_lo, new_hi, padding=0)
        self.auto_scale_y_for_visible_data()
        self._emit_viewport_changed()
        
    def jump_to_end(self):
        """Jump to latest data with Y auto-fit."""
        self._show_last_n_bars(DEFAULT_VISIBLE_BARS)
        
    def auto_scale_y_for_visible_data(self):
        """Auto-scale Y for visible data (skipped when user is in manual Y mode)."""
        if self.data is None or len(self.data) == 0:
            return

        vb = self.main_plot.getViewBox()
        if isinstance(vb, DiscreteViewBox) and vb._manual_y_mode:
            return

        x_range = vb.viewRange()[0]
        x_min = int(max(0, round(x_range[0])))
        x_max = int(min(len(self.data) - 1, round(x_range[1])))

        if x_min < len(self.data) and x_max < len(self.data) and x_min <= x_max:
            self._auto_scale_y_range(x_min, x_max)
            
    def reset_zoom(self):
        """Reset to default view with Y auto-fit."""
        vb = self.main_plot.getViewBox()
        if isinstance(vb, DiscreteViewBox):
            vb._manual_y_mode = False
        if self.last_default_x_range:
            lo, hi = self.last_default_x_range
            self.main_plot.setXRange(lo, hi, padding=0)
            self.auto_scale_y_for_visible_data()
        else:
            self.set_default_view()
            
    def zoom(self, factor, cursor_x=None):
        """Zoom in/out anchored at *cursor_x* (falls back to viewport centre)."""
        if self.data is None or len(self.data) == 0:
            return

        vb = self.main_plot.getViewBox()
        x_lo, x_hi = vb.viewRange()[0]
        span = x_hi - x_lo
        new_span = span / factor

        if cursor_x is not None and span > 0:
            frac = (cursor_x - x_lo) / span
            new_x_min = cursor_x - frac * new_span
            new_x_max = cursor_x + (1.0 - frac) * new_span
        else:
            centre = (x_lo + x_hi) / 2.0
            new_x_min = centre - new_span / 2.0
            new_x_max = centre + new_span / 2.0

        if isinstance(vb, DiscreteViewBox):
            new_x_min, new_x_max = vb._clamp_x(new_x_min, new_x_max)
        else:
            new_x_min = max(0, new_x_min)
            new_x_max = min(len(self.data) + 50, new_x_max)

        if new_x_max - new_x_min < 3:
            return

        self.main_plot.setXRange(new_x_min, new_x_max, padding=0)
        self.auto_scale_y_for_visible_data()
        self._emit_viewport_changed()
            
    def _get_timeframe_interval_minutes(self, timeframe: str) -> int:
        cfg = TIMEFRAME_CONFIG.get(timeframe or self.current_timeframe)
        return int(cfg.get('interval_minutes', 1440)) if cfg else 1440

    def auto_range(self):
        """Auto-range all plots"""
        self.set_default_view()
        for plot in self.indicator_panes.values():
            plot.autoRange()
            
    # -- axis rects (cached per call for readability) --------------------

    def _axis_rects(self):
        y = getattr(self, 'y_axis', None)
        x = getattr(self, 'x_axis', None)
        if y is None or x is None:
            return QRectF(), QRectF()
        y_rect = QRectF(y.pos(), QPointF(y.pos().x() + y._width, y.pos().y() + y._height))
        x_rect = QRectF(x.pos(), QPointF(x.pos().x() + x._width, x.pos().y() + x._height))
        return y_rect, x_rect

    # -- event filter ---------------------------------------------------

    def eventFilter(self, source, event):
        if not hasattr(self, 'y_axis') or not hasattr(self, 'x_axis'):
            return super().eventFilter(source, event)

        # Fast path: during a pan, skip all geometry / cursor work
        if self._is_panning and event.type() == QEvent.Type.MouseMove:
            return super().eventFilter(source, event)

        y_rect, x_rect = self._axis_rects()
        scene_pos = self.chart_container.mapToScene(event.position().toPoint()) \
            if hasattr(event, 'position') else None

        # ---- mouse move / axis drag -----------------------------------
        if event.type() == QEvent.Type.MouseMove and scene_pos is not None:
            # Cursor icon
            if self.dragging_y_axis or self.dragging_x_axis:
                pass  # keep resize cursor during axis drag
            elif y_rect.contains(scene_pos):
                self.setCursor(QCursor(Qt.CursorShape.SizeVerCursor))
                self.axis_hover = 'y'
            elif x_rect.contains(scene_pos):
                self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
                self.axis_hover = 'x'
            else:
                self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
                self.axis_hover = None

            # Y-axis drag: scale around the price level where the user grabbed
            if self.dragging_y_axis and self.drag_start_pos is not None:
                delta_y = scene_pos.y() - self.drag_start_pos.y()
                scale = 1.0 + delta_y / 200.0
                y_lo, y_hi = self.drag_start_y_range
                anchor = self.drag_anchor_y
                old_span = y_hi - y_lo
                new_span = float(np.clip(old_span * scale, 0.01, 1e6))
                frac = (anchor - y_lo) / old_span if old_span > 0 else 0.5
                new_lo = anchor - frac * new_span
                new_hi = anchor + (1.0 - frac) * new_span
                new_lo = max(0, new_lo)
                new_hi = max(new_lo + 0.01, new_hi)
                self.main_plot.setYRange(new_lo, new_hi, padding=0)

            # X-axis drag: scale anchored at the right edge (latest data stays pinned)
            elif self.dragging_x_axis and self.drag_start_pos is not None:
                delta_x = scene_pos.x() - self.drag_start_pos.x()
                scale = 1.0 - delta_x / 200.0
                x_lo, x_hi = self.drag_start_x_range
                old_span = x_hi - x_lo
                max_pts = len(self.data) if self.data is not None else 1000
                new_span = float(np.clip(old_span * scale, 10, max_pts * 1.2))
                new_x_min = x_hi - new_span
                new_x_max = x_hi
                vb = self.main_plot.getViewBox()
                if isinstance(vb, DiscreteViewBox):
                    new_x_min, new_x_max = vb._clamp_x(new_x_min, new_x_max)
                else:
                    new_x_min = max(0, new_x_min)
                self.main_plot.setXRange(new_x_min, new_x_max, padding=0)
                self.auto_scale_y_for_visible_data()

        # ---- press ----------------------------------------------------
        elif event.type() == QEvent.Type.MouseButtonPress and scene_pos is not None:
            if event.button() == Qt.MouseButton.LeftButton:
                if y_rect.contains(scene_pos):
                    self.dragging_y_axis = True
                    self.drag_start_pos = scene_pos
                    vb = self.main_plot.getViewBox()
                    self.drag_start_y_range = vb.viewRange()[1]
                    view_pt = vb.mapSceneToView(scene_pos)
                    self.drag_anchor_y = view_pt.y()
                    if isinstance(vb, DiscreteViewBox):
                        vb._manual_y_mode = True
                    return True
                if x_rect.contains(scene_pos):
                    self.dragging_x_axis = True
                    self.drag_start_pos = scene_pos
                    self.drag_start_x_range = self.main_plot.getViewBox().viewRange()[0]
                    return True

        # ---- release --------------------------------------------------
        elif event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton:
                if self.dragging_y_axis or self.dragging_x_axis:
                    self.dragging_y_axis = False
                    self.dragging_x_axis = False
                    self.drag_start_pos = None
                    self.drag_start_y_range = None
                    self.drag_start_x_range = None
                    self.drag_anchor_y = None
                    return True

        # ---- double-click: reset axes independently -------------------
        elif event.type() == QEvent.Type.MouseButtonDblClick and scene_pos is not None:
            if event.button() == Qt.MouseButton.LeftButton:
                if y_rect.contains(scene_pos):
                    vb = self.main_plot.getViewBox()
                    if isinstance(vb, DiscreteViewBox):
                        vb._manual_y_mode = False
                    self.auto_scale_y_for_visible_data()
                    return True
                if x_rect.contains(scene_pos):
                    vb = self.main_plot.getViewBox()
                    if isinstance(vb, DiscreteViewBox):
                        vb._manual_y_mode = False
                    self.set_default_view()
                    return True

        return super().eventFilter(source, event)
        
    def setup_chart_style(self):
        """Setup chart style"""
        if hasattr(self, 'main_plot'):
            self.main_plot.getViewBox().setBackgroundColor('#0a0a0a')
            
    def clear_all(self):
        """Clear all chart elements efficiently"""
        # Remove plot items
        for line in self.active_lines.values():
            if line.scene():  # Check if still in scene
                self.main_plot.removeItem(line)
        self.active_lines.clear()
        
        # Remove overlays
        for overlay in self.indicator_overlays.values():
            if overlay.scene():
                self.main_plot.removeItem(overlay)
        self.indicator_overlays.clear()
        self._indicator_full_data.clear()

        self.clear_forecast_data()

        # Remove indicator panes
        for indicator_name in list(self.indicator_panes.keys()):
            self.remove_indicator_pane(indicator_name)
        
        # Clear caches
        self._format_cache.clear()
        self._label_cache.clear()
        self._viewport_cache = None
        if hasattr(self, '_method_cache'):
            self._method_cache.clear()
            
    def add_indicator(self, indicator_name, indicator_data, color='#ffff00'):
        """
        Add indicator overlay to chart.
        
        EXPECTATION: MUST align indicator data with chart's x_values.
        Indicator data may have different length due to warmup periods.
        MUST handle alignment by matching lengths properly.
        """
        if indicator_name in self.indicator_overlays:
            self.remove_indicator(indicator_name)

        # Get y_values from indicator data
        if isinstance(indicator_data, pd.Series):
            y_values = indicator_data.values.astype(np.float32)
        else:
            y_values = np.array(indicator_data, dtype=np.float32)

        # Ensure x_values exists and matches the chart data length
        if self.x_values is None:
            # If x_values not set, create based on indicator length
            self.x_values = np.arange(len(y_values), dtype=np.float32)
        
        # Align indicator data with chart's x_values
        # IMPORTANT: Chart typically shows the LAST N data points (most recent)
        # So we need to take the LAST N indicator values, not the first
        x_len = len(self.x_values)
        y_len = len(y_values)
        
        if y_len < x_len:
            # Indicator is shorter - pad with NaN at the start
            y_aligned = np.full(x_len, np.nan, dtype=np.float32)
            y_aligned[-y_len:] = y_values
        elif y_len > x_len:
            # Indicator is longer - take the LAST x_len values (most recent)
            # to match the chart's visible data range
            y_aligned = y_values[-x_len:]
        else:
            # Same length - use as is
            y_aligned = y_values

        # Clip and clean values
        y_aligned = np.clip(y_aligned, -1e6, 1e6)
        y_aligned[np.isinf(y_aligned)] = np.nan

        # Store full data for viewport-windowed rendering
        self._indicator_full_data[indicator_name] = y_aligned

        # Create empty plot item, then populate with visible window only
        plot_item = pg.PlotDataItem(
            pen=pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine),
            connect='finite'
        )

        self.main_plot.addItem(plot_item)
        self.indicator_overlays[indicator_name] = plot_item
        self._refresh_visible_indicators()
        k2_logger.info(f"Added indicator overlay: {indicator_name} (aligned: {y_len} -> {x_len})", "CHART")

    def remove_indicator(self, indicator_name):
        """Remove indicator"""
        if indicator_name in self.indicator_overlays:
            self.main_plot.removeItem(self.indicator_overlays[indicator_name])
            del self.indicator_overlays[indicator_name]
            self._indicator_full_data.pop(indicator_name, None)
            k2_logger.info(f"Removed indicator: {indicator_name}", "CHART")

    # ── Forecast (dashed) lines ──────────────────────────────────────

    _FORECAST_PALETTE = [
        '#00ff00', '#0080ff', '#ff0000', '#ffff00',
        '#ff8000', '#00ffff', '#ff00ff', '#80ff00',
        '#ff4080', '#40c0ff', '#c0ff40', '#ff40c0',
    ]

    def _forecast_color(self, column_name: str) -> str:
        """Pick a color for a forecast column based on OHLC suffix or palette cycling."""
        cn_upper = column_name.upper()
        for ohlc, color in OHLC_COLORS.items():
            if ohlc.upper() in cn_upper:
                return color
        idx = hash(column_name) % len(self._FORECAST_PALETTE)
        return self._FORECAST_PALETTE[idx]

    def add_forecast_line(self, column_name: str, values: list):
        """Render a single named forecast column as a dashed line on the chart."""
        if self.data is None or len(self.data) == 0:
            return

        self.remove_forecast_line(column_name)

        clean = [v for v in values if v is not None]
        if not clean:
            return

        y = np.array(clean, dtype=np.float64)
        base_x = len(self.data)
        color = self._forecast_color(column_name)

        cn_lower = column_name.lower()
        if 'open' in cn_lower:
            anchor_order = ['Open', 'Close', 'High', 'Low']
        elif 'high' in cn_lower:
            anchor_order = ['High', 'Close', 'Open', 'Low']
        elif 'low' in cn_lower:
            anchor_order = ['Low', 'Close', 'Open', 'High']
        else:
            anchor_order = ['Close', 'Open', 'High', 'Low']

        last_val = None
        for ohlc_col in anchor_order:
            if ohlc_col in self.data.columns:
                col_vals = self.data[ohlc_col].dropna()
                if len(col_vals) > 0:
                    last_val = float(col_vals.iloc[-1])
                    break

        if last_val is not None:
            x_arr = np.empty(len(y) + 1, dtype=np.float64)
            y_arr = np.empty(len(y) + 1, dtype=np.float64)
            x_arr[0] = base_x - 1
            y_arr[0] = last_val
            x_arr[1:] = np.arange(base_x, base_x + len(y), dtype=np.float64)
            y_arr[1:] = y
        else:
            x_arr = np.arange(base_x, base_x + len(y), dtype=np.float64)
            y_arr = y

        plot_item = pg.PlotDataItem(
            x=x_arr, y=y_arr,
            pen=pg.mkPen(color=color, width=2,
                         style=Qt.PenStyle.DashLine),
            connect='finite',
        )
        self.main_plot.addItem(plot_item)
        self._forecast_lines[column_name] = plot_item
        k2_logger.info(f"Added forecast line '{column_name}' to chart", "CHART")

    def remove_forecast_line(self, column_name: str):
        """Remove a single named forecast line from the chart."""
        item = self._forecast_lines.pop(column_name, None)
        if item and item.scene():
            self.main_plot.removeItem(item)

    def add_forecast_data(self, forecast_data: dict):
        """Legacy: render dashed OHLC lines for set-indexed forecast data."""
        self.clear_forecast_data()
        if not forecast_data or self.data is None or len(self.data) == 0:
            return

        base_x = len(self.data)

        for set_idx, df in forecast_data.items():
            for ohlc_col in ['Open', 'High', 'Low', 'Close']:
                src_col = f"{ohlc_col}_P{set_idx}"
                if src_col not in df.columns:
                    continue
                series = df[src_col].dropna()
                if series.empty:
                    continue

                color = OHLC_COLORS.get(ohlc_col, '#ffffff')
                y = np.array(series.values, dtype=np.float64)

                y_start = np.empty(len(y) + 1, dtype=np.float64)
                x_start = np.empty(len(y) + 1, dtype=np.float64)

                last_val = None
                if ohlc_col in self.data.columns:
                    col_vals = self.data[ohlc_col].dropna()
                    if len(col_vals) > 0:
                        last_val = float(col_vals.iloc[-1])

                if last_val is not None:
                    x_start[0] = base_x - 1
                    y_start[0] = last_val
                    x_start[1:] = np.arange(base_x, base_x + len(y), dtype=np.float64)
                    y_start[1:] = y
                else:
                    x_start = np.arange(base_x, base_x + len(y), dtype=np.float64)
                    y_start = y

                plot_item = pg.PlotDataItem(
                    x=x_start, y=y_start,
                    pen=pg.mkPen(color=color, width=2,
                                 style=Qt.PenStyle.DashLine),
                    connect='finite',
                )
                self.main_plot.addItem(plot_item)
                key = f"P{set_idx}_{ohlc_col}"
                self._forecast_lines[key] = plot_item

        total = len(self._forecast_lines)
        if total:
            k2_logger.info(f"Added {total} forecast line(s) to chart", "CHART")

    def clear_forecast_data(self):
        """Remove all forecast (dashed) lines from the chart."""
        for key, item in list(self._forecast_lines.items()):
            if item.scene():
                self.main_plot.removeItem(item)
        self._forecast_lines.clear()

    def clear_forecast_lines(self, strategy_name: Optional[str] = None):
        """Remove forecast lines, optionally filtered by strategy prefix.

        When *strategy_name* is None, removes all lines (same as
        ``clear_forecast_data``).  Otherwise removes only lines whose key
        belongs to that strategy (tracked via data_tabs._strategy_columns).
        """
        if strategy_name is None:
            self.clear_forecast_data()
            return
        to_remove = [k for k in self._forecast_lines if k.startswith(strategy_name)]
        for key in to_remove:
            item = self._forecast_lines.pop(key, None)
            if item and item.scene():
                self.main_plot.removeItem(item)

    def toggle_forecast_line(self, key: str, visible: bool):
        """Show/hide a single forecast line by key."""
        item = self._forecast_lines.get(key)
        if item:
            item.setVisible(visible)

    def add_indicator_pane(self, indicator_name, data, chart_type='line', color='#ffffff'):
        """
        Add indicator in a separate pane below the main chart.
        
        Used for oscillators (RSI, Stochastic, MACD) that have different Y-axis scales.
        """
        if indicator_name in self.indicator_panes:
            self.remove_indicator_pane(indicator_name)

        indicator_plot = pg.PlotWidget()
        indicator_plot.setMaximumHeight(150)
        indicator_plot.setMinimumHeight(100)
        indicator_plot.showGrid(x=True, y=True, alpha=0.3)
        indicator_plot.setLabel('left', indicator_name)
        indicator_plot.setBackground('#0a0a0a')

        indicator_plot.getAxis('left').setPen(pg.mkPen(color='#666'))
        indicator_plot.getAxis('left').setTextPen(pg.mkPen(color='#999'))
        indicator_plot.getAxis('bottom').setPen(pg.mkPen(color='#666'))
        indicator_plot.getAxis('bottom').setTextPen(pg.mkPen(color='#999'))

        # Get y_values from data - use float64 for large values (OBV, Volume)
        if isinstance(data, pd.Series):
            y_values = data.values.astype(np.float64)
        else:
            y_values = np.array(data, dtype=np.float64)
        
        # For indicator panes, we need to align with the chart's current data
        # The chart displays x_values which are indices into its data array
        if self.x_values is None or len(self.x_values) == 0:
            k2_logger.warning("No x_values available for indicator pane", "CHART")
            return
        
        x_len = len(self.x_values)
        y_len = len(y_values)
        
        k2_logger.info(f"Indicator pane alignment: x_len={x_len}, y_len={y_len}", "CHART")
        
        # Align indicator data to chart's x_values
        # Chart shows indices [0, x_len), representing the LAST x_len points of full data
        # Indicator data has y_len points; we need the LAST x_len of those
        if y_len < x_len:
            # Indicator shorter than chart - pad with NaN at start
            y_aligned = np.full(x_len, np.nan, dtype=np.float64)
            y_aligned[-y_len:] = y_values
        elif y_len > x_len:
            # Indicator longer than chart - take the last x_len values
            y_aligned = y_values[-x_len:].astype(np.float64)
        else:
            y_aligned = y_values.astype(np.float64)
        
        # Clean up invalid values (use large range for volume-based indicators)
        y_aligned[np.isinf(y_aligned)] = np.nan
        
        # Check if we have any valid data to plot
        valid_count = np.count_nonzero(~np.isnan(y_aligned))
        if valid_count == 0:
            k2_logger.warning(f"No valid data points for indicator pane: {indicator_name}", "CHART")
            return
        
        k2_logger.info(f"Indicator pane {indicator_name}: {valid_count} valid points", "CHART")

        # Create x array matching chart indices
        x_array = self.x_values.copy()

        if chart_type == 'line':
            plot_item = OptimizedPlotDataItem(
                x=x_array,
                y=y_aligned,
                pen=pg.mkPen(color=color, width=2),
                connect='finite'
            )
            indicator_plot.addItem(plot_item)

        elif chart_type == 'bar':
            bargraph = pg.BarGraphItem(
                x=x_array,
                height=y_aligned,
                width=0.8,
                brush=color
            )
            indicator_plot.addItem(bargraph)

        # Store plot item reference for later updates
        indicator_plot.plot_item = plot_item if chart_type == 'line' else bargraph
        indicator_plot.y_data = y_aligned
        indicator_plot.x_data = x_array

        # Add to container and show the indicator widget
        self.indicator_container.addWidget(indicator_plot)
        self.indicator_panes[indicator_name] = indicator_plot
        
        # Show the indicator widget container (it starts hidden)
        if not self.indicator_widget.isVisible():
            self.indicator_widget.show()

        # Set up X-link AFTER adding to layout to ensure proper geometry
        # Disable auto-range first to prevent unwanted resets
        indicator_plot.getViewBox().disableAutoRange()
        indicator_plot.setXLink(self.main_plot)
        
        # Manually set Y-range based on valid data
        valid_mask = ~np.isnan(y_aligned)
        if np.any(valid_mask):
            y_min = float(np.nanmin(y_aligned))
            y_max = float(np.nanmax(y_aligned))
            y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
            indicator_plot.setYRange(y_min - y_padding, y_max + y_padding, padding=0)
        
        # Sync X-range with main chart's current view
        main_vb = self.main_plot.getViewBox()
        x_range = main_vb.viewRange()[0]
        indicator_plot.setXRange(x_range[0], x_range[1], padding=0)

        k2_logger.info(f"Added indicator pane: {indicator_name}", "CHART")
        
    def remove_indicator_pane(self, indicator_name):
        """Remove indicator pane and hide container if no panes remain"""
        if indicator_name in self.indicator_panes:
            widget = self.indicator_panes[indicator_name]
            self.indicator_container.removeWidget(widget)
            widget.deleteLater()
            del self.indicator_panes[indicator_name]
            k2_logger.info(f"Removed indicator pane: {indicator_name}", "CHART")
            
            # Hide the indicator widget container if no panes remain
            if not self.indicator_panes and self.indicator_widget.isVisible():
                self.indicator_widget.hide()
                k2_logger.info("Hidden indicator pane container (no panes active)", "CHART")
        
    def cleanup(self):
        """Cleanup resources"""
        if self._bg_worker is not None and self._bg_worker.isRunning():
            self._bg_worker.quit()
            self._bg_worker.wait(3000)

        for timer_name in ['axis_update_timer', 'range_update_timer', '_viewport_signal_timer', '_y_fit_timer']:
            if hasattr(self, timer_name):
                timer = getattr(self, timer_name)
                timer.stop()
                
        # Stop debounce timers
        if hasattr(self, '_debounce_timers'):
            for timer in self._debounce_timers.values():
                timer.stop()
            self._debounce_timers.clear()

        # Stop momentum timer on the ViewBox
        vb = self.main_plot.getViewBox()
        if isinstance(vb, DiscreteViewBox):
            vb._momentum_timer.stop()

        # Clear data
        self.clear_all()
        self.clear_all_drawings()
        
        # Clear references
        self.data = None
        self.original_data = None
        
        # Clear grid pool
        for line in self._grid_pool['v']:
            if line.scene():
                self.main_plot.removeItem(line)
        for line in self._grid_pool['h']:
            if line.scene():
                self.main_plot.removeItem(line)
        self._grid_pool['v'].clear()
        self._grid_pool['h'].clear()
        
        # Remove axes
        for attr in ['y_axis', 'x_axis']:
            if hasattr(self, attr):
                item = getattr(self, attr)
                if item and item.scene():
                    item.scene().removeItem(item)
                setattr(self, attr, None)
        
        # Remove crosshair
        crosshair_attrs = ['vLine', 'hLine', 'crosshair_date_label', 'crosshair_time_label']
        for attr in crosshair_attrs:
            if hasattr(self, attr):
                item = getattr(self, attr)
                if item and item.scene():
                    self.main_plot.removeItem(item)
                setattr(self, attr, None)
        for lbl in getattr(self, 'crosshair_ohlc_labels', {}).values():
            if lbl and lbl.scene():
                self.main_plot.removeItem(lbl)
        if hasattr(self, 'crosshair_ohlc_labels'):
            self.crosshair_ohlc_labels = {}
        
        # Clear proxy
        if hasattr(self, 'proxy'):
            self.proxy = None
        
        # Force garbage collection
        gc.collect()