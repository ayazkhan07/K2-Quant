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
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPointF, QRectF, QEvent
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


# Smart axis for units (e.g., Volume in thousands/millions)
class SmartUnitAxisItem(pg.AxisItem):
    def __init__(self, orientation='left', mode='volume', **kwargs):
        super().__init__(orientation=orientation, **kwargs)
        self.mode = mode

    def tickStrings(self, values, scale, spacing):
        if self.mode == 'volume':
            out = []
            for v in values:
                av = abs(v)
                if av >= 1_000_000:
                    out.append(f"{v/1_000_000:.1f}M")
                elif av >= 1_000:
                    out.append(f"{v/1_000:.0f}T")
                else:
                    try:
                        out.append(f"{int(v)}")
                    except Exception:
                        out.append(str(v))
            return out
        return super().tickStrings(values, scale, spacing)


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

# View range definitions (duration presets)
class ViewRange(Enum):
    M15 = "15m"
    M30 = "30m"
    H1  = "1h"
    H4  = "4h"
    D1  = "1D"
    D5  = "5D"
    M1  = "1M"
    M3  = "3M"
    YTD = "YTD"
    Y1  = "1Y"
    ALL = "All"

VIEW_RANGE_CONFIG = {
    ViewRange.M15: {"kind": "timedelta", "minutes": 15},
    ViewRange.M30: {"kind": "timedelta", "minutes": 30},
    ViewRange.H1:  {"kind": "timedelta", "hours": 1},
    ViewRange.H4:  {"kind": "timedelta", "hours": 4},
    ViewRange.D1:  {"kind": "timedelta", "days": 1},
    ViewRange.D5:  {"kind": "timedelta", "days": 5},
    ViewRange.M1:  {"kind": "months",   "months": 1},
    ViewRange.M3:  {"kind": "months",   "months": 3},
    ViewRange.YTD: {"kind": "ytd"},
    ViewRange.Y1:  {"kind": "years",    "years": 1},
    ViewRange.ALL: {"kind": "all"}
}

# Fixed Time format configurations - using MM-DD-YYYY
TIME_FORMATS = {
    TimeSpan.INTRADAY_MINUTES: {
        'major': '%m-%d-%Y %H:%M',
        'minor': '%H:%M',
        'context': None,
        'interval_func': lambda span: timedelta(minutes=5 if span < 1800 else 15 if span < 3600 else 30)
    },
    TimeSpan.INTRADAY_HOURS: {
        'major': '%m-%d-%Y %H:00',
        'minor': '%H:%M',
        'context': None,
        'interval_func': lambda span: timedelta(hours=1 if span < 21600 else 2 if span < 43200 else 4)
    },
    TimeSpan.DAILY: {
        'major': '%m-%d-%Y',
        'minor': '%m-%d',
        'context': None,
        'interval_func': lambda span: timedelta(days=1)
    },
    TimeSpan.WEEKLY: {
        'major': '%m-%d-%Y',
        'minor': '%m-%d',
        'context': None,
        'interval_func': lambda span: timedelta(days=1 if span < 604800 else 7)
    },
    TimeSpan.MONTHLY: {
        'major': '%m-%d-%Y',
        'minor': '%m-%d',
        'context': None,
        'interval_func': lambda span: timedelta(days=7 if span < 2592000 else 14)
    },
    TimeSpan.QUARTERLY: {
        'major': '%m-%d-%Y',
        'minor': '%m-%Y',
        'context': None,
        'interval_func': lambda span: timedelta(days=30)
    },
    TimeSpan.YEARLY: {
        'major': '%m-%d-%Y',
        'minor': '%m-%Y',
        'context': None,
        'interval_func': lambda span: timedelta(days=90 if span < 31536000 else 180)
    },
    TimeSpan.MULTI_YEAR: {
        'major': '%m-%d-%Y',
        'minor': '%Y',
        'context': None,
        'interval_func': lambda span: timedelta(days=365)
    }
}

NUMERIC_COLUMNS = frozenset(['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP'])
INTRADAY_TIMEFRAMES = frozenset(['1m', '5m', '15m', '30m', '1h', '4h'])
DAILY_PLUS_TIMEFRAMES = frozenset(['1D'])


# Utility functions
def safe_strftime(date_val, format_string, default=""):
    """Safely format a date, handling NaT and other edge cases"""
    try:
        if pd.isna(date_val):
            return default
        
        if isinstance(date_val, np.datetime64):
            date_val = pd.Timestamp(date_val)
            if pd.isna(date_val):
                return default
        
        if hasattr(date_val, 'strftime'):
            return date_val.strftime(format_string)
        else:
            return str(date_val)[:len(format_string)]
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
    """Manages intelligent time axis labeling and formatting with market-aware anchors."""

    def __init__(self, market_open=(9, 30), market_close=(16, 0)):
        self.label_cache = {}
        self.format_cache = {}
        self.max_labels = 20
        self.min_label_spacing = 50  # pixels
        self.market_open = market_open
        self.market_close = market_close
        self.anchor_tol_intraday_s = 180
        self.anchor_tol_daily_s = 86400
        self.anchor_drop_intraday_s = 6 * 3600
        self.anchor_drop_daily_s = 3 * 86400

    def _infer_granularity(self, data, date_column) -> str:
        try:
            ts = pd.to_datetime(data[date_column], errors='coerce').dropna()
            if len(ts) < 3:
                return 'daily'
            deltas = ts.diff().dropna().astype('timedelta64[s]')
            median_s = float(deltas.median())
            return 'minute' if median_s < 60 * 60 * 20 else 'daily'
        except Exception:
            return 'daily'

    def _visible_window(self, data, date_column, x_range):
        x_min = int(max(0, round(x_range[0])))
        x_max = int(min(len(data) - 1, round(x_range[1])))
        if x_min >= len(data) or x_max < 0 or x_min > x_max:
            return None, None, None
        ts = pd.to_datetime(data[date_column], errors='coerce')
        return x_min, x_max, (ts.iloc[x_min], ts.iloc[x_max])

    def _market_open_close(self, dt):
        ho, mo = self.market_open
        hc, mc = self.market_close
        return dt.replace(hour=ho, minute=mo, second=0, microsecond=0), dt.replace(hour=hc, minute=mc, second=0, microsecond=0)

    def _round_to_quarter_hour(self, dt):
        m = (dt.minute // 15) * 15
        return dt.replace(minute=m, second=0, microsecond=0)

    def _week_start(self, dt):
        return (dt - timedelta(days=dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)

    def _month_start(self, dt):
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def _quarter_start(self, dt):
        qm = ((dt.month - 1) // 3) * 3 + 1
        return dt.replace(month=qm, day=1, hour=0, minute=0, second=0, microsecond=0)

    def _build_intraday_anchors(self, start_dt, end_dt, span_s):
        anchors = []
        d0 = start_dt.normalize(); d1 = end_dt.normalize()
        day = d0
        while day <= d1:
            open_dt, close_dt = self._market_open_close(day)
            if open_dt <= end_dt and close_dt >= start_dt:
                if span_s <= 30 * 60:
                    step = timedelta(minutes=1 if span_s <= 10 * 60 else 5)
                    t = max(open_dt, self._round_to_quarter_hour(start_dt))
                    while t <= min(close_dt, end_dt):
                        anchors.append(t); t += step
                elif span_s <= 3 * 60 * 60:
                    t = max(open_dt, self._round_to_quarter_hour(start_dt))
                    while t <= min(close_dt, end_dt):
                        anchors.append(t); t += timedelta(minutes=15)
                elif span_s <= 7 * 60 * 60:
                    for hh, mm in [(9, 30), (10, 0), (11, 0), (12, 0), (14, 0), (15, 0), (16, 0)]:
                        t = day.replace(hour=hh, minute=mm, second=0, microsecond=0)
                        if start_dt <= t <= end_dt: anchors.append(t)
                elif span_s <= 5 * 24 * 60 * 60:
                    for t in [open_dt, day.replace(hour=12, minute=0, second=0, microsecond=0)]:
                        if start_dt <= t <= end_dt: anchors.append(t)
                else:
                    if start_dt <= open_dt <= end_dt: anchors.append(open_dt)
            day += timedelta(days=1)
        return anchors

    def _build_daily_anchors(self, start_dt, end_dt, span_d):
        anchors = []
        if span_d <= 10:
            t = start_dt.normalize()
            while t <= end_dt:
                anchors.append(t); t += timedelta(days=1)
        elif span_d <= 62:
            t = self._week_start(start_dt)
            while t <= end_dt:
                anchors.append(t); t += timedelta(days=7)
        elif span_d <= 190:
            t = self._month_start(start_dt)
            while t <= end_dt:
                anchors.append(t)
                y = t.year + (1 if t.month == 12 else 0)
                m = 1 if t.month == 12 else t.month + 1
                t = t.replace(year=y, month=m, day=1)
        elif span_d <= 730:
            t = self._quarter_start(start_dt)
            while t <= end_dt:
                anchors.append(t)
                qm = ((t.month - 1) // 3) * 3 + 4
                y = t.year + (1 if qm > 12 else 0)
                m = 1 if qm > 12 else qm
                t = t.replace(year=y, month=m, day=1)
        else:
            t = self._quarter_start(start_dt)
            while t <= end_dt:
                anchors.append(t)
                y = t.year + (1 if t.month >= 10 else 0)
                m = 1 if t.month >= 10 else t.month + 3
                t = t.replace(year=y, month=m, day=1)
        return anchors

    def _format_label_for_plan(self, t, prev, first, granularity, span_s, span_d, crosses_day=False, is_last=False):
        if granularity == 'minute':
            if span_s <= 7 * 60 * 60:
                if first:
                    return safe_strftime(t, "%b %d\n%H:%M", "")
                if crosses_day:
                    return safe_strftime(t, "%b %d\n%H:%M", "")
                if is_last and prev and t.date() != prev.date():
                    return safe_strftime(t, "%H:%M\n%b %d", "")
                return safe_strftime(t, "%H:%M", "")
            if span_s <= 5 * 24 * 60 * 60:
                if first or crosses_day:
                    return safe_strftime(t, "%b %d\n%H:%M", "")
                return safe_strftime(t, "%H:%M", "")
            return safe_strftime(t, "%b %d", "")
        if span_d <= 10:
            txt = safe_strftime(t, "%b %d", "")
            if first and (not prev or t.year != prev.year):
                return f"{txt}\n{safe_strftime(t, '%Y', '')}"
            return txt
        if span_d <= 62:
            return safe_strftime(t, "%b %d", "")
        if span_d <= 190:
            txt = safe_strftime(t, "%b", "")
            if first and (not prev or t.year != prev.year):
                return f"{txt}\n{safe_strftime(t, '%Y', '')}"
            return txt
        if span_d <= 730:
            q = (t.month - 1) // 3 + 1
            return f"Q{q} {t.year}" if first else f"Q{q}"
        return safe_strftime(t, "%Y", "")

    def calculate_time_labels(self, data, date_column, x_range, axis_width):
        if data is None or len(data) == 0 or date_column not in data.columns:
            try: k2_logger.debug("XAXIS: early return - data/date_column issue", "CHART")
            except Exception: pass
            return []
        x_min, x_max, window = self._visible_window(data, date_column, x_range)
        if window is None:
            try: k2_logger.debug("XAXIS: no visible window", "CHART")
            except Exception: pass
            return []
        start_dt, end_dt = window
        if pd.isna(start_dt) or pd.isna(end_dt):
            try: k2_logger.debug("XAXIS: NaT window", "CHART")
            except Exception: pass
            return []

        granularity = self._infer_granularity(data, date_column)
        duration_s = (end_dt - start_dt).total_seconds()
        duration_d = (end_dt - start_dt).days + 1e-6

        try:
            k2_logger.debug(str({
                'event': 'xaxis_plan',
                'granularity': granularity,
                'start_dt': str(start_dt),
                'end_dt': str(end_dt),
                'duration_s': int(duration_s),
                'axis_width_px': int(axis_width)
            }), "CHART")
        except Exception:
            pass

        anchors = self._build_intraday_anchors(start_dt, end_dt, duration_s) if granularity == 'minute' \
                  else self._build_daily_anchors(start_dt, end_dt, duration_d)

        ts = pd.to_datetime(data[date_column], errors='coerce')

        snapped = []
        for t in anchors:
            diffs = np.abs((ts - t).dt.total_seconds())
            if diffs.isna().all():
                continue
            i = int(diffs.idxmin())
            if not (x_min <= i <= x_max):
                continue
            actual_t = ts.iloc[i]
            delta_s = abs((actual_t - t).total_seconds())
            if granularity == 'minute' and delta_s > self.anchor_drop_intraday_s:
                continue
            if granularity == 'daily' and delta_s > self.anchor_drop_daily_s:
                continue
            if granularity == 'minute':
                label_time = t if delta_s <= self.anchor_tol_intraday_s else actual_t
            else:
                label_time = t if delta_s <= self.anchor_tol_daily_s else actual_t
            snapped.append((t, i, label_time))

        try:
            k2_logger.debug(str({
                'event': 'xaxis_counts',
                'anchors': int(len(anchors)),
                'snapped': int(len(snapped))
            }), "CHART")
        except Exception:
            pass

        if not snapped:
            return []

        def build_labels(min_spacing_px: int):
            labels = []
            last_x = -min_spacing_px
            prev_label_time = None
            first = True
            total = len(snapped)
            for idx, (anchor_t, i, label_time) in enumerate(snapped):
                if x_range[1] <= x_range[0] or axis_width <= 0:
                    continue
                x_pos = axis_width * ((i - x_range[0]) / (x_range[1] - x_range[0]))
                if x_pos - last_x < min_spacing_px:
                    continue
                crosses_day = bool(prev_label_time and prev_label_time.date() != label_time.date())
                is_last = (idx == total - 1)
                text = self._format_label_for_plan(
                    t=label_time, prev=prev_label_time, first=first,
                    granularity=granularity, span_s=duration_s, span_d=duration_d,
                    crosses_day=crosses_day, is_last=is_last
                )
                if text:
                    labels.append((text, x_pos))
                    last_x = x_pos
                    prev_label_time = label_time
                    first = False
            return labels

        labels = build_labels(self.min_label_spacing)
        if not labels and axis_width > 0:
            fallback_spacing = max(20, int(self.min_label_spacing * 0.7))
            labels = build_labels(fallback_spacing)

        try:
            k2_logger.debug(str({
                'event': 'xaxis_labels',
                'returned': int(len(labels))
            }), "CHART")
        except Exception:
            pass

        return labels

    def calculate_grid_positions(self, data, date_column, x_range, interval_type=None):
        if data is None or len(data) == 0:
            return []
        x_min, x_max, window = self._visible_window(data, date_column, x_range)
        if window is None or x_min >= x_max:
            return []
        start_dt, end_dt = window

        granularity = self._infer_granularity(data, date_column)
        duration_s = (end_dt - start_dt).total_seconds()
        duration_d = (end_dt - start_dt).days + 1e-6

        candidates = []
        if granularity == 'minute':
            step = None
            if duration_s <= 30 * 60:
                step = timedelta(minutes=1)
            elif duration_s <= 3 * 60 * 60:
                step = timedelta(minutes=5)
            elif duration_s <= 7 * 60 * 60:
                step = timedelta(minutes=15)
            elif duration_s <= 5 * 24 * 60 * 60:
                t = start_dt.normalize()
                while t <= end_dt:
                    o, c = self._market_open_close(t)
                    h = max(o, start_dt)
                    while h <= min(c, end_dt):
                        candidates.append(h); h += timedelta(hours=1)
                    t += timedelta(days=1)
            else:
                t = start_dt.normalize()
                while t <= end_dt:
                    o, _ = self._market_open_close(t)
                    candidates.append(o); t += timedelta(days=1)

            if step is not None:
                t = max(start_dt, self._round_to_quarter_hour(start_dt))
                while t <= end_dt:
                    candidates.append(t); t += step
        else:
            if duration_d <= 10:
                t = start_dt.normalize()
                while t <= end_dt:
                    candidates.append(t); t += timedelta(days=1)
            elif duration_d <= 62:
                t = self._week_start(start_dt)
                while t <= end_dt:
                    candidates.append(t); t += timedelta(days=7)
            else:
                t = self._month_start(start_dt)
                while t <= end_dt:
                    candidates.append(t)
                    y = t.year + (1 if t.month == 12 else 0)
                    m = 1 if t.month == 12 else t.month + 1
                    t = t.replace(year=y, month=m, day=1)

        ts = pd.to_datetime(data[date_column], errors='coerce')
        positions = []
        for t in candidates:
            diffs = np.abs((ts - t).dt.total_seconds())
            if diffs.isna().all():
                continue
            i = int(diffs.idxmin())
            if x_min <= i <= x_max:
                positions.append(i)
        return sorted(set(positions))
    
    def _format_time_label(self, dt, time_span, format_config, prev_label, context_shown):
        """
        Format a datetime for axis label with intelligent context
        """
        if pd.isna(dt):
            return ""
        
        # Use major format
        label = safe_strftime(dt, format_config['major'], '')
        
        # Add context if needed (removed since we set context to None)
        if format_config.get('context') and not context_shown:
            if time_span in [TimeSpan.DAILY, TimeSpan.WEEKLY, TimeSpan.MONTHLY]:
                # Show year on first label or year change
                if not prev_label or dt.year != pd.to_datetime(prev_label).year:
                    context = safe_strftime(dt, format_config['context'], '')
                    if context:
                        label = f"{label}\n{context}"
        
        return label
    
    def _should_show_context(self, current_time, start_time, time_span):
        """
        Determine if context information should be shown
        """
        if time_span in [TimeSpan.DAILY, TimeSpan.WEEKLY]:
            # Show year at start or on year boundary
            return current_time == start_time or current_time.month == 1
        elif time_span == TimeSpan.MONTHLY:
            # Show year on quarter boundaries
            return current_time.month in [1, 4, 7, 10]
        return False
    
    def calculate_grid_positions(self, data, date_column, x_range, interval_type=None):
        """
        Calculate grid line positions based on time intervals
        """
        if data is None or len(data) == 0:
            return []
        
        x_min = int(max(0, round(x_range[0])))
        x_max = int(min(len(data) - 1, round(x_range[1])))
        
        if x_min >= x_max:
            return []
        
        # For small ranges, show grid at each data point
        visible_points = x_max - x_min + 1
        if visible_points <= 50:
            return list(range(x_min, x_max + 1))
        
        # For larger ranges, use time-based grid
        try:
            start_date = pd.to_datetime(data.iloc[x_min][date_column])
            end_date = pd.to_datetime(data.iloc[x_max][date_column])
        except:
            # Fallback to regular intervals
            step = max(1, visible_points // 50)
            return list(range(x_min, x_max + 1, step))
        
        if pd.isna(start_date) or pd.isna(end_date):
            step = max(1, visible_points // 50)
            return list(range(x_min, x_max + 1, step))
        
        # Get appropriate interval
        time_span, duration = get_time_span(start_date, end_date)
        
        # Use smaller intervals for grid than for labels
        if time_span == TimeSpan.INTRADAY_MINUTES:
            grid_interval = timedelta(minutes=1)
        elif time_span == TimeSpan.INTRADAY_HOURS:
            grid_interval = timedelta(minutes=30)
        elif time_span == TimeSpan.DAILY:
            grid_interval = timedelta(hours=6)
        elif time_span == TimeSpan.WEEKLY:
            grid_interval = timedelta(days=1)
        elif time_span == TimeSpan.MONTHLY:
            grid_interval = timedelta(days=1)
        elif time_span == TimeSpan.QUARTERLY:
            grid_interval = timedelta(days=7)
        elif time_span == TimeSpan.YEARLY:
            grid_interval = timedelta(days=30)
        else:  # MULTI_YEAR
            grid_interval = timedelta(days=90)
        
        # Generate grid positions
        positions = []
        current_time = snap_to_time_boundary(start_date, time_span)
        
        while current_time <= end_date:
            # Find closest data point
            time_diffs = np.abs((data[date_column] - current_time).dt.total_seconds())
            if not time_diffs.isna().all():
                closest_idx = time_diffs.idxmin()
                if pd.notna(closest_idx) and x_min <= closest_idx <= x_max:
                    positions.append(closest_idx)
            
            current_time += grid_interval
            
            # Limit grid lines to prevent performance issues
            if len(positions) > 100:
                break
        
        return positions


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
    """Optimized ViewBox with discrete X-axis behavior and proper boundaries"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discrete_x = True
        # Start with reasonable defaults, will be updated when data is loaded
        self.setLimits(xMin=0, xMax=1e6, yMin=0, yMax=1e6)
        self._drag_cache = {}
        self.data_x_max = 0  # Will be updated when data is loaded
        
    def mouseDragEvent(self, ev, axis=None):
        if not self.discrete_x or axis == 1:
            super().mouseDragEvent(ev, axis)
            return
            
        if ev.button() != Qt.MouseButton.LeftButton:
            return
            
        if ev.isStart():
            self._drag_cache = {
                'start_pos': ev.pos(),
                'start_range': self.viewRange()
            }
        elif ev.isFinish():
            self._drag_cache.clear()
        elif self._drag_cache:
            delta = ev.pos() - self._drag_cache['start_pos']
            x_range = self._drag_cache['start_range'][0]
            
            # Guard against zero width
            width = self.width()
            if width == 0:
                return
                
            x_scale = (x_range[1] - x_range[0]) / width
            x_offset = round(delta.x() * x_scale)
            
            new_x_min = round(self._drag_cache['start_range'][0][0] - x_offset)
            new_x_max = round(self._drag_cache['start_range'][0][1] - x_offset)
            
            # Enforce X boundaries
            new_x_min = max(0, new_x_min)  # Can't go before first data point
            if hasattr(self, 'data_x_max') and self.data_x_max > 0:
                # Allow small buffer past last data point
                buffer = min(50, int((new_x_max - new_x_min) * 0.1))
                new_x_max = min(self.data_x_max + buffer, new_x_max)
                new_x_min = min(new_x_min, new_x_max - 10)  # Ensure at least 10 points visible
            
            self.setXRange(new_x_min, new_x_max, padding=0)
            
            if axis is None:
                y_range = self._drag_cache['start_range'][1]
                
                # Guard against zero height
                height = self.height()
                if height == 0:
                    return
                    
                y_scale = (y_range[1] - y_range[0]) / height
                y_offset = delta.y() * y_scale
                
                # Calculate new Y range
                new_y_min = self._drag_cache['start_range'][1][0] + y_offset
                new_y_max = self._drag_cache['start_range'][1][1] + y_offset
                
                # Enforce Y boundaries - prices can't be negative
                new_y_min = max(0, new_y_min)
                new_y_max = max(new_y_min + 0.01, new_y_max)  # Ensure some range
                
                self.setYRange(new_y_min, new_y_max, padding=0)
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
            self._height = 40  # Increased height for multi-level labels

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
                # Draw main labels
                for label, pos in self.labels:
                    # Check for multi-line labels (with context)
                    if '\n' in label:
                        parts = label.split('\n')
                        # Main label
                        text_rect = QRectF(pos - 40, 5, 80, 20)
                        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, parts[0])
                        # Context label (smaller, below)
                        painter.setFont(self.small_font)
                        painter.setPen(self.subtext_pen)
                        text_rect = QRectF(pos - 40, 20, 80, 15)
                        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, parts[1])
                        painter.setFont(self.font)
                        painter.setPen(self.text_pen)
                    else:
                        text_rect = QRectF(pos - 40, 5, 80, self._height - 10)
                        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)

    def setLabels(self, labels):
        if labels != self.labels:
            self.labels = labels
            self.update()


class ChartWidget(QWidget):
    """Enhanced Trading-view style chart widget with intelligent time axis"""
    
    # Signals
    drawing_added = pyqtSignal(dict)
    time_range_changed = pyqtSignal(str, str)
    time_range_selected = pyqtSignal(str)
    fetch_older_requested = pyqtSignal(object)
    timeframe_changed = pyqtSignal(str)
    view_range_changed = pyqtSignal(str)
    allowed_view_ranges_changed = pyqtSignal(list)
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
        self.current_timeframe = '1D'
        self.current_view_range = ViewRange.D5
        self.min_visible_points = 10
        self.min_granularity = None
        self.last_5_days_range = None
        
        # Time axis manager
        self.time_axis_manager = TimeAxisManager()
        
        # Collections
        self.active_lines = {}
        self.indicator_panes = {}
        self.indicator_overlays = {}
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
        self.axis_hover = None
        
        # Grid lines pool (reusable)
        self._grid_pool = {'v': [], 'h': []}
        self._active_grids = {'v': 0, 'h': 0}
        
        # Caching
        self._viewport_cache = None
        self._label_cache = {}
        self._format_cache = {}
        self._method_cache = {}
        
        # DB context
        self.current_table_name = None
        self.total_records = 0
        self._global_start_index = 0
        self._initial_chunk_size = 1000
        self.is_fetching = False
        
        # Debounce timers dict
        self._debounce_timers = {}
        
    def _setup_timers(self):
        """Setup optimized timers with single timer reuse"""
        # Axis update timer
        self.axis_update_timer = QTimer()
        self.axis_update_timer.setSingleShot(True)
        self.axis_update_timer.setInterval(16)
        self.axis_update_timer.timeout.connect(self.update_axis_labels_and_grid)
        
        # Axis geometry update timer
        self.range_update_timer = QTimer()
        self.range_update_timer.setSingleShot(True)
        self.range_update_timer.setInterval(16)
        self.range_update_timer.timeout.connect(self.update_axis_geometry)
        
        # Viewport update timer for auto-loading
        self.viewport_update_timer = QTimer()
        self.viewport_update_timer.setSingleShot(True)
        self.viewport_update_timer.setInterval(300)
        self.viewport_update_timer.timeout.connect(self._check_and_fetch_if_needed)
        
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

        # Autoload control flags (prevent pane flashing on immediate reloads)
        self._autoload_suspended = False
        self._last_ui_mutation_ms = 0

        # Caches for sub-pane series data and modes so we can reapply after reloads
        self._series_data = {}
        self._series_y_mode = {}

        # Axis retry gate to avoid repeated reschedules before layout settles
        self._axis_retry_pending = False

    # --- Autoload guards ----------------------------------------------------
    def _suspend_autoload(self):
        self._autoload_suspended = True
        try:
            if hasattr(self, 'viewport_update_timer') and self.viewport_update_timer is not None:
                self.viewport_update_timer.stop()
        except Exception:
            pass

    def _resume_autoload(self):
        self._autoload_suspended = False

    def _mark_ui_mutation(self):
        try:
            import time as _t
            self._last_ui_mutation_ms = int(_t.time() * 1000)
        except Exception:
            self._last_ui_mutation_ms = 0

    def _recent_ui_mutation(self, window_ms: int = 250) -> bool:
        try:
            import time as _t
            return int(_t.time() * 1000) - int(getattr(self, '_last_ui_mutation_ms', 0)) < window_ms
        except Exception:
            return False

    # --- Re-apply cached indicator panes after data reload ------------------
    def _reapply_indicator_panes(self):
        if not hasattr(self, 'x_values') or self.x_values is None:
            return
        if not hasattr(self, '_series_data') or not self._series_data:
            return
        for (pane_key, series_label), series in list(self._series_data.items()):
            try:
                y_values = series.values.astype(np.float32) if isinstance(series, pd.Series) else np.asarray(series, dtype=np.float32)
                y_values = np.clip(y_values, -1e6, 1e6)
                y_values[np.isinf(y_values)] = np.nan
                n = int(min(len(self.x_values), len(y_values)))
                if n <= 0:
                    continue
                x_tail = self.x_values[-n:]
                y_tail = y_values[-n:]
                if hasattr(self, 'pane_series_map') and pane_key in self.pane_series_map and series_label in self.pane_series_map[pane_key]:
                    item = self.pane_series_map[pane_key][series_label]
                    if isinstance(item, OptimizedPlotDataItem):
                        item.setData(x=x_tail, y=y_tail)
                else:
                    y_mode = self._series_y_mode.get((pane_key, series_label), 'auto') if hasattr(self, '_series_y_mode') else 'auto'
                    self.add_or_update_indicator_pane(pane_key=pane_key, series_label=series_label, series=series, y_mode=y_mode)
            except Exception:
                continue
        
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
        
        # Indicator container
        self.indicator_container = QVBoxLayout()
        self.indicator_container.setSpacing(2)
        layout.addLayout(self.indicator_container, stretch=1)
        
        return container
        
    def _setup_main_plot(self):
        """Setup main plot with optimizations"""
        self.main_plot.hideAxis('left')
        self.main_plot.hideAxis('bottom')
        self.main_plot.showGrid(x=False, y=False)
        self.main_plot.getViewBox().setMouseEnabled(x=True, y=False)
        self.main_plot.getViewBox().disableAutoRange()
        
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
        """Add crosshair with optimized updates"""
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#666', width=1))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#666', width=1))
        self.main_plot.addItem(self.vLine, ignoreBounds=True)
        self.main_plot.addItem(self.hLine, ignoreBounds=True)
        
        self.value_label = pg.TextItem(color='#fff', anchor=(0, 1))
        self.value_label.setFont(QFont('Arial', 10))
        self.main_plot.addItem(self.value_label)
        
        # Rate-limited crosshair updates
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
        vb.sigRangeChanged.connect(self._emit_viewport_changed)
        vb.sigRangeChanged.connect(
            lambda: self.viewport_update_timer.start() if self.current_table_name else None
        )
        
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

        # DIAGNOSTIC: surface current view and data state
        try:
            k2_logger.debug(str({
                'event': 'xaxis_update',
                'x_range': (float(x_range[0]), float(x_range[1])),
                'axis_width_px': int(getattr(self.x_axis, '_width', 0)),
                'date_column': self.date_column,
                'data_rows': int(len(self.data) if self.data is not None else 0)
            }), "CHART")
        except Exception:
            pass

        # NEW: gate until axis has width and datetime is ready; retry once
        try:
            axis_width_px = int(getattr(self.x_axis, '_width', 0))
        except Exception:
            axis_width_px = 0

        date_ready = False
        try:
            date_ready = (
                self.data is not None and
                isinstance(self.date_column, str) and
                self.date_column in self.data.columns and
                self.data[self.date_column].notna().any()
            )
        except Exception:
            date_ready = False

        if axis_width_px <= 0 or not date_ready:
            if not getattr(self, '_axis_retry_pending', False):
                self._axis_retry_pending = True
                def _retry_axis_update():
                    self._axis_retry_pending = False
                    self.update_axis_labels_and_grid()
                QTimer.singleShot(40, _retry_axis_update)
            return

        self._update_y_axis_and_grid(y_range)
        self._update_x_axis_and_grid_intelligent(x_range)

        # Also ensure sub-pane series stay aligned when the view changes after reload
        try:
            self._reapply_indicator_panes()
        except Exception:
            pass
        
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
        """Update X-axis with intelligent time-aware labeling"""
        # Hide all vertical grid lines first
        for i in range(self._active_grids['v']):
            if i < len(self._grid_pool['v']):
                self._grid_pool['v'][i].setVisible(False)
        
        if self.data is None or self.date_column not in self.data.columns or len(self.data) == 0:
            self.x_axis.setLabels([])
            return
        
        # Validate x_range
        x_min = max(0, int(round(x_range[0])))
        x_max = min(len(self.data) - 1, int(round(x_range[1])))
        
        # Prevent invalid ranges
        if x_min >= len(self.data) or x_max < 0 or x_min > x_max:
            self.x_axis.setLabels([])
            return
        
        # Ensure we have valid data in range
        try:
            visible_data = self.data.iloc[x_min:x_max+1]
            if visible_data.empty or visible_data[self.date_column].isna().all():
                self.x_axis.setLabels([])
                return
        except (IndexError, KeyError):
            self.x_axis.setLabels([])
            return
        
        # Convert date column to datetime if needed
        if self.date_column in self.data.columns:
            try:
                if not pd.api.types.is_datetime64_any_dtype(self.data[self.date_column]):
                    self.data[self.date_column] = pd.to_datetime(self.data[self.date_column], errors='coerce')
            except:
                pass
        
        axis_width = self.x_axis._width
        
        # Get time-aware labels
        labels = self.time_axis_manager.calculate_time_labels(
            self.data, self.date_column, x_range, axis_width
        )
        
        # Get grid positions
        grid_positions = self.time_axis_manager.calculate_grid_positions(
            self.data, self.date_column, x_range
        )
        
        # Show grid lines
        for i, pos in enumerate(grid_positions[:len(self._grid_pool['v'])]):
            if i < len(self._grid_pool['v']):
                self._grid_pool['v'][i].setPos(pos)
                self._grid_pool['v'][i].setVisible(True)
        
        self._active_grids['v'] = len(grid_positions[:len(self._grid_pool['v'])])
        
        # Set labels
        self.x_axis.setLabels(labels)
        
    def update_axis_geometry(self):
        """Update axis geometry"""
        if not hasattr(self, 'y_axis') or not hasattr(self, 'x_axis'):
            return

        vb = self.main_plot.getViewBox()
        vb_rect = vb.sceneBoundingRect()

        y_axis_width = 70
        x_axis_height = 40  # Increased for multi-level labels

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
        """Load data with optimized chunking"""
        if self.is_fetching:
            return
            
        try:
            self.is_fetching = True
            self.data_loading.emit()
            
            self.current_table_name = table_name
            self.total_records = total_records or 0
            
            if not self.total_records and stock_service:
                info = stock_service.get_table_info(table_name) or {}
                self.total_records = int(info.get('total_records', 0))
                
            if self.total_records <= 0:
                k2_logger.warning("No records to load for chart", "CHART")
                return
                
            # Load initial chunk
            chunk_size = min(self._initial_chunk_size, self.total_records)
            start_idx = max(0, self.total_records - chunk_size)
            
            if stock_service:
                df = stock_service.get_chart_data_chunk(table_name, start_idx, self.total_records)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    self._global_start_index = start_idx
                    self._process_dataframe(df)
            else:
                k2_logger.warning("Stock service not available", "CHART")
                    
        except Exception as e:
            k2_logger.error(f"Failed to load data: {e}", "CHART")
        finally:
            self.is_fetching = False
            self.data_loaded.emit()
            
    def _process_dataframe(self, df: pd.DataFrame):
        """Process dataframe with optimizations and set viewport limits"""
        # Store original
        self.original_data = df.copy()
        
        # Convert numeric columns efficiently
        numeric_cols = list(NUMERIC_COLUMNS & set(df.columns))
        if numeric_cols:
            for col in numeric_cols:
                self.original_data[col] = pd.to_numeric(self.original_data[col], errors='coerce')
            
        # Detect granularity
        self._detect_granularity_optimized()
        
        # Process for current timeframe
        self._process_timeframe_optimized()
        
        # Update ViewBox limits based on actual data
        self._update_viewport_limits()
        
        # Display lines
        self._display_ohlc_optimized()
        
        # Set view
        self.set_default_view()
        
        # Update UI
        self.update_axis_geometry()
        QTimer.singleShot(0, self.update_axis_labels_and_grid)
        
        # NEW: post-layout nudge to avoid blank top panel if first pass had axis_width=0
        QTimer.singleShot(40, self.update_axis_labels_and_grid)
        
        # Re-apply any cached indicator panes/series now that X has been rebuilt
        try:
            self._reapply_indicator_panes()
        except Exception:
            pass
        
        k2_logger.info(f"Data loaded: {len(self.data)} records", "CHART")
        
    def _update_viewport_limits(self):
        """Update ViewBox limits based on actual data - FIXED VERSION"""
        if self.data is None or len(self.data) == 0:
            return
            
        vb = self.main_plot.getViewBox()
        
        # Calculate Y limits from data
        y_min = float('inf')
        y_max = float('-inf')
        
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in self.data.columns:
                # Ensure data is numeric
                col_data = pd.to_numeric(self.data[col], errors='coerce').dropna()
                if len(col_data) > 0:
                    # Filter out any remaining inf values
                    col_data = col_data[~np.isinf(col_data)]
                    if len(col_data) > 0:
                        y_min = min(y_min, col_data.min())
                        y_max = max(y_max, col_data.max())
        
        # Ensure valid Y range
        if y_min == float('inf') or y_max == float('-inf'):
            y_min, y_max = 0, 100
        
        # Set limits with appropriate padding
        x_max = len(self.data) - 1
        y_min = max(0, y_min * 0.9)  # 10% padding below (but never negative)
        y_max = y_max * 1.2  # 20% padding above (not 2x!)
        
        # Update ViewBox limits
        vb.setLimits(
            xMin=0,
            xMax=x_max + 50,  # Fixed 50 point buffer instead of percentage
            yMin=0,  # Prices can't be negative
            yMax=y_max  # Just 20% above max, not doubled
        )
        
        # Store max X in ViewBox for reference
        if isinstance(vb, DiscreteViewBox):
            vb.data_x_max = x_max
            
    def _detect_granularity_optimized(self):
        """Optimized granularity detection"""
        if self.original_data is None or len(self.original_data) == 0:
            self.min_granularity = '1D'
            return
            
        # Quick check for time column
        if 'Time' not in self.original_data.columns:
            self.min_granularity = '1D'
            self._update_timeframe_buttons('1D')
            return
            
        try:
            # Create datetime column if needed
            if 'datetime' not in self.original_data.columns:
                self.original_data['datetime'] = pd.to_datetime(
                    self.original_data['Date'].astype(str) + ' ' +
                    self.original_data['Time'].astype(str),
                    format='%Y-%m-%d %H:%M:%S',
                    errors='coerce'
                )
                
            # Sample intervals for speed (use first 100 rows)
            sample = self.original_data['datetime'].head(100)
            diffs = sample.diff().dt.total_seconds() / 60
            diffs = diffs[diffs > 0]
            
            if len(diffs) > 0 and not diffs.isna().all():
                min_interval = diffs[~diffs.isna()].min()
                
                # Determine granularity
                for tf, config in TIMEFRAME_CONFIG.items():
                    if min_interval <= config['interval_minutes']:
                        self.min_granularity = tf
                        break
                else:
                    self.min_granularity = '1D'
            else:
                self.min_granularity = '1D'
                
            self._update_timeframe_buttons(self.min_granularity)
            
        except Exception as e:
            k2_logger.error(f"Granularity detection failed: {e}", "CHART")
            self.min_granularity = '1D'
            self._update_timeframe_buttons('1D')
            
    def _update_timeframe_buttons(self, min_tf):
        """Update timeframe button states"""
        timeframes = list(TIMEFRAME_CONFIG.keys())
        min_idx = timeframes.index(min_tf) if min_tf in timeframes else 4
        
        for i, tf in enumerate(timeframes):
            if tf in self.timeframe_buttons:
                self.timeframe_buttons[tf].setEnabled(i >= min_idx)
                
    def _process_timeframe_optimized(self):
        """Optimized timeframe processing with improved date handling"""
        self.data = self.original_data.copy()
        
        # More robust date handling
        if 'Date' in self.data.columns:
            # First ensure Date column is properly formatted
            self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')

            # If intraday, drop off-market hours and weekends before plotting
            if 'Time' in self.data.columns and self.current_timeframe in INTRADAY_TIMEFRAMES:
                try:
                    combined_dt = pd.to_datetime(
                        self.data['Date'].dt.strftime('%Y-%m-%d') + ' ' + self.data['Time'].astype(str),
                        format='%Y-%m-%d %H:%M:%S',
                        errors='coerce'
                    )
                except Exception:
                    combined_dt = pd.to_datetime(
                        self.data['Date'].astype(str) + ' ' + self.data['Time'].astype(str),
                        errors='coerce'
                    )

                try:
                    start_t = datetime.strptime('09:30:00', '%H:%M:%S').time()
                    end_t = datetime.strptime('16:00:00', '%H:%M:%S').time()
                    mask = (combined_dt.dt.weekday < 5) & (combined_dt.dt.time >= start_t) & (combined_dt.dt.time < end_t)
                    self.data = self.data[mask].reset_index(drop=True)
                except Exception:
                    # If anything fails, keep data as-is (fail-safe)
                    pass
            
            # Check if we have valid dates
            if self.data['Date'].notna().any():
                if 'Time' in self.data.columns and self.current_timeframe in INTRADAY_TIMEFRAMES:
                    # Only combine with time for intraday timeframes
                    try:
                        self.data['datetime'] = pd.to_datetime(
                            self.data['Date'].dt.strftime('%Y-%m-%d') + ' ' + 
                            self.data['Time'].astype(str),
                            format='%Y-%m-%d %H:%M:%S',
                            errors='coerce'
                        )
                    except:
                        self.data['datetime'] = self.data['Date']
                else:
                    # For daily+ timeframes, just use date without time
                    self.data['datetime'] = self.data['Date'].dt.normalize()  # Remove time component
                
                self.date_column = 'datetime'
            else:
                k2_logger.error("No valid dates found in Date column", "CHART")
                self.date_column = None
        
        # Resample if needed (only 1D remains for daily+; no weekly/monthly/yearly)
        if self.current_timeframe not in INTRADAY_TIMEFRAMES and self.current_timeframe != '1D':
            # No resampling paths for removed timeframes
            pass
            
        # Create x values (preserve panes across reload)
        self.clear_all(preserve_indicator_panes=True)
        self.x_values = np.arange(len(self.data), dtype=np.float32)
        
    def _resample_data_optimized(self):
        """Optimized data resampling"""
        config = TIMEFRAME_CONFIG.get(self.current_timeframe)
        if not config:
            return
            
        try:
            # Preserve the last actual trading row in each business period
            df = self.data.copy()
            df.set_index('datetime', inplace=True)

            tf = self.current_timeframe

            def last_per(period_alias: str):
                # Group by business period and keep the last row (preserves the real timestamp)
                return df.groupby(df.index.to_period(period_alias)).tail(1)

            if tf == '1W':
                # Weeks anchored to Friday; picks Friday's last trade (or prior day if Friday closed)
                sampled = last_per('W-FRI')
            elif tf == '1M':
                sampled = last_per('M')
            elif tf == '3M':
                sampled = last_per('3M')
            elif tf == '1Y':
                sampled = last_per('Y')
            else:
                # For intraday/1D, keep existing (no period compression here)
                sampled = df

            self.data = sampled.reset_index()

        except Exception as e:
            k2_logger.error(f"Resampling failed: {e}", "CHART")
            self.data = self.original_data.copy()
            
    def _display_ohlc_optimized(self):
        """Optimized OHLC display - FIXED to clear existing lines first"""
        # Clear any existing lines to prevent duplicates
        for line in list(self.active_lines.values()):
            if line.scene():  # Check if item is still in scene
                self.main_plot.removeItem(line)
        self.active_lines.clear()
        
        # Now add the lines that should be visible
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in self.data.columns and col in self.ohlc_buttons:
                if self.ohlc_buttons[col].isChecked():
                    self._add_ohlc_line_optimized(col)
                
    def _add_ohlc_line_optimized(self, column_name):
        """Add OHLC line with optimization"""
        if column_name in self.active_lines:
            return
            
        if column_name not in self.data.columns:
            return
            
        # Get data efficiently
        y_values = self.data[column_name].values
        
        # Ensure numeric type
        y_values = pd.to_numeric(y_values, errors='coerce')
        y_values = np.array(y_values, dtype=np.float32)
        
        # Clean data - remove inf and clip
        y_values[np.isinf(y_values)] = np.nan
        y_values = np.clip(y_values, -1e6, 1e6)
        
        # Check for valid data
        finite_mask = np.isfinite(y_values)
        if not np.any(finite_mask):
            return
        
        # Create plot item
        color = OHLC_COLORS.get(column_name, '#ffffff')
        plot_item = OptimizedPlotDataItem(
            x=self.x_values[:len(y_values)],
            y=y_values,
            pen=pg.mkPen(color=color, width=2),
            connect='finite'
        )
        
        self.main_plot.addItem(plot_item)
        self.active_lines[column_name] = plot_item
        k2_logger.info(f"Added {column_name} line", "CHART")
        
    def set_default_view(self):
        """Set default view with optimization"""
        if self.data is None or len(self.data) == 0:
            return
            
        config = TIMEFRAME_CONFIG.get(self.current_timeframe, {})
        points_per_day = config.get('points_per_day', 1)
        points_for_5_days = int(5 * points_per_day)
        points_for_5_days = max(20, min(points_for_5_days, len(self.data)))
        
        future_points = min(50, int(points_for_5_days * 0.1))  # Smaller future buffer
        x_max = len(self.data) - 1 + future_points
        x_min = max(0, len(self.data) - 1 - points_for_5_days)
        
        x_min = int(round(x_min))
        x_max = int(round(x_max))
        
        self.main_plot.setXRange(x_min, x_max, padding=0)
        self._auto_scale_y_range(x_min, min(len(self.data) - 1, x_max))
        
        # Store default range
        self.last_5_days_range = ((x_min, x_max), self.main_plot.getViewBox().viewRange()[1])
        
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
        """Optimized crosshair update with correct coordinate mapping and date formatting"""
        pos = evt[0]
        if not self.main_plot.sceneBoundingRect().contains(pos):
            return
            
        mousePoint = self.main_plot.getViewBox().mapSceneToView(pos)
        
        # Snap to discrete X
        x_raw = mousePoint.x()
        x_snapped = int(round(x_raw))
        x_snapped = max(0, min(x_snapped, len(self.data) - 1) if self.data is not None else x_snapped)
        
        self.vLine.setPos(x_snapped)
        self.hLine.setPos(mousePoint.y())
        
        # Update label with proper date formatting and price from data
        if self.data is not None and self.date_column and len(self.data) > 0:
            if 0 <= x_snapped < len(self.data):
                # Get date value
                date_val = self.data.iloc[x_snapped][self.date_column]
                
                # Format date based on current view range (≤1D shows time)
                if pd.notna(date_val):
                    if self._range_is_intraday():
                        date_str = safe_strftime(date_val, '%m-%d-%Y %H:%M:%S', 'N/A')
                    else:
                        date_str = safe_strftime(date_val, '%m-%d-%Y', 'N/A')
                else:
                    date_str = "N/A"
                
                # Hybrid snap: prefer nearest series (OHLC + indicators) within pixel tolerance,
                # else snap to nearest grid level, else free-float cursor Y
                PIX_TOL = 8  # pixels

                def _y_to_pixel(y_val: float) -> float:
                    vb_local = self.main_plot.getViewBox()
                    return vb_local.mapViewToScene(QPointF(0, y_val)).y()

                y_cursor = float(mousePoint.y())

                # Collect candidate series values at current x
                candidates = []
                # OHLC series
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in self.data.columns and col in self.active_lines:
                        val_raw = self.data.iloc[x_snapped][col]
                        val_num = pd.to_numeric(val_raw, errors='coerce')
                        if pd.notna(val_num) and np.isfinite(val_num):
                            candidates.append(float(val_num))
                # Indicator overlays
                for item in self.indicator_overlays.values():
                    try:
                        x_arr, y_arr = item.getData()
                        if x_arr is None or y_arr is None or len(x_arr) == 0:
                            continue
                        ix = int(np.clip(np.searchsorted(x_arr, x_snapped), 0, len(x_arr) - 1))
                        yv_raw = y_arr[ix]
                        yv_num = pd.to_numeric(yv_raw, errors='coerce')
                        if pd.notna(yv_num) and np.isfinite(yv_num):
                            candidates.append(float(yv_num))
                    except Exception:
                        pass

                snap_series = None
                if candidates:
                    y_cursor_px = _y_to_pixel(y_cursor)
                    best = min(candidates, key=lambda v: abs(_y_to_pixel(v) - y_cursor_px))
                    if abs(_y_to_pixel(best) - y_cursor_px) <= PIX_TOL:
                        snap_series = best

                if snap_series is not None:
                    y_display = snap_series
                else:
                    # Try grid snap
                    vb_local = self.main_plot.getViewBox()
                    _, (y_min, y_max) = vb_local.viewRange()
                    price_range = max(0.01, y_max - max(0, y_min))
                    interval = self._get_price_interval(price_range)
                    grid_snap = round(y_cursor / interval) * interval
                    y_display = grid_snap if abs(_y_to_pixel(grid_snap) - _y_to_pixel(y_cursor)) <= PIX_TOL else y_cursor

                # Set crosshair and label at snapped value (or free-float fallback)
                self.hLine.setPos(y_display)
                price_str = f"${y_display:.3f}" if y_display < 10 else (f"${y_display:.2f}" if y_display < 1000 else f"${y_display:,.2f}")
                self.value_label.setText(f"{date_str}\n{price_str}")
                self.value_label.setPos(x_snapped, y_display)
            
    def _emit_viewport_changed(self):
        """Emit viewport changed signal"""
        if self.data is None:
            return
            
        vb = self.main_plot.getViewBox()
        x_min, x_max = vb.viewRange()[0]
        
        start_local = int(max(0, round(x_min)))
        end_local = int(min(len(self.data), round(x_max)))
        
        start_global = self._global_start_index + start_local
        end_global = self._global_start_index + end_local
        total = self.total_records if self.total_records else len(self.data)
        
        self.viewport_changed.emit(start_global, end_global, total)
        
    def _check_and_fetch_if_needed(self):
        """Check if more data needs to be fetched"""
        if not self.current_table_name or self.is_fetching or self.data is None:
            return
        # Guard: avoid immediate auto fetch right after UI pane/overlay mutations
        if getattr(self, '_autoload_suspended', False):
            return
        if self._recent_ui_mutation(250):
            return
            
        vb = self.main_plot.getViewBox()
        x_min, _ = vb.viewRange()[0]
        
        # Fetch if near edge and more data available
        NEAR_EDGE_THRESHOLD = 50
        if int(round(x_min)) <= NEAR_EDGE_THRESHOLD and self._global_start_index > 0:
            self._fetch_older_data()
            
    def _fetch_older_data(self):
        """Fetch older data with optimization and update viewport limits"""
        if self.is_fetching:
            return
            
        try:
            self.is_fetching = True
            self.data_loading.emit()
            
            # Calculate chunk to fetch
            fetch_size = self._initial_chunk_size
            older_end = self._global_start_index
            older_start = max(0, older_end - fetch_size)
            
            if stock_service:
                df = stock_service.get_chart_data_chunk(
                    self.current_table_name, 
                    older_start, 
                    older_end
                )
                
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Store current view position
                    vb = self.main_plot.getViewBox()
                    x_min, x_max = vb.viewRange()[0]
                    prepend_len = len(df)
                    
                    # Update data
                    self._global_start_index = older_start
                    self.original_data = pd.concat([df, self.original_data], ignore_index=True)
                    
                    # Reprocess
                    self._process_timeframe_optimized()
                    
                    # Update viewport limits with new data
                    self._update_viewport_limits()
                    
                    self._display_ohlc_optimized()
                    
                    # Restore view position adjusted for prepend
                    new_x_min = int(round(x_min + prepend_len))
                    new_x_max = int(round(x_max + prepend_len))
                    new_x_min = max(0, min(new_x_min, len(self.data) - 1))
                    new_x_max = max(0, min(new_x_max, len(self.data)))
                    vb.setXRange(new_x_min, new_x_max, padding=0)
                    self.auto_scale_y_for_visible_data()
                    self._emit_viewport_changed()
                    
        except Exception as e:
            k2_logger.error(f"Failed to fetch older data: {e}", "CHART")
        finally:
            self.is_fetching = False
            self.data_loaded.emit()
            
    # Drawing methods (unchanged from original)
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
        """Change timeframe with optimization and boundary updates"""
        if self.current_timeframe == timeframe:
            return
            
        self.current_timeframe = timeframe
        self.timeframe_changed.emit(timeframe)
        
        if self.original_data is not None:
            self._process_timeframe_optimized()
            self._update_viewport_limits()  # Update boundaries for new timeframe
            self._display_ohlc_optimized()
            self._apply_view_range()
            try:
                allowed = [vr.value for vr in self._compute_allowed_view_ranges()]
                self.allowed_view_ranges_changed.emit(allowed)
            except Exception:
                pass
            
            self.update_axis_geometry()
            QTimer.singleShot(0, self.update_axis_labels_and_grid)
            
            k2_logger.info(f"Timeframe changed to {timeframe}", "CHART")
            
    # Navigation methods with proper boundaries
    def pan_left(self, points: int = 200):
        """Pan left with boundary constraints"""
        if self.data is None:
            return
        vb = self.main_plot.getViewBox()
        x_min, x_max = vb.viewRange()[0]
        x_width = x_max - x_min
        
        # Enforce left boundary - can't go below 0
        new_x_min = max(0, int(round(x_min - points)))
        new_x_max = int(round(new_x_min + x_width))
        
        # Ensure we have valid range
        if new_x_max - new_x_min < 10:  # Minimum 10 points visible
            return
            
        vb.setXRange(new_x_min, new_x_max, padding=0)
        self.auto_scale_y_for_visible_data()
        if self.current_table_name:
            self.viewport_update_timer.start()
        self._emit_viewport_changed()
        
    def pan_right(self, points: int = 200):
        """Pan right with boundary constraints"""
        if self.data is None:
            return

        vb = self.main_plot.getViewBox()
        x_min, x_max = vb.viewRange()[0]
        x_width = x_max - x_min

        max_points = len(self.data)
        # Allow small buffer past end
        buffer = min(50, int(x_width * 0.1))
        
        new_x_min = int(round(min(max_points - x_width + buffer, x_min + points)))
        new_x_max = int(round(new_x_min + x_width))
        
        # Ensure we don't exceed reasonable limits
        new_x_max = min(new_x_max, max_points + buffer)
        
        vb.setXRange(new_x_min, new_x_max, padding=0)
        self.auto_scale_y_for_visible_data()
        self._emit_viewport_changed()
        
    def jump_to_end(self):
        """Jump to latest data"""
        if self.data is None:
            return
        config = TIMEFRAME_CONFIG.get(self.current_timeframe, {})
        points_per_day = config.get('points_per_day', 1)
        points_for_5_days = int(5 * points_per_day)
        points_for_5_days = max(20, min(points_for_5_days, len(self.data)))
        x_max = len(self.data) - 1 + min(50, int(points_for_5_days * 0.1))
        x_min = max(0, len(self.data) - 1 - points_for_5_days)
        self.main_plot.setXRange(int(round(x_min)), int(round(x_max)), padding=0)
        self._auto_scale_y_range(x_min, min(len(self.data) - 1, x_max))
        self._emit_viewport_changed()
        
    def auto_scale_y_for_visible_data(self):
        """Auto-scale Y for visible data"""
        if self.data is None or len(self.data) == 0:
            return

        x_range = self.main_plot.getViewBox().viewRange()[0]
        x_min = int(max(0, round(x_range[0])))
        x_max = int(min(len(self.data) - 1, round(x_range[1])))

        if x_min < len(self.data) and x_max < len(self.data) and x_min <= x_max:
            self._auto_scale_y_range(x_min, x_max)
            
    def reset_zoom(self):
        """Reset zoom"""
        if self.last_5_days_range:
            x_range, y_range = self.last_5_days_range
            self.main_plot.setXRange(x_range[0], x_range[1], padding=0)
            self.main_plot.setYRange(y_range[0], y_range[1], padding=0)
        else:
            self.set_default_view()
            
    def zoom(self, factor):
        """Zoom in/out by factor with boundary constraints"""
        if self.data is None or len(self.data) == 0:
            return
            
        vb = self.main_plot.getViewBox()
        x_range, y_range = vb.viewRange()
        
        # Calculate center points
        x_center = (x_range[0] + x_range[1]) / 2
        y_center = (y_range[0] + y_range[1]) / 2
        
        # Calculate new ranges
        x_width = (x_range[1] - x_range[0]) / factor
        y_height = (y_range[1] - y_range[0]) / factor
        
        # Apply zoom
        new_x_min = int(round(x_center - x_width / 2))
        new_x_max = int(round(x_center + x_width / 2))
        
        # Enforce X boundaries
        new_x_min = max(0, new_x_min)  # Can't go before first data point
        buffer = min(50, int(x_width * 0.1))  # Small buffer
        new_x_max = min(len(self.data) + buffer, new_x_max)
        
        # Minimum 10 points visible
        if new_x_max - new_x_min < 10:
            return
        
        # Calculate Y range with boundaries
        new_y_min = y_center - y_height / 2
        new_y_max = y_center + y_height / 2
        
        # Enforce Y boundaries - prices can't be negative
        new_y_min = max(0, new_y_min)
        new_y_max = max(new_y_min + 0.01, new_y_max)  # Ensure some range
            
        self.main_plot.setXRange(new_x_min, new_x_max, padding=0)
        self.main_plot.setYRange(new_y_min, new_y_max, padding=0)
        self._emit_viewport_changed()
            
    # View Range API
    def set_view_range(self, view_range_key: str):
        """Set the visible time duration window without changing aggregation."""
        try:
            vr = ViewRange(view_range_key) if not isinstance(view_range_key, ViewRange) else view_range_key
        except Exception:
            vr = ViewRange.D5
        self.current_view_range = vr
        self._apply_view_range()
        self.view_range_changed.emit(vr.value)

    def _apply_view_range(self):
        """Apply the current view range to the viewport, enforcing min points."""
        if self.data is None or len(self.data) == 0 or not self.date_column:
            return

        dt_series = self._get_datetime_series()
        if dt_series is None or dt_series.empty:
            return

        end_dt = dt_series.iloc[-1]
        start_dt = self._compute_start_datetime(end_dt, self.current_view_range, dt_series)

        # Map datetimes to indices
        try:
            start_idx = int(dt_series.searchsorted(start_dt, side='left'))
        except Exception:
            start_idx = max(0, len(dt_series) - 200)
        end_idx = len(dt_series) - 1

        # Enforce minimum visible points
        visible = end_idx - start_idx + 1
        if visible < self.min_visible_points:
            start_idx = max(0, end_idx - (self.min_visible_points - 1))

        buffer = min(50, max(5, visible // 10))
        self.main_plot.setXRange(start_idx, end_idx + buffer, padding=0)
        self._auto_scale_y_range(start_idx, end_idx)

    def _get_datetime_series(self):
        try:
            s = pd.to_datetime(self.data[self.date_column], errors='coerce')
            return s.dropna()
        except Exception:
            return None

    def _compute_start_datetime(self, end_dt, view_range: 'ViewRange', dt_series: pd.Series):
        cfg = VIEW_RANGE_CONFIG.get(view_range, {"kind": "timedelta", "days": 5})
        kind = cfg.get("kind")

        if kind == "timedelta":
            delta = timedelta(
                minutes=cfg.get("minutes", 0),
                hours=cfg.get("hours", 0),
                days=cfg.get("days", 0)
            )
            return end_dt - delta

        if kind == "months":
            months = cfg.get("months", 1)
            year = end_dt.year + (end_dt.month - months - 1) // 12
            month = (end_dt.month - months - 1) % 12 + 1
            day = min(getattr(end_dt, 'day', 1), 28)
            try:
                return end_dt.replace(year=year, month=month, day=day)
            except Exception:
                return end_dt - timedelta(days=30 * months)

        if kind == "years":
            years = cfg.get("years", 1)
            try:
                return end_dt.replace(year=end_dt.year - years)
            except Exception:
                return end_dt - timedelta(days=365 * years)

        if kind == "ytd":
            return end_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

        if kind == "all":
            return dt_series.iloc[0]

        return end_dt - timedelta(days=5)

    def _range_is_intraday(self) -> bool:
        """Return True when the selected view range is ≤ 1D."""
        return self.current_view_range in {ViewRange.M15, ViewRange.M30, ViewRange.H1, ViewRange.H4, ViewRange.D1}

    def _get_timeframe_interval_minutes(self, timeframe: str) -> int:
        cfg = TIMEFRAME_CONFIG.get(timeframe or self.current_timeframe)
        return int(cfg.get('interval_minutes', 1440)) if cfg else 1440

    def _get_view_range_duration_minutes(self, view_range: 'ViewRange', dt_series: pd.Series) -> int:
        cfg = VIEW_RANGE_CONFIG.get(view_range, {"kind": "timedelta", "days": 5})
        kind = cfg.get("kind")
        if kind == "timedelta":
            minutes = cfg.get("minutes", 0) + 60 * cfg.get("hours", 0) + 1440 * cfg.get("days", 0)
            return int(minutes)
        if kind == "months":
            return int(30 * 1440 * cfg.get("months", 1))
        if kind == "years":
            return int(365 * 1440 * cfg.get("years", 1))
        if kind == "ytd":
            if dt_series is not None and not dt_series.empty:
                end_dt = dt_series.iloc[-1]
                start_year = end_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                delta = end_dt - start_year
                return int(delta.total_seconds() // 60)
            return 365 * 1440
        if kind == "all":
            if dt_series is not None and not dt_series.empty:
                delta = dt_series.iloc[-1] - dt_series.iloc[0]
                return max(1, int(delta.total_seconds() // 60))
            return 365 * 1440
        return 5 * 1440

    def _compute_allowed_view_ranges(self) -> list:
        if self.data is None or not self.date_column:
            return [vr for vr in ViewRange]
        dt_series = self._get_datetime_series()
        if dt_series is None or dt_series.empty:
            return [vr for vr in ViewRange]
        interval = self._get_timeframe_interval_minutes(self.current_timeframe)
        dmin = 10 * interval
        allowed = []
        for vr in ViewRange:
            dur = self._get_view_range_duration_minutes(vr, dt_series)
            if dur >= dmin or vr == ViewRange.ALL:
                allowed.append(vr)
        return allowed

    def auto_range(self):
        """Auto-range all plots"""
        self.set_default_view()
        for plot in self.indicator_panes.values():
            plot.autoRange()
            
    # Event filter for axis dragging
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

                    new_y_min = y_center - y_range / 2
                    new_y_max = y_center + y_range / 2
                    
                    # Enforce Y boundaries - prices can't be negative
                    new_y_min = max(0, new_y_min)
                    new_y_max = max(new_y_min + 0.01, new_y_max)  # Ensure some range

                    self.main_plot.setYRange(new_y_min, new_y_max, padding=0)

            elif self.dragging_x_axis and self.drag_start_pos:
                delta_x = scene_pos.x() - self.drag_start_pos.x()
                scale_factor = 1.0 - (delta_x / 200.0)

                if self.drag_start_x_range:
                    x_min, x_max = self.drag_start_x_range
                    x_center = (x_min + x_max) / 2
                    x_range = (x_max - x_min) * scale_factor

                    min_points = 10
                    max_points = len(self.data) if self.data is not None else 1000

                    x_range = np.clip(x_range, min_points, max_points * 1.2)

                    new_x_min = int(round(x_center - x_range / 2))
                    new_x_max = int(round(x_center + x_range / 2))

                    # Enforce boundaries
                    new_x_min = max(0, new_x_min)  # Can't go before first data point
                    
                    # Allow small buffer past end
                    if self.data is not None:
                        buffer = min(50, int((new_x_max - new_x_min) * 0.1))
                        new_x_max = min(len(self.data) + buffer, new_x_max)
                    
                    # Adjust if we hit boundaries
                    if new_x_min == 0 and new_x_max > x_range:
                        new_x_max = int(x_range)
                    elif self.data is not None and new_x_max >= len(self.data):
                        new_x_min = max(0, new_x_max - int(x_range))

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
            
    def clear_all(self, preserve_indicator_panes: bool = False):
        """Clear chart elements efficiently.
        If preserve_indicator_panes=True, keep sub-panes across price-data reloads.
        """
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
        
        # Remove indicator panes only if not preserving
        if not preserve_indicator_panes:
            for indicator_name in list(self.indicator_panes.keys()):
                self.remove_indicator_pane(indicator_name)
        
        # Clear caches
        self._format_cache.clear()
        self._label_cache.clear()
        self._viewport_cache = None
        if hasattr(self, '_method_cache'):
            self._method_cache.clear()
            
    def add_indicator_overlay(self, indicator_name: str, series: pd.Series, cid: str | None = None):
        """Render overlay as dotted white on main plot (ignores caller color)."""
        self._suspend_autoload()
        if indicator_name in self.indicator_overlays:
            self.remove_indicator(indicator_name)

        if self.x_values is None:
            self.x_values = np.arange(len(series), dtype=np.float32)

        y_values = series.values.astype(np.float32) if isinstance(series, pd.Series) else np.array(series, dtype=np.float32)

        y_values = np.clip(y_values, -1e6, 1e6)
        y_values[np.isinf(y_values)] = np.nan

        n = int(min(len(self.x_values), len(y_values)))
        if len(y_values) != len(self.x_values):
            try:
                k2_logger.debug(f"Aligning arrays: chart_x={len(self.x_values)}, indicator_y={len(y_values)}, using last {n} points", "CHART")
            except Exception:
                pass
        x_tail = self.x_values[-n:]
        y_tail = y_values[-n:]

        plot_item = OptimizedPlotDataItem(
            x=x_tail,
            y=y_tail,
            pen=pg.mkPen(color='#ffffff', width=2, style=Qt.PenStyle.DotLine),
            connect='finite'
        )

        self.main_plot.addItem(plot_item)
        self.indicator_overlays[indicator_name] = plot_item
        try:
            k2_logger.info(str({
                'event': 'indicator_render_overlay',
                'cid': cid,
                'indicator': indicator_name,
                'x_len': int(len(x_tail)),
                'y_len': int(len(y_tail)),
                'status': 'success'
            }), "INDICATOR")
        except Exception:
            pass
        # Debounce: allow fetches again after a short delay
        try:
            self._mark_ui_mutation()
            QTimer.singleShot(250, self._resume_autoload)
        except Exception:
            self._resume_autoload()

    def remove_indicator(self, indicator_name):
        """Remove indicator"""
        if indicator_name in self.indicator_overlays:
            self.main_plot.removeItem(self.indicator_overlays[indicator_name])
            del self.indicator_overlays[indicator_name]
            k2_logger.info(f"Removed indicator: {indicator_name}", "CHART")
            
    def add_or_update_indicator_pane(self, pane_key: str, series_label: str, series: pd.Series, y_mode: str = "auto", cid: str | None = None):
        """Create/update shared sub-pane with gridlines, guides, ticks, and stable styles."""
        self._suspend_autoload()
        if not hasattr(self, 'pane_series_map'):
            self.pane_series_map = {}
        if not hasattr(self, '_pane_guides'):
            self._pane_guides = {}
        if not hasattr(self, '_series_style'):
            self._series_style = {}

        pane_key = pane_key.upper()

        created = False
        if pane_key not in self.indicator_panes:
            # Use smart axis for VOLUME pane
            if pane_key == "VOLUME":
                pane = pg.PlotWidget(axisItems={'left': SmartUnitAxisItem(orientation='left', mode='volume')})
            else:
                pane = pg.PlotWidget()
            pane.setMaximumHeight(150)
            pane.setBackground('#0a0a0a')
            pane.getAxis('left').setPen(pg.mkPen('#666'))
            pane.getAxis('left').setTextPen(pg.mkPen('#999'))
            pane.getAxis('bottom').setPen(pg.mkPen('#666'))
            pane.getAxis('bottom').setTextPen(pg.mkPen('#999'))
            pane.setXLink(self.main_plot)
            pane.showGrid(x=True, y=True, alpha=0.3)

            # Sub-panes share the main x-axis; hide bottom axis labels in sub-panes
            try:
                pane.getPlotItem().showAxis('bottom', False)
            except Exception:
                pass

            self.indicator_container.addWidget(pane)
            self.indicator_panes[pane_key] = pane
            self.pane_series_map[pane_key] = {}
            self._pane_guides[pane_key] = {}
            created = True

            if pane_key == "RSI":
                oversold = pg.InfiniteLine(pos=30, angle=0, pen=pg.mkPen('#444', width=1, style=Qt.PenStyle.DashLine))
                overbought = pg.InfiniteLine(pos=70, angle=0, pen=pg.mkPen('#444', width=1, style=Qt.PenStyle.DashLine))
                pane.addItem(oversold); pane.addItem(overbought)
                self._pane_guides[pane_key]['oversold'] = oversold
                self._pane_guides[pane_key]['overbought'] = overbought
            elif pane_key == "STOCH":
                oversold = pg.InfiniteLine(pos=20, angle=0, pen=pg.mkPen('#444', width=1, style=Qt.PenStyle.DashLine))
                overbought = pg.InfiniteLine(pos=80, angle=0, pen=pg.mkPen('#444', width=1, style=Qt.PenStyle.DashLine))
                pane.addItem(oversold); pane.addItem(overbought)
                self._pane_guides[pane_key]['oversold'] = oversold
                self._pane_guides[pane_key]['overbought'] = overbought

        pane = self.indicator_panes[pane_key]

        if self.x_values is None:
            self.x_values = np.arange(len(series), dtype=np.float32)

        y_values = series.values.astype(np.float32) if isinstance(series, pd.Series) else np.array(series, dtype=np.float32)
        y_values = np.clip(y_values, -1e6, 1e6)
        y_values[np.isinf(y_values)] = np.nan

        n = int(min(len(self.x_values), len(y_values)))
        if len(y_values) != len(self.x_values):
            try:
                k2_logger.debug(f"Aligning arrays: chart_x={len(self.x_values)}, indicator_y={len(y_values)}, using last {n} points", "CHART")
            except Exception:
                pass
        x_tail = self.x_values[-n:]
        y_tail = y_values[-n:]

        key = (pane_key, series_label)
        if key not in self._series_style:
            palette = [
                pg.mkPen('#ffffff', width=2, style=Qt.PenStyle.SolidLine),
                pg.mkPen('#ffffff', width=2, style=Qt.PenStyle.DashLine),
                pg.mkPen('#ffffff', width=2, style=Qt.PenStyle.DotLine),
                pg.mkPen('#ffffff', width=2, style=Qt.PenStyle.DashDotLine),
            ]
            self._series_style[key] = palette[len(self.pane_series_map[pane_key]) % len(palette)]
        pen = self._series_style[key]

        if series_label in self.pane_series_map[pane_key]:
            item = self.pane_series_map[pane_key][series_label]
            if isinstance(item, OptimizedPlotDataItem):
                item.setData(x=x_tail, y=y_tail, pen=pen)
        else:
            item = OptimizedPlotDataItem(
                x=x_tail,
                y=y_tail,
                pen=pen,
                connect='finite'
            )
            pane.addItem(item)
            self.pane_series_map[pane_key][series_label] = item

        # Cache original series and y-mode so we can reapply after data reloads
        try:
            self._series_data[key] = series.copy() if isinstance(series, pd.Series) else pd.Series(y_values)
        except Exception:
            self._series_data[key] = pd.Series(y_values)
        self._series_y_mode[key] = y_mode

        try:
            k2_logger.info(str({
                'event': 'indicator_render_pane',
                'cid': cid,
                'pane_key': pane_key,
                'series_label': series_label,
                'y_mode': y_mode,
                'action': 'created' if created else 'updated',
                'status': 'success'
            }), "INDICATOR")
        except Exception:
            pass
        # Debounce: allow fetches again after a short delay
        try:
            self._mark_ui_mutation()
            QTimer.singleShot(250, self._resume_autoload)
        except Exception:
            self._resume_autoload()

        y_axis = pane.getAxis('left')
        if y_mode == "bounded_0_100":
            pane.setYRange(0, 100, padding=0)
            if pane_key == "RSI":
                y_axis.setTicks([[(0, '0'), (30, '30'), (70, '70'), (100, '100')]])
            elif pane_key == "STOCH":
                y_axis.setTicks([[(0, '0'), (20, '20'), (80, '80'), (100, '100')]])
            elif pane_key == "MFI":
                y_axis.setTicks([[(0, '0'), (30, '30'), (70, '70'), (100, '100')]])
        elif y_mode == "symmetric_0":
            finite_vals = np.nan_to_num(y_values)
            max_abs = float(np.nanmax(np.abs(finite_vals))) if finite_vals.size else 1.0
            if max_abs < 1.0:
                max_abs = 1.0
            pane.setYRange(-max_abs, max_abs, padding=0.1)
            if pane_key == "CCI":
                y_axis.setTicks([[(-200, '-200'), (-100, '-100'), (0, '0'), (100, '100'), (200, '200')]])
            else:
                y_axis.setTicks([[(-round(max_abs), f"-{int(round(max_abs))}"), (0, '0'), (round(max_abs), f"{int(round(max_abs))}")]])
        elif y_mode == "auto_positive":
            ymin = 0
            ymax = max(np.nanmax(y_values), 1.0)
            pane.setYRange(ymin, ymax, padding=0.1)
            # ticks auto
        
    def remove_indicator_pane(self, indicator_name, cid: str | None = None):
        """Remove indicator pane and clean up guides."""
        if indicator_name in self.indicator_panes:
            widget = self.indicator_panes[indicator_name]
            self.indicator_container.removeWidget(widget)
            widget.deleteLater()
            del self.indicator_panes[indicator_name]
            if hasattr(self, '_pane_guides'):
                self._pane_guides.pop(indicator_name, None)
            # Remove cached series entries for this pane
            try:
                if hasattr(self, '_series_data'):
                    for k in [k for k in list(self._series_data.keys()) if k[0] == indicator_name]:
                        self._series_data.pop(k, None)
                if hasattr(self, '_series_y_mode'):
                    for k in [k for k in list(self._series_y_mode.keys()) if k[0] == indicator_name]:
                        self._series_y_mode.pop(k, None)
            except Exception:
                pass
            try:
                k2_logger.info(str({
                    'event': 'indicator_pane_removed',
                    'cid': cid,
                    'pane_key': indicator_name
                }), "INDICATOR")
            except Exception:
                pass

    def remove_series_from_pane(self, pane_key: str, series_label: str, cid: str | None = None):
        """Remove a series from a pane; deletes the pane if it becomes empty."""
        if not hasattr(self, 'pane_series_map'):
            return
        pane_key = pane_key.upper()
        if pane_key not in self.pane_series_map:
            return
        if series_label in self.pane_series_map[pane_key]:
            item = self.pane_series_map[pane_key][series_label]
            pane = self.indicator_panes.get(pane_key)
            if pane and item:
                pane.removeItem(item)
            del self.pane_series_map[pane_key][series_label]
            if hasattr(self, '_series_style'):
                self._series_style.pop((pane_key, series_label), None)
            # Remove cached data for this series
            if hasattr(self, '_series_data'):
                self._series_data.pop((pane_key, series_label), None)
            if hasattr(self, '_series_y_mode'):
                self._series_y_mode.pop((pane_key, series_label), None)
            try:
                k2_logger.info(str({
                    'event': 'indicator_pane_series_removed',
                    'cid': cid,
                    'pane_key': pane_key,
                    'series_label': series_label
                }), "INDICATOR")
            except Exception:
                pass

        if pane_key in self.pane_series_map and len(self.pane_series_map[pane_key]) == 0:
            self.remove_indicator_pane(pane_key)
            if hasattr(self, '_pane_guides'):
                self._pane_guides.pop(pane_key, None)
            if hasattr(self, '_series_style'):
                to_delete = [k for k in list(self._series_style.keys()) if k[0] == pane_key]
                for k in to_delete:
                    self._series_style.pop(k, None)
        
    def cleanup(self):
        """Cleanup resources"""
        # Stop timers
        for timer_name in ['axis_update_timer', 'range_update_timer', 'viewport_update_timer']:
            if hasattr(self, timer_name):
                timer = getattr(self, timer_name)
                timer.stop()
                
        # Stop debounce timers
        if hasattr(self, '_debounce_timers'):
            for timer in self._debounce_timers.values():
                timer.stop()
            self._debounce_timers.clear()
                
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
        for attr in ['vLine', 'hLine', 'value_label']:
            if hasattr(self, attr):
                item = getattr(self, attr)
                if item and item.scene():
                    self.main_plot.removeItem(item)
                setattr(self, attr, None)
        
        # Clear proxy
        if hasattr(self, 'proxy'):
            self.proxy = None
        
        # Force garbage collection
        gc.collect()