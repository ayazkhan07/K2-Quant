"""
Data Tabs Widget for K2 Quant Analysis

Tabbed data interface with three tabs:
  Tab 1 – Current Data   : Read-only model OHLCV data with indicator columns
  Tab 2 – Forecast Data   : Pre-generated future timestamps with editable projection columns
  Tab 3 – Working Data    : Spreadsheet-style workspace with independent columns
                            (per-model + global scopes)

Save as: k2_quant/pages/analysis/widgets/data_tabs_widget.py
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dt_time, date as dt_date
from typing import Dict, List, Optional, Tuple, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QPushButton, QLabel, QHeaderView, QMessageBox,
    QInputDialog, QMenu, QTableView, QAbstractItemView, QStyledItemDelegate,
)
from PyQt6.QtCore import Qt, pyqtSignal, QAbstractTableModel, QModelIndex
from PyQt6.QtGui import QFont, QColor

from k2_quant.utilities.logger import k2_logger


PRICE_COLUMNS = {'open', 'high', 'low', 'close', 'vwap'}
FORECAST_OHLC = ['Open', 'High', 'Low', 'Close']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_future_timestamps(
    last_date,
    last_time,
    timespan: str,
    frequency: str,
    market_hours_only: bool,
    count: int = 500,
) -> List[Tuple[dt_date, dt_time]]:
    """Return *count* future ``(date, time)`` tuples after the last known bar."""

    freq_num = int(''.join(c for c in str(frequency) if c.isdigit()) or '1')
    ts = (timespan or '').lower()

    if ts.startswith('min'):
        delta = timedelta(minutes=freq_num)
        is_intraday = True
    elif ts.startswith('hour'):
        delta = timedelta(hours=freq_num)
        is_intraday = True
    elif ts.startswith('day'):
        delta = timedelta(days=freq_num)
        is_intraday = False
    else:
        delta = timedelta(minutes=freq_num)
        is_intraday = True

    if isinstance(last_date, str):
        last_date = datetime.strptime(str(last_date), '%Y-%m-%d').date()
    if isinstance(last_time, str):
        parts = str(last_time).split(':')
        h, m = int(parts[0]), int(parts[1])
        s = int(parts[2].split('.')[0]) if len(parts) > 2 else 0
        last_time = dt_time(h, m, s)

    current = datetime.combine(last_date, last_time) + delta
    timestamps: List[Tuple[dt_date, dt_time]] = []
    safety = count * 20

    while len(timestamps) < count and safety > 0:
        safety -= 1

        if market_hours_only:
            # skip weekends
            if current.weekday() >= 5:
                days_ahead = 7 - current.weekday()
                current += timedelta(days=days_ahead)
                if is_intraday:
                    current = current.replace(hour=9, minute=30, second=0, microsecond=0)
                continue
            if is_intraday:
                if current.time() < dt_time(9, 30):
                    current = current.replace(hour=9, minute=30, second=0, microsecond=0)
                if current.time() >= dt_time(16, 0):
                    current += timedelta(days=1)
                    current = current.replace(hour=9, minute=30, second=0, microsecond=0)
                    continue

        timestamps.append((current.date(), current.time()))
        current += delta

    return timestamps


def _format_cell(value, col_name: str, indicator_names: set = None) -> str:
    """Format a value for display in a table cell."""
    if pd.isna(value):
        return ""
    col_lower = str(col_name).lower()
    if col_lower == 'date':
        return str(value)
    if col_lower == 'time':
        return str(value)[:8] if value else ""
    if col_lower == 'volume':
        try:
            return f"{int(float(value)):,}"
        except (ValueError, TypeError):
            return str(value)
    if col_lower in PRICE_COLUMNS:
        try:
            return f"{float(value):.2f}"
        except (ValueError, TypeError):
            return str(value)
    if indicator_names and col_name in indicator_names:
        try:
            if isinstance(value, (int, np.integer)):
                return f"{value:,}"
            if isinstance(value, (float, np.floating)):
                return f"{value:.2f}" if 0.01 < abs(value) < 10000 else f"{value:.4f}"
        except (ValueError, TypeError):
            pass
        return str(value)
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:.2f}" if 0.01 < abs(value) < 10000 else f"{value:.4f}"
    return str(value)


# ---------------------------------------------------------------------------
# Spreadsheet model for Tab 3 (Working Data)
# ---------------------------------------------------------------------------

def _col_letter(index: int) -> str:
    """Convert a 0-based column index to an Excel-style letter (A, B, … Z, AA, AB, …)."""
    result = ""
    while True:
        result = chr(ord('A') + index % 26) + result
        index = index // 26 - 1
        if index < 0:
            break
    return result


def _letter_to_index(letter: str) -> int:
    """Convert an Excel-style column letter to a 0-based index."""
    letter = letter.upper().strip()
    result = 0
    for ch in letter:
        result = result * 26 + (ord(ch) - ord('A') + 1)
    return result - 1


_DEFAULT_VISIBLE_COLS = 30
_DEFAULT_VISIBLE_ROWS = 50


class SpreadsheetModel(QAbstractTableModel):
    """Virtual table model with Excel-style letter columns and numbered rows.

    Row 0 is the header row (column names). Data starts at row 1.
    Columns are addressed by letter (A, B, C …). Each column stores data
    independently and may have a different number of populated rows.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # {col_index: {'name': str, 'values': list[str]}}
        self._columns: Dict[int, Dict[str, Any]] = {}
        self._visible_cols = _DEFAULT_VISIBLE_COLS
        self._visible_rows = _DEFAULT_VISIBLE_ROWS

    # ── QAbstractTableModel interface ─────────────────────────────

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        max_data_rows = 0
        for col_data in self._columns.values():
            max_data_rows = max(max_data_rows, len(col_data['values']))
        return max(self._visible_rows, max_data_rows + 1)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        max_col = -1
        for ci in self._columns:
            max_col = max(max_col, ci)
        return max(self._visible_cols, max_col + 2)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row, col = index.row(), index.column()

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            col_data = self._columns.get(col)
            if col_data is None:
                return ""
            if row == 0:
                return col_data.get('name', '')
            data_row = row - 1
            values = col_data['values']
            if data_row < len(values):
                return str(values[data_row]) if values[data_row] is not None else ""
            return ""

        if role == Qt.ItemDataRole.ForegroundRole:
            if row == 0 and col in self._columns:
                return QColor("#4a9eff")
            return QColor("#e0e0e0")

        if role == Qt.ItemDataRole.FontRole:
            if row == 0 and col in self._columns:
                f = QFont()
                f.setBold(True)
                return f
            return None

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if row == 0:
                return int(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        if role == Qt.ItemDataRole.BackgroundRole:
            if row == 0:
                return QColor("#151515")
            return None

        return None

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False

        row, col = index.row(), index.column()
        text = str(value).strip() if value is not None else ""

        if row == 0:
            if col not in self._columns:
                if not text:
                    return False
                self._columns[col] = {'name': text, 'values': []}
            else:
                self._columns[col]['name'] = text
            self.dataChanged.emit(index, index, [role])
            return True

        data_row = row - 1
        if col not in self._columns:
            if not text:
                return False
            self._columns[col] = {'name': '', 'values': []}

        values = self._columns[col]['values']
        while len(values) <= data_row:
            values.append(None)
        values[data_row] = text if text else None
        self.dataChanged.emit(index, index, [role])
        return True

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return (Qt.ItemFlag.ItemIsEnabled |
                Qt.ItemFlag.ItemIsSelectable |
                Qt.ItemFlag.ItemIsEditable)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return _col_letter(section)
            else:
                return str(section + 1)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignCenter)
        return None

    # ── Column operations ─────────────────────────────────────────

    def set_column(self, col_index: int, name: str, values: list):
        """Write a named column at the given index. Row 0 = name, rows 1+ = values."""
        old_row_count = self.rowCount()
        old_col_count = self.columnCount()

        cleaned = []
        for v in values:
            if v is None:
                cleaned.append(None)
            elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                cleaned.append(None)
            else:
                s = str(v).strip()
                cleaned.append(s if s else None)

        needed_rows = max(self._visible_rows, len(cleaned) + 1)
        max_existing_col = max(self._columns.keys()) if self._columns else -1
        needed_cols = max(self._visible_cols, max(max_existing_col, col_index) + 2)

        if needed_cols > old_col_count:
            self.beginInsertColumns(QModelIndex(), old_col_count, needed_cols - 1)
            self._columns[col_index] = {'name': name, 'values': cleaned}
            self.endInsertColumns()
        else:
            self._columns[col_index] = {'name': name, 'values': cleaned}

        new_row_count = self.rowCount()
        if new_row_count > old_row_count:
            self.beginInsertRows(QModelIndex(), old_row_count, new_row_count - 1)
            self.endInsertRows()

        top_left = self.index(0, col_index)
        bottom_right = self.index(len(cleaned), col_index)
        self.dataChanged.emit(top_left, bottom_right)

    def remove_column_data(self, col_index: int):
        """Clear a column's name and data (the letter remains)."""
        if col_index not in self._columns:
            return
        del self._columns[col_index]
        top_left = self.index(0, col_index)
        bottom_right = self.index(self.rowCount() - 1, col_index)
        self.dataChanged.emit(top_left, bottom_right)

    def find_column_by_name(self, name: str) -> Optional[int]:
        """Return the column index for a given name, or None."""
        for ci, col_data in self._columns.items():
            if col_data.get('name') == name:
                return ci
        return None

    def next_free_column(self) -> int:
        """Return the first column index that has no data."""
        if not self._columns:
            return 0
        return max(self._columns.keys()) + 1

    def get_column_info(self) -> Dict[int, Dict[str, Any]]:
        """Return {col_index: {'name': str, 'letter': str, 'populated_rows': int}}."""
        info = {}
        for ci, col_data in sorted(self._columns.items()):
            populated = sum(
                1 for v in col_data['values']
                if v is not None and str(v).strip() != ''
            )
            info[ci] = {
                'name': col_data['name'],
                'letter': _col_letter(ci),
                'populated_rows': populated,
            }
        return info

    def clear_all(self):
        """Remove all column data and reset to empty grid."""
        self.beginResetModel()
        self._columns.clear()
        self.endResetModel()

    def to_dataframe(self) -> Optional[pd.DataFrame]:
        """Export populated columns as a DataFrame with column names as headers.

        Each column is exported at its full length; shorter columns are
        NOT padded to match the longest. The resulting DataFrame uses
        the column letter + name mapping preserved in metadata.
        """
        if not self._columns:
            return None

        col_frames = {}
        col_letters = {}
        for ci in sorted(self._columns.keys()):
            col_data = self._columns[ci]
            name = col_data['name'] or _col_letter(ci)
            values = col_data['values']
            populated = [v if v is not None else "" for v in values]
            while populated and populated[-1] == "":
                populated.pop()
            if populated:
                col_frames[name] = populated
                col_letters[name] = _col_letter(ci)

        if not col_frames:
            return None

        max_len = max(len(v) for v in col_frames.values())
        for name in col_frames:
            vals = col_frames[name]
            col_frames[name] = vals + [""] * (max_len - len(vals))

        df = pd.DataFrame(col_frames)
        df.attrs['_col_letters'] = col_letters
        return df

    def from_serialised(self, state: Dict):
        """Load from serialised state: {col_index: {name, values}}."""
        self.beginResetModel()
        self._columns.clear()
        for ci_str, col_data in state.items():
            ci = int(ci_str)
            self._columns[ci] = {
                'name': col_data.get('name', ''),
                'values': col_data.get('values', []),
            }
        self.endResetModel()

    def to_serialised(self) -> Optional[Dict]:
        """Serialise to a dict preserving column letter positions."""
        if not self._columns:
            return None
        result = {}
        for ci, col_data in self._columns.items():
            values = col_data['values']
            while values and (values[-1] is None or str(values[-1]).strip() == ''):
                values = values[:-1]
            if col_data['name'] or values:
                result[str(ci)] = {
                    'name': col_data['name'],
                    'values': [v if v is not None else '' for v in values],
                }
        return result if result else None


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class DataTabsWidget(QWidget):
    """Tabbed data interface: Current Data | Forecast Data | Working Data."""

    forecast_apply = pyqtSignal(dict)

    FORECAST_ROW_COUNT = 500

    def __init__(self, parent=None):
        super().__init__(parent)

        self._forecast_sets: int = 0
        self._forecast_timestamps: List[Tuple[dt_date, dt_time]] = []
        self._model_table_name: Optional[str] = None
        self._timespan = ''
        self._frequency = '1'
        self._market_hours_only = False

        self._init_ui()
        self._apply_styling()

    # ==================================================================
    # UI construction
    # ==================================================================

    def _init_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setLayout(root)

        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        root.addWidget(self.tab_widget)

        self._build_current_tab()
        self._build_forecast_tab()
        self._build_working_tab()

    # -- Tab 1 ----------------------------------------------------------

    def _build_current_tab(self):
        self.current_table = QTableWidget()
        self.current_table.setAlternatingRowColors(True)
        self.current_table.horizontalHeader().setStretchLastSection(False)
        self.current_table.setSortingEnabled(True)
        self.current_table.verticalHeader().setVisible(False)
        self.current_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.current_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.current_table.setSelectionMode(QTableWidget.SelectionMode.ContiguousSelection)
        self.current_table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.current_table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self._apply_header_font(self.current_table)
        self.tab_widget.addTab(self.current_table, "Current Data")

    # -- Tab 2 ----------------------------------------------------------

    def _build_forecast_tab(self):
        container = QWidget()
        vl = QVBoxLayout()
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)
        container.setLayout(vl)

        # toolbar
        toolbar = QWidget()
        toolbar.setFixedHeight(35)
        toolbar.setObjectName("forecastToolbar")
        tl = QHBoxLayout()
        tl.setContentsMargins(8, 0, 8, 0)
        toolbar.setLayout(tl)

        self.forecast_info = QLabel("No model loaded")
        self.forecast_info.setStyleSheet("color:#666;font-size:12px;")
        tl.addWidget(self.forecast_info)
        tl.addStretch()

        self.btn_add_set = QPushButton("+ Forecast Set")
        self.btn_add_set.setToolTip("Add Open/High/Low/Close projection columns")
        self.btn_add_set.clicked.connect(self.add_forecast_set)
        tl.addWidget(self.btn_add_set)

        btn_apply = QPushButton("Apply to Chart")
        btn_apply.setToolTip("Push populated forecasts to chart as dashed lines")
        btn_apply.clicked.connect(self._emit_forecast)
        tl.addWidget(btn_apply)

        btn_clear = QPushButton("Clear Forecasts")
        btn_clear.setToolTip("Erase forecast values; timestamps preserved")
        btn_clear.clicked.connect(self.clear_forecast_values)
        tl.addWidget(btn_clear)

        vl.addWidget(toolbar)

        self.forecast_table = QTableWidget()
        self.forecast_table.setAlternatingRowColors(True)
        self.forecast_table.horizontalHeader().setStretchLastSection(False)
        self.forecast_table.verticalHeader().setVisible(False)
        self.forecast_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.forecast_table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.forecast_table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self._apply_header_font(self.forecast_table)
        vl.addWidget(self.forecast_table)

        self.tab_widget.addTab(container, "Forecast Data")

    # -- Tab 3 ----------------------------------------------------------

    def _build_working_tab(self):
        container = QWidget()
        vl = QVBoxLayout()
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)
        container.setLayout(vl)

        toolbar = QWidget()
        toolbar.setFixedHeight(35)
        toolbar.setObjectName("workingToolbar")
        tl = QHBoxLayout()
        tl.setContentsMargins(8, 0, 8, 0)
        toolbar.setLayout(tl)

        self.working_info = QLabel("Working Data")
        self.working_info.setStyleSheet("color:#666;font-size:12px;")
        tl.addWidget(self.working_info)
        tl.addStretch()

        btn_col = QPushButton("+ Column")
        btn_col.setToolTip("Add a named column to the active workspace")
        btn_col.clicked.connect(self._add_working_column_dialog)
        tl.addWidget(btn_col)

        btn_clr_m = QPushButton("Clear Model")
        btn_clr_m.clicked.connect(lambda: self._clear_working('model'))
        tl.addWidget(btn_clr_m)

        btn_clr_g = QPushButton("Clear Global")
        btn_clr_g.clicked.connect(lambda: self._clear_working('global'))
        tl.addWidget(btn_clr_g)

        vl.addWidget(toolbar)

        self.working_tabs = QTabWidget()
        self.working_tabs.setDocumentMode(True)

        self.working_model_model = SpreadsheetModel()
        self.working_model_table = self._create_spreadsheet_view(
            self.working_model_model, 'model')
        self.working_tabs.addTab(self.working_model_table, "Model Workspace")

        self.working_global_model = SpreadsheetModel()
        self.working_global_table = self._create_spreadsheet_view(
            self.working_global_model, 'global')
        self.working_tabs.addTab(self.working_global_table, "Global Workspace")

        vl.addWidget(self.working_tabs)
        self.tab_widget.addTab(container, "Working Data")

    def _create_spreadsheet_view(self, model: SpreadsheetModel,
                                  scope: str) -> QTableView:
        """Create a QTableView configured as a spreadsheet grid."""
        view = QTableView()
        view.setModel(model)
        view.setAlternatingRowColors(True)
        view.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

        view.verticalHeader().setVisible(True)
        view.verticalHeader().setDefaultSectionSize(24)
        view.verticalHeader().setMinimumSectionSize(20)
        view.verticalHeader().setFixedWidth(30)

        view.horizontalHeader().setDefaultSectionSize(90)
        view.horizontalHeader().setMinimumSectionSize(50)
        view.horizontalHeader().setStretchLastSection(False)
        hdr_font = view.horizontalHeader().font()
        hdr_font.setCapitalization(QFont.Capitalization.SmallCaps)
        view.horizontalHeader().setFont(hdr_font)

        view.horizontalHeader().setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        view.horizontalHeader().customContextMenuRequested.connect(
            lambda pos, v=view, s=scope: self._on_spreadsheet_header_menu(pos, v, s))

        return view

    # -- shared helpers -------------------------------------------------

    @staticmethod
    def _apply_header_font(table: QTableWidget):
        hdr = table.horizontalHeader()
        f = hdr.font()
        f.setCapitalization(QFont.Capitalization.SmallCaps)
        hdr.setFont(f)

    @staticmethod
    def _setup_editable_table(table: QTableWidget):
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setStretchLastSection(False)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        table.setColumnCount(0)
        table.setRowCount(0)

    # -- empty-column padding --------------------------------------------

    @staticmethod
    def _data_col_count(table: QTableWidget) -> int:
        """Return the number of *real* data columns (excludes visual padding)."""
        v = table.property("_data_cols")
        return v if v is not None and v > 0 else table.columnCount()

    def _pad_empty_columns(self, table: QTableWidget):
        """Append empty grid columns so the table fills the viewport width."""
        data_cols = table.property("_data_cols")
        if not data_cols:
            return

        table.setColumnCount(data_cols)

        viewport_w = table.viewport().width()
        if viewport_w <= 0:
            viewport_w = max(self.width() - 30, 800)

        used_w = sum(table.columnWidth(c) for c in range(data_cols))
        remaining = viewport_w - used_w

        PAD_W = 90
        pad_count = max(1, (remaining // PAD_W) + 2)

        table.setColumnCount(data_cols + pad_count)
        for c in range(data_cols, data_cols + pad_count):
            table.setHorizontalHeaderItem(c, QTableWidgetItem(""))
            table.setColumnWidth(c, PAD_W)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for table in [self.current_table, self.forecast_table]:
            if table.property("_data_cols"):
                self._pad_empty_columns(table)

    # ==================================================================
    # Tab 1 – Current Data
    # ==================================================================

    def load_current_data(self, df: pd.DataFrame, active_indicators: Optional[Dict] = None):
        """Populate the Current Data tab with *df* (already merged with indicators)."""
        if df is None or df.empty:
            return

        indicator_names = set(active_indicators.keys()) if active_indicators else set()
        table = self.current_table
        table.setSortingEnabled(False)
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))

        headers = []
        for col in df.columns:
            cs = str(col)
            headers.append(cs.capitalize() if cs.lower() in (
                'open', 'high', 'low', 'close', 'volume', 'vwap', 'date', 'time') else cs)
        table.setHorizontalHeaderLabels(headers)

        for r in range(len(df)):
            for c in range(len(df.columns)):
                value = df.iloc[r, c]
                col_name = df.columns[c]
                text = _format_cell(value, col_name, indicator_names)
                item = QTableWidgetItem(text)
                if str(col_name).lower() in ('date', 'time'):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(r, c, item)

        table.resizeColumnsToContents()
        self._enforce_min_widths(table, headers)
        table.setProperty("_data_cols", len(df.columns))
        self._pad_empty_columns(table)
        table.setSortingEnabled(True)
        k2_logger.info(f"Tab 1 loaded: {len(df)} rows, {len(df.columns)} cols", "DATA_TABS")

    @staticmethod
    def _enforce_min_widths(table: QTableWidget, headers: List[str]):
        for i in range(table.columnCount()):
            min_w = 80
            if i < len(headers):
                h = headers[i].lower()
                if h == 'date':
                    min_w = 100
                elif h == 'time':
                    min_w = 90
                elif h == 'volume':
                    min_w = 110
            if table.columnWidth(i) < min_w:
                table.setColumnWidth(i, min_w)

    # ==================================================================
    # Tab 2 – Forecast Data
    # ==================================================================

    def setup_forecast(
        self,
        last_date,
        last_time,
        timespan: str,
        frequency: str,
        market_hours_only: bool,
    ):
        """Generate future timestamps and rebuild the forecast grid."""
        self._timespan = timespan
        self._frequency = frequency
        self._market_hours_only = market_hours_only
        self._forecast_sets = 0
        self._forecast_timestamps = generate_future_timestamps(
            last_date, last_time, timespan, frequency,
            market_hours_only, count=self.FORECAST_ROW_COUNT,
        )
        self._rebuild_forecast_table()
        self.forecast_info.setText(
            f"{len(self._forecast_timestamps)} future timestamps generated"
        )
        k2_logger.info(
            f"Forecast grid: {len(self._forecast_timestamps)} ts, "
            f"span={timespan}, freq={frequency}, mkt={market_hours_only}",
            "DATA_TABS",
        )

    def add_forecast_set(self):
        """Append Open_Px / High_Px / Low_Px / Close_Px columns."""
        if not self._forecast_timestamps:
            QMessageBox.information(
                self, "No Model",
                "Load a model first so timestamps can be generated.")
            return
        self._forecast_sets += 1
        self._rebuild_forecast_table()
        k2_logger.info(f"Forecast set P{self._forecast_sets} added", "DATA_TABS")

    def _rebuild_forecast_table(self):
        table = self.forecast_table
        ts = self._forecast_timestamps
        cols = ['Date', 'Time']
        for s in range(1, self._forecast_sets + 1):
            for base in FORECAST_OHLC:
                cols.append(f"{base}_P{s}")

        table.blockSignals(True)
        table.setSortingEnabled(False)
        table.setRowCount(len(ts))
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)

        for r, (d, t) in enumerate(ts):
            d_item = QTableWidgetItem(str(d))
            d_item.setFlags(d_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            d_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(r, 0, d_item)

            t_item = QTableWidgetItem(str(t)[:8])
            t_item.setFlags(t_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            t_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(r, 1, t_item)

            for c in range(2, len(cols)):
                item = QTableWidgetItem("")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(r, c, item)

        table.resizeColumnsToContents()
        for i in range(len(cols)):
            if table.columnWidth(i) < 90:
                table.setColumnWidth(i, 90)
        table.setProperty("_data_cols", len(cols))
        self._pad_empty_columns(table)
        table.blockSignals(False)

    def clear_forecast_values(self):
        """Erase editable cells; keep timestamps intact."""
        table = self.forecast_table
        for r in range(table.rowCount()):
            for c in range(2, self._data_col_count(table)):
                item = table.item(r, c)
                if item:
                    item.setText("")
        k2_logger.info("Forecast values cleared", "DATA_TABS")

    def set_forecast_values(self, column_name: str, values: list):
        """Programmatically write values into a forecast column."""
        table = self.forecast_table
        col_idx = None
        for c in range(self._data_col_count(table)):
            h = table.horizontalHeaderItem(c)
            if h and h.text() == column_name:
                col_idx = c
                break
        if col_idx is None:
            k2_logger.warning(f"Forecast column '{column_name}' not found", "DATA_TABS")
            return
        for r, val in enumerate(values):
            if r >= table.rowCount():
                break
            text = ""
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                try:
                    text = f"{float(val):.2f}"
                except (ValueError, TypeError):
                    text = str(val)
            item = table.item(r, col_idx)
            if item:
                item.setText(text)

    def get_forecast_data(self) -> Dict[int, pd.DataFrame]:
        """Return ``{set_index: DataFrame}`` for populated forecast rows only."""
        result: Dict[int, pd.DataFrame] = {}
        table = self.forecast_table

        for s in range(1, self._forecast_sets + 1):
            target_cols = [f"{base}_P{s}" for base in FORECAST_OHLC]
            col_map: Dict[str, int] = {}
            for c in range(self._data_col_count(table)):
                h = table.horizontalHeaderItem(c)
                if h and h.text() in target_cols:
                    col_map[h.text()] = c
            if len(col_map) != 4:
                continue

            rows = []
            for r in range(table.rowCount()):
                values: Dict[str, Optional[float]] = {}
                has_any = False
                for cname, ci in col_map.items():
                    item = table.item(r, ci)
                    txt = item.text().strip() if item else ""
                    if txt:
                        try:
                            values[cname] = float(txt)
                            has_any = True
                        except ValueError:
                            values[cname] = None
                    else:
                        values[cname] = None
                if has_any:
                    d_item = table.item(r, 0)
                    t_item = table.item(r, 1)
                    row = {
                        'Date': d_item.text() if d_item else '',
                        'Time': t_item.text() if t_item else '',
                    }
                    row.update(values)
                    rows.append(row)
            if rows:
                result[s] = pd.DataFrame(rows)

        return result

    def _emit_forecast(self):
        data = self.get_forecast_data()
        if not data:
            QMessageBox.information(
                self, "No Data",
                "No forecast values to apply.\n"
                "Add a forecast set and populate values first.")
            return
        self.forecast_apply.emit(data)
        k2_logger.info(f"Forecast applied: {len(data)} set(s)", "DATA_TABS")

    # ==================================================================
    # Tab 3 – Working Data (spreadsheet grid)
    # ==================================================================

    def _spreadsheet_model(self, scope: str) -> SpreadsheetModel:
        return self.working_model_model if scope == 'model' else self.working_global_model

    def _spreadsheet_view(self, scope: str) -> QTableView:
        return self.working_model_table if scope == 'model' else self.working_global_table

    def set_working_data(self, scope: str, df: pd.DataFrame):
        """Replace the working grid for *scope* with *df*."""
        model = self._spreadsheet_model(scope)
        if df is None or df.empty:
            model.clear_all()
            self._update_working_info()
            return

        model.beginResetModel()
        model._columns.clear()
        col_letters = getattr(df, 'attrs', {}).get('_col_letters', {})
        for i, col_name in enumerate(df.columns):
            letter = col_letters.get(col_name)
            col_idx = _letter_to_index(letter) if letter else i
            values = []
            for v in df[col_name]:
                s = "" if pd.isna(v) else str(v)
                values.append(s if s.strip() else None)
            while values and values[-1] is None:
                values.pop()
            if values or col_name:
                model._columns[col_idx] = {'name': str(col_name), 'values': values}
        model.endResetModel()
        self._update_working_info()

    def get_working_data(self, scope: str) -> Optional[pd.DataFrame]:
        """Read the working grid as a DataFrame (None if empty)."""
        model = self._spreadsheet_model(scope)
        return model.to_dataframe()

    def add_working_column(self, scope: str, name: str, values: list,
                           column: Optional[str] = None):
        """Add or replace a named column in the working grid.

        Parameters
        ----------
        scope : 'model' or 'global'
        name : column name (displayed in row 1 of the grid)
        values : data values (placed starting at row 2)
        column : optional Excel-style letter (e.g. 'A', 'H', 'AA') for explicit
                 placement. If None, reuses the existing column with the same name
                 or appends to the next free column.
        """
        model = self._spreadsheet_model(scope)

        if column is not None:
            col_idx = _letter_to_index(column)
        else:
            existing = model.find_column_by_name(name)
            col_idx = existing if existing is not None else model.next_free_column()

        model.set_column(col_idx, name, values)
        self._update_working_info()
        letter = _col_letter(col_idx)
        k2_logger.info(
            f"Working column '{name}' -> {scope} col {letter} ({len(values)} vals)",
            "DATA_TABS")

    def _add_working_column_dialog(self):
        scope = 'model' if self.working_tabs.currentIndex() == 0 else 'global'
        name, ok = QInputDialog.getText(self, "Add Column", "Column name:")
        if ok and name.strip():
            self.add_working_column(scope, name.strip(), [])

    def delete_working_column(self, scope: str, column_name: str) -> bool:
        """Remove a named column from the working grid. Returns True on success."""
        model = self._spreadsheet_model(scope)
        col_idx = model.find_column_by_name(column_name)

        if col_idx is None:
            k2_logger.warning(
                f"Cannot delete '{column_name}' from working [{scope}]: not found",
                "DATA_TABS")
            return False

        model.remove_column_data(col_idx)
        self._update_working_info()
        k2_logger.info(
            f"Deleted working column '{column_name}' from [{scope}]", "DATA_TABS")
        return True

    def _on_spreadsheet_header_menu(self, pos, view: QTableView, scope: str):
        """Right-click context menu on spreadsheet column header."""
        logical_idx = view.horizontalHeader().logicalIndexAt(pos)
        if logical_idx < 0:
            return

        model = self._spreadsheet_model(scope)
        col_data = model._columns.get(logical_idx)
        if col_data is None:
            return

        col_name = col_data.get('name', '')
        letter = _col_letter(logical_idx)
        display = f"{letter}: \"{col_name}\"" if col_name else letter

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border: 1px solid #2a2a2a;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #2a3f5f;
            }
        """)

        delete_action = menu.addAction(f"Delete {display}")
        action = menu.exec(view.horizontalHeader().mapToGlobal(pos))

        if action == delete_action:
            reply = QMessageBox.question(
                self, "Delete Column",
                f"Delete column {display} from {scope} workspace?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                if col_name:
                    self.delete_working_column(scope, col_name)
                else:
                    model.remove_column_data(logical_idx)
                    self._update_working_info()

    def _clear_working(self, scope: str):
        model = self._spreadsheet_model(scope)
        model.clear_all()
        self._update_working_info()
        k2_logger.info(f"Working data cleared ({scope})", "DATA_TABS")

    def _update_working_info(self):
        parts = []
        for label, mdl in [("Model", self.working_model_model),
                           ("Global", self.working_global_model)]:
            info = mdl.get_column_info()
            if info:
                col_count = len(info)
                max_rows = max(
                    (ci['populated_rows'] for ci in info.values()), default=0)
                parts.append(f"{label}: {col_count} cols, {max_rows} max rows")
        self.working_info.setText(" | ".join(parts) if parts else "Working Data")

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def set_model_context(self, table_name: Optional[str]):
        self._model_table_name = table_name

    def clear_all(self):
        """Reset Tabs 1 & 2 and the model workspace. Global workspace preserved."""
        self.current_table.setRowCount(0)
        self.current_table.setColumnCount(0)
        self.current_table.setProperty("_data_cols", None)

        self._forecast_sets = 0
        self._forecast_timestamps = []
        self.forecast_table.setRowCount(0)
        self.forecast_table.setColumnCount(0)
        self.forecast_table.setProperty("_data_cols", None)
        self.forecast_info.setText("No model loaded")

        self.working_model_model.clear_all()
        self._update_working_info()
        self._model_table_name = None

    def cleanup(self):
        self.clear_all()
        self.working_global_model.clear_all()
        self._update_working_info()

    # ==================================================================
    # Persistence  (serialise ↔ dict  for DB storage)
    # ==================================================================

    def serialise_forecast(self) -> Optional[Dict]:
        """Return forecast tab state as a JSON-serialisable dict (or None)."""
        if not self._forecast_timestamps or self._forecast_sets == 0:
            return None
        table = self.forecast_table
        data_cols = self._data_col_count(table)
        columns = [
            (table.horizontalHeaderItem(c).text()
             if table.horizontalHeaderItem(c) else f"Col_{c}")
            for c in range(data_cols)
        ]
        data = []
        for r in range(table.rowCount()):
            row = []
            for c in range(data_cols):
                item = table.item(r, c)
                row.append(item.text() if item else "")
            data.append(row)
        return {
            'columns': columns,
            'data': data,
            'forecast_sets': self._forecast_sets,
            'timespan': self._timespan,
            'frequency': self._frequency,
            'market_hours_only': self._market_hours_only,
        }

    def restore_forecast(self, state: Dict):
        """Rebuild forecast tab from a previously serialised dict."""
        if not state:
            return
        columns = state.get('columns', [])
        data = state.get('data', [])
        self._forecast_sets = state.get('forecast_sets', 0)
        self._timespan = state.get('timespan', '')
        self._frequency = state.get('frequency', '1')
        self._market_hours_only = state.get('market_hours_only', False)

        table = self.forecast_table
        table.blockSignals(True)
        table.setSortingEnabled(False)
        table.setRowCount(len(data))
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)

        self._forecast_timestamps = []
        for r, row in enumerate(data):
            for c, val in enumerate(row):
                item = QTableWidgetItem(val)
                cname = columns[c] if c < len(columns) else ''
                if cname in ('Date', 'Time'):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(r, c, item)
            if len(row) >= 2:
                try:
                    d = datetime.strptime(row[0], '%Y-%m-%d').date()
                    parts = row[1].split(':')
                    t = dt_time(int(parts[0]), int(parts[1]),
                                int(parts[2].split('.')[0]) if len(parts) > 2 else 0)
                    self._forecast_timestamps.append((d, t))
                except (ValueError, IndexError):
                    pass

        table.resizeColumnsToContents()
        for i in range(len(columns)):
            if table.columnWidth(i) < 90:
                table.setColumnWidth(i, 90)
        table.setProperty("_data_cols", len(columns))
        self._pad_empty_columns(table)
        table.blockSignals(False)

        self.forecast_info.setText(
            f"{len(self._forecast_timestamps)} timestamps, "
            f"{self._forecast_sets} forecast set(s)")
        k2_logger.info(
            f"Forecast restored: {len(data)} rows, {self._forecast_sets} sets",
            "DATA_TABS")

    def serialise_working_grid(self, scope: str) -> Optional[Dict]:
        """Serialise the spreadsheet model, preserving column letter positions."""
        model = self._spreadsheet_model(scope)
        return model.to_serialised()

    def restore_working_grid(self, scope: str, state: Optional[Dict]):
        """Restore the spreadsheet model from a previously serialised dict."""
        model = self._spreadsheet_model(scope)
        if not state:
            model.clear_all()
        else:
            model.from_serialised(state)
        self._update_working_info()

    @staticmethod
    def serialise_working(df: Optional[pd.DataFrame]) -> Optional[Dict]:
        """Convert a working-data DataFrame to a JSON-serialisable dict.

        Preserves column letter positions if _col_letters metadata is present.
        """
        if df is None or df.empty:
            return None
        col_letters = getattr(df, 'attrs', {}).get('_col_letters', {})
        if col_letters:
            result = {}
            for col_name in df.columns:
                letter = col_letters.get(col_name)
                ci = _letter_to_index(letter) if letter else None
                values = df[col_name].fillna("").tolist()
                while values and str(values[-1]).strip() == '':
                    values.pop()
                if ci is not None and (col_name or values):
                    result[str(ci)] = {
                        'name': str(col_name),
                        'values': values,
                    }
            return result if result else None
        return {
            'columns': [str(c) for c in df.columns],
            'data': df.fillna("").values.tolist(),
        }

    @staticmethod
    def deserialise_working(state: Optional[Dict]) -> Optional[pd.DataFrame]:
        """Reconstruct a working-data DataFrame from a serialised dict.

        Handles both the new grid format (keyed by column index) and the
        legacy flat format (columns + data lists).
        """
        if not state:
            return None

        if 'columns' in state and 'data' in state:
            cols = state.get('columns', [])
            data = state.get('data', [])
            if not cols:
                return None
            return pd.DataFrame(data, columns=cols)

        col_frames = {}
        col_letters = {}
        for ci_str, col_data in state.items():
            try:
                ci = int(ci_str)
            except (ValueError, TypeError):
                continue
            name = col_data.get('name', _col_letter(ci))
            values = col_data.get('values', [])
            col_frames[name] = values
            col_letters[name] = _col_letter(ci)

        if not col_frames:
            return None

        max_len = max(len(v) for v in col_frames.values())
        for name in col_frames:
            vals = col_frames[name]
            col_frames[name] = vals + [""] * (max_len - len(vals))

        df = pd.DataFrame(col_frames)
        df.attrs['_col_letters'] = col_letters
        return df

    # ==================================================================
    # Styling
    # ==================================================================

    def _apply_styling(self):
        self.setStyleSheet("""
            /* ── Tab bar ─────────────────────────────── */
            QTabWidget::pane {
                border: none;
                background-color: #0a0a0a;
            }
            QTabBar::tab {
                background-color: #111111;
                color: #888;
                padding: 8px 20px;
                border: none;
                border-bottom: 2px solid transparent;
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 0.5px;
            }
            QTabBar::tab:selected {
                background-color: #0a0a0a;
                color: #fff;
                border-bottom: 2px solid #4a9eff;
            }
            QTabBar::tab:hover:!selected {
                background-color: #1a1a1a;
                color: #ccc;
            }

            /* ── Toolbars ────────────────────────────── */
            #forecastToolbar, #workingToolbar {
                background-color: #0f0f0f;
                border-bottom: 1px solid #1a1a1a;
            }
            QPushButton {
                background-color: #1a1a1a;
                color: #999;
                border: 1px solid #2a2a2a;
                padding: 5px 15px;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #2a2a2a;
                color: #fff;
            }

            /* ── Tables ──────────────────────────────── */
            QTableWidget, QTableView {
                background-color: #0a0a0a;
                alternate-background-color: #111111;
                gridline-color: #1a1a1a;
                color: #e0e0e0;
                font-family: 'Inter', 'Segoe UI', 'Arial', sans-serif;
                font-size: 13px;
                border: none;
                outline: none;
            }
            QTableWidget::item, QTableView::item {
                padding: 5px 8px;
                border: none;
                border-bottom: 1px solid rgba(42, 42, 42, 0.3);
            }
            QTableWidget::item:selected, QTableView::item:selected {
                background-color: #2a3f5f;
                color: #ffffff;
            }
            QTableWidget::item:hover, QTableView::item:hover {
                background-color: #1e1e1e;
            }
            QHeaderView::section {
                background-color: #0a0a0a;
                color: #888;
                padding: 4px 2px;
                border: none;
                border-bottom: 2px solid #2a2a2a;
                font-weight: 600;
                font-size: 11px;
                letter-spacing: 0.5px;
            }
            QHeaderView::section:hover {
                background-color: #1a1a1a;
                color: #aaa;
            }

            /* ── Scrollbars ──────────────────────────── */
            QScrollBar:vertical {
                background: #0a0a0a; width: 10px; border: none;
            }
            QScrollBar::handle:vertical {
                background: #2a2a2a; border-radius: 5px; min-height: 20px;
            }
            QScrollBar::handle:vertical:hover { background: #3a3a3a; }
            QScrollBar:horizontal {
                background: #0a0a0a; height: 10px; border: none;
            }
            QScrollBar::handle:horizontal {
                background: #2a2a2a; border-radius: 5px; min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover { background: #3a3a3a; }
            QScrollBar::add-line, QScrollBar::sub-line { background: none; border: none; }
            QScrollBar::add-page, QScrollBar::sub-page { background: none; }
        """)
