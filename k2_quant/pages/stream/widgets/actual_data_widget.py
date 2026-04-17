"""
Actual Data Widget — table UI for the ACTUAL DATA tab in Stream windows.

Mirrors the forecast table structure (index, Date, Time, OHLCV) but is
populated by live market bars from the WebSocket + reconciliation backfill.

The last row is the "forming bar" — highlighted and updated in real-time
as partial aggregation progresses.
"""

from datetime import date, time as dt_time
from typing import List, Dict, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel, QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont

from k2_quant.utilities.logger import k2_logger

_COLUMNS = ["#", "Date", "Time", "Open", "High", "Low", "Close", "Volume", "VWAP"]
_COL_IDX = {name: i for i, name in enumerate(_COLUMNS)}

_FORMING_BG = QColor(20, 40, 20)
_NORMAL_FG = QColor(200, 200, 200)
_FORMING_FG = QColor(100, 255, 100)


class ActualDataWidget(QWidget):
    """Table showing real-time actual market bars for one stream window."""

    bar_count_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._row_count = 0
        self._forming_row: Optional[int] = None
        self._base_index = 0
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 4, 8, 4)

        self._status_label = QLabel("No data")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        toolbar.addWidget(self._status_label)

        toolbar.addStretch()

        self._bar_progress = QLabel("")
        self._bar_progress.setStyleSheet("color: #4a9eff; font-size: 11px;")
        toolbar.addWidget(self._bar_progress)

        toolbar_widget = QWidget()
        toolbar_widget.setLayout(toolbar)
        toolbar_widget.setStyleSheet("background: #0a0a0a;")
        layout.addWidget(toolbar_widget)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(False)

        header = self.table.horizontalHeader()
        header.setDefaultSectionSize(90)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)

        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #0a0a0a;
                color: #c8c8c8;
                gridline-color: #1a1a1a;
                border: none;
                font-size: 11px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QTableWidget::item { padding: 2px 6px; }
            QTableWidget::item:selected {
                background-color: #1a2a3a;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #111;
                color: #aaa;
                border: none;
                border-bottom: 1px solid #222;
                padding: 4px 6px;
                font-size: 11px;
                font-weight: 600;
            }
        """)

        layout.addWidget(self.table)

    def set_base_index(self, base: int):
        """Set the starting index so the # column continues from the model data."""
        self._base_index = base

    def load_bars(self, bars: list):
        """Bulk-load bars (e.g. from reconciliation or DB restore).
        Each bar is a tuple (date, time, O, H, L, C, V, VWAP) or a dict."""
        self.table.setRowCount(0)
        self._row_count = 0
        self._forming_row = None

        for bar in bars:
            self._append_bar_row(bar, forming=False)

        self._update_status()
        self._scroll_to_bottom()

    def append_completed_bar(self, bar: dict):
        """Add a fully aggregated bar.  If a forming row exists, finalize it first."""
        if self._forming_row is not None:
            self._finalize_forming_row(bar)
        else:
            self._append_bar_row(bar, forming=False)
        self._update_status()
        self._scroll_to_bottom()

    def update_forming_bar(self, bar: dict):
        """Update (or create) the forming bar at the bottom of the table."""
        if self._forming_row is None:
            self._append_bar_row(bar, forming=True)
            self._forming_row = self._row_count - 1
        else:
            self._update_row(self._forming_row, bar, forming=True)

        progress = bar.get("bars_accumulated", 0)
        required = bar.get("bars_required", 1)
        self._bar_progress.setText(f"Forming: {progress}/{required} min")
        self._scroll_to_bottom()

    def clear(self):
        self.table.setRowCount(0)
        self._row_count = 0
        self._forming_row = None
        self._bar_progress.setText("")
        self._update_status()

    def get_bar_count(self) -> int:
        """Number of completed bars (excludes forming row)."""
        count = self._row_count
        if self._forming_row is not None:
            count -= 1
        return max(0, count)

    def _append_bar_row(self, bar, forming: bool):
        row_idx = self._row_count
        self.table.setRowCount(row_idx + 1)
        self._row_count = row_idx + 1

        self._update_row(row_idx, bar, forming)

    def _update_row(self, row_idx: int, bar, forming: bool):
        display_num = self._base_index + row_idx + 1
        if isinstance(bar, (tuple, list)):
            d, t, o, h, l_, c, v, vw = bar[0], bar[1], bar[2], bar[3], bar[4], bar[5], bar[6], bar[7]
            values = {
                "#": display_num,
                "Date": str(d),
                "Time": str(t)[:8],
                "Open": f"{float(o):.2f}" if o is not None else "",
                "High": f"{float(h):.2f}" if h is not None else "",
                "Low": f"{float(l_):.2f}" if l_ is not None else "",
                "Close": f"{float(c):.2f}" if c is not None else "",
                "Volume": f"{int(v):,}" if v is not None else "",
                "VWAP": f"{float(vw):.2f}" if vw is not None else "",
            }
        else:
            d = bar.get("date") or bar.get("market_date", "")
            t = bar.get("time") or bar.get("market_time", "")
            values = {
                "#": display_num,
                "Date": str(d),
                "Time": str(t)[:8],
                "Open": f"{bar['open']:.2f}" if bar.get("open") is not None else "",
                "High": f"{bar['high']:.2f}" if bar.get("high") is not None else "",
                "Low": f"{bar['low']:.2f}" if bar.get("low") is not None else "",
                "Close": f"{bar['close']:.2f}" if bar.get("close") is not None else "",
                "Volume": f"{int(bar.get('volume', 0)):,}",
                "VWAP": f"{bar['vwap']:.2f}" if bar.get("vwap") is not None else "",
            }

        fg = _FORMING_FG if forming else _NORMAL_FG
        bg = _FORMING_BG if forming else None

        for col_name, col_idx in _COL_IDX.items():
            item = self.table.item(row_idx, col_idx)
            if item is None:
                item = QTableWidgetItem()
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.table.setItem(row_idx, col_idx, item)

            item.setText(str(values.get(col_name, "")))
            item.setForeground(QBrush(fg))
            if bg:
                item.setBackground(QBrush(bg))
            else:
                item.setData(Qt.ItemDataRole.BackgroundRole, None)

    def _finalize_forming_row(self, bar: dict):
        """Turn the forming row into a completed row, then reset forming state."""
        if self._forming_row is not None:
            self._update_row(self._forming_row, bar, forming=False)
        self._forming_row = None
        self._bar_progress.setText("")

    def _scroll_to_bottom(self):
        if self._row_count > 0:
            self.table.scrollToBottom()

    def _update_status(self):
        completed = self.get_bar_count()
        self._status_label.setText(f"{completed:,} bars")
        self.bar_count_changed.emit(completed)
