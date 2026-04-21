"""
Stock Fetcher UI - Page-driven version
"""

import sys
import os
import glob
from datetime import datetime, timedelta, time as dt_time
from typing import List

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QLineEdit,
                             QFrame, QTableWidget, QTableWidgetItem,
                             QHeaderView, QGridLayout, QProgressDialog, QMessageBox,
                             QCheckBox, QDateEdit, QComboBox, QCalendarWidget)
from PyQt6.QtCore import Qt, pyqtSignal, QDate, QThread
from PyQt6.QtGui import QFont, QTextCharFormat, QColor

from k2_quant.utilities.config.api_config import api_config
from k2_quant.utilities.services.stock_data_service import stock_service
from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.dialogs import (
    show_warning, show_error, show_info, show_question, show_confirm_delete, show_data_clear_options
)


# ---------------------------------------------------------------------------
# NYSE holiday / market-calendar helpers
# ---------------------------------------------------------------------------

def _easter_sunday(year: int) -> QDate:
    """Compute Easter Sunday via the Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = ((h + el - 7 * m + 114) % 31) + 1
    return QDate(year, month, day)


def _observed(date: QDate) -> QDate:
    """Shift Sat→Fri, Sun→Mon for fixed holidays observed by NYSE."""
    dow = date.dayOfWeek()
    if dow == 6:
        return date.addDays(-1)
    if dow == 7:
        return date.addDays(1)
    return date


def _nth_weekday(year: int, month: int, day_of_week: int, n: int) -> QDate:
    """Return the *n*-th occurrence of *day_of_week* (1=Mon … 7=Sun) in *month*."""
    first = QDate(year, month, 1)
    offset = (day_of_week - first.dayOfWeek()) % 7
    return first.addDays(offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, day_of_week: int) -> QDate:
    last_day = QDate(year, month, 1).addMonths(1).addDays(-1)
    offset = (last_day.dayOfWeek() - day_of_week) % 7
    return last_day.addDays(-offset)


def _nyse_holidays_for_year(year: int) -> list:
    holidays = [
        _observed(QDate(year, 1, 1)),                       # New Year's Day
        _nth_weekday(year, 1, 1, 3),                        # MLK Day
        _nth_weekday(year, 2, 1, 3),                        # Presidents' Day
        _easter_sunday(year).addDays(-2),                   # Good Friday
        _last_weekday(year, 5, 1),                          # Memorial Day
        _observed(QDate(year, 6, 19)),                      # Juneteenth
        _observed(QDate(year, 7, 4)),                       # Independence Day
        _nth_weekday(year, 9, 1, 1),                        # Labor Day
        _nth_weekday(year, 11, 4, 4),                       # Thanksgiving
        _observed(QDate(year, 12, 25)),                     # Christmas
    ]
    return holidays


def _build_nyse_holiday_set() -> set:
    holidays = set()
    for year in range(2000, 2036):
        holidays.update(_nyse_holidays_for_year(year))
    return holidays


NYSE_HOLIDAYS = _build_nyse_holiday_set()


class MarketCalendar(QCalendarWidget):
    """QCalendarWidget that visually greys-out and prevents selection of
    weekends and NYSE holidays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self._last_valid = self.selectedDate()
        self.selectionChanged.connect(self._on_selection_changed)
        self._apply_disabled_formats()

    def _is_disabled(self, date: QDate) -> bool:
        return date.dayOfWeek() in (6, 7) or date in NYSE_HOLIDAYS

    def _on_selection_changed(self):
        if self._updating:
            return
        date = self.selectedDate()
        if self._is_disabled(date):
            self._updating = True
            self.setSelectedDate(self._last_valid)
            self._updating = False
        else:
            self._last_valid = date

    def _apply_disabled_formats(self):
        disabled = QTextCharFormat()
        disabled.setForeground(QColor(60, 60, 60))
        self.setWeekdayTextFormat(Qt.DayOfWeek.Saturday, disabled)
        self.setWeekdayTextFormat(Qt.DayOfWeek.Sunday, disabled)
        for d in NYSE_HOLIDAYS:
            self.setDateTextFormat(d, disabled)


def _snap_to_trading_day(date: QDate) -> QDate:
    """Walk backwards until a trading day is found."""
    while date.dayOfWeek() in (6, 7) or date in NYSE_HOLIDAYS:
        date = date.addDays(-1)
    return date


def _generate_time_items(start_h, start_m, end_h, end_m):
    """Return list of (display_text, value_24h) tuples in 15-min steps."""
    items = []
    h, m = start_h, start_m
    while (h < end_h) or (h == end_h and m <= end_m):
        t = dt_time(h, m)
        display = t.strftime("%I:%M %p").lstrip("0")
        value = t.strftime("%H:%M")
        items.append((display, value))
        m += 15
        if m >= 60:
            h += 1
            m = 0
    return items


MARKET_HOURS_TIMES = _generate_time_items(9, 30, 16, 0)
EXTENDED_HOURS_TIMES = _generate_time_items(4, 0, 20, 0)

FREQ_TO_MINUTES = {
    '1min': 1, '5min': 5, '15min': 15, '30min': 30, '1H': 60,
}


def _last_bar_minutes(close_h: int, close_m: int, freq_min: int) -> int:
    """Minute-of-day of the last interval-aligned bar strictly before close."""
    return ((close_h * 60 + close_m - 1) // freq_min) * freq_min


def _build_time_items(open_h, open_m, close_h, close_m, freq_str):
    """Generate (display, value) time items from open to the last bar before close.

    Uses 15-minute steps for the main grid.  If the computed last-bar time
    does not land on a 15-minute boundary it is appended as a final entry
    so the combo always contains the correct default end.

    Returns (items_list, default_end_value_24h).
    """
    freq_min = FREQ_TO_MINUTES.get(freq_str, 1)
    last_min = _last_bar_minutes(close_h, close_m, freq_min)

    items = []
    cur = open_h * 60 + open_m
    step = 15
    while cur <= last_min:
        h, m = divmod(cur, 60)
        t = dt_time(h, m)
        display = t.strftime("%I:%M %p").lstrip("0")
        value = t.strftime("%H:%M")
        items.append((display, value))
        cur += step

    last_h, last_m = divmod(last_min, 60)
    last_value = dt_time(last_h, last_m).strftime("%H:%M")
    if not items or items[-1][1] != last_value:
        t = dt_time(last_h, last_m)
        display = t.strftime("%I:%M %p").lstrip("0")
        items.append((display, last_value))

    return items, last_value


# ---------------------------------------------------------------------------
# Background workers – keep the UI thread free during I/O-heavy operations
# ---------------------------------------------------------------------------

class _FetchWorker(QThread):
    """Runs stock_service.fetch_and_store_stock_data off the main thread."""

    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, symbol, active_range, active_frequency, custom_start, custom_end,
                 market_hours_only=False):
        super().__init__()
        self._symbol = symbol
        self._range = active_range
        self._frequency = active_frequency
        self._custom_start = custom_start
        self._custom_end = custom_end
        self._market_hours_only = market_hours_only

    def run(self):
        try:
            result = stock_service.fetch_and_store_stock_data(
                self._symbol, self._range, self._frequency,
                market_hours_only=self._market_hours_only,
                custom_start=self._custom_start, custom_end=self._custom_end)
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class _DbLoadWorker(QThread):
    """Queries display rows from the database off the main thread."""

    result_ready = pyqtSignal(object, int)
    error_occurred = pyqtSignal(str)

    def __init__(self, table_name, limit, time_start, time_end, cutoff_datetime=None):
        super().__init__()
        self._table_name = table_name
        self._limit = limit
        self.time_start = time_start
        self.time_end = time_end
        self.cutoff_datetime = cutoff_datetime

    def run(self):
        try:
            rows, total_count = stock_service.get_display_data(
                self._table_name,
                self._limit,
                market_hours_only=False,
                time_start=self.time_start,
                time_end=self.time_end,
                cutoff_datetime=self.cutoff_datetime,
            )
            self.result_ready.emit(rows, total_count)
        except Exception as e:
            self.error_occurred.emit(str(e))


class _ExportWorker(QThread):
    """Streams data from the database to CSV files off the main thread."""

    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, current_data, current_table, market_hours_only, time_start, time_end,
                 cutoff_datetime=None):
        super().__init__()
        self._data = current_data
        self._table = current_table
        self._market_hours = market_hours_only
        self._time_start = time_start
        self._time_end = time_end
        self._cutoff_datetime = cutoff_datetime
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import csv
        try:
            downloads_dir = os.path.expanduser("~/Downloads")
            os.makedirs(downloads_dir, exist_ok=True)
            market_suffix = "_market_hours" if self._market_hours else ""
            base_name = (
                f"{self._data['symbol']}_{self._data['range']}_"
                f"{self._data['frequency']}{market_suffix}"
            )
            MAX_ROWS_PER_FILE = 1_000_000

            def open_part(part_idx):
                part_path = os.path.join(downloads_dir, f"{base_name}_part{part_idx:02d}.csv")
                f = open(part_path, 'w', newline='', encoding='utf-8')
                w = csv.writer(f)
                market_tz_local = os.getenv('MARKET_TIMEZONE', 'US/Eastern').split('/')[-1]
                w.writerow([
                    f'Date ({market_tz_local})', f'Time ({market_tz_local})',
                    'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP',
                ])
                return f, w, part_path

            part = 1
            rows_in_part = 0
            file_handle, writer, filename = open_part(part)
            batch_count = 0

            try:
                for batch in stock_service.get_export_data_streaming(
                        self._table, market_hours_only=False,
                        time_start=self._time_start, time_end=self._time_end,
                        cutoff_datetime=self._cutoff_datetime):
                    if self._cancelled:
                        break
                    for row in batch:
                        if self._cancelled:
                            break
                        md, mt = row[0], row[1]
                        date_str = md.strftime('%Y-%m-%d') if hasattr(md, 'strftime') else str(md)
                        time_str = mt.strftime('%H:%M:%S') if hasattr(mt, 'strftime') else str(mt)
                        o = float(row[2]); h = float(row[3]); l = float(row[4]); c = float(row[5])
                        vol = int(row[6]); vw = float(row[7])
                        writer.writerow([
                            date_str, time_str,
                            f"{o:.2f}", f"{h:.2f}", f"{l:.2f}", f"{c:.2f}",
                            vol, f"{vw:.2f}",
                        ])
                        rows_in_part += 1
                        if rows_in_part >= MAX_ROWS_PER_FILE:
                            file_handle.close()
                            part += 1
                            rows_in_part = 0
                            file_handle, writer, filename = open_part(part)
                    batch_count += 1
                    self.progress.emit(min(99, batch_count * 5))
            finally:
                try:
                    file_handle.close()
                except Exception:
                    pass

            if not self._cancelled:
                self.finished_ok.emit(filename)
        except Exception as e:
            self.error_occurred.emit(str(e))


class StockFetcherWidget(QMainWindow):
    """Main stock fetcher UI widget"""

    stock_data_fetched = pyqtSignal(dict)
    back_to_landing = pyqtSignal()
    model_saved = pyqtSignal(str)  # NEW SIGNAL for model saved
    database_cleared = pyqtSignal()  # Emitted after full DB deletion & analysis reset request

    INTRADAY_FREQUENCIES = {'1min', '5min', '15min', '30min', '1H'}
    RANGE_DAYS = {
        '1D': 1, '1W': 7, '1M': 30, '3M': 90,
        '6M': 180, '1Y': 365, '2Y': 730, '5Y': 1825,
        '10Y': 3650, '20Y': 7300,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker_thread = None
        self._db_load_worker = None
        self.export_worker = None
        self.current_data = None
        self.current_table = None
        self.active_range = '1M'
        self.active_frequency = 'D'
        self.market_hours_only = False

        if not api_config.polygon_api_key:
            show_warning(
                self,
                "Configuration Error",
                "Polygon API key not found. Please add POLYGON_API_KEY to your .env file."
            )

        self.init_ui()
        self.setup_styling()
        self.check_existing_tables()

    def init_ui(self):
        self.setWindowTitle("K2 Quant - Stock Data Fetcher")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)

        self.create_sidebar(main_layout)
        self.create_main_content(main_layout)

    def create_sidebar(self, parent_layout):
        sidebar = QFrame()
        sidebar.setFixedWidth(280)
        sidebar.setObjectName("sidebar")

        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(12)
        sidebar.setLayout(sidebar_layout)

        ticker_section = self.create_ticker_section()
        sidebar_layout.addWidget(ticker_section)

        range_section = self.create_range_section()
        sidebar_layout.addWidget(range_section)

        date_range_section = self.create_date_range_section()
        sidebar_layout.addWidget(date_range_section)

        freq_section = self.create_frequency_section()
        sidebar_layout.addWidget(freq_section)

        filters_section = self.create_filters_section()
        sidebar_layout.addWidget(filters_section)

        time_range_section = self.create_time_range_section()
        sidebar_layout.addWidget(time_range_section)

        metrics_frame = self.create_metrics_frame()
        sidebar_layout.addWidget(metrics_frame)

        self.table_info_frame = self.create_table_info_frame()
        sidebar_layout.addWidget(self.table_info_frame)

        actions_section = self.create_actions_section()
        sidebar_layout.addWidget(actions_section)

        sidebar_layout.addStretch()

        parent_layout.addWidget(sidebar)

    def create_ticker_section(self) -> QFrame:
        ticker_section = QFrame()
        ticker_layout = QVBoxLayout()
        ticker_layout.setContentsMargins(0, 0, 0, 0)
        ticker_section.setLayout(ticker_layout)

        ticker_label = QLabel("TICKER")
        ticker_label.setObjectName("sectionTitle")
        ticker_layout.addWidget(ticker_label)

        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter symbol...")
        self.ticker_input.setMaxLength(5)
        self.ticker_input.returnPressed.connect(self.fetch_stock_data)
        ticker_layout.addWidget(self.ticker_input)

        return ticker_section

    def create_range_section(self) -> QFrame:
        range_section = QFrame()
        range_layout = QVBoxLayout()
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_section.setLayout(range_layout)

        range_label = QLabel("QUICK RANGE")
        range_label.setObjectName("sectionTitle")
        range_layout.addWidget(range_label)

        range_grid = QGridLayout()
        range_grid.setSpacing(6)

        ranges = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '20Y']
        for i, range_text in enumerate(ranges):
            btn = QPushButton(range_text)
            btn.setObjectName("rangeButton")
            btn.setCheckable(True)
            if range_text == '1M':
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, r=range_text: self.set_range(r))
            range_grid.addWidget(btn, i // 4, i % 4)

        range_layout.addLayout(range_grid)
        return range_section

    def create_date_range_section(self) -> QFrame:
        section = QFrame()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        section.setLayout(layout)

        title = QLabel("DATE RANGE")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        # --- FROM row ---
        from_row = QHBoxLayout()
        from_label = QLabel("FROM")
        from_label.setObjectName("fieldLabel")
        from_label.setFixedWidth(36)
        from_row.addWidget(from_label)

        self.date_from = QDateEdit()
        self.date_from.setObjectName("dateEdit")
        self.date_from.setDisplayFormat("MM-dd-yyyy")
        self.date_from.setCalendarPopup(True)
        cal_from = MarketCalendar()
        self.date_from.setCalendarWidget(cal_from)
        from_row.addWidget(self.date_from)
        layout.addLayout(from_row)

        # --- TO row ---
        to_row = QHBoxLayout()
        to_label = QLabel("TO")
        to_label.setObjectName("fieldLabel")
        to_label.setFixedWidth(36)
        to_row.addWidget(to_label)

        self.date_to = QDateEdit()
        self.date_to.setObjectName("dateEdit")
        self.date_to.setDisplayFormat("MM-dd-yyyy")
        self.date_to.setCalendarPopup(True)
        cal_to = MarketCalendar()
        self.date_to.setCalendarWidget(cal_to)
        to_row.addWidget(self.date_to)
        layout.addLayout(to_row)

        # --- Cutoff row (below TO) ---
        cutoff_row = QHBoxLayout()
        cutoff_label = QLabel("AT")
        cutoff_label.setObjectName("fieldLabel")
        cutoff_label.setFixedWidth(36)
        cutoff_row.addWidget(cutoff_label)

        self.cutoff_time = QComboBox()
        self.cutoff_time.setObjectName("timeCombo")
        self.cutoff_time.setEnabled(False)
        self.cutoff_time.currentIndexChanged.connect(self._on_time_range_changed)
        cutoff_row.addWidget(self.cutoff_time)
        layout.addLayout(cutoff_row)

        # Populate initial dates from default range (1M)
        self._populate_dates_from_range(self.active_range)

        return section

    def create_frequency_section(self) -> QFrame:
        freq_section = QFrame()
        freq_layout = QVBoxLayout()
        freq_layout.setContentsMargins(0, 0, 0, 0)
        freq_section.setLayout(freq_layout)

        freq_label = QLabel("FREQUENCY")
        freq_label.setObjectName("sectionTitle")
        freq_layout.addWidget(freq_label)

        freq_grid = QGridLayout()
        freq_grid.setSpacing(6)

        frequencies = ['1min', '5min', '15min', '30min', '1H', 'D', 'W', 'M']
        for i, freq_text in enumerate(frequencies):
            btn = QPushButton(freq_text)
            btn.setObjectName("freqButton")
            btn.setCheckable(True)
            if freq_text == 'D':
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, f=freq_text: self.set_frequency(f))
            freq_grid.addWidget(btn, i // 4, i % 4)

        freq_layout.addLayout(freq_grid)
        return freq_section

    def create_filters_section(self) -> QFrame:
        filters_section = QFrame()
        filters_layout = QVBoxLayout()
        filters_layout.setContentsMargins(0, 0, 0, 0)
        filters_section.setLayout(filters_layout)

        filters_label = QLabel("FILTERS")
        filters_label.setObjectName("sectionTitle")
        filters_layout.addWidget(filters_label)

        self.market_hours_checkbox = QCheckBox("Market Hours Only")
        self.market_hours_checkbox.setObjectName("filterCheckbox")
        self.market_hours_checkbox.setToolTip("Clamp time range to regular market hours")
        self.market_hours_checkbox.setChecked(False)
        self.market_hours_checkbox.setEnabled(False)
        self.market_hours_checkbox.stateChanged.connect(self.apply_market_hours_filter)
        filters_layout.addWidget(self.market_hours_checkbox)

        self.filter_info_label = QLabel("(9:30 AM - 4:00 PM ET)")
        self.filter_info_label.setObjectName("filterInfoLabel")
        self.filter_info_label.setStyleSheet("color: #666; font-size: 10px; margin-left: 20px;")
        filters_layout.addWidget(self.filter_info_label)

        return filters_section

    def create_time_range_section(self) -> QFrame:
        section = QFrame()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        section.setLayout(layout)

        title = QLabel("TIME RANGE")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        # --- FROM row ---
        from_row = QHBoxLayout()
        from_label = QLabel("FROM")
        from_label.setObjectName("fieldLabel")
        from_label.setFixedWidth(36)
        from_row.addWidget(from_label)

        self.time_from = QComboBox()
        self.time_from.setObjectName("timeCombo")
        self.time_from.currentIndexChanged.connect(self._on_time_range_changed)
        from_row.addWidget(self.time_from)
        layout.addLayout(from_row)

        # --- TO row ---
        to_row = QHBoxLayout()
        to_label = QLabel("TO")
        to_label.setObjectName("fieldLabel")
        to_label.setFixedWidth(36)
        to_row.addWidget(to_label)

        self.time_to = QComboBox()
        self.time_to.setObjectName("timeCombo")
        self.time_to.currentIndexChanged.connect(self._on_time_range_changed)
        to_row.addWidget(self.time_to)
        layout.addLayout(to_row)

        # Default state: disabled (default freq is D)
        self._populate_time_combos(MARKET_HOURS_TIMES, "09:30", "16:00")
        self._populate_cutoff_combo(MARKET_HOURS_TIMES, "16:00")
        self._set_time_controls_enabled(False)

        return section

    def create_metrics_frame(self) -> QFrame:
        metrics_frame = QFrame()
        metrics_frame.setObjectName("metricsFrame")
        metrics_layout = QVBoxLayout()
        metrics_layout.setContentsMargins(0, 10, 0, 10)
        metrics_frame.setLayout(metrics_layout)

        self.exec_time_label = QLabel("")
        self.exec_time_label.setObjectName("metricLabel")
        metrics_layout.addWidget(self.exec_time_label)

        self.speed_label = QLabel("")
        self.speed_label.setObjectName("metricLabel")
        metrics_layout.addWidget(self.speed_label)

        return metrics_frame

    def create_table_info_frame(self) -> QFrame:
        info_frame = QFrame()
        info_frame.setObjectName("tableInfoFrame")
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 10, 0, 10)
        info_frame.setLayout(info_layout)

        self.table_name_label = QLabel("")
        self.table_name_label.setObjectName("tableInfoLabel")
        self.table_name_label.setWordWrap(True)
        info_layout.addWidget(self.table_name_label)

        info_frame.hide()
        return info_frame

    def create_actions_section(self) -> QFrame:
        actions_section = QFrame()
        actions_section.setObjectName("bottomActions")
        actions_layout = QVBoxLayout()
        actions_layout.setContentsMargins(0, 10, 0, 0)
        actions_section.setLayout(actions_layout)

        self.fetch_button = QPushButton("Fetch Data")
        self.fetch_button.setObjectName("fetchButton")
        self.fetch_button.clicked.connect(self.fetch_stock_data)
        actions_layout.addWidget(self.fetch_button)

        self.export_button = QPushButton("Export as CSV")
        self.export_button.setObjectName("exportButton")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_data_csv)
        actions_layout.addWidget(self.export_button)

        self.save_model_btn = QPushButton("Save as Model")
        self.save_model_btn.setObjectName("actionButton")
        self.save_model_btn.clicked.connect(self.save_as_model)
        self.save_model_btn.setEnabled(False)
        actions_layout.addWidget(self.save_model_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("actionButton")
        self.clear_btn.clicked.connect(self.clear_data)
        self.clear_btn.setEnabled(False)
        actions_layout.addWidget(self.clear_btn)

        self.delete_db_btn = QPushButton("Delete Database")
        self.delete_db_btn.setObjectName("dangerButton")
        self.delete_db_btn.clicked.connect(self.delete_database)
        self.delete_db_btn.setEnabled(False)
        actions_layout.addWidget(self.delete_db_btn)

        return actions_section

    def create_main_content(self, parent_layout):
        main_content = QFrame()
        main_content.setObjectName("mainContent")

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        main_content.setLayout(content_layout)

        header = self.create_header()
        content_layout.addWidget(header)

        self.data_container = self.create_data_container()
        content_layout.addWidget(self.data_container)

        parent_layout.addWidget(main_content)

    def create_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(60)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(30, 0, 30, 0)
        header.setLayout(header_layout)

        header_layout.addStretch()

        self.record_count = QLabel("")
        self.record_count.setObjectName("recordCount")
        header_layout.addWidget(self.record_count)

        return header

    def create_data_container(self) -> QFrame:
        data_container = QFrame()
        data_container.setObjectName("dataContainer")

        data_layout = QVBoxLayout()
        data_layout.setContentsMargins(30, 20, 30, 20)
        data_container.setLayout(data_layout)

        self.empty_state = self.create_empty_state()
        data_layout.addWidget(self.empty_state)

        self.record_info = QLabel("")
        self.record_info.setObjectName("recordInfo")
        self.record_info.hide()
        data_layout.addWidget(self.record_info)

        self.data_table = self.create_data_table()
        data_layout.addWidget(self.data_table)

        return data_container

    def create_empty_state(self) -> QFrame:
        empty_state = QFrame()
        empty_layout = QVBoxLayout()
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_state.setLayout(empty_layout)

        empty_title = QLabel("No Data Loaded")
        empty_title.setFont(QFont("Arial", 18))
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_title)

        empty_text = QLabel("Enter a ticker symbol and select a time range to fetch stock data")
        empty_text.setFont(QFont("Arial", 14))
        empty_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_text.setObjectName("emptyText")
        empty_layout.addWidget(empty_text)

        tier_label = QLabel("Polygon.io API Connected")
        tier_label.setFont(QFont("Arial", 12))
        tier_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tier_label.setObjectName("tierLabel")
        empty_layout.addWidget(tier_label)

        return empty_state

    def create_data_table(self) -> QTableWidget:
        data_table = QTableWidget()
        data_table.setColumnCount(8)
        market_tz = os.getenv('MARKET_TIMEZONE', 'US/Eastern').split('/')[-1]
        data_table.setHorizontalHeaderLabels([f'Date ({market_tz})', f'Time ({market_tz})', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP'])

        header = data_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)

        # Apply small-caps to header titles
        header_font = header.font()
        header_font.setCapitalization(QFont.Capitalization.SmallCaps)
        header.setFont(header_font)

        data_table.verticalHeader().setVisible(False)
        data_table.setCornerButtonEnabled(False)
        data_table.setAlternatingRowColors(True)
        data_table.hide()
        data_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        data_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return data_table

    def setup_styling(self):
        self.setStyleSheet(
            """
            QMainWindow { background-color: #0a0a0a; }
            #sidebar { background-color: #0f0f0f; border-right: 1px solid #1a1a1a; }
            #sectionTitle { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #999; margin-bottom: 8px; font-weight: 600; }
            #fieldLabel { font-size: 11px; color: #999; }
            #metricsFrame, #tableInfoFrame { background-color: #1a1a1a; border-radius: 4px; padding: 10px; }
            #metricLabel { font-size: 11px; color: #4a4; text-align: center; margin: 2px; }
            #tableInfoLabel { font-size: 10px; color: #888; text-align: center; }
            #tierLabel { color: #4a4; margin-top: 10px; }
            QCheckBox#filterCheckbox { color: #ffffff; font-size: 12px; padding: 5px; }
            QCheckBox#filterCheckbox::indicator { width: 16px; height: 16px; background-color: #1a1a1a; border: 1px solid #3a3a3a; border-radius: 3px; }
            QCheckBox#filterCheckbox::indicator:checked { background-color: #4a4; border-color: #4a4; }
            QCheckBox#filterCheckbox:disabled { color: #555; }
            QLineEdit { background-color: #1a1a1a; border: 1px solid #2a2a2a; color: #fff; padding: 12px; font-size: 14px; border-radius: 4px; }
            QLineEdit:focus { border-color: #3a3a3a; background-color: #222; }

            QDateEdit#dateEdit { background-color: #1a1a1a; border: 1px solid #2a2a2a; color: #fff; padding: 6px 8px; font-size: 12px; border-radius: 3px; }
            QDateEdit#dateEdit:focus { border-color: #3a3a3a; background-color: #222; }
            QDateEdit#dateEdit::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 18px; border: none; }
            QDateEdit#dateEdit::down-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #999; }
            QCalendarWidget { background-color: #1a1a1a; color: #fff; }
            QCalendarWidget QWidget#qt_calendar_navigationbar { background-color: #111; }
            QCalendarWidget QToolButton { color: #fff; background-color: #1a1a1a; border: none; padding: 4px 8px; font-size: 12px; }
            QCalendarWidget QToolButton:hover { background-color: #2a2a2a; }
            QCalendarWidget QAbstractItemView { background-color: #1a1a1a; color: #ccc; selection-background-color: #fff; selection-color: #000; font-size: 11px; }
            QCalendarWidget QAbstractItemView:enabled { color: #ccc; }
            QCalendarWidget QAbstractItemView:disabled { color: #444; }

            QComboBox#timeCombo { background-color: #1a1a1a; border: 1px solid #2a2a2a; color: #fff; padding: 6px 8px; font-size: 12px; border-radius: 3px; }
            QComboBox#timeCombo:disabled { color: #555; background-color: #111; border-color: #1a1a1a; }
            QComboBox#timeCombo::drop-down { border: none; width: 18px; }
            QComboBox#timeCombo::down-arrow { border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #999; }
            QComboBox#timeCombo QAbstractItemView { background-color: #1a1a1a; color: #fff; selection-background-color: #2a2a2a; selection-color: #fff; border: 1px solid #2a2a2a; }

            #rangeButton, #freqButton { background-color: #1a1a1a; border: 1px solid #2a2a2a; color: #999; padding: 8px; font-size: 12px; border-radius: 3px; }
            #rangeButton:hover, #freqButton:hover { background-color: #2a2a2a; color: #fff; }
            #rangeButton:checked, #freqButton:checked { background-color: #fff; color: #000; font-weight: 500; }
            #fetchButton { background-color: #1a1a1a; border: 1px solid #4a4a4a; color: #fff; padding: 12px; font-size: 13px; border-radius: 4px; margin-bottom: 10px; }
            #fetchButton:hover { background-color: #2a2a2a; border-color: #5a5a5a; }
            #actionButton, #exportButton { background-color: #1a1a1a; border: 1px solid #2a2a2a; color: #fff; padding: 12px; font-size: 13px; border-radius: 4px; margin-bottom: 10px; }
            #actionButton:hover:enabled, #exportButton:hover:enabled { background-color: #2a2a2a; border-color: #3a3a3a; }
            #actionButton:disabled, #exportButton:disabled { background-color: #1a1a1a; color: #666; border-color: #2a2a2a; }
            #dangerButton { background-color: #1a1a1a; border: 1px solid #ff4444; color: #ff4444; padding: 12px; font-size: 13px; border-radius: 4px; }
            #dangerButton:hover:enabled { background-color: #ff4444; color: #fff; }
            #dangerButton:disabled { background-color: #1a1a1a; color: #666; border-color: #2a2a2a; }
            #header { background-color: #0f0f0f; border-bottom: 1px solid #1a1a1a; color: #fff; }
            #recordCount, #recordInfo { font-size: 13px; color: #666; }
            #dataContainer { background-color: #0a0a0a; }
            #emptyText { color: #666; }
            QTableWidget { background-color: #0a0a0a; border: none; gridline-color: #1a1a1a; color: #fff; font-family: 'Consolas', 'Monaco', 'Courier New', monospace; font-size: 13px; }
            QTableWidget::item { padding: 10px 15px; border: none; border-bottom: 1px solid #1a1a1a; color: #fff; }
            QTableWidget::item:selected { background-color: #2a2a2a; color: #fff; }
            QTableWidget::item:alternate { background-color: #0f0f0f; }
            QHeaderView::section { background-color: #0a0a0a; color: #999; padding: 12px 15px; border: none; border-bottom: 2px solid #2a2a2a; font-weight: 600; font-size: 11px; letter-spacing: 1px; text-align: left; }
            QHeaderView::section:first { border-left: none; }
            QHeaderView::section:last { border-right: none; }
            QTableCornerButton::section { background-color: #0a0a0a; border: none; }
            #bottomActions { border-top: 1px solid #1a1a1a; padding-top: 20px; }
            QProgressDialog { background-color: #1a1a1a; color: #fff; }
            QProgressDialog QLabel { color: #fff; }
            QProgressDialog QProgressBar { border: 1px solid #2a2a2a; border-radius: 3px; text-align: center; background-color: #0a0a0a; }
            QProgressDialog QProgressBar::chunk { background-color: #4a4; border-radius: 2px; }
            QProgressDialog QPushButton { background-color: #2a2a2a; border: 1px solid #3a3a3a; color: #fff; padding: 6px 12px; border-radius: 3px; }
            QProgressDialog QPushButton:hover { background-color: #3a3a3a; }
            """
        )

    def save_as_model(self):
        """Save current data as a model and reset the UI"""
        if not self.current_data or not self.current_table:
            show_warning(self, "No Data", "No data to save as model")
            return
        
        try:
            # Import the saved models manager
            from k2_quant.utilities.data.saved_models_manager import saved_models_manager
            
            timespan, _, _, multiplier = stock_service.convert_ui_parameters(
                self.active_range, self.active_frequency)

            model_data = {
                'table_name': self.current_table,
                'symbol': self.current_data['symbol'],
                'timespan': timespan,
                'range_val': self.active_range,
                'frequency': str(multiplier),
                'market_hours_only': self.market_hours_only,
                'record_count': self.current_data.get('total_records', 0)
            }
            
            # Save to database
            success = saved_models_manager.save_model(model_data)
            
            if success:
                # Add sequential row number column
                from k2_quant.utilities.data.db_manager import db_manager
                db_manager.add_row_number_column(self.current_table)

                # Show success message
                show_info(
                    self, 
                    "Model Saved", 
                    f"Model '{self.current_data['symbol']} - {self.active_range}' has been saved.\n"
                    f"Table: {self.current_table}\n\n"
                    f"You can now access this model in the Analysis page."
                )
                
                # Emit signal to notify other components
                self.model_saved.emit(self.current_table)
                
                # Clear the UI to prepare for next fetch
                self.clear_ui()
                
                k2_logger.info(f"Model saved and UI cleared: {self.current_table}", "STOCK_FETCHER")
            else:
                show_error(self, "Save Error", "Failed to save model to database")
                
        except Exception as e:
            k2_logger.error(f"Error saving model: {str(e)}", "STOCK_FETCHER")
            show_error(self, "Save Error", f"Failed to save model: {str(e)}")

    def clear_data(self):
        """Handle clear data with saved model check"""
        if not self.current_table:
            self.clear_ui()
            return
            
        # Check if this table is a saved model
        try:
            from k2_quant.utilities.data.saved_models_manager import saved_models_manager
            is_saved = saved_models_manager.is_model_saved(self.current_table)
            
            if is_saved:
                # Warn that this is a saved model
                msg = QMessageBox(self)
                msg.setWindowTitle("Clear Saved Model")
                msg.setText(f"'{self.current_table}' is a saved model.\n\nWhat would you like to do?")
                msg.setIcon(QMessageBox.Icon.Warning)
                
                delete_btn = msg.addButton("Delete Table & Unsave", QMessageBox.ButtonRole.DestructiveRole)
                clear_btn = msg.addButton("Clear UI Only", QMessageBox.ButtonRole.AcceptRole)
                cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
                
                msg.exec()
                
                if msg.clickedButton() == delete_btn:
                    # Delete table and remove from saved models
                    saved_models_manager.unsave_model(self.current_table, delete_table=True)
                    show_info(self, "Model Deleted", f"Model '{self.current_table}' has been deleted.")
                    self.check_existing_tables()
                    self.clear_ui()
                elif msg.clickedButton() == clear_btn:
                    # Just clear UI
                    self.clear_ui()
                # else: Cancel - do nothing
            else:
                # Not a saved model - use original logic
                reply = show_data_clear_options(self)
                if reply == QMessageBox.StandardButton.Yes:
                    self.delete_table_and_clear()
                elif reply == QMessageBox.StandardButton.No:
                    self.clear_ui()
                    
        except ImportError:
            # SavedModelsManager doesn't exist yet - use original logic
            reply = show_data_clear_options(self)
            if reply == QMessageBox.StandardButton.Yes:
                self.delete_table_and_clear()
            elif reply == QMessageBox.StandardButton.No:
                self.clear_ui()
        except Exception as e:
            k2_logger.error(f"Error checking saved model status: {str(e)}", "STOCK_FETCHER")
            # Fall back to original behavior
            reply = show_data_clear_options(self)
            if reply == QMessageBox.StandardButton.Yes:
                self.delete_table_and_clear()
            elif reply == QMessageBox.StandardButton.No:
                self.clear_ui()

    # ----- helpers for date / time controls -----

    def _populate_dates_from_range(self, range_value: str):
        days = self.RANGE_DAYS.get(range_value, 30)
        end_qdate = _snap_to_trading_day(QDate.currentDate())
        start_qdate = _snap_to_trading_day(QDate.currentDate().addDays(-days))
        self.date_from.setDate(start_qdate)
        self.date_to.setDate(end_qdate)

    def _populate_time_combos(self, items, default_start: str, default_end: str):
        """Repopulate both time combo-boxes without triggering reload."""
        self.time_from.blockSignals(True)
        self.time_to.blockSignals(True)
        self.time_from.clear()
        self.time_to.clear()
        start_idx = end_idx = 0
        for i, (display, value) in enumerate(items):
            self.time_from.addItem(display, value)
            self.time_to.addItem(display, value)
            if value == default_start:
                start_idx = i
            if value == default_end:
                end_idx = i
        self.time_from.setCurrentIndex(start_idx)
        self.time_to.setCurrentIndex(end_idx)
        self.time_from.blockSignals(False)
        self.time_to.blockSignals(False)

    def _set_time_controls_enabled(self, enabled: bool):
        self.time_from.setEnabled(enabled)
        self.time_to.setEnabled(enabled)
        self.cutoff_time.setEnabled(enabled)
        self.market_hours_checkbox.setEnabled(enabled)
        if not enabled:
            self.filter_info_label.setText("")
            self._populate_cutoff_combo([], "")
        else:
            self._update_filter_info()

    def _populate_cutoff_combo(self, items, default_end: str):
        """Repopulate the cutoff-time combo beside the TO date."""
        self.cutoff_time.blockSignals(True)
        self.cutoff_time.clear()
        end_idx = 0
        for i, (display, value) in enumerate(items):
            self.cutoff_time.addItem(display, value)
            if value == default_end:
                end_idx = i
        self.cutoff_time.setCurrentIndex(end_idx)
        self.cutoff_time.blockSignals(False)

    def _refresh_time_combos(self):
        """Rebuild time combo items and defaults for the current frequency."""
        if self.market_hours_only:
            items, default_end = _build_time_items(9, 30, 16, 0, self.active_frequency)
            default_start = "09:30"
        else:
            items, default_end = _build_time_items(4, 0, 20, 0, self.active_frequency)
            default_start = "04:00"
        self._populate_time_combos(items, default_start, default_end)
        self._populate_cutoff_combo(items, default_end)
        self._update_filter_info()

    def _update_filter_info(self):
        freq_min = FREQ_TO_MINUTES.get(self.active_frequency)
        if self.market_hours_only:
            if freq_min:
                last_min = _last_bar_minutes(16, 0, freq_min)
                h, m = divmod(last_min, 60)
                end_display = dt_time(h, m).strftime("%I:%M %p").lstrip("0")
            else:
                end_display = "4:00 PM"
            label = f"(9:30 AM - {end_display} ET)"
        else:
            if freq_min:
                last_min = _last_bar_minutes(20, 0, freq_min)
                h, m = divmod(last_min, 60)
                end_display = dt_time(h, m).strftime("%I:%M %p").lstrip("0")
            else:
                end_display = "8:00 PM"
            label = f"(4:00 AM - {end_display} ET)"
        self.filter_info_label.setText(label)
        self.market_hours_checkbox.setToolTip(f"Clamp time range to {label.strip('()')}")

    def _get_current_time_filter(self):
        """Return (time_start, time_end, cutoff_datetime) for the active filter.

        time_start / time_end are 24h strings (or None for daily+ frequencies).
        cutoff_datetime is an ISO datetime string that caps the last day's data
        when the user picks an end time earlier than the uniform time_end, or
        None when no cutoff is needed.

        The returned time_end is extended by (interval - 1) minutes so the SQL
        BETWEEN clause captures all rows belonging to the last selected bar.
        """
        if self.active_frequency in self.INTRADAY_FREQUENCIES:
            ts = self.time_from.currentData()
            te = self.time_to.currentData()
            if ts and te:
                freq_min = FREQ_TO_MINUTES.get(self.active_frequency, 1)
                if freq_min > 1:
                    parts = te.split(':')
                    end_total = int(parts[0]) * 60 + int(parts[1]) + freq_min - 1
                    eh, em = divmod(min(end_total, 23 * 60 + 59), 60)
                    te = f"{eh:02d}:{em:02d}"

                cutoff_dt = None
                cutoff_val = self.cutoff_time.currentData()
                if cutoff_val and cutoff_val != self.time_to.currentData():
                    to_date_str = self.date_to.date().toString("yyyy-MM-dd")
                    ct = cutoff_val if len(cutoff_val) > 5 else f"{cutoff_val}:00"
                    if freq_min > 1:
                        cp = cutoff_val.split(':')
                        ct_total = int(cp[0]) * 60 + int(cp[1]) + freq_min - 1
                        ch, cm = divmod(min(ct_total, 23 * 60 + 59), 60)
                        ct = f"{ch:02d}:{cm:02d}:00"
                    cutoff_dt = f"{to_date_str} {ct}"

                return ts, te, cutoff_dt
        return None, None, None

    def _on_time_range_changed(self):
        if self.current_table:
            self.load_data_from_db()

    # ----- core state handlers -----

    def apply_market_hours_filter(self):
        self.market_hours_only = self.market_hours_checkbox.isChecked()
        self._refresh_time_combos()
        if self.current_table:
            k2_logger.ui_operation("Market hours filter changed", f"Filter active: {self.market_hours_only}")
            self.load_data_from_db()

    def check_existing_tables(self):
        try:
            tables = stock_service.get_all_stock_tables()
            self.delete_db_btn.setEnabled(len(tables) > 0)
        except Exception:
            self.delete_db_btn.setEnabled(False)

    def set_range(self, range_value: str):
        self.active_range = range_value
        for btn in self.findChildren(QPushButton):
            if btn.objectName() == "rangeButton":
                btn.setChecked(btn.text() == range_value)
        self._populate_dates_from_range(range_value)

    def set_frequency(self, freq_value: str):
        self.active_frequency = freq_value
        for btn in self.findChildren(QPushButton):
            if btn.objectName() == "freqButton":
                btn.setChecked(btn.text() == freq_value)
        is_intraday = freq_value in self.INTRADAY_FREQUENCIES
        if is_intraday:
            self.market_hours_checkbox.setEnabled(True)
            was_checked = self.market_hours_checkbox.isChecked()
            if was_checked:
                self._refresh_time_combos()
            else:
                self.market_hours_checkbox.setChecked(True)  # triggers apply_market_hours_filter
            self._set_time_controls_enabled(True)
        else:
            self.market_hours_checkbox.blockSignals(True)
            self.market_hours_checkbox.setChecked(False)
            self.market_hours_only = False
            self.market_hours_checkbox.blockSignals(False)
            self._set_time_controls_enabled(False)

    def fetch_stock_data(self):
        symbol = self.ticker_input.text().strip().upper()
        if not symbol:
            show_warning(self, "Input Error", "Please enter a ticker symbol")
            return
        if not api_config.polygon_api_key:
            show_error(self, "Configuration Error", "Polygon API key not configured.")
            return

        from_date = self.date_from.date()
        to_date = self.date_to.date()
        if from_date > to_date:
            show_warning(self, "Input Error", "FROM date must be before TO date.")
            return

        custom_start = from_date.toString("yyyy-MM-dd")
        custom_end = to_date.toString("yyyy-MM-dd")

        self.fetch_button.setEnabled(False)
        self.fetch_button.setText("Fetching...")
        self.export_button.setEnabled(False)
        self.clear_metrics()
        self.show_empty_state()

        self.worker_thread = _FetchWorker(
            symbol, self.active_range, self.active_frequency,
            custom_start, custom_end,
            market_hours_only=self.market_hours_only)
        self.worker_thread.result_ready.connect(self._on_fetch_finished)
        self.worker_thread.error_occurred.connect(self._on_fetch_error)
        self.worker_thread.finished.connect(self.worker_finished)
        self.worker_thread.start()

    def _on_fetch_finished(self, result: dict):
        self.display_stock_data(result)

    def _on_fetch_error(self, error_message: str):
        k2_logger.error(f"Worker error: {error_message}", "WORKER")
        self.handle_error(error_message)

    def display_stock_data(self, data: dict):
        self.current_data = data
        self.current_table = data.get('table_name')
        self.record_count.setText(f"{data['total_records']:,} records")
        self.update_metrics(data)
        self.update_table_info(data)
        self.load_data_from_db()
        self.export_button.setEnabled(True)
        self.save_model_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.delete_db_btn.setEnabled(True)
        self.stock_data_fetched.emit(data)

    def update_table_info(self, data: dict):
        table_name = data.get('table_name', '')
        if table_name:
            parts = table_name.split('_')
            if len(parts) > 4 and parts[-1].isdigit():
                version = parts[-1]
                self.table_info_frame.show()
                self.table_name_label.setText(f"Table: {table_name}\n(Version {version})")
            else:
                self.table_info_frame.show()
                self.table_name_label.setText(f"Table: {table_name}")

    def load_data_from_db(self):
        if not self.current_table:
            return
        time_start, time_end, cutoff_dt = self._get_current_time_filter()
        worker = _DbLoadWorker(self.current_table, 1000, time_start, time_end, cutoff_dt)
        worker.result_ready.connect(self._on_db_load_finished)
        worker.error_occurred.connect(self._on_db_load_error)
        self._db_load_worker = worker
        worker.start()

    def _on_db_load_finished(self, rows, total_count):
        time_start, time_end, cutoff_dt = self._get_current_time_filter()
        if time_start and time_end:
            ts_display = self.time_from.currentText()
            te_display = self.time_to.currentText()
            suffix = ""
            if cutoff_dt:
                cutoff_display = self.cutoff_time.currentText()
                suffix = f", last day until {cutoff_display}"
            self.record_count.setText(
                f"{total_count:,} records ({ts_display} – {te_display}{suffix})")
        else:
            self.record_count.setText(f"{total_count:,} records")
        if total_count > 1000:
            self.record_info.setText(
                f"Showing first 500 and last 500 of {total_count:,} total records"
            )
            self.record_info.show()
        else:
            self.record_info.hide()
        self.show_data_table(rows)

    def _on_db_load_error(self, error_message):
        show_error(self, "Display Error", f"Failed to load data: {error_message}")

    def show_data_table(self, rows: list):
        self.empty_state.hide()
        self.data_table.show()
        self.data_table.setRowCount(len(rows))
        for i in range(len(rows)):
            self.data_table.setRowHeight(i, 40)
        for i, row in enumerate(rows):
            # rows are standardized: Date, Time, Open, High, Low, Close, Volume, VWAP
            date_val = row[0]
            time_val = row[1]
            # Date column
            try:
                date_text = date_val.strftime('%Y-%m-%d')
            except Exception:
                date_text = str(date_val)
            date_item = QTableWidgetItem(date_text)
            date_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.data_table.setItem(i, 0, date_item)
            # Time column
            try:
                time_text = time_val.strftime('%H:%M:%S')
            except Exception:
                time_text = str(time_val)
            time_item = QTableWidgetItem(time_text)
            time_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.data_table.setItem(i, 1, time_item)
            for j, value in enumerate(row[2:8], 2):
                if j == 6:
                    item = QTableWidgetItem(f"{int(value):,}")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item = QTableWidgetItem(f"{float(value):.2f}")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.data_table.setItem(i, j, item)

    def update_metrics(self, data: dict):
        exec_time = data.get('execution_time', 0)
        records_per_sec = data.get('records_per_second', 0)
        self.exec_time_label.setText(f"Execution: {exec_time:.2f}s")
        self.speed_label.setText(f"Speed: {records_per_sec:,} rec/s")

    def clear_metrics(self):
        self.exec_time_label.clear()
        self.speed_label.clear()

    def show_empty_state(self):
        self.empty_state.show()
        self.data_table.hide()
        self.data_table.setRowCount(0)
        self.record_info.hide()
        self.table_info_frame.hide()
        self.table_name_label.clear()

    def handle_error(self, error_message: str):
        show_error(self, "Error", error_message)

    def worker_finished(self):
        self.fetch_button.setEnabled(True)
        self.fetch_button.setText("Fetch Data")

    def export_data_csv(self):
        if not self.current_data or not self.current_table:
            return

        ok, msg = stock_service.validate_export_size(self.current_table)
        if not ok:
            show_warning(self, "Export too large", msg)
            return

        self.export_button.setEnabled(False)
        self.export_button.setText("Exporting...")

        self._export_dlg = QProgressDialog("Exporting CSV…", "Cancel", 0, 100, self)
        self._export_dlg.setWindowTitle("Exporting Data")
        self._export_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._export_dlg.setMinimumWidth(400)
        self._export_dlg.show()

        time_start, time_end, cutoff_dt = self._get_current_time_filter()
        self.export_worker = _ExportWorker(
            self.current_data, self.current_table,
            self.market_hours_only, time_start, time_end, cutoff_dt)
        self.export_worker.progress.connect(self._export_dlg.setValue)
        self.export_worker.finished_ok.connect(self._on_export_finished)
        self.export_worker.error_occurred.connect(self._on_export_error)
        self._export_dlg.canceled.connect(self.export_worker.cancel)
        self.export_worker.start()

    def _on_export_finished(self, filename: str):
        self._export_dlg.setValue(100)
        self._export_dlg.close()
        self.export_button.setEnabled(True)
        self.export_button.setText("Export as CSV")
        reply = show_question(
            self,
            "Export Complete",
            f"Data exported successfully to:\n{filename}\n\nOpen containing folder?",
        )
        if reply == QMessageBox.StandardButton.Yes:
            folder = os.path.dirname(filename)
            if sys.platform == 'win32':
                os.startfile(folder)
            elif sys.platform == 'darwin':
                os.system(f'open "{folder}"')
            else:
                os.system(f'xdg-open "{folder}"')

    def _on_export_error(self, error_message: str):
        self._export_dlg.close()
        self.export_button.setEnabled(True)
        self.export_button.setText("Export as CSV")
        show_error(self, "Export Error", f"Failed to export data: {error_message}")

    def delete_table_and_clear(self):
        if not self.current_table:
            return
        try:
            stock_service.delete_table(self.current_table)
            show_info(self, "Table Deleted", f"Table '{self.current_table}' deleted.")
            self.check_existing_tables()
        except Exception as e:
            show_error(self, "Delete Error", f"Failed to delete table: {str(e)}")
        self.clear_ui()

    def clear_ui(self):
        self.current_data = None
        self.current_table = None
        self.ticker_input.clear()
        self.record_count.clear()
        self.record_info.clear()
        self.record_info.hide()
        self.clear_metrics()
        self.show_empty_state()
        self.market_hours_checkbox.blockSignals(True)
        self.market_hours_checkbox.setChecked(False)
        self.market_hours_only = False
        self.market_hours_checkbox.blockSignals(False)
        self._set_time_controls_enabled(self.active_frequency in self.INTRADAY_FREQUENCIES)
        self._populate_dates_from_range(self.active_range)
        self.export_button.setEnabled(False)
        self.save_model_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

    def delete_database(self):
        reply = show_confirm_delete(self, "Delete All Data", "Delete ALL stock data tables from the database?")
        if reply == QMessageBox.StandardButton.Yes:
            try:
                tables = stock_service.get_all_stock_tables()
                if not tables:
                    show_info(self, "No Data", "No stock data tables found.")
                    return
                count = stock_service.delete_all_tables()
                # Remove any model_*.json artifacts from repo root
                try:
                    for json_path in glob.glob("model_*.json"):
                        try:
                            os.remove(json_path)
                        except Exception:
                            pass
                except Exception:
                    pass

                # Inform user and notify application for analysis reset
                show_info(
                    self,
                    "Database Cleaned",
                    f"Deleted {count} stock data tables.\n\nThe Analysis page has been reset."
                )
                # Broadcast to other components (Main -> Analysis tabs)
                self.database_cleared.emit()
                self.clear_ui()
                self.delete_db_btn.setEnabled(False)
            except Exception as e:
                show_error(self, "Error", f"Failed to delete tables: {str(e)}")

    def cleanup(self):
        """Stop any running background workers before teardown."""
        for worker in (self.worker_thread, self._db_load_worker, self.export_worker):
            if worker is not None and hasattr(worker, 'isRunning') and worker.isRunning():
                try:
                    if hasattr(worker, 'cancel'):
                        worker.cancel()
                    worker.terminate()
                    worker.wait(3000)
                except Exception:
                    pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = StockFetcherWidget()
    widget.show()
    sys.exit(app.exec())