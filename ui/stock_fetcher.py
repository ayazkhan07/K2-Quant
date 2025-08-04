"""
Stock Fetcher UI - Updated Version

Clean UI implementation with separated business logic.
Updated to display table versioning information.
"""

import sys
import json
import os
from datetime import datetime
from typing import Dict, Optional

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                           QFrame, QTableWidget, QTableWidgetItem,
                           QHeaderView, QGridLayout, QScrollBar, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor

from config.api_config import api_config
from services.stock_data_service import stock_service
from utils.logger import k2_logger
from utils.dialogs import (show_warning, show_error, show_info, 
                          show_question, show_confirm_delete, show_data_clear_options)


class StockDataWorker(QThread):
    """Worker thread for fetching stock data"""
    
    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, symbol: str, time_range: str, frequency: str):
        super().__init__()
        self.symbol = symbol.upper()
        self.time_range = time_range
        self.frequency = frequency
    
    def run(self):
        """Run the data fetching process"""
        try:
            k2_logger.ui_operation(
                f"Worker started for {self.symbol}", 
                f"Range: {self.time_range}, Frequency: {self.frequency}"
            )
            
            # Use the service layer
            result = stock_service.fetch_and_store_stock_data(
                self.symbol, 
                self.time_range, 
                self.frequency
            )
            
            self.data_ready.emit(result)
            
        except Exception as e:
            k2_logger.error(f"Worker error: {str(e)}", "WORKER")
            self.error_occurred.emit(str(e))


class StockFetcherWidget(QMainWindow):
    """Main stock fetcher UI widget"""
    
    stock_data_fetched = pyqtSignal(dict)
    back_to_landing = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker_thread = None
        self.current_data = None
        self.current_table = None
        self.active_range = '1M'
        self.active_frequency = 'D'
        
        if not api_config.polygon_api_key:
            show_warning(
                self, 
                "Configuration Error", 
                "Polygon API key not found. Please add POLYGON_API_KEY to your .env file."
            )
        
        self.init_ui()
        self.setup_styling()
        self.check_existing_tables()  # Check for existing data on startup
    
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("K2 Quant - Stock Data Fetcher")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)
        
        # Create sidebar
        self.create_sidebar(main_layout)
        
        # Create main content area
        self.create_main_content(main_layout)
    
    def create_sidebar(self, parent_layout):
        """Create the left sidebar with controls"""
        sidebar = QFrame()
        sidebar.setFixedWidth(280)
        sidebar.setObjectName("sidebar")
        
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(30)
        sidebar.setLayout(sidebar_layout)
        
        # Ticker section
        ticker_section = self.create_ticker_section()
        sidebar_layout.addWidget(ticker_section)
        
        # Quick Range section
        range_section = self.create_range_section()
        sidebar_layout.addWidget(range_section)
        
        # Frequency section
        freq_section = self.create_frequency_section()
        sidebar_layout.addWidget(freq_section)
        
        # Add stretch
        sidebar_layout.addStretch()
        
        # Performance metrics
        metrics_frame = self.create_metrics_frame()
        sidebar_layout.addWidget(metrics_frame)
        
        # Table info frame (NEW)
        self.table_info_frame = self.create_table_info_frame()
        sidebar_layout.addWidget(self.table_info_frame)
        
        # Bottom actions
        actions_section = self.create_actions_section()
        sidebar_layout.addWidget(actions_section)
        
        parent_layout.addWidget(sidebar)
    
    def create_ticker_section(self) -> QFrame:
        """Create ticker input section"""
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
        """Create time range selection section"""
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
    
    def create_frequency_section(self) -> QFrame:
        """Create frequency selection section"""
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
    
    def create_metrics_frame(self) -> QFrame:
        """Create performance metrics display"""
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
        """Create table info display (NEW)"""
        info_frame = QFrame()
        info_frame.setObjectName("tableInfoFrame")
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 10, 0, 10)
        info_frame.setLayout(info_layout)
        
        self.table_name_label = QLabel("")
        self.table_name_label.setObjectName("tableInfoLabel")
        self.table_name_label.setWordWrap(True)
        info_layout.addWidget(self.table_name_label)
        
        info_frame.hide()  # Hidden by default
        return info_frame
    
    def create_actions_section(self) -> QFrame:
        """Create action buttons section"""
        actions_section = QFrame()
        actions_section.setObjectName("bottomActions")
        actions_layout = QVBoxLayout()
        actions_layout.setContentsMargins(0, 20, 0, 0)
        actions_section.setLayout(actions_layout)
        
        self.fetch_button = QPushButton("Fetch Data")
        self.fetch_button.setObjectName("fetchButton")
        self.fetch_button.clicked.connect(self.fetch_stock_data)
        actions_layout.addWidget(self.fetch_button)
        
        self.export_button = QPushButton("Export Data")
        self.export_button.setObjectName("exportButton")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_data)
        actions_layout.addWidget(self.export_button)
        
        # Save as Model button - initially disabled
        self.save_model_btn = QPushButton("Save as Model")
        self.save_model_btn.setObjectName("actionButton")
        self.save_model_btn.clicked.connect(self.save_as_model)
        self.save_model_btn.setEnabled(False)  # Disabled by default
        actions_layout.addWidget(self.save_model_btn)
        
        # Clear button - initially disabled
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("actionButton")
        self.clear_btn.clicked.connect(self.clear_data)
        self.clear_btn.setEnabled(False)  # Disabled by default
        actions_layout.addWidget(self.clear_btn)
        
        # Delete Database button - initially disabled
        self.delete_db_btn = QPushButton("Delete Database")
        self.delete_db_btn.setObjectName("dangerButton")
        self.delete_db_btn.clicked.connect(self.delete_database)
        self.delete_db_btn.setEnabled(False)  # Disabled by default
        actions_layout.addWidget(self.delete_db_btn)
        
        return actions_section
    
    def create_main_content(self, parent_layout):
        """Create the main content area"""
        main_content = QFrame()
        main_content.setObjectName("mainContent")
        
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        main_content.setLayout(content_layout)
        
        # Header - simplified without title
        header = self.create_header()
        content_layout.addWidget(header)
        
        # Data container
        self.data_container = self.create_data_container()
        content_layout.addWidget(self.data_container)
        
        parent_layout.addWidget(main_content)
    
    def create_header(self) -> QFrame:
        """Create header section"""
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
        """Create data display container"""
        data_container = QFrame()
        data_container.setObjectName("dataContainer")
        
        data_layout = QVBoxLayout()
        data_layout.setContentsMargins(30, 20, 30, 20)
        data_container.setLayout(data_layout)
        
        # Empty state
        self.empty_state = self.create_empty_state()
        data_layout.addWidget(self.empty_state)
        
        # Record info label
        self.record_info = QLabel("")
        self.record_info.setObjectName("recordInfo")
        self.record_info.hide()
        data_layout.addWidget(self.record_info)
        
        # Data table - removed first column
        self.data_table = self.create_data_table()
        data_layout.addWidget(self.data_table)
        
        return data_container
    
    def create_empty_state(self) -> QFrame:
        """Create empty state display"""
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
        """Create data display table"""
        data_table = QTableWidget()
        data_table.setColumnCount(8)  # Removed first column
        # Get timezone from environment or default to EST
        market_tz = os.getenv('MARKET_TIMEZONE', 'US/Eastern').split('/')[-1]
        data_table.setHorizontalHeaderLabels([f'Date ({market_tz})', f'Time ({market_tz})', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP'])
        
        # Set header resize modes
        header = data_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Date
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Time
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)  # Volume
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)  # VWAP stretches
        
        # Hide the vertical header (row numbers)
        data_table.verticalHeader().setVisible(False)
        
        # Disable corner button
        data_table.setCornerButtonEnabled(False)
        
        data_table.setAlternatingRowColors(True)
        data_table.hide()
        
        # Customize scrollbar
        data_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        data_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        return data_table
    
    def setup_styling(self):
        """Apply dark terminal styling with modern scrollbar"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0a0a;
            }
            
            #sidebar {
                background-color: #0f0f0f;
                border-right: 1px solid #1a1a1a;
            }
            
            #sectionTitle {
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #999;
                margin-bottom: 12px;
                font-weight: 600;
            }
            
            #metricsFrame, #tableInfoFrame {
                background-color: #1a1a1a;
                border-radius: 4px;
                padding: 10px;
            }
            
            #metricLabel {
                font-size: 11px;
                color: #4a4;
                text-align: center;
                margin: 2px;
            }
            
            #tableInfoLabel {
                font-size: 10px;
                color: #888;
                text-align: center;
            }
            
            #tierLabel {
                color: #4a4;
                margin-top: 10px;
            }
            
            QLineEdit {
                background-color: #1a1a1a;
                border: 1px solid #2a2a2a;
                color: #fff;
                padding: 12px;
                font-size: 14px;
                border-radius: 4px;
            }
            
            QLineEdit:focus {
                border-color: #3a3a3a;
                background-color: #222;
            }
            
            #rangeButton, #freqButton {
                background-color: #1a1a1a;
                border: 1px solid #2a2a2a;
                color: #999;
                padding: 8px;
                font-size: 12px;
                border-radius: 3px;
            }
            
            #rangeButton:hover, #freqButton:hover {
                background-color: #2a2a2a;
                color: #fff;
            }
            
            #rangeButton:checked, #freqButton:checked {
                background-color: #fff;
                color: #000;
                font-weight: 500;
            }
            
            #fetchButton {
                background-color: #1a1a1a;
                border: 1px solid #4a4a4a;
                color: #fff;
                padding: 12px;
                font-size: 13px;
                border-radius: 4px;
                margin-bottom: 10px;
            }
            
            #fetchButton:hover {
                background-color: #2a2a2a;
                border-color: #5a5a5a;
            }
            
            #actionButton, #exportButton {
                background-color: #1a1a1a;
                border: 1px solid #2a2a2a;
                color: #fff;
                padding: 12px;
                font-size: 13px;
                border-radius: 4px;
                margin-bottom: 10px;
            }
            
            #actionButton:hover:enabled, #exportButton:hover:enabled {
                background-color: #2a2a2a;
                border-color: #3a3a3a;
            }
            
            #actionButton:disabled, #exportButton:disabled {
                background-color: #1a1a1a;
                color: #666;
                border-color: #2a2a2a;
            }
            
            #dangerButton {
                background-color: #1a1a1a;
                border: 1px solid #ff4444;
                color: #ff4444;
                padding: 12px;
                font-size: 13px;
                border-radius: 4px;
            }
            
            #dangerButton:hover:enabled {
                background-color: #ff4444;
                color: #fff;
            }
            
            #dangerButton:disabled {
                background-color: #1a1a1a;
                color: #666;
                border-color: #2a2a2a;
            }
            
            #header {
                background-color: #0f0f0f;
                border-bottom: 1px solid #1a1a1a;
                color: #fff;
            }
            
            #recordCount, #recordInfo {
                font-size: 13px;
                color: #666;
            }
            
            #dataContainer {
                background-color: #0a0a0a;
            }
            
            #emptyText {
                color: #666;
            }
            
            QTableWidget {
                background-color: #0a0a0a;
                border: none;
                gridline-color: #1a1a1a;
                color: #fff;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 13px;
            }
            
            QTableWidget::item {
                padding: 10px 15px;
                border: none;
                border-bottom: 1px solid #1a1a1a;
                color: #fff;
            }
            
            QTableWidget::item:selected {
                background-color: #2a2a2a;
                color: #fff;
            }
            
            QTableWidget::item:alternate {
                background-color: #0f0f0f;
            }
            
            QHeaderView::section {
                background-color: #0a0a0a;
                color: #999;
                padding: 12px 15px;
                border: none;
                border-bottom: 2px solid #2a2a2a;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 11px;
                letter-spacing: 1px;
                text-align: left;
            }
            
            QHeaderView::section:first {
                border-left: none;
            }
            
            QHeaderView::section:last {
                border-right: none;
            }
            
            /* Hide the corner button */
            QTableCornerButton::section {
                background-color: #0a0a0a;
                border: none;
            }
            
            /* Vertical header (if visible) */
            QHeaderView::section:vertical {
                background-color: #0a0a0a;
                border: none;
            }
            
            #bottomActions {
                border-top: 1px solid #1a1a1a;
                padding-top: 20px;
            }
            
            /* Modern Scrollbar Styling */
            QScrollBar:vertical {
                background-color: #0a0a0a;
                width: 12px;
                border-radius: 6px;
                margin: 0;
            }
            
            QScrollBar::handle:vertical {
                background-color: #2a2a2a;
                min-height: 30px;
                border-radius: 6px;
                margin: 2px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #3a3a3a;
            }
            
            QScrollBar::handle:vertical:pressed {
                background-color: #4a4a4a;
            }
            
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
                background: none;
            }
            
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: none;
            }
            
            QScrollBar:horizontal {
                background-color: #0a0a0a;
                height: 12px;
                border-radius: 6px;
                margin: 0;
            }
            
            QScrollBar::handle:horizontal {
                background-color: #2a2a2a;
                min-width: 30px;
                border-radius: 6px;
                margin: 2px;
            }
            
            QScrollBar::handle:horizontal:hover {
                background-color: #3a3a3a;
            }
            
            QScrollBar::handle:horizontal:pressed {
                background-color: #4a4a4a;
            }
            
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0;
                background: none;
            }
            
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: none;
            }
        """)
    
    def check_existing_tables(self):
        """Check if any stock tables exist and enable/disable delete button"""
        try:
            tables = stock_service.get_all_stock_tables()
            self.delete_db_btn.setEnabled(len(tables) > 0)
        except:
            self.delete_db_btn.setEnabled(False)
    
    def set_range(self, range_value: str):
        """Set active time range"""
        self.active_range = range_value
        for btn in self.findChildren(QPushButton):
            if btn.objectName() == "rangeButton":
                btn.setChecked(btn.text() == range_value)
    
    def set_frequency(self, freq_value: str):
        """Set active frequency"""
        self.active_frequency = freq_value
        for btn in self.findChildren(QPushButton):
            if btn.objectName() == "freqButton":
                btn.setChecked(btn.text() == freq_value)
    
    def fetch_stock_data(self):
        """Fetch stock data"""
        symbol = self.ticker_input.text().strip().upper()
        
        if not symbol:
            show_warning(self, "Input Error", "Please enter a ticker symbol")
            return
        
        if not api_config.polygon_api_key:
            show_error(self, "Configuration Error", "Polygon API key not configured.")
            return
        
        # Update UI state
        self.fetch_button.setEnabled(False)
        self.fetch_button.setText("Fetching...")
        self.export_button.setEnabled(False)
        self.clear_metrics()
        
        # Clear previous results
        self.show_empty_state()
        
        # Start worker thread
        self.worker_thread = StockDataWorker(symbol, self.active_range, self.active_frequency)
        self.worker_thread.data_ready.connect(self.display_stock_data)
        self.worker_thread.error_occurred.connect(self.handle_error)
        self.worker_thread.finished.connect(self.worker_finished)
        self.worker_thread.start()
    
    def display_stock_data(self, data: dict):
        """Display fetched stock data"""
        self.current_data = data
        self.current_table = data.get('table_name')
        
        # Update record count only
        self.record_count.setText(f"{data['total_records']:,} records")
        
        # Show performance metrics
        self.update_metrics(data)
        
        # Show table info
        self.update_table_info(data)
        
        # Load data from database
        self.load_data_from_db()
        
        # Enable buttons now that we have data
        self.export_button.setEnabled(True)
        self.save_model_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.delete_db_btn.setEnabled(True)
        
        # Emit signal
        self.stock_data_fetched.emit(data)
    
    def update_table_info(self, data: dict):
        """Update table info display (NEW)"""
        table_name = data.get('table_name', '')
        if table_name:
            # Check if this is a versioned table
            parts = table_name.split('_')
            if len(parts) > 4 and parts[-1].isdigit():
                version = parts[-1]
                self.table_info_frame.show()
                self.table_name_label.setText(f"Table: {table_name}\n(Version {version})")
            else:
                self.table_info_frame.show()
                self.table_name_label.setText(f"Table: {table_name}")
    
    def load_data_from_db(self):
        """Load data from database for display"""
        if not self.current_table:
            return
        
        try:
            # Get data from service
            rows, total_count = stock_service.get_display_data(self.current_table, 1000)
            
            # Update record info
            if total_count > 1000:
                self.record_info.setText(
                    f"Showing first 500 and last 500 of {total_count:,} total records"
                )
                self.record_info.show()
            else:
                self.record_info.hide()
            
            # Display data
            self.show_data_table(rows)
            
        except Exception as e:
            show_error(self, "Display Error", f"Failed to load data: {str(e)}")
    
    def show_data_table(self, rows: list):
        """Display data in table"""
        self.empty_state.hide()
        self.data_table.show()
        self.data_table.setRowCount(len(rows))
        
        # Set row height for better readability
        for i in range(len(rows)):
            self.data_table.setRowHeight(i, 40)
        
        for i, row in enumerate(rows):
            # Date and time (columns 0 and 1)
            date_time = row[0]
            
            # Date item
            date_item = QTableWidgetItem(date_time.strftime('%Y-%m-%d'))
            date_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.data_table.setItem(i, 0, date_item)
            
            # Time item
            time_item = QTableWidgetItem(date_time.strftime('%H:%M:%S'))
            time_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.data_table.setItem(i, 1, time_item)
            
            # OHLCV + VWAP (columns 2-7)
            for j, value in enumerate(row[1:], 2):
                if j == 6:  # Volume
                    item = QTableWidgetItem(f"{int(value):,}")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                else:  # Prices
                    item = QTableWidgetItem(f"{float(value):.2f}")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                
                self.data_table.setItem(i, j, item)
    
    def update_metrics(self, data: dict):
        """Update performance metrics display"""
        exec_time = data.get('execution_time', 0)
        records_per_sec = data.get('records_per_second', 0)
        
        self.exec_time_label.setText(f"Execution: {exec_time:.2f}s")
        self.speed_label.setText(f"Speed: {records_per_sec:,} rec/s")
    
    def clear_metrics(self):
        """Clear performance metrics"""
        self.exec_time_label.clear()
        self.speed_label.clear()
    
    def show_empty_state(self):
        """Show empty state"""
        self.empty_state.show()
        self.data_table.hide()
        self.data_table.setRowCount(0)
        self.record_info.hide()
        self.table_info_frame.hide()
        self.table_name_label.clear()
    
    def handle_error(self, error_message: str):
        """Handle errors"""
        show_error(self, "Error", error_message)
    
    def worker_finished(self):
        """Clean up after worker finishes"""
        self.fetch_button.setEnabled(True)
        self.fetch_button.setText("Fetch Data")
        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None
    
    def export_data(self):
        """Export data summary"""
        if not self.current_data:
            return
        
        filename = f"{self.current_data['symbol']}_{self.current_data['range']}_{self.current_data['frequency']}_summary.json"
        
        try:
            export_info = {
                'symbol': self.current_data['symbol'],
                'range': self.current_data['range'],
                'frequency': self.current_data['frequency'],
                'total_records': self.current_data['total_records'],
                'table_name': self.current_data['table_name'],
                'execution_time': self.current_data.get('execution_time', 0),
                'records_per_second': self.current_data.get('records_per_second', 0),
                'exported_at': datetime.now().isoformat(),
                'database_query': f"SELECT * FROM {self.current_data['table_name']} ORDER BY timestamp"
            }
            
            with open(filename, 'w') as f:
                json.dump(export_info, f, indent=2)
            
            show_info(
                self,
                "Export Complete",
                f"Export summary saved to: {filename}\n"
                f"Full data available in PostgreSQL table: {self.current_table}"
            )
        except Exception as e:
            show_warning(self, "Export Error", str(e))
    
    def save_as_model(self):
        """Save configuration as model"""
        if self.current_data:
            model_config = {
                'symbol': self.current_data['symbol'],
                'range': self.active_range,
                'frequency': self.active_frequency,
                'table_name': self.current_data['table_name'],
                'created_at': datetime.now().isoformat()
            }
            
            filename = f"model_{self.current_data['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            try:
                with open(filename, 'w') as f:
                    json.dump(model_config, f, indent=2)
                
                show_info(
                    self,
                    "Model Saved",
                    f"Model configuration saved to: {filename}"
                )
                
            except Exception as e:
                show_warning(self, "Save Error", str(e))
    
    def clear_data(self):
        """Clear UI and optionally delete database table"""
        if not self.current_table:
            self.clear_ui()
            return
        
        reply = show_data_clear_options(self)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.delete_table_and_clear()
        elif reply == QMessageBox.StandardButton.No:
            self.clear_ui()
    
    def delete_table_and_clear(self):
        """Delete table and clear UI"""
        if not self.current_table:
            return
        
        try:
            stock_service.delete_table(self.current_table)
            show_info(self, "Table Deleted", f"Table '{self.current_table}' deleted.")
            self.check_existing_tables()  # Re-check if any tables remain
        except Exception as e:
            show_error(self, "Delete Error", f"Failed to delete table: {str(e)}")
        
        self.clear_ui()
    
    def clear_ui(self):
        """Clear UI only"""
        self.current_data = None
        self.current_table = None
        self.ticker_input.clear()
        self.record_count.clear()
        self.record_info.clear()
        self.record_info.hide()
        self.clear_metrics()
        self.show_empty_state()
        
        # Disable buttons when UI is cleared
        self.export_button.setEnabled(False)
        self.save_model_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
    
    def delete_database(self):
        """Delete all stock tables"""
        reply = show_confirm_delete(
            self,
            "Delete All Data",
            "Delete ALL stock data tables from the database?"
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                tables = stock_service.get_all_stock_tables()
                
                if not tables:
                    show_info(self, "No Data", "No stock data tables found.")
                    return
                
                count = stock_service.delete_all_tables()
                
                show_info(
                    self, 
                    "Database Cleaned", 
                    f"Deleted {count} stock data tables\n"
                    f"Database space reclaimed."
                )
                
                self.clear_ui()
                self.delete_db_btn.setEnabled(False)  # Disable after deletion
                
            except Exception as e:
                show_error(self, "Error", f"Failed to delete tables: {str(e)}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.terminate()
            self.worker_thread.wait()


class StockFetcherApplication(QApplication):
    """Application wrapper"""
    
    def __init__(self, argv):
        super().__init__(argv)
        self.stock_fetcher = None
    
    def run(self):
        self.stock_fetcher = StockFetcherWidget()
        self.stock_fetcher.show()
        return self.exec()


if __name__ == "__main__":
    app = StockFetcherApplication(sys.argv)
    sys.exit(app.run())