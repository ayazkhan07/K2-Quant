import sys
import json
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                             QTextEdit, QProgressBar, QFrame, QMessageBox,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QPalette, QColor


class StockDataWorker(QThread):
    """Background worker for simulated stock data fetching"""
    
    data_ready = pyqtSignal(dict)
    progress_update = pyqtSignal(int, str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol.upper()
        
    def run(self):
        """Simulate stock data fetching process"""
        try:
            steps = [
                "Connecting to market data...",
                "Fetching historical data...", 
                "Processing price information...",
                "Calculating metrics...",
                "Finalizing data..."
            ]
            
            for i, step in enumerate(steps):
                progress = int((i + 1) / len(steps) * 100)
                self.progress_update.emit(progress, step)
                self.msleep(300)  # Simulate processing time
            
            # Generate mock stock data
            stock_data = self.generate_mock_data()
            self.data_ready.emit(stock_data)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def generate_mock_data(self):
        """Generate realistic mock stock data"""
        base_price = 150.0
        data = {
            'symbol': self.symbol,
            'company_name': f"{self.symbol} Corporation",
            'current_price': round(base_price + (hash(self.symbol) % 100), 2),
            'change': round((hash(self.symbol) % 10) - 5, 2),
            'change_percent': round(((hash(self.symbol) % 10) - 5) / base_price * 100, 2),
            'volume': (hash(self.symbol) % 10000000) + 1000000,
            'market_cap': f"${(hash(self.symbol) % 500 + 50)}B",
            'pe_ratio': round(15 + (hash(self.symbol) % 20), 1),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'historical_data': self.generate_historical_data(base_price)
        }
        return data
    
    def generate_historical_data(self, base_price):
        """Generate mock historical price data"""
        historical = []
        for i in range(30):  # Last 30 days
            date = (datetime.now() - timedelta(days=29-i)).strftime('%Y-%m-%d')
            price_variation = (hash(f"{self.symbol}{i}") % 20) - 10
            price = round(base_price + price_variation, 2)
            historical.append({
                'date': date,
                'open': round(price - 1, 2),
                'high': round(price + 2, 2),
                'low': round(price - 2, 2),
                'close': price,
                'volume': (hash(f"{self.symbol}{i}") % 5000000) + 500000
            })
        return historical


class StockFetcherWidget(QMainWindow):
    """Main stock fetcher interface"""
    
    stock_data_fetched = pyqtSignal(dict)
    back_to_landing = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker_thread = None
        self.current_data = None
        
        self.init_ui()
        self.setup_styling()
        
    def init_ui(self):
        """Initialize the stock fetcher interface"""
        self.setWindowTitle("Stock Price Projection System - Data Fetcher")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        central_widget.setLayout(main_layout)
        
        # Header
        self.create_header(main_layout)
        
        # Input section
        self.create_input_section(main_layout)
        
        # Progress section
        self.create_progress_section(main_layout)
        
        # Results section
        self.create_results_section(main_layout)
        
        # Action buttons
        self.create_action_buttons(main_layout)
    
    def create_header(self, layout):
        """Create header section"""
        header_frame = QFrame()
        header_frame.setFixedHeight(80)
        header_layout = QVBoxLayout()
        header_frame.setLayout(header_layout)
        
        title = QLabel("Stock Data Fetcher")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)
        
        subtitle = QLabel("Enter a stock symbol to fetch real-time market data")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #666666;")
        header_layout.addWidget(subtitle)
        
        layout.addWidget(header_frame)
    
    def create_input_section(self, layout):
        """Create stock symbol input section"""
        input_frame = QFrame()
        input_layout = QHBoxLayout()
        input_frame.setLayout(input_layout)
        
        # Symbol input
        input_layout.addWidget(QLabel("Stock Symbol:"))
        
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter symbol (e.g., AAPL, GOOGL, MSFT)")
        self.symbol_input.setFont(QFont("Arial", 12))
        self.symbol_input.setFixedHeight(40)
        self.symbol_input.returnPressed.connect(self.fetch_stock_data)
        input_layout.addWidget(self.symbol_input)
        
        # Fetch button
        self.fetch_button = QPushButton("Fetch Data")
        self.fetch_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.fetch_button.setFixedHeight(40)
        self.fetch_button.setFixedWidth(120)
        self.fetch_button.clicked.connect(self.fetch_stock_data)
        input_layout.addWidget(self.fetch_button)
        
        layout.addWidget(input_frame)
    
    def create_progress_section(self, layout):
        """Create progress indicator section"""
        progress_frame = QFrame()
        progress_layout = QVBoxLayout()
        progress_frame.setLayout(progress_layout)
        
        self.progress_label = QLabel("Ready to fetch stock data")
        self.progress_label.setFont(QFont("Arial", 10))
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(8)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(progress_frame)
    
    def create_results_section(self, layout):
        """Create results display section"""
        # Stock info display
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        info_layout = QVBoxLayout()
        info_frame.setLayout(info_layout)
        
        info_title = QLabel("Stock Information")
        info_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        info_layout.addWidget(info_title)
        
        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_display.setFixedHeight(200)
        self.info_display.setPlaceholderText("Stock information will appear here after fetching data...")
        info_layout.addWidget(self.info_display)
        
        # Historical data table
        table_frame = QFrame()
        table_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        table_layout = QVBoxLayout()
        table_frame.setLayout(table_layout)
        
        table_title = QLabel("Historical Data (Last 30 Days)")
        table_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        table_layout.addWidget(table_title)
        
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        self.data_table.horizontalHeader().setStretchLastSection(True)
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setFixedHeight(250)
        table_layout.addWidget(self.data_table)
        
        # Add both sections to main layout
        layout.addWidget(info_frame)
        layout.addWidget(table_frame)
    
    def create_action_buttons(self, layout):
        """Create action buttons section"""
        button_frame = QFrame()
        button_layout = QHBoxLayout()
        button_frame.setLayout(button_layout)
        
        # Back button
        back_button = QPushButton("Back to Landing")
        back_button.setFont(QFont("Arial", 11))
        back_button.setFixedHeight(35)
        back_button.clicked.connect(self.go_back_to_landing)
        button_layout.addWidget(back_button)
        
        button_layout.addStretch()
        
        # Export button
        self.export_button = QPushButton("Export Data")
        self.export_button.setFont(QFont("Arial", 11))
        self.export_button.setFixedHeight(35)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_data)
        button_layout.addWidget(self.export_button)
        
        # Continue to Analysis button
        self.analyze_button = QPushButton("Continue to Analysis")
        self.analyze_button.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.analyze_button.setFixedHeight(35)
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.continue_to_analysis)
        button_layout.addWidget(self.analyze_button)
        
        layout.addWidget(button_frame)
    
    def setup_styling(self):
        """Apply consistent styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            
            QFrame {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                padding: 15px;
                margin: 5px;
            }
            
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #005fa3;
            }
            
            QPushButton:pressed {
                background-color: #004080;
            }
            
            QPushButton:disabled {
                background-color: #cccccc;
            }
            
            QLineEdit {
                padding: 8px;
                border: 2px solid #d0d0d0;
                border-radius: 4px;
                background-color: white;
                font-size: 12px;
            }
            
            QLineEdit:focus {
                border-color: #007acc;
            }
            
            QLabel {
                color: #333333;
                font-weight: bold;
            }
            
            QTextEdit {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: #fafafa;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
            
            QTableWidget {
                gridline-color: #e0e0e0;
                background-color: white;
                border: none;
            }
            
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 8px;
                border: 1px solid #d0d0d0;
                font-weight: bold;
            }
            
            QProgressBar {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                text-align: center;
                background-color: #f0f0f0;
            }
            
            QProgressBar::chunk {
                background-color: #007acc;
                border-radius: 3px;
            }
        """)
    
    def fetch_stock_data(self):
        """Fetch stock data for entered symbol"""
        symbol = self.symbol_input.text().strip().upper()
        
        if not symbol:
            QMessageBox.warning(self, "Input Error", "Please enter a stock symbol")
            return
        
        if len(symbol) > 5:
            QMessageBox.warning(self, "Input Error", "Stock symbol too long (max 5 characters)")
            return
        
        # Update UI state
        self.fetch_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.export_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        
        # Clear previous results
        self.info_display.clear()
        self.data_table.setRowCount(0)
        
        # Start worker thread
        self.worker_thread = StockDataWorker(symbol)
        self.worker_thread.progress_update.connect(self.update_progress)
        self.worker_thread.data_ready.connect(self.display_stock_data)
        self.worker_thread.error_occurred.connect(self.handle_error)
        self.worker_thread.finished.connect(self.worker_finished)
        self.worker_thread.start()
    
    def update_progress(self, value, message):
        """Update progress display"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
    
    def display_stock_data(self, data):
        """Display fetched stock data"""
        self.current_data = data
        
        # Display stock information
        info_text = f"""
STOCK INFORMATION
==================

Symbol: {data['symbol']}
Company: {data['company_name']}
Current Price: ${data['current_price']}
Daily Change: ${data['change']} ({data['change_percent']}%)
Volume: {data['volume']:,}
Market Cap: {data['market_cap']}
P/E Ratio: {data['pe_ratio']}

Last Updated: {data['last_updated']}

Status: Data successfully fetched and ready for analysis
        """.strip()
        
        self.info_display.setPlainText(info_text)
        
        # Populate historical data table
        historical = data['historical_data']
        self.data_table.setRowCount(len(historical))
        
        for row, day_data in enumerate(historical):
            self.data_table.setItem(row, 0, QTableWidgetItem(day_data['date']))
            self.data_table.setItem(row, 1, QTableWidgetItem(f"${day_data['open']}"))
            self.data_table.setItem(row, 2, QTableWidgetItem(f"${day_data['high']}"))
            self.data_table.setItem(row, 3, QTableWidgetItem(f"${day_data['low']}"))
            self.data_table.setItem(row, 4, QTableWidgetItem(f"${day_data['close']}"))
            self.data_table.setItem(row, 5, QTableWidgetItem(f"{day_data['volume']:,}"))
        
        # Enable action buttons
        self.export_button.setEnabled(True)
        self.analyze_button.setEnabled(True)
        
        self.progress_label.setText(f"Successfully fetched data for {data['symbol']}")
        
        # Emit signal for external handling
        self.stock_data_fetched.emit(data)
    
    def handle_error(self, error_message):
        """Handle errors during data fetching"""
        QMessageBox.critical(self, "Error", f"Failed to fetch stock data:\n{error_message}")
        self.progress_label.setText("Error occurred during data fetching")
    
    def worker_finished(self):
        """Clean up after worker thread finishes"""
        self.progress_bar.setVisible(False)
        self.fetch_button.setEnabled(True)
        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None
    
    def export_data(self):
        """Export current stock data"""
        if not self.current_data:
            return
        
        filename = f"stock_data_{self.current_data['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.current_data, f, indent=2)
            
            QMessageBox.information(self, "Export Complete", f"Stock data exported to:\n{filename}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export data:\n{str(e)}")
    
    def continue_to_analysis(self):
        """Continue to analysis phase"""
        if self.current_data:
            QMessageBox.information(self, "Continue to Analysis", 
                                  f"Proceeding to analysis with {self.current_data['symbol']} data.\n\n"
                                  "This would transition to the prediction/analysis interface.")
    
    def go_back_to_landing(self):
        """Return to landing page"""
        self.back_to_landing.emit()
    
    def cleanup(self):
        """Clean up resources"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.terminate()
            self.worker_thread.wait()


class StockFetcherApplication(QApplication):
    """Standalone application for testing the stock fetcher"""
    
    def __init__(self, argv):
        super().__init__(argv)
        self.stock_fetcher = None
    
    def run(self):
        """Run the stock fetcher application"""
        self.stock_fetcher = StockFetcherWidget()
        self.stock_fetcher.show()
        
        return self.exec()


if __name__ == "__main__":
    app = StockFetcherApplication(sys.argv)
    sys.exit(app.run())