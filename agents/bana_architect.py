import os
import sys
import asyncio
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path

from .base_agent import BaseAgent


class BanaArchitect(BaseAgent):
    """Senior Developer/Architect Agent for Stock Price Projection System"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'database_path': 'data/stocks.db',
            'model_save_path': 'models/',
            'supported_models': ['LSTM', 'ARIMA', 'Prophet', 'XGBoost'],
            'default_prediction_days': 30,
            'api_keys': {}
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(
            name="Bana",
            role="Senior Developer/Architect",
            config=default_config
        )
        
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the system architecture and core components"""
        self.update_status("initializing", "Setting up system architecture")
        
        # Create necessary directories
        for directory in ['data', 'models', 'logs', 'config', 'tests', 'reports']:
            Path(directory).mkdir(exist_ok=True)
        
        # Initialize database
        self.setup_database()
        
        # Initialize model registry
        self.model_registry = {}
        
        self.update_status("ready", "System architecture initialized")
        
    def setup_database(self):
        """Setup SQLite database for stock data storage"""
        db_path = self.config['database_path']
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    prediction_date DATE NOT NULL,
                    predicted_price REAL NOT NULL,
                    confidence_interval_low REAL,
                    confidence_interval_high REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    accuracy_score REAL,
                    mae REAL,
                    rmse REAL,
                    mape REAL,
                    evaluation_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
        
        self.logger.info("Database setup completed")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute architecture and development tasks"""
        task_type = task.get('type')
        
        try:
            if task_type == 'fetch_stock_data':
                return await self.fetch_stock_data(task['symbol'], task.get('period', '1y'))
            
            elif task_type == 'create_prediction_model':
                return await self.create_prediction_model(
                    task['model_type'], 
                    task['symbol'],
                    task.get('parameters', {})
                )
            
            elif task_type == 'generate_predictions':
                return await self.generate_predictions(
                    task['symbol'],
                    task.get('model_type', 'LSTM'),
                    task.get('days', 30)
                )
            
            elif task_type == 'create_desktop_component':
                return await self.create_desktop_component(
                    task['component_type'],
                    task.get('specifications', {})
                )
            
            elif task_type == 'system_health_check':
                return await self.system_health_check()
            
            elif task_type == 'optimize_performance':
                return await self.optimize_performance(task.get('target_area'))
            
            else:
                return {
                    'success': False,
                    'error': f'Unknown task type: {task_type}'
                }
                
        except Exception as e:
            self.logger.error(f"Task execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def fetch_stock_data(self, symbol: str, period: str = '1y') -> Dict[str, Any]:
        """Fetch and store stock data"""
        self.update_status("fetching_data", f"Retrieving data for {symbol}")
        
        try:
            # Fetch data using yfinance
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            # Store in database
            db_path = self.config['database_path']
            with sqlite3.connect(db_path) as conn:
                for date, row in data.iterrows():
                    conn.execute('''
                        INSERT OR REPLACE INTO stocks 
                        (symbol, date, open, high, low, close, volume, adj_close)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, date.strftime('%Y-%m-%d'),
                        row['Open'], row['High'], row['Low'], 
                        row['Close'], row['Volume'], row['Adj Close']
                    ))
                conn.commit()
            
            return {
                'success': True,
                'symbol': symbol,
                'records_count': len(data),
                'date_range': f"{data.index.min()} to {data.index.max()}",
                'data_preview': data.tail().to_dict()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to fetch data for {symbol}: {str(e)}"
            }
    
    async def create_prediction_model(self, model_type: str, symbol: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create and configure prediction models"""
        self.update_status("creating_model", f"Building {model_type} model for {symbol}")
        
        # This is a framework - actual ML model implementations would be more complex
        model_config = {
            'type': model_type,
            'symbol': symbol,
            'parameters': parameters,
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        # Save model configuration
        model_path = f"{self.config['model_save_path']}/{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(model_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Register model
        model_id = f"{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_registry[model_id] = model_config
        
        return {
            'success': True,
            'model_id': model_id,
            'model_path': model_path,
            'configuration': model_config
        }
    
    async def generate_predictions(self, symbol: str, model_type: str, days: int) -> Dict[str, Any]:
        """Generate stock price predictions"""
        self.update_status("generating_predictions", f"Creating {days}-day predictions for {symbol}")
        
        # Fetch recent data for prediction base
        db_path = self.config['database_path']
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query('''
                SELECT * FROM stocks 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 100
            ''', conn, params=(symbol,))
        
        if df.empty:
            return {
                'success': False,
                'error': f"No data available for {symbol}"
            }
        
        # Simple prediction logic (placeholder for complex ML models)
        last_price = df.iloc[0]['close']
        predictions = []
        
        for i in range(1, days + 1):
            # Simple trend-based prediction (replace with actual ML models)
            trend_factor = np.random.normal(1.001, 0.02)  # Slight upward trend with volatility
            predicted_price = last_price * (trend_factor ** i)
            
            prediction_date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            
            predictions.append({
                'date': prediction_date,
                'predicted_price': round(predicted_price, 2),
                'confidence_low': round(predicted_price * 0.95, 2),
                'confidence_high': round(predicted_price * 1.05, 2)
            })
        
        # Store predictions in database
        with sqlite3.connect(db_path) as conn:
            for pred in predictions:
                conn.execute('''
                    INSERT INTO predictions 
                    (symbol, model_type, prediction_date, predicted_price, 
                     confidence_interval_low, confidence_interval_high)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, model_type, pred['date'], pred['predicted_price'],
                    pred['confidence_low'], pred['confidence_high']
                ))
            conn.commit()
        
        return {
            'success': True,
            'symbol': symbol,
            'model_type': model_type,
            'predictions': predictions,
            'base_price': last_price,
            'prediction_count': len(predictions)
        }
    
    async def create_desktop_component(self, component_type: str, specifications: Dict[str, Any]) -> Dict[str, Any]:
        """Create desktop application components"""
        self.update_status("creating_component", f"Building {component_type} component")
        
        component_templates = {
            'main_window': self.create_main_window_template,
            'chart_widget': self.create_chart_widget_template,
            'data_table': self.create_data_table_template,
            'prediction_panel': self.create_prediction_panel_template
        }
        
        if component_type in component_templates:
            template = component_templates[component_type](specifications)
            
            # Save component file
            component_path = f"app/components/{component_type}.py"
            with open(component_path, 'w') as f:
                f.write(template)
            
            return {
                'success': True,
                'component_type': component_type,
                'file_path': component_path,
                'specifications': specifications
            }
        
        return {
            'success': False,
            'error': f'Unknown component type: {component_type}'
        }
    
    def create_main_window_template(self, specs: Dict[str, Any]) -> str:
        """Generate main window component template"""
        return '''
import sys
import asyncio
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLineEdit, QLabel, QTableWidget, 
                             QTableWidgetItem, QComboBox, QSpinBox, QTextEdit, 
                             QSplitter, QTabWidget, QProgressBar, QStatusBar,
                             QMenuBar, QMenu, QMessageBox, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QIcon
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QDateTimeAxis, QValueAxis
import pyqtgraph as pg
from datetime import datetime, timedelta
import pandas as pd

class StockDataWorker(QThread):
    """Background worker for stock data operations"""
    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, symbol, operation):
        super().__init__()
        self.symbol = symbol
        self.operation = operation
    
    def run(self):
        try:
            # Simulate backend operations
            for i in range(101):
                self.progress_updated.emit(i)
                self.msleep(10)
            
            # Mock data - replace with actual backend calls
            result = {
                'symbol': self.symbol,
                'data': {'status': 'success'},
                'timestamp': datetime.now().isoformat()
            }
            self.data_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

class PredictionChart(QWidget):
    """Custom chart widget for displaying predictions"""
    def __init__(self):
        super().__init__()
        self.init_chart()
    
    def init_chart(self):
        layout = QVBoxLayout()
        
        # Create pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('white')
        self.plot_widget.setLabel('left', 'Stock Price ($)')
        self.plot_widget.setLabel('bottom', 'Date')
        self.plot_widget.showGrid(x=True, y=True)
        
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
    
    def update_chart(self, historical_data, predictions):
        """Update chart with historical and predicted data"""
        self.plot_widget.clear()
        
        # Plot historical data
        if historical_data:
            self.plot_widget.plot(
                historical_data['dates'], 
                historical_data['prices'], 
                pen=pg.mkPen(color='blue', width=2),
                name='Historical'
            )
        
        # Plot predictions
        if predictions:
            self.plot_widget.plot(
                predictions['dates'], 
                predictions['prices'], 
                pen=pg.mkPen(color='red', width=2, style=Qt.PenStyle.DashLine),
                name='Predicted'
            )

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Price Projection System")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        # Initialize components
        self.worker_thread = None
        self.prediction_data = {}
        
        self.init_ui()
        self.init_status_bar()
        self.init_menu_bar()
        self.setup_connections()
        
        # Apply modern styling
        self.setStyleSheet(self.get_modern_stylesheet())
    
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Model Strategy Indicator panel - Controls
        model_strat_indicator_panel = self.create_control_panel()
        splitter.addWidget(model_strat_indicator_panel)
        
        # ConvoAI panel - Results and Charts
        convoAI_panel = self.create_results_panel()
        splitter.addWidget(convoAI_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 1000])
    
    def create_control_panel(self):
        """Create the left control panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Title
        title = QLabel("Stock Analysis Controls")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Stock symbol input
        layout.addWidget(QLabel("Stock Symbol:"))
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("e.g., AAPL, GOOGL, MSFT")
        self.symbol_input.setText("AAPL")
        layout.addWidget(self.symbol_input)
        
        # Model selection
        layout.addWidget(QLabel("Prediction Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["LSTM", "ARIMA", "Prophet", "XGBoost"])
        layout.addWidget(self.model_combo)
        
        # Prediction days
        layout.addWidget(QLabel("Prediction Days:"))
        self.days_spin = QSpinBox()
        self.days_spin.setRange(1, 365)
        self.days_spin.setValue(30)
        layout.addWidget(self.days_spin)
        
        # Action buttons
        button_layout = QVBoxLayout()
        
        self.fetch_button = QPushButton("Fetch Stock Data")
        self.fetch_button.setMinimumHeight(40)
        button_layout.addWidget(self.fetch_button)
        
        self.predict_button = QPushButton("Generate Predictions")
        self.predict_button.setMinimumHeight(40)
        self.predict_button.setEnabled(False)
        button_layout.addWidget(self.predict_button)
        
        self.analyze_button = QPushButton("Run Analysis")
        self.analyze_button.setMinimumHeight(40)
        button_layout.addWidget(self.analyze_button)
        
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(200)
        self.status_text.setPlaceholderText("System status messages will appear here...")
        layout.addWidget(self.status_text)
        
        layout.addStretch()
        return panel
    
    def create_results_panel(self):
        """Create the right results panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Chart tab
        self.prediction_chart = PredictionChart()
        self.tab_widget.addTab(self.prediction_chart, "Price Chart")
        
        # Data table tab
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(7)
        self.data_table.setHorizontalHeaderLabels([
            "Date", "Open", "High", "Low", "Close", "Volume", "Predicted"
        ])
        self.tab_widget.addTab(self.data_table, "Data Table")
        
        # Analysis tab
        self.analysis_text = QTextEdit()
        self.analysis_text.setPlaceholderText("Analysis results will appear here...")
        self.tab_widget.addTab(self.analysis_text, "Analysis Results")
        
        return panel
    
    def init_status_bar(self):
        """Initialize status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def init_menu_bar(self):
        """Initialize menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        file_menu.addAction('Import Data', self.import_data)
        file_menu.addAction('Export Results', self.export_results)
        file_menu.addSeparator()
        file_menu.addAction('Exit', self.close)
        
        # View menu
        view_menu = menubar.addMenu('View')
        view_menu.addAction('Refresh Data', self.refresh_data)
        view_menu.addAction('Clear Results', self.clear_results)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        help_menu.addAction('About', self.show_about)
    
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.fetch_button.clicked.connect(self.fetch_stock_data)
        self.predict_button.clicked.connect(self.generate_predictions)
        self.analyze_button.clicked.connect(self.run_analysis)
    
    def fetch_stock_data(self):
        """Fetch stock data in background thread"""
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            self.show_message("Please enter a stock symbol")
            return
        
        self.progress_bar.setVisible(True)
        self.fetch_button.setEnabled(False)
        self.status_bar.showMessage(f"Fetching data for {symbol}...")
        
        # Start background worker
        self.worker_thread = StockDataWorker(symbol, 'fetch')
        self.worker_thread.data_ready.connect(self.on_data_ready)
        self.worker_thread.error_occurred.connect(self.on_error)
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.finished.connect(self.on_operation_finished)
        self.worker_thread.start()
    
    def generate_predictions(self):
        """Generate stock price predictions"""
        symbol = self.symbol_input.text().strip().upper()
        model = self.model_combo.currentText()
        days = self.days_spin.value()
        
        self.status_bar.showMessage(f"Generating {days}-day predictions using {model}...")
        self.log_message(f"Starting prediction generation for {symbol} using {model} model")
        
        # Mock prediction data - replace with actual backend call
        prediction_dates = pd.date_range(
            start=datetime.now(), 
            periods=days, 
            freq='D'
        )
        
        # Update chart with mock data
        historical_data = {
            'dates': list(range(30)),
            'prices': [100 + i + (i % 5) * 2 for i in range(30)]
        }
        
        predictions = {
            'dates': list(range(30, 30 + days)),
            'prices': [130 + i * 0.5 + (i % 3) for i in range(days)]
        }
        
        self.prediction_chart.update_chart(historical_data, predictions)
        self.populate_data_table(historical_data, predictions)
        
        self.status_bar.showMessage("Predictions generated successfully")
        self.log_message("Prediction generation completed")
    
    def run_analysis(self):
        """Run comprehensive analysis"""
        self.log_message("Running comprehensive analysis...")
        
        analysis_result = """
        STOCK ANALYSIS REPORT
        =====================
        
        Symbol: AAPL
        Analysis Date: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
        
        TECHNICAL INDICATORS:
        - Moving Average (20-day): $150.25
        - RSI: 65.4 (Neutral)
        - MACD: Bullish Signal
        
        PREDICTION SUMMARY:
        - Model Used: LSTM
        - Confidence Level: 78%
        - 30-day Trend: Slightly Bullish
        - Expected Range: $145 - $165
        
        RISK ASSESSMENT:
        - Volatility: Medium
        - Market Correlation: 0.72
        - Recommendation: HOLD
        """
        
        self.analysis_text.setPlainText(analysis_result)
        self.tab_widget.setCurrentIndex(2)  # Switch to analysis tab
        self.status_bar.showMessage("Analysis completed")
    
    def populate_data_table(self, historical_data, predictions):
        """Populate the data table with results"""
        total_rows = len(historical_data['dates']) + len(predictions['dates'])
        self.data_table.setRowCount(total_rows)
        
        # Add historical data
        for i, (date_idx, price) in enumerate(zip(historical_data['dates'], historical_data['prices'])):
            date_str = (datetime.now() - timedelta(days=30-i)).strftime('%Y-%m-%d')
            self.data_table.setItem(i, 0, QTableWidgetItem(date_str))
            self.data_table.setItem(i, 4, QTableWidgetItem(f"${price:.2f}"))
        
        # Add predictions
        for i, (date_idx, price) in enumerate(zip(predictions['dates'], predictions['prices'])):
            row = len(historical_data['dates']) + i
            date_str = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
            self.data_table.setItem(row, 0, QTableWidgetItem(date_str))
            self.data_table.setItem(row, 6, QTableWidgetItem(f"${price:.2f}"))
    
    def on_data_ready(self, data):
        """Handle data ready signal"""
        self.log_message(f"Data received for {data['symbol']}")
        self.predict_button.setEnabled(True)
    
    def on_error(self, error_message):
        """Handle error signal"""
        self.show_message(f"Error: {error_message}")
        self.log_message(f"Error occurred: {error_message}")
    
    def on_operation_finished(self):
        """Handle operation finished"""
        self.progress_bar.setVisible(False)
        self.fetch_button.setEnabled(True)
        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None
    
    def log_message(self, message):
        """Add message to status text"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.status_text.append(f"[{timestamp}] {message}")
    
    def show_message(self, message):
        """Show message box"""
        QMessageBox.information(self, "Information", message)
    
    def import_data(self):
        """Import data functionality"""
        self.show_message("Import data functionality would be implemented here")
    
    def export_results(self):
        """Export results functionality"""
        self.show_message("Export results functionality would be implemented here")
    
    def refresh_data(self):
        """Refresh data"""
        self.log_message("Refreshing data...")
    
    def clear_results(self):
        """Clear all results"""
        self.prediction_chart.plot_widget.clear()
        self.data_table.setRowCount(0)
        self.analysis_text.clear()
        self.status_text.clear()
        self.log_message("Results cleared")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About", 
            "Stock Price Projection System\\n\\n"
            "Professional-grade stock analysis and prediction platform\\n"
            "Built with PyQt6 and advanced ML models")
    
    def get_modern_stylesheet(self):
        """Return modern CSS stylesheet"""
        return """
        QMainWindow {
            background-color: #f5f5f5;
        }
        
        QFrame {
            background-color: white;
            border: 1px solid #d0d0d0;
            border-radius: 5px;
            margin: 2px;
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
        
        QLineEdit, QComboBox, QSpinBox {
            padding: 8px;
            border: 2px solid #d0d0d0;
            border-radius: 4px;
            background-color: white;
        }
        
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
            border-color: #007acc;
        }
        
        QLabel {
            color: #333333;
            font-weight: bold;
            margin-top: 5px;
        }
        
        QTabWidget::pane {
            border: 1px solid #d0d0d0;
            background-color: white;
        }
        
        QTabBar::tab {
            background-color: #e0e0e0;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 2px solid #007acc;
        }
        
        QTableWidget {
            gridline-color: #d0d0d0;
            background-color: white;
        }
        
        QHeaderView::section {
            background-color: #f0f0f0;
            padding: 8px;
            border: 1px solid #d0d0d0;
            font-weight: bold;
        }
        
        QTextEdit {
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            background-color: white;
            font-family: 'Consolas', 'Monaco', monospace;
        }
        
        QProgressBar {
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            text-align: center;
        }
        
        QProgressBar::chunk {
            background-color: #007acc;
            border-radius: 3px;
        }
        """

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
'''
    
    def create_chart_widget_template(self, specs: Dict[str, Any]) -> str:
        """Generate chart widget component template"""
        return '''
import sys
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
import numpy as np
from datetime import datetime, timedelta

class ChartWidget(QWidget):
    """Standalone chart widget for stock price visualization"""
    
    chart_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.historical_data = {}
        self.prediction_data = {}
    
    def init_ui(self):
        """Initialize the chart interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Chart controls
        controls_layout = QHBoxLayout()
        
        # Chart type selector
        controls_layout.addWidget(QLabel("Chart Type:"))
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["Line Chart", "Candlestick", "Volume"])
        self.chart_type_combo.currentTextChanged.connect(self.update_chart_type)
        controls_layout.addWidget(self.chart_type_combo)
        
        # Time range selector
        controls_layout.addWidget(QLabel("Time Range:"))
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(["1M", "3M", "6M", "1Y", "2Y", "All"])
        self.time_range_combo.currentTextChanged.connect(self.update_time_range)
        controls_layout.addWidget(self.time_range_combo)
        
        controls_layout.addStretch()
        
        # Export button
        export_btn = QPushButton("Export Chart")
        export_btn.clicked.connect(self.export_chart)
        controls_layout.addWidget(export_btn)
        
        layout.addLayout(controls_layout)
        
        # Main chart area
        self.plot_widget = pg.PlotWidget()
        self.setup_chart_appearance()
        layout.addWidget(self.plot_widget)
        
        # Chart legend and info
        info_layout = QHBoxLayout()
        self.info_label = QLabel("Ready for data...")
        info_layout.addWidget(self.info_label)
        info_layout.addStretch()
        layout.addLayout(info_layout)
    
    def setup_chart_appearance(self):
        """Setup chart visual appearance"""
        self.plot_widget.setBackground('white')
        self.plot_widget.setLabel('left', 'Price ($)', color='black', size=12)
        self.plot_widget.setLabel('bottom', 'Date', color='black', size=12)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Add crosshair
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen='gray')
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen='gray')
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)
        
        # Mouse tracking
        self.plot_widget.scene().sigMouseMoved.connect(self.mouse_moved)
    
    def mouse_moved(self, pos):
        """Handle mouse movement for crosshair"""
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)
            self.crosshair_v.setPos(mouse_point.x())
            self.crosshair_h.setPos(mouse_point.y())
            
            # Update info label
            self.info_label.setText(f"Price: ${mouse_point.y():.2f}")
    
    def update_data(self, historical_data, prediction_data=None):
        """Update chart with new data"""
        self.historical_data = historical_data
        self.prediction_data = prediction_data or {}
        self.refresh_chart()
    
    def refresh_chart(self):
        """Refresh the chart display"""
        self.plot_widget.clear()
        
        # Re-add crosshairs
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)
        
        if not self.historical_data:
            return
        
        # Plot historical data
        if 'dates' in self.historical_data and 'prices' in self.historical_data:
            historical_pen = pg.mkPen(color='blue', width=2)
            self.plot_widget.plot(
                self.historical_data['dates'],
                self.historical_data['prices'],
                pen=historical_pen,
                name='Historical'
            )
        
        # Plot predictions
        if self.prediction_data and 'dates' in self.prediction_data:
            prediction_pen = pg.mkPen(color='red', width=2, style=Qt.PenStyle.DashLine)
            self.plot_widget.plot(
                self.prediction_data['dates'],
                self.prediction_data['prices'],
                pen=prediction_pen,
                name='Predicted'
            )
            
            # Add confidence intervals if available
            if 'confidence_low' in self.prediction_data and 'confidence_high' in self.prediction_data:
                fill_brush = pg.mkBrush(color=(255, 0, 0, 50))
                self.plot_widget.plot(
                    self.prediction_data['dates'],
                    self.prediction_data['confidence_low'],
                    pen=pg.mkPen(color='red', width=1, style=Qt.PenStyle.DotLine)
                )
                self.plot_widget.plot(
                    self.prediction_data['dates'],
                    self.prediction_data['confidence_high'],
                    pen=pg.mkPen(color='red', width=1, style=Qt.PenStyle.DotLine)
                )
        
        # Update info
        total_points = len(self.historical_data.get('prices', []))
        pred_points = len(self.prediction_data.get('prices', []))
        self.info_label.setText(f"Historical: {total_points} points | Predictions: {pred_points} points")
        
        # Emit update signal
        self.chart_updated.emit({
            'historical_points': total_points,
            'prediction_points': pred_points
        })
    
    def update_chart_type(self, chart_type):
        """Update chart display type"""
        # Implementation would depend on chart type
        self.refresh_chart()
    
    def update_time_range(self, time_range):
        """Update visible time range"""
        # Implementation would filter data based on time range
        self.refresh_chart()
    
    def export_chart(self):
        """Export chart to image file"""
        exporter = pg.exporters.ImageExporter(self.plot_widget.plotItem)
        exporter.export(f'chart_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    widget = ChartWidget()
    
    # Sample data
    dates = list(range(30))
    prices = [100 + i + np.sin(i/5) * 10 for i in dates]
    widget.update_data({'dates': dates, 'prices': prices})
    
    widget.show()
    sys.exit(app.exec())
'''
    
    def create_data_table_template(self, specs: Dict[str, Any]) -> str:
        """Generate data table component template"""
        return '''
import sys
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QTableWidget, QTableWidgetItem, QHeaderView, 
                             QLabel, QLineEdit, QComboBox, QMessageBox,
                             QMenu, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal, QSortFilterProxyModel, QAbstractTableModel
from PyQt6.QtGui import QAction, QColor
from datetime import datetime
import pandas as pd
import json

class StockDataTableModel(QAbstractTableModel):
    """Custom table model for stock data"""
    
    def __init__(self, data=None):
        super().__init__()
        self._data = data or pd.DataFrame()
        self.headers = []
    
    def rowCount(self, parent=None):
        return len(self._data)
    
    def columnCount(self, parent=None):
        return len(self._data.columns) if not self._data.empty else 0
    
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if pd.isna(value):
                return ""
            return str(value)
        
        elif role == Qt.ItemDataRole.BackgroundRole:
            # Highlight prediction rows
            if index.column() == len(self._data.columns) - 1:  # Prediction column
                value = self._data.iloc[index.row(), index.column()]
                if pd.notna(value):
                    return QColor(255, 240, 240)  # Light red for predictions
        
        return None
    
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            else:
                return str(section + 1)
        return None
    
    def update_data(self, new_data):
        """Update the model with new data"""
        self.beginResetModel()
        self._data = new_data
        self.endResetModel()

class DataTableWidget(QWidget):
    """Advanced data table widget for stock information"""
    
    data_selected = pyqtSignal(dict)
    export_requested = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = StockDataTableModel()
        self.init_ui()
        self.setup_context_menu()
    
    def init_ui(self):
        """Initialize the table interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Table controls
        controls_layout = QHBoxLayout()
        
        # Search functionality
        controls_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search in table data...")
        self.search_input.textChanged.connect(self.filter_data)
        controls_layout.addWidget(self.search_input)
        
        # Column filter
        controls_layout.addWidget(QLabel("Show Column:"))
        self.column_filter = QComboBox()
        self.column_filter.addItem("All Columns")
        self.column_filter.currentTextChanged.connect(self.toggle_column_visibility)
        controls_layout.addWidget(self.column_filter)
        
        controls_layout.addStretch()
        
        # Action buttons
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_data)
        controls_layout.addWidget(refresh_btn)
        
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self.export_to_csv)
        controls_layout.addWidget(export_btn)
        
        layout.addLayout(controls_layout)
        
        # Main table
        self.table_widget = QTableWidget()
        self.setup_table()
        layout.addWidget(self.table_widget)
        
        # Summary information
        summary_layout = QHBoxLayout()
        self.summary_label = QLabel("No data loaded")
        summary_layout.addWidget(self.summary_label)
        summary_layout.addStretch()
        
        self.selection_label = QLabel("")
        summary_layout.addWidget(self.selection_label)
        
        layout.addLayout(summary_layout)
    
    def setup_table(self):
        """Setup table properties and behavior"""
        # Table properties
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_widget.setSortingEnabled(True)
        
        # Headers
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        
        # Signals
        self.table_widget.itemSelectionChanged.connect(self.on_selection_changed)
        self.table_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
    
    def setup_context_menu(self):
        """Setup right-click context menu"""
        self.table_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_widget.customContextMenuRequested.connect(self.show_context_menu)
    
    def show_context_menu(self, position):
        """Show context menu at position"""
        item = self.table_widget.itemAt(position)
        if item is None:
            return
        
        menu = QMenu(self)
        
        copy_action = QAction("Copy Cell", self)
        copy_action.triggered.connect(lambda: self.copy_cell_value(item))
        menu.addAction(copy_action)
        
        copy_row_action = QAction("Copy Row", self)
        copy_row_action.triggered.connect(lambda: self.copy_row_data(item.row()))
        menu.addAction(copy_row_action)
        
        menu.addSeparator()
        
        highlight_action = QAction("Highlight Row", self)
        highlight_action.triggered.connect(lambda: self.highlight_row(item.row()))
        menu.addAction(highlight_action)
        
        menu.exec(self.table_widget.mapToGlobal(position))
    
    def copy_cell_value(self, item):
        """Copy single cell value to clipboard"""
        if item:
            QApplication.clipboard().setText(item.text())
    
    def copy_row_data(self, row):
        """Copy entire row data to clipboard"""
        if row < 0 or row >= self.table_widget.rowCount():
            return
        
        row_data = []
        for col in range(self.table_widget.columnCount()):
            item = self.table_widget.item(row, col)
            row_data.append(item.text() if item else "")
        
        QApplication.clipboard().setText("\\t".join(row_data))
    
    def highlight_row(self, row):
        """Highlight a specific row"""
        for col in range(self.table_widget.columnCount()):
            item = self.table_widget.item(row, col)
            if item:
                item.setBackground(QColor(255, 255, 0, 100))  # Light yellow
    
    def update_data(self, data_dict):
        """Update table with new data"""
        try:
            # Convert dict to DataFrame if needed
            if isinstance(data_dict, dict):
                if 'historical' in data_dict and 'predictions' in data_dict:
                    # Combine historical and prediction data
                    df = self.combine_data(data_dict['historical'], data_dict['predictions'])
                else:
                    df = pd.DataFrame(data_dict)
            else:
                df = data_dict
            
            # Update table widget
            self.table_widget.setRowCount(len(df))
            self.table_widget.setColumnCount(len(df.columns))
            self.table_widget.setHorizontalHeaderLabels(list(df.columns))
            
            # Populate data
            for row in range(len(df)):
                for col in range(len(df.columns)):
                    value = df.iloc[row, col]
                    item = QTableWidgetItem(str(value) if pd.notna(value) else "")
                    
                    # Special formatting for certain columns
                    if 'price' in df.columns[col].lower() or 'close' in df.columns[col].lower():
                        try:
                            price_val = float(value)
                            item.setText(f"${price_val:.2f}")
                        except (ValueError, TypeError):
                            pass
                    
                    # Color code predictions
                    if 'predicted' in df.columns[col].lower() and pd.notna(value):
                        item.setBackground(QColor(255, 240, 240))
                    
                    self.table_widget.setItem(row, col, item)
            
            # Update column filter
            self.update_column_filter(df.columns)
            
            # Update summary
            self.update_summary(df)
            
        except Exception as e:
            QMessageBox.warning(self, "Data Update Error", f"Failed to update table: {str(e)}")
    
    def combine_data(self, historical, predictions):
        """Combine historical and prediction data"""
        hist_df = pd.DataFrame(historical)
        pred_df = pd.DataFrame(predictions)
        
        # Add prediction column to historical data
        hist_df['Predicted'] = None
        
        # Create combined DataFrame
        if not pred_df.empty:
            pred_df['Predicted'] = pred_df.get('prices', pred_df.get('close', None))
            combined = pd.concat([hist_df, pred_df], ignore_index=True, sort=False)
        else:
            combined = hist_df
        
        return combined
    
    def update_column_filter(self, columns):
        """Update the column filter dropdown"""
        self.column_filter.clear()
        self.column_filter.addItem("All Columns")
        for col in columns:
            self.column_filter.addItem(col)
    
    def update_summary(self, df):
        """Update summary information"""
        if df.empty:
            self.summary_label.setText("No data available")
            return
        
        total_rows = len(df)
        historical_rows = df[df['Predicted'].isna()].shape[0] if 'Predicted' in df.columns else total_rows
        prediction_rows = total_rows - historical_rows
        
        self.summary_label.setText(
            f"Total: {total_rows} rows | Historical: {historical_rows} | Predictions: {prediction_rows}"
        )
    
    def filter_data(self, text):
        """Filter table data based on search text"""
        for row in range(self.table_widget.rowCount()):
            show_row = False
            if not text:
                show_row = True
            else:
                for col in range(self.table_widget.columnCount()):
                    item = self.table_widget.item(row, col)
                    if item and text.lower() in item.text().lower():
                        show_row = True
                        break
            
            self.table_widget.setRowHidden(row, not show_row)
    
    def toggle_column_visibility(self, column_name):
        """Toggle column visibility"""
        if column_name == "All Columns":
            for col in range(self.table_widget.columnCount()):
                self.table_widget.setColumnHidden(col, False)
        else:
            for col in range(self.table_widget.columnCount()):
                header_item = self.table_widget.horizontalHeaderItem(col)
                if header_item:
                    is_target_column = header_item.text() == column_name
                    self.table_widget.setColumnHidden(col, not is_target_column)
    
    def on_selection_changed(self):
        """Handle selection changes"""
        selected_items = self.table_widget.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            self.selection_label.setText(f"Selected row: {row + 1}")
            
            # Emit selection data
            row_data = {}
            for col in range(self.table_widget.columnCount()):
                header = self.table_widget.horizontalHeaderItem(col)
                item = self.table_widget.item(row, col)
                if header and item:
                    row_data[header.text()] = item.text()
            
            self.data_selected.emit(row_data)
        else:
            self.selection_label.setText("")
    
    def on_item_double_clicked(self, item):
        """Handle double-click on item"""
        QMessageBox.information(self, "Cell Value", f"Value: {item.text()}")
    
    def refresh_data(self):
        """Refresh table data"""
        # Implementation would reconnect to data source
        pass
    
    def export_to_csv(self):
        """Export table data to CSV"""
        try:
            data = []
            headers = []
            
            # Get headers
            for col in range(self.table_widget.columnCount()):
                header = self.table_widget.horizontalHeaderItem(col)
                headers.append(header.text() if header else f"Column_{col}")
            
            # Get data
            for row in range(self.table_widget.rowCount()):
                if not self.table_widget.isRowHidden(row):
                    row_data = []
                    for col in range(self.table_widget.columnCount()):
                        item = self.table_widget.item(row, col)
                        row_data.append(item.text() if item else "")
                    data.append(row_data)
            
            # Create DataFrame and export
            df = pd.DataFrame(data, columns=headers)
            filename = f"stock_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            
            QMessageBox.information(self, "Export Complete", f"Data exported to {filename}")
            self.export_requested.emit(filename)
            
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export data: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = DataTableWidget()
    
    # Sample data
    sample_data = {
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Open': [100.0, 102.0, 104.0],
        'High': [105.0, 106.0, 107.0],
        'Low': [98.0, 100.0, 102.0],
        'Close': [102.0, 104.0, 106.0],
        'Volume': [1000000, 1100000, 1200000],
        'Predicted': [None, None, 108.0]
    }
    
    widget.update_data(sample_data)
    widget.show()
    sys.exit(app.exec())
'''
    
    def create_prediction_panel_template(self, specs: Dict[str, Any]) -> str:
        """Generate prediction panel component template"""
        return '''
import sys
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
                             QProgressBar, QTextEdit, QGroupBox, QCheckBox,
                             QSlider, QTabWidget, QFrame, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt6.QtGui import QFont, QColor, QPalette
from datetime import datetime, timedelta
import json

class PredictionWorker(QThread):
    """Background worker for prediction generation"""
    
    progress_update = pyqtSignal(int, str)
    prediction_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def run(self):
        """Run prediction generation"""
        try:
            steps = [
                "Loading historical data...",
                "Preprocessing data...",
                "Initializing model...",
                "Training model...",
                "Generating predictions...",
                "Calculating confidence intervals...",
                "Finalizing results..."
            ]
            
            for i, step in enumerate(steps):
                self.progress_update.emit(int((i + 1) / len(steps) * 100), step)
                self.msleep(500)  # Simulate processing time
            
            # Mock prediction results
            result = {
                'symbol': self.config.get('symbol', 'UNKNOWN'),
                'model_type': self.config.get('model_type', 'LSTM'),
                'predictions': self.generate_mock_predictions(),
                'metrics': {
                    'accuracy': 0.85,
                    'mae': 2.34,
                    'rmse': 3.12,
                    'confidence': 0.78
                },
                'timestamp': datetime.now().isoformat()
            }
            
            self.prediction_ready.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def generate_mock_predictions(self):
        """Generate mock prediction data"""
        days = self.config.get('prediction_days', 30)
        base_price = 150.0
        
        predictions = []
        for i in range(days):
            date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
            price = base_price + i * 0.5 + (i % 5) * 2
            confidence_low = price * 0.95
            confidence_high = price * 1.05
            
            predictions.append({
                'date': date,
                'predicted_price': round(price, 2),
                'confidence_low': round(confidence_low, 2),
                'confidence_high': round(confidence_high, 2)
            })
        
        return predictions

class ModelConfigWidget(QWidget):
    """Widget for model configuration"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize model configuration UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Model selection
        model_group = QGroupBox("Model Configuration")
        model_layout = QGridLayout()
        model_group.setLayout(model_layout)
        
        # Model type
        model_layout.addWidget(QLabel("Model Type:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["LSTM", "ARIMA", "Prophet", "XGBoost", "Ensemble"])
        model_layout.addWidget(self.model_combo, 0, 1)
        
        # Prediction horizon
        model_layout.addWidget(QLabel("Prediction Days:"), 1, 0)
        self.days_spin = QSpinBox()
        self.days_spin.setRange(1, 365)
        self.days_spin.setValue(30)
        model_layout.addWidget(self.days_spin, 1, 1)
        
        # Model parameters based on selection
        model_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.001, 1.0)
        self.learning_rate_spin.setValue(0.01)
        self.learning_rate_spin.setDecimals(4)
        model_layout.addWidget(self.learning_rate_spin, 2, 1)
        
        model_layout.addWidget(QLabel("Epochs:"), 3, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1000)
        self.epochs_spin.setValue(100)
        model_layout.addWidget(self.epochs_spin, 3, 1)
        
        layout.addWidget(model_group)
        
        # Advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QGridLayout()
        advanced_group.setLayout(advanced_layout)
        
        self.use_volume_check = QCheckBox("Include Volume Data")
        self.use_volume_check.setChecked(True)
        advanced_layout.addWidget(self.use_volume_check, 0, 0, 1, 2)
        
        self.use_technical_indicators_check = QCheckBox("Use Technical Indicators")
        self.use_technical_indicators_check.setChecked(True)
        advanced_layout.addWidget(self.use_technical_indicators_check, 1, 0, 1, 2)
        
        advanced_layout.addWidget(QLabel("Confidence Level:"), 2, 0)
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(50, 99)
        self.confidence_slider.setValue(95)
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        advanced_layout.addWidget(self.confidence_slider, 2, 1)
        
        self.confidence_label = QLabel("95%")
        advanced_layout.addWidget(self.confidence_label, 2, 2)
        
        layout.addWidget(advanced_group)
        layout.addStretch()
    
    def update_confidence_label(self, value):
        """Update confidence level label"""
        self.confidence_label.setText(f"{value}%")
    
    def get_config(self):
        """Get current configuration"""
        return {
            'model_type': self.model_combo.currentText(),
            'prediction_days': self.days_spin.value(),
            'learning_rate': self.learning_rate_spin.value(),
            'epochs': self.epochs_spin.value(),
            'use_volume': self.use_volume_check.isChecked(),
            'use_technical_indicators': self.use_technical_indicators_check.isChecked(),
            'confidence_level': self.confidence_slider.value() / 100.0
        }

class PredictionResultsWidget(QWidget):
    """Widget for displaying prediction results"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_results = None
    
    def init_ui(self):
        """Initialize results display UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Results summary
        summary_group = QGroupBox("Prediction Summary")
        summary_layout = QGridLayout()
        summary_group.setLayout(summary_layout)
        
        self.symbol_label = QLabel("Symbol: -")
        summary_layout.addWidget(self.symbol_label, 0, 0)
        
        self.model_label = QLabel("Model: -")
        summary_layout.addWidget(self.model_label, 0, 1)
        
        self.accuracy_label = QLabel("Accuracy: -")
        summary_layout.addWidget(self.accuracy_label, 1, 0)
        
        self.confidence_label = QLabel("Confidence: -")
        summary_layout.addWidget(self.confidence_label, 1, 1)
        
        layout.addWidget(summary_group)
        
        # Metrics display
        metrics_group = QGroupBox("Model Metrics")
        metrics_layout = QGridLayout()
        metrics_group.setLayout(metrics_layout)
        
        metrics_layout.addWidget(QLabel("MAE:"), 0, 0)
        self.mae_label = QLabel("-")
        metrics_layout.addWidget(self.mae_label, 0, 1)
        
        metrics_layout.addWidget(QLabel("RMSE:"), 0, 2)
        self.rmse_label = QLabel("-")
        metrics_layout.addWidget(self.rmse_label, 0, 3)
        
        layout.addWidget(metrics_group)
        
        # Prediction details
        details_group = QGroupBox("Prediction Details")
        details_layout = QVBoxLayout()
        details_group.setLayout(details_layout)
        
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(200)
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)
        
        layout.addWidget(details_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.save_results_btn = QPushButton("Save Results")
        self.save_results_btn.clicked.connect(self.save_results)
        self.save_results_btn.setEnabled(False)
        button_layout.addWidget(self.save_results_btn)
        
        self.export_json_btn = QPushButton("Export JSON")
        self.export_json_btn.clicked.connect(self.export_json)
        self.export_json_btn.setEnabled(False)
        button_layout.addWidget(self.export_json_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def update_results(self, results):
        """Update display with new results"""
        self.current_results = results
        
        # Update summary
        self.symbol_label.setText(f"Symbol: {results.get('symbol', 'Unknown')}")
        self.model_label.setText(f"Model: {results.get('model_type', 'Unknown')}")
        
        metrics = results.get('metrics', {})
        self.accuracy_label.setText(f"Accuracy: {metrics.get('accuracy', 0):.2%}")
        self.confidence_label.setText(f"Confidence: {metrics.get('confidence', 0):.2%}")
        
        # Update metrics
        self.mae_label.setText(f"{metrics.get('mae', 0):.2f}")
        self.rmse_label.setText(f"{metrics.get('rmse', 0):.2f}")
        
        # Update details
        self.update_details_text(results)
        
        # Enable buttons
        self.save_results_btn.setEnabled(True)
        self.export_json_btn.setEnabled(True)
    
    def update_details_text(self, results):
        """Update detailed results text"""
        details = []
        details.append(f"Prediction Generated: {results.get('timestamp', 'Unknown')}")
        details.append(f"Model: {results.get('model_type', 'Unknown')}")
        details.append("")
        
        predictions = results.get('predictions', [])
        if predictions:
            details.append(f"Predictions for next {len(predictions)} days:")
            details.append("-" * 50)
            
            for i, pred in enumerate(predictions[:10]):  # Show first 10
                date = pred.get('date', 'Unknown')
                price = pred.get('predicted_price', 0)
                conf_low = pred.get('confidence_low', 0)
                conf_high = pred.get('confidence_high', 0)
                
                details.append(f"{date}: ${price:.2f} (${conf_low:.2f} - ${conf_high:.2f})")
            
            if len(predictions) > 10:
                details.append(f"... and {len(predictions) - 10} more days")
        
        self.details_text.setPlainText("\\n".join(details))
    
    def save_results(self):
        """Save results to file"""
        if not self.current_results:
            return
        
        filename = f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(self.current_results, f, indent=2)
            
            QMessageBox.information(self, "Save Complete", f"Results saved to {filename}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Failed to save results: {str(e)}")
    
    def export_json(self):
        """Export results as JSON"""
        self.save_results()  # Same functionality for now

class PredictionPanelWidget(QWidget):
    """Main prediction panel widget"""
    
    prediction_generated = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker_thread = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the prediction panel interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title = QLabel("Stock Price Prediction Panel")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Stock symbol input
        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(QLabel("Stock Symbol:"))
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter stock symbol (e.g., AAPL)")
        self.symbol_input.setText("AAPL")
        symbol_layout.addWidget(self.symbol_input)
        layout.addLayout(symbol_layout)
        
        # Tab widget for configuration and results
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Configuration tab
        self.config_widget = ModelConfigWidget()
        self.tab_widget.addTab(self.config_widget, "Model Configuration")
        
        # Results tab
        self.results_widget = PredictionResultsWidget()
        self.tab_widget.addTab(self.results_widget, "Results")
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Generate Predictions")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.clicked.connect(self.generate_predictions)
        button_layout.addWidget(self.generate_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_prediction)
        button_layout.addWidget(self.stop_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Progress section
        progress_layout = QVBoxLayout()
        
        self.progress_label = QLabel("Ready to generate predictions")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addLayout(progress_layout)
    
    def generate_predictions(self):
        """Start prediction generation"""
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Input Error", "Please enter a stock symbol")
            return
        
        # Get configuration
        config = self.config_widget.get_config()
        config['symbol'] = symbol
        
        # Update UI state
        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start worker thread
        self.worker_thread = PredictionWorker(config)
        self.worker_thread.progress_update.connect(self.update_progress)
        self.worker_thread.prediction_ready.connect(self.on_prediction_ready)
        self.worker_thread.error_occurred.connect(self.on_error)
        self.worker_thread.finished.connect(self.on_worker_finished)
        self.worker_thread.start()
    
    def stop_prediction(self):
        """Stop prediction generation"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.terminate()
            self.worker_thread.wait()
        self.reset_ui_state()
    
    def update_progress(self, value, message):
        """Update progress display"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
    
    def on_prediction_ready(self, results):
        """Handle prediction results"""
        self.results_widget.update_results(results)
        self.tab_widget.setCurrentIndex(1)  # Switch to results tab
        self.progress_label.setText("Predictions generated successfully")
        self.prediction_generated.emit(results)
    
    def on_error(self, error_message):
        """Handle prediction errors"""
        QMessageBox.critical(self, "Prediction Error", f"Failed to generate predictions:\\n{error_message}")
        self.progress_label.setText(f"Error: {error_message}")
    
    def on_worker_finished(self):
        """Handle worker thread completion"""
        self.reset_ui_state()
    
    def reset_ui_state(self):
        """Reset UI to initial state"""
        self.generate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    
    panel = PredictionPanelWidget()
    panel.show()
    
    sys.exit(app.exec())
'''
    
    async def system_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        self.update_status("health_check", "Performing system diagnostics")
        
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        # Check database connection
        try:
            with sqlite3.connect(self.config['database_path']) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM stocks')
                stock_count = cursor.fetchone()[0]
                
            health_report['components']['database'] = {
                'status': 'healthy',
                'stock_records': stock_count
            }
        except Exception as e:
            health_report['components']['database'] = {
                'status': 'error',
                'error': str(e)
            }
            health_report['overall_status'] = 'degraded'
        
        # Check model registry
        health_report['components']['models'] = {
            'status': 'healthy',
            'registered_models': len(self.model_registry)
        }
        
        # Check disk space
        import shutil
        total, used, free = shutil.disk_usage('.')
        health_report['components']['storage'] = {
            'status': 'healthy' if free > 1024**3 else 'warning',  # 1GB threshold
            'free_space_gb': round(free / (1024**3), 2),
            'total_space_gb': round(total / (1024**3), 2)
        }
        
        return health_report
    
    async def optimize_performance(self, target_area: str = None) -> Dict[str, Any]:
        """Optimize system performance"""
        self.update_status("optimizing", f"Optimizing {target_area or 'system'}")
        
        optimizations = []
        
        if not target_area or target_area == 'database':
            # Database optimization
            with sqlite3.connect(self.config['database_path']) as conn:
                conn.execute('VACUUM')
                conn.execute('ANALYZE')
            optimizations.append('Database vacuumed and analyzed')
        
        if not target_area or target_area == 'models':
            # Clean old model files
            model_path = Path(self.config['model_save_path'])
            old_files = [f for f in model_path.glob('*.json') 
                        if (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days > 7]
            
            for file in old_files[:5]:  # Keep some history
                file.unlink()
            
            optimizations.append(f'Cleaned {len(old_files)} old model files')
        
        return {
            'success': True,
            'optimizations_applied': optimizations,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_capabilities(self) -> List[str]:
        """Return list of Bana's capabilities"""
        return [
            'system_architecture_design',
            'database_management',
            'stock_data_fetching',
            'prediction_model_creation',
            'desktop_app_development',
            'performance_optimization',
            'system_health_monitoring',
            'code_generation',
            'technical_documentation'
        ]
    
    def get_architecture_overview(self) -> Dict[str, Any]:
        """Get comprehensive architecture overview"""
        return {
            'agent': self.name,
            'role': self.role,
            'system_architecture': {
                'database': self.config['database_path'],
                'supported_models': self.config['supported_models'],
                'model_registry': len(self.model_registry),
                'components': ['main_window', 'chart_widget', 'data_table', 'prediction_panel']
            },
            'capabilities': self.get_capabilities(),
            'status': self.get_status_report()
        } 