"""
K2 Quant Analysis Page

Three-pane interface for comprehensive stock market analysis.
"""

import sys
from typing import Dict, Optional, Any
from datetime import datetime
import pandas as pd

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QStatusBar, QLabel, QFrame, QPushButton,
                             QListWidget, QTextEdit, QLineEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QCheckBox, QComboBox,
                             QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

from k2_quant.utilities.logger import k2_logger


class AnalysisPageWidget(QWidget):
    """Main Analysis page widget with three-pane layout"""
    
    # Signals
    back_to_stock_fetcher = pyqtSignal()
    
    def __init__(self, tab_id: int = 0, parent=None):
        super().__init__(parent)
        self.tab_id = tab_id
        self.current_model = None
        self.current_data = None
        
        self.init_ui()
        self.setup_styling()
        self.load_saved_models()
        
        k2_logger.info(f"Analysis page initialized (Tab ID: {tab_id})", "ANALYSIS")
    
    def init_ui(self):
        """Initialize the three-pane UI layout"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)
        
        # Create header
        self.create_header(main_layout)
        
        # Create splitter for three panes
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #1a1a1a;
            }
        """)
        
        # Create three panes
        self.left_pane = self.create_left_pane()
        self.middle_pane = self.create_middle_pane()
        self.right_pane = self.create_right_pane()
        
        # Add to splitter
        self.splitter.addWidget(self.left_pane)
        self.splitter.addWidget(self.middle_pane)
        self.splitter.addWidget(self.right_pane)
        
        # Set sizes - left: 280px, middle: flexible, right: 380px
        self.splitter.setSizes([280, 740, 380])
        
        main_layout.addWidget(self.splitter)
        
        # Create status bar
        self.status_widget = self.create_status_bar()
        main_layout.addWidget(self.status_widget)
    
    def create_header(self, parent_layout):
        """Create header with title"""
        header = QWidget()
        header.setFixedHeight(40)
        header.setObjectName("analysisHeader")
        
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(20, 0, 20, 0)
        header.setLayout(header_layout)
        
        title = QLabel(f"K2 QUANT - ANALYSIS (Tab {self.tab_id})")
        title.setFont(QFont("Arial", 14))
        title.setStyleSheet("color: #999; letter-spacing: 1px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        parent_layout.addWidget(header)
    
    def create_left_pane(self) -> QFrame:
        """Create left control panel"""
        frame = QFrame()
        frame.setFixedWidth(280)
        frame.setObjectName("leftPane")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        frame.setLayout(layout)
        
        # Saved Models Section
        models_label = QLabel("SAVED MODELS")
        models_label.setObjectName("sectionTitle")
        layout.addWidget(models_label)
        
        self.model_list = QListWidget()
        self.model_list.setMaximumHeight(200)
        self.model_list.itemClicked.connect(self.load_model)
        layout.addWidget(self.model_list)
        
        # Technical Indicators Section
        indicators_label = QLabel("TECHNICAL INDICATORS")
        indicators_label.setObjectName("sectionTitle")
        layout.addWidget(indicators_label)
        
        # Create indicator checkboxes
        self.indicator_checks = {}
        indicators = ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands", "Volume"]
        
        for indicator in indicators:
            checkbox = QCheckBox(indicator)
            checkbox.stateChanged.connect(lambda state, ind=indicator: self.toggle_indicator(ind, state))
            self.indicator_checks[indicator] = checkbox
            layout.addWidget(checkbox)
        
        layout.addStretch()
        
        return frame
    
    def create_middle_pane(self) -> QFrame:
        """Create middle visualization pane"""
        frame = QFrame()
        frame.setObjectName("middlePane")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        frame.setLayout(layout)
        
        # Chart controls
        controls = QWidget()
        controls.setFixedHeight(40)
        controls.setObjectName("chartControls")
        
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(10, 0, 10, 0)
        controls.setLayout(controls_layout)
        
        controls_layout.addWidget(QLabel("VIEW:"))
        
        self.view_selector = QComboBox()
        self.view_selector.addItems(["Chart", "Data Table", "Both"])
        self.view_selector.currentTextChanged.connect(self.change_view)
        controls_layout.addWidget(self.view_selector)
        
        controls_layout.addStretch()
        
        # Zoom controls
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedSize(30, 30)
        controls_layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setFixedSize(30, 30)
        controls_layout.addWidget(zoom_out_btn)
        
        reset_btn = QPushButton("Reset")
        controls_layout.addWidget(reset_btn)
        
        layout.addWidget(controls)
        
        # Data display area (placeholder for chart/table)
        self.data_display = QFrame()
        self.data_display.setObjectName("dataDisplay")
        self.data_display.setStyleSheet("""
            #dataDisplay {
                background-color: #0a0a0a;
                border: 1px solid #1a1a1a;
            }
        """)
        
        display_layout = QVBoxLayout()
        self.data_display.setLayout(display_layout)
        
        # Data table
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.horizontalHeader().setStretchLastSection(True)
        display_layout.addWidget(self.data_table)
        
        layout.addWidget(self.data_display)
        
        return frame
    
    def create_right_pane(self) -> QFrame:
        """Create right AI chat pane"""
        frame = QFrame()
        frame.setFixedWidth(380)
        frame.setObjectName("rightPane")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        frame.setLayout(layout)
        
        # Header
        ai_label = QLabel("CONVERSATIONAL AI")
        ai_label.setObjectName("sectionTitle")
        layout.addWidget(ai_label)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #0a0a0a;
                color: #ccc;
                border: 1px solid #1a1a1a;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.chat_display)
        
        # Add welcome message
        self.chat_display.append("AI: Hello! I can help you analyze stock data and create custom strategies. Load a model to get started.")
        
        # Input area
        input_widget = QWidget()
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_widget.setLayout(input_layout)
        
        self.ai_input = QLineEdit()
        self.ai_input.setPlaceholderText("Type your message...")
        self.ai_input.returnPressed.connect(self.send_ai_message)
        input_layout.addWidget(self.ai_input)
        
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.send_ai_message)
        input_layout.addWidget(send_btn)
        
        layout.addWidget(input_widget)
        
        return frame
    
    def create_status_bar(self) -> QWidget:
        """Create status bar"""
        widget = QWidget()
        widget.setFixedHeight(32)
        widget.setObjectName("analysisStatusBar")
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        widget.setLayout(layout)
        
        # Status indicator
        indicator = QLabel("●")
        indicator.setStyleSheet("color: #4a4; font-size: 8px;")
        layout.addWidget(indicator)
        
        # Model info
        self.model_label = QLabel("No model loaded")
        self.model_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.model_label)
        
        layout.addStretch()
        
        # System status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.status_label)
        
        return widget
    
    def load_saved_models(self):
        """Load list of saved models"""
        try:
            # Get stock tables from database
            from k2_quant.utilities.services.stock_data_service import stock_service
            tables = stock_service.get_all_stock_tables()
            
            for table_name, size in tables:
                self.model_list.addItem(f"{table_name} ({size})")
                
            k2_logger.info(f"Loaded {len(tables)} saved models", "ANALYSIS")
        except Exception as e:
            k2_logger.error(f"Failed to load models: {str(e)}", "ANALYSIS")
    
    def load_model(self, item):
        """Load selected model"""
        if not item:
            return
            
        # Extract table name from item text
        table_name = item.text().split(" (")[0]
        k2_logger.info(f"Loading model: {table_name}", "ANALYSIS")
        
        try:
            from k2_quant.utilities.services.stock_data_service import stock_service
            
            # Get data from table
            rows, total_count = stock_service.get_display_data(table_name, limit=1000)
            
            if rows:
                # Update status
                self.model_label.setText(f"Model: {table_name} ({total_count:,} records)")
                self.status_label.setText("Model loaded")
                
                # Load into table
                self.load_data_into_table(rows)
                
                # Update AI context
                self.chat_display.append(f"\nAI: Loaded {table_name} with {total_count:,} records. How can I help you analyze this data?")
                
                # Store current data
                self.current_model = table_name
                self.current_data = rows
                
        except Exception as e:
            k2_logger.error(f"Failed to load model: {str(e)}", "ANALYSIS")
            self.status_label.setText("Error loading model")
    
    def load_data_into_table(self, rows):
        """Load data into the table widget"""
        if not rows:
            return
            
        # Set up table
        self.data_table.setRowCount(len(rows))
        self.data_table.setColumnCount(7)
        self.data_table.setHorizontalHeaderLabels(['Date/Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP'])
        
        # Populate table
        for i, row in enumerate(rows):
            # Date/Time
            self.data_table.setItem(i, 0, QTableWidgetItem(str(row[0])))
            # OHLCV + VWAP
            for j in range(1, 7):
                if j == 5:  # Volume
                    self.data_table.setItem(i, j, QTableWidgetItem(f"{int(row[j]):,}"))
                else:
                    self.data_table.setItem(i, j, QTableWidgetItem(f"{float(row[j]):.2f}"))
    
    def toggle_indicator(self, indicator_name: str, state: int):
        """Toggle technical indicator"""
        enabled = state == 2  # Qt.CheckState.Checked
        k2_logger.info(f"Indicator {'enabled' if enabled else 'disabled'}: {indicator_name}", "ANALYSIS")
        
        # In a full implementation, this would calculate and display the indicator
        self.status_label.setText(f"{indicator_name} {'enabled' if enabled else 'disabled'}")
    
    def change_view(self, view_type: str):
        """Change the middle pane view"""
        k2_logger.info(f"View changed to: {view_type}", "ANALYSIS")
        # In a full implementation, this would switch between chart/table views
    
    def send_ai_message(self):
        """Send message to AI"""
        message = self.ai_input.text().strip()
        if not message:
            return
            
        # Add user message to chat
        self.chat_display.append(f"\nYOU: {message}")
        
        # Clear input
        self.ai_input.clear()
        
        # Simulate AI response (in full implementation, would call AI service)
        self.chat_display.append(f"\nAI: I understand you want to analyze the {self.current_model if self.current_model else 'data'}. Let me help you with that...")
        
        # Handle special commands
        if "create strategy" in message.lower():
            self.chat_display.append("\nAI: I'll help you create a custom trading strategy. What conditions would you like to use?")
        elif "projection" in message.lower():
            self.chat_display.append("\nAI: I can create price projections based on historical patterns. What time frame are you interested in?")
    
    def setup_styling(self):
        """Apply consistent styling"""
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0a0a;
                color: #ffffff;
            }
            
            #analysisHeader {
                background-color: #0f0f0f;
                border-bottom: 1px solid #1a1a1a;
            }
            
            #analysisStatusBar {
                background-color: #0f0f0f;
                border-top: 1px solid #1a1a1a;
            }
            
            #leftPane, #rightPane {
                background-color: #0f0f0f;
                border: 1px solid #1a1a1a;
            }
            
            #middlePane {
                background-color: #0a0a0a;
            }
            
            #chartControls {
                background-color: #0f0f0f;
                border-bottom: 1px solid #1a1a1a;
            }
            
            #sectionTitle {
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #999;
                font-weight: 600;
                background-color: #1a1a1a;
                padding: 5px 10px;
                border-radius: 3px;
            }
            
            QListWidget {
                background-color: #1a1a1a;
                border: 1px solid #2a2a2a;
                border-radius: 3px;
                color: #fff;
            }
            
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #2a2a2a;
            }
            
            QListWidget::item:selected {
                background-color: #2a2a2a;
            }
            
            QListWidget::item:hover {
                background-color: #252525;
            }
            
            QCheckBox {
                color: #ccc;
                spacing: 8px;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                background-color: #1a1a1a;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
            }
            
            QCheckBox::indicator:checked {
                background-color: #4a4;
                border-color: #4a4;
            }
            
            QPushButton {
                background-color: #1a1a1a;
                color: #fff;
                border: 1px solid #2a2a2a;
                padding: 5px 15px;
                border-radius: 3px;
            }
            
            QPushButton:hover {
                background-color: #2a2a2a;
            }
            
            QComboBox {
                background-color: #1a1a1a;
                color: #fff;
                border: 1px solid #2a2a2a;
                padding: 5px;
                border-radius: 3px;
            }
            
            QLineEdit {
                background-color: #1a1a1a;
                color: #fff;
                border: 1px solid #2a2a2a;
                padding: 8px;
                border-radius: 3px;
            }
            
            QTableWidget {
                background-color: #0a0a0a;
                gridline-color: #1a1a1a;
                color: #fff;
            }
            
            QTableWidget::item {
                padding: 5px;
            }
            
            QHeaderView::section {
                background-color: #1a1a1a;
                color: #999;
                padding: 8px;
                border: none;
                font-weight: 600;
            }
        """)
    
    def cleanup(self):
        """Cleanup when closing tab"""
        k2_logger.info(f"Cleaning up Analysis tab {self.tab_id}", "ANALYSIS")