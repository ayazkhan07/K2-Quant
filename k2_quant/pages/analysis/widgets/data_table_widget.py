"""
Data Table Widget for K2 Quant Analysis

Displays data with export capabilities and efficient handling of large datasets.
Save as: k2_quant/pages/analysis/widgets/data_table_widget.py
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Any
from datetime import datetime

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QPushButton, QLabel, QHeaderView,
                             QFileDialog, QMessageBox, QProgressDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QThread

from k2_quant.utilities.logger import k2_logger


class DataExportThread(QThread):
    """Thread for exporting large datasets"""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, data, filepath, format):
        super().__init__()
        self.data = data
        self.filepath = filepath
        self.format = format
    
    def run(self):
        """Export data in background"""
        try:
            total_rows = len(self.data)
            chunk_size = 10000
            
            if self.format == 'csv':
                mode = 'w'
                header = True
                
                for i in range(0, total_rows, chunk_size):
                    chunk = self.data.iloc[i:i+chunk_size]
                    chunk.to_csv(self.filepath, mode=mode, header=header, index=False)
                    mode = 'a'
                    header = False
                    
                    progress = int((i + chunk_size) / total_rows * 100)
                    self.progress.emit(min(progress, 100))
                    
            elif self.format == 'excel':
                self.data.to_excel(self.filepath, index=False, engine='openpyxl')
                self.progress.emit(100)
            
            self.finished.emit(self.filepath)
            
        except Exception as e:
            self.error.emit(str(e))


class DataTableWidget(QWidget):
    """Advanced data table widget with export capabilities"""
    
    # Signals
    row_selected = pyqtSignal(dict)
    data_exported = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.data = None
        self.original_data = None
        self.display_limit = 1000
        self.export_thread = None
        
        self.init_ui()
        self.setup_styling()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        
        # Control bar
        control_bar = self.create_control_bar()
        layout.addWidget(control_bar)
        
        # Table
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ContiguousSelection)
        self.table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        
        # Connect signals
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        
        layout.addWidget(self.table)
        
        # Info bar
        self.info_bar = QLabel("No data loaded")
        self.info_bar.setFixedHeight(30)
        self.info_bar.setStyleSheet("""
            QLabel {
                background: linear-gradient(to right, #0f0f0f, #1a1a1a);
                color: #666;
                padding: 8px 12px;
                font-size: 12px;
                border-top: 1px solid #2a2a2a;
            }
        """)
        layout.addWidget(self.info_bar)
    
    def create_control_bar(self):
        """Create control bar with export buttons"""
        control_bar = QWidget()
        control_bar.setFixedHeight(35)
        control_bar.setStyleSheet("""
            QWidget {
                background-color: #0f0f0f;
                border-bottom: 1px solid #1a1a1a;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 0, 5, 0)
        control_bar.setLayout(layout)
        
        # Data info
        self.data_label = QLabel("Table View")
        self.data_label.setStyleSheet("color: #999; font-size: 12px;")
        layout.addWidget(self.data_label)
        
        layout.addStretch()
        
        # Export buttons
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.clicked.connect(lambda: self.export_data('csv'))
        self.export_csv_btn.setEnabled(False)
        self.style_button(self.export_csv_btn)
        layout.addWidget(self.export_csv_btn)
        
        self.export_excel_btn = QPushButton("Export Excel")
        self.export_excel_btn.clicked.connect(lambda: self.export_data('excel'))
        self.export_excel_btn.setEnabled(False)
        self.style_button(self.export_excel_btn)
        layout.addWidget(self.export_excel_btn)
        
        return control_bar
    
    def style_button(self, button):
        """Apply consistent button styling"""
        button.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1a;
                color: #999;
                border: 1px solid #2a2a2a;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover:enabled {
                background-color: #2a2a2a;
                color: #fff;
            }
            QPushButton:disabled {
                background-color: #0a0a0a;
                color: #444;
                border-color: #1a1a1a;
            }
        """)
    
    def load_data(self, data):
        """Load data into table"""
        k2_logger.info("Loading data into table", "DATA_TABLE")
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            if len(data) > 0:
                num_cols = len(data[0])
                
                if num_cols == 8:
                    columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
                elif num_cols == 7:
                    columns = ['date_time', 'open', 'high', 'low', 'close', 'volume', 'vwap']
                else:
                    columns = [f'Column_{i}' for i in range(num_cols)]
                
                self.original_data = pd.DataFrame(data, columns=columns)
        elif isinstance(data, pd.DataFrame):
            self.original_data = data.copy()
        else:
            k2_logger.error("Unsupported data format", "DATA_TABLE")
            return
        
        # Process data for display
        self.data = self.process_display_data(self.original_data)
        
        # Update display
        self.update_table_display()
        
        # Enable export buttons
        self.export_csv_btn.setEnabled(True)
        self.export_excel_btn.setEnabled(True)
        
        # Update info
        total_rows = len(self.data)
        displayed_rows = min(total_rows, self.display_limit)
        
        if total_rows > self.display_limit:
            self.info_bar.setText(
                f"Total: {total_rows:,} rows | Displaying: first {displayed_rows:,} rows (for performance)"
            )
        else:
            self.info_bar.setText(f"Total: {total_rows:,} rows | Displaying: {displayed_rows:,} rows")
    
    def process_display_data(self, data):
        """Process data for display"""
        display_data = data.copy()
        
        # Check if we already have Date and Time columns
        existing_cols_lower = [col.lower() for col in display_data.columns]
        if 'date' in existing_cols_lower and 'time' in existing_cols_lower:
            return display_data
        
        # Check if we have a datetime column that needs splitting
        datetime_columns = ['date_time', 'datetime', 'date_time_market', 'timestamp']
        datetime_col = None
        
        for col in display_data.columns:
            if col.lower() in datetime_columns:
                datetime_col = col
                break
        
        # Split datetime into Date and Time columns if found
        if datetime_col:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(display_data[datetime_col]):
                if datetime_col.lower() == 'timestamp' and display_data[datetime_col].dtype in [np.int64, np.float64]:
                    if display_data[datetime_col].iloc[0] > 1e10:
                        display_data[datetime_col] = pd.to_datetime(display_data[datetime_col], unit='ms')
                    else:
                        display_data[datetime_col] = pd.to_datetime(display_data[datetime_col], unit='s')
                else:
                    display_data[datetime_col] = pd.to_datetime(display_data[datetime_col])
            
            # Get the index position of the datetime column
            datetime_idx = display_data.columns.get_loc(datetime_col)
            
            # Create Date and Time columns
            date_series = display_data[datetime_col].dt.date
            time_series = display_data[datetime_col].dt.time
            
            # Drop the original datetime column
            display_data = display_data.drop(columns=[datetime_col])
            
            # Insert Date and Time at the original position
            display_data.insert(datetime_idx, 'Date', date_series)
            display_data.insert(datetime_idx + 1, 'Time', time_series)
        
        return display_data
    
    def update_table_display(self):
        """Update table with data (limited for performance)"""
        if self.data is None:
            return
        
        # Limit display for performance
        display_data = self.data.head(self.display_limit)
        
        # Set up table
        self.table.setRowCount(len(display_data))
        self.table.setColumnCount(len(display_data.columns))
        
        # Set column headers with proper formatting
        headers = []
        for col in display_data.columns:
            col_str = str(col)
            if col_str.lower() in ['open', 'high', 'low', 'close', 'volume', 'vwap', 'date', 'time']:
                headers.append(col_str.capitalize())
            else:
                headers.append(col_str)
        self.table.setHorizontalHeaderLabels(headers)
        
        # Populate table
        for row_idx in range(len(display_data)):
            for col_idx in range(len(display_data.columns)):
                value = display_data.iloc[row_idx, col_idx]
                col_name = display_data.columns[col_idx]
                col_name_lower = str(col_name).lower()
                
                # Format value based on column type
                if pd.isna(value):
                    text = ""
                elif col_name_lower == 'date':
                    text = str(value)
                elif col_name_lower == 'time':
                    text = str(value)[:8] if value else ""
                elif col_name_lower == 'volume':
                    try:
                        text = f"{int(float(value)):,}"
                    except (ValueError, TypeError):
                        text = str(value)
                elif col_name_lower in ['open', 'high', 'low', 'close', 'vwap']:
                    try:
                        text = f"{float(value):.2f}"
                    except (ValueError, TypeError):
                        text = str(value)
                elif isinstance(value, (int, np.integer)):
                    text = f"{value:,}"
                elif isinstance(value, (float, np.floating)):
                    if abs(value) < 10000 and abs(value) > 0.01:
                        text = f"{value:.2f}"
                    else:
                        text = f"{value:.4f}"
                else:
                    text = str(value)
                
                item = QTableWidgetItem(text)
                
                # Align text based on data type
                if col_name_lower in ['date', 'time']:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                
                self.table.setItem(row_idx, col_idx, item)
        
        # Adjust column widths
        self.table.resizeColumnsToContents()
        
        # Set minimum column widths
        for i in range(self.table.columnCount()):
            current_width = self.table.columnWidth(i)
            min_width = 80
            if i < len(headers):
                if headers[i].lower() == 'date':
                    min_width = 100
                elif headers[i].lower() == 'time':
                    min_width = 90
                elif headers[i].lower() == 'volume':
                    min_width = 110
            
            if current_width < min_width:
                self.table.setColumnWidth(i, min_width)
        
        k2_logger.info(f"Table updated with {len(display_data)} rows", "DATA_TABLE")
    
    def on_selection_changed(self):
        """Handle row selection"""
        selected_items = self.table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            
            # Get row data
            row_data = {}
            for col in range(self.table.columnCount()):
                header = self.table.horizontalHeaderItem(col)
                item = self.table.item(row, col)
                if header and item:
                    row_data[header.text()] = item.text()
            
            self.row_selected.emit(row_data)
    
    def export_data(self, format):
        """Export data to file"""
        if self.original_data is None:
            return
        
        # Get file path
        if format == 'csv':
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Export CSV", "", "CSV Files (*.csv)"
            )
        elif format == 'excel':
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Export Excel", "", "Excel Files (*.xlsx)"
            )
        else:
            return
        
        if not filepath:
            return
        
        # Show progress dialog
        progress = QProgressDialog("Exporting data...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        
        # Start export thread
        self.export_thread = DataExportThread(self.original_data, filepath, format)
        self.export_thread.progress.connect(progress.setValue)
        self.export_thread.finished.connect(lambda path: self.export_complete(path, progress))
        self.export_thread.error.connect(lambda err: self.export_error(err, progress))
        self.export_thread.start()
        
        k2_logger.info(f"Exporting data to {format}: {filepath}", "DATA_TABLE")
    
    def export_complete(self, filepath, progress):
        """Handle export completion"""
        progress.close()
        
        QMessageBox.information(
            self,
            "Export Complete",
            f"Data exported successfully to:\n{filepath}"
        )
        
        self.data_exported.emit(filepath)
        k2_logger.info(f"Export complete: {filepath}", "DATA_TABLE")
    
    def export_error(self, error, progress):
        """Handle export error"""
        progress.close()
        
        QMessageBox.critical(
            self,
            "Export Error",
            f"Failed to export data:\n{error}"
        )
        
        k2_logger.error(f"Export failed: {error}", "DATA_TABLE")
    
    def setup_styling(self):
        """Apply modern table styling"""
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #0f0f0f;
                alternate-background-color: #1a1a1a;
                gridline-color: transparent;
                color: #e0e0e0;
                font-family: 'Inter', 'Segoe UI', 'Arial', sans-serif;
                font-size: 13px;
                border: none;
                outline: none;
            }
            
            QTableWidget::item {
                padding: 8px 12px;
                border: none;
                border-bottom: 1px solid rgba(42, 42, 42, 0.3);
            }
            
            QTableWidget::item:selected {
                background-color: #2a3f5f;
                color: #ffffff;
            }
            
            QTableWidget::item:hover {
                background-color: #1e1e1e;
            }
            
            QHeaderView::section {
                background-color: #0a0a0a;
                color: #888;
                padding: 10px 12px;
                border: none;
                border-bottom: 2px solid #2a2a2a;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 11px;
                letter-spacing: 0.5px;
            }
            
            QHeaderView::section:hover {
                background-color: #1a1a1a;
                color: #aaa;
            }
            
            QTableCornerButton::section {
                background-color: #0a0a0a;
                border: none;
            }
            
            QScrollBar:vertical {
                background: #0a0a0a;
                width: 10px;
                border: none;
            }
            
            QScrollBar::handle:vertical {
                background: #2a2a2a;
                border-radius: 5px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background: #3a3a3a;
            }
            
            QScrollBar:horizontal {
                background: #0a0a0a;
                height: 10px;
                border: none;
            }
            
            QScrollBar::handle:horizontal {
                background: #2a2a2a;
                border-radius: 5px;
                min-width: 20px;
            }
            
            QScrollBar::handle:horizontal:hover {
                background: #3a3a3a;
            }
            
            QScrollBar::add-line, QScrollBar::sub-line {
                background: none;
                border: none;
            }
            
            QScrollBar::add-page, QScrollBar::sub-page {
                background: none;
            }
        """)
        
        # Set alternating row colors for better readability
        self.table.setAlternatingRowColors(True)
        
        # Remove focus rectangle
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    
    def get_selected_data(self):
        """Get currently selected rows as DataFrame"""
        selected_rows = set()
        for item in self.table.selectedItems():
            selected_rows.add(item.row())
        
        if selected_rows and self.data is not None:
            indices = sorted(list(selected_rows))
            return self.data.iloc[indices]
        
        return None
    
    def refresh(self):
        """Refresh table display"""
        if self.data is not None:
            self.update_table_display()
    
    def clear(self):
        """Clear table data"""
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.data = None
        self.original_data = None
        self.info_bar.setText("No data loaded")
        self.export_csv_btn.setEnabled(False)
        self.export_excel_btn.setEnabled(False)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.export_thread and self.export_thread.isRunning():
            self.export_thread.terminate()
            self.export_thread.wait()
        
        self.clear()