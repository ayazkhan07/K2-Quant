"""
Data Table Widget for K2 Quant Analysis

Displays data with export capabilities and efficient handling of large datasets.
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
    
    def __init__(self, data: pd.DataFrame, filepath: str, format: str):
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
                # Export in chunks for large datasets
                mode = 'w'
                header = True
                
                for i in range(0, total_rows, chunk_size):
                    chunk = self.data.iloc[i:i+chunk_size]
                    chunk.to_csv(self.filepath, mode=mode, header=header, index=False)
                    mode = 'a'  # Append mode after first chunk
                    header = False
                    
                    progress = int((i + chunk_size) / total_rows * 100)
                    self.progress.emit(min(progress, 100))
                    
            elif self.format == 'excel':
                # Excel export (single shot due to library limitations)
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
        self.display_limit = 1000  # Display limit for performance
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
        
        # Connect signals
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        
        layout.addWidget(self.table)
        
        # Info bar
        self.info_bar = QLabel("No data loaded")
        self.info_bar.setFixedHeight(25)
        self.info_bar.setStyleSheet("""
            QLabel {
                background-color: #0f0f0f;
                color: #666;
                padding: 5px;
                font-size: 11px;
                border-top: 1px solid #1a1a1a;
            }
        """)
        layout.addWidget(self.info_bar)
    
    def create_control_bar(self) -> QWidget:
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
    
    def style_button(self, button: QPushButton):
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
    
    def load_data(self, data: Any):
        """Load data into table"""
        k2_logger.info("Loading data into table", "DATA_TABLE")
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            # Assuming list of tuples from database
            if len(data) > 0:
                num_cols = len(data[0])
                columns = ['date_time', 'open', 'high', 'low', 'close', 'volume', 'vwap'][:num_cols]
                self.data = pd.DataFrame(data, columns=columns)
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            k2_logger.error("Unsupported data format", "DATA_TABLE")
            return
        
        # Update display
        self.update_table_display()
        
        # Enable export buttons
        self.export_csv_btn.setEnabled(True)
        self.export_excel_btn.setEnabled(True)
        
        # Update info
        total_rows = len(self.data)
        displayed_rows = min(total_rows, self.display_limit)
        self.info_bar.setText(
            f"Total: {total_rows:,} rows | Displaying: {displayed_rows:,} rows"
        )
        
        if total_rows > self.display_limit:
            self.info_bar.setText(
                f"Total: {total_rows:,} rows | Displaying: first {displayed_rows:,} rows (for performance)"
            )
    
    def update_table_display(self):
        """Update table with data (limited for performance)"""
        if self.data is None:
            return
        
        # Limit display for performance
        display_data = self.data.head(self.display_limit)
        
        # Set up table
        self.table.setRowCount(len(display_data))
        self.table.setColumnCount(len(display_data.columns))
        self.table.setHorizontalHeaderLabels([str(col) for col in display_data.columns])
        
        # Populate table
        for row_idx in range(len(display_data)):
            for col_idx in range(len(display_data.columns)):
                value = display_data.iloc[row_idx, col_idx]
                
                # Format value
                if pd.isna(value):
                    text = ""
                elif isinstance(value, (int, np.integer)):
                    text = f"{value:,}"
                elif isinstance(value, (float, np.floating)):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row_idx, col_idx, item)
        
        # Adjust column widths
        self.table.resizeColumnsToContents()
        
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
    
    def export_data(self, format: str):
        """Export data to file"""
        if self.data is None:
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
        self.export_thread = DataExportThread(self.data, filepath, format)
        self.export_thread.progress.connect(progress.setValue)
        self.export_thread.finished.connect(lambda path: self.export_complete(path, progress))
        self.export_thread.error.connect(lambda err: self.export_error(err, progress))
        self.export_thread.start()
        
        k2_logger.info(f"Exporting data to {format}: {filepath}", "DATA_TABLE")
    
    def export_complete(self, filepath: str, progress: QProgressDialog):
        """Handle export completion"""
        progress.close()
        
        QMessageBox.information(
            self,
            "Export Complete",
            f"Data exported successfully to:\n{filepath}"
        )
        
        self.data_exported.emit(filepath)
        k2_logger.info(f"Export complete: {filepath}", "DATA_TABLE")
    
    def export_error(self, error: str, progress: QProgressDialog):
        """Handle export error"""
        progress.close()
        
        QMessageBox.critical(
            self,
            "Export Error",
            f"Failed to export data:\n{error}"
        )
        
        k2_logger.error(f"Export failed: {error}", "DATA_TABLE")
    
    def setup_styling(self):
        """Apply table styling"""
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #0a0a0a;
                gridline-color: #1a1a1a;
                color: #fff;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 3px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #2a2a2a;
            }
            QHeaderView::section {
                background-color: #1a1a1a;
                color: #999;
                padding: 5px;
                border: none;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 11px;
            }
            QTableCornerButton::section {
                background-color: #1a1a1a;
                border: none;
            }
        """)
    
    def get_selected_data(self) -> Optional[pd.DataFrame]:
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
        self.info_bar.setText("No data loaded")
        self.export_csv_btn.setEnabled(False)
        self.export_excel_btn.setEnabled(False)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.export_thread and self.export_thread.isRunning():
            self.export_thread.terminate()
            self.export_thread.wait()
        
        self.clear()