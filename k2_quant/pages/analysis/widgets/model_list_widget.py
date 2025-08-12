"""
Model List Widget for K2 Quant Analysis

Displays saved models from PostgreSQL with metadata and quick actions.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
                             QLabel, QHBoxLayout, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from k2_quant.utilities.logger import k2_logger


class ModelListWidget(QWidget):
    """Widget for displaying and selecting saved models"""
    
    # Signals
    model_selected = pyqtSignal(str)  # table_name
    model_deleted = pyqtSignal(str)  # table_name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.models = []
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.setLayout(layout)
        
        # List widget
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #1a1a1a;
                border: 1px solid #2a2a2a;
                border-radius: 3px;
                color: #fff;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #2a2a2a;
            }
            QListWidget::item:selected {
                background-color: #2a2a2a;
                color: #4a4;
            }
            QListWidget::item:hover {
                background-color: #252525;
            }
        """)
        
        # Connect signals
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self.on_item_clicked)
        
        layout.addWidget(self.list_widget)
        
        # Info label
        self.info_label = QLabel("No models loaded")
        self.info_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 11px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.info_label)
    
    def populate_models(self, models: List[Dict[str, Any]]):
        """Populate the list with models"""
        self.models = models
        self.list_widget.clear()
        
        if not models:
            self.info_label.setText("No saved models found")
            return
        
        for model in models:
            # Create list item with formatted text
            item_text = self.format_model_text(model)
            item = QListWidgetItem(item_text)
            
            # Store table name in item data
            item.setData(Qt.ItemDataRole.UserRole, model['table_name'])
            
            # Add to list
            self.list_widget.addItem(item)
        
        self.info_label.setText(f"{len(models)} models available")
        k2_logger.info(f"Populated {len(models)} models", "MODEL_LIST")
    
    def format_model_text(self, model: Dict[str, Any]) -> str:
        """Format model information for display"""
        symbol = model.get('symbol', 'UNKNOWN')
        timespan = model.get('timespan', '')
        range_val = model.get('range', '')
        records = model.get('record_count', 0)
        size = model.get('size', '')
        
        # Format record count
        if records > 1000000:
            record_str = f"{records/1000000:.1f}M"
        elif records > 1000:
            record_str = f"{records/1000:.1f}K"
        else:
            record_str = str(records)
        
        # Build display text
        text = f"{symbol} - {range_val.upper()}"
        if timespan:
            text += f" ({timespan})"
        text += f"\n{record_str} records | {size}"
        
        # Add date range if available
        date_range = model.get('date_range')
        if date_range and date_range[0]:
            start_date = str(date_range[0])[:10]
            end_date = str(date_range[1])[:10] if date_range[1] else ''
            text += f"\n{start_date} to {end_date}"
        
        return text
    
    def on_item_clicked(self, item: QListWidgetItem):
        """Handle item selection (single or double click)."""
        table_name = item.data(Qt.ItemDataRole.UserRole)
        if table_name:
            self.model_selected.emit(table_name)
            k2_logger.info(f"Model selected: {table_name}", "MODEL_LIST")
    
    def get_selected_model(self) -> Optional[str]:
        """Get currently selected model table name"""
        current = self.list_widget.currentItem()
        if current:
            return current.data(Qt.ItemDataRole.UserRole)
        return None
    
    def refresh(self):
        """Refresh the model list"""
        # Prefer reloading from the saved models service
        try:
            from k2_quant.utilities.data.saved_models_manager import saved_models_manager
            models = saved_models_manager.get_saved_models()
            self.populate_models(models)
        except Exception as e:
            k2_logger.error(f"Failed to refresh models: {e}", "MODEL_LIST")

    def reload_from_service(self):
        """Fetch models directly from the saved models manager and populate."""
        try:
            from k2_quant.utilities.data.saved_models_manager import saved_models_manager
            models = saved_models_manager.get_saved_models()
            self.populate_models(models)
        except Exception as e:
            k2_logger.error(f"Failed to reload saved models: {e}", "MODEL_LIST")
            self.populate_models([])
    
    def filter_models(self, filter_text: str):
        """Filter displayed models"""
        filter_lower = filter_text.lower()
        
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item_text = item.text().lower()
            
            # Show/hide based on filter
            if filter_lower in item_text:
                item.setHidden(False)
            else:
                item.setHidden(True)
    
    def clear(self):
        """Clear the model list"""
        self.list_widget.clear()
        self.models.clear()
        self.info_label.setText("No models loaded")