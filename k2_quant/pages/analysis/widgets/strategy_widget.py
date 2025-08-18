"""
Strategy Widget for K2 Quant Analysis

Displays custom strategies with toggle controls.
Save as: k2_quant/pages/analysis/widgets/strategy_widget.py
"""

from typing import List, Dict, Any

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QCheckBox,
                             QPushButton, QLabel, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal

from k2_quant.utilities.logger import k2_logger


class StrategyWidget(QWidget):
    """Widget for displaying and toggling custom strategies"""
    
    # Signals
    strategy_toggled = pyqtSignal(str, bool)
    strategy_edited = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.strategy_checkboxes = {}
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(5)
        self.setLayout(self.layout)
        
        # Add placeholder
        self.placeholder = QLabel("No strategies available")
        self.placeholder.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 12px;
                padding: 10px;
            }
        """)
        self.layout.addWidget(self.placeholder)
    
    def populate_strategies(self, strategies):
        """Populate widget with strategies"""
        # Clear existing
        self.clear()
        
        if not strategies:
            self.placeholder.show()
            return
        
        self.placeholder.hide()
        
        for strategy in strategies:
            strategy_item = self.create_strategy_item(strategy)
            self.layout.addWidget(strategy_item)
        
        # Add stretch at the end
        self.layout.addStretch()
        
        k2_logger.info(f"Populated {len(strategies)} strategies", "STRATEGY_WIDGET")
    
    def create_strategy_item(self, strategy):
        """Create a strategy item widget"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #2a2a2a;
                border-radius: 3px;
                padding: 5px;
                margin: 2px;
            }
            QFrame:hover {
                background-color: #1f1f1f;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        frame.setLayout(layout)
        
        # Strategy name and checkbox
        header_layout = QHBoxLayout()
        
        checkbox = QCheckBox(strategy['name'])
        checkbox.setStyleSheet("""
            QCheckBox {
                color: #fff;
                font-size: 12px;
                font-weight: bold;
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
        """)
        checkbox.stateChanged.connect(
            lambda state, name=strategy['name']: self.on_strategy_toggled(name, state)
        )
        header_layout.addWidget(checkbox)
        
        # Store reference
        self.strategy_checkboxes[strategy['name']] = checkbox
        
        # Edit button
        edit_btn = QPushButton("Edit")
        edit_btn.setFixedSize(40, 20)
        edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #999;
                border: 1px solid #3a3a3a;
                border-radius: 2px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                color: #fff;
            }
        """)
        edit_btn.clicked.connect(
            lambda _, name=strategy['name']: self.strategy_edited.emit(name)
        )
        header_layout.addWidget(edit_btn)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Description
        if strategy.get('description'):
            desc_label = QLabel(strategy['description'][:100])
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("""
                QLabel {
                    color: #888;
                    font-size: 11px;
                    padding-left: 20px;
                }
            """)
            layout.addWidget(desc_label)
        
        # Metrics
        if strategy.get('execution_count'):
            metrics_label = QLabel(f"Executed: {strategy['execution_count']} times")
            metrics_label.setStyleSheet("""
                QLabel {
                    color: #666;
                    font-size: 10px;
                    padding-left: 20px;
                }
            """)
            layout.addWidget(metrics_label)
        
        return frame
    
    def on_strategy_toggled(self, strategy_name, state):
        """Handle strategy toggle"""
        enabled = state == 2
        self.strategy_toggled.emit(strategy_name, enabled)
        k2_logger.info(
            f"Strategy {'enabled' if enabled else 'disabled'}: {strategy_name}",
            "STRATEGY_WIDGET"
        )
    
    def set_strategy_enabled(self, strategy_name, enabled):
        """Programmatically set strategy state"""
        if strategy_name in self.strategy_checkboxes:
            self.strategy_checkboxes[strategy_name].setChecked(enabled)
    
    def get_enabled_strategies(self):
        """Get list of enabled strategies"""
        enabled = []
        for name, checkbox in self.strategy_checkboxes.items():
            if checkbox.isChecked():
                enabled.append(name)
        return enabled
    
    def clear(self):
        """Clear all strategies"""
        # Remove all widgets except placeholder
        while self.layout.count() > 1:
            item = self.layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        
        self.strategy_checkboxes.clear()
        self.placeholder.show()