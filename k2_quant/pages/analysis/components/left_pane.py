"""
Left Pane Component - Simplified Control Panel

Equal distribution layout with models, strategies, and indicators.
Save as: k2_quant/pages/analysis/components/left_pane.py
"""

import re
from typing import Dict, List, Optional
from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QLabel, QScrollArea,
                             QWidget, QCheckBox, QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, pyqtSignal

from k2_quant.utilities.logger import k2_logger


class LeftPaneWidget(QFrame):
    """Simplified left control panel with equal distribution"""
    
    # Signals
    model_selected = pyqtSignal(str)  # table_name
    strategy_toggled = pyqtSignal(str, bool)  # strategy_name, enabled
    indicator_toggled = pyqtSignal(str, bool)  # indicator_name, enabled
    
    def __init__(self):
        super().__init__()
        self.setFixedWidth(280)
        self.setObjectName("leftPane")
        
        # Initialize tracking sets FIRST
        self.active_indicators = set()
        self.active_strategies = set()
        
        # Initialize widgets
        self.model_list = None
        self.model_scroll = None
        self.strategy_scroll = None
        self.strategy_container = None
        self.strategy_layout = None
        self.indicator_scroll = None
        self.indicator_container = None
        self.indicator_layout = None
        
        self.init_ui()
        self.setup_styling()
        
    def init_ui(self):
        """Initialize the UI with equal 1/3 distribution"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        self.setLayout(main_layout)
        
        # Calculate equal heights for each section
        section_height = 240
        
        # === SAVED MODELS SECTION (1/3) ===
        models_label = QLabel("SAVED MODELS")
        models_label.setObjectName("sectionTitle")
        main_layout.addWidget(models_label)
        
        # Models list with scroll
        self.model_scroll = QScrollArea()
        self.model_scroll.setWidgetResizable(True)
        self.model_scroll.setFixedHeight(section_height)
        self.model_scroll.setObjectName("scrollArea")
        
        self.model_list = QListWidget()
        self.model_list.setObjectName("modelList")
        self.model_list.itemClicked.connect(self.on_model_clicked)
        self.model_scroll.setWidget(self.model_list)
        
        main_layout.addWidget(self.model_scroll)
        
        # === STRATEGIES SECTION (1/3) ===
        strategies_label = QLabel("STRATEGIES")
        strategies_label.setObjectName("sectionTitle")
        main_layout.addWidget(strategies_label)
        
        # Strategies container with scroll
        self.strategy_scroll = QScrollArea()
        self.strategy_scroll.setWidgetResizable(True)
        self.strategy_scroll.setFixedHeight(section_height)
        self.strategy_scroll.setObjectName("scrollArea")
        
        # Container for strategy checkboxes
        self.strategy_container = QWidget()
        self.strategy_layout = QVBoxLayout()
        self.strategy_layout.setContentsMargins(5, 5, 5, 5)
        self.strategy_layout.setSpacing(5)
        self.strategy_container.setLayout(self.strategy_layout)
        self.strategy_scroll.setWidget(self.strategy_container)
        
        main_layout.addWidget(self.strategy_scroll)
        
        # === TECHNICAL INDICATORS SECTION (1/3) ===
        indicators_label = QLabel("TECHNICAL INDICATORS")
        indicators_label.setObjectName("sectionTitle")
        main_layout.addWidget(indicators_label)
        
        # Indicators container with scroll
        self.indicator_scroll = QScrollArea()
        self.indicator_scroll.setWidgetResizable(True)
        self.indicator_scroll.setFixedHeight(section_height)
        self.indicator_scroll.setObjectName("scrollArea")
        
        # Container for indicator checkboxes
        self.indicator_container = QWidget()
        self.indicator_layout = QVBoxLayout()
        self.indicator_layout.setContentsMargins(5, 5, 5, 5)
        self.indicator_layout.setSpacing(5)
        self.indicator_container.setLayout(self.indicator_layout)
        self.indicator_scroll.setWidget(self.indicator_container)
        
        main_layout.addWidget(self.indicator_scroll)
        
        # Add stretch at bottom to push everything up
        main_layout.addStretch()
        
        # Populate indicators in alphabetical order
        self.populate_indicators()
    
    def on_model_clicked(self, item: QListWidgetItem):
        """Handle model selection"""
        if item.flags() == Qt.ItemFlag.NoItemFlags:  # "No saved models" item
            return
            
        table_name = item.data(Qt.ItemDataRole.UserRole)
        if table_name:
            k2_logger.info(f"Model selected: {table_name}", "LEFT_PANE")
            self.model_selected.emit(table_name)
    
    def on_strategy_toggled(self, name: str, checked: bool):
        """Handle strategy toggle - immediate action"""
        if checked:
            self.active_strategies.add(name)
        else:
            self.active_strategies.discard(name)
        
        k2_logger.info(f"Strategy '{name}' toggled: {checked}", "LEFT_PANE")
        self.strategy_toggled.emit(name, checked)
    
    def on_indicator_toggled(self, name: str, checked: bool):
        """Handle indicator toggle - immediate action"""
        if checked:
            self.active_indicators.add(name)
        else:
            self.active_indicators.discard(name)
        
        k2_logger.info(f"Indicator '{name}' toggled: {checked}", "LEFT_PANE")
        self.indicator_toggled.emit(name, checked)
    
    def populate_models(self, models: List[Dict]):
        """Populate the saved models list"""
        self.model_list.clear()
        
        if not models:
            item = QListWidgetItem("No saved models")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.model_list.addItem(item)
            return
        
        for model in models:
            # Use display_name if available, otherwise create from metadata
            if 'display_name' in model:
                label = model['display_name']
            else:
                symbol = model.get('symbol', 'UNKNOWN')
                range_val = model.get('range', '')
                label = f"{symbol} - {range_val}"
            
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, model.get('table_name'))
            self.model_list.addItem(item)
        
        k2_logger.info(f"Populated {len(models)} models", "LEFT_PANE")
    
    def populate_strategies(self, strategies: List[Dict]):
        """Populate the strategies section"""
        # Clear existing
        while self.strategy_layout.count():
            child = self.strategy_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if not strategies:
            no_strategies = QLabel("No strategies available")
            no_strategies.setStyleSheet("color: #666; padding: 10px;")
            self.strategy_layout.addWidget(no_strategies)
        else:
            # Add strategy checkboxes
            for strategy in strategies:
                name = strategy.get('name', 'Unknown')
                checkbox = QCheckBox(name)
                checkbox.setObjectName("strategyCheckbox")
                checkbox.stateChanged.connect(
                    lambda state, n=name: self.on_strategy_toggled(n, state == Qt.CheckState.Checked.value)
                )
                self.strategy_layout.addWidget(checkbox)
        
        # Add spacing at end
        self.strategy_layout.addStretch()
        
        k2_logger.info(f"Populated {len(strategies)} strategies", "LEFT_PANE")
    
    def populate_indicators(self):
        """Populate technical indicators in alphabetical order"""
        # Clear existing
        while self.indicator_layout.count():
            child = self.indicator_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Define indicators (alphabetically sorted)
        indicators = [
            "Bollinger Bands",
            "EMA (20)",
            "EMA (50)",
            "MACD",
            "OBV",
            "RSI",
            "SMA (20)",
            "SMA (50)",
            "SMA (200)",
            "Stochastic",
            "VWAP",
            "Volume"
        ]
        
        # Add indicator checkboxes
        for indicator in indicators:
            checkbox = QCheckBox(indicator)
            checkbox.setObjectName("indicatorCheckbox")
            checkbox.stateChanged.connect(
                lambda state, ind=indicator: self.on_indicator_toggled(ind, state == Qt.CheckState.Checked.value)
            )
            self.indicator_layout.addWidget(checkbox)
        
        # Add spacing at end
        self.indicator_layout.addStretch()
        
        k2_logger.info(f"Populated {len(indicators)} indicators", "LEFT_PANE")
    
    def clear_all_indicators(self):
        """Uncheck all indicators"""
        for i in range(self.indicator_layout.count()):
            widget = self.indicator_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setChecked(False)
        self.active_indicators.clear()
    
    def clear_all_strategies(self):
        """Uncheck all strategies"""
        for i in range(self.strategy_layout.count()):
            widget = self.strategy_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setChecked(False)
        self.active_strategies.clear()
    
    def refresh_models(self):
        """Refresh the models list from database"""
        try:
            from k2_quant.utilities.data.saved_models_manager import saved_models_manager
            models = saved_models_manager.get_saved_models()
            self.populate_models(models)
        except Exception as e:
            k2_logger.error(f"Failed to refresh models: {str(e)}", "LEFT_PANE")
    
    def refresh_strategies(self):
        """Refresh the strategies list from database"""
        try:
            from k2_quant.utilities.services.strategy_service import strategy_service
            strategies = strategy_service.get_all_strategies()
            self.populate_strategies(strategies)
        except Exception:
            # If strategy service fails, show empty list
            self.populate_strategies([])
    
    def get_selected_model(self) -> Optional[str]:
        """Get currently selected model table name"""
        current_item = self.model_list.currentItem()
        if current_item:
            return current_item.data(Qt.ItemDataRole.UserRole)
        return None
    
    def setup_styling(self):
        """Apply consistent styling"""
        self.setStyleSheet("""
            QFrame#leftPane {
                background-color: #0f0f0f;
                border-right: 1px solid #1a1a1a;
            }
            
            QLabel#sectionTitle {
                color: #999;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
                padding: 8px;
                background-color: #1a1a1a;
                border-radius: 3px;
            }
            
            QScrollArea#scrollArea {
                background-color: #0a0a0a;
                border: 1px solid #1a1a1a;
                border-radius: 4px;
            }
            
            QScrollArea#scrollArea QScrollBar:vertical {
                background-color: #0a0a0a;
                width: 10px;
                border-radius: 5px;
            }
            
            QScrollArea#scrollArea QScrollBar::handle:vertical {
                background-color: #2a2a2a;
                border-radius: 5px;
                min-height: 20px;
            }
            
            QScrollArea#scrollArea QScrollBar::handle:vertical:hover {
                background-color: #3a3a3a;
            }
            
            QScrollArea#scrollArea QScrollBar::add-line:vertical,
            QScrollArea#scrollArea QScrollBar::sub-line:vertical {
                height: 0px;
            }
            
            QListWidget#modelList {
                background-color: transparent;
                border: none;
                outline: none;
            }
            
            QListWidget#modelList::item {
                color: #ccc;
                padding: 8px;
                border-radius: 4px;
            }
            
            QListWidget#modelList::item:hover {
                background-color: #1a1a1a;
            }
            
            QListWidget#modelList::item:selected {
                background-color: #2a2a2a;
                color: #fff;
            }
            
            QCheckBox#strategyCheckbox,
            QCheckBox#indicatorCheckbox {
                color: #ccc;
                padding: 6px;
                font-size: 13px;
            }
            
            QCheckBox#strategyCheckbox:hover,
            QCheckBox#indicatorCheckbox:hover {
                color: #fff;
            }
            
            QCheckBox#strategyCheckbox::indicator,
            QCheckBox#indicatorCheckbox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #3a3a3a;
                background-color: #0a0a0a;
            }
            
            QCheckBox#strategyCheckbox::indicator:checked,
            QCheckBox#indicatorCheckbox::indicator:checked {
                background-color: #4a4;
                border-color: #4a4;
            }
            
            QCheckBox#strategyCheckbox::indicator:hover,
            QCheckBox#indicatorCheckbox::indicator:hover {
                border-color: #5a5a5a;
            }
        """)