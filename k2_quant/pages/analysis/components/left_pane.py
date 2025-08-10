"""
Left Pane Component - Controls Panel

Contains saved models list, custom strategies, and technical indicators.
"""

from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QLabel, QScrollArea,
                             QWidget, QCheckBox, QPushButton, QListWidget,
                             QListWidgetItem, QHBoxLayout, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal

from k2_quant.pages.analysis.widgets.model_list_widget import ModelListWidget
from k2_quant.pages.analysis.widgets.strategy_widget import StrategyWidget
from k2_quant.pages.analysis.widgets.indicator_widget import IndicatorWidget
from k2_quant.utilities.services.model_loader_service import model_loader_service
from k2_quant.utilities.services.strategy_service import strategy_service
from k2_quant.utilities.logger import k2_logger


class LeftPaneWidget(QFrame):
    """Left control panel with models, strategies, and indicators"""
    
    # Signals
    model_selected = pyqtSignal(str)  # table_name
    strategy_toggled = pyqtSignal(str, bool)  # strategy_name, enabled
    indicator_toggled = pyqtSignal(str, bool)  # indicator_name, enabled
    new_strategy_requested = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setFixedWidth(280)
        self.setObjectName("leftPane")
        
        self.init_ui()
        self.setup_styling()
        self.load_saved_models()
        self.load_strategies()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        self.setLayout(layout)
        
        # Search bar
        self.search_bar = self.create_search_bar()
        layout.addWidget(self.search_bar)
        
        # Saved Models Section
        models_header = self.create_section_header("SAVED MODELS")
        layout.addWidget(models_header)
        
        self.model_list = ModelListWidget()
        self.model_list.model_selected.connect(self.model_selected.emit)
        self.model_list.setMaximumHeight(200)
        layout.addWidget(self.model_list)
        
        # Custom Strategies Section
        strategies_header = self.create_section_header("CUSTOM STRATEGIES")
        layout.addWidget(strategies_header)
        
        # New Strategy button
        new_strategy_btn = QPushButton("+ New Strategy")
        new_strategy_btn.setObjectName("newStrategyBtn")
        new_strategy_btn.clicked.connect(self.new_strategy_requested.emit)
        layout.addWidget(new_strategy_btn)
        
        # Scrollable strategies container
        self.strategies_scroll = QScrollArea()
        self.strategies_scroll.setWidgetResizable(True)
        self.strategies_scroll.setMaximumHeight(150)
        self.strategies_scroll.setObjectName("scrollArea")
        
        self.strategy_widget = StrategyWidget()
        self.strategy_widget.strategy_toggled.connect(self.strategy_toggled.emit)
        self.strategies_scroll.setWidget(self.strategy_widget)
        layout.addWidget(self.strategies_scroll)
        
        # Technical Indicators Section
        indicators_header = self.create_section_header("TECHNICAL INDICATORS")
        layout.addWidget(indicators_header)
        
        # Scrollable indicators container
        self.indicators_scroll = QScrollArea()
        self.indicators_scroll.setWidgetResizable(True)
        self.indicators_scroll.setObjectName("scrollArea")
        
        self.indicator_widget = IndicatorWidget()
        self.indicator_widget.indicator_toggled.connect(self.indicator_toggled.emit)
        self.indicators_scroll.setWidget(self.indicator_widget)
        layout.addWidget(self.indicators_scroll)
        
        layout.addStretch()
    
    def create_search_bar(self) -> QWidget:
        """Create search bar widget"""
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        container.setLayout(layout)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search models, strategies...")
        self.search_input.textChanged.connect(self.on_search_changed)
        self.search_input.setObjectName("searchInput")
        layout.addWidget(self.search_input)
        
        return container
    
    def create_section_header(self, text: str) -> QWidget:
        """Create a section header widget"""
        header = QWidget()
        header.setFixedHeight(30)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header.setLayout(header_layout)
        
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        header_layout.addWidget(label)
        
        return header
    
    def on_search_changed(self, text: str):
        """Handle search text change"""
        # Filter models
        self.model_list.filter_models(text)
        
        # Filter indicators
        self.indicator_widget.filter_indicators(text)
        
        k2_logger.info(f"Search filter: {text}", "LEFT_PANE")
    
    def load_saved_models(self):
        """Load saved models from database"""
        try:
            if model_loader_service:
                models = model_loader_service.get_saved_models()
                self.model_list.populate_models(models)
                k2_logger.info(f"Loaded {len(models)} saved models", "LEFT_PANE")
            else:
                k2_logger.warning("Model loader service not available", "LEFT_PANE")
        except Exception as e:
            k2_logger.error(f"Failed to load models: {str(e)}", "LEFT_PANE")
    
    def load_strategies(self):
        """Load custom strategies from database"""
        try:
            if strategy_service:
                strategies = strategy_service.get_all_strategies()
                self.strategy_widget.populate_strategies(strategies)
                k2_logger.info(f"Loaded {len(strategies)} strategies", "LEFT_PANE")
            else:
                k2_logger.warning("Strategy service not available", "LEFT_PANE")
        except Exception as e:
            k2_logger.error(f"Failed to load strategies: {str(e)}", "LEFT_PANE")
    
    def refresh_models(self):
        """Refresh the models list"""
        self.load_saved_models()
    
    def refresh_strategies(self):
        """Refresh the strategies list"""
        self.load_strategies()
    
    def setup_styling(self):
        """Apply styling to the pane"""
        self.setStyleSheet("""
            #leftPane {
                background-color: #0f0f0f;
                border-right: 1px solid #1a1a1a;
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
            
            #searchInput {
                background-color: #1a1a1a;
                border: 1px solid #2a2a2a;
                color: #fff;
                padding: 8px;
                font-size: 12px;
                border-radius: 3px;
            }
            
            #searchInput:focus {
                border-color: #3a3a3a;
                background-color: #222;
            }
            
            #newStrategyBtn {
                background-color: #1a1a1a;
                color: #4a4;
                border: 1px solid #2a2a2a;
                padding: 8px;
                font-size: 12px;
                border-radius: 3px;
            }
            
            #newStrategyBtn:hover {
                background-color: #2a2a2a;
                border-color: #3a3a3a;
            }
            
            #scrollArea {
                background-color: transparent;
                border: none;
            }
            
            QScrollBar:vertical {
                background-color: #1a1a1a;
                width: 8px;
                border-radius: 4px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #3a3a3a;
                border-radius: 4px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #4a4a4a;
            }
            
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
    
    def cleanup(self):
        """Cleanup resources"""
        pass