"""
Browser-style Tab Bar Component for K2 Quant

Manages navigation between Stock Fetcher and Analysis pages with support for
multiple Analysis tab instances.
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QTabBar, 
                             QVBoxLayout, QLabel)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont

from k2_quant.utilities.logger import k2_logger


class TabBarWidget(QWidget):
    """Browser-style tab bar for page navigation"""
    
    # Signals
    tab_changed = pyqtSignal(str, int)  # page_type, tab_id
    new_tab_requested = pyqtSignal(str)  # page_type
    tab_closed = pyqtSignal(str, int)  # page_type, tab_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabs = {}  # {tab_index: (page_type, tab_id, title)}
        self.current_analysis_id = 0
        self.permanent_tabs = ['stock_fetcher', 'analysis_0']  # Can't be closed
        
        self.init_ui()
        self.setup_permanent_tabs()
        
    def init_ui(self):
        """Initialize the tab bar UI"""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        
        # Tab bar
        self.tab_bar = QTabBar()
        self.tab_bar.setTabsClosable(True)
        self.tab_bar.setMovable(True)
        self.tab_bar.setExpanding(False)
        self.tab_bar.setUsesScrollButtons(True)
        
        # Style the tab bar
        self.tab_bar.setStyleSheet("""
            QTabBar {
                background-color: #0f0f0f;
                border: none;
                border-bottom: 1px solid #1a1a1a;
            }
            QTabBar::tab {
                background-color: #1a1a1a;
                color: #999;
                padding: 8px 12px;
                margin-right: 2px;
                border: 1px solid #2a2a2a;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background-color: #0a0a0a;
                color: #fff;
                border-bottom: 1px solid #0a0a0a;
            }
            QTabBar::tab:hover:!selected {
                background-color: #2a2a2a;
                color: #ccc;
            }
            QTabBar::close-button {
                image: none;
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 16px;
                height: 16px;
                background-color: transparent;
            }
            QTabBar::close-button:hover {
                background-color: #ff4444;
                border-radius: 2px;
            }
        """)
        
        # Connect signals
        self.tab_bar.currentChanged.connect(self.on_tab_changed)
        self.tab_bar.tabCloseRequested.connect(self.on_tab_close_requested)
        
        layout.addWidget(self.tab_bar)
        
        # Add new tab button
        self.new_tab_btn = QPushButton("+")
        self.new_tab_btn.setFixedSize(30, 30)
        self.new_tab_btn.clicked.connect(self.add_new_analysis_tab)
        self.new_tab_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1a;
                color: #999;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2a2a2a;
                color: #fff;
            }
        """)
        layout.addWidget(self.new_tab_btn)
        
        layout.addStretch()
    
    def setup_permanent_tabs(self):
        """Setup permanent tabs that can't be closed"""
        # Stock Fetcher tab
        idx = self.tab_bar.addTab("Stock Fetcher")
        self.tabs[idx] = ('stock_fetcher', 0, "Stock Fetcher")
        self.tab_bar.setTabButton(idx, QTabBar.ButtonPosition.RightSide, None)  # No close button
        
        # Default Analysis tab
        idx = self.tab_bar.addTab("Analysis")
        self.tabs[idx] = ('analysis', 0, "Analysis")
        self.tab_bar.setTabButton(idx, QTabBar.ButtonPosition.RightSide, None)  # No close button
        
        # Select Stock Fetcher by default
        self.tab_bar.setCurrentIndex(0)
        k2_logger.ui_operation("Tab bar initialized", "Permanent tabs created")
    
    def add_new_analysis_tab(self):
        """Add a new Analysis tab instance"""
        self.current_analysis_id += 1
        tab_id = self.current_analysis_id
        title = f"Analysis {tab_id}"
        
        idx = self.tab_bar.addTab(title)
        self.tabs[idx] = ('analysis', tab_id, title)
        
        # Make it closeable (default behavior)
        self.tab_bar.setCurrentIndex(idx)
        
        k2_logger.ui_operation(f"New analysis tab created", f"Tab ID: {tab_id}")
        self.new_tab_requested.emit('analysis')
    
    def on_tab_changed(self, index):
        """Handle tab selection change"""
        if index in self.tabs:
            page_type, tab_id, title = self.tabs[index]
            k2_logger.ui_operation(f"Tab switched", f"{title} (ID: {tab_id})")
            self.tab_changed.emit(page_type, tab_id)
    
    def on_tab_close_requested(self, index):
        """Handle tab close request"""
        if index in self.tabs:
            page_type, tab_id, title = self.tabs[index]
            
            # Check if it's a permanent tab
            tab_key = f"{page_type}_{tab_id}" if page_type == 'analysis' and tab_id == 0 else page_type
            if tab_key in self.permanent_tabs or (page_type == 'analysis' and tab_id == 0):
                k2_logger.warning(f"Cannot close permanent tab: {title}", "TAB_BAR")
                return
            
            # Remove the tab
            self.tab_bar.removeTab(index)
            
            # Update tabs dictionary
            del self.tabs[index]
            
            # Reindex remaining tabs
            new_tabs = {}
            for i in range(self.tab_bar.count()):
                for old_idx, tab_data in self.tabs.items():
                    if old_idx > index:
                        new_tabs[old_idx - 1] = tab_data
                    elif old_idx < index:
                        new_tabs[old_idx] = tab_data
            self.tabs = new_tabs
            
            k2_logger.ui_operation(f"Tab closed", f"{title} (ID: {tab_id})")
            self.tab_closed.emit(page_type, tab_id)
    
    def get_current_tab(self):
        """Get current tab information"""
        index = self.tab_bar.currentIndex()
        if index in self.tabs:
            return self.tabs[index]
        return None
    
    def select_tab(self, page_type, tab_id=0):
        """Programmatically select a tab"""
        for index, (p_type, t_id, _) in self.tabs.items():
            if p_type == page_type and t_id == tab_id:
                self.tab_bar.setCurrentIndex(index)
                return True
        return False