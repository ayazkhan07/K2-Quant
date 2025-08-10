#!/usr/bin/env python3
"""
K2 QUANT - Main Application with Tab Navigation

Integrated application with browser-style tabs for Stock Fetcher and Analysis pages.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QStackedWidget
from PyQt6.QtCore import QObject

from k2_quant.pages.landing.page import LandingPageWidget
from k2_quant.pages.stock_fetcher.page import StockFetcherWidget
from k2_quant.pages.analysis.page import AnalysisPageWidget
from k2_quant.components.tab_bar import TabBarWidget

from k2_quant.utilities.logger import k2_logger


class MainWindow(QMainWindow):
    """Main window with tab navigation"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("K2 QUANT - Stock Price Projection System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Track Analysis tab instances
        self.analysis_tabs = {}  # {tab_id: widget}
        
        self.init_ui()
        self.setup_styling()
        
    def init_ui(self):
        """Initialize the UI with tab bar and stacked widget"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        central_widget.setLayout(layout)
        
        # Add tab bar
        self.tab_bar = TabBarWidget()
        self.tab_bar.tab_changed.connect(self.on_tab_changed)
        self.tab_bar.new_tab_requested.connect(self.on_new_tab_requested)
        self.tab_bar.tab_closed.connect(self.on_tab_closed)
        layout.addWidget(self.tab_bar)
        
        # Create stacked widget for pages
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)
        
        # Create Stock Fetcher page
        self.stock_fetcher = StockFetcherWidget()
        self.stock_fetcher.stock_data_fetched.connect(self.handle_stock_data)
        self.stacked_widget.addWidget(self.stock_fetcher)
        
        # Create default Analysis page (tab_id=0)
        self.default_analysis = AnalysisPageWidget(tab_id=0)
        self.analysis_tabs[0] = self.default_analysis
        self.stacked_widget.addWidget(self.default_analysis)
        
        # Show Stock Fetcher by default
        self.stacked_widget.setCurrentWidget(self.stock_fetcher)
    
    def on_tab_changed(self, page_type: str, tab_id: int):
        """Handle tab selection change"""
        k2_logger.ui_operation(f"Tab changed", f"Type: {page_type}, ID: {tab_id}")
        
        if page_type == 'stock_fetcher':
            self.stacked_widget.setCurrentWidget(self.stock_fetcher)
        elif page_type == 'analysis':
            if tab_id in self.analysis_tabs:
                self.stacked_widget.setCurrentWidget(self.analysis_tabs[tab_id])
            else:
                k2_logger.error(f"Analysis tab {tab_id} not found", "MAIN")
    
    def on_new_tab_requested(self, page_type: str):
        """Handle new tab request"""
        if page_type == 'analysis':
            # Get the next tab ID from tab bar
            tab_id = self.tab_bar.current_analysis_id
            
            # Create new Analysis instance
            new_analysis = AnalysisPageWidget(tab_id=tab_id)
            self.analysis_tabs[tab_id] = new_analysis
            
            # Add to stacked widget
            self.stacked_widget.addWidget(new_analysis)
            
            # Switch to new tab
            self.stacked_widget.setCurrentWidget(new_analysis)
            
            k2_logger.ui_operation(f"New Analysis tab created", f"Tab ID: {tab_id}")
    
    def on_tab_closed(self, page_type: str, tab_id: int):
        """Handle tab close"""
        if page_type == 'analysis' and tab_id in self.analysis_tabs:
            # Get the widget
            widget = self.analysis_tabs[tab_id]
            
            # Clean up
            widget.cleanup()
            
            # Remove from stacked widget
            self.stacked_widget.removeWidget(widget)
            
            # Delete from tracking
            del self.analysis_tabs[tab_id]
            
            # Delete widget
            widget.deleteLater()
            
            k2_logger.ui_operation(f"Analysis tab closed", f"Tab ID: {tab_id}")
    
    def handle_stock_data(self, data):
        """Handle stock data when fetched"""
        symbol = data.get('symbol', 'Unknown')
        total_records = data.get('total_records', 0)
        
        k2_logger.data_processing("Data fetched", total_records)
        
        # Optional: Auto-switch to Analysis tab
        # self.tab_bar.select_tab('analysis', 0)
    
    def setup_styling(self):
        """Apply global styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0a0a;
            }
            QStackedWidget {
                background-color: #0a0a0a;
                border: none;
            }
        """)
    
    def cleanup(self):
        """Clean up all components"""
        # Clean up Analysis tabs
        for tab_id, widget in self.analysis_tabs.items():
            widget.cleanup()
        
        # Clean up Stock Fetcher
        if hasattr(self, 'stock_fetcher'):
            self.stock_fetcher.cleanup()


class MainApplication(QObject):
    """Main application controller"""

    def __init__(self):
        super().__init__()
        self.landing_page = None
        self.main_window = None

    def start(self):
        """Start the application with landing page"""
        k2_logger.ui_operation("Starting K2 Quant application", "Initializing landing page")
        self.show_landing_page()

    def show_landing_page(self):
        """Display the landing page"""
        k2_logger.ui_operation("Displaying landing page", "Video playback ready")
        
        # Create and show landing page
        self.landing_page = LandingPageWidget()
        self.landing_page.continue_requested.connect(self.transition_to_main_app)
        self.landing_page.show()
        k2_logger.ui_operation("Landing page active", "Click anywhere to continue")

    def transition_to_main_app(self):
        """Transition from landing page to main application"""
        k2_logger.ui_operation("Transitioning to main application", "User clicked to continue")

        # Clean up landing page
        if self.landing_page:
            k2_logger.ui_operation("Cleaning up landing page", "Memory management")
            self.landing_page.cleanup()
            self.landing_page.close()
            self.landing_page = None

        # Create and show main window
        k2_logger.ui_operation("Initializing main application", "Tab navigation ready")
        self.main_window = MainWindow()
        self.main_window.show()
        k2_logger.ui_operation("Main application ready", "Multi-tab interface active")

    def cleanup(self):
        """Clean up all components"""
        if self.landing_page:
            self.landing_page.cleanup()
        if self.main_window:
            self.main_window.cleanup()


def setup_global_styling(app):
    """Setup global application styling"""
    app.setStyleSheet("""
        /* Global styling for consistency */
        * { border-radius: 0px; }

        /* QMessageBox Styling */
        QMessageBox { 
            background-color: #0a0a0a; 
            color: #ffffff; 
            border: 1px solid #3a3a3a; 
            min-width: 400px; 
            min-height: 150px; 
        }
        QMessageBox QLabel { 
            color: #ffffff; 
            background-color: transparent; 
            font-size: 14px; 
            padding: 20px; 
        }
        QMessageBox QPushButton { 
            background-color: #1a1a1a; 
            color: #ffffff; 
            border: 1px solid #3a3a3a; 
            padding: 10px 30px; 
            font-size: 13px; 
            min-width: 100px; 
        }
        QMessageBox QPushButton:hover { 
            background-color: #2a2a2a; 
            border-color: #4a4a4a; 
        }
        QMessageBox QPushButton:default { 
            background-color: #ffffff; 
            color: #0a0a0a; 
        }
        
        /* Dialog styling */
        QDialog { 
            background-color: #0a0a0a; 
            border: 1px solid #3a3a3a; 
        }
    """)


def main():
    """Main function to run the application"""
    print("K2 QUANT - Stock Price Projection System")
    print("=" * 40)
    print("Starting integrated application...")

    k2_logger.info("K2 Quant application starting", "MAIN")

    app = QApplication(sys.argv)
    k2_logger.ui_operation("PyQt6 application created", "QApplication initialized")

    setup_global_styling(app)
    k2_logger.ui_operation("Global styling applied", "Dark theme active")

    main_app = MainApplication()
    k2_logger.ui_operation("Main application controller created", "Ready for landing page")

    app.aboutToQuit.connect(main_app.cleanup)
    k2_logger.ui_operation("Cleanup handlers registered", "Memory management ready")

    k2_logger.info("Starting application flow", "MAIN")
    main_app.start()

    k2_logger.info("Entering PyQt6 event loop", "MAIN")
    result = app.exec()
    k2_logger.info("Application ended", "MAIN")
    return result


if __name__ == "__main__":
    sys.exit(main())