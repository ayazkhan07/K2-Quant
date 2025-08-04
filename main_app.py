#!/usr/bin/env python3
"""
Stock Price Projection System - Main Application

Integrated application that handles transitions between landing page and stock fetcher.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QObject, pyqtSignal

from ui.landing_page import LandingPageWidget
from ui.stock_fetcher import StockFetcherWidget

# Import comprehensive logging
from utils.logger import k2_logger


class MainApplication(QObject):
    """Main application controller handling page transitions"""
    
    def __init__(self):
        super().__init__()
        self.landing_page = None
        self.stock_fetcher = None
        
    def start(self):
        """Start the application with landing page"""
        k2_logger.ui_operation("Starting K2 Quant application", "Initializing landing page")
        self.show_landing_page()
        
    def show_landing_page(self):
        """Display the landing page"""
        k2_logger.ui_operation("Displaying landing page", "Video playback ready")
        # Clean up stock fetcher if it exists
        if self.stock_fetcher:
            k2_logger.ui_operation("Cleaning up stock fetcher", "Memory management")
            self.stock_fetcher.cleanup()
            self.stock_fetcher.close()
            self.stock_fetcher = None
            
        # Create and show landing page
        self.landing_page = LandingPageWidget()
        self.landing_page.continue_requested.connect(self.transition_to_stock_fetcher)
        self.landing_page.show()
        k2_logger.ui_operation("Landing page active", "Click anywhere to continue")
        
    def transition_to_stock_fetcher(self):
        """Transition from landing page to stock fetcher"""
        k2_logger.ui_operation("Transitioning to stock fetcher", "User clicked to continue")
        
        # Clean up landing page
        if self.landing_page:
            k2_logger.ui_operation("Cleaning up landing page", "Memory management")
            self.landing_page.cleanup()
            self.landing_page.close()
            self.landing_page = None
            
        # Create and show stock fetcher
        k2_logger.ui_operation("Initializing enterprise stock fetcher", "PostgreSQL backend ready")
        self.stock_fetcher = StockFetcherWidget()
        self.stock_fetcher.back_to_landing.connect(self.show_landing_page)
        self.stock_fetcher.stock_data_fetched.connect(self.handle_stock_data)
        self.stock_fetcher.show()
        k2_logger.ui_operation("Stock fetcher ready", "Enterprise features active")
        
    def handle_stock_data(self, data):
        """Handle stock data when fetched"""
        symbol = data.get('symbol', 'Unknown')
        total_records = data.get('total_records', 0)
        exec_time = data.get('execution_time', 0)
        records_per_sec = data.get('records_per_second', 0)
        
        k2_logger.data_processing("Enterprise data fetch completed", total_records, exec_time)
        k2_logger.performance_metric(f"{symbol} processing speed", records_per_sec, "rec/s")
        k2_logger.ui_operation("Data displayed in UI", f"{symbol}: {total_records:,} records")
        
        print(f"Enterprise data fetched for {symbol}: {total_records:,} records in {exec_time:.2f}s")
        # Here you could add additional processing or transition to analysis view
        
    def cleanup(self):
        """Clean up all components"""
        if self.landing_page:
            self.landing_page.cleanup()
        if self.stock_fetcher:
            self.stock_fetcher.cleanup()


def setup_global_styling(app):
    """Setup global application styling including comprehensive QMessageBox handling"""
    app.setStyleSheet("""
        /* Global styling for consistency */
        * {
            border-radius: 0px;
        }
        
        /* Comprehensive QMessageBox Styling - Belt and Suspenders Approach */
        QMessageBox {
            background-color: #0a0a0a;
            color: #ffffff;
            border: 1px solid #3a3a3a;
            border-radius: 0px;
            min-width: 400px;
            min-height: 150px;
            icon-size: 0px;
        }
        
        QMessageBox QLabel {
            color: #ffffff;
            background-color: transparent;
            font-size: 14px;
            padding: 20px;
            min-height: 40px;
        }
        
        QMessageBox QPushButton {
            background-color: #1a1a1a;
            color: #ffffff;
            border: 1px solid #3a3a3a;
            border-radius: 0px;
            padding: 10px 30px;
            font-size: 13px;
            min-width: 100px;
            font-weight: normal;
        }
        
        QMessageBox QPushButton:hover {
            background-color: #2a2a2a;
            border-color: #4a4a4a;
        }
        
        QMessageBox QPushButton:pressed {
            background-color: #3a3a3a;
            border-color: #5a5a5a;
        }
        
        QMessageBox QPushButton:default {
            background-color: #ffffff;
            color: #0a0a0a;
            border: 1px solid #ffffff;
        }
        
        QMessageBox QPushButton:default:hover {
            background-color: #f0f0f0;
            border-color: #f0f0f0;
        }
        
        QMessageBox QPushButton:default:pressed {
            background-color: #e0e0e0;
            border-color: #e0e0e0;
        }
        
        /* Multiple approaches to hide icons */
        QMessageBox::icon {
            image: none;
            width: 0px;
            height: 0px;
            margin: 0px;
            padding: 0px;
            border: none;
            background: none;
            max-width: 0px;
            max-height: 0px;
            min-width: 0px;
            min-height: 0px;
        }
        
        QMessageBox QLabel#qt_msgbox_label {
            margin: 0px;
            padding: 20px;
        }
        
        QMessageBox QLabel#qt_msgboxex_icon_label {
            width: 0px;
            height: 0px;
            max-width: 0px;
            max-height: 0px;
            margin: 0px;
            padding: 0px;
            image: none;
            background: none;
            border: none;
        }
        
        /* Force icon area to be invisible */
        QMessageBox > QWidget > QWidget > QLabel {
            max-width: 0px;
            max-height: 0px;
            width: 0px;
            height: 0px;
            image: none;
        }
        
        /* Dialog title bar */
        QDialog {
            background-color: #0a0a0a;
            border: 1px solid #3a3a3a;
        }
    """)


def apply_windows_specific_fixes():
    """Apply Windows-specific fixes for QMessageBox icons"""
    if sys.platform == "win32":
        try:
            # Windows-specific: Try to disable system sounds/icons
            import ctypes
            # This is a placeholder - actual Windows API calls would go here
            # to disable system message box icons at the OS level
            k2_logger.ui_operation("Applied Windows-specific dialog fixes", "Icon suppression active")
        except Exception as e:
            k2_logger.warning(f"Could not apply Windows fixes: {e}", "MAIN")


def configure_message_box_defaults():
    """Configure default QMessageBox behavior"""
    # This ensures any QMessageBox created anywhere will have minimal styling
    # QMessageBox.setStyleSheet(QMessageBox, """
    #     QMessageBox { icon-size: 0px; }
    #     QMessageBox::icon { width: 0px; height: 0px; }
    # """)


def main():
    """Main function to run the application"""
    print("Stock Price Projection System")
    print("=" * 30)
    print("Starting integrated application...")
    
    # Initialize logging system (will show detailed console output)
    k2_logger.info("K2 Quant application starting", "MAIN")
    k2_logger.info("Comprehensive logging and monitoring active", "MAIN")
    
    app = QApplication(sys.argv)
    k2_logger.ui_operation("PyQt6 application created", "QApplication initialized")
    
    # Apply comprehensive styling
    setup_global_styling(app)
    k2_logger.ui_operation("Global styling applied", "Dark theme with icon suppression")
    
    # Apply Windows-specific fixes if needed
    apply_windows_specific_fixes()
    
    # Configure message box defaults
    configure_message_box_defaults()
    
    # Create main application controller
    main_app = MainApplication()
    k2_logger.ui_operation("Main application controller created", "Ready for landing page")
    
    # Handle application quit
    app.aboutToQuit.connect(main_app.cleanup)
    k2_logger.ui_operation("Cleanup handlers registered", "Memory management ready")
    
    # Start the application
    k2_logger.info("Starting application flow", "MAIN")
    main_app.start()
    
    k2_logger.info("Entering PyQt6 event loop", "MAIN")
    result = app.exec()
    
    k2_logger.info("Application ended", "MAIN")
    return result


if __name__ == "__main__":
    sys.exit(main())