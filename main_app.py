#!/usr/bin/env python3
"""
Stock Price Projection System - Main Application

Integrated application that handles transitions between landing page and stock fetcher.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal

from ui.landing_page import LandingPageWidget
from ui.stock_fetcher import StockFetcherWidget


class MainApplication(QObject):
    """Main application controller handling page transitions"""
    
    def __init__(self):
        super().__init__()
        self.landing_page = None
        self.stock_fetcher = None
        
    def start(self):
        """Start the application with landing page"""
        self.show_landing_page()
        
    def show_landing_page(self):
        """Display the landing page"""
        # Clean up stock fetcher if it exists
        if self.stock_fetcher:
            self.stock_fetcher.cleanup()
            self.stock_fetcher.close()
            self.stock_fetcher = None
            
        # Create and show landing page
        self.landing_page = LandingPageWidget()
        self.landing_page.continue_requested.connect(self.transition_to_stock_fetcher)
        self.landing_page.show()
        
    def transition_to_stock_fetcher(self):
        """Transition from landing page to stock fetcher"""
        # Clean up landing page
        if self.landing_page:
            self.landing_page.cleanup()
            self.landing_page.close()
            self.landing_page = None
            
        # Create and show stock fetcher
        self.stock_fetcher = StockFetcherWidget()
        self.stock_fetcher.back_to_landing.connect(self.show_landing_page)
        self.stock_fetcher.stock_data_fetched.connect(self.handle_stock_data)
        self.stock_fetcher.show()
        
    def handle_stock_data(self, data):
        """Handle stock data when fetched"""
        print(f"Stock data received for {data['symbol']}: ${data['current_price']}")
        # Here you could add additional processing or transition to analysis view
        
    def cleanup(self):
        """Clean up all components"""
        if self.landing_page:
            self.landing_page.cleanup()
        if self.stock_fetcher:
            self.stock_fetcher.cleanup()


def main():
    """Main function to run the application"""
    print("Stock Price Projection System")
    print("=" * 30)
    print("Starting integrated application...")
    
    app = QApplication(sys.argv)
    
    # Create main application controller
    main_app = MainApplication()
    
    # Handle application quit
    app.aboutToQuit.connect(main_app.cleanup)
    
    # Start the application
    main_app.start()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())