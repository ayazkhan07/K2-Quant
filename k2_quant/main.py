#!/usr/bin/env python3
"""
K2 QUANT - Main Application with Tab Navigation

Integrated application with browser-style tabs for Stock Fetcher and Analysis pages.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QStackedWidget
from PyQt6.QtCore import QObject, QSettings, QTimer

from k2_quant.pages.landing.page import LandingPageWidget
from k2_quant.pages.stock_fetcher.page import StockFetcherWidget
from k2_quant.pages.analysis.page import AnalysisPageWidget
from k2_quant.pages.stream.page import StreamPageWidget
from k2_quant.components.tab_bar import TabBarWidget

from k2_quant.utilities.logger import k2_logger


class MainWindow(QMainWindow):
    """Main window with tab navigation"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("K2 QUANT - Stock Price Projection System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Track Analysis and Stream tab instances
        self.analysis_tabs = {}  # {tab_id: widget}
        self.stream_tabs = {}   # {tab_id: widget}
        
        self.init_ui()
        self.setup_styling()
        QTimer.singleShot(0, self._restore_session)
        
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
        # Connect the new model_saved signal
        self.stock_fetcher.model_saved.connect(self.on_model_saved)
        # Connect database cleared broadcast
        try:
            self.stock_fetcher.database_cleared.connect(self.handle_database_cleared)
        except Exception:
            pass
        self.stacked_widget.addWidget(self.stock_fetcher)
        
        # Create default Analysis page (tab_id=0)
        self.default_analysis = AnalysisPageWidget(tab_id=0)
        self.analysis_tabs[0] = self.default_analysis
        self.stacked_widget.addWidget(self.default_analysis)
        
        # Create default Stream page (tab_id=0)
        self.default_stream = StreamPageWidget(tab_id=0)
        self.stream_tabs[0] = self.default_stream
        self.stacked_widget.addWidget(self.default_stream)
        
        # Show Stock Fetcher by default
        self.stacked_widget.setCurrentWidget(self.stock_fetcher)
    
    def on_model_saved(self, table_name: str):
        """Handle model saved signal from Stock Fetcher"""
        k2_logger.info(f"Model saved signal received: {table_name}", "MAIN")
        
        # Refresh all Analysis tabs
        for tab_id, widget in self.analysis_tabs.items():
            try:
                if hasattr(widget, 'refresh_models'):
                    widget.refresh_models()
                elif hasattr(widget, 'load_saved_models'):
                    widget.load_saved_models()
                k2_logger.info(f"Refreshed Analysis tab {tab_id}", "MAIN")
            except Exception as e:
                k2_logger.error(f"Failed to refresh Analysis tab {tab_id}: {str(e)}", "MAIN")
        
        # Refresh all Stream tabs
        for tab_id, widget in self.stream_tabs.items():
            try:
                if hasattr(widget, 'refresh_models'):
                    widget.refresh_models()
                elif hasattr(widget, 'load_saved_models'):
                    widget.load_saved_models()
                k2_logger.info(f"Refreshed Stream tab {tab_id}", "MAIN")
            except Exception as e:
                k2_logger.error(f"Failed to refresh Stream tab {tab_id}: {str(e)}", "MAIN")
    
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
        elif page_type == 'stream':
            if tab_id in self.stream_tabs:
                self.stacked_widget.setCurrentWidget(self.stream_tabs[tab_id])
            else:
                k2_logger.error(f"Stream tab {tab_id} not found", "MAIN")
    
    def on_new_tab_requested(self, page_type: str):
        """Handle new tab request"""
        if page_type == 'analysis':
            tab_id = self.tab_bar.current_analysis_id
            new_analysis = AnalysisPageWidget(tab_id=tab_id)
            self.analysis_tabs[tab_id] = new_analysis
            self.stacked_widget.addWidget(new_analysis)
            self.stacked_widget.setCurrentWidget(new_analysis)
            k2_logger.ui_operation(f"New Analysis tab created", f"Tab ID: {tab_id}")
        elif page_type == 'stream':
            tab_id = self.tab_bar.current_stream_id
            new_stream = StreamPageWidget(tab_id=tab_id)
            self.stream_tabs[tab_id] = new_stream
            self.stacked_widget.addWidget(new_stream)
            self.stacked_widget.setCurrentWidget(new_stream)
            k2_logger.ui_operation(f"New Stream tab created", f"Tab ID: {tab_id}")
    
    def on_tab_closed(self, page_type: str, tab_id: int):
        """Handle tab close"""
        if page_type == 'analysis' and tab_id in self.analysis_tabs:
            widget = self.analysis_tabs[tab_id]
            widget.cleanup()
            self.stacked_widget.removeWidget(widget)
            del self.analysis_tabs[tab_id]
            widget.deleteLater()
            k2_logger.ui_operation(f"Analysis tab closed", f"Tab ID: {tab_id}")
        elif page_type == 'stream' and tab_id in self.stream_tabs:
            widget = self.stream_tabs[tab_id]
            widget.cleanup()
            self.stacked_widget.removeWidget(widget)
            del self.stream_tabs[tab_id]
            widget.deleteLater()
            k2_logger.ui_operation(f"Stream tab closed", f"Tab ID: {tab_id}")

    def closeEvent(self, event):
        """Save session state before the window closes."""
        self._save_session()
        super().closeEvent(event)

    def _save_session(self):
        """Persist open analysis and stream tabs to QSettings."""
        settings = QSettings("K2Quant", "K2Quant")

        # Analysis tabs
        session_tabs = []
        for tab_id, widget in self.analysis_tabs.items():
            session_tabs.append({
                'tab_id': tab_id,
                'model': widget.current_model or '',
            })
        settings.setValue("session/analysis_tabs", json.dumps(session_tabs))

        # Stream tabs
        stream_sessions = []
        for tab_id, widget in self.stream_tabs.items():
            stream_sessions.append(widget.get_session_state())
        settings.setValue("session/stream_tabs", json.dumps(stream_sessions))

        current = self.tab_bar.get_current_tab()
        if current:
            page_type, tab_id, _title = current
            settings.setValue("session/active_page_type", page_type)
            settings.setValue("session/active_tab_id", tab_id)

        k2_logger.info(
            f"Session saved ({len(session_tabs)} analysis, "
            f"{len(stream_sessions)} stream tabs)", "MAIN")

    def _restore_session(self):
        """Recreate analysis and stream tabs from a previous session."""
        settings = QSettings("K2Quant", "K2Quant")

        # --- Restore analysis tabs ---
        raw = settings.value("session/analysis_tabs", "")
        analysis_count = 0
        if raw:
            try:
                session_tabs = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                session_tabs = []
            if isinstance(session_tabs, list):
                for tab_info in session_tabs:
                    tab_id = tab_info.get('tab_id', 0)
                    model = tab_info.get('model', '')
                    if tab_id != 0 and tab_id not in self.analysis_tabs:
                        self.tab_bar.restore_analysis_tab(tab_id)
                        new_analysis = AnalysisPageWidget(tab_id=tab_id)
                        self.analysis_tabs[tab_id] = new_analysis
                        self.stacked_widget.addWidget(new_analysis)
                    if model and tab_id in self.analysis_tabs:
                        try:
                            self.analysis_tabs[tab_id].load_model_by_table(model)
                        except Exception as e:
                            k2_logger.error(
                                f"Failed to restore model '{model}' on tab {tab_id}: {e}",
                                "MAIN")
                analysis_count = len(session_tabs)

        # --- Restore stream tabs ---
        raw_stream = settings.value("session/stream_tabs", "")
        stream_count = 0
        if raw_stream:
            try:
                stream_sessions = json.loads(raw_stream)
            except (json.JSONDecodeError, TypeError):
                stream_sessions = []
            if isinstance(stream_sessions, list):
                for state in stream_sessions:
                    tab_id = state.get('tab_id', 0)
                    if tab_id != 0 and tab_id not in self.stream_tabs:
                        self.tab_bar.restore_stream_tab(tab_id)
                        new_stream = StreamPageWidget(tab_id=tab_id)
                        self.stream_tabs[tab_id] = new_stream
                        self.stacked_widget.addWidget(new_stream)
                    if tab_id in self.stream_tabs:
                        try:
                            self.stream_tabs[tab_id].restore_session_state(state)
                        except Exception as e:
                            k2_logger.error(
                                f"Failed to restore stream tab {tab_id}: {e}", "MAIN")
                stream_count = len(stream_sessions)

        saved_page = settings.value("session/active_page_type", "stock_fetcher")
        saved_tab = settings.value("session/active_tab_id", 0)
        try:
            saved_tab = int(saved_tab)
        except (TypeError, ValueError):
            saved_tab = 0
        self.tab_bar.select_tab(saved_page, saved_tab)

        k2_logger.info(
            f"Session restored ({analysis_count} analysis, "
            f"{stream_count} stream tabs)", "MAIN")

    def handle_database_cleared(self):
        """Reset analysis and stream views and caches after database deletion."""
        k2_logger.info("Handling database cleared broadcast", "MAIN")
        settings = QSettings("K2Quant", "K2Quant")
        settings.remove("session/analysis_tabs")
        settings.remove("session/stream_tabs")
        settings.remove("session/active_page_type")
        settings.remove("session/active_tab_id")

        # Reset default analysis tab (ID 0)
        if 0 in self.analysis_tabs:
            widget = self.analysis_tabs[0]
            if hasattr(widget, 'reset_after_database_cleared'):
                try:
                    widget.reset_after_database_cleared()
                except Exception as e:
                    k2_logger.error(f"Failed to reset default analysis tab: {str(e)}", "MAIN")

        # Close extra analysis tabs
        for tid in [t for t in self.analysis_tabs if t != 0]:
            try:
                widget = self.analysis_tabs[tid]
                widget.cleanup()
                self.stacked_widget.removeWidget(widget)
                widget.deleteLater()
                del self.analysis_tabs[tid]
            except Exception as e:
                k2_logger.error(f"Failed closing analysis tab {tid}: {str(e)}", "MAIN")
        try:
            if hasattr(self.tab_bar, 'close_all_analysis_tabs_except_default'):
                self.tab_bar.close_all_analysis_tabs_except_default()
        except Exception:
            pass

        # Reset default stream tab (ID 0)
        if 0 in self.stream_tabs:
            widget = self.stream_tabs[0]
            if hasattr(widget, 'reset_after_database_cleared'):
                try:
                    widget.reset_after_database_cleared()
                except Exception as e:
                    k2_logger.error(f"Failed to reset default stream tab: {str(e)}", "MAIN")

        # Close extra stream tabs
        for tid in [t for t in self.stream_tabs if t != 0]:
            try:
                widget = self.stream_tabs[tid]
                widget.cleanup()
                self.stacked_widget.removeWidget(widget)
                widget.deleteLater()
                del self.stream_tabs[tid]
            except Exception as e:
                k2_logger.error(f"Failed closing stream tab {tid}: {str(e)}", "MAIN")
        try:
            if hasattr(self.tab_bar, 'close_all_stream_tabs_except_default'):
                self.tab_bar.close_all_stream_tabs_except_default()
        except Exception:
            pass

        # Clear model caches
        try:
            from k2_quant.utilities.services.model_loader_service import model_loader_service
            model_loader_service.clear_cache()
        except Exception as e:
            k2_logger.warning(f"Model cache clear failed: {str(e)}", "MAIN")
    
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
        for tab_id, widget in self.analysis_tabs.items():
            widget.cleanup()
        for tab_id, widget in self.stream_tabs.items():
            widget.cleanup()
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