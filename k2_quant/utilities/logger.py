"""
K2 Quant Comprehensive Logging System

Provides detailed logging for troubleshooting and monitoring enterprise operations.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
import threading
import queue
import time


class K2QuantLogger:
    """Enhanced logging system for K2 Quant enterprise operations"""

    def __init__(self, log_level=logging.INFO):
        self.setup_logging(log_level)
        self.setup_console_handler()

    def setup_logging(self, log_level):
        """Setup comprehensive logging system"""
        Path("logs").mkdir(exist_ok=True)

        self.logger = logging.getLogger("K2Quant")
        self.logger.setLevel(log_level)
        self.logger.handlers = []

        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s',
            datefmt='%H:%M:%S'
        )

        console_formatter = logging.Formatter(
            '🔹 %(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )

        file_handler = logging.FileHandler(
            f"logs/k2_quant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)

        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setFormatter(console_formatter)
        self.console_handler.setLevel(log_level)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(self.console_handler)

    def setup_console_handler(self):
        """Setup enhanced console output"""
        print("\n" + "="*80)
        print("** K2 QUANT ENTERPRISE STOCK DATA PLATFORM **")
        print("="*80)
        print("[INFO] Real-time logging and monitoring active")
        print("[DEBUG] Troubleshooting mode enabled")
        print("="*80 + "\n")

    def info(self, message, component="SYSTEM"):
        self.logger.info(f"[{component}] {message}")

    def debug(self, message, component="DEBUG"):
        self.logger.debug(f"[{component}] {message}")

    def warning(self, message, component="WARNING"):
        self.logger.warning(f"[{component}] {message}")

    def error(self, message, component="ERROR"):
        self.logger.error(f"[{component}] {message}")

    def critical(self, message, component="CRITICAL"):
        self.logger.critical(f"[{component}] {message}")

    def database_operation(self, operation, details=""):
        self.info(f"[DB] {operation} {details}", "DATABASE")

    def api_operation(self, operation, details=""):
        self.info(f"[API] {operation} {details}", "API")

    def performance_metric(self, metric, value, unit=""):
        self.info(f"[PERF] {metric}: {value} {unit}", "PERFORMANCE")

    def data_processing(self, operation, count=None, time_taken=None):
        msg = f"[DATA] {operation}"
        if count is not None:
            msg += f" | Records: {count:,}"
        if time_taken is not None:
            msg += f" | Time: {time_taken:.2f}s"
        self.info(msg, "DATA")

    def ui_operation(self, operation, details=""):
        self.info(f"[UI] {operation} {details}", "UI")

    def step(self, step_number, total_steps, description):
        progress = f"({step_number}/{total_steps})"
        self.info(f"[STEP] {progress}: {description}", "PROCESS")


k2_logger = K2QuantLogger()


def log_exception(func):
    """Decorator to log exceptions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            k2_logger.error(f"Exception in {func.__name__}: {str(e)}", "EXCEPTION")
            raise
    return wrapper


def log_performance(func):
    """Decorator to log function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        k2_logger.debug(f"Starting {func.__name__}", "PERFORMANCE")
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            k2_logger.performance_metric(f"{func.__name__} execution", execution_time, "seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            k2_logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {str(e)}", "PERFORMANCE")
            raise
    return wrapper



