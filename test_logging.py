#!/usr/bin/env python3
"""
Test script to demonstrate the comprehensive logging system
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import k2_logger
import time


def test_logging_system():
    """Test the comprehensive logging system"""
    
    k2_logger.info("Testing K2 Quant logging system", "TEST")
    time.sleep(1)
    
    k2_logger.ui_operation("Starting mock UI operations", "Testing user interface logging")
    time.sleep(0.5)
    
    k2_logger.database_operation("Mock database connection", "PostgreSQL k2_quant")
    time.sleep(0.5)
    
    k2_logger.api_operation("Mock API call", "Polygon.io symbol validation")
    time.sleep(0.5)
    
    k2_logger.step(1, 4, "Mock data preparation")
    time.sleep(0.5)
    
    k2_logger.step(2, 4, "Mock parallel processing")
    time.sleep(0.5)
    
    k2_logger.data_processing("Mock data processing", 50000, 2.5)
    time.sleep(0.5)
    
    k2_logger.performance_metric("Mock processing speed", 20000, "records/sec")
    time.sleep(0.5)
    
    k2_logger.step(3, 4, "Mock database storage")
    time.sleep(0.5)
    
    k2_logger.database_operation("Mock bulk insert", "50,000 records inserted")
    time.sleep(0.5)
    
    k2_logger.step(4, 4, "Mock completion")
    time.sleep(0.5)
    
    k2_logger.warning("This is a test warning", "TEST")
    time.sleep(0.5)
    
    k2_logger.info("Logging system test completed successfully", "TEST")
    
    print("\n🎉 Logging system is working! This is what you'll see during real operations.")
    print("📋 Log files are saved in the 'logs/' directory for detailed analysis.")


if __name__ == "__main__":
    test_logging_system()