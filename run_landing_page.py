#!/usr/bin/env python3
"""
Quick launcher for the Stock Price Projection System Landing Page

This is a simple launcher script for easy execution of the landing page.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from test_landing_page import main

if __name__ == "__main__":
    sys.exit(main())