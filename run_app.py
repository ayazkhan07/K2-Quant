#!/usr/bin/env python3
"""
Quick launcher for the complete Stock Price Projection System

This launches the integrated application starting with the landing page.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main_app import main

if __name__ == "__main__":
    sys.exit(main())