#!/usr/bin/env python3
"""
Test script for the Stock Price Projection System Landing Page

This script demonstrates how to run the landing page component.
Make sure to place your MP4 video file in the 'assets' directory before running.

Usage:
    python test_landing_page.py

Requirements:
    - PyQt6 (with multimedia support)
    - MP4 video file in assets/ directory (optional - fallback UI will show if missing)
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ui.landing_page import LandingPageApplication
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure PyQt6 is installed: pip install PyQt6")
    sys.exit(1)


def main():
    """Main function to run the landing page test"""
    print("Stock Price Projection System - Landing Page Test")
    print("=" * 50)
    
    # Check for k2_logo.mp4 video file in assets directory
    assets_path = Path("assets")
    if not assets_path.exists():
        print("Assets directory not found. Creating it...")
        assets_path.mkdir(exist_ok=True)
    
    k2_logo_path = assets_path / "k2_logo.mp4"
    if k2_logo_path.exists():
        print("Found k2_logo.mp4 video file.")
        print("Video will be played in the landing page.")
    else:
        print("k2_logo.mp4 not found in assets/ directory.")
        print("Fallback UI will be displayed with dark gradient background.")
        print("\nTo test with video:")
        print("1. Place your video file in the 'assets' directory")
        print("2. Rename it to 'k2_logo.mp4'")
        print("3. Run this script again")
    
    print("\nInstructions:")
    print("- The landing page will open in fullscreen")
    print("- Click anywhere on the screen to trigger the fade-out transition")
    print("- The application will close after the transition (in a real app, this would switch to the stock fetcher page)")
    print("\nStarting landing page...")
    
    # Create and run the application
    app = LandingPageApplication(sys.argv)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())