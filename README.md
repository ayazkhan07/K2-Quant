# Stock Price Projection System

A professional desktop application for stock data fetching and price projection analysis, built with PyQt6.

## 🚀 Features

### Landing Page
- **Video Background**: Auto-playing MP4 video with professional presentation
- **Click-to-Continue**: Smooth fade transition to stock fetcher
- **Responsive Design**: Scales beautifully across different screen sizes
- **Fallback UI**: Elegant gradient background if video unavailable

### Stock Fetcher
- **Real-time Data**: Stock symbol input with simulated market data fetching
- **Historical Data**: 30-day historical price information display
- **Professional Interface**: Clean, modern design with progress indicators
- **Export Functionality**: Save fetched data to JSON format
- **Navigation**: Seamless transitions between components

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- PyQt6 with multimedia support

### Quick Setup
```bash
# Clone or download the project
cd "K2 Quant"

# Install dependencies
pip install -r requirements.txt

# Run the complete application
python run_app.py
```

### Individual Component Testing
```bash
# Test landing page only
python test_landing_page.py

# Test stock fetcher only  
python test_stock_fetcher.py
```

## 🎯 Usage

### Complete Application Flow
1. **Landing Page**: Application starts with video background
2. **Click Anywhere**: Triggers smooth fade transition
3. **Stock Fetcher**: Enter stock symbols (AAPL, GOOGL, MSFT, etc.)
4. **View Data**: See real-time quotes and historical information
5. **Export/Continue**: Save data or proceed to analysis

### Stock Symbol Examples
- `AAPL` - Apple Inc.
- `GOOGL` - Google/Alphabet
- `MSFT` - Microsoft Corporation
- `TSLA` - Tesla Inc.
- `AMZN` - Amazon

## 📁 Project Structure

```
K2 Quant/
├── ui/                          # UI Components
│   ├── __init__.py             # Package initialization
│   ├── landing_page.py         # Video landing page
│   └── stock_fetcher.py        # Stock data interface
├── assets/                     # Media assets
│   └── README_video_placement.txt
├── agents/                     # AI Agent system
│   ├── bana_architect.py       # Senior architect agent
│   ├── mussadiq_tester.py      # Testing specialist  
│   └── mussahih_fixer.py       # Issue resolution agent
├── main_app.py                 # Integrated application
├── run_app.py                  # Quick launcher
├── test_landing_page.py        # Landing page test
├── test_stock_fetcher.py       # Stock fetcher test
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🎨 Interface Design

### Landing Page
- **Black Theme**: Professional, cinematic appearance
- **Centered Video**: 16:9 aspect ratio, 80% viewport height
- **Fade Transition**: Smooth 0.5-second animation
- **Full-screen Click**: Any click triggers transition

### Stock Fetcher  
- **Clean Layout**: Modern card-based design
- **Progress Indicators**: Real-time feedback during data fetching
- **Data Tables**: Organized historical information display
- **Action Buttons**: Export, navigation, and analysis options

## 🔧 Technical Details

### Architecture
- **PyQt6**: Modern Qt framework for desktop applications
- **Component-based**: Modular UI components with clean separation
- **Signal-slot**: Event-driven communication between components
- **Thread-safe**: Background data fetching without UI blocking

### Data Simulation
The current version uses intelligent data simulation:
- **Realistic Prices**: Generated based on symbol characteristics
- **Historical Trends**: 30-day mock historical data
- **Market Metrics**: Volume, P/E ratios, market cap simulation
- **Progress Simulation**: Realistic data fetching delays

### Error Handling
- **Video Fallback**: Graceful handling of missing MP4 files
- **Input Validation**: Symbol format and length validation
- **Resource Cleanup**: Proper cleanup of threads and media players
- **User Feedback**: Clear error messages and status updates

## 🎥 Video Setup

### Adding Your Video
1. Place your video file in `assets/` directory and name it `k2_logo.mp4`
2. Recommended: 1920x1080 resolution, H.264 codec
3. Any duration (will auto-loop)
4. Audio optional (automatically muted)

### Supported Formats
- **Primary**: MP4 with H.264 video codec
- **Fallback**: Automatic gradient background if video unavailable
- **Responsive**: Scales to any screen size maintaining aspect ratio

## 🚀 Running the Application

### Full Application
```bash
python run_app.py
```
**Experience**: Landing page → click anywhere → stock fetcher interface

### Individual Components
```bash
# Landing page only
python test_landing_page.py

# Stock fetcher only
python test_stock_fetcher.py
```

## 💡 Development Features

### Simple Structure
- **Contained**: All components in logical directories
- **No Over-engineering**: Clean, straightforward architecture
- **Easy Integration**: Components communicate via Qt signals
- **Professional Quality**: Enterprise-grade code standards

### Agent System
The project includes an AI agent architecture:
- **Bana**: Senior architect for system design
- **Mussadiq**: Testing and quality assurance specialist  
- **Mussahih**: Issue resolution and debugging expert

## 🔮 Future Enhancements

### Next Phase Features
- **Real Market Data**: Integration with financial APIs
- **Advanced Analytics**: Technical indicators and trend analysis
- **Prediction Models**: ML-based price projection algorithms
- **Portfolio Management**: Multi-stock tracking and analysis
- **Charting**: Interactive price charts and visualization

### Technical Roadmap
- **Database Integration**: SQLite for local data storage
- **API Connectivity**: Real-time market data feeds
- **Advanced UI**: Charts, graphs, and interactive elements
- **Export Options**: Multiple data format support

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, or Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 100MB for application, additional for video assets

### Recommended Setup
- **Python**: 3.10+ for optimal performance
- **RAM**: 8GB+ for smooth video playback
- **GPU**: Dedicated graphics for hardware acceleration
- **Display**: 1920x1080 or higher resolution

---

## 📄 License
Professional enterprise software - all rights reserved

## 👨‍💻 Development Team
**Bana Architect**: Senior Developer/System Architect  
**Built with**: Professional-grade standards and enterprise-level quality assurance