# Stock Price Projection System - Landing Page

A professional landing page component built with PyQt6 featuring video background and smooth transitions.

## Features

### Core Functionality
- **Video Background**: Auto-playing, looping MP4 video with muted audio
- **Responsive Design**: 16:9 aspect ratio, scales to 80% of viewport height
- **Click-to-Continue**: Any click anywhere on screen triggers transition
- **Smooth Transitions**: 0.5-second fade-out effect before page switching
- **Error Handling**: Elegant fallback UI with dark gradient if video fails to load

### Technical Specifications
- Built with PyQt6 and QtMultimedia
- Supports MP4 video format
- Responsive video scaling maintains aspect ratio
- Professional black background theme
- Full-screen click detection
- Resource cleanup on exit

## Installation

### Prerequisites
- Python 3.8 or higher
- PyQt6 with multimedia support

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify PyQt6 Multimedia Installation
```bash
python -c "from PyQt6.QtMultimedia import QMediaPlayer; print('PyQt6 Multimedia is installed')"
```

## Usage

### Basic Usage
1. **Add Your Video**: Place your video file in the `assets/` directory and name it `k2_logo.mp4`
2. **Run the Landing Page**:
   ```bash
   python test_landing_page.py
   ```
3. **Interact**: Click anywhere on the screen to trigger the transition

### Video Requirements
- **Filename**: Must be named exactly `k2_logo.mp4`
- **Format**: MP4 (H.264 codec recommended)
- **Aspect Ratio**: 16:9 (1920x1080, 1280x720, etc.)
- **Duration**: Any duration (will loop automatically)
- **Audio**: Optional (will be muted by default)

### Integration Example
```python
from ui.landing_page import LandingPageWidget

class MainApplication:
    def __init__(self):
        self.landing_page = LandingPageWidget()
        self.landing_page.continue_requested.connect(self.show_stock_fetcher)
        
    def show_landing_page(self):
        self.landing_page.show()
        
    def show_stock_fetcher(self):
        self.landing_page.cleanup()
        # Transition to your stock fetcher page
        self.stock_fetcher.show()
```

## Architecture

### Component Structure
```
ui/
├── landing_page.py          # Main landing page component
├── __init__.py             # Package initialization
assets/
├── your_video.mp4          # Your MP4 video file
test_landing_page.py        # Test/demo script
```

### Key Classes

#### `LandingPageWidget(QMainWindow)`
Main landing page component with video playback and interaction handling.

**Signals:**
- `continue_requested`: Emitted when user clicks to continue

**Key Methods:**
- `handle_click(event)`: Processes click events
- `start_transition()`: Initiates fade-out transition
- `cleanup()`: Cleans up resources

#### `LandingPageApplication(QApplication)`
Standalone application wrapper for testing and demonstration.

## Customization

### Styling
The landing page uses CSS-like styling. Key style elements:

```python
# Black background
background-color: #000000

# Fallback gradient
background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
    stop:0 #1a1a1a, stop:0.5 #000000, stop:1 #1a1a1a)
```

### Animation Timing
```python
# Fade duration (currently 0.5 seconds)
self.fade_animation.setDuration(500)

# Easing curve
self.fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
```

### Video Sizing
```python
# Video takes 80% of viewport height
target_height = int(screen_size.height() * 0.8)

# Maximum width is 90% of screen width
if target_width > screen_size.width() * 0.9:
    target_width = int(screen_size.width() * 0.9)
```

## Fallback Behavior

If video loading fails or no MP4 file is found:
1. **Dark Gradient Background**: Professional-looking gradient backdrop
2. **Title Display**: "Stock Price Projection System" in white text
3. **Continue Prompt**: "Click anywhere to continue" with subtle pulsing animation
4. **Full Functionality**: Click detection still works normally

## Error Handling

### Video Loading Errors
- Automatically switches to fallback UI
- Logs error information to console
- Maintains all click functionality

### Missing Assets Directory
- Creates assets directory if it doesn't exist
- Shows fallback UI with appropriate messaging
- Continues normal operation

## Performance Considerations

### Resource Management
- Video player resources are properly cleaned up
- Timers are stopped when component is destroyed
- Memory usage is optimized for smooth playback

### Responsive Performance
- Video resizing is debounced to prevent excessive updates
- Animation uses hardware acceleration when available
- Efficient event handling for click detection

## Troubleshooting

### Common Issues

**Video Not Playing**
- Ensure MP4 file is in the `assets/` directory
- Check that the video codec is supported (H.264 recommended)
- Verify PyQt6-multimedia is installed

**Click Detection Not Working**
- Ensure the window has focus
- Check that mouse tracking is enabled
- Verify event handling chain

**Performance Issues**
- Use smaller video file sizes
- Ensure adequate system resources
- Check video codec compatibility

### Debug Mode
Enable debug output by setting environment variable:
```bash
export QT_LOGGING_RULES="qt.multimedia.debug=true"
python test_landing_page.py
```

## Integration with Existing System

The landing page component is designed to integrate with your existing PyQt6 application:

1. **Import the Component**: `from ui.landing_page import LandingPageWidget`
2. **Create Instance**: `landing_page = LandingPageWidget()`
3. **Connect Signal**: `landing_page.continue_requested.connect(your_handler)`
4. **Show Page**: `landing_page.show()`
5. **Handle Transition**: In your handler, hide landing page and show next component

## Future Enhancements

Planned improvements for future versions:
- Multiple video format support
- Configurable animation effects
- Theme customization options
- Progress indicators for long videos
- Accessibility features

---

## Author
Built by Bana Architect as part of the Stock Price Projection System

## License
Professional enterprise software - all rights reserved