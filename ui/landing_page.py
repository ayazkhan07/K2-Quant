import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGraphicsOpacityEffect)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QUrl
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap, QPainter, QBrush, QLinearGradient
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

# Import logger without emojis for Windows compatibility
from utils.logger import k2_logger


class LandingPageWidget(QMainWindow):
    """Professional landing page with video background and click-to-continue functionality"""
    
    # Signal emitted when user clicks to continue
    continue_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_loaded = False
        self.fade_animation = None
        self.opacity_effect = None
        
        self.init_ui()
        self.setup_video_player()
        self.setup_animations()
        
    def init_ui(self):
        """Initialize the landing page user interface"""
        self.setWindowTitle("K2 Quant - Stock Price Projection System")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        
        # Set black background
        self.setStyleSheet("""
            QMainWindow {
                background-color: #000000;
            }
        """)
        
        # Central widget setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)
        
        # Video container with responsive sizing
        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: transparent;")
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_container.setLayout(video_layout)
        
        # Video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.video_widget.setStyleSheet("background-color: transparent;")
        
        # Center the video widget
        video_wrapper = QHBoxLayout()
        video_wrapper.addStretch()
        video_wrapper.addWidget(self.video_widget)
        video_wrapper.addStretch()
        
        video_layout.addStretch()
        video_layout.addLayout(video_wrapper)
        video_layout.addStretch()
        
        main_layout.addWidget(self.video_container)
        
        # Fallback UI (initially hidden)
        self.create_fallback_ui()
        
        # Enable click detection on entire window
        self.setMouseTracking(True)
        self.video_widget.setMouseTracking(True)
        self.video_container.setMouseTracking(True)
        central_widget.setMouseTracking(True)
        
        # Make all widgets clickable
        self.video_widget.mousePressEvent = self.handle_click
        self.video_container.mousePressEvent = self.handle_click
        central_widget.mousePressEvent = self.handle_click
        
    def create_fallback_ui(self):
        """Create fallback UI for when video fails to load"""
        self.fallback_widget = QWidget()
        self.fallback_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a1a, stop:0.5 #000000, stop:1 #1a1a1a);
            }
        """)
        self.fallback_widget.hide()
        
        fallback_layout = QVBoxLayout()
        fallback_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fallback_widget.setLayout(fallback_layout)
        
        # K2 Quant branding
        k2_label = QLabel("K2 QUANT")
        k2_label.setFont(QFont("Arial", 48, QFont.Weight.Bold))
        k2_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background: transparent;
                margin: 10px;
                text-align: center;
                letter-spacing: 8px;
            }
        """)
        k2_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fallback_layout.addWidget(k2_label)
        
        # Fallback title
        fallback_title = QLabel("Stock Price Projection System")
        fallback_title.setFont(QFont("Arial", 24))
        fallback_title.setStyleSheet("""
            QLabel {
                color: #cccccc;
                background: transparent;
                margin: 10px;
                text-align: center;
            }
        """)
        fallback_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fallback_layout.addWidget(fallback_title)
        
        # Add spacing
        fallback_layout.addSpacing(50)
        
        # Click to continue message
        continue_label = QLabel("Click anywhere to continue")
        continue_label.setFont(QFont("Arial", 16))
        continue_label.setStyleSheet("""
            QLabel {
                color: #999999;
                background: transparent;
                margin: 10px;
                text-align: center;
            }
        """)
        continue_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fallback_layout.addWidget(continue_label)
        
        # Add pulsing animation to continue label
        self.setup_pulsing_animation(continue_label)
        
        # Add fallback to main layout (will be shown only if video fails)
        self.centralWidget().layout().addWidget(self.fallback_widget)
        
        # Make fallback clickable
        self.fallback_widget.mousePressEvent = self.handle_click
        
    def setup_pulsing_animation(self, widget):
        """Setup pulsing animation for the continue label"""
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(lambda: self.pulse_effect(widget))
        self.pulse_timer.start(2000)  # Pulse every 2 seconds
        
    def pulse_effect(self, widget):
        """Create a subtle pulsing effect"""
        # This is a simple implementation - could be enhanced with opacity animation
        original_style = widget.styleSheet()
        widget.setStyleSheet(original_style.replace("color: #999999", "color: #ffffff"))
        
        QTimer.singleShot(200, lambda: widget.setStyleSheet(original_style))
        
    def setup_video_player(self):
        """Setup video player for MP4 playback"""
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        
        # Set up video output
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.setAudioOutput(self.audio_output)
        
        # Mute by default
        self.audio_output.setMuted(True)
        
        # Connect signals
        self.media_player.mediaStatusChanged.connect(self.handle_media_status)
        self.media_player.errorOccurred.connect(self.handle_media_error)
        
        # Load video file
        self.load_video()
        
    def load_video(self):
        """Load the k2_logo.mp4 video file from assets directory"""
        # Look specifically for k2_logo.mp4
        assets_path = Path("assets")
        k2_logo_path = assets_path / "videos" / "k2_logo.mp4"
        
        k2_logger.ui_operation("Loading video", f"Path: {k2_logo_path.absolute()}")
        
        if k2_logo_path.exists():
            try:
                # Convert to QUrl using fromLocalFile for proper handling
                video_url = QUrl.fromLocalFile(str(k2_logo_path.absolute()))
                k2_logger.ui_operation("Video URL created", f"URL: {video_url.toString()}")
                
                # Set the source using QUrl
                self.media_player.setSource(video_url)
                self.video_loaded = True
                k2_logger.info("Video loaded successfully", "LANDING")
            except Exception as e:
                k2_logger.error(f"Failed to load video: {str(e)}", "LANDING")
                self.show_fallback_ui()
        else:
            # k2_logo.mp4 not found, show fallback
            k2_logger.warning("Video file not found - showing fallback UI", "LANDING")
            self.show_fallback_ui()
            
    def handle_media_status(self, status):
        """Handle media player status changes"""
        status_messages = {
            QMediaPlayer.MediaStatus.NoMedia: "No media loaded",
            QMediaPlayer.MediaStatus.LoadingMedia: "Loading media...",
            QMediaPlayer.MediaStatus.LoadedMedia: "Media loaded successfully",
            QMediaPlayer.MediaStatus.StalledMedia: "Media stalled",
            QMediaPlayer.MediaStatus.BufferingMedia: "Buffering media...",
            QMediaPlayer.MediaStatus.BufferedMedia: "Media buffered",
            QMediaPlayer.MediaStatus.EndOfMedia: "End of media reached",
            QMediaPlayer.MediaStatus.InvalidMedia: "Invalid media"
        }
        status_msg = status_messages.get(status, f"Unknown status: {status}")
        k2_logger.ui_operation("Media status", status_msg)
        
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            # Video loaded successfully, start playback
            k2_logger.info("Starting video playback", "LANDING")
            self.media_player.play()
            self.resize_video_widget()
        elif status == QMediaPlayer.MediaStatus.EndOfMedia:
            # Loop the video
            k2_logger.info("Looping video", "LANDING")
            self.media_player.setPosition(0)
            self.media_player.play()
        elif status == QMediaPlayer.MediaStatus.InvalidMedia:
            k2_logger.error("Invalid media - showing fallback", "LANDING")
            self.show_fallback_ui()
            
    def handle_media_error(self, error):
        """Handle media player errors"""
        error_messages = {
            QMediaPlayer.Error.NoError: "No error",
            QMediaPlayer.Error.ResourceError: "Resource error - file not found or inaccessible",
            QMediaPlayer.Error.FormatError: "Format error - unsupported file format",
            QMediaPlayer.Error.NetworkError: "Network error",
            QMediaPlayer.Error.AccessDeniedError: "Access denied error"
        }
        error_msg = error_messages.get(error, f"Unknown error: {error}")
        k2_logger.error(f"Video playback error: {error_msg}", "LANDING")
        self.show_fallback_ui()
        
    def show_fallback_ui(self):
        """Show fallback UI when video fails"""
        self.video_container.hide()
        self.fallback_widget.show()
        self.video_loaded = False
        
    def resize_video_widget(self):
        """Resize video widget to maintain aspect ratio and responsive design"""
        if not self.video_loaded:
            return
            
        # Get screen dimensions
        screen_size = self.size()
        
        # Calculate 80% of viewport height
        target_height = int(screen_size.height() * 0.8)
        
        # Calculate width maintaining 16:9 aspect ratio
        target_width = int(target_height * 16 / 9)
        
        # Ensure it doesn't exceed screen width
        if target_width > screen_size.width() * 0.9:
            target_width = int(screen_size.width() * 0.9)
            target_height = int(target_width * 9 / 16)
            
        self.video_widget.setFixedSize(target_width, target_height)
        
    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        if self.video_loaded:
            # Delay resize to avoid too frequent updates
            QTimer.singleShot(100, self.resize_video_widget)
            
    def setup_animations(self):
        """Setup fade-out animation"""
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(500)  # 0.5 seconds
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        # Connect animation finished signal
        self.fade_animation.finished.connect(self.emit_continue_signal)
        
    def handle_click(self, event):
        """Handle click events anywhere on the screen"""
        if event.button() == Qt.MouseButton.LeftButton:
            k2_logger.ui_operation("User clicked", "Starting transition")
            self.start_transition()
            event.accept()
            
    def mousePressEvent(self, event):
        """Handle mouse press events on the main window"""
        if event.button() == Qt.MouseButton.LeftButton:
            k2_logger.ui_operation("User clicked window", "Starting transition")
            self.start_transition()
            event.accept()
        else:
            super().mousePressEvent(event)
            
    def start_transition(self):
        """Start the fade-out transition"""
        # Stop video playback
        if self.video_loaded:
            self.media_player.stop()
            
        # Start fade animation
        self.fade_animation.start()
        
    def emit_continue_signal(self):
        """Emit the continue signal after fade completes"""
        k2_logger.info("Transitioning to stock fetcher", "LANDING")
        self.continue_requested.emit()
        
    def cleanup(self):
        """Cleanup resources when closing"""
        if self.media_player:
            self.media_player.stop()
        if hasattr(self, 'pulse_timer'):
            self.pulse_timer.stop()


class LandingPageApplication(QApplication):
    """Standalone application for testing the landing page"""
    
    def __init__(self, argv):
        super().__init__(argv)
        self.landing_page = None
        
    def run(self):
        """Run the landing page application"""
        self.landing_page = LandingPageWidget()
        self.landing_page.continue_requested.connect(self.handle_continue)
        self.landing_page.show()
        
        return self.exec()
        
    def handle_continue(self):
        """Handle continue request from landing page"""
        k2_logger.info("User clicked to continue - transitioning to stock fetcher page", "LANDING")
        # Here you would transition to the next page
        # For now, we'll just close the application
        self.landing_page.cleanup()
        self.quit()


if __name__ == "__main__":
    app = LandingPageApplication(sys.argv)
    sys.exit(app.run())