import sys
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QGraphicsOpacityEffect)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QUrl
from PyQt6.QtGui import QFont
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

from k2_quant.utilities.logger import k2_logger


class LandingPageWidget(QMainWindow):
    """Professional landing page with video background and click-to-continue functionality"""

    continue_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_loaded = False
        self.fade_animation = None
        self.opacity_effect = None
        self.media_player = None  # Initialize as None
        self.is_transitioning = False  # Add flag to prevent multiple transitions

        self.init_ui()
        self.setup_video_player()
        self.setup_animations()

    def init_ui(self):
        self.setWindowTitle("K2 Quant - Stock Price Projection System")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.setStyleSheet("""
            QMainWindow { background-color: #000000; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)

        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: transparent;")
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_container.setLayout(video_layout)

        self.video_widget = QVideoWidget()
        self.video_widget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.video_widget.setStyleSheet("background-color: transparent;")

        video_wrapper = QHBoxLayout()
        video_wrapper.addStretch()
        video_wrapper.addWidget(self.video_widget)
        video_wrapper.addStretch()

        video_layout.addStretch()
        video_layout.addLayout(video_wrapper)
        video_layout.addStretch()

        main_layout.addWidget(self.video_container)

        self.create_fallback_ui()

        # Enable click detection on entire window
        self.setMouseTracking(True)
        self.video_widget.setMouseTracking(True)
        self.video_container.setMouseTracking(True)
        central_widget.setMouseTracking(True)

        self.video_widget.mousePressEvent = self.handle_click
        self.video_container.mousePressEvent = self.handle_click
        central_widget.mousePressEvent = self.handle_click

    def create_fallback_ui(self):
        self.fallback_widget = QWidget()
        self.fallback_widget.setStyleSheet(
            """
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a1a, stop:0.5 #000000, stop:1 #1a1a1a);
            }
            """
        )
        self.fallback_widget.hide()

        fallback_layout = QVBoxLayout()
        fallback_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fallback_widget.setLayout(fallback_layout)

        k2_label = QLabel("K2 QUANT")
        k2_label.setFont(QFont("Arial", 48, QFont.Weight.Bold))
        k2_label.setStyleSheet(
            """
            QLabel { color: #ffffff; background: transparent; margin: 10px; text-align: center; letter-spacing: 8px; }
            """
        )
        k2_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fallback_layout.addWidget(k2_label)

        fallback_title = QLabel("Stock Price Projection System")
        fallback_title.setFont(QFont("Arial", 24))
        fallback_title.setStyleSheet("""
            QLabel { color: #cccccc; background: transparent; margin: 10px; text-align: center; }
        """)
        fallback_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fallback_layout.addWidget(fallback_title)

        fallback_layout.addSpacing(50)

        continue_label = QLabel("Click anywhere to continue")
        continue_label.setFont(QFont("Arial", 16))
        continue_label.setStyleSheet("""
            QLabel { color: #999999; background: transparent; margin: 10px; text-align: center; }
        """)
        continue_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fallback_layout.addWidget(continue_label)

        self.setup_pulsing_animation(continue_label)

        self.centralWidget().layout().addWidget(self.fallback_widget)
        self.fallback_widget.mousePressEvent = self.handle_click

    def setup_pulsing_animation(self, widget):
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(lambda: self.pulse_effect(widget))
        self.pulse_timer.start(2000)

    def pulse_effect(self, widget):
        original_style = widget.styleSheet()
        widget.setStyleSheet(original_style.replace("color: #999999", "color: #ffffff"))
        QTimer.singleShot(200, lambda: widget.setStyleSheet(original_style))

    def setup_video_player(self):
        try:
            self.media_player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.media_player.setVideoOutput(self.video_widget)
            self.media_player.setAudioOutput(self.audio_output)
            self.audio_output.setMuted(True)
            
            # Connect signals
            self.media_player.mediaStatusChanged.connect(self.handle_media_status)
            self.media_player.errorOccurred.connect(self.handle_media_error)
            
            self.load_video()
        except Exception as e:
            k2_logger.error(f"Failed to setup video player: {str(e)}", "LANDING")
            self.show_fallback_ui()

    def load_video(self):
        # Prefer local assets under page folder, then fall back to legacy root assets
        candidates = [
            (Path(__file__).resolve().parent / "assets" / "videos" / "k2_logo.mp4"),
            (Path.cwd() / "assets" / "videos" / "k2_logo.mp4"),
        ]

        for k2_logo_path in candidates:
            k2_logger.ui_operation("Loading video", f"Path: {k2_logo_path}")
            if k2_logo_path.exists():
                try:
                    video_url = QUrl.fromLocalFile(str(k2_logo_path))
                    k2_logger.ui_operation("Video URL created", f"URL: {video_url.toString()}")
                    self.media_player.setSource(video_url)
                    self.video_loaded = True
                    k2_logger.info("Video loaded successfully", "LANDING")
                    return
                except Exception as e:
                    k2_logger.error(f"Failed to load video: {str(e)}", "LANDING")
                    break

        k2_logger.warning("Video file not found - showing fallback UI", "LANDING")
        self.show_fallback_ui()

    def handle_media_status(self, status):
        # Don't process media status if we're transitioning
        if self.is_transitioning:
            return
            
        from PyQt6.QtMultimedia import QMediaPlayer as MP
        status_messages = {
            MP.MediaStatus.NoMedia: "No media loaded",
            MP.MediaStatus.LoadingMedia: "Loading media...",
            MP.MediaStatus.LoadedMedia: "Media loaded successfully",
            MP.MediaStatus.StalledMedia: "Media stalled",
            MP.MediaStatus.BufferingMedia: "Buffering media...",
            MP.MediaStatus.BufferedMedia: "Media buffered",
            MP.MediaStatus.EndOfMedia: "End of media reached",
            MP.MediaStatus.InvalidMedia: "Invalid media",
        }
        status_msg = status_messages.get(status, f"Unknown status: {status}")
        k2_logger.ui_operation("Media status", status_msg)

        if status == MP.MediaStatus.LoadedMedia:
            k2_logger.info("Starting video playback", "LANDING")
            self.media_player.play()
            self.resize_video_widget()
        elif status == MP.MediaStatus.EndOfMedia:
            # Only loop if we're not transitioning
            if not self.is_transitioning and self.media_player:
                k2_logger.info("Looping video", "LANDING")
                self.media_player.setPosition(0)
                self.media_player.play()
        elif status == MP.MediaStatus.InvalidMedia:
            k2_logger.error("Invalid media - showing fallback", "LANDING")
            self.show_fallback_ui()

    def handle_media_error(self, error):
        from PyQt6.QtMultimedia import QMediaPlayer as MP
        error_messages = {
            MP.Error.NoError: "No error",
            MP.Error.ResourceError: "Resource error - file not found or inaccessible",
            MP.Error.FormatError: "Format error - unsupported file format",
            MP.Error.NetworkError: "Network error",
            MP.Error.AccessDeniedError: "Access denied error",
        }
        error_msg = error_messages.get(error, f"Unknown error: {error}")
        k2_logger.error(f"Video playback error: {error_msg}", "LANDING")
        self.show_fallback_ui()

    def show_fallback_ui(self):
        self.video_container.hide()
        self.fallback_widget.show()
        self.video_loaded = False

    def resize_video_widget(self):
        if not self.video_loaded:
            return
        screen_size = self.size()
        target_height = int(screen_size.height() * 0.8)
        target_width = int(target_height * 16 / 9)
        if target_width > screen_size.width() * 0.9:
            target_width = int(screen_size.width() * 0.9)
            target_height = int(target_width * 9 / 16)
        self.video_widget.setFixedSize(target_width, target_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.video_loaded and not self.is_transitioning:
            QTimer.singleShot(100, self.resize_video_widget)

    def setup_animations(self):
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(500)
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.fade_animation.finished.connect(self.emit_continue_signal)

    def handle_click(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self.is_transitioning:
            k2_logger.ui_operation("User clicked", "Starting transition")
            self.start_transition()
            event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self.is_transitioning:
            k2_logger.ui_operation("User clicked window", "Starting transition")
            self.start_transition()
            event.accept()
        else:
            super().mousePressEvent(event)

    def start_transition(self):
        # Prevent multiple transitions
        if self.is_transitioning:
            return
            
        self.is_transitioning = True
        
        # Stop video immediately if loaded
        if self.video_loaded and self.media_player:
            try:
                # Disconnect signals to prevent further processing
                self.media_player.mediaStatusChanged.disconnect()
                self.media_player.errorOccurred.disconnect()
                
                # Stop and clear the media
                self.media_player.stop()
                self.media_player.setSource(QUrl())  # Clear the source
                
                k2_logger.info("Video stopped for transition", "LANDING")
            except Exception as e:
                k2_logger.error(f"Error stopping video: {str(e)}", "LANDING")
        
        # Start fade animation
        self.fade_animation.start()

    def emit_continue_signal(self):
        k2_logger.info("Transitioning to stock fetcher", "LANDING")
        self.continue_requested.emit()

    def cleanup(self):
        """Properly cleanup all resources"""
        k2_logger.info("Starting landing page cleanup", "LANDING")
        
        # Set transitioning flag to prevent any further media operations
        self.is_transitioning = True
        
        # Stop and cleanup media player
        if hasattr(self, 'media_player') and self.media_player:
            try:
                # Disconnect all signals
                try:
                    self.media_player.mediaStatusChanged.disconnect()
                    self.media_player.errorOccurred.disconnect()
                except:
                    pass  # Signals might already be disconnected
                
                # Stop playback
                self.media_player.stop()
                
                # Clear the source
                self.media_player.setSource(QUrl())
                
                # Delete the media player
                self.media_player.deleteLater()
                self.media_player = None
                
                k2_logger.info("Media player cleaned up", "LANDING")
            except Exception as e:
                k2_logger.error(f"Error during media cleanup: {str(e)}", "LANDING")
        
        # Stop audio output
        if hasattr(self, 'audio_output') and self.audio_output:
            try:
                self.audio_output.deleteLater()
                self.audio_output = None
            except:
                pass
        
        # Stop pulse timer
        if hasattr(self, 'pulse_timer') and self.pulse_timer:
            self.pulse_timer.stop()
            self.pulse_timer.deleteLater()
            self.pulse_timer = None
        
        # Stop fade animation
        if hasattr(self, 'fade_animation') and self.fade_animation:
            self.fade_animation.stop()
            self.fade_animation.deleteLater()
            self.fade_animation = None
        
        k2_logger.info("Landing page cleanup complete", "LANDING")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LandingPageWidget()
    window.show()
    sys.exit(app.exec())