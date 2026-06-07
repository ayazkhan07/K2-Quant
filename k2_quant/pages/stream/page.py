"""
K2 Quant Stream Page — Multi-window model viewer.

Layout:
  Fixed left pane (Saved Models, Strategies, Technical Indicators)
  QMdiArea filling the rest — each model opens in a tiled sub-window.

Behavioral rules:
  * One window per model (re-click raises existing).
  * Strategies / TI apply only to the focused window.
  * Left pane checkboxes sync to the focused window's state.
  * Auto-tile on every open / close.
  * Close requires confirmation.
  * Session state persists across restart.
"""

import json
from typing import Dict, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QMdiArea, QMdiSubWindow, QMessageBox, QPushButton, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint, QSize
from PyQt6.QtGui import QCursor

from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.data.saved_models_manager import saved_models_manager
from k2_quant.utilities.services.strategy_service import strategy_service
from k2_quant.utilities.services.strategy_runner import strategy_runner

from k2_quant.pages.analysis.components.left_pane import LeftPaneWidget
from k2_quant.pages.stream.components.stream_window import StreamWindowWidget


_EDGE = 6  # pixel margin for resize detection


class ResizableSubWindow(QMdiSubWindow):
    """Frameless QMdiSubWindow with edge/corner resize handles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._resize_edge = None
        self._resize_origin = None
        self._resize_geo = None
        self.setMouseTracking(True)
        self.setMinimumSize(200, 150)

    def _hit_edge(self, pos: QPoint) -> str:
        """Return edge/corner identifier based on local position."""
        r = self.rect()
        e = _EDGE
        left = pos.x() < e
        right = pos.x() > r.width() - e
        top = pos.y() < e
        bottom = pos.y() > r.height() - e
        if top and left:
            return 'tl'
        if top and right:
            return 'tr'
        if bottom and left:
            return 'bl'
        if bottom and right:
            return 'br'
        if left:
            return 'l'
        if right:
            return 'r'
        if top:
            return 't'
        if bottom:
            return 'b'
        return ''

    _CURSOR_MAP = {
        'l': Qt.CursorShape.SizeHorCursor,
        'r': Qt.CursorShape.SizeHorCursor,
        't': Qt.CursorShape.SizeVerCursor,
        'b': Qt.CursorShape.SizeVerCursor,
        'tl': Qt.CursorShape.SizeFDiagCursor,
        'br': Qt.CursorShape.SizeFDiagCursor,
        'tr': Qt.CursorShape.SizeBDiagCursor,
        'bl': Qt.CursorShape.SizeBDiagCursor,
    }

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            edge = self._hit_edge(event.position().toPoint())
            if edge:
                self._resize_edge = edge
                self._resize_origin = event.globalPosition().toPoint()
                self._resize_geo = self.geometry()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resize_edge and self._resize_origin:
            delta = event.globalPosition().toPoint() - self._resize_origin
            geo = QRect(self._resize_geo)
            mn = self.minimumSize()
            e = self._resize_edge

            if 'r' in e:
                geo.setRight(geo.right() + delta.x())
            if 'b' in e:
                geo.setBottom(geo.bottom() + delta.y())
            if 'l' in e:
                geo.setLeft(geo.left() + delta.x())
            if 't' in e:
                geo.setTop(geo.top() + delta.y())

            if geo.width() >= mn.width() and geo.height() >= mn.height():
                self.setGeometry(geo)
            event.accept()
            return

        edge = self._hit_edge(event.position().toPoint())
        cursor = self._CURSOR_MAP.get(edge)
        if cursor:
            self.setCursor(cursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._resize_edge:
            self._resize_edge = None
            self._resize_origin = None
            self._resize_geo = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class StreamTitleBar(QWidget):
    """Custom title bar for stream MDI sub-windows.

    Black background, right-aligned white model name,
    stream toggle button, red square close button, white square maximize/restore button.
    Supports click-and-drag to move the parent QMdiSubWindow.
    """

    close_requested = pyqtSignal()
    maximize_requested = pyqtSignal()
    stream_toggled = pyqtSignal(bool)  # True = start, False = stop

    def __init__(self, title: str, mdi_subwindow: QMdiSubWindow = None, parent=None):
        super().__init__(parent)
        self._mdi_sub = mdi_subwindow
        self._drag_pos = None
        self._streaming = False
        self.setFixedHeight(28)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        layout = QHBoxLayout()
        layout.setContentsMargins(4, 0, 0, 0)
        layout.setSpacing(4)
        self.setLayout(layout)

        # Stream toggle button
        self._stream_btn = QPushButton("▶ Stream")
        self._stream_btn.setFixedHeight(20)
        self._stream_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self._stream_btn.setStyleSheet(
            "QPushButton { background: #1a2a1a; color: #4a4; border: 1px solid #2a3a2a; "
            "border-radius: 3px; padding: 0 8px; font-size: 10px; font-weight: 600; }"
            "QPushButton:hover { background: #2a3a2a; color: #6c6; }"
        )
        self._stream_btn.setToolTip("Start live data streaming")
        self._stream_btn.clicked.connect(self._on_stream_clicked)
        layout.addWidget(self._stream_btn)

        # Stream status indicator
        self._stream_status = QLabel("")
        self._stream_status.setStyleSheet(
            "color: #666; font-size: 10px; background: transparent;"
        )
        layout.addWidget(self._stream_status)

        layout.addStretch()

        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(
            "color: #ffffff; font-size: 12px; font-weight: 600; "
            "background: transparent; padding-right: 8px;"
        )
        self._title_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # Never let the layout squeeze the name below the width its text needs.
        self._title_label.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        self._refresh_title_min_width()
        layout.addWidget(self._title_label)

        self._max_btn = QPushButton()
        self._max_btn.setFixedSize(18, 18)
        self._max_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self._max_btn.setStyleSheet(
            "QPushButton { background-color: #ffffff; border: none; }"
            "QPushButton:hover { background-color: #cccccc; }"
        )
        self._max_btn.setToolTip("Maximize / Restore")
        self._max_btn.clicked.connect(self.maximize_requested.emit)
        layout.addWidget(self._max_btn)

        self._close_btn = QPushButton()
        self._close_btn.setFixedSize(18, 18)
        self._close_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self._close_btn.setStyleSheet(
            "QPushButton { background-color: #ff4444; border: none; }"
            "QPushButton:hover { background-color: #cc0000; }"
        )
        self._close_btn.setToolTip("Close")
        self._close_btn.clicked.connect(self.close_requested.emit)
        layout.addWidget(self._close_btn)

        self.setStyleSheet("StreamTitleBar { background-color: #000000; }")

    def _on_stream_clicked(self):
        self._streaming = not self._streaming
        self.stream_toggled.emit(self._streaming)
        self._update_stream_appearance()

    def _update_stream_appearance(self):
        if self._streaming:
            self._stream_btn.setText("■ Stop")
            self._stream_btn.setStyleSheet(
                "QPushButton { background: #2a1a1a; color: #f44; border: 1px solid #3a2a2a; "
                "border-radius: 3px; padding: 0 8px; font-size: 10px; font-weight: 600; }"
                "QPushButton:hover { background: #3a2a2a; color: #f66; }"
            )
            self._stream_btn.setToolTip("Stop live data streaming")
        else:
            self._stream_btn.setText("▶ Stream")
            self._stream_btn.setStyleSheet(
                "QPushButton { background: #1a2a1a; color: #4a4; border: 1px solid #2a3a2a; "
                "border-radius: 3px; padding: 0 8px; font-size: 10px; font-weight: 600; }"
                "QPushButton:hover { background: #2a3a2a; color: #6c6; }"
            )
            self._stream_btn.setToolTip("Start live data streaming")

    def set_stream_status(self, text: str):
        self._stream_status.setText(text)

    def set_streaming(self, streaming: bool):
        self._streaming = streaming
        self._update_stream_appearance()

    def set_mdi_subwindow(self, sub: QMdiSubWindow):
        self._mdi_sub = sub

    def set_title(self, title: str):
        self._title_label.setText(title)
        self._refresh_title_min_width()

    def _refresh_title_min_width(self):
        """Reserve enough width for the full title text so it is never clipped."""
        fm = self._title_label.fontMetrics()
        # +16 covers the 8px right padding plus a small safety margin.
        width = fm.horizontalAdvance(self._title_label.text()) + 16
        self._title_label.setMinimumWidth(width)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._mdi_sub:
            self._drag_pos = event.globalPosition().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and self._mdi_sub:
            delta = event.globalPosition().toPoint() - self._drag_pos
            self._mdi_sub.move(self._mdi_sub.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.maximize_requested.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


class StreamPageWidget(QWidget):
    """Stream page — left pane + MDI area with floating model windows."""

    back_to_stock_fetcher = pyqtSignal()

    def __init__(self, tab_id: int = 0, parent=None):
        super().__init__(parent)
        self.tab_id = tab_id

        # {table_name: QMdiSubWindow}
        self._windows: Dict[str, QMdiSubWindow] = {}
        self._active_table: Optional[str] = None

        self._init_ui()
        self._setup_styling()
        self._load_left_pane_data()

        k2_logger.info(f"Stream page initialized (Tab ID: {tab_id})", "STREAM")

    # ── UI construction ───────────────────────────────────────────

    def _init_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setLayout(root)

        # Body: left pane + MDI area
        body = QSplitter(Qt.Orientation.Horizontal)
        body.setHandleWidth(1)
        body.setStyleSheet("QSplitter::handle { background-color: #1a1a1a; }")

        self.left_pane = LeftPaneWidget()
        body.addWidget(self.left_pane)

        self.mdi_area = QMdiArea()
        self.mdi_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.mdi_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.mdi_area.setBackground(Qt.GlobalColor.black)
        self.mdi_area.setStyleSheet("""
            QMdiArea { background: #0a0a0a; border: none; }
            QMdiSubWindow {
                background: #0f0f0f;
                border: 1px solid #2a2a2a;
            }
        """)
        self.mdi_area.subWindowActivated.connect(self._on_subwindow_activated)
        body.addWidget(self.mdi_area)

        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([280, 1100])

        root.addWidget(body)

        # Status bar
        self._status_widget = self._create_status_bar()
        root.addWidget(self._status_widget)

        # Wire left pane signals
        self.left_pane.model_selected.connect(self._on_model_selected)
        self.left_pane.strategy_toggled.connect(self._on_strategy_toggled)
        self.left_pane.strategy_deleted.connect(self._on_strategy_deleted)
        self.left_pane.indicator_toggled.connect(self._on_indicator_toggled)

        # Cross-page sync: pick up saves/deletes made on any other tab.
        # Without these, this tab's STRATEGIES / SAVED MODELS list stays
        # at the snapshot taken when the tab was first constructed.
        strategy_service.strategies_changed.connect(
            self.left_pane.refresh_strategies)
        saved_models_manager.models_changed.connect(
            self.left_pane.refresh_models)

    def _create_status_bar(self) -> QWidget:
        widget = QWidget()
        widget.setFixedHeight(32)
        widget.setObjectName("streamStatusBar")
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        widget.setLayout(layout)

        indicator = QLabel("●")
        indicator.setStyleSheet("color: #4a4; font-size: 8px;")
        layout.addWidget(indicator)

        self._model_label = QLabel("No windows open")
        self._model_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self._model_label)

        layout.addStretch()

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self._status_label)

        return widget

    # ── Left pane data ────────────────────────────────────────────

    def _load_left_pane_data(self):
        try:
            models = saved_models_manager.get_saved_models()
            self.left_pane.populate_models(models)
            try:
                strategies = strategy_service.get_all_strategies()
                self.left_pane.populate_strategies(strategies)
            except Exception:
                self.left_pane.populate_strategies([])
        except Exception as e:
            k2_logger.error(f"Stream left pane load failed: {e}", "STREAM")

    def refresh_models(self):
        self.left_pane.refresh_models()

    def load_saved_models(self):
        self.left_pane.refresh_models()

    # ── Model selection → window management ───────────────────────

    def _on_model_selected(self, table_name: str):
        if table_name in self._windows:
            sub = self._windows[table_name]
            self.mdi_area.setActiveSubWindow(sub)
            sub.showNormal()
            self._sync_left_pane(table_name)
            return

        self._open_window(table_name)

    def _open_window(self, table_name: str):
        content = StreamWindowWidget(table_name)
        content.apply_styling()

        display_name = self._display_name_for(table_name)

        # Wrap content in a container: custom title bar + stream window
        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container.setLayout(container_layout)

        sub = ResizableSubWindow()

        title_bar = StreamTitleBar(display_name, mdi_subwindow=sub)
        container_layout.addWidget(title_bar)
        container_layout.addWidget(content)

        sub.setWidget(container)
        sub.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        sub.setWindowTitle(display_name)

        # Hide the native title bar
        empty_bar = QWidget()
        empty_bar.setFixedHeight(0)
        sub.layout().insertWidget(0, empty_bar)
        sub.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        title_bar.close_requested.connect(lambda tn=table_name: self._request_close(tn))
        title_bar.maximize_requested.connect(lambda s=sub: self._toggle_maximize(s))
        title_bar.stream_toggled.connect(content.toggle_streaming)
        content.stream_status_changed.connect(title_bar.set_stream_status)
        content.stream_rejected.connect(lambda tb=title_bar: tb.set_streaming(False))

        # Store the actual content widget and title bar on the sub for easy retrieval
        sub._stream_content = content
        sub._title_bar = title_bar

        self.mdi_area.addSubWindow(sub)
        sub.show()

        self._windows[table_name] = sub

        sub.installEventFilter(self)

        self.mdi_area.tileSubWindows()

        self._active_table = table_name
        self._sync_left_pane(table_name)
        self._update_status()

        k2_logger.info(f"Stream window opened: {table_name}", "STREAM")

    def eventFilter(self, obj, event):
        """Intercept sub-window close to ask for confirmation."""
        if isinstance(obj, QMdiSubWindow) and event.type() == event.Type.Close:
            table_name = self._table_name_for_sub(obj)
            if table_name:
                reply = QMessageBox.question(
                    self, "Close Window",
                    f"Close the window for {table_name}?\n\n"
                    "All unsaved state for this window will be lost.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return True

                self._close_window(table_name, from_event=True)
                event.ignore()
                return True

        return super().eventFilter(obj, event)

    def _request_close(self, table_name: str):
        """Close button clicked on custom title bar — show confirmation."""
        reply = QMessageBox.question(
            self, "Close Window",
            f"Close the window for {table_name}?\n\n"
            "All unsaved state for this window will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._close_window(table_name)

    def _toggle_maximize(self, sub: QMdiSubWindow):
        """Maximize button clicked — toggle between maximized and tiled."""
        if sub.isMaximized():
            sub.showNormal()
            self.mdi_area.tileSubWindows()
        else:
            sub.showMaximized()

    def _close_window(self, table_name: str, from_event: bool = False):
        sub = self._windows.pop(table_name, None)
        if sub is None:
            return

        content: StreamWindowWidget = self._content_of(sub)
        if content:
            content.cleanup()

        sub.removeEventFilter(self)
        self.mdi_area.removeSubWindow(sub)
        sub.deleteLater()

        self.mdi_area.tileSubWindows()

        if self._active_table == table_name:
            active = self.mdi_area.activeSubWindow()
            self._active_table = self._table_name_for_sub(active) if active else None
            if self._active_table:
                self._sync_left_pane(self._active_table)
            else:
                self.left_pane.clear_all_indicators()
                self.left_pane.clear_all_strategies()

        self._update_status()
        k2_logger.info(f"Stream window closed: {table_name}", "STREAM")

    def _display_name_for(self, table_name: str) -> str:
        """Return the user-facing model name shown in the Saved Models list.

        Falls back to the raw table name if metadata lookup fails.
        """
        try:
            # Use the same list the Saved Models sidebar is built from so the
            # title matches it exactly (e.g. "GOOG-1D-20Y-06/05").
            for model in saved_models_manager.get_saved_models():
                if model.get("table_name") == table_name:
                    name = model.get("display_name")
                    if name:
                        return name
                    break
        except Exception as e:
            k2_logger.warning(
                f"Display name lookup failed for {table_name}: {e}", "STREAM")
        return table_name

    def _table_name_for_sub(self, sub: Optional[QMdiSubWindow]) -> Optional[str]:
        if sub is None:
            return None
        for tname, s in self._windows.items():
            if s is sub:
                return tname
        return None

    # ── Focus tracking & left pane sync ───────────────────────────

    def _on_subwindow_activated(self, sub: Optional[QMdiSubWindow]):
        table_name = self._table_name_for_sub(sub)
        if table_name and table_name != self._active_table:
            self._active_table = table_name
            self._sync_left_pane(table_name)
            self._update_status()

    def _sync_left_pane(self, table_name: str):
        """Update strategy/indicator checkboxes to reflect the focused window's state."""
        sub = self._windows.get(table_name)
        if sub is None:
            return
        content: StreamWindowWidget = self._content_of(sub)
        if content is None:
            return

        # Sync indicators
        self.left_pane.clear_all_indicators()
        active_inds = content.get_applied_indicators()
        for i in range(self.left_pane.indicator_layout.count()):
            widget = self.left_pane.indicator_layout.itemAt(i).widget()
            if widget is None:
                continue
            from PyQt6.QtWidgets import QCheckBox
            if isinstance(widget, QCheckBox) and widget.text() in active_inds:
                widget.blockSignals(True)
                widget.setChecked(True)
                widget.blockSignals(False)
                self.left_pane.active_indicators.add(widget.text())

        # Sync strategies
        self.left_pane.clear_all_strategies()
        active_strats = content.get_applied_strategies()
        for i in range(self.left_pane.strategy_layout.count()):
            widget = self.left_pane.strategy_layout.itemAt(i).widget()
            if widget is None:
                continue
            from PyQt6.QtWidgets import QCheckBox
            if isinstance(widget, QCheckBox) and widget.text() in active_strats:
                widget.blockSignals(True)
                widget.setChecked(True)
                widget.blockSignals(False)
                self.left_pane.active_strategies.add(widget.text())

    # ── Strategy / indicator routing to active window ─────────────

    @staticmethod
    def _content_of(sub: QMdiSubWindow) -> Optional['StreamWindowWidget']:
        """Get the StreamWindowWidget from a sub-window (may be wrapped in a container)."""
        return getattr(sub, '_stream_content', None) or sub.widget()

    def _get_active_content(self) -> Optional[StreamWindowWidget]:
        if self._active_table and self._active_table in self._windows:
            return self._content_of(self._windows[self._active_table])
        return None

    def _on_indicator_toggled(self, indicator_name: str, enabled: bool):
        content = self._get_active_content()
        if content is None:
            k2_logger.warning("No active stream window for indicator toggle", "STREAM")
            return

        if enabled:
            params = StreamWindowWidget.extract_default_indicator_params(indicator_name)
            content.apply_indicator(indicator_name, params)
        else:
            content.remove_indicator(indicator_name)

    def _on_strategy_toggled(self, strategy_name: str, enabled: bool):
        content = self._get_active_content()
        if content is None:
            k2_logger.warning("No active stream window for strategy toggle", "STREAM")
            return

        # Reentrancy guard: silently ignore a duplicate "enable" while a run
        # for this (table, strategy) pair is still in flight. This prevents
        # double-submits when users click the checkbox twice in rapid
        # succession or when session restore re-fires signals.
        key = strategy_runner.make_key(content.table_name, strategy_name)
        if enabled and strategy_runner.is_running(key):
            k2_logger.info(
                f"Strategy '{strategy_name}' already running on "
                f"{content.table_name}; ignoring duplicate toggle.",
                "STREAM",
            )
            return

        if enabled:
            content.apply_strategy(strategy_name)
        else:
            content.remove_strategy(strategy_name)

    def _on_strategy_deleted(self, name: str):
        name = (name or "").strip()
        if not name:
            return
        ok = strategy_service.delete_strategy(name)
        if not ok:
            k2_logger.error(f"Strategy delete failed for {name!r}", "STREAM")

        for table_name, sub in list(self._windows.items()):
            content: StreamWindowWidget = self._content_of(sub)
            if content:
                content.remove_strategy(name)
                content.applied_strategies.discard(name)
                content.outputs_panel.handle_strategy_deleted(name)

        self._load_left_pane_data()

    def _update_status(self):
        n = len(self._windows)
        if n == 0:
            self._model_label.setText("No windows open")
        elif self._active_table:
            self._model_label.setText(
                f"Active: {self._active_table} | {n} window{'s' if n != 1 else ''}")
        else:
            self._model_label.setText(f"{n} window{'s' if n != 1 else ''}")

    # ── Session persistence ───────────────────────────────────────

    def get_session_state(self) -> dict:
        """Return serializable state for QSettings persistence."""
        windows = []
        for table_name, sub in self._windows.items():
            geo = sub.geometry()
            windows.append({
                'table_name': table_name,
                'x': geo.x(), 'y': geo.y(),
                'w': geo.width(), 'h': geo.height(),
                'maximized': sub.isMaximized(),
            })
        return {
            'tab_id': self.tab_id,
            'windows': windows,
            'active': self._active_table or '',
        }

    def restore_session_state(self, state: dict):
        """Recreate windows from a saved session."""
        for win_info in state.get('windows', []):
            table_name = win_info.get('table_name')
            if not table_name:
                continue
            self._open_window(table_name)
            sub = self._windows.get(table_name)
            if sub and win_info.get('maximized'):
                sub.showMaximized()

        active = state.get('active', '')
        if active and active in self._windows:
            self.mdi_area.setActiveSubWindow(self._windows[active])
            self._active_table = active
            self._sync_left_pane(active)

    # ── Lifecycle ─────────────────────────────────────────────────

    def cleanup(self):
        """Persist all windows and release resources."""
        for table_name in list(self._windows.keys()):
            sub = self._windows.get(table_name)
            if sub:
                content: StreamWindowWidget = self._content_of(sub)
                if content:
                    content.cleanup()

        self._windows.clear()

        # Drop singleton -> deleted-widget references that would otherwise
        # fire `refresh_*` on a Python object whose underlying QWidget
        # has been destroyed (RuntimeError at next emit).
        for sig, slot in (
            (strategy_service.strategies_changed,
             self.left_pane.refresh_strategies),
            (saved_models_manager.models_changed,
             self.left_pane.refresh_models),
        ):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

        k2_logger.info(f"Stream page cleaned up (Tab ID: {self.tab_id})", "STREAM")

    def reset_after_database_cleared(self):
        for table_name in list(self._windows.keys()):
            sub = self._windows.pop(table_name)
            content: StreamWindowWidget = self._content_of(sub)
            if content:
                content.cleanup()
            sub.removeEventFilter(self)
            self.mdi_area.removeSubWindow(sub)
            sub.deleteLater()

        self._active_table = None
        self.left_pane.populate_models([])
        self.left_pane.clear_all_indicators()
        self.left_pane.clear_all_strategies()
        self._model_label.setText("No windows open")
        self._status_label.setText("Ready")
        k2_logger.info(f"Stream tab {self.tab_id} reset after DB clear", "STREAM")

    # ── Styling ───────────────────────────────────────────────────

    def _setup_styling(self):
        self.setStyleSheet("""
            #streamStatusBar {
                background-color: #0f0f0f;
                border-top: 1px solid #1a1a1a;
            }
            #streamStatusBar QLabel {
                color: #666;
                background: transparent;
            }
        """)
