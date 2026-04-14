"""
Right Pane Component - Conversational AI with Agent Loop

Scroll-based chat layout that supports inline interactive charts
(matplotlib FigureCanvasQTAgg) alongside text messages.

Save as: k2_quant/pages/analysis/components/right_pane.py
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import base64
import html as html_mod
import json
import re

from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QPushButton, QLabel, QWidget,
                             QSizePolicy, QComboBox, QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer, QEvent
from PyQt6.QtGui import QColor, QFontMetrics, QPixmap, QKeySequence, QShortcut

from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.services import table_controller
from k2_quant.utilities.services.table_controller import AI_MODELS, DEFAULT_MODEL
from k2_quant.utilities.text.math_formatter import MathFormatter
from k2_quant.utilities.text.math_renderer import latex_to_pixmap
from k2_quant.utilities.text.markdown_renderer import markdown_to_html
from k2_quant.pages.analysis.widgets.matrix_widget import MatrixWidget
from k2_quant.pages.analysis.widgets.data_view_widget import InlineDataView

# Thinkspace width: resizable via main splitter; max = 2x default.
THINKSPACE_DEFAULT_WIDTH = 998
THINKSPACE_MIN_WIDTH = 380
THINKSPACE_MAX_WIDTH = THINKSPACE_DEFAULT_WIDTH * 2


class CommandWorker(QThread):
    """Worker to execute the agent loop in background."""

    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    step_update = pyqtSignal(str)

    def __init__(
        self,
        table_name: str,
        command_text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        initial_workspace: Optional[Dict] = None,
        model_key: str = DEFAULT_MODEL,
        market_hours_only: bool = False,
    ):
        super().__init__()
        self.table_name = table_name
        self.command_text = command_text
        self.conversation_history = conversation_history or []
        self.initial_workspace = initial_workspace
        self.model_key = model_key
        self.market_hours_only = market_hours_only
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self):
        try:
            if table_controller is None:
                raise RuntimeError("table_controller service is not available")
            result = table_controller.execute_command(
                self.table_name,
                self.command_text,
                self.conversation_history,
                step_callback=self._on_step,
                initial_workspace=self.initial_workspace,
                cancel_check=self.is_cancelled,
                model_key=self.model_key,
                market_hours_only=self.market_hours_only,
            )
            if self._cancelled:
                self.result_ready.emit({
                    "success": True,
                    "display_message": "Request cancelled.",
                    "data_modified": False,
                    "cancelled": True,
                })
            else:
                self.result_ready.emit(result)
        except Exception as e:
            if self._cancelled:
                self.result_ready.emit({
                    "success": True,
                    "display_message": "Request cancelled.",
                    "data_modified": False,
                    "cancelled": True,
                })
            else:
                self.error_occurred.emit(str(e))

    def _on_step(self, text: str):
        self.step_update.emit(text)


class RightPaneWidget(QFrame):
    """Right pane with AI chat interface (scroll-based with interactive charts)."""

    # Signals
    message_sent = pyqtSignal(str)
    # Emitted after any strategy tool mutates the library (save/rename/etc.).
    strategy_generated = pyqtSignal(str, str)
    # Thinkspace delete_strategy already removed this name from the DB; analysis page syncs UI.
    strategy_removed_remotely = pyqtSignal(str)
    projection_requested = pyqtSignal(dict)
    data_modified = pyqtSignal()
    tab_writes_ready = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(THINKSPACE_MIN_WIDTH)
        self.setMaximumWidth(THINKSPACE_MAX_WIDTH)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        self.setObjectName("rightPane")

        self.current_context: Optional[Dict[str, Any]] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.worker: Optional[CommandWorker] = None
        self.streaming_timer = None
        self._typing_dot_timer: Optional[QTimer] = None
        self._typing_dot_state = 0
        self.streaming_text = ""
        self.streaming_index = 0
        self._streaming_prefix = "AI: "
        self._streaming_label: Optional[QLabel] = None
        self.math_formatter = MathFormatter(use_block_markers=True)
        self.workspace_provider: Optional[callable] = None
        self.save_chat_callback: Optional[callable] = None
        self.load_chat_callback: Optional[callable] = None
        self._pending_charts: list = []
        self._message_records: list = []

        self._chat_store: Dict[str, Dict[str, Any]] = {}
        self._active_table: Optional[str] = None

        self.init_ui()
        self.setup_styling()

    # ── UI setup ─────────────────────────────────────────────────

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        self.setLayout(layout)

        # Header row
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.addStretch()

        self.model_combo = QComboBox()
        self.model_combo.setObjectName("modelCombo")
        self.model_combo.setFixedHeight(24)
        for name in AI_MODELS:
            self.model_combo.addItem(name)
        self.model_combo.setCurrentText(DEFAULT_MODEL)
        header_row.addWidget(self.model_combo)

        self.copy_all_btn = QPushButton("Copy All")
        self.copy_all_btn.setObjectName("clearChatBtn")
        self.copy_all_btn.setFixedHeight(24)
        self.copy_all_btn.clicked.connect(self._copy_all_chat)
        header_row.addWidget(self.copy_all_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clearChatBtn")
        self.clear_btn.setFixedHeight(24)
        self.clear_btn.clicked.connect(self.clear_chat)
        header_row.addWidget(self.clear_btn)
        layout.addLayout(header_row)

        # Chat display — QScrollArea with vertical widget list
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("chatScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("chatScrollContent")
        self.chat_layout = QVBoxLayout(self.scroll_content)
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        self.chat_layout.setSpacing(2)

        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

        QShortcut(QKeySequence("Ctrl+Shift+C"), self, self._copy_all_chat)

        # Typing indicator
        self.typing_indicator = QLabel("...")
        self.typing_indicator.setObjectName("typingIndicator")
        self.typing_indicator.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.typing_indicator.setFixedHeight(18)
        self.typing_indicator.hide()
        layout.addWidget(self.typing_indicator)

        # Input area
        input_widget = QWidget()
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_widget.setLayout(input_layout)

        self.ai_input = QTextEdit()
        self.ai_input.setPlaceholderText("Type your message...")
        self.ai_input.setObjectName("chatInput")
        self.ai_input.setAcceptRichText(False)
        self.ai_input.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.ai_input.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ai_input.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ai_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.ai_input.document().setDocumentMargin(8)
        self._input_line_height = QFontMetrics(
            self.ai_input.font()).lineSpacing()
        self._input_max_lines = 5
        self._input_padding = 46
        self.ai_input.setFixedHeight(
            self._input_line_height + self._input_padding)
        self.ai_input.textChanged.connect(self._adjust_input_height)
        self.ai_input.installEventFilter(self)
        input_layout.addWidget(self.ai_input)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_ai_message)
        self.send_btn.setObjectName("sendBtn")
        input_layout.addWidget(self.send_btn)

        layout.addWidget(input_widget)

    # ── chat widget helpers ──────────────────────────────────────

    def _is_near_bottom(self) -> bool:
        sb = self.scroll_area.verticalScrollBar()
        return sb.value() >= sb.maximum() - 80

    def _scroll_to_bottom(self, force: bool = False):
        def _do():
            sb = self.scroll_area.verticalScrollBar()
            if force or sb.value() >= sb.maximum() - 80:
                sb.setValue(sb.maximum())
        QTimer.singleShot(10, _do)

    def _add_widget(self, widget):
        self.chat_layout.addWidget(widget)
        self._scroll_to_bottom()

    def _start_typing_indicator(self):
        self._typing_dot_state = 0
        self.typing_indicator.setText(".")
        self.typing_indicator.show()
        self._typing_dot_timer = QTimer()
        self._typing_dot_timer.timeout.connect(self._update_typing_dots)
        self._typing_dot_timer.start(400)

    def _stop_typing_indicator(self):
        if self._typing_dot_timer:
            self._typing_dot_timer.stop()
            self._typing_dot_timer = None
        self.typing_indicator.hide()

    def _update_typing_dots(self):
        dots = (".", "..", "...")
        self._typing_dot_state = (self._typing_dot_state + 1) % 3
        self.typing_indicator.setText(dots[self._typing_dot_state])

    def _make_user_bubble(self, text: str) -> QLabel:
        label = QLabel()
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setStyleSheet("background: transparent; padding: 0; margin: 0;")
        label.setContentsMargins(0, 14, 0, 0)
        safe = html_mod.escape(text).replace('\n', '<br>')
        label.setText(
            f'<span style="color:#666666;">YOU: </span>'
            f'<span style="color:#ffffff;">{safe}</span>')
        return label

    def _make_ai_bubble(self, html_content: str = "") -> QLabel:
        label = QLabel()
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setStyleSheet("background: transparent; padding: 0; margin: 0;")
        label.setContentsMargins(0, 4, 0, 14)
        if html_content:
            label.setText(html_content)
        return label

    def _make_step_label(self, text: str) -> QLabel:
        label = QLabel(f"  {text}")
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        label.setStyleSheet(
            "color: #666666; background: transparent;"
            "font-size: 12px; padding: 0; margin: 0;")
        label.setContentsMargins(0, 1, 0, 1)
        return label

    def _build_ai_html(self, text: str, prefix: str = "AI: ") -> str:
        """Build complete HTML for an AI message bubble."""
        prefix_html = (
            f'<span style="color:#666666;">'
            f'{html_mod.escape(prefix)}</span>')

        if self._has_markdown_table(text):
            lines = text.split('\n')
            parts = [prefix_html]
            text_buffer: list = []
            i = 0

            def _flush_text():
                if text_buffer:
                    chunk = '\n'.join(text_buffer)
                    text_buffer.clear()
                    parts.append(markdown_to_html(chunk))

            while i < len(lines):
                stripped = lines[i].strip()

                if '|' in stripped and not self._is_separator_line(stripped):
                    table_lines = [stripped]
                    j = i + 1
                    found_sep = False
                    while j < len(lines):
                        s = lines[j].strip()
                        if not s:
                            j += 1
                            continue
                        if self._is_separator_line(s):
                            found_sep = True
                            j += 1
                            continue
                        if '|' in s:
                            table_lines.append(s)
                            j += 1
                            continue
                        break
                    if found_sep and len(table_lines) >= 2:
                        _flush_text()
                        rows = []
                        for tl in table_lines:
                            cells = [c.strip() for c in tl.split('|')]
                            if cells and cells[0] == '':
                                cells = cells[1:]
                            if cells and cells[-1] == '':
                                cells = cells[:-1]
                            if cells:
                                rows.append(cells)
                        if rows:
                            parts.append(self._build_table_html(rows))
                        i = j
                        continue

                text_buffer.append(lines[i])
                i += 1

            _flush_text()
            return ''.join(parts)

        formatted = self.math_formatter.format_full(text)
        return prefix_html + markdown_to_html(formatted)

    # ── code block helpers ────────────────────────────────────────

    @staticmethod
    def _has_code_blocks(text: str) -> bool:
        return text.count('```') >= 2

    @staticmethod
    def _parse_code_blocks(text: str) -> list:
        """Split *text* into ``('text', content)`` and
        ``('code', content, lang)`` segments."""
        segments: list = []
        parts = re.split(r'```', text)
        for i, part in enumerate(parts):
            if i % 2 == 0:
                if part.strip():
                    segments.append(('text', part.strip()))
            else:
                lines = part.split('\n', 1)
                lang_hint = lines[0].strip()
                code = lines[1].rstrip() if len(lines) > 1 else ''
                segments.append(('code', code, lang_hint))
        return segments

    def _make_code_block_widget(self, code: str, lang: str = '') -> QFrame:
        container = QFrame()
        container.setObjectName("codeBlock")
        container.setStyleSheet(
            "#codeBlock {"
            "  background-color: #0d0d0d;"
            "  border-left: 3px solid #2a6496;"
            "  border-radius: 4px;"
            "}")
        v = QVBoxLayout(container)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(2)

        if lang:
            lang_label = QLabel(lang.upper())
            lang_label.setStyleSheet(
                "color: #555; font-size: 10px; background: transparent;"
                " letter-spacing: 0.5px; padding: 0; margin: 0;")
            v.addWidget(lang_label)

        code_label = QLabel()
        code_label.setWordWrap(True)
        code_label.setTextFormat(Qt.TextFormat.RichText)
        code_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        safe = (html_mod.escape(code)
                .replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
                .replace(' ', '&nbsp;')
                .replace('\n', '<br>'))
        code_label.setText(
            f'<span style="font-family: Consolas, monospace;'
            f' font-size: 12px; color: #a8c7e8;">{safe}</span>')
        code_label.setStyleSheet(
            "background: transparent; padding: 0; margin: 0;")
        v.addWidget(code_label)
        return container

    def _render_code_block_response(
            self, text: str, prefix: str = "AI: "):
        segments = self._parse_code_blocks(text)
        is_first_text = True
        for seg in segments:
            if seg[0] == 'text':
                p = prefix if is_first_text else ""
                is_first_text = False
                formatted = self.math_formatter.format_full(seg[1])
                rendered = markdown_to_html(formatted)
                prefix_html = (
                    f'<span style="color:#666666;">'
                    f'{html_mod.escape(p)}</span>') if p else ''
                label = self._make_ai_bubble(prefix_html + rendered)
                self._add_widget(label)
            elif seg[0] == 'code':
                is_first_text = False
                widget = self._make_code_block_widget(
                    seg[1], seg[2] if len(seg) > 2 else '')
                self._add_widget(widget)

        if self._pending_charts:
            self._insert_charts(self._pending_charts)
            self._pending_charts = []

    # ── send message ─────────────────────────────────────────────

    def send_ai_message(self):
        message = self.ai_input.toPlainText().strip()
        if not message:
            return

        self._add_widget(self._make_user_bubble(message))
        self._scroll_to_bottom(force=True)
        self._message_records.append({'type': 'user', 'content': message})

        self.ai_input.clear()
        self.ai_input.setFixedHeight(
            self._input_line_height + self._input_padding)
        self.ai_input.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.conversation_history.append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })

        table_name = None
        if self.current_context:
            table_name = self.current_context.get('table_name')

        if not table_name:
            self.stream_response(
                "Please load a model first so I know which table to operate on.")
            return

        if self.worker and self.worker.isRunning():
            self.stream_response(
                "Previous command is still executing. Please wait.")
            return

        self._start_typing_indicator()

        history_for_agent = [
            {"role": h["role"], "content": h["content"]}
            for h in self.conversation_history
            if h.get("role") in ("user", "assistant") and h.get("content")
        ][:-1]

        initial_workspace = None
        if self.workspace_provider:
            try:
                initial_workspace = self.workspace_provider()
            except Exception:
                pass

        model_key = self.model_combo.currentText()
        mkt_hours = self.current_context.get('market_hours_only', False) if self.current_context else False
        self.worker = CommandWorker(
            table_name, message, history_for_agent, initial_workspace,
            model_key, market_hours_only=mkt_hours)
        self.worker.result_ready.connect(self._on_worker_result)
        self.worker.error_occurred.connect(self._on_worker_error)
        self.worker.step_update.connect(self._on_step_update)
        self.worker.start()

        self._set_cancel_mode(True)
        self.message_sent.emit(message)

    def _cancel_request(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self._stop_typing_indicator()
            k2_logger.info("AI request cancelled by user", "AI_CHAT")

    def _set_cancel_mode(self, active: bool):
        try:
            self.send_btn.clicked.disconnect()
        except TypeError:
            pass
        if active:
            self.send_btn.setText("Cancel")
            self.send_btn.setObjectName("cancelBtn")
            self.send_btn.clicked.connect(self._cancel_request)
        else:
            self.send_btn.setText("Send")
            self.send_btn.setObjectName("sendBtn")
            self.send_btn.clicked.connect(self.send_ai_message)
        self.send_btn.style().unpolish(self.send_btn)
        self.send_btn.style().polish(self.send_btn)

    # ── intermediate step display ────────────────────────────────

    def _on_step_update(self, text: str):
        self._add_widget(self._make_step_label(text))
        self._message_records.append({'type': 'step', 'content': text})

    # ── markdown table helpers ───────────────────────────────────

    @staticmethod
    def _is_separator_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped or '|' not in stripped:
            return False
        return bool(re.match(r'^[\|\s\-:]+$', stripped) and '--' in stripped)

    def _has_markdown_table(self, text: str) -> bool:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        has_sep = any(self._is_separator_line(l) for l in lines)
        pipe_data = sum(
            1 for l in lines
            if '|' in l and not self._is_separator_line(l))
        return has_sep and pipe_data >= 2

    def _build_table_html(self, rows: list) -> str:
        if not rows:
            return ""

        num_cols = max(len(r) for r in rows)
        for r in rows:
            while len(r) < num_cols:
                r.append('')

        hdr = (
            "border-collapse:collapse; margin:8px 0; width:100%;"
            "table-layout:fixed; word-wrap:break-word;"
            "font-family:'Consolas','Courier New',monospace; font-size:12px;"
        )
        th = (
            "padding:6px 12px; border:1px solid #333; background:#1e1e1e;"
            "color:#e0e0e0; text-align:left; font-weight:600;"
            "overflow:hidden; word-wrap:break-word;"
        )
        td = (
            "padding:5px 12px; border:1px solid #2a2a2a;"
            "color:#ccc; overflow:hidden; word-wrap:break-word;"
        )
        td_alt = td + "background:#131313;"

        table_html = f"<table style='{hdr}'><thead><tr>"
        for cell in rows[0]:
            safe = html_mod.escape(cell)
            table_html += f"<th style='{th}'>{safe}</th>"
        table_html += "</tr></thead><tbody>"

        for idx, row in enumerate(rows[1:]):
            row_style = td_alt if idx % 2 else td
            table_html += "<tr>"
            for cell in row:
                safe = html_mod.escape(cell)
                table_html += f"<td style='{row_style}'>{safe}</td>"
            table_html += "</tr>"

        table_html += "</tbody></table>"
        return table_html

    # ── response rendering ───────────────────────────────────────

    def stream_response(self, text: str, prefix: str = "AI: "):
        self._message_records.append(
            {'type': 'ai', 'content': text, 'prefix': prefix})

        if self._has_markdown_table(text):
            self._render_formatted_response(text, prefix)
            return

        if self._has_code_blocks(text):
            self._render_code_block_response(text, prefix)
            return

        self.streaming_text = self.math_formatter.format_full(text)
        self.streaming_index = 0
        self._streaming_prefix = prefix

        label = self._make_ai_bubble()
        prefix_html = (
            f'<span style="color:#666666;">'
            f'{html_mod.escape(prefix)}</span>')
        label.setText(prefix_html)
        self._streaming_label = label
        self._add_widget(label)

        if self.streaming_timer:
            self.streaming_timer.stop()

        self.streaming_timer = QTimer()
        self.streaming_timer.timeout.connect(self._stream_next_chunk)
        self.streaming_timer.start(20)

    def _render_formatted_response(self, text: str, prefix: str = "AI: "):
        label = self._make_ai_bubble(self._build_ai_html(text, prefix))
        self._add_widget(label)

        if self._pending_charts:
            self._insert_charts(self._pending_charts)
            self._pending_charts = []

    def _stream_next_chunk(self):
        if self.streaming_index < len(self.streaming_text):
            chunk_size = min(
                2, len(self.streaming_text) - self.streaming_index)
            self.streaming_index += chunk_size

            text_so_far = self.streaming_text[:self.streaming_index]
            rendered = markdown_to_html(text_so_far)
            prefix_html = (
                f'<span style="color:#666666;">'
                f'{html_mod.escape(self._streaming_prefix)}</span>')
            self._streaming_label.setText(prefix_html + rendered)
            self._scroll_to_bottom()
        else:
            self.streaming_timer.stop()
            self.streaming_timer = None
            self._streaming_label = None
            if self._pending_charts:
                self._insert_charts(self._pending_charts)
                self._pending_charts = []

    # ── chart insertion ──────────────────────────────────────────

    def _insert_charts(self, charts: list):
        for chart_data in charts:
            if isinstance(chart_data, dict):
                fig = chart_data.get('figure')
                png = chart_data.get('png', '')
            else:
                fig = None
                png = chart_data

            inserted = False
            if fig is not None:
                try:
                    widget = self._create_interactive_chart(fig)
                    self._add_widget(widget)
                    inserted = True
                except Exception as e:
                    k2_logger.error(
                        f"Interactive chart failed, falling back to PNG: {e}",
                        "AI_CHAT")

            if not inserted and png:
                self._add_static_chart(png)

            self._message_records.append(
                {'type': 'chart', 'png': png if isinstance(png, str) else ''})

    def _create_interactive_chart(self, figure):
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT)

        container = QFrame()
        container.setObjectName("chartContainer")
        container.setStyleSheet("""
            #chartContainer {
                background-color: #1a1a1a;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
            }
        """)
        clayout = QVBoxLayout(container)
        clayout.setContentsMargins(4, 4, 4, 4)
        clayout.setSpacing(2)

        canvas = FigureCanvasQTAgg(figure)
        canvas.setMinimumHeight(350)
        canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        toolbar = NavigationToolbar2QT(canvas, container)
        toolbar.setStyleSheet(
            "background: #1a1a1a; border: none; color: #ccc;")

        clayout.addWidget(toolbar)
        clayout.addWidget(canvas)
        container.setMinimumHeight(400)
        return container

    def _add_static_chart(self, png_b64: str):
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        label.setStyleSheet("background: transparent;")
        label.setContentsMargins(0, 6, 0, 6)

        img_data = base64.b64decode(png_b64)
        pixmap = QPixmap()
        pixmap.loadFromData(img_data)

        max_w = self.scroll_area.viewport().width() - 30
        if pixmap.width() > max_w > 0:
            pixmap = pixmap.scaledToWidth(
                max_w, Qt.TransformationMode.SmoothTransformation)

        label.setPixmap(pixmap)
        self._add_widget(label)

    # ── interactive widget insertion ──────────────────────────────

    def _insert_matrices(self, items: list):
        for item in items:
            data = item.get('data', [])
            if not data:
                continue
            widget = MatrixWidget(
                data=data,
                label=item.get('label', ''),
                editable=item.get('editable', True),
                headers=item.get('headers'),
            )
            widget.matrix_edited.connect(self._on_matrix_edited)
            widget.send_to_workspace.connect(self._on_widget_to_workspace)
            self._add_widget(widget)
            self._message_records.append({
                'type': 'matrix',
                'data': data,
                'label': item.get('label', ''),
                'headers': item.get('headers'),
            })

    def _insert_tables(self, items: list):
        for item in items:
            rows = item.get('rows', [])
            headers = item.get('headers', [])
            if not rows:
                continue
            widget = InlineDataView(
                rows=rows,
                headers=headers,
                label=item.get('label', ''),
                editable=item.get('editable', False),
            )
            widget.cell_edited.connect(self._on_table_cell_edited)
            widget.send_to_workspace.connect(self._on_table_to_workspace)
            self._add_widget(widget)
            self._message_records.append({
                'type': 'table',
                'rows': rows,
                'headers': headers,
                'label': item.get('label', ''),
            })

    def _insert_equations(self, items: list):
        for item in items:
            latex = item.get('latex', '')
            if not latex:
                continue
            label_text = item.get('label', '')
            pixmap = latex_to_pixmap(latex)
            if pixmap is not None:
                lbl = QLabel()
                lbl.setPixmap(pixmap)
                lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
                lbl.setStyleSheet("background: transparent; padding: 4px 0;")
                if label_text:
                    container = QFrame()
                    container.setStyleSheet("background: transparent;")
                    v = QVBoxLayout(container)
                    v.setContentsMargins(0, 0, 0, 0)
                    v.setSpacing(2)
                    cap = QLabel(f'<span style="color:#666; font-size:10px;">'
                                 f'{html_mod.escape(label_text)}</span>')
                    cap.setTextFormat(Qt.TextFormat.RichText)
                    cap.setStyleSheet("background: transparent;")
                    v.addWidget(cap)
                    v.addWidget(lbl)
                    self._add_widget(container)
                else:
                    self._add_widget(lbl)
            else:
                formatted = self.math_formatter.format_full(latex)
                safe = html_mod.escape(formatted)
                fallback = self._make_ai_bubble(
                    f'<span style="color:#ffffff;">{safe}</span>')
                self._add_widget(fallback)
            self._message_records.append({
                'type': 'equation',
                'latex': latex,
                'label': label_text,
            })

    # ── edit feedback handlers ────────────────────────────────────

    def _on_matrix_edited(self, data: list):
        """User edited a matrix cell; inject context into conversation."""
        note = f"[User edited the matrix to: {self._summarize_matrix(data)}]"
        self.conversation_history.append({
            'role': 'user',
            'content': note,
            'timestamp': datetime.now().isoformat(),
        })

    @staticmethod
    def _summarize_matrix(data: list) -> str:
        rows = len(data)
        cols = len(data[0]) if data else 0
        if rows <= 4 and cols <= 6:
            row_strs = [", ".join(str(v) for v in r) for r in data]
            return "[" + "; ".join(row_strs) + "]"
        return f"{rows}×{cols} matrix (too large to display inline)"

    def _on_table_cell_edited(self, row: int, col: int, value: str):
        note = f"[User edited table cell ({row},{col}) to: {value}]"
        self.conversation_history.append({
            'role': 'user',
            'content': note,
            'timestamp': datetime.now().isoformat(),
        })

    def _on_widget_to_workspace(self, data: list, label: str):
        """Matrix → workspace: emit as tab_writes columns."""
        if not data or not data[0]:
            return
        n_cols = len(data[0])
        writes = []
        for c in range(n_cols):
            col_name = label or f"Matrix_col{c}"
            if n_cols > 1:
                col_name = f"{col_name}_{c}"
            values = [row[c] if c < len(row) else None for row in data]
            writes.append({
                "type": "working",
                "column_name": col_name,
                "values": values,
                "scope": "model",
                "sheet": "Sheet 1",
            })
        self.tab_writes_ready.emit(writes)

    def _on_table_to_workspace(self, rows: list, headers: list):
        """Table → workspace: emit each column as a tab_write."""
        if not rows or not headers:
            return
        writes = []
        for c, hdr in enumerate(headers):
            values = [row[c] if c < len(row) else None for row in rows]
            writes.append({
                "type": "working",
                "column_name": hdr,
                "values": values,
                "scope": "model",
                "sheet": "Sheet 1",
            })
        self.tab_writes_ready.emit(writes)

    # ── worker result handling ───────────────────────────────────

    def _on_worker_result(self, result: Dict[str, Any]):
        self._stop_typing_indicator()
        self._set_cancel_mode(False)
        self._pending_charts = result.get('_charts', [])
        pending_matrices = result.get('_matrices', [])
        pending_tables = result.get('_tables', [])
        pending_equations = result.get('_equations', [])

        if result.get('cancelled'):
            self._pending_charts = []
            self.stream_response("Request cancelled.", prefix="")
            self.conversation_history.append({
                'role': 'assistant',
                'content': "Request cancelled.",
                'timestamp': datetime.now().isoformat()
            })
        elif result.get('success'):
            response = result.get(
                'display_message', 'Operation completed successfully.')
            self.stream_response(response)

            self.conversation_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })

            if result.get('data_modified'):
                self.data_modified.emit()
            if result.get('_tab_writes'):
                self.tab_writes_ready.emit(result['_tab_writes'])

            # ── Embed interactive widgets ────────────────────────
            self._insert_matrices(pending_matrices)
            self._insert_tables(pending_tables)
            self._insert_equations(pending_equations)

            strategies_changed = False
            for strat in result.get('_strategies_saved', []):
                s_name = strat.get('name', '')
                if strat.get('deleted') and s_name:
                    self.strategy_removed_remotely.emit(s_name)
                s_code = strat.get('code', '')
                strategies_changed = True
                if s_name and s_code:
                    header = QLabel(
                        f'<span style="color:#4a9eff; font-size:12px;">'
                        f'Strategy saved: {s_name}</span>')
                    header.setTextFormat(Qt.TextFormat.RichText)
                    header.setStyleSheet("background:transparent; padding:4px 0 0 0;")
                    self._add_widget(header)
                    self._add_widget(
                        self._make_code_block_widget(s_code, 'python'))
            if strategies_changed:
                self.strategy_generated.emit(
                    result['_strategies_saved'][-1].get('name', ''), '')
        else:
            self._pending_charts = []
            error = result.get('error', 'Operation failed')
            self.stream_response(
                f"Unable to complete that request. {error}")

            self.conversation_history.append({
                'role': 'assistant',
                'content': f"Error: {error}",
                'timestamp': datetime.now().isoformat()
            })

        self.worker = None

    def _on_worker_error(self, error_msg: str):
        self._stop_typing_indicator()
        self._set_cancel_mode(False)
        self._pending_charts = []
        self.stream_response(
            f"Unable to process that request. {error_msg}")
        self.worker = None

    # ── context and state management ─────────────────────────────

    def set_data_context(self, context: Dict[str, Any]):
        """Save current chat, load or create chat for the new model."""
        new_table = context.get('table_name') if context else None

        if self._active_table:
            chat_state = {
                'records': list(self._message_records),
                'history': list(self.conversation_history),
            }
            self._chat_store[self._active_table] = chat_state
            if self.save_chat_callback:
                try:
                    self.save_chat_callback(
                        self._active_table,
                        json.dumps(self._message_records),
                        chat_state['history'])
                except Exception:
                    pass

        if new_table and new_table in self._chat_store:
            saved = self._chat_store[new_table]
            self._rebuild_chat(saved.get('records', []))
            self.conversation_history = list(saved.get('history', []))
        elif new_table and self.load_chat_callback:
            try:
                saved = self.load_chat_callback(new_table)
                if saved:
                    records = None
                    raw = saved.get('html', '')
                    if raw:
                        try:
                            records = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            pass

                    if isinstance(records, list):
                        self._rebuild_chat(records)
                    else:
                        self._rebuild_from_history(
                            saved.get('history', []))

                    self.conversation_history = list(
                        saved.get('history', []))
                    self._chat_store[new_table] = {
                        'records': list(self._message_records),
                        'history': self.conversation_history,
                    }
                else:
                    self._clear_chat_widgets()
                    self.conversation_history.clear()
            except Exception:
                self._clear_chat_widgets()
                self.conversation_history.clear()
        else:
            self._clear_chat_widgets()
            self.conversation_history.clear()

        self._active_table = new_table
        self.current_context = context

    def _rebuild_chat(self, records: list):
        """Reconstruct chat widgets from saved message records."""
        self._clear_chat_widgets()
        self._message_records = []
        for rec in records:
            rtype = rec.get('type', '')
            content = rec.get('content', '')
            prefix = rec.get('prefix', 'AI: ')

            if rtype == 'user':
                self.chat_layout.addWidget(self._make_user_bubble(content))
                self._message_records.append(rec)
            elif rtype == 'ai':
                html = self._build_ai_html(content, prefix)
                self.chat_layout.addWidget(self._make_ai_bubble(html))
                self._message_records.append(rec)
            elif rtype == 'step':
                self.chat_layout.addWidget(self._make_step_label(content))
                self._message_records.append(rec)
            elif rtype == 'chart':
                png = rec.get('png', '')
                if png:
                    self._add_static_chart(png)
                self._message_records.append(rec)
            elif rtype == 'matrix':
                data = rec.get('data', [])
                if data:
                    widget = MatrixWidget(
                        data=data,
                        label=rec.get('label', ''),
                        editable=True,
                        headers=rec.get('headers'),
                    )
                    widget.matrix_edited.connect(self._on_matrix_edited)
                    widget.send_to_workspace.connect(
                        self._on_widget_to_workspace)
                    self.chat_layout.addWidget(widget)
                self._message_records.append(rec)
            elif rtype == 'table':
                rows = rec.get('rows', [])
                headers = rec.get('headers', [])
                if rows:
                    widget = InlineDataView(
                        rows=rows, headers=headers,
                        label=rec.get('label', ''),
                    )
                    widget.cell_edited.connect(self._on_table_cell_edited)
                    widget.send_to_workspace.connect(
                        self._on_table_to_workspace)
                    self.chat_layout.addWidget(widget)
                self._message_records.append(rec)
            elif rtype == 'equation':
                latex = rec.get('latex', '')
                if latex:
                    self._insert_equations([{
                        'latex': latex,
                        'label': rec.get('label', ''),
                    }])
                else:
                    self._message_records.append(rec)
        self._scroll_to_bottom()

    def _rebuild_from_history(self, history: list):
        """Fallback: rebuild chat from conversation history (old format)."""
        self._clear_chat_widgets()
        self._message_records = []
        for msg in history:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if not content:
                continue
            if role == 'user':
                self.chat_layout.addWidget(self._make_user_bubble(content))
                self._message_records.append(
                    {'type': 'user', 'content': content})
            elif role == 'assistant':
                html = self._build_ai_html(content, 'AI: ')
                self.chat_layout.addWidget(self._make_ai_bubble(html))
                self._message_records.append(
                    {'type': 'ai', 'content': content, 'prefix': 'AI: '})
        self._scroll_to_bottom()

    def _clear_chat_widgets(self):
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._message_records = []

    def _copy_all_chat(self):
        """Walk _message_records and copy the full chat transcript to clipboard."""
        from PyQt6.QtWidgets import QApplication
        lines: list = []
        for rec in self._message_records:
            rtype = rec.get('type', '')
            content = rec.get('content', '')
            if rtype == 'user':
                lines.append(f"YOU: {content}")
            elif rtype == 'ai':
                prefix = rec.get('prefix', 'AI: ')
                lines.append(f"{prefix}{content}")
            elif rtype == 'step':
                lines.append(f"  [{content}]")
            elif rtype == 'matrix':
                label = rec.get('label', 'Matrix')
                data = rec.get('data', [])
                headers = rec.get('headers')
                parts = [f"[{label}]" if label else "[Matrix]"]
                if headers:
                    parts.append("\t".join(str(h) for h in headers))
                for row in data:
                    parts.append("\t".join(
                        "" if v is None else f"{v:.6g}" if isinstance(v, float) else str(v)
                        for v in row))
                lines.append("\n".join(parts))
            elif rtype == 'table':
                label = rec.get('label', 'Table')
                headers = rec.get('headers', [])
                rows = rec.get('rows', [])
                parts = [f"[{label}]" if label else "[Table]"]
                if headers:
                    parts.append("\t".join(str(h) for h in headers))
                for row in rows:
                    parts.append("\t".join(
                        "" if v is None else f"{v:.6g}" if isinstance(v, float) else str(v)
                        for v in row))
                lines.append("\n".join(parts))
            elif rtype == 'equation':
                latex = rec.get('latex', '')
                label = rec.get('label', '')
                lines.append(f"[Equation{': ' + label if label else ''}] {latex}")
            elif rtype == 'chart':
                lines.append("[Chart]")

        text = "\n\n".join(lines)
        QApplication.clipboard().setText(text)
        k2_logger.info(f"Chat copied to clipboard ({len(lines)} items)", "AI_CHAT")

    def clear_chat(self):
        self._clear_chat_widgets()
        self.conversation_history.clear()
        if self._active_table and self._active_table in self._chat_store:
            del self._chat_store[self._active_table]
        if self._active_table and self.save_chat_callback:
            try:
                self.save_chat_callback(self._active_table, '', [])
            except Exception:
                pass
        k2_logger.info("Chat cleared", "AI_CHAT")

    # ── auto-growing input ───────────────────────────────────────

    def eventFilter(self, obj, event):
        if obj is self.ai_input and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    return False
                self.send_ai_message()
                return True
        return super().eventFilter(obj, event)

    def _adjust_input_height(self):
        doc = self.ai_input.document()
        doc.setTextWidth(self.ai_input.viewport().width())
        content_h = int(doc.size().height())
        pad = self._input_padding
        one_line = self._input_line_height + pad
        max_h = self._input_line_height * self._input_max_lines + pad
        new_h = max(one_line, min(content_h + pad, max_h))
        self.ai_input.setFixedHeight(new_h)

        if content_h + pad > max_h:
            self.ai_input.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        else:
            self.ai_input.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    # ── styling ──────────────────────────────────────────────────

    def setup_styling(self):
        self.setStyleSheet("""
            #rightPane {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #111111, stop:1 #0a0a0a);
                border-left: 1px solid #1a1a1a;
            }

            #sectionTitle {
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #999;
                font-weight: 600;
                font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
                background-color: #1a1a1a;
                padding: 5px 10px;
                border-radius: 3px;
            }

            #chatScrollArea {
                background-color: #0a0a0a;
                border: 1px solid #1a1a1a;
                border-radius: 4px;
            }

            #chatScrollContent {
                background-color: #0a0a0a;
            }

            #chatScrollContent QLabel {
                font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }

            #typingIndicator {
                color: #666;
                font-size: 18px;
                letter-spacing: 4px;
                background: transparent;
                padding: 0 4px;
                font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
            }

            #chatInput {
                background-color: #1a1a1a;
                color: #fff;
                border: 1px solid #2a2a2a;
                padding: 14px 14px;
                border-radius: 12px;
                font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }

            #chatInput:focus {
                border: 1px solid #2a6496;
                background-color: #111;
            }

            #sendBtn {
                background-color: #1a1a1a;
                color: #fff;
                border: 1px solid #2a2a2a;
                padding: 8px 15px;
                border-radius: 3px;
                font-weight: bold;
            }

            #sendBtn:hover {
                background-color: #2a2a2a;
            }

            #cancelBtn {
                background-color: #3a1a1a;
                color: #ff6b6b;
                border: 1px solid #5a2a2a;
                padding: 8px 15px;
                border-radius: 3px;
                font-weight: bold;
            }

            #cancelBtn:hover {
                background-color: #4a2222;
                border-color: #ff6b6b;
            }

            #modelCombo {
                background-color: #1a1a1a;
                color: #ccc;
                border: 1px solid #2a2a2a;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 11px;
                font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
                min-width: 140px;
            }

            #modelCombo:hover {
                border-color: #444;
            }

            #modelCombo QAbstractItemView {
                background-color: #1a1a1a;
                color: #ccc;
                selection-background-color: #2a2a2a;
                border: 1px solid #333;
            }

            #modelCombo::drop-down {
                border: none;
                width: 20px;
            }

            #clearChatBtn {
                background-color: transparent;
                color: #666;
                border: 1px solid #2a2a2a;
                padding: 2px 10px;
                border-radius: 3px;
                font-size: 11px;
            }

            #clearChatBtn:hover {
                background-color: #2a2a2a;
                color: #fff;
            }
        """)

    # ── cleanup ──────────────────────────────────────────────────

    def cleanup(self):
        self._stop_typing_indicator()
        if self.streaming_timer:
            self.streaming_timer.stop()
        if self._active_table and self.save_chat_callback:
            try:
                self.save_chat_callback(
                    self._active_table,
                    json.dumps(self._message_records),
                    list(self.conversation_history))
            except Exception:
                pass
        self._clear_chat_widgets()
        self.conversation_history.clear()
        self._chat_store.clear()
        self._active_table = None
        self.current_context = None
