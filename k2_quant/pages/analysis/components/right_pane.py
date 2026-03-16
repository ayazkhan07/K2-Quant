"""
Right Pane Component - Conversational AI with Agent Loop

Contains AI chat interface backed by an agent loop that can execute
multiple tool calls (SQL, Python) per user message, showing intermediate
steps as the agent works.

Save as: k2_quant/pages/analysis/components/right_pane.py
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import html as html_mod
import re

from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QLineEdit, QPushButton, QLabel, QWidget, QProgressBar,
                             QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer, QEvent
from PyQt6.QtGui import QTextCursor, QTextBlockFormat, QTextCharFormat, QColor, QFontMetrics, QTextOption

from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.services import table_controller
from k2_quant.utilities.text.math_formatter import MathFormatter


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
    ):
        super().__init__()
        self.table_name = table_name
        self.command_text = command_text
        self.conversation_history = conversation_history or []
        self.initial_workspace = initial_workspace
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
    """Right pane with AI chat interface"""
    
    # Signals
    message_sent = pyqtSignal(str)  # message
    strategy_generated = pyqtSignal(str, str)  # name, code
    projection_requested = pyqtSignal(dict)  # parameters
    data_modified = pyqtSignal()  # emitted when the agent modifies table data
    tab_writes_ready = pyqtSignal(list)  # emitted when agent routes data to Tab 2/3
    
    def __init__(self):
        super().__init__()
        self.setFixedWidth(998)
        self.setObjectName("rightPane")
        
        self.current_context: Optional[Dict[str, Any]] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.worker: Optional[CommandWorker] = None
        self.streaming_timer = None
        self.streaming_text = ""
        self.streaming_index = 0
        self.math_formatter = MathFormatter(use_block_markers=True)
        self.workspace_provider: Optional[callable] = None
        self.save_chat_callback: Optional[callable] = None
        self.load_chat_callback: Optional[callable] = None

        # Per-model chat persistence: {table_name: {'html': str, 'history': list}}
        self._chat_store: Dict[str, Dict[str, Any]] = {}
        self._active_table: Optional[str] = None
        
        self.init_ui()
        self.setup_styling()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        self.setLayout(layout)
        
        # Header row
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        ai_label = QLabel("CONVERSATIONAL AI")
        ai_label.setObjectName("sectionTitle")
        header_row.addWidget(ai_label)
        header_row.addStretch()
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clearChatBtn")
        self.clear_btn.setFixedHeight(24)
        self.clear_btn.clicked.connect(self.clear_chat)
        header_row.addWidget(self.clear_btn)
        layout.addLayout(header_row)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setObjectName("chatDisplay")
        self.chat_display.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.chat_display.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.chat_display.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_display.document().setDefaultStyleSheet(
            "body { word-wrap: break-word; }"
            "table { table-layout: fixed; width: 100%; }"
            "td, th { word-wrap: break-word; overflow-wrap: break-word; }"
        )
        layout.addWidget(self.chat_display)
        
        # Loading indicator (initially hidden)
        self.loading_bar = QProgressBar()
        self.loading_bar.setObjectName("loadingBar")
        self.loading_bar.setMaximum(0)  # Indeterminate progress
        self.loading_bar.setMinimum(0)
        self.loading_bar.setTextVisible(False)
        self.loading_bar.setFixedHeight(2)
        self.loading_bar.hide()
        layout.addWidget(self.loading_bar)
        
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
        self.ai_input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ai_input.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ai_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.ai_input.document().setDocumentMargin(4)
        self._input_line_height = QFontMetrics(self.ai_input.font()).lineSpacing()
        self._input_max_lines = 5
        self._input_padding = 28
        self.ai_input.setFixedHeight(self._input_line_height + self._input_padding)
        self.ai_input.textChanged.connect(self._adjust_input_height)
        self.ai_input.installEventFilter(self)
        input_layout.addWidget(self.ai_input)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_ai_message)
        self.send_btn.setObjectName("sendBtn")
        input_layout.addWidget(self.send_btn)
        
        layout.addWidget(input_widget)
    
    def send_ai_message(self):
        """Send message to AI and execute via agent loop"""
        message = self.ai_input.toPlainText().strip()
        if not message:
            return
        
        # Add user message to chat (right-aligned)
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        block_fmt = QTextBlockFormat()
        block_fmt.setAlignment(Qt.AlignmentFlag.AlignRight)
        block_fmt.setTopMargin(18)
        block_fmt.setBottomMargin(0)
        cursor.insertBlock(block_fmt)

        label_fmt = QTextCharFormat()
        label_fmt.setForeground(QColor("#666666"))
        cursor.insertText("YOU: ", label_fmt)

        text_fmt = QTextCharFormat()
        text_fmt.setForeground(QColor("#ffffff"))
        cursor.insertText(message, text_fmt)

        self.chat_display.setTextCursor(cursor)
        
        # Clear input and reset height
        self.ai_input.clear()
        self.ai_input.setFixedHeight(self._input_line_height + self._input_padding)
        self.ai_input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Add to history
        self.conversation_history.append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Determine current table from context
        table_name = None
        if self.current_context:
            table_name = self.current_context.get('table_name')

        if not table_name:
            self.stream_response("Please load a model first so I know which table to operate on.")
            return

        # Prevent concurrent requests
        if self.worker and self.worker.isRunning():
            self.stream_response("Previous command is still executing. Please wait.")
            return

        # Show loading indicator
        self.loading_bar.show()
        
        # Build clean history for the agent (exclude the message we just added)
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

        self.worker = CommandWorker(
            table_name, message, history_for_agent, initial_workspace)
        self.worker.result_ready.connect(self._on_worker_result)
        self.worker.error_occurred.connect(self._on_worker_error)
        self.worker.step_update.connect(self._on_step_update)
        self.worker.start()

        self._set_cancel_mode(True)
        self.message_sent.emit(message)

    def _cancel_request(self):
        """Cancel the running AI request."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            k2_logger.info("AI request cancelled by user", "AI_CHAT")

    def _set_cancel_mode(self, active: bool):
        """Toggle the send button between Send and Cancel states."""
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

    # ── intermediate step display ──────────────────────────────────

    def _on_step_update(self, text: str):
        """Display an intermediate agent step in subdued style."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        block_fmt = QTextBlockFormat()
        block_fmt.setAlignment(Qt.AlignmentFlag.AlignLeft)
        block_fmt.setTopMargin(2)
        block_fmt.setBottomMargin(2)
        cursor.insertBlock(block_fmt)

        step_fmt = QTextCharFormat()
        step_fmt.setForeground(QColor("#666666"))
        cursor.insertText(f"  {text}", step_fmt)

        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    # ── response rendering ─────────────────────────────────────────

    # ── markdown table helpers ─────────────────────────────────────

    @staticmethod
    def _is_separator_line(line: str) -> bool:
        """True for markdown table separators like |---|---| or :---: | :---:"""
        stripped = line.strip()
        if not stripped or '|' not in stripped:
            return False
        return bool(re.match(r'^[\|\s\-:]+$', stripped) and '--' in stripped)

    def _has_markdown_table(self, text: str) -> bool:
        """Check whether *text* contains at least one markdown-style table."""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        has_sep = any(self._is_separator_line(l) for l in lines)
        pipe_data = sum(
            1 for l in lines if '|' in l and not self._is_separator_line(l)
        )
        return has_sep and pipe_data >= 2

    # ── response rendering ─────────────────────────────────────────

    def stream_response(self, text: str, prefix: str = "AI: "):
        """Stream text with markdown table support."""
        if self._has_markdown_table(text):
            self._render_formatted_response(text)
            return

        self.streaming_text = self.math_formatter.format_full(text)
        self.streaming_index = 0

        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        block_fmt = QTextBlockFormat()
        block_fmt.setAlignment(Qt.AlignmentFlag.AlignLeft)
        block_fmt.setTopMargin(6)
        block_fmt.setBottomMargin(18)
        cursor.insertBlock(block_fmt)

        label_fmt = QTextCharFormat()
        label_fmt.setForeground(QColor("#666666"))
        cursor.insertText(prefix, label_fmt)

        text_fmt = QTextCharFormat()
        text_fmt.setForeground(QColor("#ffffff"))
        cursor.setCharFormat(text_fmt)
        self.chat_display.setTextCursor(cursor)
        
        if self.streaming_timer:
            self.streaming_timer.stop()
        
        self.streaming_timer = QTimer()
        self.streaming_timer.timeout.connect(self._stream_next_chunk)
        self.streaming_timer.start(20)

    def _render_formatted_response(self, text: str):
        """Render response converting markdown tables to proper HTML tables."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        block_fmt = QTextBlockFormat()
        block_fmt.setAlignment(Qt.AlignmentFlag.AlignLeft)
        block_fmt.setTopMargin(6)
        block_fmt.setBottomMargin(18)
        cursor.insertBlock(block_fmt)

        label_fmt = QTextCharFormat()
        label_fmt.setForeground(QColor("#666666"))
        cursor.insertText("AI: ", label_fmt)

        text_fmt = QTextCharFormat()
        text_fmt.setForeground(QColor("#ffffff"))
        cursor.setCharFormat(text_fmt)

        lines = text.split('\n')
        i = 0

        while i < len(lines):
            stripped = lines[i].strip()

            if '|' in stripped and not self._is_separator_line(stripped):
                table_lines = [stripped]
                j = i + 1
                found_separator = False

                while j < len(lines):
                    s = lines[j].strip()
                    if not s:
                        j += 1
                        continue
                    if self._is_separator_line(s):
                        found_separator = True
                        j += 1
                        continue
                    if '|' in s:
                        table_lines.append(s)
                        j += 1
                        continue
                    break

                if found_separator and len(table_lines) >= 2:
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
                        self._insert_table(cursor, rows)
                    i = j
                    continue

            if stripped:
                cursor.insertText(stripped + '\n')
            i += 1

        self.chat_display.ensureCursorVisible()

    def _insert_table(self, cursor, rows):
        """Insert a well-formatted HTML table into the chat display."""
        if not rows:
            return

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
        cursor.insertHtml(table_html)
    
    def _stream_next_chunk(self):
        """Stream next chunk of text"""
        if self.streaming_index < len(self.streaming_text):
            chunk_size = min(2, len(self.streaming_text) - self.streaming_index)
            chunk = self.streaming_text[self.streaming_index:self.streaming_index + chunk_size]
            
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("#ffffff"))
            cursor.insertText(chunk, fmt)
            self.chat_display.setTextCursor(cursor)
            
            self.chat_display.ensureCursorVisible()
            
            self.streaming_index += chunk_size
        else:
            self.streaming_timer.stop()
            self.streaming_timer = None

    # ── worker result handling ─────────────────────────────────────
    
    def _on_worker_result(self, result: Dict[str, Any]):
        """Handle worker completion."""
        self.loading_bar.hide()
        self._set_cancel_mode(False)

        if result.get('cancelled'):
            self.stream_response("Request cancelled.", prefix="")
            self.conversation_history.append({
                'role': 'assistant',
                'content': "Request cancelled.",
                'timestamp': datetime.now().isoformat()
            })
        elif result.get('success'):
            response = result.get('display_message', 'Operation completed successfully.')
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
        else:
            error = result.get('error', 'Operation failed')
            self.stream_response(f"Unable to complete that request. {error}")

            self.conversation_history.append({
                'role': 'assistant',
                'content': f"Error: {error}",
                'timestamp': datetime.now().isoformat()
            })

        self.worker = None

    def _on_worker_error(self, error_msg: str):
        """Handle worker error"""
        self.loading_bar.hide()
        self._set_cancel_mode(False)
        self.stream_response(f"Unable to process that request. {error_msg}")
        self.worker = None
    
    # ── context and state management ───────────────────────────────

    def set_data_context(self, context: Dict[str, Any]):
        """Set the data context for AI, saving and restoring chat per model."""
        new_table = context.get('table_name') if context else None

        if self._active_table:
            chat_state = {
                'html': self.chat_display.toHtml(),
                'history': list(self.conversation_history),
            }
            self._chat_store[self._active_table] = chat_state
            if self.save_chat_callback:
                try:
                    self.save_chat_callback(
                        self._active_table,
                        chat_state['html'],
                        chat_state['history'])
                except Exception:
                    pass

        if new_table and new_table in self._chat_store:
            saved = self._chat_store[new_table]
            self.chat_display.setHtml(saved['html'])
            self.conversation_history = list(saved['history'])
        elif new_table and self.load_chat_callback:
            try:
                saved = self.load_chat_callback(new_table)
                if saved and saved.get('html'):
                    self.chat_display.setHtml(saved['html'])
                    self.conversation_history = list(saved.get('history', []))
                    self._chat_store[new_table] = saved
                else:
                    self.chat_display.clear()
                    self.conversation_history.clear()
            except Exception:
                self.chat_display.clear()
                self.conversation_history.clear()
        else:
            self.chat_display.clear()
            self.conversation_history.clear()

        self._active_table = new_table
        self.current_context = context
    
    def clear_chat(self):
        """Clear chat history for the active model and persist the empty state."""
        self.chat_display.clear()
        self.conversation_history.clear()
        if self._active_table and self._active_table in self._chat_store:
            del self._chat_store[self._active_table]
        if self._active_table and self.save_chat_callback:
            try:
                self.save_chat_callback(self._active_table, '', [])
            except Exception:
                pass
        k2_logger.info("Chat cleared", "AI_CHAT")
    
    # ── auto-growing input ───────────────────────────────────────────

    def eventFilter(self, obj, event):
        """Enter sends the message; Shift+Enter inserts a newline."""
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
            self.ai_input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        else:
            self.ai_input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def setup_styling(self):
        """Apply styling to the pane"""
        self.setStyleSheet("""
            #rightPane {
                background-color: #0f0f0f;
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
            
            #chatDisplay {
                background-color: #0a0a0a;
                color: #ccc;
                border: 1px solid #1a1a1a;
                border-radius: 4px;
                padding: 10px;
                font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
                line-height: 1.6;
            }
            
            #loadingBar {
                background-color: #0a0a0a;
                border: none;
            }
            
            #loadingBar::chunk {
                background-color: #ffffff;
                border-radius: 1px;
            }
            
            #chatInput {
                background-color: #1a1a1a;
                color: #fff;
                border: 1px solid #2a2a2a;
                padding: 8px;
                border-radius: 3px;
                font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
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
    
    def cleanup(self):
        """Cleanup resources, persisting active chat before clearing."""
        if self.streaming_timer:
            self.streaming_timer.stop()
        if self._active_table and self.save_chat_callback:
            try:
                self.save_chat_callback(
                    self._active_table,
                    self.chat_display.toHtml(),
                    list(self.conversation_history))
            except Exception:
                pass
        self.chat_display.clear()
        self.conversation_history.clear()
        self._chat_store.clear()
        self._active_table = None
        self.current_context = None
