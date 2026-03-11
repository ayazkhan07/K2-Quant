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

from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QLineEdit, QPushButton, QLabel, QWidget, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QTextCursor, QTextBlockFormat, QTextCharFormat, QColor

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
    ):
        super().__init__()
        self.table_name = table_name
        self.command_text = command_text
        self.conversation_history = conversation_history or []

    def run(self):
        try:
            if table_controller is None:
                raise RuntimeError("table_controller service is not available")
            result = table_controller.execute_command(
                self.table_name,
                self.command_text,
                self.conversation_history,
                step_callback=self._on_step,
            )
            self.result_ready.emit(result)
        except Exception as e:
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
    
    def __init__(self):
        super().__init__()
        self.setFixedWidth(570)
        self.setObjectName("rightPane")
        
        self.current_context: Optional[Dict[str, Any]] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.worker: Optional[CommandWorker] = None
        self.streaming_timer = None
        self.streaming_text = ""
        self.streaming_index = 0
        self.math_formatter = MathFormatter(use_block_markers=True)

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
        
        # Header
        ai_label = QLabel("CONVERSATIONAL AI")
        ai_label.setObjectName("sectionTitle")
        layout.addWidget(ai_label)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setObjectName("chatDisplay")
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
        
        self.ai_input = QLineEdit()
        self.ai_input.setPlaceholderText("Type your message...")
        self.ai_input.returnPressed.connect(self.send_ai_message)
        self.ai_input.setObjectName("chatInput")
        input_layout.addWidget(self.ai_input)
        
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.send_ai_message)
        send_btn.setObjectName("sendBtn")
        input_layout.addWidget(send_btn)
        
        layout.addWidget(input_widget)
    
    def send_ai_message(self):
        """Send message to AI and execute via agent loop"""
        message = self.ai_input.text().strip()
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
        
        # Clear input
        self.ai_input.clear()
        
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

        self.worker = CommandWorker(table_name, message, history_for_agent)
        self.worker.result_ready.connect(self._on_worker_result)
        self.worker.error_occurred.connect(self._on_worker_error)
        self.worker.step_update.connect(self._on_step_update)
        self.worker.start()

        self.message_sent.emit(message)
    
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

    def stream_response(self, text: str, prefix: str = "AI: "):
        """Stream text with markdown table support."""
        if '|' in text and '\n|' in text and '---' in text:
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
        """Render response with markdown table formatting."""
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
        
        parts = text.split('\n')
        in_table = False
        table_rows = []
        
        for line in parts:
            if '|' in line and not line.strip().startswith('|--') and not set(line.strip()) == {'|'}:
                if not in_table:
                    in_table = True
                    table_rows = []
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if cells:
                    table_rows.append(cells)
            elif '|--' in line or '---' in line:
                continue
            else:
                if in_table and table_rows:
                    self._insert_table(cursor, table_rows)
                    table_rows = []
                    in_table = False
                if line.strip():
                    cursor.insertText(line + '\n')
        
        if in_table and table_rows:
            self._insert_table(cursor, table_rows)
        
        self.chat_display.ensureCursorVisible()

    def _insert_table(self, cursor, rows):
        """Insert a formatted table into the chat."""
        if not rows:
            return
        
        table_html = """
        <table style='border-collapse: collapse; margin: 10px 0; font-family: monospace;'>
        """
        
        # Header
        table_html += "<tr style='background: #2a2a2a;'>"
        for cell in rows[0]:
            table_html += f"<th style='padding: 5px 10px; border: 1px solid #444; color: #fff; text-align: left;'>{cell}</th>"
        table_html += "</tr>"
        
        # Body
        for row in rows[1:]:
            table_html += "<tr>"
            for cell in row:
                table_html += f"<td style='padding: 5px 10px; border: 1px solid #444; color: #ccc;'>{cell}</td>"
            table_html += "</tr>"
        
        table_html += "</table>"
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
        
        if result.get('success'):
            response = result.get('display_message', 'Operation completed successfully.')
            self.stream_response(response)

            self.conversation_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })

            if result.get('data_modified'):
                self.data_modified.emit()
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
        self.stream_response(f"Unable to process that request. {error_msg}")
        self.worker = None
    
    # ── context and state management ───────────────────────────────

    def set_data_context(self, context: Dict[str, Any]):
        """Set the data context for AI, saving and restoring chat per model."""
        new_table = context.get('table_name') if context else None

        if self._active_table:
            self._chat_store[self._active_table] = {
                'html': self.chat_display.toHtml(),
                'history': list(self.conversation_history),
            }

        if new_table and new_table in self._chat_store:
            saved = self._chat_store[new_table]
            self.chat_display.setHtml(saved['html'])
            self.conversation_history = list(saved['history'])
        else:
            self.chat_display.clear()
            self.conversation_history.clear()

        self._active_table = new_table
        self.current_context = context
    
    def clear_chat(self):
        """Clear chat history for the active model"""
        self.chat_display.clear()
        self.conversation_history.clear()
        if self._active_table and self._active_table in self._chat_store:
            del self._chat_store[self._active_table]
        k2_logger.info("Chat cleared", "AI_CHAT")
    
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
                font-family: 'Segoe UI', Arial, sans-serif;
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
        """)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.streaming_timer:
            self.streaming_timer.stop()
        self.chat_display.clear()
        self.conversation_history.clear()
        self._chat_store.clear()
        self._active_table = None
        self.current_context = None
