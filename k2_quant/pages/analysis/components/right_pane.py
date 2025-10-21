"""
Right Pane Component - Conversational AI with streaming

Contains AI chat interface for strategy development.
Save as: k2_quant/pages/analysis/components/right_pane.py
"""

from typing import Dict, Any, Optional
from datetime import datetime

from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QLineEdit, QPushButton, QLabel, QWidget, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QTextCursor

from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.services import table_controller
from k2_quant.utilities.text.math_formatter import MathFormatter


class CommandWorker(QThread):
    """Worker to execute AI-driven table commands in background."""

    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, table_name: str, command_text: str, conversation_state: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.table_name = table_name
        self.command_text = command_text
        self.conversation_state = conversation_state or {}

    def run(self):
        try:
            if table_controller is None:
                raise RuntimeError("table_controller service is not available")
            result = table_controller.execute_command(self.table_name, self.command_text, self.conversation_state)
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class RightPaneWidget(QFrame):
    """Right pane with AI chat interface"""
    
    # Signals
    message_sent = pyqtSignal(str)  # message
    strategy_generated = pyqtSignal(str, str)  # name, code
    projection_requested = pyqtSignal(dict)  # parameters
    
    def __init__(self):
        super().__init__()
        self.setFixedWidth(684)
        self.setObjectName("rightPane")
        
        self.current_context: Optional[Dict[str, Any]] = None
        self.conversation_history = []
        self.worker: Optional[CommandWorker] = None
        self.streaming_timer = None
        self.streaming_text = ""
        self.streaming_index = 0
        self.math_formatter = MathFormatter(use_block_markers=True)
        
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
        """Send message to AI and execute via controller"""
        message = self.ai_input.text().strip()
        if not message:
            return
        
        # Add user message to chat
        self.chat_display.append(f"\nYOU: {message}")
        
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

        # Launch background worker
        if self.worker and self.worker.isRunning():
            self.stream_response("Previous command is still executing. Please wait.")
            return

        # Show loading indicator
        self.loading_bar.show()
        
        # Build structured conversation_state for LLM follow-ups
        structured_answers = [
            {
                'label': h.get('label', ''),
                'value': h.get('value'),
                'column': h.get('column')
            }
            for h in self.conversation_history if h.get('role') == 'answer'
        ][-5:]
        # Fallback to assistant text if no structured answers yet
        if not structured_answers:
            structured_answers = [
                {'label': '', 'value': h.get('content')}
                for h in self.conversation_history if h.get('role') == 'assistant'
            ][-5:]
        conversation_state = {'last_answers': structured_answers}

        self.worker = CommandWorker(table_name, message, conversation_state)
        self.worker.result_ready.connect(self._on_worker_result)
        self.worker.error_occurred.connect(self._on_worker_error)
        self.worker.start()

        # Emit signal for external listeners if needed
        self.message_sent.emit(message)
    
    def stream_response(self, text: str, prefix: str = "\nAI: "):
        """Stream text with markdown table support."""
        if '|' in text and '\n|' in text and '---' in text:
            self._render_formatted_response(prefix, text)
            return

        # Apply math formatting only within explicit delimiters; keep others intact
        self.streaming_text = self.math_formatter.format_full(text)
        self.streaming_index = 0
        self.chat_display.append(prefix)
        
        if self.streaming_timer:
            self.streaming_timer.stop()
        
        self.streaming_timer = QTimer()
        self.streaming_timer.timeout.connect(self._stream_next_chunk)
        self.streaming_timer.start(20)

    def _render_formatted_response(self, prefix: str, text: str):
        """Render response with markdown table formatting."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(prefix)
        
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
            # Stream 1-3 characters at a time for natural appearance
            chunk_size = min(2, len(self.streaming_text) - self.streaming_index)
            chunk = self.streaming_text[self.streaming_index:self.streaming_index + chunk_size]
            
            # Move cursor to end and insert text
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(chunk)
            self.chat_display.setTextCursor(cursor)
            
            # Ensure visible
            self.chat_display.ensureCursorVisible()
            
            self.streaming_index += chunk_size
        else:
            # Streaming complete
            self.streaming_timer.stop()
            self.streaming_timer = None
    
    def _on_worker_result(self, result: Dict[str, Any]):
        """Handle worker completion - show only natural language results"""
        # Hide loading indicator
        self.loading_bar.hide()
        
        if result.get('success'):
            # Determine response text
            if result.get('interpreted_result'):
                response = result['interpreted_result']
            elif result.get('query_result') is not None:
                # Only use raw query_result as last resort
                query_result = result['query_result']
                response = str(query_result)
            elif result.get('new_columns'):
                cols = [c for c in result['new_columns'] if c]
                response = f"Successfully added {len(cols)} new column{'s' if len(cols) != 1 else ''}: {', '.join(cols)}"
            elif result.get('rows_deleted'):
                response = f"Deleted {result['rows_deleted']:,} rows from the dataset"
            elif result.get('rows_inserted'):
                response = f"Inserted {result['rows_inserted']:,} new rows into the dataset"
            elif result.get('rows_affected') is not None:
                response = f"Operation completed. {result['rows_affected']:,} rows were affected"
            else:
                response = "Operation completed successfully"
            
            # Stream the response
            self.stream_response(response)

            # Capture structured answers for follow-up pronoun resolution
            answer_to_remember = result.get('answer_to_remember')
            if answer_to_remember:
                self.conversation_history.append({
                    'role': 'answer',
                    'label': answer_to_remember.get('label', ''),
                    'value': answer_to_remember.get('value'),
                    'column': answer_to_remember.get('column'),
                    'timestamp': datetime.now().isoformat()
                })
        else:
            error = result.get('error', 'Operation failed')
            self.stream_response(f"Unable to complete that request. {error}")

        # Add to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response if result.get('success') else error,
            'timestamp': datetime.now().isoformat()
        })

        self.worker = None

    def _on_worker_error(self, error_msg: str):
        """Handle worker error"""
        # Hide loading indicator
        self.loading_bar.hide()
        
        self.stream_response(f"Unable to process that request. {error_msg}")
        self.worker = None
    
    def set_data_context(self, context: Dict[str, Any]):
        """Set the data context for AI - no announcement"""
        self.current_context = context
    
    def clear_chat(self):
        """Clear chat history"""
        self.chat_display.clear()
        self.conversation_history.clear()
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
        self.clear_chat()
        self.current_context = None