"""
AI Chat Widget for K2 Quant Analysis

Provides AI chat interface with streaming responses for strategy development.
Save as: k2_quant/pages/analysis/widgets/ai_chat_widget.py
"""

import json
import re
from typing import Dict, Any, Optional, List, Generator
from datetime import datetime

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QLineEdit, QPushButton, QLabel, QComboBox,
                             QProgressBar, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QTextCursor, QFont, QTextCharFormat, QColor

from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.services import OllamaService, OllamaConfig
from k2_quant.utilities.data.db_manager import db_manager


class AIStreamThread(QThread):
    """Thread for streaming AI responses"""
    
    text_chunk = pyqtSignal(str)
    complete = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, message, provider=None):
        super().__init__()
        self.message = message
        self.provider = provider
    
    def run(self):
        """Stream AI response"""
        try:
            # Check if AI service is available
            try:
                from k2_quant.utilities.services.ai_chat_service import ai_chat_service
                
                if not ai_chat_service:
                    self.error.emit("AI service not available")
                    return
                
                # Stream response
                for chunk in ai_chat_service.get_streaming_response(self.message):
                    self.text_chunk.emit(chunk)
                
            except ImportError:
                # Fallback if AI service not available
                self.text_chunk.emit("AI service not configured. Please set up API keys.")
            
            self.complete.emit()
            
        except Exception as e:
            self.error.emit(str(e))


class AIChatWidget(QWidget):
    """AI chat widget with streaming support"""
    
    # Signals
    code_generated = pyqtSignal(str, str)
    strategy_saved = pyqtSignal(str, str)
    message_sent = pyqtSignal(str)
    # Emitted when a [QUERY] response provides executable lookup code
    query_generated = pyqtSignal(str)
    # New: SQL preview and mutation request
    sql_generated = pyqtSignal(list)
    schema_change_requested = pyqtSignal(list, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_context = None
        self.conversation_history = []
        self.ai_thread = None
        self.current_code_block = ""
        self.current_ai_text = ""
        self.is_streaming = False
        
        self.init_ui()
        self.setup_styling()
        self.show_welcome_message()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.setLayout(layout)
        
        # Provider and Model selector
        model_selector = self.create_model_selector()
        layout.addWidget(model_selector)

        # NEW: DB table selector + mutation toggle row
        top_controls = QWidget()
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_controls.setLayout(top_layout)

        top_layout.addWidget(QLabel("Target Table:"))
        self.table_selector = QComboBox(self)
        self.table_selector.addItem("(loading...)")
        top_layout.addWidget(self.table_selector)

        self.enable_mutations_checkbox = QCheckBox("Enable Schema/Data Modifications", self)
        self.enable_mutations_checkbox.setChecked(False)
        self.enable_mutations_checkbox.stateChanged.connect(self._on_mutation_toggle)
        top_layout.addWidget(self.enable_mutations_checkbox)

        top_layout.addStretch()
        layout.addWidget(top_controls)
        
        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setObjectName("chatDisplay")
        layout.addWidget(self.chat_display)
        
        # Streaming indicator
        self.streaming_indicator = QProgressBar()
        self.streaming_indicator.setMaximum(0)
        self.streaming_indicator.setTextVisible(False)
        self.streaming_indicator.setFixedHeight(3)
        self.streaming_indicator.hide()
        layout.addWidget(self.streaming_indicator)
        
        # Input area
        input_widget = self.create_input_area()
        layout.addWidget(input_widget)
    
    
    
    def create_model_selector(self):
        """Create model selector widget"""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)
        
        # Provider selector
        layout.addWidget(QLabel("Provider:"))
        
        self.provider_selector = QComboBox()
        self.provider_selector.addItems(["Anthropic", "OpenAI", "Local"])
        self.provider_selector.currentTextChanged.connect(self.on_provider_changed)
        self.provider_selector.setObjectName("providerSelector")
        layout.addWidget(self.provider_selector)
        
        layout.addSpacing(10)
        
        # Model selector
        layout.addWidget(QLabel("Model:"))
        
        self.model_selector = QComboBox()
        self.update_model_list()
        self.model_selector.currentTextChanged.connect(self.on_model_changed)
        self.model_selector.setObjectName("modelSelector")
        layout.addWidget(self.model_selector)

        # Move Clear button here (from removed header)
        layout.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(50)
        clear_btn.clicked.connect(self.clear_chat)
        clear_btn.setObjectName("clearChatBtn")
        layout.addWidget(clear_btn)
        
        return widget
    
    
    def create_input_area(self):
        """Create input area widget"""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Describe your strategy or ask a question...")
        self.input_field.returnPressed.connect(self.send_message)
        self.input_field.setObjectName("chatInput")
        layout.addWidget(self.input_field)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setFixedWidth(60)
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setObjectName("sendBtn")
        layout.addWidget(self.send_btn)
        
        return widget
    
    def update_model_list(self):
        """Update model list based on provider"""
        provider = self.provider_selector.currentText().lower()
        
        if provider == "openai":
            models = ["gpt-4-turbo-preview", "gpt-4o", "gpt-3.5-turbo"]
        elif provider == "anthropic":
            models = [
                "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229",
                "claude-3-haiku-20240307"
            ]
        elif provider == "local":
            models = ["No models (simulated)"]
        else:
            models = []
        
        self.model_selector.clear()
        self.model_selector.addItems(models)
    
    def on_provider_changed(self, provider):
        """Handle provider change"""
        self.update_model_list()
        
        # Force select first model for new provider
        if self.model_selector.count() > 0:
            self.model_selector.setCurrentIndex(0)
        
        # Try to set provider and current model in AI service
        try:
            from k2_quant.utilities.services.ai_chat_service import ai_chat_service
            if ai_chat_service:
                ai_chat_service.set_provider(provider.lower())
                if self.model_selector.currentText():
                    ai_chat_service.set_model(self.model_selector.currentText())
                k2_logger.info(f"Provider changed: {provider}/{self.model_selector.currentText()}", "AI_CHAT")
        except ImportError:
            k2_logger.debug("AI service not available", "AI_CHAT")
    
    def on_model_changed(self, model):
        """Handle model change"""
        try:
            from k2_quant.utilities.services.ai_chat_service import ai_chat_service
            if ai_chat_service and model:
                ai_chat_service.set_model(model)
                k2_logger.info(f"AI model changed to {model}", "AI_CHAT")
        except ImportError:
            k2_logger.debug("AI service not available", "AI_CHAT")
    
    def show_welcome_message(self):
        """Show welcome message"""
        welcome = """<div style='color: #999; font-style: italic;'>
        Welcome to K2 Quant AI Assistant!<br><br>
        I can help you:
        <ul>
        <li>Create custom trading strategies</li>
        <li>Generate price projections</li>
        <li>Analyze patterns in your data</li>
        <li>Write Python code for complex calculations</li>
        </ul>
        Load a model and describe what you'd like to analyze.
        </div>"""
        
        self.chat_display.setHtml(welcome)

        # Initialize Ollama service (mutations off by default) and populate tables
        try:
            self.ollama = OllamaService(
                db=db_manager,
                config=OllamaConfig(
                    allow_schema_changes=False,
                    allow_data_modifications=False,
                    auto_backup_before_changes=False,
                    create_audit_tables=False,
                ),
            )
        except Exception as e:
            k2_logger.error(f"Failed to initialize OllamaService: {e}", "AI_CHAT")
            self.ollama = None
        QTimer.singleShot(0, self._load_table_selector)
    
    def set_data_context(self, data, metadata):
        """Set the data context for AI and pass dataset awareness"""
        self.current_context = {
            'metadata': metadata,
            'data_shape': data.shape if hasattr(data, 'shape') else len(data),
            'columns': list(data.columns) if hasattr(data, 'columns') else None
        }
        
        # Build compact recent sample and quick stats for grounding
        recent_sample = []
        quick_stats = {}
        try:
            if hasattr(data, "columns") and hasattr(data, "tail"):
                sample_cols = [c for c in ['date_time', 'open', 'high', 'low', 'close', 'volume'] if c in data.columns]
                if sample_cols:
                    recent_sample = data.tail(10)[sample_cols].to_dict(orient='records')
                quick_stats = self._calculate_quick_stats(data)
        except Exception:
            pass
        
        # Update AI service context (system + dataset)
        try:
            from k2_quant.utilities.services.ai_chat_service import ai_chat_service
            if ai_chat_service:
                context = f"""
                Working with stock data:
                - Symbol: {metadata.get('symbol', 'Unknown')}
                - Records: {self.current_context['data_shape'][0] if isinstance(self.current_context['data_shape'], tuple) else self.current_context['data_shape']}
                - Columns: {', '.join(self.current_context['columns']) if self.current_context['columns'] else 'Unknown'}
                """
                ai_chat_service.set_system_context(context)
                ai_chat_service.set_dataset_context(
                    metadata,
                    list(data.columns) if hasattr(data, "columns") else [],
                    recent_sample,
                    quick_stats,
                    exchange="NYSE",
                    tz="US/Eastern"
                )
        except ImportError:
            pass
        
        # Show in chat
        records = self.current_context['data_shape'][0] if isinstance(self.current_context['data_shape'], tuple) else self.current_context['data_shape']
        self.add_system_message(f"📊 Loaded {metadata.get('symbol', 'Unknown')} data with {records} records")
    
    
    def send_message(self):
        """Send message to AI (auto-mutation + auto-reload flow)."""
        message = self.input_field.text().strip()
        if not message or self.is_streaming:
            return

        # Show user message and start streaming assistant text
        self.add_user_message(message)
        self.input_field.clear()
        self.message_sent.emit(message)
        self.start_streaming(message)

        # Auto-apply default elasticity formula when ambiguous
        m_lower = message.lower()
        if "elasticity" in m_lower and self._formula_is_ambiguous(message):
            message = f"{message} using formula ((high - low) / low) * 100"

        # Dispatch background work
        try:
            if self._is_mutation_intent(message):
                # If mutations are disabled, enable them for this action
                if hasattr(self, "enable_mutations_checkbox") and not self.enable_mutations_checkbox.isChecked():
                    self.enable_mutations_checkbox.setChecked(True)
                self._start_schema_generation(message)  # generate → execute (no preview)
            else:
                self._start_read_query(message)
        except Exception as e:
            k2_logger.error(f"Dispatch error: {e}", "AI_CHAT")
    
    def start_streaming(self, message):
        """Start streaming AI response"""
        self.is_streaming = True
        self.current_code_block = ""
        self.current_ai_text = ""
        
        # Disable input
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        
        # Show streaming indicator
        self.streaming_indicator.show()
        
        # Add AI message placeholder
        self.add_ai_message("")
        
        # Start thread (legacy local AIStreamThread kept for compatibility)
        self.ai_thread = AIStreamThread(message)
        self.ai_thread.text_chunk.connect(self.append_ai_text)
        self.ai_thread.complete.connect(self.streaming_complete)
        self.ai_thread.error.connect(self.streaming_error)
        self.ai_thread.start()
    
    def append_ai_text(self, chunk):
        """Append streamed text to AI message"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Accumulate full AI text for tag parsing
        self.current_ai_text += chunk
        
        # Check for code blocks
        if "```python" in chunk or "```" in chunk:
            self.current_code_block += chunk
        else:
            cursor.insertText(chunk)
        
        # Auto-scroll
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def streaming_complete(self):
        """Handle streaming completion"""
        self.is_streaming = False
        
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        
        # Hide streaming indicator
        self.streaming_indicator.hide()
        
        # Legacy [QUERY]/[DIRECT] flow removed; we handle in background threads now
        k2_logger.info("AI streaming complete", "AI_CHAT")
    
    def streaming_error(self, error):
        """Handle streaming error"""
        self.is_streaming = False
        
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        
        # Hide streaming indicator
        self.streaming_indicator.hide()
        
        # Show error
        self.add_system_message(f"❌ Error: {error}")
        k2_logger.error(f"AI streaming error: {error}", "AI_CHAT")
    
    def process_code_block(self, text):
        """Process code block from AI response"""
        code = self._extract_python_code(text)
        if not code:
            return
        
        # Show code in formatted block
        self.add_code_block(code)
        
        # Ask if user wants to execute
        self.add_system_message("📝 Code generated. Click 'Execute' to run this strategy.")
        
        # Store for execution
        self.current_code_block = code
        
        # Emit signal
        self.code_generated.emit(code, "AI Generated Strategy")

    def _extract_python_code(self, text: str) -> Optional[str]:
        """Extract Python code from a markdown fenced block"""
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
                return code
        return None

    def _parse_tag(self, text: str) -> (Optional[str], str):
        """Parse leading tag like [DIRECT]/[QUERY]/[STRATEGY]/[DECLINE]"""
        try:
            m = re.match(r'^\s*\[(DIRECT|QUERY|STRATEGY|DECLINE)\]\s*', text, re.IGNORECASE)
            if m:
                tag = m.group(1).upper()
                remainder = text[m.end():]
                return tag, remainder
        except Exception:
            pass
        return None, text
    
    def add_user_message(self, message):
        """Add user message to chat"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Add spacing
        cursor.insertText("\n\n")
        
        # Add user label
        format = QTextCharFormat()
        format.setForeground(QColor("#4a4"))
        format.setFontWeight(QFont.Weight.Bold)
        cursor.setCharFormat(format)
        cursor.insertText("YOU: ")
        
        # Add message
        format.setForeground(QColor("#fff"))
        format.setFontWeight(QFont.Weight.Normal)
        cursor.setCharFormat(format)
        cursor.insertText(message)
        
        # Auto-scroll
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Add to history
        self.conversation_history.append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_ai_message(self, message):
        """Add AI message to chat"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Add spacing
        cursor.insertText("\n\n")
        
        # Add AI label
        format = QTextCharFormat()
        format.setForeground(QColor("#4aa"))
        format.setFontWeight(QFont.Weight.Bold)
        cursor.setCharFormat(format)
        cursor.insertText("AI: ")
        
        # Add message
        format.setForeground(QColor("#ccc"))
        format.setFontWeight(QFont.Weight.Normal)
        cursor.setCharFormat(format)
        
        if message:
            cursor.insertText(message)
        
        # Auto-scroll
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def add_system_message(self, message):
        """Add system message to chat"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Add spacing
        cursor.insertText("\n\n")
        
        # Add message
        format = QTextCharFormat()
        format.setForeground(QColor("#999"))
        format.setFontItalic(True)
        cursor.setCharFormat(format)
        cursor.insertText(message)
        
        # Auto-scroll
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def add_code_block(self, code):
        """Add formatted code block to chat"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Add spacing
        cursor.insertText("\n\n")
        
        # Add code in monospace
        format = QTextCharFormat()
        format.setForeground(QColor("#4f4"))
        format.setFontFamily("Consolas, Monaco, monospace")
        format.setBackground(QColor("#1a1a1a"))
        cursor.setCharFormat(format)
        cursor.insertText(code)
        
        # Reset format
        format = QTextCharFormat()
        cursor.setCharFormat(format)
    
    def clear_chat(self):
        """Clear chat history"""
        self.chat_display.clear()
        self.conversation_history.clear()
        self.show_welcome_message()
        
        try:
            from k2_quant.utilities.services.ai_chat_service import ai_chat_service
            if ai_chat_service:
                ai_chat_service.clear_history()
        except ImportError:
            pass
        
        k2_logger.info("Chat cleared", "AI_CHAT")
    
    def show_error(self, error):
        """Show error message in chat"""
        self.add_system_message(f"⚠️ {error}")
    
    def setup_styling(self):
        """Apply styling to the widget"""
        self.setStyleSheet("""
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
            
            #chatInput {
                background-color: #1a1a1a;
                color: #fff;
                border: 1px solid #2a2a2a;
                padding: 8px;
                border-radius: 3px;
                font-size: 13px;
            }
            
            #chatInput:focus {
                border-color: #3a3a3a;
            }
            
            #sendBtn {
                background-color: #1a1a1a;
                color: #fff;
                border: 1px solid #4a4a4a;
                padding: 8px;
                border-radius: 3px;
                font-weight: bold;
            }
            
            #sendBtn:hover:enabled {
                background-color: #2a2a2a;
                border-color: #5a5a5a;
            }
            
            #sendBtn:disabled {
                background-color: #0a0a0a;
                color: #444;
                border-color: #2a2a2a;
            }
            
            #clearChatBtn {
                background-color: transparent;
                color: #666;
                border: 1px solid #2a2a2a;
                padding: 3px 8px;
                border-radius: 2px;
                font-size: 10px;
            }
            
            #clearChatBtn:hover {
                background-color: #1a1a1a;
                color: #999;
            }
            
            #providerSelector, #modelSelector {
                background-color: #1a1a1a;
                color: #fff;
                border: 1px solid #2a2a2a;
                padding: 4px;
                border-radius: 3px;
                font-size: 11px;
            }
            
            QComboBox::drop-down {
                border: none;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #666;
                width: 0;
                height: 0;
                margin-right: 4px;
            }
            
            QLabel {
                color: #999;
                font-size: 11px;
            }
            
            QProgressBar {
                background-color: #1a1a1a;
                border: none;
            }
            
            QProgressBar::chunk {
                background-color: #4aa;
            }
        """)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.ai_thread and self.ai_thread.isRunning():
            self.ai_thread.terminate()
            self.ai_thread.wait()

    # -------- Helpers ----------

    def _calculate_quick_stats(self, df) -> dict:
        """Compute small set of quick stats for grounding."""
        stats = {}
        try:
            if df is not None and hasattr(df, "columns") and len(df) > 0:
                if 'high' in df.columns:
                    stats['week_high'] = float(df['high'].tail(25).max())
                if 'low' in df.columns:
                    stats['week_low'] = float(df['low'].tail(25).min())
                if 'close' in df.columns:
                    stats['last_close'] = float(df['close'].iloc[-1])
                if 'volume' in df.columns:
                    stats['avg_volume'] = float(df['volume'].tail(20).mean())
        except Exception:
            pass
        return stats

    # =================== NEW: Ollama integration helpers ===================

    def _on_mutation_toggle(self, state):
        try:
            if not self.ollama:
                return
            enabled = bool(state)
            self.ollama.config.allow_schema_changes = enabled
            self.ollama.config.allow_data_modifications = enabled
            self.add_system_message("Schema/Data modifications " + ("ENABLED" if enabled else "DISABLED"))
        except Exception:
            pass

    def _load_table_selector(self):
        if not getattr(self, 'ollama', None):
            return
        try:
            schema = self.ollama.introspect_schema()
            stock_tables = [t for t in schema.keys() if t.startswith('stock_')]
            self.table_selector.clear()
            if stock_tables:
                self.table_selector.addItems(stock_tables)
            else:
                self.table_selector.addItem("(no stock_* tables)")
        except Exception as exc:
            self.table_selector.clear()
            self.table_selector.addItem(f"(schema error: {exc})")

    def _is_mutation_intent(self, text: str) -> bool:
        t = text.lower()
        return any(kw in t for kw in ["add column", "alter table", "delete rows", "update set", "remove outlier", "modify column"])

    def _start_schema_generation(self, request_text: str):
        if not getattr(self, 'ollama', None):
            return
        # Run only if mutations might be intended; clarification handled in chat flow by user
        self.setCursor(Qt.CursorShape.WaitCursor)
        self._gen_thread = SchemaGenThread(self.ollama, request_text)
        self._gen_thread.generated.connect(self._on_ops_generated)
        self._gen_thread.error.connect(lambda e: self.add_system_message(f"Schema generation error: {e}"))
        self._gen_thread.finished.connect(lambda: self.setCursor(Qt.CursorShape.ArrowCursor))
        self._gen_thread.start()

    def _start_read_query(self, prompt: str):
        if not getattr(self, 'ollama', None):
            return
        self.setCursor(Qt.CursorShape.WaitCursor)
        self._read_thread = ReadQueryThread(self.ollama, prompt)
        self._read_thread.succeeded.connect(self._render_read_result)
        self._read_thread.error.connect(lambda e: self.add_system_message(f"Query error: {e}"))
        self._read_thread.finished.connect(lambda: self.setCursor(Qt.CursorShape.ArrowCursor))
        self._read_thread.start()

    def _on_ops_generated(self, operations: List[str], rationale: str):
        if not operations:
            self.add_system_message("No SQL operations were generated.")
            return

        table_hint = self.table_selector.currentText() if hasattr(self, 'table_selector') else None
        if not table_hint or table_hint.startswith("("):
            self.add_system_message("⚠️ No valid table selected")
            return

        try:
            if not db_manager.validate_table_exists(table_hint):
                self.add_system_message(f"⚠️ Table does not exist: {table_hint}")
                return
        except Exception:
            pass

        # Log and execute immediately (skip preview/confirm)
        self.add_system_message(f"Executing {len(operations)} operation(s) on {table_hint}...")
        self._execute_operations(operations, table_hint)

    def _execute_operations(self, operations: List[str], table_hint: str):
        if not getattr(self, 'ollama', None):
            return
        self.setCursor(Qt.CursorShape.WaitCursor)
        self._exec_thread = SchemaExecThread(self.ollama, operations, table_hint, request_text=None)
        self._exec_thread.succeeded.connect(self._on_mutation_complete)
        self._exec_thread.error.connect(lambda e: self.add_system_message(f"Execution error: {e}"))
        self._exec_thread.finished.connect(lambda: self.setCursor(Qt.CursorShape.ArrowCursor))
        self._exec_thread.start()

    def _on_mutation_complete(self, result: Dict[str, Any]):
        self.add_system_message("✓ Changes applied successfully.")
        try:
            if not hasattr(self, 'operation_history'):
                self.operation_history = []
            self.operation_history.append({
                'timestamp': datetime.now(),
                'operations': result.get('operations', []),
                'snapshot': result.get('snapshot'),
                'table': self.table_selector.currentText() if hasattr(self, 'table_selector') else None,
            })
        except Exception:
            pass

        # Auto-reload immediately
        self._reload_current_table()

    def _reload_current_table(self):
        table_name = self.table_selector.currentText() if hasattr(self, 'table_selector') else None
        if not table_name or table_name.startswith("("):
            return
        try:
            rows, total = db_manager.fetch_display_data(table_name, limit=1000, market_hours_only=False)
            import pandas as pd
            df = pd.DataFrame(rows, columns=[
                "market_date", "market_time", "open", "high", "low", "close", "volume", "vwap"
            ])
            # TODO: Replace with your real table widget integration
            self.add_system_message(f"Reloaded {table_name} (showing {len(df)} rows, total {total}).")
        except Exception as exc:
            self.add_system_message(f"Reload failed: {exc}")

    def _render_read_result(self, result: Dict[str, Any]):
        table = result.get("table_update")
        chart = result.get("chart_update")

        if table:
            import pandas as pd
            rows = table.get("rows", [])
            df = pd.DataFrame(rows)
            # TODO: Replace with your real table widget integration
            self.add_system_message(
                f"Loaded {table.get('total_rows', len(rows))} rows (page {table.get('page', 1)})."
            )

        if chart:
            # TODO: Map to chart widget
            pass