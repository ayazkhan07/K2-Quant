"""
AI Chat Widget for K2 Quant Analysis

Provides AI chat interface with streaming responses for strategy development.
Save as: k2_quant/pages/analysis/widgets/ai_chat_widget.py
"""

import json
from typing import Dict, Any, Optional, List, Generator
from datetime import datetime

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QLineEdit, QPushButton, QLabel, QComboBox,
                             QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QTextCursor, QFont, QTextCharFormat, QColor

from k2_quant.utilities.logger import k2_logger


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
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_context = None
        self.conversation_history = []
        self.ai_thread = None
        self.current_code_block = ""
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
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Provider and Model selector
        model_selector = self.create_model_selector()
        layout.addWidget(model_selector)
        
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
        
        # Quick actions
        quick_actions = self.create_quick_actions()
        layout.addWidget(quick_actions)
        
        # Input area
        input_widget = self.create_input_area()
        layout.addWidget(input_widget)
    
    def create_header(self):
        """Create header widget"""
        header = QWidget()
        header.setFixedHeight(30)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        header.setLayout(layout)
        
        label = QLabel("AI ASSISTANT")
        label.setObjectName("headerLabel")
        layout.addWidget(label)
        
        layout.addStretch()
        
        # Clear chat button
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(50)
        clear_btn.clicked.connect(self.clear_chat)
        clear_btn.setObjectName("clearChatBtn")
        layout.addWidget(clear_btn)
        
        return header
    
    def create_model_selector(self):
        """Create model selector widget"""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)
        
        # Provider selector
        layout.addWidget(QLabel("Provider:"))
        
        self.provider_selector = QComboBox()
        self.provider_selector.addItems(["OpenAI", "Anthropic", "Local"])
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
        
        layout.addStretch()
        
        return widget
    
    def create_quick_actions(self):
        """Create quick action buttons"""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 5)
        widget.setLayout(layout)
        
        # Quick prompts
        elasticity_btn = QPushButton("📊 Elasticity")
        elasticity_btn.setToolTip("Generate elasticity strategy")
        elasticity_btn.clicked.connect(lambda: self.send_quick_prompt("elasticity"))
        elasticity_btn.setObjectName("quickBtn")
        layout.addWidget(elasticity_btn)
        
        projection_btn = QPushButton("📈 Projection")
        projection_btn.setToolTip("Create price projections")
        projection_btn.clicked.connect(lambda: self.send_quick_prompt("projection"))
        projection_btn.setObjectName("quickBtn")
        layout.addWidget(projection_btn)
        
        pattern_btn = QPushButton("🔍 Pattern")
        pattern_btn.setToolTip("Find pattern matches")
        pattern_btn.clicked.connect(lambda: self.send_quick_prompt("pattern"))
        pattern_btn.setObjectName("quickBtn")
        layout.addWidget(pattern_btn)
        
        layout.addStretch()
        
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
            models = ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"]
        elif provider == "anthropic":
            models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]
        elif provider == "local":
            models = ["No models (simulated)"]
        else:
            models = []
        
        self.model_selector.clear()
        self.model_selector.addItems(models)
    
    def on_provider_changed(self, provider):
        """Handle provider change"""
        self.update_model_list()
        
        # Try to set provider in AI service
        try:
            from k2_quant.utilities.services.ai_chat_service import ai_chat_service
            if ai_chat_service:
                ai_chat_service.set_provider(provider.lower())
                k2_logger.info(f"AI provider changed to {provider}", "AI_CHAT")
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
    
    def set_data_context(self, data, metadata):
        """Set the data context for AI"""
        self.current_context = {
            'metadata': metadata,
            'data_shape': data.shape if hasattr(data, 'shape') else len(data),
            'columns': list(data.columns) if hasattr(data, 'columns') else None
        }
        
        # Update AI service context
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
        except ImportError:
            pass
        
        # Show in chat
        records = self.current_context['data_shape'][0] if isinstance(self.current_context['data_shape'], tuple) else self.current_context['data_shape']
        self.add_system_message(f"📊 Loaded {metadata.get('symbol', 'Unknown')} data with {records} records")
    
    def send_quick_prompt(self, prompt_type):
        """Send a quick prompt based on type"""
        prompts = {
            'elasticity': "Create an elasticity strategy that calculates (high-low)/low*100, finds patterns within 5% tolerance, and projects 10 days forward",
            'projection': "Generate a 30-day price projection based on historical patterns and trends",
            'pattern': "Find repeating patterns in the price data and identify similar historical movements"
        }
        
        if prompt_type in prompts:
            self.input_field.setText(prompts[prompt_type])
            self.send_message()
    
    def send_message(self):
        """Send message to AI"""
        message = self.input_field.text().strip()
        if not message or self.is_streaming:
            return
        
        # Add user message to display
        self.add_user_message(message)
        
        # Clear input
        self.input_field.clear()
        
        # Emit signal
        self.message_sent.emit(message)
        
        # Start streaming
        self.start_streaming(message)
    
    def start_streaming(self, message):
        """Start streaming AI response"""
        self.is_streaming = True
        self.current_code_block = ""
        
        # Disable input
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        
        # Show streaming indicator
        self.streaming_indicator.show()
        
        # Add AI message placeholder
        self.add_ai_message("")
        
        # Start thread
        self.ai_thread = AIStreamThread(message)
        self.ai_thread.text_chunk.connect(self.append_ai_text)
        self.ai_thread.complete.connect(self.streaming_complete)
        self.ai_thread.error.connect(self.streaming_error)
        self.ai_thread.start()
    
    def append_ai_text(self, chunk):
        """Append streamed text to AI message"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
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
        
        # Process any code blocks
        if self.current_code_block:
            self.process_code_block(self.current_code_block)
        
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
        # Extract Python code
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
                
                # Show code in formatted block
                self.add_code_block(code)
                
                # Ask if user wants to execute
                self.add_system_message("📝 Code generated. Click 'Execute' to run this strategy.")
                
                # Store for execution
                self.current_code_block = code
                
                # Emit signal
                self.code_generated.emit(code, "AI Generated Strategy")
    
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
            #headerLabel {
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #999;
                font-weight: 600;
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
            
            #quickBtn {
                background-color: #1a1a1a;
                color: #888;
                border: 1px solid #2a2a2a;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 11px;
            }
            
            #quickBtn:hover {
                background-color: #2a2a2a;
                color: #fff;
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