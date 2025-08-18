"""
Right Pane Component - Conversational AI

Contains AI chat interface for strategy development.
Save as: k2_quant/pages/analysis/components/right_pane.py
"""

from typing import Dict, Any, Optional
from datetime import datetime

from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QLineEdit, QPushButton, QLabel, QWidget)
from PyQt6.QtCore import Qt, pyqtSignal

from k2_quant.utilities.logger import k2_logger


class RightPaneWidget(QFrame):
    """Right pane with AI chat interface"""
    
    # Signals
    message_sent = pyqtSignal(str)  # message
    strategy_generated = pyqtSignal(str, str)  # name, code
    projection_requested = pyqtSignal(dict)  # parameters
    
    def __init__(self):
        super().__init__()
        self.setFixedWidth(380)
        self.setObjectName("rightPane")
        
        self.current_context = None
        self.conversation_history = []
        
        self.init_ui()
        self.setup_styling()
        self.show_welcome_message()
    
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
    
    def show_welcome_message(self):
        """Show welcome message"""
        self.chat_display.append("AI: Hello! I can help you analyze stock data and create custom strategies. Load a model to get started.")
    
    def send_ai_message(self):
        """Send message to AI"""
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
        
        # Emit signal
        self.message_sent.emit(message)
        
        # Simulate AI response (in full implementation, would call AI service)
        self.process_ai_response(message)
    
    def process_ai_response(self, message: str):
        """Process and display AI response"""
        # Simulate different responses based on keywords
        response = ""
        
        if "create strategy" in message.lower():
            response = "I'll help you create a custom trading strategy. What conditions would you like to use?"
        elif "projection" in message.lower():
            response = "I can create price projections based on historical patterns. What time frame are you interested in?"
            self.projection_requested.emit({'timeframe': 30})
        elif "elasticity" in message.lower():
            response = "Creating an elasticity strategy that calculates (high-low)/low*100..."
            # Simulate code generation
            code = """def elasticity_strategy(df):
    df['elasticity'] = (df['high'] - df['low']) / df['low'] * 100
    return df"""
            self.strategy_generated.emit("Elasticity Strategy", code)
        elif self.current_context:
            response = f"I understand you want to analyze the {self.current_context.get('symbol', 'data')}. Let me help you with that..."
        else:
            response = "Please load a model first so I can help you analyze the data."
        
        # Display response
        self.chat_display.append(f"\nAI: {response}")
        
        # Add to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
    
    def set_data_context(self, context: Dict[str, Any]):
        """Set the data context for AI"""
        self.current_context = context
        
        # Show in chat
        symbol = context.get('symbol', 'Unknown')
        records = context.get('records', 0)
        self.chat_display.append(f"\nAI: Loaded {symbol} with {records:,} records. How can I help you analyze this data?")
    
    def clear_chat(self):
        """Clear chat history"""
        self.chat_display.clear()
        self.conversation_history.clear()
        self.show_welcome_message()
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
        self.clear_chat()
        self.current_context = None