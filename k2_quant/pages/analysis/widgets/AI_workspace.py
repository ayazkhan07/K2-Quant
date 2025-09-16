"""
AI Chat Widget for K2 Quant Analysis

Minimal conversational chat UI with streaming responses.
No intents/tags, no NL→SQL, no code-block execution.
"""

from typing import Optional
from datetime import datetime

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
						 QLineEdit, QPushButton, QLabel, QComboBox,
						 QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QTextCursor, QFont, QTextCharFormat, QColor

from k2_quant.utilities.logger import k2_logger


class AIStreamThread(QThread):
	"""Thread for streaming AI responses"""

	text_chunk = pyqtSignal(str)
	complete = pyqtSignal()
	error = pyqtSignal(str)

	def __init__(self, message: str):
		super().__init__()
		self.message = message

	def run(self):
		"""Stream AI response"""
		try:
			try:
				from k2_quant.utilities.services.ai_chat_service import ai_chat_service
				if not ai_chat_service:
					self.error.emit("AI service not available")
					return
				for chunk in ai_chat_service.get_streaming_response(self.message):
					self.text_chunk.emit(chunk)
			except ImportError:
				self.text_chunk.emit("AI service not configured. Please set up API keys or local model.")
			self.complete.emit()
		except Exception as e:
			self.error.emit(str(e))


class AIChatWidget(QWidget):
	"""Conversational AI chat widget with streaming support"""

	# Signals
	message_sent = pyqtSignal(str)

	def __init__(self, parent=None):
		super().__init__(parent)

		self.ai_thread: Optional[AIStreamThread] = None
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

		# Clear button
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
			models = ["gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo"]
		elif provider == "anthropic":
			models = [
				"claude-3-5-sonnet-20241022",
				"claude-3-opus-20240229",
				"claude-3-haiku-20240307"
			]
		elif provider == "local":
			models = []
			try:
				import ollama
				client = ollama.Client(host="http://localhost:11434")
				response = client.list()
				try:
					model_list = response.get("models")
				except Exception:
					model_list = getattr(response, "models", None)
				if not model_list:
					model_list = []
				names = []
				for m in model_list:
					name = getattr(m, "model", None) or getattr(m, "name", None)
					if name is None and isinstance(m, dict):
						name = m.get("model") or m.get("name")
					if name:
						names.append(name)
				# Filter out heavy 70B models by default
				names = [n for n in names if "70b" not in n.lower()]
				models = names or ["(Ollama: no models found)"]
			except Exception as exc:
				models = [f"(Ollama error: {exc})"]
		else:
			models = []

		self.model_selector.clear()
		self.model_selector.addItems(models)
		if self.model_selector.count() > 0 and "70b" in self.model_selector.currentText().lower():
			self.model_selector.setCurrentIndex(0)

	def on_provider_changed(self, provider: str):
		"""Handle provider change"""
		self.update_model_list()
		if self.model_selector.count() > 0:
			self.model_selector.setCurrentIndex(0)
		try:
			from k2_quant.utilities.services.ai_chat_service import ai_chat_service
			if ai_chat_service:
				ai_chat_service.set_provider(provider.lower())
				if self.model_selector.currentText():
					ai_chat_service.set_model(self.model_selector.currentText())
				k2_logger.info(f"Provider changed: {provider}/{self.model_selector.currentText()}", "AI_CHAT")
		except ImportError:
			k2_logger.debug("AI service not available", "AI_CHAT")

	def on_model_changed(self, model: str):
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
		welcome = """<div style='color: #fff; font-style: italic;'>
		Welcome to K2 Quant AI Assistant!<br><br>
		I can help you:
		<ul>
		<li>Create custom trading strategies</li>
		<li>Generate price projections</li>
		<li>Analyze patterns in your data</li>
		<li>Write Python code for complex calculations</li>
		</ul>
		Ask a question or describe what you'd like to analyze.
		</div>"""
		self.chat_display.setHtml(welcome)

	def set_data_context(self, data, metadata):
		"""No-op in simplified mode; kept for compatibility."""
		return

	def send_message(self):
		"""Send message to AI"""
		message = self.input_field.text().strip()
		if not message or self.is_streaming:
			return

		# Show user message and start streaming assistant text
		self.add_user_message(message)
		self.input_field.clear()
		self.message_sent.emit(message)
		self.start_streaming(message)

	def start_streaming(self, message: str):
		"""Start streaming AI response"""
		self.is_streaming = True

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

	def append_ai_text(self, chunk: str):
		"""Append streamed text to AI message"""
		cursor = self.chat_display.textCursor()
		cursor.movePosition(QTextCursor.MoveOperation.End)
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

		k2_logger.info("AI streaming complete", "AI_CHAT")

	def streaming_error(self, error: str):
		"""Handle streaming error"""
		self.is_streaming = False

		# Re-enable input
		self.input_field.setEnabled(True)
		self.send_btn.setEnabled(True)

		# Hide streaming indicator
		self.streaming_indicator.hide()

		# Show error
		self.add_system_message(f"Error: {error}")
		k2_logger.error(f"AI streaming error: {error}", "AI_CHAT")

	def add_user_message(self, message: str):
		"""Add user message to chat"""
		cursor = self.chat_display.textCursor()
		cursor.movePosition(QTextCursor.MoveOperation.End)

		# Add spacing
		cursor.insertText("\n\n")

		# Add user label (white, non-bold)
		fmt = QTextCharFormat()
		fmt.setForeground(QColor("#fff"))
		fmt.setFontWeight(QFont.Weight.Normal)
		cursor.setCharFormat(fmt)
		cursor.insertText("YOU: ")

		# Add message (white, non-bold)
		fmt.setForeground(QColor("#fff"))
		fmt.setFontWeight(QFont.Weight.Normal)
		cursor.setCharFormat(fmt)
		cursor.insertText(message)

		# Auto-scroll
		scrollbar = self.chat_display.verticalScrollBar()
		scrollbar.setValue(scrollbar.maximum())

	def add_ai_message(self, message: str):
		"""Add AI message to chat"""
		cursor = self.chat_display.textCursor()
		cursor.movePosition(QTextCursor.MoveOperation.End)

		# Add spacing
		cursor.insertText("\n\n")

		# Add AI label (white, non-bold)
		fmt = QTextCharFormat()
		fmt.setForeground(QColor("#fff"))
		fmt.setFontWeight(QFont.Weight.Normal)
		cursor.setCharFormat(fmt)
		cursor.insertText("AI: ")

		# Add message (white, non-bold)
		fmt.setForeground(QColor("#fff"))
		fmt.setFontWeight(QFont.Weight.Normal)
		cursor.setCharFormat(fmt)
		if message:
			cursor.insertText(message)

		# Auto-scroll
		scrollbar = self.chat_display.verticalScrollBar()
		scrollbar.setValue(scrollbar.maximum())

	def add_system_message(self, message: str):
		"""Add system message to chat"""
		cursor = self.chat_display.textCursor()
		cursor.movePosition(QTextCursor.MoveOperation.End)

		# Add spacing
		cursor.insertText("\n\n")

		# Add message (white, italic, non-bold)
		fmt = QTextCharFormat()
		fmt.setForeground(QColor("#fff"))
		fmt.setFontItalic(True)
		cursor.setCharFormat(fmt)
		cursor.insertText(message)

		# Auto-scroll
		scrollbar = self.chat_display.verticalScrollBar()
		scrollbar.setValue(scrollbar.maximum())

	def clear_chat(self):
		"""Clear chat history"""
		self.chat_display.clear()
		try:
			from k2_quant.utilities.services.ai_chat_service import ai_chat_service
			if ai_chat_service:
				ai_chat_service.clear_history()
		except ImportError:
			pass
		self.show_welcome_message()
		k2_logger.info("Chat cleared", "AI_CHAT")

	def setup_styling(self):
		"""Apply styling to the widget"""
		self.setStyleSheet("""
			#chatDisplay {
				background-color: #0a0a0a;
				color: #fff;
				border: 1px solid #1a1a1a;
				border-radius: 4px;
				padding: 10px;
				font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif;
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
				font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif;
				font-weight: normal;
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
				font-weight: normal;
				font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif;
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
				color: #fff;
				border: 1px solid #2a2a2a;
				padding: 3px 8px;
				border-radius: 2px;
				font-size: 10px;
				font-weight: normal;
				font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif;
			}

			#clearChatBtn:hover {
				background-color: #1a1a1a;
				color: #fff;
			}

			#providerSelector, #modelSelector {
				background-color: #1a1a1a;
				color: #fff;
				border: 1px solid #2a2a2a;
				padding: 4px;
				border-radius: 3px;
				font-size: 11px;
				font-weight: normal;
				font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif;
			}

			QComboBox::drop-down { border: none; }
			QComboBox::down-arrow { image: none; }
			QProgressBar { background-color: #1a1a1a; border: none; }
			QProgressBar::chunk { background-color: #4aa; }
		""")

	def cleanup(self):
		"""Cleanup resources"""
		if self.ai_thread and self.ai_thread.isRunning():
			self.ai_thread.terminate()
			self.ai_thread.wait()