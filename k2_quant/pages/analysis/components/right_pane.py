"""
Right Pane Component - Conversational AI

Contains AI chat interface for strategy development.
Save as: k2_quant/pages/analysis/components/right_pane.py
"""

from typing import Dict, Any

from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QLabel)
from PyQt6.QtCore import pyqtSignal

from k2_quant.utilities.logger import k2_logger
from k2_quant.pages.analysis.widgets.AI_workspace import AIChatWidget


class RightPaneWidget(QFrame):
	"""Right pane with AI chat interface"""

	# Signals
	message_sent = pyqtSignal(str)  # message

	def __init__(self):
		super().__init__()
		self.setFixedWidth(513)
		self.setObjectName("rightPane")

		self.current_context = None

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

		# Embedded AI chat widget (streams via AIChatService)
		self.chat = AIChatWidget(self)
		layout.addWidget(self.chat)

		# Wire widget signals to pane-level signals/handlers
		self._wire_widget_signals()

	def show_welcome_message(self):
		"""Show welcome message"""
		# Delegate welcome rendering to the embedded widget
		pass

	def _wire_widget_signals(self):
		"""Bridge widget signals to existing right pane signals/handlers."""
		# Mirror user messages if higher-level code is listening
		self.chat.message_sent.connect(self.message_sent.emit)

	def set_data_context(self, context: Dict[str, Any]):
		"""Set the data context for AI (optional in simplified mode)"""
		self.current_context = context or {}

	def clear_chat(self):
		"""Clear chat history"""
		self.chat.clear_chat()
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
		""")

	def cleanup(self):
		"""Cleanup resources"""
		self.clear_chat()
		self.current_context = None