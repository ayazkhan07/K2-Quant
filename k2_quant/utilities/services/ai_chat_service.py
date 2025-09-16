from __future__ import annotations

import time
import traceback
from typing import Dict, List, Optional, Generator, Any

# Optional SDKs (graceful if missing)
try:
	from anthropic import Anthropic
except Exception:
	Anthropic = None  # type: ignore

try:
	from openai import OpenAI
except Exception:
	OpenAI = None  # type: ignore

# Optional local client
try:
	import ollama
except Exception:
	ollama = None  # type: ignore

from k2_quant.utilities.config.api_config import api_config
from k2_quant.utilities.logger import k2_logger


class AIChatService:
	"""
	Minimal provider-agnostic, streaming AI chat service.

	- Conversational only (no intents/tags, no NL→SQL).
	- Keeps a short rolling history of recent turns.
	- Streams via Anthropic/OpenAI if configured or local Ollama.
	"""

	MAX_HISTORY_TURNS = 8

	def __init__(self) -> None:
		# Provider/model
		self.provider: str = "local"  # default to local if available
		self.model: str = "llama3.2:3b-instruct-q8_0"

		# Persona/system prompt (concise and helpful, no tags)
		self.system_prompt: str = (
			"You are a helpful quantitative analysis assistant. "
			"Be concise and ask clarifying questions when needed. "
			"Do not invent timestamps or data that are not provided. "
			"Default OHLC aggregation uses the close."
		)

		# Conversation memory
		self.history: List[Dict[str, str]] = []  # [{role, content}]

		# Clients (lazy-init where possible)
		self._anthropic: Optional[Any] = None
		self._openai_client: Optional[Any] = None
		self._ollama_base_url: str = "http://localhost:11434"

		# Initialize cloud clients if keys exist
		self._init_anthropic()
		self._init_openai()

	# ---------- Public API ----------

	def set_provider(self, provider: str) -> None:
		val = (provider or "").strip().lower()
		if val not in {"anthropic", "openai", "local"}:
			val = "local"
		self.provider = val
		k2_logger.info(f"Provider set to {self.provider}", "AI_CHAT")

	def set_model(self, model: str) -> None:
		self.model = (model or "").strip()
		k2_logger.info(f"Model set to {self.model}", "AI_CHAT")

	def set_system_context(self, text: str) -> None:
		"""Optional: allow callers to set a lightweight system hint."""
		self.system_prompt = text or self.system_prompt

	def set_dataset_context(
		self,
		meta: Dict[str, Any],
		columns: List[str],
		recent_rows: List[Dict[str, Any]],
		quick_stats: Optional[Dict[str, Any]] = None,
		exchange: str = "NYSE",
		tz: str = "US/Eastern",
	) -> None:
		"""No-op in simplified conversational mode (kept for compatibility)."""
		return

	def get_streaming_response(self, message: str) -> Generator[str, None, None]:
		"""Stream a response based on the active provider."""
		user_msg = (message or "").strip()
		if not user_msg:
			return

		start_time = time.time()
		assistant_accum: List[str] = []
		try:
			messages = self._build_messages(user_msg)
			system_prompt = self.system_prompt

			if self.provider == "anthropic" and self._anthropic is not None:
				# Anthropic streaming
				for delta in self._stream_anthropic(messages, system_prompt):
					assistant_accum.append(delta)
					yield delta

			elif self.provider == "openai" and self._openai_client is not None:
				# OpenAI streaming
				for delta in self._stream_openai(messages, system_prompt):
					assistant_accum.append(delta)
					yield delta

			elif self.provider == "local" and ollama is not None:
				# Local Ollama streaming
				client = ollama.Client(host=self._ollama_base_url)
				stream = client.chat(
					model=self.model or "llama3.2:3b-instruct-q8_0",
					messages=[
						{"role": "system", "content": system_prompt},
						*messages,
					],
					stream=True,
				)
				for chunk in stream:
					content = (chunk.get("message", {}) or {}).get("content", "")
					if content:
						assistant_accum.append(content)
						yield content
			else:
				warn = "Local AI service not available and no cloud keys configured."
				k2_logger.warning(warn, "AI_CHAT")
				yield warn

		except Exception as e:
			err = f"[ERROR] {type(e).__name__}: {str(e)}"
			k2_logger.error(f"{err}\n{traceback.format_exc()}", "AI_CHAT")
			yield err
		finally:
			# Persist both user and assistant messages to a small rolling history
			self._append_history("user", user_msg)
			final_text = "".join(assistant_accum)
			if final_text:
				self._append_history("assistant", final_text)
			elapsed = round(time.time() - start_time, 3)
			k2_logger.performance_metric("AI response time", elapsed, "seconds")

	def clear_history(self) -> None:
		self.history.clear()
		k2_logger.info("AI chat history cleared", "AI_CHAT")

	def get_available_providers(self) -> List[str]:
		available = []
		if self._anthropic is not None:
			available.append("anthropic")
		if self._openai_client is not None:
			available.append("openai")
		available.append("local")
		return sorted(set(available))

	def export_conversation(self) -> List[Dict[str, str]]:
		return self.history.copy()

	# ---------- Internals ----------

	def _init_anthropic(self) -> None:
		if Anthropic is None or not api_config.anthropic_api_key:
			self._anthropic = None
			return
		self._anthropic = Anthropic(api_key=api_config.anthropic_api_key)

	def _init_openai(self) -> None:
		if OpenAI is None or not api_config.openai_api_key:
			self._openai_client = None
			return
		self._openai_client = OpenAI(api_key=api_config.openai_api_key)

	def _build_messages(self, new_user_message: str) -> List[Dict[str, str]]:
		trimmed = self.history[-self.MAX_HISTORY_TURNS :]
		return trimmed + [{"role": "user", "content": new_user_message}]

	def _append_history(self, role: str, content: str) -> None:
		if not content:
			return
		self.history.append({"role": role, "content": content})
		if len(self.history) > self.MAX_HISTORY_TURNS:
			self.history = self.history[-self.MAX_HISTORY_TURNS :]

	# ---------- Provider streams ----------

	def _stream_anthropic(self, messages: List[Dict[str, str]], system_prompt: str) -> Generator[str, None, None]:
		if self._anthropic is None:
			yield "Anthropic client not initialized."
			return

		assistant_accum: List[str] = []
		for attempt in range(2):
			try:
				with self._anthropic.messages.stream(  # type: ignore[attr-defined]
						model=self.model or "claude-3-5-sonnet-20241022",
						system=system_prompt,
						messages=[{"role": m["role"], "content": m["content"]} for m in messages],
						max_tokens=4000,
					) as stream:
					for delta in stream.text_stream:
						assistant_accum.append(delta)
						yield delta
				break
			except Exception as e:
				if "rate" in str(e).lower() and attempt == 0:
					msg = "[INFO] Rate limited. Retrying in 5s..."
					k2_logger.warning(msg, "AI_CHAT")
					yield msg
					time.sleep(5)
					continue
				raise

	def _stream_openai(self, messages: List[Dict[str, str]], system_prompt: str) -> Generator[str, None, None]:
		if self._openai_client is None:
			yield "OpenAI client not initialized."
			return

		chat_messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
		chat_messages.extend(messages)

		for attempt in range(2):
			try:
				stream = self._openai_client.chat.completions.create(
					model=self.model or "gpt-4o",
					messages=chat_messages,
					stream=True,
				)
				for chunk in stream:
					delta = getattr(chunk.choices[0].delta, "content", None)
					if delta:
						yield delta
				break
			except Exception as e:
				if "rate" in str(e).lower() and attempt == 0:
					msg = "[INFO] Rate limited. Retrying in 5s..."
					k2_logger.warning(msg, "AI_CHAT")
					yield msg
					time.sleep(5)
					continue
				raise


# Singleton instance
ai_chat_service = AIChatService()


