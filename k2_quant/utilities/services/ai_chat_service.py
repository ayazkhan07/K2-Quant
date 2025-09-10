from __future__ import annotations

import time
import traceback
from typing import Dict, List, Optional, Generator, Any
from datetime import datetime

# Optional SDKs (graceful if missing)
try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None  # type: ignore

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# Optional tokenizers (accurate counting if present)
try:
    import tiktoken  # OpenAI tokenizer
except Exception:
    tiktoken = None  # type: ignore

from k2_quant.utilities.config.api_config import api_config
from k2_quant.utilities.logger import k2_logger


class AIChatService:
    """
    Provider-agnostic, streaming AI chat service for K2 Quant.

    UI contract:
      - set_provider(str)
      - set_model(str)
      - set_system_context(str)
      - set_dataset_context(meta: dict, columns: list[str], recent_rows: list[dict], quick_stats: dict | None = None, exchange: str = "NYSE", tz: str = "US/Eastern")
      - get_streaming_response(message: str) -> Generator[str, None, None]
      - clear_history()
      - get_available_providers() -> list[str]
      - export_conversation() -> list[dict]
    """

    # Rolling history and token window management
    MAX_HISTORY_TURNS = 8  # keep recent turns compact
    DEFAULT_PROVIDER = "anthropic"
    DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
    DEFAULT_OPENAI_MODEL = "gpt-4-turbo-preview"

    # Model context windows (approx tokens)
    CONTEXT_WINDOWS = {
        "anthropic": 200000,  # Claude 3 family large windows
        "openai": 128000,     # GPT-4 Turbo
    }

    def __init__(self) -> None:
        # Provider/model
        self.provider: str = self.DEFAULT_PROVIDER
        self.model: str = self.DEFAULT_ANTHROPIC_MODEL

        # Persona/system prompt and dataset card
        self.system_prompt: str = self._default_system_prompt()
        self.dataset_context: Dict[str, Any] = {}
        self.environment_card: str = ""

        # Conversation memory
        self.history: List[Dict[str, str]] = []  # [{role, content}]
        self.summary_memory: str = ""  # summarized older turns

        # Token accounting
        self.total_tokens: int = 0
        self.refresh_threshold: float = 0.8

        # Clients
        self._anthropic: Optional[Any] = None
        self._openai_client: Optional[Any] = None
        self._initialize_clients()

        # Build initial environment card
        self._rebuild_environment_card()

    # ---------- Public API ----------

    def set_provider(self, provider: str) -> None:
        val = (provider or "").strip().lower()
        if val not in {"anthropic", "openai", "local"}:
            val = "anthropic"
        self.provider = val

        # Ensure the selected model matches the provider family
        if self.provider == "openai":
            if not self.model or "claude" in (self.model or "").lower():
                self.model = self.DEFAULT_OPENAI_MODEL
            if self._openai_client is None:
                self._init_openai()
        elif self.provider == "anthropic":
            if not self.model or (self.model or "").lower().startswith("gpt"):
                self.model = self.DEFAULT_ANTHROPIC_MODEL
            if self._anthropic is None:
                self._init_anthropic()

        k2_logger.info(f"Provider: {self.provider}, Model: {self.model}", "AI_CHAT")

    def set_model(self, model: str) -> None:
        self.model = (model or "").strip()
        k2_logger.info(f"AI model set to {self.model}", "AI_CHAT")

    def set_system_context(self, text: str) -> None:
        self.system_prompt = text or ""
        self._rebuild_environment_card()

    def set_dataset_context(
        self,
        meta: Dict[str, Any],
        columns: List[str],
        recent_rows: List[Dict[str, Any]],
        quick_stats: Optional[Dict[str, Any]] = None,
        exchange: str = "NYSE",
        tz: str = "US/Eastern",
    ) -> None:
        # Compact recent sample (<= 10 rows, key fields if available)
        sample = recent_rows or []
        if len(sample) > 10:
            sample = sample[-10:]

        self.dataset_context = {
            "metadata": meta or {},
            "columns": columns or [],
            "recent_sample": sample,
            "quick_stats": quick_stats or {},
            "exchange": exchange,
            "timezone": tz,
            "last_updated": datetime.now().isoformat(),
        }
        self._rebuild_environment_card()
        k2_logger.info(
            f"Dataset context set: {meta.get('symbol', 'UNKNOWN')} "
            f"rows={meta.get('total_records')} range={meta.get('date_range')}",
            "AI_CHAT",
        )

    def get_streaming_response(self, message: str) -> Generator[str, None, None]:
        """
        Stream a response in chunks. Handles provider selection, token management,
        and appends to history on completion.
        """
        user_msg = (message or "").strip()
        if not user_msg:
            return

        start_time = time.time()
        try:
            sys_prompt = self.environment_card or self.system_prompt
            messages = self._build_messages(user_msg)

            # Pre-send token check (approx)
            self._maybe_refresh_context(sys_prompt, messages)

            if self.provider == "anthropic" and self._anthropic is not None:
                yield from self._stream_anthropic(messages, sys_prompt)
            elif self.provider == "openai" and self._openai_client is not None:
                yield from self._stream_openai(messages, sys_prompt)
            else:
                warn = "[LOCAL] AI service not configured. Please set API keys."
                k2_logger.warning(warn, "AI_CHAT")
                yield warn

        except Exception as e:
            err = f"[ERROR] {type(e).__name__}: {str(e)}"
            k2_logger.error(f"{err}\n{traceback.format_exc()}", "AI_CHAT")
            yield err
        finally:
            self._append_history("user", user_msg)
            elapsed = round(time.time() - start_time, 3)
            k2_logger.performance_metric("AI response time", elapsed, "seconds")

    def clear_history(self) -> None:
        self.history.clear()
        self.summary_memory = ""
        k2_logger.info("AI chat history cleared", "AI_CHAT")

    def get_available_providers(self) -> List[str]:
        available = []
        if self._anthropic is not None:
            available.append("anthropic")
        if self._openai_client is not None:
            available.append("openai")
        if not available:
            available.append("local")
        return available

    def export_conversation(self) -> List[Dict[str, str]]:
        return self.history.copy()

    # ---------- Internals ----------

    def _initialize_clients(self) -> None:
        self._init_anthropic()
        self._init_openai()

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

    def _default_system_prompt(self) -> str:
        return (
            "You are K2 Quant's conversational strategy development partner.\n"
            "Style: explore → refine → design → confirm → create; ask at most two clarifying questions before proposing a plan.\n"
            "Use tags as the first token: [DIRECT] | [QUERY] | [STRATEGY] | [DECLINE].\n"
            "Rules: no synthetic timestamps; use exchange trading sessions (default NYSE). Default OHLC aggregation uses close.\n"
            "No file/network/OS access; produce a single ```python block``` when outputting code.\n"
            "QUERY: small data lookups (≤1000 rows), auto-run by app. Return dict for single points or a small table for ranges.\n"
            "STRATEGY: projections/patterns; include projection_mid/low/high and is_projection=True; code only after confirmation.\n"
        )

    def _rebuild_environment_card(self) -> None:
        ctx = self.dataset_context or {}
        meta = ctx.get("metadata", {})
        cols = ctx.get("columns", [])
        sample = ctx.get("recent_sample", [])
        qstats = ctx.get("quick_stats", {})
        exchange = ctx.get("exchange", "NYSE")
        tz = ctx.get("timezone", "US/Eastern")

        parts: List[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt.strip())

        if meta or cols or sample:
            parts.append("ENVIRONMENT CARD:")
            if meta:
                parts.append(
                    f"- Symbol: {meta.get('symbol', 'UNKNOWN')} | "
                    f"Records: {meta.get('total_records', 'UNKNOWN')} | "
                    f"Date range: {meta.get('date_range', 'UNKNOWN')}"
                )
            if cols:
                parts.append(f"- Columns: {', '.join(cols)}")
            if qstats:
                parts.append(f"- Quick stats: {qstats}")
            parts.append(f"- Exchange: {exchange} | Timezone: {tz}")
            if sample:
                parts.append(f"- Recent sample (last {len(sample)} rows): {sample}")

        # Compact summary memory if present
        if self.summary_memory:
            parts.append("CONVERSATION SUMMARY:")
            parts.append(self.summary_memory)

        # Operating rules (concise, reasserted)
        parts.append(
            "OPERATING RULES:\n"
            "- Tags: [DIRECT] | [QUERY] | [STRATEGY] | [DECLINE]\n"
            "- No synthetic timestamps; default NYSE calendar; default aggregation uses close\n"
            "- No file/network/OS access; one ```python block``` when coding\n"
            "- QUERY ≤1000 rows; STRATEGY ≤10000 rows; suggest narrowing if limits exceeded\n"
            "- Projections: projection_mid/low/high + is_projection=True"
        )

        self.environment_card = "\n".join(parts)

    def _build_messages(self, new_user_message: str) -> List[Dict[str, str]]:
        trimmed = self.history[-self.MAX_HISTORY_TURNS :]
        return trimmed + [{"role": "user", "content": new_user_message}]

    # ---------- Token management ----------

    def _maybe_refresh_context(self, sys_prompt: str, messages: List[Dict[str, str]]) -> None:
        approx_tokens = self._estimate_tokens(sys_prompt) + sum(self._estimate_tokens(m["content"]) for m in messages)
        self.total_tokens += approx_tokens
        if self._should_refresh_context():
            self._summarize_history()
            self.total_tokens = approx_tokens  # reset to current context size

    def _should_refresh_context(self) -> bool:
        window = self.CONTEXT_WINDOWS.get(self.provider, 100000)
        return self.total_tokens > window * self.refresh_threshold

    def _estimate_tokens(self, text: str) -> int:
        # Prefer tiktoken for OpenAI
        if tiktoken and self.provider == "openai":
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(text))
            except Exception:
                pass
        # Fallback rough estimate: ~4 chars/token
        return max(1, len(text) // 4)

    def _summarize_history(self) -> None:
        # Simple compaction: keep last 2 turns and summarize the rest into 1 paragraph
        old = self.history[:-2]
        if not old:
            return
        points = []
        for turn in old:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if content:
                points.append(f"{role}: {content[:200]}{'...' if len(content) > 200 else ''}")
        self.summary_memory = " | ".join(points[:10])  # compact
        self.history = self.history[-2:]
        self._rebuild_environment_card()
        k2_logger.info("Conversation summarized to maintain token budget", "AI_CHAT")

    # ---------- Provider streams ----------

    def _stream_anthropic(self, messages: List[Dict[str, str]], system_prompt: str) -> Generator[str, None, None]:
        # Anthropic expects a list of dicts with roles and content. Send rolling window.
        assistant_accum: List[str] = []

        # Retry-once for rate limits/transients
        for attempt in range(2):
            try:
                with self._anthropic.messages.stream(  # type: ignore[attr-defined]
                    model=self.model or self.DEFAULT_ANTHROPIC_MODEL,
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

        final_text = "".join(assistant_accum)
        self._append_history("assistant", final_text)

    def _stream_openai(self, messages: List[Dict[str, str]], system_prompt: str) -> Generator[str, None, None]:
        if self._openai_client is None:
            yield "[LOCAL] OpenAI client not initialized."
            return

        chat_messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        chat_messages.extend(messages)

        assistant_accum: List[str] = []

        # Retry-once policy
        for attempt in range(2):
            try:
                stream = self._openai_client.chat.completions.create(
                    model=self.model or self.DEFAULT_OPENAI_MODEL,
                    messages=chat_messages,
                    stream=True,
                )
                for chunk in stream:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
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

        final_text = "".join(assistant_accum)
        self._append_history("assistant", final_text)

    # ---------- Validation helpers (used by caller when needed) ----------

    @staticmethod
    def validate_code_security(code: str) -> List[str]:
        """Static scan for disallowed operations."""
        banned = ["exec(", "eval(", "__import__", "open(", "os.", "subprocess.", "socket.", "requests.", "httpx."]
        hits = [kw for kw in banned if kw in (code or "")]
        return hits

    @staticmethod
    def validate_strategy_output(code: str) -> (bool, str):
        """Ensure strategy code intends to produce required columns."""
        required = ["projection_mid", "projection_low", "projection_high", "is_projection"]
        for req in required:
            if req not in (code or ""):
                return False, f"Strategy must include '{req}' column"
        return True, ""

    # ---------- History helpers ----------

    def _append_history(self, role: str, content: str) -> None:
        if not content:
            return
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.MAX_HISTORY_TURNS:
            self.history = self.history[-self.MAX_HISTORY_TURNS :]


# Singleton instance
ai_chat_service = AIChatService()


