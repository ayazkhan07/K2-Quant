"""
Polygon.io WebSocket Manager — singleton real-time market data connection.

Manages a single WebSocket connection to Polygon's stocks endpoint,
multiplexing subscriptions for multiple symbols.  Raw 1-minute aggregate
bars are emitted as Qt signals so any number of UI consumers can connect.

Usage:
    from k2_quant.utilities.services.polygon_websocket import polygon_ws_manager

    polygon_ws_manager.bar_received.connect(my_handler)   # receives dict
    polygon_ws_manager.subscribe("AAPL")
    polygon_ws_manager.start()
"""

import json
import threading
from typing import Set, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from k2_quant.utilities.config.api_config import api_config
from k2_quant.utilities.logger import k2_logger

try:
    import websocket as _ws_lib  # websocket-client
except ImportError:
    _ws_lib = None


_POLYGON_WS_URL = "wss://socket.polygon.io/stocks"


class PolygonWebSocketManager(QObject):
    """Singleton WebSocket connection to Polygon.io real-time stocks feed.

    Signals
    -------
    bar_received(dict)
        Emitted for every Aggregate-Minute (AM.*) message.  The dict keys
        mirror the Polygon message:  sym, s (start epoch ms), e (end epoch ms),
        o, h, l, c, v, vw, z, a, op, ...
    connection_status(str)
        Emitted when connection state changes: "connected", "disconnected",
        "auth_ok", "auth_failed", "error: <msg>".
    """

    bar_received = pyqtSignal(dict)
    connection_status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._subscriptions: Set[str] = set()
        self._ws: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._authenticated = False
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._running and self._authenticated

    def subscribe(self, symbol: str):
        symbol = symbol.upper()
        with self._lock:
            if symbol in self._subscriptions:
                return
            self._subscriptions.add(symbol)
        if self._authenticated and self._ws:
            self._send_subscribe([symbol])
        k2_logger.info(f"WS subscribed: AM.{symbol}", "POLYGON_WS")

    def unsubscribe(self, symbol: str):
        symbol = symbol.upper()
        with self._lock:
            self._subscriptions.discard(symbol)
        if self._authenticated and self._ws:
            self._send_unsubscribe([symbol])
        k2_logger.info(f"WS unsubscribed: AM.{symbol}", "POLYGON_WS")

    def start(self):
        if self._running:
            return
        if _ws_lib is None:
            k2_logger.error("websocket-client not installed", "POLYGON_WS")
            self.connection_status.emit("error: websocket-client not installed")
            return
        api_key = api_config.polygon_api_key
        if not api_key:
            k2_logger.error("No Polygon API key configured", "POLYGON_WS")
            self.connection_status.emit("error: no API key")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="PolygonWS")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        self._authenticated = False
        k2_logger.info("WebSocket stopped", "POLYGON_WS")

    def _run(self):
        api_key = api_config.polygon_api_key
        while self._running:
            try:
                ws = _ws_lib.WebSocketApp(
                    _POLYGON_WS_URL,
                    on_open=lambda ws_obj: self._on_open(ws_obj, api_key),
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws = ws
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as exc:
                k2_logger.error(f"WS run_forever error: {exc}", "POLYGON_WS")

            self._authenticated = False
            self.connection_status.emit("disconnected")

            if not self._running:
                break

            import time
            k2_logger.info("Reconnecting in 5 s …", "POLYGON_WS")
            time.sleep(5)

    def _on_open(self, ws_obj, api_key: str):
        k2_logger.info("WS connection opened, authenticating …", "POLYGON_WS")
        self.connection_status.emit("connected")
        ws_obj.send(json.dumps({"action": "auth", "params": api_key}))

    def _on_message(self, _ws_obj, raw: str):
        try:
            messages = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(messages, list):
            messages = [messages]

        for msg in messages:
            ev = msg.get("ev")
            if ev == "status":
                status = msg.get("status", "")
                if status == "auth_success":
                    self._authenticated = True
                    self.connection_status.emit("auth_ok")
                    k2_logger.info("WS authenticated", "POLYGON_WS")
                    with self._lock:
                        syms = list(self._subscriptions)
                    if syms:
                        self._send_subscribe(syms)
                elif status == "auth_failed":
                    self.connection_status.emit("auth_failed")
                    k2_logger.error("WS auth failed", "POLYGON_WS")
            elif ev == "AM":
                self.bar_received.emit(msg)

    def _on_error(self, _ws_obj, error):
        k2_logger.error(f"WS error: {error}", "POLYGON_WS")
        self.connection_status.emit(f"error: {error}")

    def _on_close(self, _ws_obj, close_status_code, close_msg):
        k2_logger.info(
            f"WS closed (code={close_status_code}, msg={close_msg})", "POLYGON_WS"
        )
        self._authenticated = False
        self.connection_status.emit("disconnected")

    def _send_subscribe(self, symbols: list):
        if not self._ws:
            return
        params = ",".join(f"AM.{s}" for s in symbols)
        try:
            self._ws.send(json.dumps({"action": "subscribe", "params": params}))
            k2_logger.info(f"WS subscribe sent: {params}", "POLYGON_WS")
        except Exception as exc:
            k2_logger.error(f"WS subscribe send failed: {exc}", "POLYGON_WS")

    def _send_unsubscribe(self, symbols: list):
        if not self._ws:
            return
        params = ",".join(f"AM.{s}" for s in symbols)
        try:
            self._ws.send(json.dumps({"action": "unsubscribe", "params": params}))
        except Exception as exc:
            k2_logger.error(f"WS unsubscribe send failed: {exc}", "POLYGON_WS")


# Singleton
polygon_ws_manager = PolygonWebSocketManager()
