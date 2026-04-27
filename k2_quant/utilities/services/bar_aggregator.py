"""
Bar Aggregator — accumulates 1-minute Polygon WS bars into model-frequency bars.

Given a model with frequency=30 and timespan=minute, this class collects
1-minute bars that fall within the same clock-aligned window and emits a
single aggregated bar when the window closes.

Clock alignment means a 30-minute model always produces bars starting at
:00 and :30 past the hour (09:30, 10:00, 10:30, …), regardless of when
the stream was started.

Signals (Qt):
    bar_updated(dict)     — partial / forming bar (every incoming 1-min tick)
    bar_completed(dict)   — fully aggregated bar ready for persistence

Bar dict format (emitted):
    {
        "symbol": "AAPL",
        "start_ts": <epoch ms>,
        "end_ts":   <epoch ms>,
        "open": 182.30,
        "high": 183.10,
        "low":  181.90,
        "close": 182.75,
        "volume": 48230,
        "vwap": 182.55,
        "bars_accumulated": 15,
        "bars_required": 30,
        "is_complete": False,
    }
"""

from typing import Optional, Dict, Any
from datetime import datetime, time as dt_time

import pytz
from PyQt6.QtCore import QObject, pyqtSignal

from k2_quant.utilities.logger import k2_logger

_ET = pytz.timezone("US/Eastern")


class BarAggregator(QObject):
    """Accumulates 1-min bars into clock-aligned N-min bars for one symbol.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "AAPL").
    frequency : int
        Number of minutes per aggregated bar (e.g. 30 for a 30-min model).
    timespan : str
        Model timespan ("minute", "hour", "day").
    """

    bar_updated = pyqtSignal(dict)
    bar_completed = pyqtSignal(dict)

    def __init__(self, symbol: str, frequency: int = 1, timespan: str = "minute",
                 parent=None):
        super().__init__(parent)
        self.symbol = symbol.upper()
        self.timespan = (timespan or "minute").lower()

        # Daily models collapse an entire trading date into one bar. Any
        # intra-day frequency/multiplier on a "day" timespan (e.g. 5-day,
        # weekly stored as days) still buckets per calendar date — the
        # minute-slice arithmetic used for intraday sizes is wrong here
        # because it tiles the 24-hour clock with session-length chunks.
        self._is_daily = self.timespan.startswith("day")

        if self._is_daily:
            # Retained only for progress display (bars_required / "N/390 min")
            # A regular U.S. session is 390 one-minute bars; pre-/post-market
            # minutes will count too but progress is advisory only.
            self._window_minutes = 390
        elif self.timespan.startswith("hour"):
            self._window_minutes = frequency * 60
        else:
            self._window_minutes = max(1, frequency)

        self._reset_accumulator()

    @property
    def bars_required(self) -> int:
        return self._window_minutes

    @property
    def is_daily(self) -> bool:
        return self._is_daily

    def _bar_window_key(self, ts_ms: int):
        """Return a hashable key identifying the aggregation window a raw
        1-minute bar belongs to.

        For daily models the key is simply the ET calendar date, so every
        1-minute bar between 04:00 ET (Polygon AM feed start) and 19:59 ET
        of the same trading date collapses into a single bucket.

        For intraday models the key is (date, clock-aligned window start
        minute) as before.
        """
        utc_dt = datetime.utcfromtimestamp(ts_ms / 1000)
        et_dt = pytz.utc.localize(utc_dt).astimezone(_ET)

        if self._is_daily:
            return (et_dt.date(),)

        minutes_of_day = et_dt.hour * 60 + et_dt.minute
        window_start = (minutes_of_day // self._window_minutes) * self._window_minutes
        return (et_dt.date(), window_start)

    def _window_start_ts_ms(self, ts_ms: int) -> int:
        """Snap a timestamp to the window-start epoch-ms used by the DB.

        Daily bars are stamped at **09:30 ET of the trading date** (DST
        aware). Downstream code in ``StreamWindow._on_bar_completed``
        converts this back to naive ET, yielding ``market_date = trading
        date`` and ``market_time = 09:30:00`` — matching ``stock_data_service``'s
        re-stamped historical rows and ``StreamReconciler``'s re-stamped
        backfill rows. Live, historical, and backfilled daily bars all
        share one ``timestamp_ms`` primary key which makes the upsert-on-
        tick behavior idempotent.

        Intraday bars keep their original clock-aligned window start.
        """
        utc_dt = datetime.utcfromtimestamp(ts_ms / 1000)
        et_dt = pytz.utc.localize(utc_dt).astimezone(_ET)

        if self._is_daily:
            trading_date = et_dt.date()
            session_open_et = _ET.localize(
                datetime(trading_date.year, trading_date.month,
                         trading_date.day, 9, 30)
            )
            return int(session_open_et.timestamp() * 1000)

        minutes_of_day = et_dt.hour * 60 + et_dt.minute
        window_start_min = (minutes_of_day // self._window_minutes) * self._window_minutes
        snapped = et_dt.replace(
            hour=window_start_min // 60,
            minute=window_start_min % 60,
            second=0, microsecond=0,
        )
        return int(snapped.timestamp() * 1000)

    def ingest(self, ws_bar: dict):
        """Feed a raw Polygon AM message.  Ignored if symbol doesn't match."""
        sym = (ws_bar.get("sym") or "").upper()
        if sym != self.symbol:
            return

        start_ts = int(ws_bar.get("s", 0))
        new_key = self._bar_window_key(start_ts)

        if self._count > 0 and self._current_window != new_key:
            completed = self._to_dict()
            completed["is_complete"] = True
            self.bar_completed.emit(completed)
            self._reset_accumulator()

        o = float(ws_bar.get("o", 0))
        h = float(ws_bar.get("h", 0))
        l_ = float(ws_bar.get("l", 0))
        c = float(ws_bar.get("c", 0))
        v = int(ws_bar.get("v", 0))
        vw = float(ws_bar.get("vw", 0))
        end_ts = int(ws_bar.get("e", 0))

        if self._count == 0:
            self._start_ts = self._window_start_ts_ms(start_ts)
            self._open = o
            self._high = h
            self._low = l_
            self._vw_sum = 0.0
            self._vol_total = 0
            self._current_window = new_key
        else:
            self._high = max(self._high, h)
            self._low = min(self._low, l_)

        self._close = c
        self._end_ts = end_ts
        self._vol_total += v
        self._vw_sum += vw * v
        self._count += 1

        bar_dict = self._to_dict()
        bar_dict["is_complete"] = False
        self.bar_updated.emit(bar_dict)

    def try_seed(self, bar: dict, now_ts_ms: Optional[int] = None) -> bool:
        """Pre-load the accumulator from an already-existing bar.

        Used on stream start: once the reconciler has backfilled today's
        partial daily bar (real session open, accumulated H/L/V from market
        open), we seed the in-memory accumulator so the first live WS 1-
        minute tick follows the ``_count > 0`` branch of ``ingest()`` —
        taking ``max(seed_high, new_high)``, ``min(seed_low, new_low)``,
        ``close = new_close``, ``volume += new_volume`` — instead of the
        ``_count == 0`` branch that would blindly overwrite the real
        session open with whatever GOOG is trading at stream-start moment.

        Only seeds if ``bar``'s ``timestamp_ms`` matches the aggregator's
        current trading-window key (computed from ``now_ts_ms``). Returns
        True if seeded, False if the bar belongs to a prior window (caller
        should not treat it as the current in-progress bar).
        """
        import time as _time
        ts_ms = int(bar.get("timestamp_ms") or bar.get("start_ts") or 0)
        if ts_ms <= 0:
            return False
        now_ms = now_ts_ms if now_ts_ms is not None else int(_time.time() * 1000)
        if self._window_start_ts_ms(now_ms) != ts_ms:
            return False

        self._start_ts = ts_ms
        self._current_window = self._bar_window_key(ts_ms)
        self._open = float(bar.get("open") or 0.0)
        self._high = float(bar.get("high") or self._open)
        self._low = float(bar.get("low") or self._open)
        self._close = float(bar.get("close") or self._open)
        vol = int(bar.get("volume") or 0)
        self._vol_total = vol
        vwap_val = bar.get("vwap")
        self._vw_sum = (float(vwap_val) * vol) if (vwap_val is not None and vol > 0) else 0.0
        self._end_ts = int(bar.get("end_ts") or ts_ms)
        # Use elapsed minutes since the window's session-open ts as a
        # best-guess sample count for the progress display ("N/390 min").
        # The seed summarises an unknown number of 1-minute bars so this
        # is advisory only; cap at the window size.
        elapsed_min = max(1, (now_ms - ts_ms) // 60000)
        self._count = min(int(elapsed_min), self._window_minutes)
        return True

    def flush(self) -> Optional[Dict[str, Any]]:
        """Force-emit whatever is accumulated (e.g. at market close).
        Returns the bar dict or None if empty."""
        if self._count == 0:
            return None
        bar = self._to_dict()
        bar["is_complete"] = True
        self.bar_completed.emit(bar)
        self._reset_accumulator()
        return bar

    def reset(self):
        self._reset_accumulator()

    def _reset_accumulator(self):
        self._open = 0.0
        self._high = 0.0
        self._low = 0.0
        self._close = 0.0
        self._vol_total = 0
        self._vw_sum = 0.0
        self._start_ts = 0
        self._end_ts = 0
        self._count = 0
        self._current_window = None

    def _to_dict(self) -> dict:
        vwap = (self._vw_sum / self._vol_total) if self._vol_total > 0 else self._close
        return {
            "symbol": self.symbol,
            "start_ts": self._start_ts,
            "end_ts": self._end_ts,
            "open": round(self._open, 2),
            "high": round(self._high, 2),
            "low": round(self._low, 2),
            "close": round(self._close, 2),
            "volume": self._vol_total,
            "vwap": round(vwap, 2),
            "bars_accumulated": self._count,
            "bars_required": self._window_minutes,
            "is_complete": False,
        }
