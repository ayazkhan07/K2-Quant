"""
Stream Reconciler — fills the gap between a model's last data point and now.

On stream start the model may be hours or days behind.  This utility
calculates the gap, bulk-fetches the missing bars from Polygon REST,
and returns them as a list of normalised bar dicts ready for the
ActualDataManager and the UI table.

Signals
-------
reconciliation_progress(int, int)
    (fetched_so_far, total_estimated)
reconciliation_complete(list)
    List[dict] of backfilled bars, chronologically ordered.
reconciliation_error(str)
    Error description if the fetch fails.
"""

import math
from datetime import datetime, timedelta, date, time as dt_time
from typing import List, Dict, Optional, Tuple

import requests
from PyQt6.QtCore import QObject, pyqtSignal, QThread

from k2_quant.utilities.config.api_config import api_config
from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.helpers.market_hours import (
    is_trading_day, market_session_times, _MARKET_OPEN, _MARKET_CLOSE, _ET,
)

BACKFILL_CAP = 2000


class _ReconcileWorker(QThread):
    """Background thread for REST backfill — keeps UI responsive."""

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, symbol: str, timespan: str, frequency: int,
                 gap_start: datetime, gap_end: datetime,
                 market_hours_only: bool, parent=None):
        super().__init__(parent)
        self.symbol = symbol
        self.timespan = timespan
        self.frequency = frequency
        self.gap_start = gap_start
        self.gap_end = gap_end
        self.market_hours_only = market_hours_only

    def run(self):
        try:
            bars = _fetch_gap_bars(
                symbol=self.symbol,
                timespan=self.timespan,
                frequency=self.frequency,
                start_dt=self.gap_start,
                end_dt=self.gap_end,
                market_hours_only=self.market_hours_only,
                progress_cb=lambda done, total: self.progress.emit(done, total),
            )
            self.finished.emit(bars)
        except Exception as exc:
            self.error.emit(str(exc))


class StreamReconciler(QObject):
    """Orchestrates gap detection + background backfill for one stream window."""

    reconciliation_progress = pyqtSignal(int, int)
    reconciliation_complete = pyqtSignal(list)
    reconciliation_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[_ReconcileWorker] = None

    def reconcile(self, *,
                  symbol: str,
                  timespan: str,
                  frequency: int,
                  last_bar_date: date,
                  last_bar_time: dt_time,
                  market_hours_only: bool):
        """Start an async backfill from `last_bar + 1 bar` to now."""

        ts_lower = timespan.lower()
        if ts_lower.startswith("min"):
            delta = timedelta(minutes=frequency)
        elif ts_lower.startswith("hour"):
            delta = timedelta(hours=frequency)
        elif ts_lower.startswith("day"):
            delta = timedelta(days=frequency)
        else:
            delta = timedelta(minutes=frequency)

        gap_start = datetime.combine(last_bar_date, last_bar_time) + delta
        gap_end = datetime.now()

        if gap_end <= gap_start:
            self.reconciliation_complete.emit([])
            return

        k2_logger.info(
            f"Reconciling {symbol}: {gap_start} -> {gap_end} "
            f"(timespan={timespan}, freq={frequency})",
            "RECONCILER",
        )

        self._worker = _ReconcileWorker(
            symbol=symbol, timespan=timespan, frequency=frequency,
            gap_start=gap_start, gap_end=gap_end,
            market_hours_only=market_hours_only,
        )
        self._worker.progress.connect(self.reconciliation_progress.emit)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self.reconciliation_error.emit)
        self._worker.start()

    def _on_done(self, bars: list):
        k2_logger.info(f"Reconciliation complete — {len(bars)} bars fetched", "RECONCILER")
        self.reconciliation_complete.emit(bars)

    def cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)


def _fetch_gap_bars(
    symbol: str,
    timespan: str,
    frequency: int,
    start_dt: datetime,
    end_dt: datetime,
    market_hours_only: bool,
    progress_cb=None,
) -> List[dict]:
    """Fetch aggregated bars from Polygon REST for the given time range.

    Returns a list of normalised bar dicts sorted chronologically.
    """
    api_key = api_config.polygon_api_key
    if not api_key:
        raise RuntimeError("Polygon API key not configured")

    ts_lower = (timespan or "minute").lower()
    if ts_lower.startswith("min"):
        api_timespan = "minute"
    elif ts_lower.startswith("hour"):
        api_timespan = "hour"
    elif ts_lower.startswith("day"):
        api_timespan = "day"
    else:
        api_timespan = "minute"

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range"
        f"/{frequency}/{api_timespan}/{start_str}/{end_str}"
    )
    params = {
        "apiKey": api_key,
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
    }

    all_results: List[dict] = []

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "OK":
        raise RuntimeError(f"Polygon API error: {data.get('status')}")

    raw_bars = data.get("results") or []
    total_est = min(len(raw_bars), BACKFILL_CAP)

    import pytz
    et = pytz.timezone("US/Eastern")

    # Polygon returns full calendar days — we must drop bars before gap_start
    gap_start_epoch_ms = int(start_dt.timestamp() * 1000)

    for i, bar in enumerate(raw_bars):
        if len(all_results) >= BACKFILL_CAP:
            break

        ts_ms = bar.get("t", 0)

        if ts_ms < gap_start_epoch_ms:
            continue

        utc_dt = datetime.utcfromtimestamp(ts_ms / 1000)
        market_dt = pytz.utc.localize(utc_dt).astimezone(et).replace(tzinfo=None)

        if market_hours_only:
            mt = market_dt.time()
            if not (dt_time(9, 30) <= mt < dt_time(16, 0)):
                continue

        all_results.append({
            "symbol": symbol.upper(),
            "timestamp_ms": ts_ms,
            "datetime": market_dt,
            "date": market_dt.date(),
            "time": market_dt.time(),
            "open": round(float(bar.get("o", 0)), 2),
            "high": round(float(bar.get("h", 0)), 2),
            "low": round(float(bar.get("l", 0)), 2),
            "close": round(float(bar.get("c", 0)), 2),
            "volume": int(bar.get("v", 0)),
            "vwap": round(float(bar.get("vw", 0)), 2),
        })

        if progress_cb and (i % 100 == 0 or i == len(raw_bars) - 1):
            progress_cb(len(all_results), total_est)

    all_results.sort(key=lambda b: b["timestamp_ms"])
    return all_results
