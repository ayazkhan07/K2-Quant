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
        import pytz
        import re as _re
        et = pytz.timezone("US/Eastern")

        ts_lower = timespan.lower().strip()
        _m = _re.match(r'^(\d+)\s*(min|minute|hour|day|week|month)', ts_lower)
        if _m:
            if frequency <= 1:
                frequency = int(_m.group(1))
            ts_lower = _m.group(2)
        if ts_lower.startswith("min"):
            delta = timedelta(minutes=frequency)
        elif ts_lower.startswith("hour"):
            delta = timedelta(hours=frequency)
        elif ts_lower.startswith("day"):
            delta = timedelta(days=frequency)
        else:
            delta = timedelta(minutes=frequency)

        naive_start = datetime.combine(last_bar_date, last_bar_time) + delta
        gap_start = et.localize(naive_start)
        gap_end = datetime.now(et)

        k2_logger.info(
            f"[RECONCILER] Input: last_bar={last_bar_date} {last_bar_time} | "
            f"delta={delta} | timespan={timespan} freq={frequency}",
            "RECONCILER",
        )
        k2_logger.info(
            f"[RECONCILER] Gap: {gap_start} -> {gap_end} "
            f"(gap_days={(gap_end - gap_start).total_seconds() / 86400:.1f})",
            "RECONCILER",
        )

        if gap_end <= gap_start:
            k2_logger.info(
                "[RECONCILER] No gap to fill (gap_end <= gap_start), skipping",
                "RECONCILER",
            )
            self.reconciliation_complete.emit([])
            return

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

    import re as _re
    ts_lower = (timespan or "minute").lower().strip()
    _m = _re.match(r'^(\d+)\s*(min|minute|hour|day|week|month)', ts_lower)
    if _m:
        if frequency <= 1:
            frequency = int(_m.group(1))
        ts_lower = _m.group(2)

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

    k2_logger.info(
        f"[RECONCILER] REST request: {url.replace(api_key, '***')}",
        "RECONCILER",
    )
    k2_logger.info(
        f"[RECONCILER] Date range: {start_str} to {end_str} | "
        f"frequency={frequency} | timespan={api_timespan}",
        "RECONCILER",
    )

    all_results: List[dict] = []

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "OK":
        raise RuntimeError(f"Polygon API error: {data.get('status')}")

    raw_bars = data.get("results") or []
    total_est = min(len(raw_bars), BACKFILL_CAP)

    k2_logger.info(
        f"[RECONCILER] Polygon returned {len(raw_bars)} raw bars | "
        f"status={data.get('status')} | resultsCount={data.get('resultsCount', '?')}",
        "RECONCILER",
    )

    import pytz
    et = pytz.timezone("US/Eastern")

    # Daily bars get re-stamped to 09:30 ET of their trading date (see the
    # per-bar loop below), which means the gap filter can't use
    # ``start_dt.timestamp()`` directly — Polygon's raw ``t`` for a daily
    # bar is midnight UTC of the trading date, a few hours EARLIER than the
    # re-stamped epoch. Instead, for daily we filter on "trading date of
    # this Polygon bar >= trading date at or after gap_start", expressed as
    # midnight UTC of the calendar date of ``start_dt`` viewed in ET.
    if api_timespan == "day":
        start_dt_et_date = start_dt.astimezone(et).date()
        gap_start_epoch_ms = int(pytz.utc.localize(datetime(
            start_dt_et_date.year, start_dt_et_date.month, start_dt_et_date.day
        )).timestamp() * 1000)
    else:
        gap_start_epoch_ms = int(start_dt.timestamp() * 1000)
    skipped_before_gap = 0
    skipped_market_hours = 0

    if raw_bars:
        first_ts = raw_bars[0].get("t", 0)
        last_ts = raw_bars[-1].get("t", 0)
        first_dt = datetime.utcfromtimestamp(first_ts / 1000) if first_ts else None
        last_dt = datetime.utcfromtimestamp(last_ts / 1000) if last_ts else None
        k2_logger.info(
            f"[RECONCILER] Raw bar range: {first_dt} UTC -> {last_dt} UTC",
            "RECONCILER",
        )
        k2_logger.info(
            f"[RECONCILER] Gap filter: epoch_ms >= {gap_start_epoch_ms} "
            f"(= {start_dt})",
            "RECONCILER",
        )

    for i, bar in enumerate(raw_bars):
        if len(all_results) >= BACKFILL_CAP:
            break

        ts_ms = bar.get("t", 0)

        if ts_ms < gap_start_epoch_ms:
            skipped_before_gap += 1
            continue

        utc_dt = datetime.utcfromtimestamp(ts_ms / 1000)

        if api_timespan == "day":
            # Polygon's ``t`` for a daily aggregate is midnight UTC of the
            # trading date (so ``utc_dt.date()`` IS the trading date D).
            # Re-stamp to 09:30 ET of D so stored rows show the real
            # trading date, a meaningful session-open time, and a DST-
            # aware epoch that matches ``BarAggregator`` and
            # ``stock_data_service`` for seamless upsert on overlap.
            trading_date = utc_dt.date()
            session_open_et = et.localize(datetime(
                trading_date.year, trading_date.month, trading_date.day,
                9, 30,
            ))
            market_dt = session_open_et.replace(tzinfo=None)
            ts_ms = int(session_open_et.timestamp() * 1000)
        else:
            market_dt = pytz.utc.localize(utc_dt).astimezone(et).replace(tzinfo=None)

        if market_hours_only:
            mt = market_dt.time()
            if not (dt_time(9, 30) <= mt < dt_time(16, 0)):
                skipped_market_hours += 1
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

    first_bar_str = (
        f"{all_results[0]['date']} {all_results[0]['time']}"
        if all_results else "N/A"
    )
    last_bar_str = (
        f"{all_results[-1]['date']} {all_results[-1]['time']}"
        if all_results else "N/A"
    )
    k2_logger.info(
        f"[RECONCILER] Result: {len(all_results)} bars kept | "
        f"{skipped_before_gap} skipped (before gap) | "
        f"{skipped_market_hours} skipped (outside market hours)",
        "RECONCILER",
    )
    k2_logger.info(
        f"[RECONCILER] Actual data range: {first_bar_str} -> {last_bar_str}",
        "RECONCILER",
    )

    return all_results
