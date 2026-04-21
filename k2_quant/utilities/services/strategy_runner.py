"""
Off-GUI-thread strategy execution.

``StrategyRunner`` wraps ``dpe_service.execute_strategy`` in a ``QRunnable``
so Qt pages can fire-and-forget a strategy apply without blocking the event
loop. Each run is keyed by ``(table_name, strategy_name)`` and owns a
``threading.Event`` that the GUI can set to request cooperative cancellation
(observed by DPE via ``sys.settrace``).

Design notes
------------
* Signals are emitted from the worker thread and delivered to the GUI thread
  via Qt's default AutoConnection, so subscribers in widgets see them on the
  main thread with no extra glue.
* The runner is a process-wide singleton (``strategy_runner``) and uses a
  dedicated ``QThreadPool`` (``setMaxThreadCount=2``) so strategy work does
  not starve other background tasks that share the global pool.
* ``submit()`` is a no-op if the same ``(table, strategy)`` key is already
  running; callers should treat this as idempotent.
* ``cancel(key)`` sets the event and leaves DPE's trace hook to raise
  ``StrategyCancelled`` at the next Python line boundary. On numpy-heavy
  strategies like RPP this typically aborts within tens of milliseconds of
  the next outer-loop iteration.
* On app shutdown, call ``strategy_runner.shutdown()`` to cancel in-flight
  runs and wait briefly for workers to unwind. See ``main_app``.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import pandas as pd
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.services.dynamic_python_engine import (
    StrategyCancelled,
    dpe_service,
)


Loader = Callable[[], pd.DataFrame]


@dataclass
class StrategyJob:
    key: str
    table_name: str
    strategy_name: str
    code: str
    loader: Loader
    cancel_event: threading.Event


class _StrategyWorkerSignals(QObject):
    """Internal signal relay; lives on the thread that *constructed* the
    ``StrategyRunner`` (i.e. the GUI thread)."""
    started = pyqtSignal(str)
    progress = pyqtSignal(str, str)
    finished = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)
    cancelled = pyqtSignal(str)


class _StrategyWorker(QRunnable):
    def __init__(self, job: StrategyJob, signals: _StrategyWorkerSignals):
        super().__init__()
        self._job = job
        self._signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:  # noqa: D401 - Qt override
        job = self._job
        sig = self._signals
        key = job.key
        t_start = time.time()
        try:
            k2_logger.info(f"[{key}] worker started", "STRATEGY_RUNNER")
            sig.started.emit(key)
            if job.cancel_event.is_set():
                sig.cancelled.emit(key)
                return

            # ── Phase: loader ───────────────────────────────────────
            k2_logger.info(f"[{key}] phase=load (begin)", "STRATEGY_RUNNER")
            sig.progress.emit(key, "Loading data")
            t_load_0 = time.time()
            try:
                df = job.loader()
            except Exception as load_err:
                k2_logger.error(
                    f"[{key}] loader failed: {load_err!r}",
                    "STRATEGY_RUNNER",
                )
                k2_logger.error(traceback.format_exc(), "STRATEGY_RUNNER")
                sig.failed.emit(key, f"Load failed: {load_err}")
                return
            t_load_ms = (time.time() - t_load_0) * 1000.0
            k2_logger.info(
                f"[{key}] phase=load (done) took={t_load_ms:,.0f} ms "
                f"rows={len(df):,} cols={len(df.columns)}",
                "STRATEGY_RUNNER",
            )

            if job.cancel_event.is_set():
                sig.cancelled.emit(key)
                return

            # ── Phase: kernel (DPE) ─────────────────────────────────
            k2_logger.info(f"[{key}] phase=kernel (begin)", "STRATEGY_RUNNER")
            sig.progress.emit(
                key, f"Executing on {len(df):,} rows x {len(df.columns)} cols"
            )

            def monitor(status: str, info: Any) -> None:
                if status in ("start", "complete", "error", "cancelled"):
                    sig.progress.emit(key, f"{status}: {info}")
                elif status == "progress":
                    # DPE already logged the progress line; relay to the GUI.
                    phase = info.get('phase') if isinstance(info, dict) else ''
                    sig.progress.emit(key, f"progress: {phase}")

            t_kernel_0 = time.time()
            result = dpe_service.execute_strategy(
                job.code, df,
                monitor_callback=monitor,
                cancel_event=job.cancel_event,
            )
            t_kernel_ms = (time.time() - t_kernel_0) * 1000.0
            k2_logger.info(
                f"[{key}] phase=kernel (done) took={t_kernel_ms:,.0f} ms "
                f"success={result.get('success')} "
                f"cancelled={result.get('cancelled')}",
                "STRATEGY_RUNNER",
            )

            if result.get("cancelled"):
                sig.cancelled.emit(key)
                return

            t_total_ms = (time.time() - t_start) * 1000.0
            k2_logger.info(
                f"[{key}] worker complete total={t_total_ms:,.0f} ms "
                f"(load={t_load_ms:,.0f} + kernel={t_kernel_ms:,.0f})",
                "STRATEGY_RUNNER",
            )
            sig.finished.emit(key, result)

        except StrategyCancelled:
            sig.cancelled.emit(key)
        except Exception as e:
            k2_logger.error(
                f"[{key}] worker crashed: {e!r}", "STRATEGY_RUNNER",
            )
            k2_logger.error(traceback.format_exc(), "STRATEGY_RUNNER")
            sig.failed.emit(key, str(e))


class StrategyRunner(QObject):
    """Public facade. Connect to the five signals below.

    Signals
    -------
    started(str key)
    progress(str key, str message)
    finished(str key, dict result)       # DPE result dict
    failed(str key, str error_message)
    cancelled(str key)
    """
    started = pyqtSignal(str)
    progress = pyqtSignal(str, str)
    finished = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)
    cancelled = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None, max_concurrent: int = 2):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(1, int(max_concurrent)))
        self._active: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._signals = _StrategyWorkerSignals()
        self._signals.started.connect(self._on_started)
        self._signals.progress.connect(self._on_progress)
        self._signals.finished.connect(self._on_finished)
        self._signals.failed.connect(self._on_failed)
        self._signals.cancelled.connect(self._on_cancelled)

    # ── Public API ───────────────────────────────────────────────────
    @staticmethod
    def make_key(table_name: str, strategy_name: str) -> str:
        return f"{table_name}::{strategy_name}"

    def is_running(self, key: str) -> bool:
        with self._lock:
            return key in self._active

    def active_keys(self) -> list[str]:
        with self._lock:
            return list(self._active.keys())

    def submit(
        self,
        table_name: str,
        strategy_name: str,
        code: str,
        loader: Loader,
    ) -> Optional[str]:
        """Queue a strategy for execution. Returns the job key, or ``None`` if
        an identical job is already running."""
        key = self.make_key(table_name, strategy_name)
        with self._lock:
            if key in self._active:
                k2_logger.warning(
                    f"Strategy already running; ignoring duplicate submit: {key}",
                    "STRATEGY_RUNNER",
                )
                return None
            cancel = threading.Event()
            self._active[key] = cancel

        job = StrategyJob(
            key=key,
            table_name=table_name,
            strategy_name=strategy_name,
            code=code,
            loader=loader,
            cancel_event=cancel,
        )
        worker = _StrategyWorker(job, self._signals)
        self._pool.start(worker)
        k2_logger.info(f"Strategy queued: {key}", "STRATEGY_RUNNER")
        return key

    def cancel(self, key: str) -> bool:
        with self._lock:
            ev = self._active.get(key)
        if ev is None:
            return False
        ev.set()
        k2_logger.info(f"Cancel requested: {key}", "STRATEGY_RUNNER")
        return True

    def cancel_all(self) -> None:
        with self._lock:
            evs = list(self._active.values())
        for ev in evs:
            ev.set()

    def shutdown(self, wait_ms: int = 5000) -> None:
        self.cancel_all()
        self._pool.waitForDone(wait_ms)

    # ── Internal relays ──────────────────────────────────────────────
    def _drop(self, key: str) -> None:
        with self._lock:
            self._active.pop(key, None)

    def _on_started(self, key: str) -> None:
        self.started.emit(key)

    def _on_progress(self, key: str, msg: str) -> None:
        self.progress.emit(key, msg)

    def _on_finished(self, key: str, result: object) -> None:
        self._drop(key)
        self.finished.emit(key, result)

    def _on_failed(self, key: str, err: str) -> None:
        self._drop(key)
        self.failed.emit(key, err)

    def _on_cancelled(self, key: str) -> None:
        self._drop(key)
        self.cancelled.emit(key)


# Process-wide singleton. Consumers may also construct their own instance.
strategy_runner = StrategyRunner()
