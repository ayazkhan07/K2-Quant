"""
Dynamic Python Engine (DPE) for K2 Quant

Executes saved strategies via exec() with forecast-tab and working-sheet output support.
"""

import io
import sys
import textwrap
import threading
import time
import traceback
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from contextlib import redirect_stdout, redirect_stderr

from k2_quant.utilities.logger import k2_logger


class StrategyCancelled(Exception):
    """Raised inside a strategy's execution when the caller requests cancel."""
    pass


# Watchdog heartbeat interval (seconds). The DPE logs a single "still
# running" line this often during strategy execution so the terminal shows
# forward progress even when strategies are spending all their time inside
# long numpy calls. Configurable via env var K2_DPE_HEARTBEAT_SEC.
import os as _os
try:
    _HEARTBEAT_SEC = float(_os.environ.get("K2_DPE_HEARTBEAT_SEC", "5.0"))
except ValueError:
    _HEARTBEAT_SEC = 5.0
if _HEARTBEAT_SEC < 0.5:
    _HEARTBEAT_SEC = 0.5


from k2_quant.utilities.numeric_rounding import round_price_scalar
from k2_quant.utilities.report_helpers import (
    report_header,
    report_config,
    report_table,
    start_report_capture,
    end_report_capture,
)
from k2_quant.utilities.strategy_execution_reporting import (
    format_strategy_execution_failure,
    snapshot_dataframe_for_report,
)


class StrategyExecutor:
    """Executes strategy code in an isolated context."""

    def execute_code(self, code: str, data: pd.DataFrame,
                     monitor_callback: Optional[Callable] = None,
                     cancel_event: Optional[threading.Event] = None) -> Dict[str, Any]:
        """Execute strategy code with monitoring and optional cancellation.

        Strategy code may call ``to_forecast(column_name, values)`` to write
        price projections to the Forecast tab.  Any such writes are
        collected in ``result['_tab_writes']`` for the caller to route.

        The code is wrapped in a function so that bare ``return`` statements
        (used for early-exit guard clauses) work correctly under exec().

        ``cancel_event`` is an optional ``threading.Event``.  When set, a
        low-overhead ``sys.settrace`` hook raises :class:`StrategyCancelled`
        at the next Python line boundary in the executing strategy.  Strategy
        authors may additionally call ``__k2_check_cancel__()`` at hot-loop
        boundaries for sub-line-granularity cancellation.  The runner never
        fires from inside a numpy C call; granularity is one Python line,
        which is sufficient for the nested ``tp/ts/ti`` loops in RPP-style
        strategies.
        """
        start_time = time.time()
        tab_writes: List[Dict[str, Any]] = []
        result = {
            'success': False,
            'data': None,
            'output': '',
            'error': None,
            'execution_time': 0,
            'metrics': {},
            'cancelled': False,
            '_tab_writes': tab_writes,
        }

        if cancel_event is not None and cancel_event.is_set():
            result['cancelled'] = True
            result['error'] = 'Strategy execution cancelled before start.'
            if monitor_callback:
                try:
                    monitor_callback('cancelled', {'reason': 'pre-start'})
                except Exception:
                    pass
            return result

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        exec_context: Optional[Dict[str, Any]] = None
        report_blocks = start_report_capture()

        try:
            def _to_forecast(column_name_or_set_index, values_or_open=None,
                             high_values=None, low_values=None,
                             close_values=None, anchor_price=None):
                def _clean(vals):
                    if vals is None:
                        return None
                    if isinstance(vals, (pd.Series, np.ndarray)):
                        vals = vals.tolist()
                    out = []
                    for v in vals:
                        if v is None:
                            out.append(None)
                        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                            out.append(None)
                        else:
                            out.append(round_price_scalar(float(v)))
                    return out[:500]

                if isinstance(column_name_or_set_index, str):
                    entry = {
                        "type": "forecast",
                        "column_name": column_name_or_set_index,
                        "values": _clean(values_or_open),
                    }
                    if anchor_price is not None:
                        entry["anchor_price"] = float(anchor_price)
                    tab_writes.append(entry)
                else:
                    tab_writes.append({
                        "type": "forecast",
                        "set_index": int(column_name_or_set_index),
                        "open_values": _clean(values_or_open),
                        "high_values": _clean(high_values),
                        "low_values": _clean(low_values),
                        "close_values": _clean(close_values),
                    })

            # Shared observability state written by trace hook + progress
            # hook, read by the heartbeat watchdog thread.
            exec_state = {
                'phase': 'starting',
                'phase_detail': '',
                'phase_since': time.time(),
                'last_line': '',
                'n_progress': 0,
                'n_trace': 0,
            }

            def _check_cancel():
                if cancel_event is not None and cancel_event.is_set():
                    raise StrategyCancelled()

            def _progress(phase, detail=None):
                """Strategy-side progress hook. Logs immediately, updates the
                watchdog's phase tracker, and routes through
                ``monitor_callback`` so the runner can relay it to the GUI."""
                detail_str = '' if detail is None else str(detail)
                exec_state['phase'] = str(phase)
                exec_state['phase_detail'] = detail_str
                exec_state['phase_since'] = time.time()
                exec_state['n_progress'] += 1
                msg = f"{phase}" if not detail_str else f"{phase} | {detail_str}"
                k2_logger.info(f"[progress] {msg}", "DPE")
                if monitor_callback is not None:
                    try:
                        monitor_callback('progress', {
                            'phase': str(phase),
                            'detail': detail_str,
                        })
                    except Exception:
                        pass

            exec_context = {
                'pd': pd,
                'np': np,
                'datetime': datetime,
                'timedelta': timedelta,
                'data': data.copy(),
                'to_forecast': _to_forecast,
                'report_header': report_header,
                'report_config': report_config,
                'report_table': report_table,
                '__k2_check_cancel__': _check_cancel,
                '__k2_progress__': _progress,
            }
            exec_context['df'] = exec_context['data']

            if monitor_callback:
                monitor_callback('start', {'rows': len(data), 'columns': len(data.columns)})

            # Wrap in a function so bare ``return`` statements work
            wrapped = (
                "def __k2_strategy__():\n"
                "    global df, data, result\n"
                + textwrap.indent(code, "    ") + "\n"
                "__k2_strategy__()\n"
            )

            # sys.settrace hook: serves two purposes on the worker thread:
            #   1) observes ``cancel_event`` at Python line boundaries and
            #      raises ``StrategyCancelled`` if set.
            #   2) records the last-seen ``file:line func`` into
            #      ``exec_state['last_line']`` so the heartbeat watchdog has
            #      something to report even when strategies don't call
            #      ``__k2_progress__`` themselves.
            # Trace only fires between Python bytecode ops; it does NOT fire
            # inside numpy C code, so during long numpy calls ``last_line``
            # will be stale until the numpy call returns.
            trace_state = exec_state  # closure grab
            _watcher_stop = threading.Event()

            def _trace(frame, event, arg, _ev=cancel_event, _st=trace_state):
                if _ev is not None and _ev.is_set():
                    raise StrategyCancelled()
                if event == 'line':
                    co = frame.f_code
                    _st['last_line'] = (
                        f"{co.co_filename}:{frame.f_lineno} in {co.co_name}"
                    )
                    _st['n_trace'] += 1
                return _trace

            def _watchdog():
                """Runs on a daemon thread; logs a heartbeat every N seconds
                so terminals show movement even when the kernel is silent."""
                t0 = exec_state['phase_since']
                while not _watcher_stop.wait(_HEARTBEAT_SEC):
                    now = time.time()
                    elapsed = now - t0
                    in_phase = now - exec_state['phase_since']
                    phase = exec_state['phase']
                    detail = exec_state['phase_detail']
                    last = exec_state['last_line'] or '<no python line seen>'
                    np_calls = exec_state['n_trace']
                    segs = [
                        f"elapsed={elapsed:.1f}s",
                        f"phase={phase!r}",
                    ]
                    if detail:
                        segs.append(f"detail={detail!r}")
                    segs.append(f"in_phase={in_phase:.1f}s")
                    segs.append(f"trace_hits={np_calls}")
                    segs.append(f"last={last}")
                    k2_logger.info("[heartbeat] " + " ".join(segs), "DPE")

            watchdog_thread = threading.Thread(
                target=_watchdog, name="DPE-watchdog", daemon=True)
            watchdog_thread.start()

            prior_trace = sys.gettrace()
            sys.settrace(_trace)
            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(wrapped, exec_context)
            finally:
                sys.settrace(prior_trace)
                _watcher_stop.set()
                # Don't block long on the watchdog; it's a daemon.
                watchdog_thread.join(timeout=1.0)

            if 'result' in exec_context:
                result['data'] = exec_context['result']
            elif 'data' in exec_context:
                result['data'] = exec_context['data']
            elif 'df' in exec_context:
                result['data'] = exec_context['df']

            result['metrics'] = {
                'execution_time': time.time() - start_time,
                'original_rows': len(data),
            }

            result['success'] = True

            if monitor_callback:
                monitor_callback('complete', result['metrics'])

        except StrategyCancelled:
            result['cancelled'] = True
            result['error'] = 'Strategy execution cancelled.'
            result['metrics'] = {
                'original_rows': len(data),
                'original_columns': len(data.columns),
            }
            if monitor_callback:
                try:
                    monitor_callback('cancelled', {'reason': 'user'})
                except Exception:
                    pass
            k2_logger.info("Strategy execution cancelled by caller", "DPE")
        except Exception as e:
            result['traceback'] = traceback.format_exc()
            df_any = None
            if exec_context is not None:
                df_any = exec_context.get('df')
                if df_any is None:
                    df_any = exec_context.get('data')
            snap = snapshot_dataframe_for_report(df_any)
            fail_metrics = {
                'original_rows': len(data),
                'original_columns': len(data.columns),
            }
            result['metrics'] = fail_metrics
            result['error'] = format_strategy_execution_failure(
                e,
                code,
                metrics=fail_metrics,
                df_summary=snap,
            )
            stderr_output = stderr_capture.getvalue()
            if stderr_output:
                result['error'] += f"\n\n--- captured stderr ---\n{stderr_output.strip()}"

            if monitor_callback:
                monitor_callback('error', {'error': str(e)})

            k2_logger.error(f"Strategy execution failed: {str(e)}", "DPE")

        finally:
            end_report_capture()
            # stdout: optional print() diagnostics only; tables live in _report_blocks
            result['output'] = stdout_capture.getvalue()
            result['_report_blocks'] = list(report_blocks)

            elapsed = time.time() - start_time
            result['execution_time'] = elapsed
            m = result.get('metrics')
            if isinstance(m, dict):
                m['execution_time'] = elapsed

        return result


class DynamicPythonEngine:
    """Main DPE service for strategy execution."""

    def __init__(self):
        self.executor = StrategyExecutor()

    def execute_strategy(self, strategy_code: str, data: pd.DataFrame,
                         metadata: Dict[str, Any] = None,
                         monitor_callback: Optional[Callable] = None,
                         cancel_event: Optional[threading.Event] = None) -> Dict[str, Any]:
        """Execute a trading strategy.

        ``monitor_callback`` receives ``(status, info_dict)`` pairs for
        ``start``, ``complete``, ``error`` and ``cancelled`` events, plus any
        ad-hoc progress updates a caller-supplied callback layers on top.
        ``cancel_event`` enables cooperative cancellation; see
        :meth:`StrategyExecutor.execute_code` for granularity.
        """
        k2_logger.info("Executing strategy", "DPE")

        def default_monitor(status, info):
            k2_logger.info(f"Strategy execution {status}: {info}", "DPE")

        def chained_monitor(status, info):
            default_monitor(status, info)
            if monitor_callback is not None:
                try:
                    monitor_callback(status, info)
                except Exception as cb_err:
                    k2_logger.warning(
                        f"monitor_callback raised {cb_err!r}; ignoring", "DPE"
                    )

        return self.executor.execute_code(
            strategy_code, data, chained_monitor, cancel_event=cancel_event
        )


# Singleton instance
dpe_service = DynamicPythonEngine()
