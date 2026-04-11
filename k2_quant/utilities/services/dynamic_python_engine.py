"""
Dynamic Python Engine (DPE) for K2 Quant

Executes saved strategies via exec() with forecast-tab and working-sheet output support.
"""

import io
import textwrap
import time
import traceback
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from contextlib import redirect_stdout, redirect_stderr

from k2_quant.utilities.logger import k2_logger


class StrategyExecutor:
    """Executes strategy code in an isolated context."""

    def execute_code(self, code: str, data: pd.DataFrame,
                     monitor_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute strategy code with monitoring.

        Strategy code may call ``to_forecast(column_name, values)`` to write
        price projections to the Forecast tab.  Any such writes are
        collected in ``result['_tab_writes']`` for the caller to route.

        The code is wrapped in a function so that bare ``return`` statements
        (used for early-exit guard clauses) work correctly under exec().
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
            '_tab_writes': tab_writes,
        }

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            def _to_forecast(column_name_or_set_index, values_or_open=None,
                             high_values=None, low_values=None,
                             close_values=None):
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
                            out.append(float(v))
                    return out[:500]

                if isinstance(column_name_or_set_index, str):
                    tab_writes.append({
                        "type": "forecast",
                        "column_name": column_name_or_set_index,
                        "values": _clean(values_or_open),
                    })
                else:
                    tab_writes.append({
                        "type": "forecast",
                        "set_index": int(column_name_or_set_index),
                        "open_values": _clean(values_or_open),
                        "high_values": _clean(high_values),
                        "low_values": _clean(low_values),
                        "close_values": _clean(close_values),
                    })

            def _to_working(column_name, values, scope='model', column=None,
                           sheet=None):
                """Write a column to a Working Data sheet (Tab 3)."""
                if isinstance(values, (pd.Series, np.ndarray)):
                    values = values.tolist()
                cleaned = []
                for v in values:
                    if v is None:
                        cleaned.append(None)
                    elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                        cleaned.append(None)
                    else:
                        cleaned.append(v)
                write_entry = {
                    "type": "working",
                    "column_name": str(column_name),
                    "values": cleaned,
                    "scope": str(scope),
                    "sheet": sheet or "Sheet 1",
                }
                if column is not None:
                    write_entry["column"] = str(column).upper().strip()
                tab_writes.append(write_entry)

            exec_context = {
                'pd': pd,
                'np': np,
                'datetime': datetime,
                'timedelta': timedelta,
                'data': data.copy(),
                'to_forecast': _to_forecast,
                'to_working': _to_working,
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

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(wrapped, exec_context)

            if 'result' in exec_context:
                result['data'] = exec_context['result']
            elif 'data' in exec_context:
                result['data'] = exec_context['data']
            elif 'df' in exec_context:
                result['data'] = exec_context['df']

            result['output'] = stdout_capture.getvalue()
            result['metrics'] = {
                'execution_time': time.time() - start_time,
                'original_rows': len(data),
            }

            result['success'] = True

            if monitor_callback:
                monitor_callback('complete', result['metrics'])

        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            stderr_output = stderr_capture.getvalue()
            if stderr_output:
                result['error'] += f"\nStderr: {stderr_output}"

            if monitor_callback:
                monitor_callback('error', {'error': str(e)})

            k2_logger.error(f"Strategy execution failed: {str(e)}", "DPE")

        finally:
            result['execution_time'] = time.time() - start_time

        return result


class DynamicPythonEngine:
    """Main DPE service for strategy execution."""

    def __init__(self):
        self.executor = StrategyExecutor()

    def execute_strategy(self, strategy_code: str, data: pd.DataFrame,
                         metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a trading strategy."""
        k2_logger.info("Executing strategy", "DPE")

        def monitor(status, info):
            k2_logger.info(f"Strategy execution {status}: {info}", "DPE")

        return self.executor.execute_code(strategy_code, data, monitor)


# Singleton instance
dpe_service = DynamicPythonEngine()
