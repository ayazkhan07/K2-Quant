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

            result['metrics'] = {
                'execution_time': time.time() - start_time,
                'original_rows': len(data),
            }

            result['success'] = True

            if monitor_callback:
                monitor_callback('complete', result['metrics'])

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
                         metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a trading strategy."""
        k2_logger.info("Executing strategy", "DPE")

        def monitor(status, info):
            k2_logger.info(f"Strategy execution {status}: {info}", "DPE")

        return self.executor.execute_code(strategy_code, data, monitor)


# Singleton instance
dpe_service = DynamicPythonEngine()
