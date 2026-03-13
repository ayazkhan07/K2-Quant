"""
Table Controller - Agent Loop Architecture

Implements an iterative agent loop where the LLM can:
1. Execute SQL queries against the database
2. Execute Python code against pandas DataFrames
3. Inspect results and decide next steps
4. Compose a final natural language response

The loop continues until the LLM provides a final text response
or hits the maximum iteration limit.
"""

import json
import re
from decimal import Decimal
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

import pandas as pd
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.data.db_manager import db_manager
from k2_quant.utilities.config import api_config

MAX_AGENT_ITERATIONS = 15


class TableController(QObject):
    """Agent-loop driven table manipulation"""

    operation_complete = pyqtSignal(dict)
    operation_failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    # ── public entry point ──────────────────────────────────────────

    def execute_command(
        self,
        table: str,
        command: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        step_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Run the agent loop: plan, tool, observe, repeat, respond."""
        try:
            api_key = api_config.openai_api_key
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY")

            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            system_prompt = self._build_system_prompt(table)
            tools = self._build_tools(table)

            messages: list = [{"role": "system", "content": system_prompt}]

            if conversation_history:
                for msg in conversation_history[-20:]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role in ("user", "assistant") and content:
                        messages.append({"role": role, "content": content})

            messages.append({"role": "user", "content": command})

            data_modified = False
            tab_writes: list = []
            iterations = 0

            while iterations < MAX_AGENT_ITERATIONS:
                iterations += 1

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=tools,
                    temperature=0.1,
                )

                choice = response.choices[0]
                assistant_msg = choice.message

                if assistant_msg.tool_calls:
                    messages.append(assistant_msg)

                    for tool_call in assistant_msg.tool_calls:
                        fn_name = tool_call.function.name
                        try:
                            fn_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            fn_args = {}

                        purpose = fn_args.get("purpose", fn_name)
                        if step_callback:
                            step_callback(purpose)

                        if fn_name == "run_sql":
                            tool_result = self._tool_run_sql(fn_args.get("sql", ""))
                        elif fn_name == "run_python":
                            tool_result = self._tool_run_python(table, fn_args.get("code", ""), tab_writes)
                            if tool_result.get("_data_modified"):
                                data_modified = True
                        else:
                            tool_result = {"error": f"Unknown tool: {fn_name}"}

                        result_str = json.dumps(
                            {k: v for k, v in tool_result.items() if not k.startswith("_")},
                            default=str,
                            indent=2,
                        )

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str[:8000],
                        })

                    continue

                final_text = assistant_msg.content or "Done."

                result = {
                    "success": True,
                    "display_message": final_text,
                    "data_modified": data_modified,
                    "iterations": iterations,
                    "_tab_writes": tab_writes,
                }
                self.operation_complete.emit(result)
                return result

            result = {
                "success": True,
                "display_message": "I've completed my analysis. Please check the intermediate steps above.",
                "data_modified": data_modified,
                "iterations": iterations,
                "_tab_writes": tab_writes,
            }
            self.operation_complete.emit(result)
            return result

        except Exception as e:
            msg = str(e)
            k2_logger.error(f"Agent loop failed: {msg}", "TABLE_CTRL")
            self.operation_failed.emit(msg)
            return {"success": False, "error": msg}

    # ── system prompt ──────────────────────────────────────────────

    def _build_system_prompt(self, table: str) -> str:
        """Build system prompt with schema, sample data, and summary stats."""
        try:
            with db_manager.get_connection() as conn:
                with db_manager.get_cursor(conn) as cur:
                    cur.execute(
                        """SELECT column_name, data_type
                           FROM information_schema.columns
                           WHERE table_name = %s
                           ORDER BY ordinal_position""",
                        (table,),
                    )
                    columns = cur.fetchall()

                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cur.fetchone()[0]

                    sample_col_names = [c[0] for c in columns[:12]]
                    sample_cols_sql = ", ".join(sample_col_names)

                    cur.execute(f"SELECT {sample_cols_sql} FROM {table} ORDER BY timestamp ASC LIMIT 3")
                    first_rows = cur.fetchall()
                    col_headers = [desc[0] for desc in cur.description]

                    cur.execute(f"SELECT {sample_cols_sql} FROM {table} ORDER BY timestamp DESC LIMIT 3")
                    last_rows = cur.fetchall()

                    price_cols = [c[0] for c in columns if c[0] in ("open", "high", "low", "close", "volume", "vwap")]
                    stats = {}
                    for pc in price_cols:
                        try:
                            cur.execute(f"SELECT MIN({pc}), MAX({pc}), AVG({pc})::numeric(12,2) FROM {table}")
                            row = cur.fetchone()
                            stats[pc] = {
                                "min": float(row[0]) if row[0] is not None else None,
                                "max": float(row[1]) if row[1] is not None else None,
                                "avg": float(row[2]) if row[2] is not None else None,
                            }
                        except Exception:
                            pass

            column_info = [f"{c[0]} ({c[1]})" for c in columns]

            def fmt_rows(rows):
                lines = []
                for r in rows:
                    vals = [f"{col_headers[i]}={r[i]}" for i in range(len(r))]
                    lines.append("  " + ", ".join(str(v) for v in vals))
                return "\n".join(lines)

            stats_text = "\n".join(
                f"  {col}: min={s['min']}, max={s['max']}, avg={s['avg']}"
                for col, s in stats.items()
            ) if stats else "  (none available)"

            return f"""You are a quantitative financial analyst working with a stock market dataset.

TABLE: {table}
COLUMNS: {', '.join(column_info)}
ROWS: {row_count:,}

EARLIEST DATA (first 3 rows):
{fmt_rows(first_rows)}

LATEST DATA (last 3 rows):
{fmt_rows(last_rows)}

SUMMARY STATISTICS:
{stats_text}

INSTRUCTIONS:
- You have two tools: run_sql (execute SQL against PostgreSQL) and run_python (execute Python/pandas code).
- Break complex tasks into steps. After each step, inspect the result before continuing.
- Validate your results. If a count seems implausible given the summary stats, double-check.
- When finished, provide a clear natural-language answer summarizing what you found or did.
- Show your reasoning and key numbers so the user can verify.
- For conversational messages (greetings, clarifications), respond naturally without running tools.
- When modifying data, briefly describe what changed so the user knows to check the dataframe.

TAB SYSTEM — The UI has three data tabs the user can see:
  Tab 1 (Current Data): Read-only stock data. Refreshes automatically.
  Tab 2 (Forecast Data): Pre-generated future timestamps for price projections.
  Tab 3 (Working Data): Editable workspace for intermediate results.

ROUTING RULES (important):
- Intermediate computation results (elasticity, pattern matches, filtered lists, etc.)
  MUST go to Tab 3 via to_working(). Do NOT add intermediate columns to the main DB table.
- Final price projections MUST go to Tab 2 via to_forecast().
- Only use direct df column modifications when the user explicitly asks to alter the main dataset.

SQL NOTES:
- Table name is: {table}
- Use standard PostgreSQL syntax.
- For floating-point comparisons use tolerance: ABS(col - value) < 0.01

PYTHON NOTES:
- 'df' is the full table as a pandas DataFrame.
- 'pd', 'np', 'datetime' are available.
- Only numeric columns (float/int) will be persisted. Non-numeric new columns are ignored.
- The DataFrame must always retain a 'timestamp' column.
- To return a computed value without modifying the table, assign to 'result' variable.

TAB HELPER FUNCTIONS (available inside run_python):
- to_working(column_name, values, scope='model')
    Write a column to the Working Data tab (Tab 3).
    column_name: string label for the column.
    values: list, Series, or ndarray of values.
    scope: 'model' (per-model workspace) or 'global' (shared workspace).
    All values will be displayed; no row limit.

- to_forecast(set_index, open_values=None, high_values=None, low_values=None, close_values=None)
    Write price projections to the Forecast Data tab (Tab 2).
    set_index: integer (1, 2, 3…) identifying the forecast scenario.
    Each list should align with the pre-generated future timestamps (up to 500 values).
    You may provide any subset of OHLC columns.

- delete_working(column_name, scope='model')
    Remove a column from the Working Data tab (Tab 3).
    column_name: exact name of the column to delete.
    scope: 'model' or 'global'."""

        except Exception as e:
            k2_logger.error(f"Failed to build system prompt: {e}", "TABLE_CTRL")
            return f"You are a data analyst. Table: {table}. Use the run_sql and run_python tools."

    # ── tool definitions ───────────────────────────────────────────

    def _build_tools(self, table: str) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_sql",
                    "description": (
                        f"Execute a SQL query against the PostgreSQL database. "
                        f"The main table is '{table}'."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "The SQL query to execute",
                            },
                            "purpose": {
                                "type": "string",
                                "description": "One-line explanation of what this query does",
                            },
                        },
                        "required": ["sql", "purpose"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_python",
                    "description": (
                        f"Execute Python code on the '{table}' data. The full table is loaded "
                        f"as 'df' (pandas DataFrame). 'pd', 'np', 'datetime' are available. "
                        f"Changes to df are persisted back to the database (numeric columns only). "
                        f"Assign to 'result' variable to return a computed value."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": (
                                    "Python code to execute. Use 'df' for the DataFrame. "
                                    "Assign to 'result' for output values."
                                ),
                            },
                            "purpose": {
                                "type": "string",
                                "description": "One-line explanation of what this code does",
                            },
                        },
                        "required": ["code", "purpose"],
                    },
                },
            },
        ]

    # ── tool executors ─────────────────────────────────────────────

    def _tool_run_sql(self, sql: str) -> Dict[str, Any]:
        """Execute SQL and return structured result for the LLM."""
        try:
            with db_manager.get_connection() as conn:
                with db_manager.get_cursor(conn) as cur:
                    cur.execute(sql)

                    if sql.strip().upper().startswith("SELECT"):
                        col_names = [desc[0] for desc in cur.description] if cur.description else []
                        rows = cur.fetchall()

                        if len(rows) == 1 and len(rows[0]) == 1:
                            return {"result": self._serialize(rows[0][0]), "type": "single_value"}

                        if len(rows) <= 50:
                            formatted = [
                                {col_names[i]: self._serialize(row[i]) for i in range(len(col_names))}
                                for row in rows
                            ]
                            return {"columns": col_names, "rows": formatted, "count": len(rows), "type": "table"}

                        first_10 = [
                            {col_names[i]: self._serialize(row[i]) for i in range(len(col_names))}
                            for row in rows[:10]
                        ]
                        return {
                            "columns": col_names,
                            "total_rows": len(rows),
                            "first_10": first_10,
                            "type": "large_table",
                            "note": f"Showing first 10 of {len(rows)} rows.",
                        }

                    conn.commit()
                    return {"rows_affected": cur.rowcount, "type": "modification"}

        except Exception as e:
            return {"error": str(e), "type": "error"}

    def _tool_run_python(self, table: str, code: str, tab_writes: list = None) -> Dict[str, Any]:
        """Execute Python and return structured result for the LLM."""
        try:
            df = db_manager.fetch_dataframe(table)
            original = df.copy(deep=True)
            original_cols = set(df.columns)
            original_len = len(df)

            if tab_writes is None:
                tab_writes = []

            def _to_working(column_name, values, scope='model'):
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
                tab_writes.append({
                    "type": "working",
                    "column_name": str(column_name),
                    "values": cleaned,
                    "scope": str(scope),
                })

            def _to_forecast(set_index, open_values=None, high_values=None,
                             low_values=None, close_values=None):
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
                tab_writes.append({
                    "type": "forecast",
                    "set_index": int(set_index),
                    "open_values": _clean(open_values),
                    "high_values": _clean(high_values),
                    "low_values": _clean(low_values),
                    "close_values": _clean(close_values),
                })

            def _delete_working(column_name, scope='model'):
                tab_writes.append({
                    "type": "delete_working",
                    "column_name": str(column_name),
                    "scope": str(scope),
                })

            exec_globals = {
                "df": df,
                "pd": pd,
                "np": np,
                "datetime": datetime,
                "result": None,
                "to_working": _to_working,
                "to_forecast": _to_forecast,
                "delete_working": _delete_working,
            }
            exec(code, exec_globals)

            df_result = exec_globals.get("df", df)
            explicit_result = exec_globals.get("result")

            data_modified = False
            modifications: List[str] = []

            if "timestamp" not in df_result.columns:
                if explicit_result is not None:
                    return {
                        "result": self._serialize(explicit_result),
                        "type": "computed_value",
                        "_data_modified": False,
                    }
                raise ValueError("DataFrame must contain 'timestamp' column")

            original_ts = set(original["timestamp"].tolist())
            result_ts = set(df_result["timestamp"].tolist())
            new_ts = sorted(result_ts - original_ts)
            deleted_ts = sorted(original_ts - result_ts)

            candidate_new_cols = list(set(df_result.columns) - original_cols)
            new_numeric_cols = [
                c for c in candidate_new_cols
                if c != "timestamp" and (
                    pd.api.types.is_float_dtype(df_result[c])
                    or pd.api.types.is_integer_dtype(df_result[c])
                )
            ]
            skipped_cols = [c for c in candidate_new_cols if c not in new_numeric_cols and c != "timestamp"]

            if deleted_ts:
                with db_manager.get_connection() as conn:
                    with db_manager.get_cursor(conn) as cur:
                        cur.execute(f"DELETE FROM {table} WHERE timestamp = ANY(%s)", (deleted_ts,))
                    conn.commit()
                modifications.append(f"Deleted {len(deleted_ts)} rows")
                data_modified = True

            if new_numeric_cols:
                with db_manager.get_connection() as conn:
                    with db_manager.get_cursor(conn) as cur:
                        for col in new_numeric_cols:
                            cur.execute(
                                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION"
                            )
                    conn.commit()
                modifications.append(f"Added columns: {', '.join(new_numeric_cols)}")
                data_modified = True

            if new_ts:
                df_new = df_result[df_result["timestamp"].isin(new_ts)].copy()
                allowed = ["timestamp"] + [
                    c for c in df_new.columns if c in original_cols or c in new_numeric_cols
                ]
                df_new = df_new[allowed]
                inserted = db_manager.bulk_insert_dataframe(table, df_new)
                modifications.append(f"Inserted {inserted} new rows")
                data_modified = True

            for col in df_result.columns:
                if col == "timestamp":
                    continue
                if not (
                    pd.api.types.is_float_dtype(df_result[col])
                    or pd.api.types.is_integer_dtype(df_result[col])
                ):
                    continue
                if col in original.columns:
                    try:
                        if original[col].equals(df_result[col]):
                            continue
                    except Exception:
                        pass
                db_manager.bulk_update_column_by_timestamp(
                    table, col, df_result["timestamp"], df_result[col]
                )
                if col not in new_numeric_cols:
                    modifications.append(f"Updated column: {col}")
                data_modified = True

            resp: Dict[str, Any] = {
                "type": "python_executed",
                "rows_before": original_len,
                "rows_after": len(df_result),
                "modifications": modifications if modifications else ["No data changes"],
                "_data_modified": data_modified,
            }

            if explicit_result is not None:
                resp["result"] = self._serialize(explicit_result)

            if skipped_cols:
                resp["skipped_non_numeric_columns"] = skipped_cols

            return resp

        except Exception as e:
            return {"error": str(e), "type": "error", "_data_modified": False}

    # ── helpers ─────────────────────────────────────────────────────

    def _serialize(self, value: Any) -> Any:
        """Make a value JSON-safe."""
        if value is None:
            return None
        if isinstance(value, (Decimal, np.floating)):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, (np.ndarray, pd.Series)):
            return value.tolist()
        if isinstance(value, pd.DataFrame):
            return value.head(20).to_dict(orient="records")
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (int, float, str, bool)):
            return value
        return str(value)
