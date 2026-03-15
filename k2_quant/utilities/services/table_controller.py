"""
Table Controller - Agent Loop Architecture (Persistent Engine)

Implements an iterative agent loop where the LLM can:
1. Execute SQL queries against the database
2. Execute Python code in a persistent namespace (variables survive across calls)
3. Read/write the Working Data workspace directly
4. Inspect results and decide next steps
5. Compose a final natural language response

The Python environment persists for the duration of one execute_command() call,
so computed variables, df modifications, and workspace state carry across
successive run_python tool invocations within the same agent turn.
"""

import json
import re
import requests
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
    """Agent-loop driven table manipulation with persistent Python engine."""

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
        initial_workspace: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """Run the agent loop with a persistent Python namespace.

        Parameters
        ----------
        table : str
            Database table backing the current model.
        command : str
            User message / instruction.
        conversation_history : list, optional
            Prior chat messages for context.
        step_callback : callable, optional
            Called with a short string for each intermediate tool step.
        initial_workspace : dict, optional
            ``{'model': DataFrame, 'global': DataFrame}`` — current state of
            the Working Data tab so the AI can read what is already displayed.
        """
        try:
            api_key = api_config.openai_api_key
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY")

            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            system_prompt = self._build_system_prompt(table, initial_workspace)
            tools = self._build_tools(table)

            messages: list = [{"role": "system", "content": system_prompt}]

            if conversation_history:
                for msg in conversation_history[-200:]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role in ("user", "assistant") and content:
                        messages.append({"role": role, "content": content})

            messages.append({"role": "user", "content": command})

            data_modified = False
            tab_writes: list = []

            # ── Persistent engine: load data & workspace once ────────
            df = db_manager.fetch_dataframe(table)

            workspace: Dict[str, Dict[str, list]] = {'model': {}, 'global': {}}
            if initial_workspace:
                for scope in ('model', 'global'):
                    ws_df = initial_workspace.get(scope)
                    if ws_df is not None and not ws_df.empty:
                        workspace[scope] = {
                            str(col): ws_df[col].tolist()
                            for col in ws_df.columns
                        }

            # ── Helper closures ──────────────────────────────────────

            def _clean_values(values):
                if isinstance(values, (pd.Series, np.ndarray)):
                    values = values.tolist()
                out = []
                for v in values:
                    if v is None:
                        out.append(None)
                    elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                        out.append(None)
                    else:
                        out.append(v)
                return out

            write_verifications: list = []

            def _to_working(column_name, values, scope='model', column=None):
                cleaned = _clean_values(values)
                workspace.setdefault(scope, {})[str(column_name)] = cleaned
                write_entry = {
                    "type": "working",
                    "column_name": str(column_name),
                    "values": cleaned,
                    "scope": str(scope),
                }
                if column is not None:
                    write_entry["column"] = str(column).upper().strip()
                tab_writes.append(write_entry)
                preview_n = min(10, len(cleaned))
                write_verifications.append({
                    "column": str(column_name),
                    "scope": str(scope),
                    "total_values": len(cleaned),
                    "first_values": cleaned[:preview_n],
                    "grid_column": str(column).upper().strip() if column else "auto",
                })

            def _read_working(scope='model', columns=None, head=None, tail=None):
                cols = workspace.get(scope, {})
                if not cols:
                    return pd.DataFrame()
                max_len = max(len(v) for v in cols.values())
                padded = {}
                target_cols = columns if columns else list(cols.keys())
                for k in target_cols:
                    if k in cols:
                        v = cols[k]
                        padded[k] = v + [None] * (max_len - len(v))
                ws_df = pd.DataFrame(padded)
                if head is not None:
                    ws_df = ws_df.head(head)
                if tail is not None:
                    ws_df = ws_df.tail(tail)
                return ws_df

            def _delete_working(column_name, scope='model'):
                workspace.get(scope, {}).pop(str(column_name), None)
                tab_writes.append({
                    "type": "delete_working",
                    "column_name": str(column_name),
                    "scope": str(scope),
                })

            def _workspace_info(scope='model'):
                cols = workspace.get(scope, {})
                if not cols:
                    return {}
                col_names = list(cols.keys())
                info = {}
                for idx, name in enumerate(col_names):
                    values = cols[name]
                    populated = sum(
                        1 for v in values
                        if v is not None and str(v).strip() != ''
                    )
                    info[name] = {
                        'letter': TableController._col_letter(idx),
                        'populated_rows': populated,
                        'first_3': values[:3] if values else [],
                    }
                return info

            def _to_forecast(set_index, open_values=None, high_values=None,
                             low_values=None, close_values=None):
                def _clean_forecast(vals):
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
                    "open_values": _clean_forecast(open_values),
                    "high_values": _clean_forecast(high_values),
                    "low_values": _clean_forecast(low_values),
                    "close_values": _clean_forecast(close_values),
                })

            # Persistent namespace shared across all run_python calls
            exec_globals = {
                "df": df,
                "pd": pd,
                "np": np,
                "datetime": datetime,
                "result": None,
                "to_working": _to_working,
                "read_working": _read_working,
                "to_forecast": _to_forecast,
                "delete_working": _delete_working,
                "workspace_info": _workspace_info,
                "_write_verifications": write_verifications,
            }

            # ── Agent loop ───────────────────────────────────────────
            iterations = 0

            while iterations < MAX_AGENT_ITERATIONS:
                iterations += 1

                response = client.chat.completions.create(
                    model="o3",
                    messages=messages,
                    tools=tools,
                    temperature=1,
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
                            tool_result = self._tool_run_python(
                                table, fn_args.get("code", ""), exec_globals)
                            if tool_result.get("_data_modified"):
                                data_modified = True
                        elif fn_name == "run_web_search":
                            tool_result = self._tool_run_web_search(
                                fn_args.get("query", ""),
                                fn_args.get("max_results", 5))
                        else:
                            tool_result = {"error": f"Unknown tool: {fn_name}"}

                        result_str = json.dumps(
                            {k: v for k, v in tool_result.items()
                             if not k.startswith("_")},
                            default=str,
                            indent=2,
                        )

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str[:50000],
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
                "display_message": "I've completed my analysis. "
                                   "Please check the intermediate steps above.",
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

    def _build_system_prompt(self, table: str,
                             initial_workspace: Optional[Dict] = None) -> str:
        """Build system prompt with schema, sample data, summary stats, and workspace state."""
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
- You have three tools: run_sql (execute SQL against PostgreSQL), run_python (execute Python/pandas code), and run_web_search (search the web for news, events, and qualitative data).
- Break complex tasks into steps. After each step, inspect the result before continuing.
- Validate your results. If a count seems implausible given the summary stats, double-check.
- When finished, provide a clear natural-language answer summarizing what you found or did.
- Show your reasoning and key numbers so the user can verify.
- For conversational messages (greetings, clarifications), respond naturally without running tools.
- When modifying data, briefly describe what changed so the user knows to check the dataframe.

TAB SYSTEM — The UI has three data tabs the user can see:
  Tab 1 (Current Data): Read-only stock data. Refreshes automatically.
  Tab 2 (Forecast Data): Pre-generated future timestamps for price projections.
  Tab 3 (Working Data): Editable grid workspace for intermediate results.
    IMPORTANT: Tab 3 works like a spreadsheet. Each column is INDEPENDENT and may
    have a different number of populated rows. A column with 370 values and a column
    with 5,000 values coexist in the same grid — they are NOT row-aligned.
    Always consult the WORKSPACE GRID MAP (below) to see each column's position,
    name, and populated row count before operating on workspace data.

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
- The Python environment is PERSISTENT across tool calls within this conversation turn.
  Variables, computed results, and workspace data survive between run_python calls.
  You do NOT need to re-derive values that were computed in a previous step.
- Only numeric columns (float/int) will be persisted. Non-numeric new columns are ignored.
- The DataFrame must always retain a 'timestamp' column.
- To return a computed value without modifying the table, assign to 'result' variable.

VERIFICATION:
- After every to_working() call, the system automatically returns a verification summary
  showing the column name and the first values written. ALWAYS inspect this verification
  to confirm the data looks correct before reporting success to the user.
- If the verification values look wrong, investigate and fix BEFORE moving on.

WEB SEARCH NOTES:
- Use run_web_search for qualitative analysis: news events, economic data, Fed decisions,
  geopolitical events, earnings, or any context requiring real-world information.
- Be specific in queries — include dates, ticker symbols, event names for best results.
- Results include source URLs. ALWAYS cite sources when presenting qualitative analysis.
- For historical date analysis, search for events around the specific date.

TAB HELPER FUNCTIONS (available inside run_python):
- to_working(column_name, values, scope='model', column=None)
    Write a column to the Working Data tab (Tab 3).
    column_name: string label for the column (written to row 1).
    values: list, Series, or ndarray of values (written starting at row 2).
    scope: 'model' (per-model workspace) or 'global' (shared workspace).
    column: optional Excel-style letter (e.g. 'A', 'H', 'AA') to place the data
    in a specific grid column. If None, auto-assigns to the next free column
    (or replaces an existing column with the same name).
    ALIGNMENT: Match the length of values to the source column you are deriving
    from. If computing from a 370-row column, write exactly 370 values — not the
    full grid row count. Check workspace_info() to verify dimensions before writing.

- read_working(scope='model', columns=None, head=None, tail=None)
    Read the current state of the Working Data tab (Tab 3) as a pandas DataFrame.
    scope: 'model' or 'global'.
    columns: optional list of column names to include (None = all columns).
    head: optional int, return only the first N rows.
    tail: optional int, return only the last N rows.
    Returns a DataFrame padded to a uniform row count. Shorter columns will have
    empty/None values at the end. Use workspace_info() to know each column's
    actual populated row count.
    IMPORTANT: Always use read_working() to answer questions about workspace data
    instead of re-querying the database.

- workspace_info(scope='model')
    Returns per-column metadata as a dict:
    {{column_name: {{'letter': 'A', 'populated_rows': int, 'first_3': list}}, ...}}
    Use this BEFORE computing derived columns to check how many populated rows
    the source column actually has. This prevents row-count mismatches.

- to_forecast(set_index, open_values=None, high_values=None, low_values=None, close_values=None)
    Write price projections to the Forecast Data tab (Tab 2).
    set_index: integer (1, 2, 3…) identifying the forecast scenario.
    Each list should align with the pre-generated future timestamps (up to 500 values).
    You may provide any subset of OHLC columns.

- delete_working(column_name, scope='model')
    Remove a column from the Working Data tab (Tab 3).
    column_name: exact name of the column to delete.
    scope: 'model' or 'global'.

{self._format_workspace_snapshot(initial_workspace)}"""

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
                        f"Execute Python code on the '{table}' data. The environment "
                        f"is persistent — variables, 'df', and workspace state survive "
                        f"between calls. 'pd', 'np', 'datetime' are available. "
                        f"Use read_working()/to_working() for workspace I/O. "
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
            {
                "type": "function",
                "function": {
                    "name": "run_web_search",
                    "description": (
                        "Search the web for real-time information using Tavily. "
                        "Use this for qualitative research: news events, economic data, "
                        "Fed announcements, geopolitical events, earnings reports, "
                        "or any context that requires current or historical web data. "
                        "Returns sourced results with URLs for citation."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "The search query. Be specific — include dates, "
                                    "ticker symbols, or event names for best results."
                                ),
                            },
                            "purpose": {
                                "type": "string",
                                "description": "One-line explanation of what this search is for",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Number of results to return (default 5, max 10)",
                            },
                        },
                        "required": ["query", "purpose"],
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

    def _tool_run_python(self, table: str, code: str,
                         exec_globals: dict) -> Dict[str, Any]:
        """Execute Python code in the persistent namespace."""
        try:
            df_before = exec_globals["df"].copy(deep=True)
            original_cols = set(df_before.columns)
            original_len = len(df_before)

            exec_globals["result"] = None

            exec(code, exec_globals)

            df_result = exec_globals.get("df", df_before)
            explicit_result = exec_globals.get("result")

            exec_globals["df"] = df_result

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

            original_ts = set(df_before["timestamp"].tolist())
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
                if col in df_before.columns:
                    try:
                        if df_before[col].equals(df_result[col]):
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

            verifications = exec_globals.get("_write_verifications", [])
            if verifications:
                pending = list(verifications)
                verifications.clear()
                resp["workspace_writes_verification"] = pending

            return resp

        except Exception as e:
            return {"error": str(e), "type": "error", "_data_modified": False}

    # ── web search executor ─────────────────────────────────────────

    def _tool_run_web_search(self, query: str,
                             max_results: int = 5) -> Dict[str, Any]:
        """Execute a Tavily web search and return structured results."""
        try:
            tavily_key = api_config.tavily_api_key
            if not tavily_key:
                return {"error": "TAVILY_API_KEY not configured", "type": "error"}

            max_results = min(max(1, max_results), 10)

            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True,
                    "search_depth": "advanced",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for r in data.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:1500],
                })

            return {
                "type": "web_search",
                "query": query,
                "answer": data.get("answer", ""),
                "results": results,
                "result_count": len(results),
            }
        except requests.exceptions.Timeout:
            return {"error": "Web search timed out after 30s", "type": "error"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Web search failed: {e}", "type": "error"}
        except Exception as e:
            return {"error": f"Web search error: {e}", "type": "error"}

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _col_letter(index: int) -> str:
        """Convert a 0-based column index to an Excel-style letter (A, B, … Z, AA, AB, …)."""
        result = ""
        while True:
            result = chr(ord('A') + index % 26) + result
            index = index // 26 - 1
            if index < 0:
                break
        return result

    @staticmethod
    def _format_workspace_snapshot(initial_workspace: Optional[Dict] = None) -> str:
        """Format workspace as a column grid map for AI spatial awareness."""
        if not initial_workspace:
            return "WORKSPACE GRID MAP: Empty (no columns in Working Data tab)."

        parts = ["WORKSPACE GRID MAP (Tab 3 — Working Data):"]
        parts.append("  Row 1 = column name (header).  Data starts at row 2.")
        has_content = False

        for scope in ('model', 'global'):
            ws_df = initial_workspace.get(scope)
            if ws_df is None or (hasattr(ws_df, 'empty') and ws_df.empty):
                continue
            if not isinstance(ws_df, pd.DataFrame):
                continue

            has_content = True
            col_letters = getattr(ws_df, 'attrs', {}).get('_col_letters', {})
            parts.append(f"\n  [{scope.upper()}]")

            for col_name in ws_df.columns:
                letter = col_letters.get(col_name,
                         TableController._col_letter(
                             list(ws_df.columns).index(col_name)))
                populated = int(
                    (ws_df[col_name].astype(str).str.strip() != '').sum()
                )
                if populated > 0:
                    range_str = f"{letter}2:{letter}{populated + 1}"
                    parts.append(
                        f"    {letter}1: {col_name:<40s} [{range_str}, {populated} values]")
                else:
                    parts.append(
                        f"    {letter}1: {col_name:<40s} [empty]")

        if not has_content:
            return "WORKSPACE GRID MAP: Empty (no columns in Working Data tab)."

        parts.append(
            "\n  WORKSPACE RULES:"
            "\n  - The grid uses Excel-style addressing: column letters (A, B, …) are static,"
            "\n    row 1 holds the column name, data starts at row 2."
            "\n  - Each column is INDEPENDENT and may have a different number of populated rows."
            "\n  - When the user references a column (e.g. 'for each Key Date'), operate on"
            "\n    THAT column's populated rows — not the total grid row count."
            "\n  - When writing derived columns, match the row count to the source column."
            "\n  - Columns at adjacent positions with the same row count are likely related."
            "\n  - Use workspace_info() at runtime to check per-column populated row counts."
            "\n  - Use read_working() to access data. Do NOT recompute existing columns"
            "\n    unless the user explicitly asks."
        )
        return "\n".join(parts)

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
