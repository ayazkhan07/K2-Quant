"""
Table Controller - AI-driven table manipulation with enhanced result interpretation

Implements:
- Plan generation via OpenAI (unrestricted operations per user preference)
- Direct SQL execution with natural language interpretation
- Python-based DataFrame transforms restricted to numeric columns
- New row inserts (upsert behavior for inserts only; existing rows updated per-column)
- Enhanced result interpretation for complex queries

Notes:
- Only numeric columns (float/integer) are created/updated by Python path
- Non-numeric new columns are ignored and reported as skipped
"""

import json
import re
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime

import pandas as pd
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.data.db_manager import db_manager
from k2_quant.utilities.config import api_config


class TableController(QObject):
    """Full table manipulation focused on numeric operations"""

    # Signals for UI communication
    operation_complete = pyqtSignal(dict)
    operation_failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def execute_command(self, table: str, command: str, conversation_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Plan → execute SQL/Python → describe results with LLM prose (including tables)."""
        try:
            plan = self._get_ai_execution_plan(table, command, conversation_state)
            if not plan:
                result = {'success': False, 'error': 'Could not generate execution plan'}
                self.operation_failed.emit(result['error'])
                return result

            # Clarification short-circuit
            if plan.get('needs_clarification'):
                return {
                    'success': False,
                    'reason': 'needs_clarification',
                    'clarifying_question': plan.get('clarifying_question', 'Could you clarify your request?')
                }

            result_type = plan.get('result_type', 'unknown')
            execute_then_describe = plan.get('execute_then_describe', True)

            # Execute the plan
            if plan.get('sql'):
                result = self._execute_sql(plan['sql'], plan.get('params'))
            elif plan.get('python_code'):
                result = self._execute_python(table, plan['python_code'])
            else:
                result = {'success': False, 'error': 'No executable code in plan'}

            if not result.get('success'):
                self.operation_failed.emit(result.get('error', 'Unknown error'))
                return result

            # Second LLM pass for natural language with table support
            if execute_then_describe and result.get('query_result') is not None:
                try:
                    final_response = self._generate_natural_response(
                        command=command,
                        query_result=result.get('query_result'),
                        columns=result.get('columns', []),
                        result_type=result_type,
                        conversation_state=conversation_state or {},
                        plan_metadata={
                            'referent_resolution': plan.get('referent_resolution'),
                            'column_candidates': plan.get('column_candidates')
                        }
                    )
                    if final_response:
                        result['interpreted_result'] = final_response
                        result['display_message'] = final_response
                except Exception as e:
                    k2_logger.warning(f"Natural response generation failed: {e}", "TABLE_CTRL")
                    result['display_message'] = "Query completed successfully"

            # Update answer_to_remember with actual result
            if plan.get('answer_to_remember'):
                try:
                    answer = plan['answer_to_remember'].copy()
                    answer['value'] = result.get('query_result')
                    # Persist which column was used when available
                    if 'column' not in answer:
                        col_used = plan.get('column_used')
                        if col_used:
                            answer['column'] = col_used
                    result['answer_to_remember'] = answer
                except Exception:
                    pass

            # Pass-through metadata
            result['referent_resolution'] = plan.get('referent_resolution')
            result['follow_up_suggestion'] = plan.get('follow_up_suggestion')

            self.operation_complete.emit(result)
            return result

        except Exception as e:
            msg = str(e)
            k2_logger.error(f"execute_command failed: {msg}", "TABLE_CTRL")
            self.operation_failed.emit(msg)
            return {'success': False, 'error': msg}

    def _interpret_result(self, query_result: Any, template: str, result_type: str = 'unknown') -> str:
        """Enhanced interpretation that handles various result types"""
        try:
            # Normalize common placeholder variants like "{{0}}" -> "{0}"
            template = self._normalize_template_placeholders(template)
            # Handle empty results
            if query_result is None:
                return "No data found for your query"
            
            # Handle list results
            if isinstance(query_result, list):
                if len(query_result) == 0:
                    return "No results found matching your criteria"
                
                # Single row with single value
                elif len(query_result) == 1 and isinstance(query_result[0], tuple) and len(query_result[0]) == 1:
                    value = query_result[0][0]
                    return self._format_value_with_template(value, template)
                
                # Single row with multiple values
                elif len(query_result) == 1 and isinstance(query_result[0], tuple):
                    try:
                        return template.format(*query_result[0])
                    except:
                        # If template doesn't match, describe the result
                        return f"{template}. Found {len(query_result[0])} values"
                
                # Multiple rows with single values each
                elif all(isinstance(row, tuple) and len(row) == 1 for row in query_result):
                    values = [row[0] for row in query_result]
                    
                    # For small lists, show all values
                    if len(values) <= 10:
                        value_list = ', '.join(self._format_value(v) for v in values)
                        if '{values}' in template:
                            return template.replace('{values}', value_list)
                        elif '{0}' in template:
                            # AI expects single placeholder but got list
                            return f"{template.split('{0}')[0]}{value_list}"
                        else:
                            return f"{template}: {value_list}"
                    else:
                        # For large lists, summarize
                        sample = ', '.join(self._format_value(v) for v in values[:5])
                        return f"{template}. Found {len(values)} results (first 5: {sample}...)"
                
                # Complex multi-row, multi-column results
                else:
                    row_count = len(query_result)
                    col_count = len(query_result[0]) if query_result else 0
                    return f"{template}. Retrieved {row_count} rows with {col_count} columns each"
            
            # Handle single value results
            elif isinstance(query_result, (int, float, str, bool, Decimal, np.floating, np.integer)):
                return self._format_value_with_template(query_result, template)
            
            # Fallback
            else:
                # Don't append "Result:" - just format the template properly
                if isinstance(query_result, (int, float, str, Decimal, np.floating, np.integer)):
                    return self._format_value_with_template(query_result, template)
                else:
                    return f"{template}: {str(query_result)[:100]}"
                
        except Exception as e:
            k2_logger.warning(f"Could not interpret result: {e}", "TABLE_CTRL")
            # Provide useful fallback without exposing error
            if isinstance(query_result, list):
                return f"Found {len(query_result)} results"
            else:
                return "Query completed successfully"

    def _format_value(self, value: Any) -> str:
        """Format a single value for display"""
        if value is None:
            return "NULL"
        elif isinstance(value, (float, np.floating, Decimal)):
            # Format floats with appropriate precision
            numeric_value = float(value)
            if abs(numeric_value) < 0.01 and numeric_value != 0:
                return f"{numeric_value:.4f}"
            elif abs(numeric_value) >= 1000:
                return f"{numeric_value:,.2f}"
            else:
                return f"{numeric_value:.2f}"
        elif isinstance(value, (int, np.integer)):
            return f"{value:,}"
        elif isinstance(value, bool):
            return str(value)
        else:
            return str(value)

    def _format_value_with_template(self, value: Any, template: str) -> str:
        """Apply template to a single value with proper formatting"""
        formatted = self._format_value(value)
        template = self._normalize_template_placeholders(template)
        
        # Try Python formatting first
        try:
            return template.format(formatted)
        except Exception:
            pass

        # Direct replacement of common tokens
        try:
            replaced = template.replace('{0}', formatted)
            replaced = replaced.replace('{value}', formatted).replace('{values}', formatted)
            # Regex-based replacement to catch variants with hidden spaces/characters
            replaced = re.sub(r"\{\s*0\s*\}", formatted, replaced)
            replaced = re.sub(r"\{\s*value\s*\}", formatted, replaced)
            if replaced != template:
                return replaced
        except Exception:
            pass

        # As a last resort, replace any brace-wrapped token with the value
        try:
            generic = re.sub(r"\{[^\}]*\}", formatted, template)
            if generic:
                return generic
        except Exception:
            pass

        return f"{template}: {formatted}"

    def _normalize_template_placeholders(self, template: str) -> str:
        """Normalize AI-provided placeholders to Python format style.
        Supports variants like "{{0}}", "{ 0 }", and "{{values}}" -> "{values}".
        """
        try:
            if not isinstance(template, str):
                return str(template)
            normalized = template
            # Convert double-brace escaped placeholders to single braces
            normalized = normalized.replace('{{values}}', '{values}')
            normalized = normalized.replace('{{0}}', '{0}')
            normalized = normalized.replace('{{1}}', '{1}')
            normalized = normalized.replace('{{2}}', '{2}')
            # Remove spaces inside braces like "{ 0 }" or "{ values }"
            normalized = re.sub(r'\{\s*values\s*\}', '{values}', normalized)
            normalized = re.sub(r'\{\s*0\s*\}', '{0}', normalized)
            normalized = re.sub(r'\{\s*1\s*\}', '{1}', normalized)
            normalized = re.sub(r'\{\s*2\s*\}', '{2}', normalized)
            return normalized
        except Exception:
            return template

    def _get_ai_execution_plan(self, table: str, command: str, conversation_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get execution plan from OpenAI with enhanced financial analysis guidance"""
        try:
            # Gather context
            with db_manager.get_connection() as conn:
                with db_manager.get_cursor(conn) as cur:
                    cur.execute(
                        """
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = %s
                        ORDER BY ordinal_position
                        """,
                        (table,),
                    )
                    columns = cur.fetchall()

                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cur.fetchone()[0]

            column_info = [f"{col[0]} ({col[1]})" for col in columns]

            api_key = api_config.openai_api_key
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY for plan generation")

            prompt = f"""
Table: {table}
Columns: {', '.join(column_info)}
Primary Key: timestamp
Rows: {row_count}

User request: {command}
Recent context for pronoun resolution: {json.dumps((conversation_state or {}).get('last_answers', [])[-5:], separators=(',', ':'))}

Generate SQL or Python to accomplish this.

SQL Guidelines:
- For data operations: query {table} directly
- For counting rows: SELECT COUNT(*) FROM {table}
- For counting columns: SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{table}'
- For lists of values (like column names): Use STRING_AGG(column_name, ', ') to combine into single string
- For financial metrics: Use appropriate aggregation functions (MIN, MAX, AVG, STDDEV, etc.)
- For time series: Use window functions with OVER (ORDER BY date_time_market)
- You may DELETE/UPDATE/ALTER/INSERT as needed

CRITICAL SQL Rules:
- Resolve pronouns using the recent context above.
- Use ACTUAL resolved numeric values; NEVER output placeholders like {{value}}, {{open_price}}, ?, $1.
- For floating point comparison use tolerance: WHERE ABS(col - 123.45) < 0.01 or ROUND(col, 2) = 123.45.
- Choose the correct price column (open/high/low/close) per the request/context.

Python Guidelines:
- If Python code is used, operate on 'df' (pandas DataFrame)
- Python updates must be NUMERIC-ONLY (float/integer)
- For new rows, include unique 'timestamp'

Return strict JSON only:
{{
    "explanation": "what this does",
    "operation_type": "ADD_COLUMN|UPDATE|DELETE|DDL|QUERY|COMPLEX",
    "sql": "SQL command if applicable",
    "params": [],
    "python_code": "Python code if needed",
    "result_interpretation": "Natural language template for results. Use {{0}} for single values, {{0}}, {{1}} for multiple values, or {{values}} for lists",
    "result_type": "single_value|single_row|multiple_rows|list|aggregate"
}}

Examples of good result_interpretation:
- For COUNT: "The dataset contains {{0}} rows"
- For MIN/MAX: "The lowest value is {{0}}"
- For lists: "The columns are: {{0}}"
- For multiple metrics: "Average: {{0}}, Minimum: {{1}}, Maximum: {{2}}"
"""

            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a financial data expert. Output only valid JSON. Resolve pronouns using provided context. Never return SQL with placeholders; embed actual resolved numeric values with float tolerance (e.g., ABS(col - 123.45) < 0.01). Always use STRING_AGG for combining multiple text values into one result."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                content = response.choices[0].message.content
            except Exception:
                import openai
                openai.api_key = api_key
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a financial data expert. Output only valid JSON. Resolve pronouns using provided context. Never return SQL with placeholders; embed actual resolved numeric values with float tolerance (e.g., ABS(col - 123.45) < 0.01). Always use STRING_AGG for combining multiple text values into one result."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                content = response.choices[0].message.content

            # Parse JSON response
            content_str = str(content).strip()
            try:
                return json.loads(content_str)
            except Exception:
                match = re.search(r'\{[\s\S]*\}', content_str)
                if match:
                    return json.loads(match.group(0))
                raise ValueError("Model did not return valid JSON")
                
        except Exception as e:
            k2_logger.error(f"AI plan generation failed: {e}", "TABLE_CTRL")
            return None

    def _execute_sql(self, sql: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Execute SQL and return results with column names."""
        try:
            with db_manager.get_connection() as conn:
                with db_manager.get_cursor(conn) as cur:
                    if params:
                        cur.execute(sql, params)
                    else:
                        cur.execute(sql)

                    if sql.strip().upper().startswith('SELECT'):
                        columns = [desc[0] for desc in cur.description] if cur.description else []
                        results = cur.fetchall()

                        # For single value results, extract directly
                        if len(results) == 1 and len(results[0]) == 1:
                            query_result = results[0][0]
                        else:
                            query_result = results

                        return {
                            'success': True,
                            'sql_executed': sql[:200] + ('...' if len(sql) > 200 else ''),
                            'query_result': query_result,
                            'columns': columns
                        }
                    else:
                        affected = cur.rowcount or 0
                        conn.commit()
                        return {
                            'success': True,
                            'sql_executed': sql[:200] + ('...' if len(sql) > 200 else ''),
                            'rows_affected': affected
                        }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _generate_natural_response(self, command: str, query_result: Any, columns: List[str], 
                                   result_type: str, conversation_state: Dict[str, Any], 
                                   plan_metadata: Dict[str, Any]) -> Optional[str]:
        """Generate natural language with table formatting when appropriate."""
        if query_result is None:
            return "No data found for your query"

        api_key = api_config.openai_api_key
        if not api_key:
            return None

        # Smart summarization
        summary = self._summarize_result_for_prompt(query_result, result_type, columns)
        
        # Detect if user wants tabular display
        wants_table = any(word in command.lower() for word in ['table', 'show', 'list', 'display', 'compare'])

        recent_answers = (conversation_state or {}).get('last_answers', [])[-3:]
        recent_ctx = [{'label': a.get('label', ''), 'value': a.get('value')} for a in recent_answers]

        system_prompt = """You are a conversational data analyst.
Write a COMPLETE natural-language answer.
- NEVER use placeholders like {0} or {values}
- If data is tabular and user wants details, format as markdown table
- For tables >10 rows, show first 5-10 rows and note "Showing X of Y rows"
- Keep tables to essential columns only

For tables, use this format:
| Column1 | Column2 | Column3 |
|---------|---------|---------|
| Value1  | Value2  | Value3  |

Return ONLY the final prose and/or table, no JSON."""

        user_prompt = f"""User query: {command}
Wants table: {wants_table}
Columns available: {columns}
Result summary: {json.dumps(summary, separators=(',', ':'))}
Recent context: {json.dumps(recent_ctx, separators=(',', ':'))}"""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            k2_logger.error(f"LLM call failed: {e}", "TABLE_CTRL")
            return None

    def _summarize_result_for_prompt(self, query_result: Any, result_type: str, columns: List[str] = None) -> Dict[str, Any]:
        """Smart summarization that preserves structure for tables."""
        try:
            # Single value
            if isinstance(query_result, (int, float, str, bool, Decimal)):
                return {'type': 'single_value', 'value': self._format_value(query_result)}
            
            # List results
            if isinstance(query_result, list) and len(query_result) > 0:
                # Multi-column table data
                if isinstance(query_result[0], tuple) and len(query_result[0]) > 1:
                    sample_size = min(10, len(query_result))
                    return {
                        'type': 'table',
                        'columns': columns or [],
                        'rows': len(query_result),
                        'sample': [
                            [self._format_value(cell) for cell in row]
                            for row in query_result[:sample_size]
                        ],
                        'truncated': len(query_result) > sample_size
                    }
                
                # Single column list
                if len(query_result) <= 10:
                    return {
                        'type': 'list',
                        'count': len(query_result),
                        'values': [self._format_value(row[0] if isinstance(row, tuple) else row) 
                                  for row in query_result]
                    }
                
                # Large list - intelligent sampling
                return {
                    'type': 'large_list',
                    'count': len(query_result),
                    'first_5': [self._format_value(row[0] if isinstance(row, tuple) else row) 
                               for row in query_result[:5]],
                    'last_2': [self._format_value(row[0] if isinstance(row, tuple) else row) 
                              for row in query_result[-2:]] if len(query_result) > 7 else []
                }
            
            return {'type': 'unknown', 'preview': str(query_result)[:200]}
        except Exception as e:
            k2_logger.warning(f"Summarization failed: {e}", "TABLE_CTRL")
            return {'type': 'unknown'}

    def _execute_python(self, table: str, code: str) -> Dict[str, Any]:
        """Execute Python to transform a DataFrame and persist numeric changes."""
        try:
            # Load
            df = db_manager.fetch_dataframe(table)
            original = df.copy(deep=True)
            original_cols = set(df.columns)

            # Provide execution environment
            exec_globals = {
                'df': df,
                'pd': pd,
                'np': np,
                'datetime': datetime
            }
            exec(code, exec_globals)
            df_result = exec_globals.get('df', df)

            if 'timestamp' not in df_result.columns:
                raise ValueError("Resulting DataFrame must contain 'timestamp' as the row identity")

            # Determine row diffs
            original_ts = set(original['timestamp'].tolist())
            result_ts = set(df_result['timestamp'].tolist())
            new_ts = sorted(result_ts - original_ts)
            deleted_ts = sorted(original_ts - result_ts)

            # Determine new columns (numeric only)
            candidate_new_cols = list(set(df_result.columns) - original_cols)
            new_numeric_cols = [
                c for c in candidate_new_cols
                if c != 'timestamp' and (
                    pd.api.types.is_float_dtype(df_result[c]) or pd.api.types.is_integer_dtype(df_result[c])
                )
            ]
            skipped_new_cols = [c for c in candidate_new_cols if c not in new_numeric_cols and c != 'timestamp']

            # Persist deletions
            if deleted_ts:
                with db_manager.get_connection() as conn:
                    with db_manager.get_cursor(conn) as cur:
                        cur.execute(
                            f"DELETE FROM {table} WHERE timestamp = ANY(%s)",
                            (deleted_ts,)
                        )
                    conn.commit()

            # Ensure numeric new columns exist
            if new_numeric_cols:
                with db_manager.get_connection() as conn:
                    with db_manager.get_cursor(conn) as cur:
                        for col in new_numeric_cols:
                            dtype = self._infer_sql_type(df_result[col])
                            cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {dtype}")
                    conn.commit()

            # Insert new rows
            rows_inserted = 0
            if new_ts:
                df_new = df_result[df_result['timestamp'].isin(new_ts)].copy()
                allowed_cols = ['timestamp'] + [c for c in df_new.columns if c in original_cols or c in new_numeric_cols]
                df_new = df_new[allowed_cols]
                rows_inserted = db_manager.bulk_insert_dataframe(table, df_new)

            # Update existing rows
            modified_cols: List[str] = []
            for col in df_result.columns:
                if col == 'timestamp':
                    continue
                if not (pd.api.types.is_float_dtype(df_result[col]) or pd.api.types.is_integer_dtype(df_result[col])):
                    continue

                if col in original.columns:
                    try:
                        if original[col].equals(df_result[col]):
                            continue
                    except Exception:
                        pass

                modified_cols.append(col)
                ts_series = df_result['timestamp']
                val_series = df_result[col]
                db_manager.bulk_update_column_by_timestamp(table, col, ts_series, val_series)

            return {
                'success': True,
                'new_columns': new_numeric_cols if new_numeric_cols else None,
                'skipped_columns': skipped_new_cols if skipped_new_cols else None,
                'modified_columns': modified_cols if modified_cols else None,
                'rows_deleted': len(deleted_ts) if deleted_ts else None,
                'rows_inserted': rows_inserted if rows_inserted else None,
                'rows_remaining': len(df_result)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _infer_sql_type(self, series: pd.Series) -> str:
        """Infer SQL type for new numeric columns."""
        if pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
            return 'DOUBLE PRECISION'
        return 'DOUBLE PRECISION'


# Singleton instance
table_controller = TableController()