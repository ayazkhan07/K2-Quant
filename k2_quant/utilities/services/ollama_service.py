from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import ollama
import pandas as pd
from psycopg2 import sql

from k2_quant.utilities.data.db_manager import DatabaseManager, db_manager
from k2_quant.utilities.services.schema_manager import SchemaManager


FORBIDDEN_SQL_TOKENS = ["DROP", "TRUNCATE"]


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    # Defaults optimized for responsiveness on 16 GB VRAM; override if you accept latency.
    reasoning_model: str = "llama3.2:3b-instruct-q8_0"
    sql_model: str = "sqlcoder:7b-q4_K_M"
    request_timeout_s: int = 30
    schema_cache_ttl_s: int = 300
    max_rows: int = 100_000
    read_only_default: bool = True

    # Protection and mutability defaults (safer by default)
    protected_columns: List[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume", "vwap",
        "symbol", "date", "timestamp", "id"
    ])
    allow_schema_changes: bool = True
    allow_data_modifications: bool = True
    auto_backup_before_changes: bool = False
    create_audit_tables: bool = False


class OllamaService:
    """
    Synchronous service for AI chat and NL→SQL using the official ollama client.

    - Compatible with PyQt via QThread adapters (no asyncio required in the UI path).
    - Enforces SQL safety; read-only by default; preview + confirm for writes.
    - 5-minute schema cache; paginated reads; 100k reported-row cap.
    """

    def __init__(self, db: DatabaseManager | None = None, config: Optional[OllamaConfig] = None) -> None:
        self.db: DatabaseManager = db if db is not None else db_manager
        self.config = config or OllamaConfig()
        self._client = ollama.Client(host=self.config.base_url)
        self._schema_cache: Optional[Dict[str, Any]] = None
        self._schema_cache_ts: float = 0.0
        self._verify_models_safely()
        self.schema_mgr = SchemaManager(self.db, protected_columns=set(self.config.protected_columns))

    # ---------------- Chat ----------------

    def chat(self, prompt: str, system: Optional[str] = None) -> str:
        """Return full assistant message (non-stream)."""
        resp = self._client.chat(
            model=self.config.reasoning_model,
            messages=[
                {"role": "system", "content": system or "You are a concise quantitative analysis assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        return resp.get("message", {}).get("content", "")

    def stream_chat(self, prompt: str, system: Optional[str] = None) -> Iterable[str]:
        """
        Blocking generator of content chunks (safe for use inside a QThread).
        """
        stream = self._client.chat(
            model=self.config.reasoning_model,
            messages=[
                {"role": "system", "content": system or "You are a concise quantitative analysis assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    # ---------------- NL→SQL ----------------

    def generate_sql(self, nl_query: str, schema: Dict[str, Any], read_only: bool = True) -> Dict[str, Any]:
        """
        Generate a single PostgreSQL statement from NL.
        Returns: {"sql": str, "rationale": str}
        """
        system = (
            "You are an expert SQL generator for PostgreSQL. "
            "Output exactly one executable SQL statement on a single line. "
            "Use only the provided schema names and columns. "
            "Never include comments or markdown fences. Do not perform destructive operations."
        )
        schema_text = self._format_schema_for_prompt(schema)
        policy = (
            "Policy:\n"
            "- Read-only by default.\n"
            "- Forbidden: DROP, TRUNCATE.\n"
            "- Prefer LIMIT when scope is unspecified.\n"
        )
        prompt = (
            f"{system}\n\n"
            f"SCHEMA:\n{schema_text}\n\n"
            f"{policy}\n"
            f"USER REQUEST:\n{nl_query}\n\n"
            "Return two lines:\n"
            "RATIONALE: one short sentence\n"
            "SQL: <single-line-sql>"
        )

        resp = self._client.generate(model=self.config.sql_model, prompt=prompt)
        text = resp.get("response", "") or ""
        rationale, sql = self._extract_rationale_and_sql(text)
        self._enforce_sql_policies(sql, read_only=read_only)
        return {"sql": sql, "rationale": rationale}

    # ---------------- Schema ----------------

    def introspect_schema(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch tables/columns from information_schema and cache for 5 minutes."""
        now = time.time()
        if self._schema_cache is not None and (now - self._schema_cache_ts) < self.config.schema_cache_ttl_s:
            return self._schema_cache

        include_patterns = include_patterns or ["stock_%", "computed_features", "projections", "strategies"]
        exclude_patterns = exclude_patterns or []

        like_clauses = []
        params: Dict[str, Any] = {}
        for i, pat in enumerate(include_patterns):
            like_clauses.append(f"table_name LIKE %(inc_{i})s")
            params[f"inc_{i}"] = pat

        exclude_clause = ""
        for i, pat in enumerate(exclude_patterns):
            exclude_clause += f" AND table_name NOT LIKE %(exc_{i})s"
            params[f"exc_{i}"] = pat

        where = " OR ".join(like_clauses) if like_clauses else "TRUE"
        sql = f"""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE ({where}) {exclude_clause}
              AND table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_name, ordinal_position
        """

        with self.db.get_connection() as conn:
            df = pd.read_sql(sql, conn, params=params)

        schema = self._schema_df_to_dict(df)
        self._schema_cache = schema
        self._schema_cache_ts = now
        return schema

    # ---------------- Read execution (paginated) ----------------

    def execute_sql_readonly(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        Execute a read-only query with LIMIT/OFFSET pagination,
        statement timeout, and a hard cap on reported total rows.
        """
        self._enforce_sql_policies(sql, read_only=True)

        wrapped = f"SELECT * FROM ({sql}) AS sub_query LIMIT {int(page_size)} OFFSET {int((page - 1) * page_size)}"
        count_wrapped = f"SELECT COUNT(1) AS total_rows FROM ({sql}) AS sub_query"

        start = time.time()
        with self.db.get_connection() as conn:
            with self.db.get_cursor(conn) as cur:
                cur.execute(f"SET LOCAL statement_timeout = {int(self.config.request_timeout_s * 1000)}")
            df = pd.read_sql(wrapped, conn, params=params)
            cnt = pd.read_sql(count_wrapped, conn, params=params)
        execution_ms = int((time.time() - start) * 1000)

        total_rows = int(cnt.iloc[0]["total_rows"]) if not cnt.empty else 0
        if total_rows > self.config.max_rows:
            total_rows = self.config.max_rows

        return {
            "columns": list(df.columns),
            "rows": df.values.tolist(),
            "total_rows": total_rows,
            "page": page,
            "page_size": page_size,
            "execution_ms": execution_ms,
        }

    # ---------------- Writes (preview + confirm) ----------------

    def preview_write(self, sql: str) -> Dict[str, Any]:
        """Validate and preview a write; estimate rows for INSERT ... SELECT when feasible."""
        self._enforce_sql_policies(sql, read_only=False)
        normalized = sql.strip().upper()

        if normalized.startswith("INSERT") and "SELECT" in normalized:
            try:
                count_sql = f"SELECT COUNT(1) AS cnt FROM ({self._extract_select(sql)}) AS src"
                with self.db.get_connection() as conn:
                    with self.db.get_cursor(conn) as cur:
                        cur.execute(f"SET LOCAL statement_timeout = {int(self.config.request_timeout_s * 1000)}")
                    df = pd.read_sql(count_sql, conn)
                est = int(df.iloc[0]["cnt"]) if not df.empty else None
                return {"ok": True, "operation": "INSERT", "estimated_rows": est, "requires_confirmation": True}
            except Exception as exc:
                return {"ok": False, "error": f"Could not estimate write impact: {exc}", "requires_confirmation": True}

        return {"ok": True, "operation": "WRITE", "estimated_rows": None, "requires_confirmation": True}

    def confirm_write(self, sql: str) -> Dict[str, Any]:
        """Execute a confirmed write with statement timeout. DB server handles timestamps."""
        self._enforce_sql_policies(sql, read_only=False)
        with self.db.get_connection() as conn:
            with self.db.get_cursor(conn) as cur:
                cur.execute(f"SET LOCAL statement_timeout = {int(self.config.request_timeout_s * 1000)}")
                cur.execute(sql)
                rows_affected = cur.rowcount
            conn.commit()
        return {"ok": True, "rows_affected": rows_affected}

    # ---------------- Orchestrator ----------------

    def analyze_request(self, user_message: str) -> Dict[str, Any]:
        """NL→SQL: schema → generate SQL (read-only) → execute → structured payload."""
        schema = self.introspect_schema()
        sql_obj = self.generate_sql(nl_query=user_message, schema=schema, read_only=True)
        sql = sql_obj["sql"]

        result = self.execute_sql_readonly(sql)
        table_update = {
            "rows": result["rows"],
            "total_rows": result["total_rows"],
            "page": result["page"],
            "page_size": result["page_size"],
        }

        chart_update = None
        cols_lower = [c.lower() for c in result["columns"]]
        if any(x in cols_lower for x in ("open", "high", "low", "close")):
            chart_update = {
                "type": "ohlc",
                "timeframe": "raw",
                "points": table_update["rows"],
                "columns": result["columns"],
            }

        summary = sql_obj.get("rationale", "Query executed.")
        return {
            "query": user_message,
            "sql": sql,
            "data": "paginated",
            "chart_update": chart_update,
            "table_update": table_update,
            "summary": summary,
        }

    # ---------------- Optional persistence ----------------

    def save_chat_history(self, user_msg: str, ai_response: str, sql: Optional[str] = None) -> None:
        """Persist chat history with server timestamp; no-op if table absent."""
        try:
            with self.db.get_connection() as conn:
                with self.db.get_cursor(conn) as cur:
                    cur.execute(
                        """
                        INSERT INTO chat_history (timestamp, user_message, ai_response, sql_generated)
                        VALUES (NOW(), %s, %s, %s)
                        """,
                        (user_msg, ai_response, sql),
                    )
                conn.commit()
        except Exception:
            # Silently skip if table doesn't exist or other non-critical errors occur.
            pass

    # ---------------- Helpers ----------------

    def _verify_models_safely(self) -> None:
        try:
            models = self._client.list().get("models", [])
            names = {m.get("name") for m in models}
            _ = (self.config.reasoning_model in names) and (self.config.sql_model in names)
        except Exception:
            # Ollama might not be running at import-time; defer failures to first call.
            pass

    def _enforce_sql_policies(self, sql_text: str, read_only: bool) -> None:
        normalized = sql_text.strip().upper()
        for token in FORBIDDEN_SQL_TOKENS:
            if re.search(rf"\b{token}\b", normalized):
                raise ValueError(f"Forbidden SQL token detected: {token}")
        if read_only and re.search(r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|REPLACE|GRANT|REVOKE|ALTER)\b", normalized):
            raise ValueError("Write operation detected in read-only mode.")

    # ---------------- Mutation generation & execution ----------------

    def generate_schema_sql(self, request: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a sequence of SQL statements to apply a schema/data change workflow.
        Returns: {"operations": List[str], "rationale": str}
        """
        protected = ", ".join(sorted(self.config.protected_columns))
        concise_schema = self._format_schema_for_prompt(schema)

        system = (
            "You are an expert PostgreSQL data engineering assistant. "
            "You may add columns, update non-protected columns, and delete rows. "
            "NEVER modify protected columns."
        )
        policy = (
            f"Protected columns (never modify): {protected}\n"
            "- For a calculated column: use ALTER TABLE ADD COLUMN (if missing) then UPDATE.\n"
            "- For data cleanup: use DELETE ... WHERE ...\n"
            "- For modifying user-created columns: ALTER TABLE ALTER COLUMN or UPDATE.\n"
            "Output each SQL statement on its own line; no comments or markdown."
        )
        prompt = (
            f"{system}\n\nSCHEMA:\n{concise_schema}\n\n"
            f"{policy}\n\nUSER REQUEST:\n{request}\n\n"
            "Return two sections:\n"
            "RATIONALE: one short sentence\n"
            "SQL:\n<statement 1>\n<statement 2>\n..."
        )

        resp = self._client.generate(model=self.config.sql_model, prompt=prompt)
        text = resp.get("response", "") or ""
        rationale, ops = self._extract_rationale_and_operations(text)
        for sql_stmt in ops:
            self._validate_protected_core_columns(sql_stmt)
        return {"operations": ops, "rationale": rationale}

    def execute_schema_operations(self, operations: List[str], table_hint: Optional[str] = None, request_text: Optional[str] = None) -> Dict[str, Any]:
        if not self.config.allow_schema_changes and not self.config.allow_data_modifications:
            return {"success": False, "error": "Mutations are disabled by configuration."}

        with self.db.get_connection() as conn:
            snapshot_token = None
            try:
                # Validate all operations up front
                for sql_stmt in operations:
                    self._validate_protected_core_columns(sql_stmt)
                    self._enforce_sql_mutability_policies(sql_stmt)

                # Optional snapshot or lightweight audit token
                if self.config.auto_backup_before_changes and table_hint:
                    try:
                        snapshot_token = self._create_snapshot_or_audit(conn, table_hint)
                    except Exception:
                        snapshot_token = None  # non-fatal

                # Explicit transaction and savepoint
                with self.db.get_cursor(conn) as cur:
                    cur.execute("BEGIN")
                    cur.execute("SAVEPOINT before_schema_change")
                    cur.execute(f"SET LOCAL statement_timeout = {int(self.config.request_timeout_s * 1000)}")

                results = []
                total_rows = 0
                with self.db.get_cursor(conn) as cur:
                    for sql_stmt in operations:
                        cur.execute(sql_stmt)
                        affected = cur.rowcount or 0
                        results.append({"sql": sql_stmt, "rows_affected": affected})
                        total_rows += affected

                conn.commit()
                self._schema_cache = None

                if self.config.create_audit_tables:
                    try:
                        self._ensure_audit_tables()
                        self._log_audit_entry(
                            conn,
                            request_text=request_text,
                            sql_executed=operations,
                            rows_affected=total_rows,
                            status="success",
                            rolled_back=False,
                            snapshot_table=snapshot_token,
                        )
                        conn.commit()
                    except Exception:
                        pass

                return {"success": True, "operations": results, "snapshot": snapshot_token}

            except Exception as e:
                try:
                    with self.db.get_cursor(conn) as cur:
                        cur.execute("ROLLBACK TO SAVEPOINT before_schema_change")
                    conn.commit()
                    if self.config.create_audit_tables:
                        try:
                            self._ensure_audit_tables()
                            self._log_audit_entry(
                                conn,
                                request_text=request_text,
                                sql_executed=operations,
                                rows_affected=0,
                                status="failed",
                                error_message=str(e),
                                rolled_back=True,
                                snapshot_table=snapshot_token,
                            )
                            conn.commit()
                        except Exception:
                            pass
                except Exception:
                    pass
                return {"success": False, "error": str(e)}

    # ---------------- Protection, snapshots, audit ----------------

    def _validate_protected_core_columns(self, sql_text: str) -> None:
        """
        Block changes to protected columns while allowing reads and references.
        Handles quoted identifiers and normalizes whitespace.
        """
        sql_normalized = " ".join(sql_text.upper().split())

        for col in self.config.protected_columns:
            col_upper = col.upper()
            dangerous_patterns = [
                rf"\bDROP\s+COLUMN\s+\"?{col_upper}\"?\b",
                rf"\bALTER\s+COLUMN\s+\"?{col_upper}\"?\b",
                rf"\bRENAME\s+COLUMN\s+\"?{col_upper}\"?\b",
                rf"\bSET\s+\"?{col_upper}\"?\s*=",
            ]
            for pattern in dangerous_patterns:
                if re.search(pattern, sql_normalized):
                    raise ValueError(f"Cannot modify protected column: {col}")

    def _enforce_sql_mutability_policies(self, sql_text: str) -> None:
        """
        Require explicit config for both data and schema modifications.
        Allow ALTER TABLE only when allow_schema_changes is True.
        Keep hard-destructive ops blocked via FORBIDDEN_SQL_TOKENS.
        """
        sql_u = sql_text.strip().upper()

        for token in FORBIDDEN_SQL_TOKENS:
            if re.search(rf"\b{token}\b", sql_u):
                raise ValueError(f"Forbidden SQL operation: {token}")

        if not self.config.allow_data_modifications and re.search(r"\b(UPDATE|DELETE|INSERT|MERGE)\b", sql_u):
            raise ValueError("Data modifications are disabled by configuration.")

        if not self.config.allow_schema_changes and re.search(r"\b(ALTER\s+TABLE|CREATE\s+TABLE)\b", sql_u):
            raise ValueError("Schema changes are disabled by configuration.")

    def _create_snapshot(self, conn, table_name: str) -> str:
        """Create a snapshot from the target table using identifier-safe composition."""
        ts = int(time.time())
        snapshot = f"{table_name}_snap_{ts}"
        with self.db.get_cursor(conn) as cur:
            cur.execute(
                sql.SQL("CREATE TABLE {} AS SELECT * FROM {}")
                   .format(sql.Identifier(snapshot), sql.Identifier(table_name))
            )
        return snapshot

    def _should_use_lightweight_audit(self, conn, table_name: str, threshold_bytes: int = 5 * 1024**3) -> bool:
        """Decide whether to avoid full snapshot based on physical size."""
        with self.db.get_cursor(conn) as cur:
            cur.execute(
                "SELECT pg_total_relation_size(%s) > %s AS is_large",
                (table_name, threshold_bytes)
            )
            row = cur.fetchone()
            return bool(row[0]) if row else False

    def _create_snapshot_or_audit(self, conn, table_name: str) -> str:
        """Record txid_current() for large tables; otherwise clone table."""
        if self._should_use_lightweight_audit(conn, table_name):
            with self.db.get_cursor(conn) as cur:
                cur.execute("SELECT txid_current()")
                txid = cur.fetchone()[0]
            return f"txid_{txid}"
        return self._create_snapshot(conn, table_name)

    def _ensure_audit_tables(self) -> None:
        with self.db.get_connection() as conn:
            with self.db.get_cursor(conn) as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ai_modifications (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        request_text TEXT,
                        sql_executed TEXT[],
                        rows_affected INTEGER,
                        status VARCHAR(20),
                        error_message TEXT,
                        rolled_back BOOLEAN DEFAULT FALSE,
                        snapshot_table TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ai_columns (
                        id SERIAL PRIMARY KEY,
                        table_name TEXT NOT NULL,
                        column_name TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        created_by TEXT,
                        rationale TEXT,
                        is_protected BOOLEAN DEFAULT FALSE,
                        UNIQUE(table_name, column_name)
                    )
                    """
                )
            conn.commit()

    def _log_audit_entry(self, conn, request_text: Optional[str], sql_executed: List[str], rows_affected: int, status: str, error_message: Optional[str] = None, rolled_back: bool = False, snapshot_table: Optional[str] = None) -> None:
        with self.db.get_cursor(conn) as cur:
            cur.execute(
                """
                INSERT INTO ai_modifications (request_text, sql_executed, rows_affected, status, error_message, rolled_back, snapshot_table)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (request_text, sql_executed, rows_affected, status, error_message, rolled_back, snapshot_table)
            )

    def _format_schema_for_prompt(self, schema: Dict[str, Any]) -> str:
        # Concise for token efficiency
        lines: List[str] = []
        for table, meta in schema.items():
            cols = [c["name"] for c in meta.get("columns", [])]
            lines.append(f"{table}: {', '.join(cols)}")
        return "\n".join(lines)

    def _schema_df_to_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if df is None or df.empty:
            return out
        for (table_name), g in df.groupby("table_name"):
            out[table_name] = {
                "columns": [{"name": row["column_name"], "type": row["data_type"]} for _, row in g.iterrows()]
            }
        return out

    def _extract_select(self, sql: str) -> str:
        m = re.search(r"SELECT\s+.*", sql, re.IGNORECASE | re.DOTALL)
        if not m:
            raise ValueError("Could not extract SELECT part for estimation.")
        return m.group(0)

    def _extract_rationale_and_sql(self, text: str) -> Tuple[str, str]:
        rationale = ""
        sql_stmt = text.strip()
        rat_match = re.search(r"RATIONALE:\s*(.+)\n", text, re.IGNORECASE)
        if rat_match:
            rationale = rat_match.group(1).strip()
        sql_match = re.search(r"SQL:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql_stmt = sql_match.group(1).strip()
        sql_stmt = re.sub(r"^```(sql|SQL)?", "", sql_stmt).strip()
        sql_stmt = re.sub(r"```$", "", sql_stmt).strip()
        sql_stmt = " ".join(sql_stmt.split())
        return rationale, sql_stmt

    def _extract_rationale_and_operations(self, text: str) -> Tuple[str, List[str]]:
        rationale = ""
        ops: List[str] = []
        m = re.search(r"RATIONALE:\s*(.+)\n", text, re.IGNORECASE)
        if m:
            rationale = m.group(1).strip()
        m2 = re.search(r"SQL:\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
        if m2:
            body = m2.group(1).strip()
            ops = [line.strip().rstrip(';') for line in body.splitlines() if line.strip()]
        return rationale, ops


