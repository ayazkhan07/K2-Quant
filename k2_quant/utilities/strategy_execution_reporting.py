"""
Rich error reports for strategy execution (Dynamic Python Engine).

Maps wrapped exec() line numbers back to the user's strategy source and adds
context (what was running, where, cause hints).
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional, Tuple

# User strategy is injected after these lines in the exec() string:
#   1: def __k2_strategy__():
#   2:     global df, data, result
#   3+: indented user code
WRAPPER_LINES_BEFORE_USER_BODY = 2


def _user_line_from_wrapped(wrapped_lineno: int) -> int:
    """1-based line number in the saved strategy source."""
    return wrapped_lineno - WRAPPER_LINES_BEFORE_USER_BODY


def _source_context_lines(source: str, user_line: int, margin: int = 2) -> List[str]:
    """Return numbered context lines around user_line (1-based)."""
    lines = source.splitlines()
    if user_line < 1 or user_line > len(lines):
        return []
    i0 = max(0, user_line - 1 - margin)
    i1 = min(len(lines), user_line + margin)
    out = []
    for i in range(i0, i1):
        prefix = ">" if i + 1 == user_line else " "
        out.append(f"  {prefix} {i + 1:4d} | {lines[i]}")
    return out


def _find_user_frame_lineno(exc: BaseException) -> Optional[Tuple[int, str]]:
    """
    Return (wrapped_lineno, function_name) for the deepest frame running
    user strategy code inside __k2_strategy__, or None.
    """
    tb = exc.__traceback__
    while tb is not None:
        code = tb.tb_frame.f_code
        if code.co_filename == "<string>" and code.co_name == "__k2_strategy__":
            return tb.tb_lineno, code.co_name
        tb = tb.tb_next
    return None


def _syntax_error_user_line(exc: SyntaxError) -> Optional[int]:
    """Map SyntaxError.lineno (in wrapped exec source) to user strategy line."""
    ln = exc.lineno
    if ln is None:
        return None
    ul = _user_line_from_wrapped(ln)
    return ul if ul >= 1 else None


def snapshot_dataframe_for_report(df: Any) -> str:
    """Short description of the frame DataFrame for the report."""
    if df is None:
        return "DataFrame was not available in the execution context."
    try:
        n = len(df)
        cols = list(df.columns)
        col_preview = ", ".join(str(c) for c in cols[:24])
        if len(cols) > 24:
            col_preview += f", … (+{len(cols) - 24} more)"
        return f"{n} rows by {len(cols)} columns. Columns: {col_preview}"
    except Exception as e:
        return f"(Could not describe DataFrame: {e})"


def _what_was_running(metrics: Dict[str, Any], df_summary: str) -> str:
    rows = metrics.get("original_rows")
    parts = [
        "The Dynamic Python Engine (DPE) was executing your saved strategy against "
        "the loaded model.",
        "The model table was copied into `df` and `data` before your code ran.",
    ]
    if rows is not None:
        parts.append(f"Execution metrics recorded {rows} row(s) in that frame.")
    parts.append(f"Frame summary: {df_summary}")
    return "\n".join(parts)


def _where_it_failed(
    exc: BaseException,
    strategy_source: str,
) -> Tuple[str, Optional[int]]:
    """Narrative for location + user line number if known."""
    if isinstance(exc, SyntaxError):
        ul = _syntax_error_user_line(exc)
        if ul is not None and strategy_source:
            ctx = _source_context_lines(strategy_source, ul)
            block = "\n".join(ctx) if ctx else ""
            return (
                f"A syntax error was reported while compiling the strategy. "
                f"Strategy source line {ul} (1-based, relative to your saved code).\n{block}",
                ul,
            )
        return (
            "A syntax error was reported while compiling the strategy "
            f"(wrapped line {exc.lineno!r}).",
            None,
        )

    found = _find_user_frame_lineno(exc)
    if found is None:
        return (
            "The failure did not map to a line inside the strategy body "
            "(e.g. it may have occurred in the runner harness or a built-in). "
            "See the traceback section below.",
            None,
        )

    wrapped_ln, _fn = found
    user_ln = _user_line_from_wrapped(wrapped_ln)
    if user_ln < 1:
        return (
            f"Failure inside __k2_strategy__ at wrapped line {wrapped_ln} "
            f"(could not map to your source — report this if it persists).",
            None,
        )

    ctx = _source_context_lines(strategy_source, user_ln) if strategy_source else []
    block = "\n".join(ctx) if ctx else "  (source context unavailable)"
    return (
        f"Exception raised while running your strategy body (function __k2_strategy__).\n"
        f"Strategy source line: {user_ln} (1-based)\n{block}",
        user_ln,
    )


def _exception_type_and_message(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _likely_cause(exc: BaseException) -> str:
    """Plain-language hint; expandable over time."""
    if isinstance(exc, KeyError):
        key = exc.args[0] if exc.args else "?"
        return (
            f"Missing key {key!r}. In strategies this usually means `df[...]`, "
            f"`data[...]`, or similar does not have a column or index with that name. "
            f"Compare with k2_quant/utilities/strategy_frame_contract.json (canonical names)."
        )

    if isinstance(exc, (TypeError, ValueError)):
        msg = str(exc).lower()
        if "concatenate" in msg or "column_stack" in msg:
            return (
                "A NumPy stacking/concatenation call received no arrays (or an empty list). "
                "Typical causes: zero iterations from a filter (e.g. cascade left no survivors), "
                "or an empty candidate index list before np.column_stack / np.concatenate."
            )
        if "division" in msg or "zero" in msg and "div" in msg:
            return (
                "Numeric operation failed, often division by zero or an invalid value "
                "propagated from the data."
            )

    if isinstance(exc, IndexError):
        return (
            "An index was out of range — e.g. iloc/loc on a position past the end of "
            "the DataFrame or array, or an empty sequence where an element was expected."
        )

    if isinstance(exc, AttributeError):
        return (
            "An attribute lookup failed — often a variable was not a DataFrame/Series "
            "when the code expected one, or a typo in a method name."
        )

    if isinstance(exc, NameError):
        return (
            "An undefined name was used. Check spelling and that imports define the symbol."
        )

    if isinstance(exc, SyntaxError):
        return (
            "Python could not parse the strategy source. Fix the syntax at the indicated line."
        )

    return (
        "See the exception type and message above. If this is from a library call, "
        "check inputs (empty data, wrong dtypes, NaNs)."
    )


def _traceback_tail(exc: BaseException, max_frames: int = 12) -> str:
    """Compact traceback for debugging."""
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    text = "".join(lines)
    all_lines = text.strip().splitlines()
    if len(all_lines) <= max_frames:
        return text.strip()
    head = "\n".join(all_lines[: max_frames // 2])
    tail = "\n".join(all_lines[-(max_frames // 2) :])
    return f"{head}\n  … ({len(all_lines) - max_frames} lines omitted) …\n{tail}"


def format_strategy_execution_failure(
    exc: BaseException,
    strategy_source: str,
    *,
    metrics: Optional[Dict[str, Any]] = None,
    df_summary: str = "",
    include_full_traceback: bool = True,
) -> str:
    """
    Build the full user-facing error block for the Outputs panel and DB error_output.
    """
    metrics = metrics or {}
    where_text, user_line = _where_it_failed(exc, strategy_source)

    sections: List[str] = [
        "--- what was running ---",
        _what_was_running(metrics, df_summary or "(no frame summary)"),
        "",
        "--- where it failed ---",
        where_text,
        "",
        "--- what went wrong ---",
        _exception_type_and_message(exc),
        "",
        "--- likely cause ---",
        _likely_cause(exc),
        "",
        "--- line reference ---",
    ]
    if user_line is not None:
        sections.append(f"Strategy source line (1-based): {user_line}")
    else:
        sections.append(
            "Strategy source line: could not be mapped (see traceback)."
        )

    if include_full_traceback:
        sections.extend(["", "--- python traceback ---", _traceback_tail(exc)])

    return "\n".join(sections).strip()
