"""
Report formatting helpers injected into the DPE strategy execution context.

Strategies call these to produce structured, tabular stdout output that
appears in the OUTPUTS Report tab.  The validator enforces their use.
"""

from typing import List, Tuple, Any, Optional


def _fmt_val(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if abs(v) >= 1_000:
            return f"{v:,.2f}"
        if abs(v) < 0.01 and v != 0:
            return f"{v:.6f}"
        return f"{v:.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def report_header(title: str) -> None:
    """Print a strategy title banner."""
    width = max(64, len(title) + 4)
    border = "=" * width
    print(border)
    print(f"  {title}")
    print(border)
    print()


def report_config(params: List[Tuple[str, Any, str]]) -> None:
    """Print a configuration table.

    params: list of (name, value, description) tuples.
    """
    if not params:
        return

    col_name = "Parameter"
    col_val = "Value"
    col_desc = "Description"

    w_name = max(len(col_name), *(len(str(p[0])) for p in params))
    w_val = max(len(col_val), *(len(_fmt_val(p[1])) for p in params))
    w_desc = max(len(col_desc), *(len(str(p[2])) for p in params))

    sep = f"+-{'-' * w_name}-+-{'-' * w_val}-+-{'-' * w_desc}-+"
    hdr = f"| {col_name:<{w_name}} | {col_val:<{w_val}} | {col_desc:<{w_desc}} |"

    print("--- Configuration ---")
    print(sep)
    print(hdr)
    print(sep)
    for name, value, desc in params:
        print(f"| {str(name):<{w_name}} | {_fmt_val(value):<{w_val}} | {str(desc):<{w_desc}} |")
    print(sep)
    print()


def report_table(
    title: str,
    headers: List[str],
    rows: List[List[Any]],
    max_rows: int = 200,
) -> None:
    """Print a titled data table.

    title:   section heading (e.g. 'Step 1: Survivor Filtering')
    headers: column names
    rows:    list of row-lists (same length as headers)
    max_rows: truncate display after this many rows
    """
    if not headers:
        return

    n_cols = len(headers)
    str_rows = []
    for row in rows[:max_rows]:
        padded = list(row) + [""] * (n_cols - len(row))
        str_rows.append([_fmt_val(v) for v in padded[:n_cols]])

    widths = [len(h) for h in headers]
    for sr in str_rows:
        for i, cell in enumerate(sr):
            widths[i] = max(widths[i], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"

    print(f"--- {title} ---")
    print(sep)
    print(hdr_line)
    print(sep)
    for sr in str_rows:
        print("| " + " | ".join(cell.ljust(w) for cell, w in zip(sr, widths)) + " |")
    print(sep)

    if len(rows) > max_rows:
        print(f"  ... {len(rows) - max_rows} more rows truncated ...")
    print()
