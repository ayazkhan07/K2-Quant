"""
Report helpers injected into the DPE strategy execution context.

When capture is active (during normal DPE runs), structured blocks are recorded
for the OUTPUTS panel (dark grid tables). Plain ``print`` box-drawing is
skipped so stdout stays clean. If capture is off, legacy ASCII tables are
printed for ad-hoc use.

Invariant — per-cell character cap
----------------------------------
Every string cell that lands in a captured report block is capped at
``MAX_REPORT_CELL_CHARS`` (1,500 chars) before being appended. The cap is
applied centrally in ``_cap_cell_text`` and wired in through:

* ``_serialize_cell`` (string branch)        — covers ``report_table`` cells
* ``report_config``  (inline per-cell)       — covers config rows

If a cell is a comma-separated list (``', '`` present), the cap preserves
whole tokens and appends ``" \u2026 (+K more)"``. Otherwise the cell is
hard-sliced and marked ``" \u2026 (+K chars)"``.

Rationale: ``QTableWidget.resizeColumnsToContents()`` in
``OutputsPanel._append_data_table`` measures the natural text width of every
cell. Pathological cells (observed >100 KB, e.g., comma-joined index lists
from large-model RPP runs) produce column widths on the order of 10\u2076 pixels,
which collapses the OutputsPanel layout so *neither* the offending table nor
its siblings render. The cap protects three things simultaneously:

1. UI layout in ``OutputsPanel._append_data_table``.
2. DB row size for ``strategy_runs.report_blocks_json`` (~5x shrink on RPP).
3. Thinkspace / LLM context payloads when a run is referenced.

Strategy authors who need to expose the true item count should pass it as its
own column (e.g. ``'Count'``) alongside any list column; only the list-column
string is truncated for display.

See also: ``strategy_lifecycle_rules`` RULE 7.
"""

from __future__ import annotations

import math
import numbers
from contextvars import ContextVar
from typing import Any, List, Tuple, Optional

from k2_quant.utilities.numeric_rounding import (
    COMPUTATION_DECIMALS,
    format_computation_for_display,
)

_active_blocks: ContextVar[Optional[List[dict]]] = ContextVar(
    "_active_blocks", default=None
)

MAX_REPORT_CELL_CHARS = 1500
_CELL_MARKER_RESERVE = 40


def _cap_cell_text(text: str, max_chars: int = MAX_REPORT_CELL_CHARS) -> str:
    """Cap a string cell so the OUTPUTS renderer and DB stay well-behaved.

    Comma-separated lists are truncated on token boundaries with a
    ``" \u2026 (+K more)"`` marker; everything else is hard-sliced with a
    ``" \u2026 (+K chars)"`` marker. See module docstring (invariant).
    """
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_chars:
        return text

    if ", " in text:
        tokens = text.split(", ")
        budget = max(0, max_chars - _CELL_MARKER_RESERVE)
        kept: List[str] = []
        running = 0
        for tok in tokens:
            addl = len(tok) + (2 if kept else 0)
            if running + addl > budget:
                break
            kept.append(tok)
            running += addl
        if kept:
            remaining = len(tokens) - len(kept)
            if remaining > 0:
                return ", ".join(kept) + f" \u2026 (+{remaining:,} more)"
            return ", ".join(kept)

    head = text[:max(0, max_chars - _CELL_MARKER_RESERVE)]
    return head + f" \u2026 (+{len(text) - len(head):,} chars)"


def start_report_capture() -> List[dict]:
    """Begin collecting structured report blocks. Returns the live list."""
    blocks: List[dict] = []
    _active_blocks.set(blocks)
    return blocks


def end_report_capture() -> None:
    _active_blocks.set(None)


def _blocks() -> Optional[List[dict]]:
    return _active_blocks.get()


def _serialize_cell(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, COMPUTATION_DECIMALS)
    return _cap_cell_text(str(v))


def _fmt_val(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, numbers.Integral):
        return f"{int(v):,}"
    if isinstance(v, numbers.Real):
        return format_computation_for_display(v)
    return str(v)


def _plain_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, numbers.Integral, numbers.Real)):
        return _fmt_val(v)
    return str(v)


def format_blocks_plain(blocks: Optional[List[dict]]) -> str:
    """Readable plain text (tab-separated) for DB stdout / Thinkspace — no ASCII art."""
    if not blocks:
        return ""
    lines: List[str] = []
    for b in blocks:
        kind = b.get("kind")
        if kind == "header":
            lines.append(str(b.get("title", "")))
            lines.append("")
        elif kind == "config":
            lines.append("Configuration")
            hdrs = b.get("headers") or []
            lines.append("\t".join(str(h) for h in hdrs))
            for row in b.get("rows") or []:
                lines.append("\t".join(_plain_cell(x) for x in row))
            lines.append("")
        elif kind == "table":
            lines.append(str(b.get("title", "")))
            hdrs = b.get("headers") or []
            lines.append("\t".join(str(h) for h in hdrs))
            for row in b.get("rows") or []:
                lines.append("\t".join(_plain_cell(x) for x in row))
            lines.append("")
    return "\n".join(lines).rstrip()


def report_header(title: str) -> None:
    title = str(title).strip()
    bl = _blocks()
    if bl is not None:
        bl.append({"kind": "header", "title": title})
        return

    width = max(64, len(title) + 4)
    border = "=" * width
    print(border)
    print(f"  {title}")
    print(border)
    print()


def report_config(params: List[Tuple[str, Any, str]]) -> None:
    if not params:
        return

    bl = _blocks()
    if bl is not None:
        rows = [
            [
                _cap_cell_text(str(name)),
                _cap_cell_text(_fmt_val(value)),
                _cap_cell_text(str(desc)),
            ]
            for name, value, desc in params
        ]
        bl.append({
            "kind": "config",
            "title": "Configuration",
            "headers": ["Parameter", "Value", "Description"],
            "rows": rows,
        })
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
        print(
            f"| {str(name):<{w_name}} | {_fmt_val(value):<{w_val}} | "
            f"{str(desc):<{w_desc}} |"
        )
    print(sep)
    print()


def report_table(
    title: str,
    headers: List[str],
    rows: List[List[Any]],
    max_rows: int = 200,
) -> None:
    if not headers:
        return

    n_cols = len(headers)
    str_rows: List[List[Any]] = []
    for row in rows[:max_rows]:
        padded = list(row) + [""] * (n_cols - len(row))
        str_rows.append([_serialize_cell(v) for v in padded[:n_cols]])

    bl = _blocks()
    if bl is not None:
        bl.append({
            "kind": "table",
            "title": str(title),
            "headers": [str(h) for h in headers],
            "rows": str_rows,
            "truncated": max(0, len(rows) - max_rows),
        })
        return

    def _legacy_cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, numbers.Real):
            return _fmt_val(float(v))
        if isinstance(v, numbers.Integral):
            return _fmt_val(int(v))
        return str(v)

    widths = [len(h) for h in headers]
    display_rows = []
    for sr in str_rows:
        dr = [_legacy_cell(v) for v in sr]
        display_rows.append(dr)
        for i, cell in enumerate(dr):
            widths[i] = max(widths[i], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"

    print(f"--- {title} ---")
    print(sep)
    print(hdr_line)
    print(sep)
    for dr in display_rows:
        print("| " + " | ".join(cell.ljust(w) for cell, w in zip(dr, widths)) + " |")
    print(sep)

    if len(rows) > max_rows:
        print(f"  ... {len(rows) - max_rows} more rows truncated ...")
    print()
