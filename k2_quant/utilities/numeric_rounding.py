"""
K2 numeric display / computation policy (single source of truth).

- **Prices** (open, high, low, close, vwap, forecast dollars): at most 2 decimal places.
- **Computations** (percent changes, indicators, metrics, other derived floats): at most 3.

Apply ``round_dataframe_numeric_columns`` when building analysis DataFrames so in-memory
data and strategy inputs match what users see. This does not rewrite PostgreSQL tables;
it normalizes values at load / compute boundaries.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

import pandas as pd

PRICE_DECIMALS = 2
COMPUTATION_DECIMALS = 3

# Canonical + UI column names (normalized with _norm_col)
_PRICE_NAMES = frozenset({"open", "high", "low", "close", "vwap"})


def _norm_col(name: Any) -> str:
    return str(name).lower().replace("-", "_")


def is_price_column(name: Any) -> bool:
    n = _norm_col(name)
    if n in _PRICE_NAMES:
        return True
    # e.g. "Open" -> open
    return False


def is_percent_or_elasticity_column(name: Any) -> bool:
    n = _norm_col(name)
    if n == "elasticity":
        return True
    if "close_open" in n and "%" in str(name).lower():
        return True
    if n.endswith("_pct"):
        return True
    if "_%" in str(name).lower():
        return True
    return False


def is_volume_column(name: Any) -> bool:
    return _norm_col(name) == "volume"


def round_price_scalar(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, numbers.Real):
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return x
        return round(xf, PRICE_DECIMALS)
    return x


def round_computation_scalar(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, numbers.Real):
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return x
        return round(xf, COMPUTATION_DECIMALS)
    return x


def round_dataframe_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with OHLCV prices, volume, percents, and other floats rounded."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        raw_name = str(col)
        n = _norm_col(col)
        if n in ("date", "time") or "date_time" in n:
            continue
        if n == "#":
            try:
                s = pd.to_numeric(out[col], errors="coerce")
                out[col] = s.round(0)
            except Exception:
                pass
            continue
        if is_volume_column(col):
            s = pd.to_numeric(out[col], errors="coerce")
            out[col] = s.round(0)
            continue
        if is_price_column(col):
            out[col] = pd.to_numeric(out[col], errors="coerce").round(PRICE_DECIMALS)
            continue
        if is_percent_or_elasticity_column(col):
            out[col] = pd.to_numeric(out[col], errors="coerce").round(COMPUTATION_DECIMALS)
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").round(COMPUTATION_DECIMALS)
    return out


def format_price_for_display(x: Any) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, numbers.Integral):
        return f"{int(x):,}"
    if isinstance(x, numbers.Real):
        v = round(float(x), PRICE_DECIMALS)
        if abs(v) >= 1_000:
            return f"{v:,.2f}"
        return f"{v:.2f}"
    return str(x)


def format_computation_for_display(x: Any) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, numbers.Integral):
        return f"{int(x):,}"
    if isinstance(x, numbers.Real):
        v = round(float(x), COMPUTATION_DECIMALS)
        if abs(v) >= 1_000:
            return f"{v:,.3f}"
        return f"{v:.3f}"
    return str(x)


def report_column_looks_price(header: str) -> bool:
    h = str(header).lower()
    if "$" in h or "nominal" in h:
        return True
    if "price" in h and "pct" not in h and "%" not in h:
        return True
    if h in {"open", "high", "low", "close", "vwap"}:
        return True
    return False
