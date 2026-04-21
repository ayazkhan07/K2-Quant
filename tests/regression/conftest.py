"""
Shared fixtures for strategy-execution regression tests.

These tests exercise ``dpe_service.execute_strategy`` directly with a synthetic
DataFrame. They do not touch PostgreSQL, ``stock_service``, or Qt. The point is
to lock the *output* of strategies so that later performance refactors
(threading, algorithmic kernel rewrites, Numba JIT, etc.) can prove they have
not changed forecast values.

Snapshot files live next to the test modules. On the first run of a new test,
if no snapshot is present the test writes one and fails with a clear message.
Regenerate intentionally with::

    pytest tests/regression -q --update-snapshots
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def pytest_addoption(parser):
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Overwrite golden snapshots instead of comparing.",
    )


@pytest.fixture(scope="session")
def update_snapshots(request) -> bool:
    return bool(request.config.getoption("--update-snapshots"))


def _synthetic_price_frame(n_rows: int, seed: int) -> pd.DataFrame:
    """Deterministic OHLC frame with the columns the stream/analysis pipeline
    hands to strategies.

    Random walk on close with mild intrabar spread so ``high``/``low`` are
    sensible. All derived percent/elasticity columns match the formulas the
    production loader uses so the strategy sees exactly the same shape it sees
    in the real app.
    """
    rng = np.random.default_rng(seed)
    base = 100.0
    # Random-walk returns in %, bounded so we don't explode.
    rets = rng.normal(loc=0.0, scale=0.8, size=n_rows) / 100.0
    close = base * np.cumprod(1.0 + rets)
    # Intrabar spread ~0.5% of close, direction random.
    spread = np.abs(rng.normal(loc=0.0, scale=0.005, size=n_rows)) * close
    open_ = np.empty(n_rows, dtype=float)
    open_[0] = base
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + spread * 0.5
    low = np.minimum(open_, close) - spread * 0.5
    volume = rng.integers(1_000, 10_000, size=n_rows).astype(float)
    vwap = (high + low + close) / 3.0

    # Timestamps on business days, 09:30 ET style.
    start = pd.Timestamp("2020-01-02 09:30:00")
    dt = pd.date_range(start=start, periods=n_rows, freq="B")

    df = pd.DataFrame({
        "#": np.arange(1, n_rows + 1, dtype=int),
        "date_time_market": dt,
        "Date": dt.strftime("%Y-%m-%d"),
        "Time": dt.strftime("%H:%M:%S"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "vwap": vwap,
    })

    # Derived columns (mirror db_manager.compute_derived_columns / loader).
    for c in ("open", "high", "low", "close"):
        prev = df[c].shift(1)
        df[f"{c}_pct"] = np.where(
            (prev != 0) & prev.notna() & df[c].notna(),
            (df[c] - prev) / prev * 100.0,
            np.nan,
        )
    df["elasticity"] = np.where(
        df["low"] != 0, (df["high"] - df["low"]) / df["low"] * 100.0, np.nan
    )
    df["close_open_pct"] = np.where(
        df["open"] != 0, (df["close"] - df["open"]) / df["open"] * 100.0, np.nan
    )

    # Apply the same rounding the production pipeline applies before DPE.
    from k2_quant.utilities.numeric_rounding import round_dataframe_numeric_columns
    df = round_dataframe_numeric_columns(df)
    return df


@pytest.fixture(scope="session")
def small_fixture_df() -> pd.DataFrame:
    return _synthetic_price_frame(n_rows=600, seed=42)


@pytest.fixture(scope="session")
def medium_fixture_df() -> pd.DataFrame:
    return _synthetic_price_frame(n_rows=5_000, seed=42)


@pytest.fixture(scope="session")
def rpp_strategy_code() -> str:
    path = REPO_ROOT / "RPP_8_18_RPP_strategy_with_report.py"
    return path.read_text(encoding="utf-8")


# ────────────────────────────────────────────────────────────────────────────
# Snapshot helpers


def _round(x, digits=6):
    if x is None:
        return None
    if isinstance(x, float):
        if np.isnan(x) or np.isinf(x):
            return None
        return round(float(x), digits)
    return x


def summarize_strategy_result(result: dict) -> dict:
    """Produce a stable, JSON-serialisable dict of everything a performance
    refactor must preserve bit-for-bit.

    Specifically: success flag, every forecast/working tab-write (name +
    rounded values), metrics that don't include timing.
    """
    out = {
        "success": bool(result.get("success", False)),
        "tab_writes": [],
        "has_error": result.get("error") is not None,
        "original_rows": None,
    }
    m = result.get("metrics") or {}
    out["original_rows"] = m.get("original_rows")

    for entry in result.get("_tab_writes") or []:
        kind = entry.get("type")
        if kind == "forecast":
            vals = entry.get("values")
            if vals is None:
                vals_summary = None
            else:
                vals_summary = [_round(v) for v in vals]
            out["tab_writes"].append({
                "type": "forecast",
                "column_name": entry.get("column_name"),
                "length": len(vals) if vals is not None else 0,
                "values": vals_summary,
                "anchor_price": _round(entry.get("anchor_price")),
                "set_index": entry.get("set_index"),
            })
        elif kind == "working":
            out["tab_writes"].append({
                "type": "working",
                "column_name": entry.get("column_name"),
                "length": len(entry.get("values") or []),
            })
        else:
            out["tab_writes"].append({"type": kind})
    return out


def assert_matches_snapshot(name: str, payload: dict, update: bool) -> None:
    snap_path = Path(__file__).parent / "snapshots" / f"{name}.json"
    snap_path.parent.mkdir(parents=True, exist_ok=True)

    if update or not snap_path.exists():
        snap_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        if not update:
            pytest.fail(
                f"Snapshot {snap_path} did not exist; wrote it now. "
                f"Review it, commit, and re-run."
            )
        return

    expected = json.loads(snap_path.read_text(encoding="utf-8"))
    if expected != payload:
        # Emit a compact diff summary so CI logs are useful.
        diff_lines = []
        ek = set(expected.keys()) | set(payload.keys())
        for k in sorted(ek):
            if expected.get(k) != payload.get(k):
                diff_lines.append(f"  {k}: expected={expected.get(k)!r} got={payload.get(k)!r}")
        pytest.fail(
            "Strategy output diverged from snapshot "
            f"{snap_path}:\n" + "\n".join(diff_lines[:40])
        )
