"""
Golden-master regression for DPE strategy execution.

Runs the production ``8-18 RPP`` strategy through the DPE against a small,
deterministic synthetic DataFrame and asserts the resulting forecast writes
(names, lengths, anchor prices, per-value contents rounded to 6 dp) are
bit-identical to a committed snapshot. Any later performance refactor that
changes strategy output - even by a single epsilon - will fail loudly here.
"""
from __future__ import annotations

import pytest

from k2_quant.utilities.services.dynamic_python_engine import dpe_service

from tests.regression.conftest import (
    assert_matches_snapshot,
    summarize_strategy_result,
)


def test_rpp_8_18_on_small_fixture(small_fixture_df, rpp_strategy_code, update_snapshots):
    result = dpe_service.execute_strategy(rpp_strategy_code, small_fixture_df)
    payload = summarize_strategy_result(result)
    # Pin row count so a fixture-size drift is caught too.
    payload["fixture_rows"] = len(small_fixture_df)
    assert_matches_snapshot("rpp_8_18_small", payload, update_snapshots)


@pytest.mark.slow
def test_rpp_8_18_on_medium_fixture(medium_fixture_df, rpp_strategy_code, update_snapshots):
    result = dpe_service.execute_strategy(rpp_strategy_code, medium_fixture_df)
    payload = summarize_strategy_result(result)
    payload["fixture_rows"] = len(medium_fixture_df)
    assert_matches_snapshot("rpp_8_18_medium", payload, update_snapshots)


def test_dpe_returns_cancelled_flag_when_event_set(small_fixture_df, rpp_strategy_code):
    """PR 1 wiring: DPE exposes a cancel_event parameter; setting it before
    exec starts yields a result with ``cancelled=True`` and ``success=False``.
    """
    import threading

    ev = threading.Event()
    ev.set()
    result = dpe_service.execute_strategy(
        rpp_strategy_code, small_fixture_df, cancel_event=ev
    )
    assert result["success"] is False
    assert result.get("cancelled") is True
