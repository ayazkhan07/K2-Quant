"""
Market hours detection utility.

Provides timezone-aware checks for US equity market sessions,
including early close days and federal holidays.
"""

from datetime import datetime, time, date, timedelta
from typing import Optional, Tuple

import pytz


_ET = pytz.timezone("US/Eastern")

_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)
_EARLY_CLOSE = time(13, 0)

# Federal holidays observed by NYSE (approximate — covers most years).
# This list should be updated annually or fetched from an external source.
_FIXED_HOLIDAYS_2025_2027 = {
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
    date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
    date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25),
    date(2027, 12, 24),
}

# Early close days (day before Independence Day, day after Thanksgiving, Christmas Eve)
_EARLY_CLOSE_DATES_2025_2027 = {
    date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24),
    date(2026, 7, 2), date(2026, 11, 27), date(2026, 12, 24),
    date(2027, 7, 2), date(2027, 11, 26), date(2027, 12, 23),
}


def now_et() -> datetime:
    """Current time in US/Eastern."""
    return datetime.now(_ET)


def is_market_holiday(d: date) -> bool:
    return d in _FIXED_HOLIDAYS_2025_2027


def is_early_close_day(d: date) -> bool:
    return d in _EARLY_CLOSE_DATES_2025_2027


def is_trading_day(d: date) -> bool:
    """True if *d* is a weekday that is not an NYSE holiday."""
    if d.weekday() >= 5:
        return False
    return not is_market_holiday(d)


def market_session_times(d: date) -> Optional[Tuple[time, time]]:
    """Return (open, close) for *d*, or None if market is closed."""
    if not is_trading_day(d):
        return None
    close = _EARLY_CLOSE if is_early_close_day(d) else _MARKET_CLOSE
    return (_MARKET_OPEN, close)


def is_market_open() -> bool:
    """True if the US equity market is currently in session."""
    dt = now_et()
    session = market_session_times(dt.date())
    if session is None:
        return False
    mkt_open, mkt_close = session
    return mkt_open <= dt.time() < mkt_close


def market_status() -> str:
    """Human-readable market status string."""
    dt = now_et()
    session = market_session_times(dt.date())
    if session is None:
        if dt.date().weekday() >= 5:
            return "Market Closed — Weekend"
        return "Market Closed — Holiday"
    mkt_open, mkt_close = session
    t = dt.time()
    if t < mkt_open:
        return f"Pre-Market — Opens {mkt_open.strftime('%I:%M %p')} ET"
    if t >= mkt_close:
        return "Market Closed — After Hours"
    return "Market Open"


def next_market_open() -> datetime:
    """Return the next market open as an ET-aware datetime."""
    dt = now_et()
    d = dt.date()
    session = market_session_times(d)
    if session and dt.time() < session[0]:
        return _ET.localize(datetime.combine(d, session[0]))
    d += timedelta(days=1)
    for _ in range(10):
        if is_trading_day(d):
            return _ET.localize(datetime.combine(d, _MARKET_OPEN))
        d += timedelta(days=1)
    return _ET.localize(datetime.combine(d, _MARKET_OPEN))


def seconds_until_market_open() -> float:
    """Seconds until the market next opens (0 if already open)."""
    if is_market_open():
        return 0.0
    nxt = next_market_open()
    return max(0.0, (nxt - now_et()).total_seconds())
