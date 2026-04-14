"""
32-20 RPP (15% initial tolerance) — compact variant with cascade guards.

Paste into the strategy editor (or save via save_strategy) if you hit:
  ValueError: need at least one array to concatenate
That error means np.column_stack got an empty list because res[k] was empty.

Guards added:
  - non-empty initial candidate range
  - finite T(0) pct anchors
  - non-empty survivor pool per OHLC before column_stack
"""
import numpy as np
import pandas as pd
from datetime import datetime

# =============================================================
# 32-20 RPP (15% initial tolerance)
# =============================================================

N_FORWARD       = 32
N_PAST_MAX      = 20
INITIAL_TOL     = 0.15   # note: wider than the base version
CASCADE_TOL_MUL = 3.0
MIN_SURVIVORS   = 7
VERBOSE         = False

# ------------------------- REPORT HEADER ---------------------
report_header('32-20 RPP (15% Cascade)')

report_config([
    ('N_FORWARD',       N_FORWARD,       'Bars projected into future'),
    ('N_PAST_MAX',      N_PAST_MAX,      'Max cascade depth'),
    ('INITIAL_TOL',     INITIAL_TOL,     'Initial tolerance (relative)'),
    ('CASCADE_TOL_MUL', CASCADE_TOL_MUL, 'Multiplier per cascade step'),
    ('MIN_SURVIVORS',   MIN_SURVIVORS,   'Minimum survivors per step'),
])

# ------------------------- HELPERS ---------------------------

def log(m):
    if VERBOSE:
        print(m)

def tol(step):
    return INITIAL_TOL * (CASCADE_TOL_MUL ** step)

def compound(vec, base):
    p, out = base, []
    for pct in vec:
        p *= 1 + pct / 100.0
        out.append(p)
    return out

def write(pairs):
    for n, v in pairs:
        to_forecast(n, [float(x) for x in v])

# ------------------------- DATA ------------------------------
pct_cols   = {'O': 'open_pct', 'H': 'high_pct', 'L': 'low_pct', 'C': 'close_pct'}
price_cols = {'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close'}

pct  = {k: df[c].to_numpy(float) for k, c in pct_cols.items()}
px   = {k: df[c].to_numpy(float) for k, c in price_cols.items()}
maxI = len(df) - 1

T0_pct   = {k: float(pct[k][maxI]) for k in 'OHLC'}
T0_price = {k: float(px[k][maxI]) for k in 'OHLC'}

for k in 'OHLC':
    if not np.isfinite(T0_pct[k]):
        report_table(
            'Error: invalid anchor',
            ['Column', 'Detail'],
            [[{'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close'}[k],
              f'T(0) pct is non-finite ({T0_pct[k]}). Check data at the last bar.']],
        )
        return

# ------------------------- CASCADE ---------------------------
# Candidate indices j need j + N_FORWARD <= maxI and j >= N_PAST_MAX - 1
#  => maxI - N_FORWARD > N_PAST_MAX - 1  =>  maxI > N_FORWARD + N_PAST_MAX - 1
if maxI <= N_FORWARD + (N_PAST_MAX - 1):
    report_table(
        'Error: insufficient history',
        ['Requirement', 'Detail'],
        [[
            'len(df)',
            f'Need max index > {N_FORWARD + (N_PAST_MAX - 1)} '
            f'(i.e. len(df) > {N_FORWARD + N_PAST_MAX}). Got len(df)={len(df)}.',
        ]],
    )
    return

res = {}
for k in 'OHLC':
    pool = list(range(N_PAST_MAX - 1, maxI - N_FORWARD))
    if len(pool) == 0:
        report_table(
            'Error: empty candidate pool',
            ['Column', 'Detail'],
            [[{'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close'}[k],
              f'No valid start indices in range({N_PAST_MAX - 1}, {maxI - N_FORWARD}).']],
        )
        return
    for s in range(N_PAST_MAX):
        anchor = T0_pct[k] if s == 0 else pct[k][maxI - s]
        if not np.isfinite(anchor):
            break
        band = abs(anchor) * tol(s)
        pool = [j for j in pool if abs(pct[k][j - s] - anchor) <= band]
        if len(pool) <= MIN_SURVIVORS:
            break
    res[k] = pool

# ------------------------- STATS -----------------------------
for k in 'OHLC':
    if len(res[k]) == 0:
        report_table(
            'Error: no cascade survivors',
            ['Column', 'Detail'],
            [[{'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close'}[k],
              'All candidates were filtered out, or cascade stopped with an empty pool. '
              'Try increasing INITIAL_TOL, lowering MIN_SURVIVORS, or using a longer dataset.']],
        )
        return

stats = {}
for k in 'OHLC':
    mat = np.column_stack([pct[k][j + 1:j + 1 + N_FORWARD] for j in res[k]])
    stats[k] = {'avg': np.nanmean(mat, 1), 'min': np.nanmin(mat, 1), 'max': np.nanmax(mat, 1)}

proj = {}
for k in 'OHLC':
    for s in ('avg', 'min', 'max'):
        proj[f'{k}_{s}'] = compound(stats[k][s], T0_price[k])

pairs = [
    ('RPP32_15_Open_Avg_P', proj['O_avg']),
    ('RPP32_15_Open_Min_P', proj['O_min']),
    ('RPP32_15_Open_Max_P', proj['O_max']),
    ('RPP32_15_High_Avg_P', proj['H_avg']),
    ('RPP32_15_High_Min_P', proj['H_min']),
    ('RPP32_15_High_Max_P', proj['H_max']),
    ('RPP32_15_Low_Avg_P', proj['L_avg']),
    ('RPP32_15_Low_Min_P', proj['L_min']),
    ('RPP32_15_Low_Max_P', proj['L_max']),
    ('RPP32_15_Close_Avg_P', proj['C_avg']),
    ('RPP32_15_Close_Min_P', proj['C_min']),
    ('RPP32_15_Close_Max_P', proj['C_max']),
]
write(pairs)

# --------------------- STEP TABLES ---------------------------
report_table('Step 1: Survivor Counts', ['Column', 'Survivors'],
             [[{'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close'}[k], len(res[k])] for k in 'OHLC'])
report_table('Forecast Columns Written', ['Column', 'Length'], [[n, len(v)] for n, v in pairs])
