import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
# 32-20 Recursive Price Projection (RPP)
# 32 = forward projection indices, 20 = max past look depth
# Time agnostic — works on any model frequency
# ═══════════════════════════════════════════════════════════════════

# ── CONFIGURATION ──────────────────────────────────────────────────
N_FORWARD       = 32      # forward projection indices
N_PAST_MAX      = 20      # maximum cascade depth (past look)
MIN_SURVIVORS   = 7       # minimum surviving pool to proceed
INITIAL_TOL     = 0.15    # M_I0 tolerance: 15% relative
CASCADE_TOL_MUL = 3.0     # multiply tolerance by 3x each subsequent step
PCT_CLAMP       = 50.0    # warn if any pct move exceeds this threshold
VERBOSE         = True    # set False to mute debug prints

# ── HELPERS ────────────────────────────────────────────────────────
def log(msg):
    if VERBOSE:
        print(msg)


def write_forecasts(forecast_pairs):
    """Write all 12 forecast columns, halt on any failure."""
    failed = []
    for entry in forecast_pairs:
        col_name, vals = entry[0], entry[1]
        anchor = entry[2] if len(entry) > 2 else None
        try:
            to_forecast(col_name, [float(v) for v in vals], anchor_price=anchor)
            log(f"  ✓ {col_name} ({len(vals)} values)")
        except Exception as e:
            failed.append((col_name, str(e)))
    if failed:
        for col_name, err in failed:
            print(f"  ✗ FAILED: {col_name} — {err}")
        raise RuntimeError(
            f"RPP 32-20: {len(failed)} forecast write(s) failed"
        )


def get_tolerance(step):
    """Return the relative tolerance for cascade step n.
    M_I0 = 15%, M_I1 = 45%, M_I2 = 135%, etc."""
    return INITIAL_TOL * (CASCADE_TOL_MUL ** step)


# ═══════════════════════════════════════════════════════════════════
# STEP 0 — Define I Series
# ═══════════════════════════════════════════════════════════════════

log("=" * 65)
log("STEP 0 — Define I Series")
log("=" * 65)

# I series = the DataFrame's integer index (0-based row positions)
# Max(I) = T(0)
n = len(df)
max_I = n - 1

log(f"  Total indices in dataset: {n}")
log(f"  Max(I) = T(0) = index {max_I}")

# Percentage column names — DPE context keeps original casing
pct_cols = {
    'O': 'Open_%',
    'H': 'High_%',
    'L': 'Low_%',
    'C': 'Close_%',
}
price_cols = {
    'O': 'open',
    'H': 'high',
    'L': 'low',
    'C': 'close',
}

# Convert to numpy arrays for fast access
pct_arrays = {}
price_arrays = {}
for key in ['O', 'H', 'L', 'C']:
    pct_arrays[key] = df[pct_cols[key]].to_numpy(dtype=np.float64)
    price_arrays[key] = df[price_cols[key]].to_numpy(dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Establish T(0)
# ═══════════════════════════════════════════════════════════════════

log("\n" + "=" * 65)
log("STEP 1 — Establish T(0)")
log("=" * 65)

T0_price = {}
T0_pct = {}
for key in ['O', 'H', 'L', 'C']:
    T0_price[key] = float(price_arrays[key][max_I])
    T0_pct[key] = float(pct_arrays[key][max_I])
    log(f"  T_{key}_price = {T0_price[key]:.4f}    T_{key}_pct = {T0_pct[key]:.6f}")

# Extract T(-1) through T(-19) pct values for cascade
# T(-n) = index max_I - n
T_neg_pct = {}  # T_neg_pct['O'][n] = open_pct at T(-n)
for key in ['O', 'H', 'L', 'C']:
    T_neg_pct[key] = {}
    for step in range(0, N_PAST_MAX):
        idx = max_I - step
        if idx >= 0:
            T_neg_pct[key][step] = float(pct_arrays[key][idx])
        else:
            T_neg_pct[key][step] = None

log(f"\n  T(-1) through T(-{N_PAST_MAX-1}) pct values extracted for cascade")


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Cascade Match: M_I0 through M_I(final)
#
# Each OHLC column is processed INDEPENDENTLY.
# M_I0: all indices where pct matches T(0) within 15% relative tol
# M_I1: subset of M_I0 where (j-1) matches T(-1) within 45% relative tol
# M_I2: subset of M_I1 where (j-2) matches T(-2) within 135% relative tol
# General: M_I(n) tolerance = 15% × 3^n
# ... continue until pool stabilizes or max depth reached
# ═══════════════════════════════════════════════════════════════════

log("\n" + "=" * 65)
log("STEP 2 — Cascade Match")
log("=" * 65)

# Boundary: a candidate j must allow:
#   - j - (N_PAST_MAX - 1) >= 0   (lookback room for full cascade)
#   - j + N_FORWARD < max_I        (forward room without overlapping T(0))
min_j = N_PAST_MAX - 1          # minimum valid candidate index
max_j = max_I - N_FORWARD - 1   # maximum valid candidate index

log(f"  Boundary: valid candidates in range [{min_j}, {max_j}]")
log(f"  Excluding T(0) and its forward overlap zone")

# Store cascade results per OHLC column
cascade_results = {}  # key -> {'pools': {step: [indices]}, 'counts': {step: int}, 'final_step': int}

for key in ['O', 'H', 'L', 'C']:
    log(f"\n  ── {key} column cascade ──")

    pools = {}
    counts = {}
    pct_arr = pct_arrays[key]

    # M_I0: tolerance match against T(0)
    tol_0 = get_tolerance(0)
    anchor_0 = T0_pct[key]
    band_0 = abs(anchor_0) * tol_0

    m_i0 = []
    for j in range(min_j, max_j + 1):
        if np.isnan(pct_arr[j]):
            continue
        if abs(pct_arr[j] - anchor_0) <= band_0:
            m_i0.append(j)

    pools[0] = m_i0
    counts[0] = len(m_i0)
    log(f"    M_I0: anchor={anchor_0:.6f}  tol=±{band_0:.6f} ({tol_0*100:.1f}% rel)  "
        f"count={counts[0]}")

    if counts[0] < MIN_SURVIVORS:
        raise ValueError(
            f"RPP 32-20: {key} M_I0 returned only {counts[0]} candidates "
            f"(min={MIN_SURVIVORS}). Widen initial tolerance."
        )

    # Cascade: M_I1 through M_I(N_PAST_MAX-1)
    prev_pool = m_i0
    prev_count = counts[0]
    final_step = 0

    for step in range(1, N_PAST_MAX):
        tol = get_tolerance(step)
        anchor = T_neg_pct[key].get(step)

        if anchor is None:
            log(f"    M_I{step}: T(-{step}) out of bounds, stopping cascade")
            break

        band = abs(anchor) * tol

        new_pool = []
        for j in prev_pool:
            lookback_idx = j - step
            if lookback_idx < 0:
                continue
            val = pct_arr[lookback_idx]
            if np.isnan(val):
                continue
            if abs(val - anchor) <= band:
                new_pool.append(j)

        pools[step] = new_pool
        counts[step] = len(new_pool)

        log(f"    M_I{step}: anchor=T(-{step})={anchor:.6f}  "
            f"tol=±{band:.6f} ({tol*100:.1f}% rel)  count={counts[step]}")

        final_step = step

        # Check stabilization: if pool stopped shrinking or hit minimum
        if counts[step] <= MIN_SURVIVORS:
            log(f"    Pool stabilized at {counts[step]} survivors, stopping cascade")
            break

        if counts[step] == prev_count:
            log(f"    Pool unchanged from previous step, stopping cascade")
            break

        prev_pool = new_pool
        prev_count = counts[step]

    # If the last pool is empty, fall back to the previous step
    if counts[final_step] == 0:
        for s in range(final_step - 1, -1, -1):
            if counts[s] > 0:
                final_step = s
                log(f"    Fell back to M_I{s} with {counts[s]} survivors")
                break

    if counts[final_step] < MIN_SURVIVORS:
        log(f"    ⚠ WARNING: {key} final pool has only {counts[final_step]} "
            f"survivors (wanted {MIN_SURVIVORS})")

    cascade_results[key] = {
        'pools': pools,
        'counts': counts,
        'final_step': final_step,
        'survivors': pools[final_step],
        'survivor_count': counts[final_step],
    }

    log(f"    FINAL: M_I{final_step} → {counts[final_step]} survivors")

# Summary
log("\n  ── Cascade Summary ──")
for key in ['O', 'H', 'L', 'C']:
    cr = cascade_results[key]
    log(f"    {key}: depth={cr['final_step']}  survivors={cr['survivor_count']}  "
        f"indices={cr['survivors']}")


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — Build Forward Matrices (32 × N_survivors per column)
# ═══════════════════════════════════════════════════════════════════

log("\n" + "=" * 65)
log("STEP 3 — Build Forward Matrices")
log("=" * 65)

forward_matrices = {}  # key -> np.array shape (N_FORWARD, n_survivors)

for key in ['O', 'H', 'L', 'C']:
    survivors = cascade_results[key]['survivors']
    pct_arr = pct_arrays[key]
    n_surv = len(survivors)

    if n_surv == 0:
        raise ValueError(f"RPP 32-20: {key} has 0 survivors, cannot build forward matrix")

    cols = []
    for j in survivors:
        # Forward indices: j+1 through j+N_FORWARD
        fwd = pct_arr[j + 1: j + N_FORWARD + 1]

        if len(fwd) < N_FORWARD:
            log(f"    ⚠ {key} survivor j={j}: forward window truncated "
                f"({len(fwd)}/{N_FORWARD}), padding with NaN")
            fwd = np.concatenate([fwd, np.full(N_FORWARD - len(fwd), np.nan)])

        cols.append(fwd)

    matrix = np.column_stack(cols)
    forward_matrices[key] = matrix
    log(f"  {key}: forward matrix shape {matrix.shape}")


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — Aggregate Forward Portion (avg / min / max)
# ═══════════════════════════════════════════════════════════════════

log("\n" + "=" * 65)
log("STEP 4 — Aggregate Forward Percentage Vectors")
log("=" * 65)

pct_stats = {}  # key -> {'avg': array, 'min': array, 'max': array}

for key in ['O', 'H', 'L', 'C']:
    matrix = forward_matrices[key]

    # Use nanmean/nanmin/nanmax in case of any NaN from truncation
    avg = np.nanmean(matrix, axis=1)
    mn  = np.nanmin(matrix, axis=1)
    mx  = np.nanmax(matrix, axis=1)

    pct_stats[key] = {'avg': avg, 'min': mn, 'max': mx}

    log(f"  T_{key}_pct_avg: {np.round(avg[:5], 6)}... (showing first 5 of {len(avg)})")
    log(f"  T_{key}_pct_min: {np.round(mn[:5], 6)}...")
    log(f"  T_{key}_pct_max: {np.round(mx[:5], 6)}...")


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — Compound Price Projection
# ═══════════════════════════════════════════════════════════════════

log("\n" + "=" * 65)
log("STEP 5 — Compound Price Projections")
log(f"  Anchors — O:{T0_price['O']:.2f}  H:{T0_price['H']:.2f}  "
    f"L:{T0_price['L']:.2f}  C:{T0_price['C']:.2f}")
log("=" * 65)


def compound_prices(pct_series, base_price, series_name):
    """Compound a percentage series from a base price."""
    prices = []
    prev = base_price
    for idx, pct in enumerate(pct_series):
        if np.isnan(pct):
            log(f"    ⚠ {series_name} T(+{idx+1}): NaN pct, carrying forward")
            prices.append(float(prev))
            continue
        if abs(pct) > PCT_CLAMP:
            log(f"    ⚠ {series_name} T(+{idx+1}): pct={pct:.4f} exceeds "
                f"±{PCT_CLAMP} clamp threshold")
        prev = prev * (1.0 + pct / 100.0)
        prices.append(float(prev))
    return prices


price_projections = {}  # e.g. 'O_avg' -> [32 prices]

for key in ['O', 'H', 'L', 'C']:
    base = T0_price[key]
    for stat in ['avg', 'min', 'max']:
        label = f"{key}_{stat}"
        price_projections[label] = compound_prices(
            pct_stats[key][stat], base, f"T_{key}_{stat}"
        )

# Log first and last projected prices
for key in ['O', 'H', 'L', 'C']:
    for stat in ['avg', 'min', 'max']:
        label = f"{key}_{stat}"
        p = price_projections[label]
        log(f"  T_{key}_{stat}: [{p[0]:.2f} ... {p[-1]:.2f}] ({len(p)} values)")


# ═══════════════════════════════════════════════════════════════════
# STEP 6 — Write to Forecast Tab
# ═══════════════════════════════════════════════════════════════════

log("\n" + "=" * 65)
log("STEP 6 — Write to Forecast Tab")
log("=" * 65)

write_forecasts([
    ('RPP32_Open_Avg_P',  price_projections['O_avg'],  T0_price['O']),
    ('RPP32_Open_Min_P',  price_projections['O_min'],  T0_price['O']),
    ('RPP32_Open_Max_P',  price_projections['O_max'],  T0_price['O']),
    ('RPP32_High_Avg_P',  price_projections['H_avg'],  T0_price['H']),
    ('RPP32_High_Min_P',  price_projections['H_min'],  T0_price['H']),
    ('RPP32_High_Max_P',  price_projections['H_max'],  T0_price['H']),
    ('RPP32_Low_Avg_P',   price_projections['L_avg'],  T0_price['L']),
    ('RPP32_Low_Min_P',   price_projections['L_min'],  T0_price['L']),
    ('RPP32_Low_Max_P',   price_projections['L_max'],  T0_price['L']),
    ('RPP32_Close_Avg_P', price_projections['C_avg'],  T0_price['C']),
    ('RPP32_Close_Min_P', price_projections['C_min'],  T0_price['C']),
    ('RPP32_Close_Max_P', price_projections['C_max'],  T0_price['C']),
])

# ═══════════════════════════════════════════════════════════════════
# COMPLETE
# ═══════════════════════════════════════════════════════════════════

log("\n" + "=" * 65)
log("RPP 32-20 COMPLETE")
log("=" * 65)
for key in ['O', 'H', 'L', 'C']:
    cr = cascade_results[key]
    p_avg = price_projections[f"{key}_avg"]
    log(f"  {key}: {cr['survivor_count']} analogues, cascade depth {cr['final_step']}, "
        f"projected {T0_price[key]:.2f} \u2192 {p_avg[-1]:.2f} (avg, +{N_FORWARD} indices)")
log(f"  12 price paths \u00d7 {N_FORWARD} values written to Forecast Tab")


# =====================================================================
# STEP 7 -- Diagnostic Report to Working Data
# =====================================================================

REPORT_SHEET = 'RPP 32-20 Report'
_col_names = {'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close'}

def _w(col_name, values, column=None):
    to_working(col_name, values, sheet=REPORT_SHEET, column=column)

log("\n" + "=" * 65)
log("STEP 7 -- Writing diagnostic report to Working Data")
log("=" * 65)

# -- Section 1: Header / Config --
row = 0
header_labels = [
    '=== RPP 32-20 DIAGNOSTIC REPORT ===',
    '',
    'Strategy',
    'Run Timestamp',
    'Dataset Size',
    'Max Index (T0)',
    'N_FORWARD',
    'N_PAST_MAX',
    'INITIAL_TOL',
    'CASCADE_TOL_MUL',
    'MIN_SURVIVORS',
    'Candidate Range',
    '',
]
header_values = [
    '',
    '',
    '32-20 RPP_15%',
    str(df.iloc[max_I].get('date_time_market', 'N/A')),
    n,
    max_I,
    N_FORWARD,
    N_PAST_MAX,
    INITIAL_TOL,
    CASCADE_TOL_MUL,
    MIN_SURVIVORS,
    f'iloc [{min_j}, {max_j}]',
    '',
]
_w('Label', header_labels, column='A')
_w('Value', header_values, column='B')
row = len(header_labels)

# -- Section 2: Step 1 -- T(0) Prices and Anchors --
s1_labels = ['=== STEP 1: T(0) PRICES ===', '']
s1_vals = ['', '']

for key in ['O', 'H', 'L', 'C']:
    s1_labels.append(f'T(0) {_col_names[key]} Price')
    s1_vals.append(round(T0_price[key], 4))
    s1_labels.append(f'T(0) {_col_names[key]} %')
    s1_vals.append(round(T0_pct[key], 6))

s1_labels += ['', '=== STEP 1: CASCADE ANCHORS ===', '']
s1_vals += ['', '', '']

anchor_header_labels = ['T']
anchor_open = ['Open_%']
anchor_high = ['High_%']
anchor_low = ['Low_%']
anchor_close = ['Close_%']

for step in range(N_PAST_MAX):
    t_label = f'T(0)' if step == 0 else f'T(-{step})'
    anchor_header_labels.append(t_label)
    anchor_open.append(round(T_neg_pct['O'].get(step, 0) or 0, 6))
    anchor_high.append(round(T_neg_pct['H'].get(step, 0) or 0, 6))
    anchor_low.append(round(T_neg_pct['L'].get(step, 0) or 0, 6))
    anchor_close.append(round(T_neg_pct['C'].get(step, 0) or 0, 6))

all_labels = header_labels + s1_labels + anchor_header_labels
all_values = header_values + s1_vals + anchor_open

_w('Label', all_labels, column='A')
_w('Value', all_values, column='B')
_w('High_%', header_labels + s1_labels + anchor_high, column='C')
_w('Low_%', header_labels + s1_vals + anchor_low, column='D')
_w('Close_%', header_labels + s1_vals + anchor_close, column='E')

# -- Section 3: Step 2 -- M_I0 Tolerance Bands --
s2_start = len(all_labels) + 2
s2_labels = ['', '=== STEP 2: M_I0 TOLERANCE BANDS ===', '', 'Column', '']
s2_b = ['', '', '', '', '']
s2_c = ['', '', '', '', '']
s2_d = ['', '', '', '', '']
s2_e = ['', '', '', '', '']

for key_idx, key in enumerate(['O', 'H', 'L', 'C']):
    anchor_val = T0_pct[key]
    band = abs(anchor_val) * INITIAL_TOL
    col_data = [s2_b, s2_c, s2_d, s2_e][key_idx]
    col_data[3] = _col_names[key]

s2_labels += ['Anchor %']
for key_idx, key in enumerate(['O', 'H', 'L', 'C']):
    [s2_b, s2_c, s2_d, s2_e][key_idx].append(round(T0_pct[key], 6))

s2_labels += ['Tolerance']
for key_idx, key in enumerate(['O', 'H', 'L', 'C']):
    [s2_b, s2_c, s2_d, s2_e][key_idx].append(f'{INITIAL_TOL*100:.1f}% relative')

s2_labels += ['Band (+/-)']
for key_idx, key in enumerate(['O', 'H', 'L', 'C']):
    band = abs(T0_pct[key]) * INITIAL_TOL
    [s2_b, s2_c, s2_d, s2_e][key_idx].append(round(band, 6))

s2_labels += ['Lower Bound']
for key_idx, key in enumerate(['O', 'H', 'L', 'C']):
    band = abs(T0_pct[key]) * INITIAL_TOL
    [s2_b, s2_c, s2_d, s2_e][key_idx].append(round(T0_pct[key] - band, 6))

s2_labels += ['Upper Bound']
for key_idx, key in enumerate(['O', 'H', 'L', 'C']):
    band = abs(T0_pct[key]) * INITIAL_TOL
    [s2_b, s2_c, s2_d, s2_e][key_idx].append(round(T0_pct[key] + band, 6))

s2_labels += ['M_I0 Pool Size']
for key_idx, key in enumerate(['O', 'H', 'L', 'C']):
    [s2_b, s2_c, s2_d, s2_e][key_idx].append(cascade_results[key]['counts'].get(0, 0))

# -- Section 4: Cascade step-by-step pool counts --
s2_labels += ['', '=== STEP 2: CASCADE DEPTH SUMMARY ===', '', 'Step']
for lst in [s2_b, s2_c, s2_d, s2_e]:
    lst += ['', '', '', '']

# Fill header row with column names
s2_b[-1] = 'Open'
s2_c[-1] = 'High'
s2_d[-1] = 'Low'
s2_e[-1] = 'Close'

max_depth = max(cascade_results[k]['final_step'] for k in ['O', 'H', 'L', 'C'])
for step in range(max_depth + 1):
    tol = get_tolerance(step)
    s2_labels.append(f'M_I{step} (tol={tol*100:.1f}%)')
    for key_idx, key in enumerate(['O', 'H', 'L', 'C']):
        count = cascade_results[key]['counts'].get(step, '-')
        [s2_b, s2_c, s2_d, s2_e][key_idx].append(count)

s2_labels += ['', 'Final Step']
for key_idx, key in enumerate(['O', 'H', 'L', 'C']):
    lst = [s2_b, s2_c, s2_d, s2_e][key_idx]
    lst.append('')
    lst.append(cascade_results[key]['final_step'])

s2_labels.append('Final Survivors')
for key_idx, key in enumerate(['O', 'H', 'L', 'C']):
    [s2_b, s2_c, s2_d, s2_e][key_idx].append(cascade_results[key]['survivor_count'])

# Append section 2 to all columns
full_a = all_labels + s2_labels
full_b = all_values + s2_b
full_c = list(header_labels + s1_labels + anchor_high) + s2_c
full_d = list(header_labels + s1_vals + anchor_low) + s2_d
full_e = list(header_labels + s1_vals + anchor_close) + s2_e

_w('Label', full_a, column='A')
_w('Open / Value', full_b, column='B')
_w('High', full_c, column='C')
_w('Low', full_d, column='D')
_w('Close', full_e, column='E')

# -- Section 5: Full M_I0 candidate lists per OHLC column --
# Written as separate column groups starting at column G onward
col_offset_map = {'O': ('G', 'H', 'I'), 'H': ('K', 'L', 'M'),
                  'L': ('O', 'P', 'Q'), 'C': ('S', 'T', 'U')}

for key in ['O', 'H', 'L', 'C']:
    cr = cascade_results[key]
    final_pool = cr['survivors']
    pct_arr = pct_arrays[key]
    anchor_val = T0_pct[key]

    col_idx, col_pct, col_delta = col_offset_map[key]

    idx_header = [f'{_col_names[key]} M_I0 Candidates', 'Index (iloc)']
    pct_header = ['', f'{_col_names[key]}_%']
    delta_header = ['', 'Delta']

    idx_vals = []
    pct_vals = []
    delta_vals = []

    m_i0_pool = cr['pools'].get(0, [])
    for j in m_i0_pool:
        val = float(pct_arr[j])
        idx_vals.append(j)
        pct_vals.append(round(val, 6))
        delta_vals.append(round(val - anchor_val, 6))

    _w(f'{_col_names[key]}_M_I0_Index', idx_header + idx_vals, column=col_idx)
    _w(f'{_col_names[key]}_M_I0_Pct', pct_header + pct_vals, column=col_pct)
    _w(f'{_col_names[key]}_M_I0_Delta', delta_header + delta_vals, column=col_delta)

    log(f"  {_col_names[key]}: wrote {len(m_i0_pool)} M_I0 candidates to columns {col_idx}-{col_delta}")

# -- Section 6: Final survivors detail --
surv_offset_map = {'O': ('W', 'X', 'Y'), 'H': ('AA', 'AB', 'AC'),
                   'L': ('AE', 'AF', 'AG'), 'C': ('AI', 'AJ', 'AK')}

for key in ['O', 'H', 'L', 'C']:
    cr = cascade_results[key]
    final_survivors = cr['survivors']
    pct_arr = pct_arrays[key]
    anchor_val = T0_pct[key]

    col_idx, col_pct, col_delta = surv_offset_map[key]

    idx_header = [f'{_col_names[key]} Final Survivors (M_I{cr["final_step"]})', 'Index (iloc)']
    pct_header = ['', f'{_col_names[key]}_%']
    delta_header = ['', 'Delta']

    idx_vals = []
    pct_vals = []
    delta_vals = []

    for j in final_survivors:
        val = float(pct_arr[j])
        idx_vals.append(j)
        pct_vals.append(round(val, 6))
        delta_vals.append(round(val - anchor_val, 6))

    _w(f'{_col_names[key]}_Surv_Index', idx_header + idx_vals, column=col_idx)
    _w(f'{_col_names[key]}_Surv_Pct', pct_header + pct_vals, column=col_pct)
    _w(f'{_col_names[key]}_Surv_Delta', delta_header + delta_vals, column=col_delta)

    log(f"  {_col_names[key]}: wrote {len(final_survivors)} final survivors to columns {col_idx}-{col_delta}")

# -- Section 7: Projection summary appended to main report --
proj_labels = ['', '=== STEP 5: PROJECTION SUMMARY ===', '', 'Metric']
proj_b = ['', '', '', 'Open']
proj_c = ['', '', '', 'High']
proj_d = ['', '', '', 'Low']
proj_e = ['', '', '', 'Close']

proj_labels.append('T(0) Base Price')
for key_idx, key in enumerate(['O', 'H', 'L', 'C']):
    [proj_b, proj_c, proj_d, proj_e][key_idx].append(round(T0_price[key], 4))

for stat, stat_label in [('avg', 'Avg'), ('min', 'Min'), ('max', 'Max')]:
    proj_labels.append(f'{stat_label} T+1 Price')
    for key_idx, key in enumerate(['O', 'H', 'L', 'C']):
        [proj_b, proj_c, proj_d, proj_e][key_idx].append(
            round(price_projections[f"{key}_{stat}"][0], 4))

    proj_labels.append(f'{stat_label} T+{N_FORWARD} Price')
    for key_idx, key in enumerate(['O', 'H', 'L', 'C']):
        [proj_b, proj_c, proj_d, proj_e][key_idx].append(
            round(price_projections[f"{key}_{stat}"][-1], 4))

final_a = full_a + proj_labels
final_b = full_b + proj_b
final_c = full_c + proj_c
final_d = full_d + proj_d
final_e = full_e + proj_e

_w('Label', final_a, column='A')
_w('Open / Value', final_b, column='B')
_w('High', final_c, column='C')
_w('Low', final_d, column='D')
_w('Close', final_e, column='E')

log(f"\n  Diagnostic report written to Working Data sheet '{REPORT_SHEET}'")