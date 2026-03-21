import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
# 8-8 Recursive Price Projection (RPP)
# Timeframe agnostic — works on any model frequency
# ═══════════════════════════════════════════════════════════════════

# ── CONFIGURATION ──────────────────────────────────────────────────
# FIX: TOLERANCE was absolute ±1.0 pp. Spec requires relative 2.5% of anchor
#      magnitude: "tolerance = abs(anchor_value) * 0.025". Replaced with
#      REL_TOLERANCE multiplier so the band scales with the anchor value.
REL_TOLERANCE   = 0.025  # 2.5% relative tolerance per spec
MIN_POOL        = 8      # minimum candidates required per series
N_SURVIVORS     = 8      # number of survivors to select
N_INDICES       = 8      # number of forward indices to project
PCT_CLAMP       = 50.0   # warn if any pct move exceeds this threshold
VERBOSE         = True   # set False to mute debug prints in production

# ── HELPERS ────────────────────────────────────────────────────────
def log(msg):
    if VERBOSE:
        print(msg)

def write_forecasts(forecast_pairs):
    failed = []
    for col_name, vals in forecast_pairs:
        try:
            to_forecast(col_name, vals)
            log(f"  ✓ {col_name}")
        except Exception as e:
            failed.append((col_name, str(e)))
    if failed:
        for col_name, err in failed:
            print(f"  ✗ FAILED: {col_name} — {err}")
        raise RuntimeError(
            f"RPP 8-8: {len(failed)} forecast write(s) failed — check above"
        )

# ── STEP 1 — Establish T(0) and baseline vectors ──────────────────

log("=" * 60)
log("STEP 1 — Establish T(0)")
log("=" * 60)

df_sorted = df.sort_values('date_time_market', ascending=True)
t0        = df_sorted.iloc[-1]

# T(0) price anchors
T_O_price_T0 = float(t0['open'])
T_H_price_T0 = float(t0['high'])
T_L_price_T0 = float(t0['low'])
T_C_price_T0 = float(t0['close'])

T_O_pct_T0   = float(t0['open_pct'])
T_H_pct_T0   = float(t0['high_pct'])
T_L_pct_T0   = float(t0['low_pct'])
T_C_pct_T0   = float(t0['close_pct'])

log(f"  T(0) date : {t0['date_time_market']}")
log(f"  T_O  price={T_O_price_T0:.2f}  pct={T_O_pct_T0:.4f}")
log(f"  T_H  price={T_H_price_T0:.2f}  pct={T_H_pct_T0:.4f}")
log(f"  T_L  price={T_L_price_T0:.2f}  pct={T_L_pct_T0:.4f}")
log(f"  T_C  price={T_C_price_T0:.2f}  pct={T_C_pct_T0:.4f}")

# Baseline vectors — iloc[-9:-1]: the 8 completed days before today
# FIX: Removed [::-1] reversal. Spec defines baseline in ascending
#      chronological order: baseline[0] = iloc[-9] (oldest),
#      baseline[7] = iloc[-2] (newest). Candidate vectors in Step 3 use
#      the same ascending slice [i-8:i], so element-for-element alignment
#      is correct without reversal.
T_O_price_baseline = df_sorted['open'].iloc[-9:-1].to_numpy()
T_H_price_baseline = df_sorted['high'].iloc[-9:-1].to_numpy()
T_L_price_baseline = df_sorted['low'].iloc[-9:-1].to_numpy()
T_C_price_baseline = df_sorted['close'].iloc[-9:-1].to_numpy()

T_O_pct_baseline   = df_sorted['open_pct'].iloc[-9:-1].to_numpy()
T_H_pct_baseline   = df_sorted['high_pct'].iloc[-9:-1].to_numpy()
T_L_pct_baseline   = df_sorted['low_pct'].iloc[-9:-1].to_numpy()
T_C_pct_baseline   = df_sorted['close_pct'].iloc[-9:-1].to_numpy()

log(f"\n  T_O_pct_baseline   : {T_O_pct_baseline}")
log(f"  T_H_pct_baseline   : {T_H_pct_baseline}")
log(f"  T_L_pct_baseline   : {T_L_pct_baseline}")
log(f"  T_C_pct_baseline   : {T_C_pct_baseline}")
log(f"  T_O_price_baseline : {T_O_price_baseline}")
log(f"  T_H_price_baseline : {T_H_price_baseline}")
log(f"  T_L_price_baseline : {T_L_price_baseline}")
log(f"  T_C_price_baseline : {T_C_price_baseline}")

# ── STEP 2 — Tolerance Pool Filter ────────────────────────────────
#
# FIX (anchor): Was using T(0)'s own percentage value as the anchor.
#   Spec says: "take baseline[0] (the first element of the baseline
#   vector, i.e. the value at iloc[-9]) as the anchor value."
#   Now passes baseline[0] to build_pool instead of T_X_pct_T0.
#
# FIX (tolerance): Was absolute ±1.0 pp. Now relative:
#   tolerance = abs(anchor) * REL_TOLERANCE  (2.5% of anchor magnitude)
#
# FIX (comparison position): Was checking pct_arr[i] — the pivot value.
#   Spec says: "abs(df[col].iloc[i - 8] - anchor_value) <= tolerance"
#   Now checks pct_arr[i - N_INDICES] — the start of the candidate's
#   context window, which must align with baseline[0].
#
# FIX (boundary): Was range(8, n-9) allowing forward windows to overlap
#   the baseline. Spec says i <= len(df) - 18 so the full 17-row window
#   (8 context + 1 pivot + 8 forward) cannot reach the baseline or today.
#   Now range(N_INDICES, n - 17) → max i = n-18.

log("\n" + "=" * 60)
log("STEP 2 — Tolerance Pool Filter")
log("=" * 60)

n             = len(df_sorted)
O_pct_arr     = df_sorted['open_pct'].to_numpy()
H_pct_arr     = df_sorted['high_pct'].to_numpy()
L_pct_arr     = df_sorted['low_pct'].to_numpy()
C_pct_arr     = df_sorted['close_pct'].to_numpy()
O_price_arr   = df_sorted['open'].to_numpy()
H_price_arr   = df_sorted['high'].to_numpy()
L_price_arr   = df_sorted['low'].to_numpy()
C_price_arr   = df_sorted['close'].to_numpy()

def build_pool(pct_arr, anchor, series_name):
    tolerance = abs(anchor) * REL_TOLERANCE
    cands = []
    for i in range(N_INDICES, n - 17):
        if abs(pct_arr[i - N_INDICES] - anchor) <= tolerance:
            cands.append(i)
    pool_count = len(cands)
    log(f"  {series_name}: anchor={anchor:.4f}  tol=±{tolerance:.4f} (2.5% rel)  "
        f"pool_count={pool_count}")
    if pool_count < MIN_POOL:
        raise ValueError(
            f"RPP 8-8: {series_name} pool returned only {pool_count} "
            f"candidates (min={MIN_POOL}) — increase tolerance and re-run"
        )
    return cands, pool_count

# FIX: All four anchors now use baseline[0] consistently.
#      Previously T_L used L_pct_arr[n-1] instead of T_L_pct_T0 (copy-paste
#      inconsistency), and all four used T(0) values instead of baseline[0].
T_O_pool, T_O_pool_count = build_pool(O_pct_arr, T_O_pct_baseline[0], 'T_O')
T_H_pool, T_H_pool_count = build_pool(H_pct_arr, T_H_pct_baseline[0], 'T_H')
T_L_pool, T_L_pool_count = build_pool(L_pct_arr, T_L_pct_baseline[0], 'T_L')
T_C_pool, T_C_pool_count = build_pool(C_pct_arr, T_C_pct_baseline[0], 'T_C')

# ── STEP 3 — Vector Scoring & Survivors ───────────────────────────
#
# FIX: Candidate vector was pct_arr[i-7:i+1][::-1] — shifted 1 position
#      forward and included the pivot while missing the oldest context value.
#      Spec says: "df[col].iloc[i-8:i] (8 values ending at position i-1)"
#      Now uses pct_arr[i - N_INDICES : i] — the 8 values BEFORE the pivot,
#      in ascending chronological order matching the baseline.

log("\n" + "=" * 60)
log("STEP 3 — Vector Scoring & Survivors")
log("=" * 60)

def score_pool(pct_arr, pool, baseline_vec, series_name):
    scored = []
    for i in pool:
        vec = pct_arr[i - N_INDICES: i]
        if len(vec) < N_INDICES or np.any(np.isnan(vec)):
            continue
        score = float(np.sum(np.abs(vec - baseline_vec)))
        scored.append((i, score, vec))

    scored.sort(key=lambda x: x[1])
    actual_count = len(scored)

    if actual_count < N_SURVIVORS:
        raise ValueError(
            f"RPP 8-8: {series_name} produced only {actual_count} valid "
            f"scored candidates (need {N_SURVIVORS}) — increase tolerance"
        )

    top                  = scored[:N_SURVIVORS]
    survivors            = [x[0] for x in top]
    survivor_scores      = [x[1] for x in top]
    survivor_vectors     = np.array([x[2] for x in top])
    survivor_count       = len(survivors)

    log(f"  {series_name}: pool={len(pool)}  scored={actual_count}  "
        f"survivors={survivor_count}")
    for rank, (pos, score, _) in enumerate(top):
        date_val = df_sorted.iloc[pos]['date_time_market']
        log(f"    Rank {rank+1}: pos={pos}  date={date_val}  score={score:.4f}")

    return scored, survivors, survivor_scores, survivor_vectors, survivor_count

T_O_scored, T_O_survivors, T_O_survivor_scores, T_O_survivor_vectors, T_O_survivor_count = \
    score_pool(O_pct_arr, T_O_pool, T_O_pct_baseline, 'T_O')

T_H_scored, T_H_survivors, T_H_survivor_scores, T_H_survivor_vectors, T_H_survivor_count = \
    score_pool(H_pct_arr, T_H_pool, T_H_pct_baseline, 'T_H')

T_L_scored, T_L_survivors, T_L_survivor_scores, T_L_survivor_vectors, T_L_survivor_count = \
    score_pool(L_pct_arr, T_L_pool, T_L_pct_baseline, 'T_L')

T_C_scored, T_C_survivors, T_C_survivor_scores, T_C_survivor_vectors, T_C_survivor_count = \
    score_pool(C_pct_arr, T_C_pool, T_C_pct_baseline, 'T_C')

# ── STEP 4 — Build 17x8 Matrices ──────────────────────────────────
# FIX: Context rows were reversed with [::-1]. Spec defines rows 0-7
#      as iloc[i-8:i] in ascending chronological order. Removed reversal.

log("\n" + "=" * 60)
log("STEP 4 — Build 17x8 Matrices")
log("=" * 60)

def build_matrix(pct_arr, price_arr, survivors, series_name):
    pct_cols_list   = []
    price_cols_list = []

    for pos in survivors:
        ctx_pct   = pct_arr  [pos - N_INDICES: pos]
        ctx_price = price_arr[pos - N_INDICES: pos]

        piv_pct   = np.array([pct_arr  [pos]])
        piv_price = np.array([price_arr[pos]])

        fwd_pct   = pct_arr  [pos + 1: pos + N_INDICES + 1]
        fwd_price = price_arr[pos + 1: pos + N_INDICES + 1]

        col_pct   = np.concatenate([ctx_pct,   piv_pct,   fwd_pct])
        col_price = np.concatenate([ctx_price, piv_price, fwd_price])

        pct_cols_list  .append(col_pct)
        price_cols_list.append(col_price)

    pct_matrix   = np.column_stack(pct_cols_list)
    price_matrix = np.column_stack(price_cols_list)

    log(f"  {series_name}_pct_matrix   shape: {pct_matrix.shape}")
    log(f"  {series_name}_price_matrix shape: {price_matrix.shape}")

    return pct_matrix, price_matrix

T_O_pct_matrix, T_O_price_matrix = build_matrix(O_pct_arr, O_price_arr, T_O_survivors, 'T_O')
T_H_pct_matrix, T_H_price_matrix = build_matrix(H_pct_arr, H_price_arr, T_H_survivors, 'T_H')
T_L_pct_matrix, T_L_price_matrix = build_matrix(L_pct_arr, L_price_arr, T_L_survivors, 'T_L')
T_C_pct_matrix, T_C_price_matrix = build_matrix(C_pct_arr, C_price_arr, T_C_survivors, 'T_C')

# ── STEP 5 — Aggregate Forward Portion ────────────────────────────

log("\n" + "=" * 60)
log("STEP 5 — Aggregate Forward Percentage Vectors")
log("=" * 60)

def aggregate_forward(pct_matrix, series_name):
    fwd = pct_matrix[9:, :]  # rows 9-16: the 8 forward indices
    avg = fwd.mean(axis=1)
    mn  = fwd.min(axis=1)
    mx  = fwd.max(axis=1)
    log(f"  {series_name}_pct_avg : {np.round(avg, 4)}")
    log(f"  {series_name}_pct_min : {np.round(mn,  4)}")
    log(f"  {series_name}_pct_max : {np.round(mx,  4)}")
    return avg, mn, mx

T_O_pct_avg, T_O_pct_min, T_O_pct_max = aggregate_forward(T_O_pct_matrix, 'T_O')
T_H_pct_avg, T_H_pct_min, T_H_pct_max = aggregate_forward(T_H_pct_matrix, 'T_H')
T_L_pct_avg, T_L_pct_min, T_L_pct_max = aggregate_forward(T_L_pct_matrix, 'T_L')
T_C_pct_avg, T_C_pct_min, T_C_pct_max = aggregate_forward(T_C_pct_matrix, 'T_C')

# ── STEP 6 — Compound into Price Paths ────────────────────────────

log("\n" + "=" * 60)
log("STEP 6 — Compound Price Projections")
log(f"  Anchors — T_O:{T_O_price_T0:.2f}  T_H:{T_H_price_T0:.2f}  "
    f"T_L:{T_L_price_T0:.2f}  T_C:{T_C_price_T0:.2f}")
log("=" * 60)

def compound_prices(pct_series, base_price, series_name):
    prices = []
    prev   = base_price
    for idx, pct in enumerate(pct_series):
        if abs(pct) > PCT_CLAMP:
            print(f"  ⚠ WARNING: {series_name} T(+{idx+1}) pct={pct:.2f} "
                  f"exceeds clamp threshold ±{PCT_CLAMP}")
        prev = prev * (1 + pct / 100.0)
        prices.append(float(prev))
    return prices

T_O_price_avg = compound_prices(T_O_pct_avg, T_O_price_T0, 'T_O_avg')
T_O_price_min = compound_prices(T_O_pct_min, T_O_price_T0, 'T_O_min')
T_O_price_max = compound_prices(T_O_pct_max, T_O_price_T0, 'T_O_max')

T_H_price_avg = compound_prices(T_H_pct_avg, T_H_price_T0, 'T_H_avg')
T_H_price_min = compound_prices(T_H_pct_min, T_H_price_T0, 'T_H_min')
T_H_price_max = compound_prices(T_H_pct_max, T_H_price_T0, 'T_H_max')

T_L_price_avg = compound_prices(T_L_pct_avg, T_L_price_T0, 'T_L_avg')
T_L_price_min = compound_prices(T_L_pct_min, T_L_price_T0, 'T_L_min')
T_L_price_max = compound_prices(T_L_pct_max, T_L_price_T0, 'T_L_max')

T_C_price_avg = compound_prices(T_C_pct_avg, T_C_price_T0, 'T_C_avg')
T_C_price_min = compound_prices(T_C_pct_min, T_C_price_T0, 'T_C_min')
T_C_price_max = compound_prices(T_C_pct_max, T_C_price_T0, 'T_C_max')

log(f"  T_O_price_avg : {[round(v,2) for v in T_O_price_avg]}")
log(f"  T_O_price_min : {[round(v,2) for v in T_O_price_min]}")
log(f"  T_O_price_max : {[round(v,2) for v in T_O_price_max]}")
log(f"  T_H_price_avg : {[round(v,2) for v in T_H_price_avg]}")
log(f"  T_H_price_min : {[round(v,2) for v in T_H_price_min]}")
log(f"  T_H_price_max : {[round(v,2) for v in T_H_price_max]}")
log(f"  T_L_price_avg : {[round(v,2) for v in T_L_price_avg]}")
log(f"  T_L_price_min : {[round(v,2) for v in T_L_price_min]}")
log(f"  T_L_price_max : {[round(v,2) for v in T_L_price_max]}")
log(f"  T_C_price_avg : {[round(v,2) for v in T_C_price_avg]}")
log(f"  T_C_price_min : {[round(v,2) for v in T_C_price_min]}")
log(f"  T_C_price_max : {[round(v,2) for v in T_C_price_max]}")

# ── STEP 7 — Write to Forecast Tab ────────────────────────────────

log("\n" + "=" * 60)
log("STEP 7 — Writing to Forecast Tab")
log("=" * 60)

write_forecasts([
    ('RPP_Open_Avg_P',  T_O_price_avg),
    ('RPP_Open_Min_P',  T_O_price_min),
    ('RPP_Open_Max_P',  T_O_price_max),
    ('RPP_High_Avg_P',  T_H_price_avg),
    ('RPP_High_Min_P',  T_H_price_min),
    ('RPP_High_Max_P',  T_H_price_max),
    ('RPP_Low_Avg_P',   T_L_price_avg),
    ('RPP_Low_Min_P',   T_L_price_min),
    ('RPP_Low_Max_P',   T_L_price_max),
    ('RPP_Close_Avg_P', T_C_price_avg),
    ('RPP_Close_Min_P', T_C_price_min),
    ('RPP_Close_Max_P', T_C_price_max),
])

log(f"\n  RPP 8-8 complete — 12 price paths written to Forecast Tab")
log(f"  Projection: T(+1) to T(+{N_INDICES}) from T(0) = {t0['date_time_market']}")
