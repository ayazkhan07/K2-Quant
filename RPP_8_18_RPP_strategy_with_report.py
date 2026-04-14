"""
8-18 RPP - Recursive Price Projection with full OUTPUTS report tables.

Paste into the strategy editor or save via save_strategy. Uses canonical
frame columns: high_pct, low_pct, high, low, date_time_market, #.
"""
import numpy as np
import pandas as pd

# ============================================================
# 8-18 RPP: Recursive Price Projection Strategy
# 8 = Lookback / Survivors | 18 = Forward forecasting
# RPP = Recursive Price Projection
# ============================================================

# --- RULES ---
R1 = 8         # Minimum recursive steps
R2_Lo = 7      # Minimum survivors after final step
R2_Hi = 10     # Maximum survivors after final step
N_FORWARD = 18 # Forward projection steps
N_FOCAL = 24   # Focal series length

# ------------------------- REPORT HEADER ---------------------
report_header('8-18 RPP (Recursive Price Projection)')

report_config([
    # --- Nomenclature (frame + strategy symbols) ---
    ('#', "Row index in strategy space; last bar = focal (today).",
     'Integer bar id from the execution frame; maps to dataframe position via idx_to_pos.'),
    ('i_focal', '(set at runtime)', 'Same as max # - anchor bar for tolerance matching.'),
    ('high_pct / low_pct', 'Frame columns', 'Percent change vs prior bar same field (runner contract).'),
    ('P_h / P_L', '(set at runtime)', 'Nominal high / low price at focal bar (anchor levels).'),
    ('H / L arrays', 'Vectors over rows', 'high_pct and low_pct as float arrays aligned to #.'),
    ('R1', R1, 'Minimum recursive filter steps applied along the focal history.'),
    ('R2_Lo, R2_Hi', f'{R2_Lo} .. {R2_Hi}', 'Survivor count band after recursion completes.'),
    ('N_FORWARD', N_FORWARD, 'Forward steps: % rows pulled after each survivor index.'),
    ('N_FOCAL', N_FOCAL, 'Length of focal history (bars before focal, inclusive).'),
    ('Mh_*, ML_*', 'Matrices', 'Forward high% / low% windows stacked per survivor (rows=survivors).'),
    ('Mh_Avg_price ...', 'Forecast series', 'Compounded prices from P_h / P_L using avg/min/max % paths.'),
    # --- Tunables ---
    ('R1 (tunable)', R1, 'Recursive depth for tolerance cascade.'),
    ('R2_Lo (tunable)', R2_Lo, 'Minimum survivors required to accept a match.'),
    ('R2_Hi (tunable)', R2_Hi, 'Maximum survivors allowed for a valid match.'),
    ('N_FORWARD (tunable)', N_FORWARD, 'Number of forward % steps per survivor history.'),
    ('N_FOCAL (tunable)', N_FOCAL, 'How many past # values feed the focal series.'),
])

# --- Clear stale forecasts from prior runs ---
for _fc in ('Mh_Avg', 'Mh_Min', 'Mh_Max', 'ML_Avg', 'ML_Min', 'ML_Max'):
    to_forecast(_fc, [])

# --- STEP 1: Base Series ---
df_sorted = df.sort_values('#').reset_index(drop=True) if '#' in df.columns else df.sort_values('date_time_market').reset_index(drop=True)

if '#' not in df_sorted.columns:
    df_sorted['#'] = range(1, len(df_sorted) + 1)

I = df_sorted['#'].values.astype(int)
H = df_sorted['high_pct'].values.astype(float)
L = df_sorted['low_pct'].values.astype(float)

max_idx = int(I.max())
idx_to_pos = np.full(max_idx + 2, -1, dtype=int)
for pos in range(len(I)):
    idx_to_pos[int(I[pos])] = pos

i_focal = max_idx
pos_i = idx_to_pos[i_focal]
h_focal = H[pos_i]
l_focal = L[pos_i]
P_h = float(df_sorted.loc[pos_i, 'high'])
P_L = float(df_sorted.loc[pos_i, 'low'])

report_table(
    'Step 1: Focal anchor (today)',
    ['Quantity', 'Value', 'Notes'],
    [
        ['Rows in frame', len(df_sorted), 'After sort by # or date_time_market'],
        ['i_focal (#)', i_focal, 'Maximum # - treated as current bar'],
        ['pos_i (row index)', pos_i, 'iloc position for i_focal'],
        ['high_pct at focal', f'{h_focal:.3f}', 'H at T(0)'],
        ['low_pct at focal', f'{l_focal:.3f}', 'L at T(0)'],
        ['P_h (nominal high)', f'{P_h:.2f}', 'Price anchor for high projections'],
        ['P_L (nominal low)', f'{P_L:.2f}', 'Price anchor for low projections'],
    ],
)

# --- STEP 1b: Focal Series ---
If_arr = np.arange(i_focal, i_focal - N_FOCAL, -1)
Hf_arr = np.array([H[idx_to_pos[idx]] for idx in If_arr])
Lf_arr = np.array([L[idx_to_pos[idx]] for idx in If_arr])

focal_preview_n = min(6, len(If_arr))
focal_rows = []
for j in range(focal_preview_n):
    focal_rows.append([
        str(If_arr[j]),
        f'{Hf_arr[j]:.3f}',
        f'{Lf_arr[j]:.3f}',
        'Older -> newer toward focal' if j == 0 else '',
    ])
report_table(
    'Step 1b: Focal history sample (first bars in focal window)',
    ['# index', 'high_pct', 'low_pct', ''],
    focal_rows,
)

# --- STEP 2 & 3: Recursive Tolerance Matching ---


def run_recursive_match(series_name, focal_val, focal_series, I_arr, H_arr, L_arr, itp, r1, r2lo, r2hi):
    all_vals = H_arr if series_name == 'H' else L_arr
    mx = int(I_arr.max())

    target_mid = (r2lo + r2hi) / 2.0
    best_surv_count = 0
    best_params = None
    best_step_log = None
    deepest_step = 0
    combos_tried = 0
    pools_entered = 0

    for tp in range(1, 30):
        t = tp / 100.0
        tL_bound = focal_val - abs(focal_val) * t
        tU_bound = focal_val + abs(focal_val) * t

        msk = (all_vals >= tL_bound) & (all_vals <= tU_bound)
        init_I = I_arr[msk].copy()
        ic = len(init_I)

        if ic < r2hi:
            continue

        pools_entered += 1

        for ts in range(10, 200, 5):
            t1s = ts / 100.0
            for ti in range(5, 200, 5):
                t1i = ti / 100.0
                combos_tried += 1

                surv = init_I.copy()
                slog = []
                ok = True

                for k in range(r1):
                    tk = t1s + k * t1i
                    fk = focal_series[k + 1]

                    if np.isnan(fk):
                        ok = False
                        break

                    tLk = fk - abs(fk) * tk
                    tUk = fk + abs(fk) * tk

                    sh = surv - 1
                    vld = (sh >= 1) & (sh <= mx)
                    sh = sh[vld]
                    orig = sh + 1

                    ps = itp[sh]
                    vp = ps >= 0
                    sh = sh[vp]
                    orig = orig[vp]
                    ps = ps[vp]

                    sv = all_vals[ps]
                    pm = (sv >= tLk) & (sv <= tUk)

                    inc = len(sh)
                    surv = orig[pm]
                    outc = len(surv)

                    slog.append({
                        'step': k, 't_k': tk, 'focal_k': fk,
                        'tL_k': tLk, 'tU_k': tUk,
                        'in_count': inc, 'out_count': outc,
                    })

                    if outc < r2lo:
                        ok = False
                        break

                final_count = len(surv)
                last_step = slog[-1]['step'] if slog else 0

                if last_step > deepest_step:
                    deepest_step = last_step

                if abs(final_count - target_mid) < abs(best_surv_count - target_mid) or best_params is None:
                    best_surv_count = final_count
                    best_params = {'t': t, 't1_start': t1s, 't1_inc': t1i}
                    best_step_log = list(slog)

                if ok and r2lo <= final_count <= r2hi:
                    return {
                        'success': True,
                        't': t, 't1_start': t1s, 't1_inc': t1i,
                        'survivors': surv, 'step_log': slog,
                        'tL': tL_bound, 'tU': tU_bound,
                        'initial_matches': ic,
                    }

    return {
        'success': False,
        'best_surv_count': best_surv_count,
        'best_params': best_params,
        'best_step_log': best_step_log,
        'deepest_step': deepest_step,
        'combos_tried': combos_tried,
        'pools_entered': pools_entered,
    }


h_result = run_recursive_match('H', h_focal, Hf_arr, I, H, L, idx_to_pos, R1, R2_Lo, R2_Hi)
l_result = run_recursive_match('L', l_focal, Lf_arr, I, H, L, idx_to_pos, R1, R2_Lo, R2_Hi)


def _summarize_match(label, res):
    if not res.get('success', False):
        bp = res.get('best_params') or {}
        return [
            label,
            f"FAIL (best t={bp.get('t', '-')})",
            str(res.get('pools_entered', 0)),
            f"{res.get('best_surv_count', 0)} (need {R2_Lo}-{R2_Hi})",
            f"deepest step={res.get('deepest_step', 0)}, combos={res.get('combos_tried', 0)}",
        ]
    slog = res.get('step_log') or []
    last = slog[-1] if slog else {}
    return [
        label,
        f"{res['t']*100:.3f}%",
        str(res['initial_matches']),
        str(len(res['survivors'])),
        f"k={last.get('step', '-')} in->{last.get('in_count', '-')} out->{last.get('out_count', '-')}",
    ]


report_table(
    'Step 2-3: Recursive match summary',
    ['Track', 'Initial tol (rel)', 'Initial pool', 'Final survivors', 'Last step (in/out)'],
    [
        _summarize_match('High (H)', h_result),
        _summarize_match('Low (L)', l_result),
    ],
)

h_ok = h_result.get('success', False)
l_ok = l_result.get('success', False)

if not h_ok or not l_ok:
    # --- Diagnostic summary for failed tracks ---
    fail_diag_rows = []
    for label, res in [('High (H)', h_result), ('Low (L)', l_result)]:
        if not res.get('success', False):
            bp = res.get('best_params') or {}
            fail_diag_rows.append([
                label,
                res.get('best_surv_count', 0),
                f"t={bp.get('t', '-')}, t1s={bp.get('t1_start', '-')}, t1i={bp.get('t1_inc', '-')}",
                res.get('deepest_step', 0),
                res.get('combos_tried', 0),
                res.get('pools_entered', 0),
            ])
        else:
            fail_diag_rows.append([label, 'OK', '-', '-', '-', '-'])

    report_table(
        'Step 2-3: Failure diagnostics',
        ['Track', 'Best surv count', 'Best params', 'Deepest step', 'Combos tried', 'Tol pools entered'],
        fail_diag_rows,
    )

    # --- Best near-miss step traces ---
    for label, res in [('High (H)', h_result), ('Low (L)', l_result)]:
        if not res.get('success', False):
            bsl = res.get('best_step_log')
            if bsl:
                trace_rows = [
                    [s['step'], f"{s['t_k']:.3f}", f"{s['focal_k']:.3f}",
                     f"[{s['tL_k']:.4f}, {s['tU_k']:.4f}]",
                     s['in_count'], s['out_count']]
                    for s in bsl
                ]
                report_table(
                    f'Step 2-3: {label} best near-miss trace',
                    ['k', 't_k', 'focal_k', 'Band [tL, tU]', 'in_count', 'out_count'],
                    trace_rows,
                )

    report_table(
        'Step 2-3: Outcome',
        ['Status', 'Detail'],
        [
            [
                'FAILED',
                'No parameter set satisfied R1 recursive steps and '
                f'[{R2_Lo}, {R2_Hi}] survivor band for both H and L. '
                'Forecast columns cleared.',
            ],
        ],
    )

    for _fc in ('Mh_Avg', 'Mh_Min', 'Mh_Max', 'ML_Avg', 'ML_Min', 'ML_Max'):
        to_forecast(_fc, [])
    report_table(
        'Forecast Columns Cleared',
        ['Action', 'Detail'],
        [['Cleared all 6 forecast columns', 'No valid projection available this run']],
    )
else:
    h_slog = h_result.get('step_log') or []
    h_trace_rows = []
    for row in h_slog[:R1]:
        h_trace_rows.append([
            row['step'],
            f"{row['t_k']:.3f}",
            f"{row['focal_k']:.3f}",
            row['in_count'],
            row['out_count'],
        ])
    report_table(
        'Step 2-3: High track recursive trace (first passes)',
        ['k', 't_k', 'focal_k', 'in_count', 'out_count'],
        h_trace_rows if h_trace_rows else [['-', '-', '-', '-', '-']],

    )

    l_slog = l_result.get('step_log') or []
    l_trace_rows = []
    for row in l_slog[:R1]:
        l_trace_rows.append([
            row['step'],
            f"{row['t_k']:.3f}",
            f"{row['focal_k']:.3f}",
            row['in_count'],
            row['out_count'],
        ])
    report_table(
        'Step 2-3: Low track recursive trace (first passes)',
        ['k', 't_k', 'focal_k', 'in_count', 'out_count'],
        l_trace_rows if l_trace_rows else [['-', '-', '-', '-', '-']],
    )

    # --- STEP 4: Future of the Past ---
    h_surv_no_focal = h_result['survivors'][h_result['survivors'] != i_focal]
    l_surv_no_focal = l_result['survivors'][l_result['survivors'] != i_focal]

    report_table(
        'Step 4: Survivors (excluding focal)',
        ['Series', 'Count', 'Sample # indices'],
        [
            ['High', len(h_surv_no_focal), str(h_surv_no_focal[: min(8, len(h_surv_no_focal))])],
            ['Low', len(l_surv_no_focal), str(l_surv_no_focal[: min(8, len(l_surv_no_focal))])],
        ],
    )

    def build_matrices(survivor_indices, H_arr, L_arr, itp, n_fwd, mx):
        n = len(survivor_indices)
        h_m = np.zeros((n, n_fwd))
        l_m = np.zeros((n, n_fwd))
        for r, si in enumerate(survivor_indices):
            for c in range(n_fwd):
                fi = int(si) + c + 1
                if fi <= mx and itp[fi] >= 0:
                    p = itp[fi]
                    h_m[r, c] = H_arr[p]
                    l_m[r, c] = L_arr[p]
                else:
                    h_m[r, c] = np.nan
                    l_m[r, c] = np.nan
        return h_m, l_m

    Mh, Mh_L = build_matrices(h_surv_no_focal, H, L, idx_to_pos, N_FORWARD, max_idx)
    ML_H, ML = build_matrices(l_surv_no_focal, H, L, idx_to_pos, N_FORWARD, max_idx)

    report_table(
        'Step 4: Forward % matrices',
        ['Matrix', 'Shape (rows x cols)', 'Role'],
        [
            ['Mh (high % forward)', f'{Mh.shape[0]} x {Mh.shape[1]}', 'Per high survivor, N_FORWARD high_pct'],
            ['ML (low % forward)', f'{ML.shape[0]} x {ML.shape[1]}', 'Per low survivor, N_FORWARD low_pct'],
        ],
    )

    # --- STEP 5: Summary Statistics ---
    Mh_avg = np.nanmean(Mh, axis=0)
    Mh_min = np.nanmin(Mh, axis=0)
    Mh_max = np.nanmax(Mh, axis=0)
    ML_avg = np.nanmean(ML, axis=0)
    ML_min = np.nanmin(ML, axis=0)
    ML_max = np.nanmax(ML, axis=0)

    show_n = min(6, N_FORWARD)
    stat_rows = []
    for j in range(show_n):
        stat_rows.append([
            j + 1,
            f'{Mh_avg[j]:.3f}',
            f'{Mh_min[j]:.3f}',
            f'{Mh_max[j]:.3f}',
            f'{ML_avg[j]:.3f}',
            f'{ML_min[j]:.3f}',
            f'{ML_max[j]:.3f}',
        ])
    report_table(
        'Step 5: Aggregated forward % vectors (first steps)',
        ['Fwd step', 'Mh avg', 'Mh min', 'Mh max', 'ML avg', 'ML min', 'ML max'],
        stat_rows,
    )

    # --- STEP 6: Recursive Price Projections ---
    def recursive_prices(base_price, pct_vector):
        prices = np.zeros(len(pct_vector))
        p = round(float(base_price), 2)
        for j in range(len(pct_vector)):
            p = round(p * (1 + float(pct_vector[j]) / 100.0), 2)
            prices[j] = p
        return prices

    Mh_Avg_price = recursive_prices(P_h, Mh_avg)
    Mh_Min_price = recursive_prices(P_h, Mh_min)
    Mh_Max_price = recursive_prices(P_h, Mh_max)
    ML_Avg_price = recursive_prices(P_L, ML_avg)
    ML_Min_price = recursive_prices(P_L, ML_min)
    ML_Max_price = recursive_prices(P_L, ML_max)

    price_rows = []
    for j in range(show_n):
        price_rows.append([
            j + 1,
            f'{Mh_Avg_price[j]:.2f}',
            f'{Mh_Min_price[j]:.2f}',
            f'{Mh_Max_price[j]:.2f}',
            f'{ML_Avg_price[j]:.2f}',
            f'{ML_Min_price[j]:.2f}',
            f'{ML_Max_price[j]:.2f}',
        ])
    report_table(
        'Step 6: Compounded price paths (first steps)',
        ['Fwd step', 'Mh Avg $', 'Mh Min $', 'Mh Max $', 'ML Avg $', 'ML Min $', 'ML Max $'],
        price_rows,
    )

    # --- STEP 7: Write to Forecast (Tab 2) ---
    to_forecast('Mh_Avg', list(Mh_Avg_price))
    to_forecast('Mh_Min', list(Mh_Min_price))
    to_forecast('Mh_Max', list(Mh_Max_price))
    to_forecast('ML_Avg', list(ML_Avg_price))
    to_forecast('ML_Min', list(ML_Min_price))
    to_forecast('ML_Max', list(ML_Max_price))

    forecast_pairs = [
        ('Mh_Avg', Mh_Avg_price),
        ('Mh_Min', Mh_Min_price),
        ('Mh_Max', Mh_Max_price),
        ('ML_Avg', ML_Avg_price),
        ('ML_Min', ML_Min_price),
        ('ML_Max', ML_Max_price),
    ]
    report_table(
        'Forecast Columns Written',
        ['Column', 'Length'],
        [[n, len(v)] for n, v in forecast_pairs],
    )
