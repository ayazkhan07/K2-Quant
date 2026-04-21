import numpy as np
import pandas as pd
from datetime import timedelta
# ============================================================
# 8-18 RPP v2 : Adaptive Recursive Price Projection Strategy
# ============================================================
# ------------------------- TUNABLES --------------------------
R1 = 8          # Recursive depth (number of history steps)
R2_Lo = 7       # Min survivors required after a step
R2_Hi = 10      # Max survivors allowed after final step
N_FORWARD = 18  # Forward projection horizon
N_FOCAL = 24    # Length of focal series (bars before focal inclusive)
# New adaptive-search knobs
ADAPT_FACTOR = 1.25   # Multiplier when a recursion step falls short
MAX_T_ALLOWED = 2.0   # Absolute ceiling for any step's tolerance (200 %)
INIT_TOL_MAX = 0.29   # Maximum initial tolerance scanned (29 %)
# ----------------------------- REPORT HEADER -----------------------------
report_header('8-18 RPP v2 (adaptive recursive tolerance)')
report_config([
    ('R1', R1, 'Recursive steps to apply'),
    ('R2_Lo', R2_Lo, 'Minimum survivors per step'),
    ('R2_Hi', R2_Hi, 'Maximum survivors at completion'),
    ('N_FORWARD', N_FORWARD, 'Forward price projection steps'),
    ('N_FOCAL', N_FOCAL, 'Bars composing focal history'),
    ('ADAPT_FACTOR', ADAPT_FACTOR, 'Tolerance widening factor when survivor count < R2_Lo'),
    ('MAX_T_ALLOWED', f'{MAX_T_ALLOWED*100:.0f} %', 'Hard ceiling for any step tolerance'),
    ('INIT_TOL_MAX', f'{INIT_TOL_MAX*100:.0f} %', 'Upper bound scanned for initial tolerance'),
])
# Progress hook - no-op when strategy is run outside DPE.
try:
    __k2_progress__
except NameError:
    def __k2_progress__(*_a, **_k):
        pass

# --------------------------- PREP THE DATA -------------------------------
__k2_progress__('prep', f'n_rows={len(df):,}')
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
pos_i   = idx_to_pos[i_focal]
P_h     = float(df_sorted.loc[pos_i, 'high'])
P_L     = float(df_sorted.loc[pos_i, 'low'])
h_focal = H[pos_i]
l_focal = L[pos_i]
__k2_progress__('prep_done', f'max_idx={max_idx} focal_h={h_focal:.4f} focal_l={l_focal:.4f}')
report_table(
    'Step 1: Focal anchor',
    ['Quantity', 'Value', 'Notes'],
    [
        ['Rows in frame', len(df_sorted), 'After sort'],
        ['i_focal (#)', i_focal, 'Current bar'],
        ['pos_i (iloc)', pos_i, 'location inside dataframe'],
        ['high_pct focal', f"{h_focal:.6g}", ''],
        ['low_pct focal',  f"{l_focal:.6g}", ''],
        ['P_h (nom high)', f"{P_h:.6g}", 'Price anchor'],
        ['P_L (nom low)',  f"{P_L:.6g}", 'Price anchor'],
    ],
)
# --- Build focal history arrays (latest -> older)
If_arr = np.arange(i_focal, i_focal - N_FOCAL, -1)
Hf_arr = np.array([H[idx_to_pos[idx]] for idx in If_arr])
Lf_arr = np.array([L[idx_to_pos[idx]] for idx in If_arr])
report_table(
    'Step 1b: Focal history sample',
    ['# index', 'high_pct', 'low_pct'],
    [[int(If_arr[j]), f"{Hf_arr[j]:.6g}", f"{Lf_arr[j]:.6g}"] for j in range(len(If_arr))]
)
# -------------------- ADAPTIVE RECURSIVE MATCH ---------------------------
def run_recursive_match(series_name, focal_val, focal_series, I_arr, H_arr, L_arr, itp,
                        r1, r2lo, r2hi, adapt_factor, max_t_allowed):
    all_vals = H_arr if series_name == 'H' else L_arr
    mx = int(I_arr.max())
    best_failed = None
    max_loop = int(INIT_TOL_MAX * 100)
    __k2_progress__(f'track_{series_name}_start',
                    f'focal={focal_val:.4f} max_loop={max_loop} N={len(all_vals):,}')
    for tp in range(1, max_loop + 1):
        t0 = tp / 100.0
        lo0 = focal_val - abs(focal_val) * t0
        hi0 = focal_val + abs(focal_val) * t0
        init_idx = I_arr[(all_vals >= lo0) & (all_vals <= hi0)]
        if len(init_idx) < r2lo:
            continue
        __k2_progress__(f'track_{series_name}_tp',
                        f'tp={tp} t0={t0:.2f} init_matches={len(init_idx):,}')
        for ts in range(10, 200, 5):
            t1_start = ts / 100.0
            for ti in range(5, 200, 5):
                t1_inc = ti / 100.0
                survivors = init_idx.copy()
                step_log = []
                ok_flag = True
                for k in range(r1):
                    tk = t1_start + k * t1_inc
                    fk = focal_series[k + 1]
                    if np.isnan(fk):
                        ok_flag = False
                        break
                    adapt_ct = 0
                    while True:
                        lo_k = fk - abs(fk) * tk
                        hi_k = fk + abs(fk) * tk
                        sh = survivors - 1
                        vld = (sh >= 1) & (sh <= mx)
                        sh = sh[vld]
                        orig = sh + 1
                        ps = itp[sh]
                        vp = ps >= 0
                        sh, orig, ps = sh[vp], orig[vp], ps[vp]
                        sv = all_vals[ps]
                        mask = (sv >= lo_k) & (sv <= hi_k)
                        in_cnt = len(sh)
                        survivors = orig[mask]
                        out_cnt = len(survivors)
                        if out_cnt >= r2lo or tk >= max_t_allowed:
                            break
                        tk *= adapt_factor
                        adapt_ct += 1
                    step_log.append({'step': k, 't_k': tk, 'adapt_ct': adapt_ct,
                                     'in_count': in_cnt, 'out_count': out_cnt,
                                     'survivors': survivors.copy()})
                    if out_cnt < r2lo:
                        ok_flag = False
                        deepest_k = k
                        if best_failed is None or deepest_k > best_failed['deepest_k'] or \
                           (deepest_k == best_failed['deepest_k'] and out_cnt > best_failed['last_out']):
                            best_failed = {
                                't0': t0,
                                't1_start': t1_start,
                                't1_inc': t1_inc,
                                'initial_matches': len(init_idx),
                                'initial_indices': init_idx.copy(),
                                'deepest_k': deepest_k,
                                'last_out': out_cnt,
                                'step_log': list(step_log),
                            }
                        break
                if ok_flag and r2lo <= len(survivors) <= r2hi:
                    return {
                        'success': True,
                        't0': t0,
                        't1_start': t1_start,
                        't1_inc': t1_inc,
                        'initial_matches': len(init_idx),
                        'initial_indices': init_idx.copy(),
                        'survivors': survivors,
                        'step_log': step_log,
                    }
    return {'success': False, 'best_failed': best_failed}
__k2_progress__('match_H', 'dispatching high track')
h_result = run_recursive_match('H', h_focal, Hf_arr, I, H, L, idx_to_pos,
                               R1, R2_Lo, R2_Hi, ADAPT_FACTOR, MAX_T_ALLOWED)
__k2_progress__('match_H_done', f"success={h_result.get('success')}")
__k2_progress__('match_L', 'dispatching low track')
l_result = run_recursive_match('L', l_focal, Lf_arr, I, H, L, idx_to_pos,
                               R1, R2_Lo, R2_Hi, ADAPT_FACTOR, MAX_T_ALLOWED)
__k2_progress__('match_L_done', f"success={l_result.get('success')}")
# --------------- STEP 2 READOUT: TRACK DIAGNOSTICS -----------------------
def _track_readout(label, focal_val, price_anchor, res):
    rows = []
    if res.get('success'):
        t0 = res['t0']
        band_lo = focal_val - abs(focal_val) * t0
        band_hi = focal_val + abs(focal_val) * t0
        last_step = res['step_log'][-1] if res['step_log'] else {}
        last_tk = last_step.get('t_k', t0)
        total_adapts = sum(s['adapt_ct'] for s in res['step_log'])
        rows.append(['Focal Value',      f'{focal_val:.6g}',                   f'From {label.lower()}_pct at bar #{i_focal}'])
        rows.append(['Price Anchor',     f'{price_anchor:.6g}',                ''])
        rows.append(['Initial Tol (t0)', f'{t0*100:.1f}%',                     f'First t0 that yielded >={R2_Lo} matches'])
        rows.append(['Band [lo, hi]',    f'[{band_lo:.4f}, {band_hi:.4f}]',    f'focal +/- {t0*100:.1f}% of |focal|'])
        rows.append(['Initial Pool',     str(res['initial_matches']),           'Candidates inside band'])
        rows.append(['Steps Completed',  f'{R1}/{R1}',                          'All recursive steps passed'])
        rows.append(['Final Tol (t_k)',  f'{last_tk*100:.2f}%',                 f'{total_adapts} adaptive widen(s) across all steps'])
        rows.append(['Final Survivors',  str(len(res['survivors'])),             f'Target window: {R2_Lo}-{R2_Hi}'])
        rows.append(['Outcome',          'PASSED',                              'Ready for forward projection'])
    else:
        bf = res.get('best_failed')
        if bf:
            t0 = bf['t0']
            band_lo = focal_val - abs(focal_val) * t0
            band_hi = focal_val + abs(focal_val) * t0
            last_step = bf['step_log'][-1] if bf['step_log'] else {}
            last_tk = last_step.get('t_k', t0)
            deepest = bf['deepest_k']
            total_adapts = sum(s['adapt_ct'] for s in bf['step_log'])
            rows.append(['Focal Value',          f'{focal_val:.6g}',                    f'From {label.lower()}_pct at bar #{i_focal}'])
            rows.append(['Price Anchor',         f'{price_anchor:.6g}',                 ''])
            rows.append(['Best t0 Found',        f'{t0*100:.1f}%',                      f'Best of 1%-{INIT_TOL_MAX*100:.0f}% scan'])
            rows.append(['Band [lo, hi]',        f'[{band_lo:.4f}, {band_hi:.4f}]',     f'focal +/- {t0*100:.1f}% of |focal|'])
            rows.append(['Initial Pool',         str(bf['initial_matches']),             'Candidates inside band'])
            rows.append(['Steps Completed',      f'{deepest}/{R1}',                      f'Cascade broke at k={deepest}'])
            rows.append(['Tol at Failure (t_k)', f'{last_tk*100:.2f}%',                  f'{total_adapts} adaptive widen(s) before giving up'])
            rows.append(['Survivors at Failure',  str(bf['last_out']),                   f'Needed >={R2_Lo}, had {bf["last_out"]} -- shortfall of {R2_Lo - bf["last_out"]}'])
            rows.append(['Outcome',              'FAILED',                               f'No (t0, t1_start, t1_inc) combo survived {R1} steps'])
        else:
            rows.append(['Focal Value',  f'{focal_val:.6g}',   f'From {label.lower()}_pct at bar #{i_focal}'])
            rows.append(['Price Anchor', f'{price_anchor:.6g}', ''])
            rows.append(['Outcome',      'FAILED',             f'No t0 in 1%-{INIT_TOL_MAX*100:.0f}% produced >={R2_Lo} initial matches'])
    return rows
report_table('Step 2: High track readout',
             ['Parameter', 'Value', 'Commentary'],
             _track_readout('High', h_focal, P_h, h_result))
report_table('Step 2: Low track readout',
             ['Parameter', 'Value', 'Commentary'],
             _track_readout('Low', l_focal, P_L, l_result))
# --------------- STEP 2d: FOCAL TRACE (per-bar bands + tolerances) -------
def _fmt_tol(t):
    pct = t * 100
    return f'{pct:.0f}%' if abs(pct - round(pct)) < 0.05 else f'{pct:.1f}%'
def _bar_dt(arr_idx, If_arr, df_sorted, idx_to_pos):
    bar_num = int(If_arr[arr_idx])
    pos = idx_to_pos[bar_num]
    d, tm = '', ''
    if 'date_time_market' in df_sorted.columns:
        dt = str(df_sorted.loc[pos, 'date_time_market'])
        if len(dt) >= 10:
            d = dt[:10]
        if len(dt) > 11:
            tm = dt[11:19] if len(dt) >= 19 else dt[11:]
    return bar_num, d, tm
def _focal_trace(focal_series, res, If_arr, df_sorted, idx_to_pos, r1, r2lo):
    rows = []
    if res.get('success'):
        t0 = res['t0']
        step_log = res['step_log']
        init_pool = res['initial_matches']
    else:
        bf = res.get('best_failed')
        if bf is None:
            return rows
        t0 = bf['t0']
        step_log = bf['step_log']
        init_pool = bf['initial_matches']
    # T_0 row (initial filter on focal bar)
    bar_num, d, tm = _bar_dt(0, If_arr, df_sorted, idx_to_pos)
    fv = focal_series[0]
    bhi = fv + abs(fv) * t0
    blo = fv - abs(fv) * t0
    rows.append(['T_0', bar_num, d, tm,
                 f'{bhi:.4f}', f'{fv:.4f}', f'{blo:.4f}',
                 _fmt_tol(t0), '-', str(init_pool), 'Initial filter'])
    # Recursive steps T-1 .. T-R1
    for s in step_log:
        k = s['step']
        fk = focal_series[k + 1]
        tk = s['t_k']
        bhi = fk + abs(fk) * tk
        blo = fk - abs(fk) * tk
        bar_num, d, tm = _bar_dt(k + 1, If_arr, df_sorted, idx_to_pos)
        comment = ''
        if s['adapt_ct'] > 0:
            comment = f'Widened {s["adapt_ct"]}x'
        if k == step_log[-1]['step'] and s['out_count'] < r2lo:
            comment = f'BROKE: {s["out_count"]} < {r2lo}'
            if s['adapt_ct'] > 0:
                comment += f', widened {s["adapt_ct"]}x'
        rows.append([f'T-{k+1}', bar_num, d, tm,
                     f'{bhi:.4f}', f'{fk:.4f}', f'{blo:.4f}',
                     _fmt_tol(tk), str(s['in_count']), str(s['out_count']), comment])
    return rows
for lbl, res, fseries in [('High', h_result, Hf_arr), ('Low', l_result, Lf_arr)]:
    trace = _focal_trace(fseries, res, If_arr, df_sorted, idx_to_pos, R1, R2_Lo)
    if trace:
        report_table(f'Step 2d: {lbl} focal trace',
                     ['Step', '#', 'Date', 'Time', 'UB', 'Focal', 'LB', 'Tol', 'In', 'Out', 'Commentary'],
                     trace)
# --------------- STEP 2f: MATCHED INDICES PER STEP -----------------------
def _index_trace(res, label):
    """Build a table showing which # indices survived at every recursive step."""
    if res.get('success'):
        init_indices = res['initial_indices']
        step_log = res['step_log']
    else:
        bf = res.get('best_failed')
        if bf is None:
            return
        init_indices = bf.get('initial_indices')
        if init_indices is None:
            return
        step_log = bf['step_log']

    rows = []
    sorted_init = sorted(int(x) for x in init_indices)
    rows.append(['T_0 (initial)', len(sorted_init),
                 ', '.join(str(x) for x in sorted_init)])

    for s in step_log:
        surv = s['survivors']
        sorted_surv = sorted(int(x) for x in surv)
        rows.append([f'T-{s["step"]+1}', len(sorted_surv),
                     ', '.join(str(x) for x in sorted_surv)])

    report_table(f'Step 2f: {label} matched # indices per step',
                 ['Step', 'Count', 'Matched # indices'],
                 rows)

for lbl, res in [('High', h_result), ('Low', l_result)]:
    _index_trace(res, lbl)

# Search params for failed tracks
for lbl, res in [('High', h_result), ('Low', l_result)]:
    if res.get('success'):
        continue
    bf = res.get('best_failed')
    if bf is None:
        continue
    report_table(f'Step 2e: {lbl} best-attempt params',
                 ['Parameter', 'Value'],
                 [['t0',       f"{bf['t0']*100:.1f}%"],
                  ['t1_start', f"{bf['t1_start']:.2f}"],
                  ['t1_inc',   f"{bf['t1_inc']:.2f}"]])
# Abort if any track failed
h_ok = h_result.get('success', False)
l_ok = l_result.get('success', False)
if not h_ok or not l_ok:
    failed_tracks = []
    if not h_ok:
        bf_h = h_result.get('best_failed')
        detail_h = f"High (best: k={bf_h['deepest_k']}, {bf_h['last_out']} survivors)" if bf_h else "High (no viable t0)"
        failed_tracks.append(detail_h)
    if not l_ok:
        bf_l = l_result.get('best_failed')
        detail_l = f"Low (best: k={bf_l['deepest_k']}, {bf_l['last_out']} survivors)" if bf_l else "Low (no viable t0)"
        failed_tracks.append(detail_l)
    report_table('Outcome', ['Status', 'Detail'],
                 [['FAILED', f"Track(s) failed: {'; '.join(failed_tracks)}. No forecast written."]])
    return
# --------------------- STEP 4: SURVIVOR LISTS -----------------------------
def survivor_array(res):
    return res['survivors'][res['survivors'] != i_focal]
h_surv = survivor_array(h_result)
l_surv = survivor_array(l_result)
report_table('Step 4: Survivors (ex-focal)',
             ['Series', 'Count', 'Sample indices'],
             [['High', len(h_surv), str(h_surv[:8])],
              ['Low',  len(l_surv), str(l_surv[:8])]])
# ---------------- STEP 4b: FORWARD % MATRICES ----------------------------
def build_matrix(indices, H_arr, L_arr, itp):
    m_h = np.full((len(indices), N_FORWARD), np.nan)
    m_l = np.full_like(m_h, np.nan)
    for r, si in enumerate(indices):
        for c in range(N_FORWARD):
            fi = int(si) + c + 1
            if fi <= max_idx and itp[fi] >= 0:
                p = itp[fi]
                m_h[r, c] = H_arr[p]
                m_l[r, c] = L_arr[p]
    return m_h, m_l
__k2_progress__('build_matrix', f'h_surv={len(h_surv)} l_surv={len(l_surv)}')
Mh, _  = build_matrix(h_surv, H, L, idx_to_pos)
_,  ML = build_matrix(l_surv, H, L, idx_to_pos)
__k2_progress__('build_matrix_done', f'Mh={Mh.shape} ML={ML.shape}')
report_table('Step 4b: Forward % matrices',
             ['Matrix', 'Shape', 'Role'],
             [['Mh', f"{Mh.shape[0]}x{Mh.shape[1]}", 'High survivors -> high_pct'],
              ['ML', f"{ML.shape[0]}x{ML.shape[1]}", 'Low survivors -> low_pct']])
# --------------------- STEP 5: STATISTICS --------------------------------
Mh_avg = np.nanmean(Mh, axis=0); Mh_min = np.nanmin(Mh, axis=0); Mh_max = np.nanmax(Mh, axis=0)
ML_avg = np.nanmean(ML, axis=0); ML_min = np.nanmin(ML, axis=0); ML_max = np.nanmax(ML, axis=0)
report_table('Step 5: Aggregated forward %',
             ['Fwd', 'Mh_avg', 'Mh_min', 'Mh_max', 'ML_avg', 'ML_min', 'ML_max'],
             [[j+1, f"{Mh_avg[j]:.4g}", f"{Mh_min[j]:.4g}", f"{Mh_max[j]:.4g}",
                       f"{ML_avg[j]:.4g}", f"{ML_min[j]:.4g}", f"{ML_max[j]:.4g}"]
              for j in range(N_FORWARD)])
# ------------------ STEP 6: COMPOUND PRICES ------------------------------
def compound(base_price, pct_vec):
    price = base_price
    out   = []
    for p in pct_vec:
        price *= (1 + p/100.0)
        out.append(price)
    return np.array(out)
Mh_Avg_price = compound(P_h, Mh_avg)
Mh_Min_price = compound(P_h, Mh_min)
Mh_Max_price = compound(P_h, Mh_max)
ML_Avg_price = compound(P_L, ML_avg)
ML_Min_price = compound(P_L, ML_min)
ML_Max_price = compound(P_L, ML_max)
report_table('Step 6: Compounded price paths',
             ['Fwd', 'Mh_Avg$', 'Mh_Min$', 'Mh_Max$', 'ML_Avg$', 'ML_Min$', 'ML_Max$'],
             [[j+1, f"{Mh_Avg_price[j]:.4g}", f"{Mh_Min_price[j]:.4g}", f"{Mh_Max_price[j]:.4g}",
                       f"{ML_Avg_price[j]:.4g}", f"{ML_Min_price[j]:.4g}", f"{ML_Max_price[j]:.4g}"]
              for j in range(N_FORWARD)])
# ------------------- STEP 7: FORECAST OUTPUT -----------------------------
forecast_pairs = [
    ('Mh_Avg', Mh_Avg_price, P_h),
    ('Mh_Min', Mh_Min_price, P_h),
    ('Mh_Max', Mh_Max_price, P_h),
    ('ML_Avg', ML_Avg_price, P_L),
    ('ML_Min', ML_Min_price, P_L),
    ('ML_Max', ML_Max_price, P_L),
]
__k2_progress__('forecast_write', f'{len(forecast_pairs)} columns')
for name, vec, anchor in forecast_pairs:
    to_forecast(name, list(vec), anchor_price=anchor)
report_table('Forecast Columns Written',
             ['Column', 'Length'],
             [[n, len(v)] for n, v, _ in forecast_pairs])
__k2_progress__('done', 'forecast columns written')
