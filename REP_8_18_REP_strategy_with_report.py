import numpy as np
import pandas as pd
from datetime import timedelta
# ============================================================
# 8-18 REP : Recursive Elasticity Projection Strategy
# ============================================================
# Adapted from 8-18 RPP. Instead of matching on high_pct / low_pct
# (two tracks), this matches on a SINGLE Elasticity track and
# projects Center / High / Low using signed and absolute elasticity.
# ============================================================

# ------------------------- TUNABLES --------------------------
R1 = 8          # Recursive depth (number of history steps)
R2_Lo = 7       # Min survivors required after a step
R2_Hi = 10      # Max survivors allowed after final step
N_FORWARD = 18  # Forward projection horizon
N_FOCAL = 24    # Length of focal series (bars before focal inclusive)
# Adaptive-search knobs
ADAPT_FACTOR = 1.25   # Multiplier when a recursion step falls short
MAX_T_ALLOWED = 2.0   # Absolute ceiling for any step's tolerance (200%)
INIT_TOL_MAX = 0.29   # Maximum initial tolerance scanned (29%)

# ----------------------------- REPORT HEADER -----------------------------
report_header('8-18 REP')
report_config([
    ('R1', R1, 'Recursive steps to apply'),
    ('R2_Lo', R2_Lo, 'Minimum survivors per step'),
    ('R2_Hi', R2_Hi, 'Maximum survivors at completion'),
    ('N_FORWARD', N_FORWARD, 'Forward elasticity projection steps'),
    ('N_FOCAL', N_FOCAL, 'Bars composing focal history'),
    ('ADAPT_FACTOR', ADAPT_FACTOR, 'Tolerance widening factor when survivor count < R2_Lo'),
    ('MAX_T_ALLOWED', f'{MAX_T_ALLOWED*100:.0f}%', 'Hard ceiling for any step tolerance'),
    ('INIT_TOL_MAX', f'{INIT_TOL_MAX*100:.0f}%', 'Upper bound scanned for initial tolerance'),
])

# --------------------------- PREP THE DATA -------------------------------
df_sorted = df.sort_values('#').reset_index(drop=True) if '#' in df.columns else df.sort_values('date_time_market').reset_index(drop=True)
if '#' not in df_sorted.columns:
    df_sorted['#'] = range(1, len(df_sorted) + 1)

I = df_sorted['#'].values.astype(int)
E = df_sorted['elasticity'].values.astype(float)
max_idx = int(I.max())

idx_to_pos = np.full(max_idx + 2, -1, dtype=int)
for pos in range(len(I)):
    idx_to_pos[int(I[pos])] = pos

i_focal = max_idx
pos_i   = idx_to_pos[i_focal]
P_close = float(df_sorted.loc[pos_i, 'close'])
e_focal = E[pos_i]

report_table(
    'Step 1: Focal anchor',
    ['Quantity', 'Value', 'Notes'],
    [
        ['Rows in frame', len(df_sorted), 'After sort'],
        ['i_focal (#)', i_focal, 'Current bar'],
        ['pos_i (iloc)', pos_i, 'Location inside dataframe'],
        ['elasticity focal', f"{e_focal:.6g}", 'Signed elasticity at focal bar'],
        ['P_close (anchor)', f"{P_close:.6g}", 'Close price anchor for projection'],
    ],
)

# --- Build focal history array (latest -> older)
If_arr = np.arange(i_focal, i_focal - N_FOCAL, -1)
Ef_arr = np.array([E[idx_to_pos[idx]] for idx in If_arr])

report_table(
    'Step 1b: Focal history sample',
    ['# index', 'elasticity'],
    [[int(If_arr[j]), f"{Ef_arr[j]:.6g}"] for j in range(len(If_arr))]
)

# -------------------- ADAPTIVE RECURSIVE MATCH ---------------------------

def run_recursive_match(focal_val, focal_series, I_arr, E_arr, itp,
                        r1, r2lo, r2hi, adapt_factor, max_t_allowed):
    """Single-track recursive match on Elasticity."""
    mx = int(I_arr.max())
    best_failed = None
    max_loop = int(INIT_TOL_MAX * 100)

    for tp in range(1, max_loop + 1):
        t0 = tp / 100.0
        lo0 = focal_val - abs(focal_val) * t0
        hi0 = focal_val + abs(focal_val) * t0
        init_idx = I_arr[(E_arr >= lo0) & (E_arr <= hi0)]
        if len(init_idx) < r2lo:
            continue

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
                        sv = E_arr[ps]
                        mask = (sv >= lo_k) & (sv <= hi_k)
                        in_cnt = len(sh)
                        survivors = orig[mask]
                        out_cnt = len(survivors)

                        if out_cnt >= r2lo or tk >= max_t_allowed:
                            break
                        tk *= adapt_factor
                        adapt_ct += 1

                    step_log.append({
                        'step': k, 't_k': tk, 'adapt_ct': adapt_ct,
                        'in_count': in_cnt, 'out_count': out_cnt,
                        'survivors': survivors.copy()
                    })

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


e_result = run_recursive_match(e_focal, Ef_arr, I, E, idx_to_pos,
                               R1, R2_Lo, R2_Hi, ADAPT_FACTOR, MAX_T_ALLOWED)

# --------------- STEP 2: TRACK DIAGNOSTICS -----------------------

def _track_readout(label, focal_val, price_anchor, res):
    rows = []
    if res.get('success'):
        t0 = res['t0']
        band_lo = focal_val - abs(focal_val) * t0
        band_hi = focal_val + abs(focal_val) * t0
        last_step = res['step_log'][-1] if res['step_log'] else {}
        last_tk = last_step.get('t_k', t0)
        total_adapts = sum(s['adapt_ct'] for s in res['step_log'])
        rows.append(['Focal Value',      f'{focal_val:.6g}',                   f'From {label.lower()} at bar #{i_focal}'])
        rows.append(['Price Anchor',     f'{price_anchor:.6g}',                'Close price for projection'])
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
            rows.append(['Focal Value',          f'{focal_val:.6g}',                    f'From {label.lower()} at bar #{i_focal}'])
            rows.append(['Price Anchor',         f'{price_anchor:.6g}',                 'Close price for projection'])
            rows.append(['Best t0 Found',        f'{t0*100:.1f}%',                      f'Best of 1%-{INIT_TOL_MAX*100:.0f}% scan'])
            rows.append(['Band [lo, hi]',        f'[{band_lo:.4f}, {band_hi:.4f}]',     f'focal +/- {t0*100:.1f}% of |focal|'])
            rows.append(['Initial Pool',         str(bf['initial_matches']),             'Candidates inside band'])
            rows.append(['Steps Completed',      f'{deepest}/{R1}',                      f'Cascade broke at k={deepest}'])
            rows.append(['Tol at Failure (t_k)', f'{last_tk*100:.2f}%',                  f'{total_adapts} adaptive widen(s) before giving up'])
            rows.append(['Survivors at Failure',  str(bf['last_out']),                   f'Needed >={R2_Lo}, had {bf["last_out"]} -- shortfall of {R2_Lo - bf["last_out"]}'])
            rows.append(['Outcome',              'FAILED',                               f'No (t0, t1_start, t1_inc) combo survived {R1} steps'])
        else:
            rows.append(['Focal Value',  f'{focal_val:.6g}',   f'From {label.lower()} at bar #{i_focal}'])
            rows.append(['Price Anchor', f'{price_anchor:.6g}', 'Close price for projection'])
            rows.append(['Outcome',      'FAILED',             f'No t0 in 1%-{INIT_TOL_MAX*100:.0f}% produced >={R2_Lo} initial matches'])
    return rows

report_table('Step 2: Elasticity track readout',
             ['Parameter', 'Value', 'Commentary'],
             _track_readout('Elasticity', e_focal, P_close, e_result))

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

    bar_num, d, tm = _bar_dt(0, If_arr, df_sorted, idx_to_pos)
    fv = focal_series[0]
    bhi = fv + abs(fv) * t0
    blo = fv - abs(fv) * t0
    rows.append(['T_0', bar_num, d, tm,
                 f'{bhi:.4f}', f'{fv:.4f}', f'{blo:.4f}',
                 _fmt_tol(t0), '-', str(init_pool), 'Initial filter'])

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

trace = _focal_trace(Ef_arr, e_result, If_arr, df_sorted, idx_to_pos, R1, R2_Lo)
if trace:
    report_table('Step 2d: Elasticity focal trace',
                 ['Step', '#', 'Date', 'Time', 'UB', 'Focal', 'LB', 'Tol', 'In', 'Out', 'Commentary'],
                 trace)

# --------------- STEP 2f: MATCHED INDICES PER STEP -----------------------

def _index_trace(res, label):
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

_index_trace(e_result, 'Elasticity')

# --------------- STEP 2e: BEST-ATTEMPT PARAMS (failed track) -------------
if not e_result.get('success'):
    bf = e_result.get('best_failed')
    if bf:
        report_table('Step 2e: Elasticity best-attempt params',
                     ['Parameter', 'Value'],
                     [['t0',       f"{bf['t0']*100:.1f}%"],
                      ['t1_start', f"{bf['t1_start']:.2f}"],
                      ['t1_inc',   f"{bf['t1_inc']:.2f}"]])

# --------------- ABORT IF TRACK FAILED -----------------------------------
if not e_result.get('success'):
    bf = e_result.get('best_failed')
    if bf:
        detail = f"Elasticity (best: k={bf['deepest_k']}, {bf['last_out']} survivors)"
    else:
        detail = "Elasticity (no viable t0)"
    report_table('Outcome', ['Status', 'Detail'],
                 [['FAILED', f"Track failed: {detail}. No forecast written."]])
    to_forecast('REP_Center', [])
    to_forecast('REP_High', [])
    to_forecast('REP_Low', [])
    report_table('Forecast Columns Written',
                 ['Column', 'Length'],
                 [['REP_Center', 0], ['REP_High', 0], ['REP_Low', 0]])
    return

# --------------------- STEP 3: SEARCH PARAMS SUMMARY --------------------
report_table('Step 3: Winning search parameters',
             ['Parameter', 'Value', 'Commentary'],
             [['t0',       f"{e_result['t0']*100:.1f}%",       'Initial tolerance that seeded the cascade'],
              ['t1_start', f"{e_result['t1_start']:.2f}",      'Starting recursive tolerance'],
              ['t1_inc',   f"{e_result['t1_inc']:.2f}",        'Tolerance increment per recursive step'],
              ['Steps',    f'{R1}',                             f'All {R1} recursive steps completed'],
              ['Survivors', str(len(e_result['survivors'])),    f'Inside target window {R2_Lo}-{R2_Hi}']])

# --------------------- STEP 4: SURVIVOR LIST -----------------------------
e_surv = e_result['survivors']
e_surv = e_surv[e_surv != i_focal]

report_table('Step 4: Survivors (ex-focal)',
             ['Track', 'Count', 'Sample indices'],
             [['Elasticity', len(e_surv), str(e_surv[:8])]])

# ---------------- STEP 4b: FORWARD ELASTICITY MATRIX ---------------------
def build_matrix(indices, E_arr, itp):
    m_e = np.full((len(indices), N_FORWARD), np.nan)
    for r, si in enumerate(indices):
        for c in range(N_FORWARD):
            fi = int(si) + c + 1
            if fi <= max_idx and itp[fi] >= 0:
                p = itp[fi]
                m_e[r, c] = E_arr[p]
    return m_e

ME = build_matrix(e_surv, E, idx_to_pos)

report_table('Step 4b: Forward elasticity matrix',
             ['Matrix', 'Shape', 'Role'],
             [['ME', f"{ME.shape[0]}x{ME.shape[1]}", 'Elasticity survivors -> forward elasticity values']])

# --------------------- STEP 4c: RAW MATRIX SAMPLE -----------------------
sample_rows = min(ME.shape[0], 5)
sample_cols = min(ME.shape[1], 8)
matrix_sample = []
for r in range(sample_rows):
    row_data = [int(e_surv[r])]
    for c in range(sample_cols):
        row_data.append(f"{ME[r, c]:.4g}" if not np.isnan(ME[r, c]) else 'NaN')
    matrix_sample.append(row_data)

report_table('Step 4c: Forward elasticity matrix sample',
             ['Surv #'] + [f'F+{c+1}' for c in range(sample_cols)],
             matrix_sample)

# --------------------- STEP 5: STATISTICS --------------------------------
ME_avg = np.nanmean(ME, axis=0)
ME_min = np.nanmin(ME, axis=0)
ME_max = np.nanmax(ME, axis=0)
ME_abs_avg = np.nanmean(np.abs(ME), axis=0)

report_table('Step 5: Aggregated forward elasticity',
             ['Fwd', 'Avg (signed)', 'Avg |E|', 'Min', 'Max'],
             [[j+1, f"{ME_avg[j]:.4g}", f"{ME_abs_avg[j]:.4g}",
               f"{ME_min[j]:.4g}", f"{ME_max[j]:.4g}"]
              for j in range(N_FORWARD)])

# ------------------ STEP 6: COMPOUND PRICES ------------------------------
def compound(base_price, pct_vec):
    price = base_price
    out = []
    for p in pct_vec:
        price *= (1 + p / 100.0)
        out.append(price)
    return np.array(out)

center_prices = compound(P_close, ME_avg)
high_prices = center_prices * (1 + ME_abs_avg / 200.0)
low_prices  = center_prices * (1 - ME_abs_avg / 200.0)

report_table('Step 6: Compounded price paths',
             ['Fwd', 'Center$', 'High$', 'Low$', 'Spread$'],
             [[j+1, f"{center_prices[j]:.4f}", f"{high_prices[j]:.4f}",
               f"{low_prices[j]:.4f}", f"{high_prices[j] - low_prices[j]:.4f}"]
              for j in range(N_FORWARD)])

# ------------------- STEP 7: FORECAST OUTPUT -----------------------------
forecast_pairs = [
    ('REP_Center', list(center_prices), P_close),
    ('REP_High',   list(high_prices),   P_close),
    ('REP_Low',    list(low_prices),    P_close),
]
for name, vals, anchor in forecast_pairs:
    to_forecast(name, vals, anchor_price=anchor)

report_table('Forecast Columns Written',
             ['Column', 'Length', 'Anchor'],
             [[name, len(vals), f"{anchor:.4f}"] for name, vals, anchor in forecast_pairs])

# ------------------- OUTCOME SUMMARY -------------------------------------
report_table('Outcome', ['Status', 'Detail'],
             [['PASSED', f'{len(forecast_pairs)} forecast columns written from {len(e_surv)} survivors, '
                         f'{N_FORWARD} forward steps, anchored at Close={P_close:.4f}']])
