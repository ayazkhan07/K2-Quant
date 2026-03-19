STRATEGY: 8-8 Recursive Price Projection (RPP)

You are implementing a quantitative pattern-matching price projection strategy in Python. The strategy is called the 8-8 Recursive Price Projection (RPP). You have access to run_sql and run_python tools, a persistent Python environment where df is the full model data table as a pandas DataFrame, and to_forecast() to write final projections to Tab 2. All intermediate results must go to a newly created sheet via to_working() at each step as described below.

OVERVIEW

The strategy finds 8 historical 8-day windows per OHLC column that most closely match the current market's recent behaviour, constructs a 17x8 matrix per column, then generates 12 forward price paths (average, min, max per column) across 8 future trading days, compounding from today's actual OHLC prices.

THE DATA

The DataFrame df contains daily OHLCV data. The relevant columns for this strategy are:

open_pct — daily open percentage change (open relative to previous open)
high_pct — daily high percentage change (high relative to previous high)
low_pct — daily low percentage change (low relative to previous low)
close_pct — daily close percentage change (close relative to previous close)
open — actual open price
high — actual high price
low — actual low price
close — actual close price

Rows are ordered ascending by date. The last row is "today". Use .iloc for all positional access throughout — do not use df.index for positional lookups.

SHEET SETUP

Before writing any intermediate data, create a new sheet for this run's working output. All to_working() calls in the steps below write to this newly created sheet. This ensures no stale data from previous runs contaminates the current results.

STEP 1 — ESTABLISH BASELINE VECTORS

For each of the 4 percentage columns extract the baseline vector of length 8 from the most recent 8 rows of df, excluding the very last row (which is today's anchor). So the baseline covers iloc[-9:-1] — the 8 completed trading days immediately prior to today.

OP_baseline = open_pct values at iloc -9 through -2
HP_baseline = high_pct values at the same positions
LP_baseline = low_pct values at the same positions
CP_baseline = close_pct values at the same positions

Publish to the working sheet immediately after:
Write 4 columns named RPP_Base_Open, RPP_Base_High, RPP_Base_Low, RPP_Base_Close — each containing their 8 baseline values. This confirms the baseline vectors are correct before any further computation proceeds. Verify each to_working() call before moving on.

STEP 2 — TOLERANCE PRE-FILTER (per column independently)

For each column, take baseline[0] (the first element of the baseline vector, i.e. the value at iloc[-9]) as the anchor value. Scan all prior candidate positions i where the value at the START of the candidate context window falls within the relative tolerance of this anchor:

tolerance = abs(anchor_value) * 0.025
abs(df[col].iloc[i - 8] - anchor_value) <= tolerance

This is a relative 2.5% tolerance — the band is 2.5% of the anchor value's magnitude, with no minimum floor. The filter checks df[col].iloc[i - 8] (the first element of the candidate's context window) against baseline[0], because in Step 3 the candidate vector is df[col].iloc[i-8:i] and its first element must align with baseline[0].

Apply independently for each of the 4 columns. Each column produces its own shortlist of eligible candidate row positions.

Boundary rule: a candidate at position i is only valid if:

i >= 8 — enough history to form the context window iloc[i-8:i]
i <= len(df) - 18 — the candidate's entire 17-row window (8 context + 1 pivot + 8 forward) does not overlap with the baseline window or today

If any column produces fewer than 8 candidates after this filter, STOP execution and print a message: "[column] filter returned only [N] candidates. Increase tolerance and re-run." Do not proceed with the strategy.

Publish to the working sheet immediately after:
Write 4 single-value columns to the working sheet:

RPP_Anchor_Open, RPP_Anchor_High, RPP_Anchor_Low, RPP_Anchor_Close — the anchor value used for each filter

Print to console how many candidates survived per column so the user can assess filter tightness. Do not write the full candidate lists to the working sheet.

Verify all to_working() calls before proceeding.

STEP 3 — VECTOR SCORING (per column independently)

For each shortlisted candidate position i in a given column's shortlist, construct a candidate vector: df[col].iloc[i-8:i] (8 values ending at position i-1). This candidate vector aligns element-for-element with the baseline vector — candidate[0] corresponds to baseline[0], candidate[1] to baseline[1], and so on.

Compute the score as the sum of absolute differences (L1 distance) between the candidate vector and the baseline vector across all 8 positions.

For each column, rank all shortlisted candidates by score ascending and select the 8 lowest scoring candidate positions.

You will end up with:

open_winners — list of 8 row positions
high_winners — list of 8 row positions
low_winners — list of 8 row positions
close_winners — list of 8 row positions

Publish to the working sheet immediately after:
Write 8 columns total:

RPP_Win_Open — the 8 winning row positions for open_pct
RPP_Score_Open — the corresponding 8 scores
RPP_Win_High — the 8 winning row positions for high_pct
RPP_Score_High — the corresponding 8 scores
RPP_Win_Low — the 8 winning row positions for low_pct
RPP_Score_Low — the corresponding 8 scores
RPP_Win_Close — the 8 winning row positions for close_pct
RPP_Score_Close — the corresponding 8 scores

Also write 4 columns showing the actual dates of the winning pivot points for human readability:

RPP_Dates_Open — dates corresponding to the 8 winning open_pct positions
RPP_Dates_High — dates corresponding to the 8 winning high_pct positions
RPP_Dates_Low — dates corresponding to the 8 winning low_pct positions
RPP_Dates_Close — dates corresponding to the 8 winning close_pct positions

Verify all to_working() calls before proceeding.

STEP 4 — BUILD THE 17x8 MATRICES (per column)

For each column and its 8 winning positions, construct a 17-row by 8-column matrix where each column corresponds to one winning analogue:

Rows 0–7: df[col].iloc[i-8:i] — the 8-day historical context window leading up to the pivot
Row 8: df[col].iloc[i] — the pivot point itself, the "today equivalent" in that analogue
Rows 9–16: df[col].iloc[i+1:i+9] — the 8 days that actually followed in history

Build 4 matrices as numpy arrays shape (17, 8):

matrix_open, matrix_high, matrix_low, matrix_close

Publish to the working sheet immediately after — write only rows 9–16 (the forward portion):
For open_pct: columns RPP_O_V1 through RPP_O_V8
For high_pct: columns RPP_H_V1 through RPP_H_V8
For low_pct: columns RPP_L_V1 through RPP_L_V8
For close_pct: columns RPP_C_V1 through RPP_C_V8

That is 32 columns total, each containing 8 values — the raw historical forward percentage paths of every winning analogue before any aggregation. Verify all to_working() calls before proceeding.

STEP 5 — COMPUTE PROJECTION STATISTICS

For each of the 4 matrices take only rows 9–16 (shape 8x8). Across the 8 columns compute per row:

avg — mean of the 8 values
min — minimum of the 8 values
max — maximum of the 8 values

Producing 12 series of length 8:

open_avg, open_min, open_max
high_avg, high_min, high_max
low_avg, low_min, low_max
close_avg, close_min, close_max

Publish to the working sheet immediately after:
Write 12 columns:

RPP_Open_Avg, RPP_Open_Min, RPP_Open_Max
RPP_High_Avg, RPP_High_Min, RPP_High_Max
RPP_Low_Avg, RPP_Low_Min, RPP_Low_Max
RPP_Close_Avg, RPP_Close_Min, RPP_Close_Max

Each column contains 8 values — the aggregated percentage change inputs for the compounding step. Verify all to_working() calls before proceeding.

STEP 6 — COMPOUND PRICE PROJECTION

Take today's actual prices from the last row of df:

today_open = df['open'].iloc[-1]
today_high = df['high'].iloc[-1]
today_low = df['low'].iloc[-1]
today_close = df['close'].iloc[-1]

Each OHLC column's percentage series must compound from its own respective price. The percentage values are expressed as percentages (e.g. 1.5 means 1.5%) so divide by 100 when applying. The compounding pattern for each series is:

price_day_1 = base_price * (1 + series[0] / 100)
price_day_2 = price_day_1 * (1 + series[1] / 100)
...
price_day_8 = price_day_7 * (1 + series[7] / 100)

Apply with the following explicit base price mapping — each series compounds from the price of its own OHLC column:

open_avg, open_min, open_max → compound from today_open (projected open prices)
high_avg, high_min, high_max → compound from today_high (projected high prices)
low_avg, low_min, low_max → compound from today_low (projected low prices)
close_avg, close_min, close_max → compound from today_close (projected close prices)

This produces 12 lists of 8 projected prices each. Do NOT use a single base price for all columns — each column must use its own.

Publish to the working sheet immediately after:
Write 12 columns of compounded dollar prices:

RPP_Open_Avg_P, RPP_Open_Min_P, RPP_Open_Max_P
RPP_High_Avg_P, RPP_High_Min_P, RPP_High_Max_P
RPP_Low_Avg_P, RPP_Low_Min_P, RPP_Low_Max_P
RPP_Close_Avg_P, RPP_Close_Min_P, RPP_Close_Max_P

Each column contains 8 values representing the actual projected dollar prices for each future trading day. These are the final price paths before being written to Tab 2. Verify all to_working() calls before proceeding.

STEP 7 — WRITE TO FORECAST TAB

Use to_forecast(column_name, values) to write each of the 12 projected price columns to Tab 2. Cast all values to plain Python float before passing. Each call writes one named column:

to_forecast("RPP_Open_Avg_P", [float(v) for v in open_avg_prices])
to_forecast("RPP_Open_Min_P", [float(v) for v in open_min_prices])
to_forecast("RPP_Open_Max_P", [float(v) for v in open_max_prices])
to_forecast("RPP_High_Avg_P", [float(v) for v in high_avg_prices])
to_forecast("RPP_High_Min_P", [float(v) for v in high_min_prices])
to_forecast("RPP_High_Max_P", [float(v) for v in high_max_prices])
to_forecast("RPP_Low_Avg_P", [float(v) for v in low_avg_prices])
to_forecast("RPP_Low_Min_P", [float(v) for v in low_min_prices])
to_forecast("RPP_Low_Max_P", [float(v) for v in low_max_prices])
to_forecast("RPP_Close_Avg_P", [float(v) for v in close_avg_prices])
to_forecast("RPP_Close_Min_P", [float(v) for v in close_min_prices])
to_forecast("RPP_Close_Max_P", [float(v) for v in close_max_prices])

Each list must contain exactly 8 plain Python float values. The columns will auto-appear in the Forecast Data tab and can be individually toggled on/off on the chart by clicking the column header.

STEP 8 — SUMMARY REPORT

After all Tab 2 writes are confirmed, print a clean natural language summary including:

How many candidates passed the tolerance filter per column
The match quality scores of the 8 winning analogues per column
The dates of each winning analogue's pivot point
Today's actual OHLC prices used as the compounding base
The 8 projected average prices for each OHLC column day by day
A one-line interpretation such as: "Based on 8 historical analogues, the average projection suggests Open will move from $X to $Y over the next 8 trading days, with a range of $MinY to $MaxY"

STEP 9 — SAVE AS STRATEGY

After the summary report is printed, package the entire RPP pipeline into a single self-contained Python script and save it using the save_strategy tool so the user can toggle it on/off from the Strategies panel.

CRITICAL — the saved strategy runs in the DPE execution context, which has different column names than the interactive environment:

Interactive (run_python): open_pct, high_pct, low_pct, close_pct
DPE (saved strategy): Open_%, High_%, Low_%, Close_%

The DPE DataFrame also contains: Date, Time, open, high, low, close, volume, vwap, date_time_market, Elasticity, Close-Open_%

The saved script must:
- Use the DPE column names (Open_%, High_%, Low_%, Close_%) for percentage columns
- Use lowercase (open, high, low, close) for price columns (same in both contexts)
- Be fully self-contained — import numpy, define all variables, run Steps 1 through 7 in sequence
- Write final projections via to_forecast(column_name, values) — the only output mechanism available in DPE
- NOT include any to_working() calls (the DPE context does not have access to the working sheet)
- NOT include any print statements for the summary report (not visible in DPE)

Call save_strategy with:
- name: "RPP 8-8"
- code: the complete self-contained script
- description: "8-8 Recursive Price Projection — finds 8 best-matching historical 8-day windows per OHLC column and projects 8 days forward"

After saving, confirm to the user that the strategy was saved and is available in the Strategies panel.

IMPORTANT CONSTRAINTS

Never modify the main df table directly — all intermediate work stays in memory or goes to the working sheet
Always verify working sheet writes after every to_working() call before proceeding to the next step
Use numpy for all matrix operations
Cast all forecast values to plain Python float before passing to to_forecast()
The Python environment is persistent — variables computed in earlier steps are available in later steps without recomputation
If any column produces fewer than 8 candidates after the tolerance filter, stop execution and prompt the user to increase the tolerance level
