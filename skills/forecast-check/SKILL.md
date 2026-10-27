---
name: forecast-check
description: Check a time-series forecast before its accuracy is reported or it goes into planning — detects forecasts that lose to a naive or seasonal-naive baseline, test periods that overlap training, systematic bias, suspiciously perfect forecasts (leakage), duplicate or missing dates, and MAPE computed over zeros. Use whenever building, backtesting or comparing a forecast (demand, sales, traffic, revenue, capacity, inventory), reporting MAPE, WAPE, RMSE or MASE, or choosing a forecasting model.
---

# Forecast check

A forecast's accuracy only means something next to the simplest possible forecast. "Same as last week" is surprisingly hard to beat, and models that seem to beat it by a lot are often leaking the answer.

## Steps

1. **State the setup in one line**: the series and its frequency, the horizon (how far ahead each forecast is made), the forecast origin, and the training cut-off. Example: "Daily units per store, 8 weeks ahead, origin 2026-06-30, trained on data up to 2026-06-30."

2. **Build one file** with the full series: history rows (actual filled, forecast empty) followed by evaluation rows (actual and forecast filled). Include at least one full season of history before the first forecast date, and a series id column if there are many series.

3. **Run the checks** (requires pandas and numpy):

   ```bash
   python <skill-dir>/scripts/check_forecast.py --data <file> --date <col> --actual <col> --forecast <col> \
     [--series <id col>] [--train-end YYYY-MM-DD] [--season <periods>]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. The season is inferred from the date step (7 for daily, 52 for weekly, 12 for monthly) unless given. Exit code 1 means a FAIL. Add `--json` for structured output.

4. **Act on the findings**:

   | Finding | Meaning | What to do |
   | --- | --- | --- |
   | `no_better_than_naive` FAIL | "Repeat the last season" (or the last value) is at least as accurate | Do not report the model as an improvement; check seasonality, the horizon and features |
   | `train_overlap` FAIL | Some evaluation dates are inside the training period | Re-split by time; evaluate only after the cut-off |
   | `duplicate_dates` FAIL | A series has repeated timestamps | Deduplicate; find the double load or fan-out join |
   | `too_good` | Error far below what a one-step baseline can achieve | Audit features for leakage: anything computed from the target or recorded after the origin |
   | `bias` | Consistently over or under | Check level shifts, trend handling and the target transform (log without bias correction) |
   | `mape_undefined` | Zeros in the actuals | Report WAPE or MAE; consider intermittent-demand methods |
   | `short_evaluation` | Test window shorter than one season | Use a longer window or several rolling origins |
   | `gaps` | Missing periods | Fill the calendar so lags point at the right dates |
   | `barely_beats_naive` | Under 5% better than the baseline | Report the baseline beside the model and judge whether the gain is worth the complexity |

5. **Check what the script cannot see**:
   - Were features such as promotions, prices or weather available *as forecasts* at the origin, or were the actual values used?
   - Is the evaluation horizon the one the business uses? A 1-day-ahead score says little about an 8-week plan.
   - With several models or settings tried, use rolling-origin backtests rather than one split.
   - Holidays and one-off events in the test window can make any model look good or bad; name them.

## Output

A table with MAE, WAPE and bias for the model and both baselines, the ratio to the best baseline, and the evaluation window. Then one sentence on the biggest caveat. Never report a forecast's accuracy without its baseline, or while a `train_overlap` or `duplicate_dates` FAIL is open.
