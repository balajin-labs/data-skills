#!/usr/bin/env python3
"""Check a time-series forecast evaluation: worse than a naive baseline, test period overlapping training,
systematic bias, suspiciously perfect forecasts (leakage), duplicate or missing dates, and MAPE on zeros.

Usage examples
--------------
    python check_forecast.py --data sales.csv --date date --actual units --forecast forecast
    python check_forecast.py --data sales.csv --date date --actual units --forecast forecast \
        --series store_id --train-end 2026-06-30 --season 7

The file holds the full series: history rows (actual filled, forecast empty) followed by the evaluation
rows (actual and forecast both filled). History is needed to build the naive baselines; include at least
one full season of it. Exit code: 1 if any FAIL finding, 0 otherwise, 2 on bad input. Add --json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: file not found: {path}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, sep="\t" if p.suffix.lower() in {".tsv", ".tab"} else ",")


def finding(check, severity, message, columns=None, value=None, examples=None):
    out = {"check": check, "severity": severity, "message": message, "columns": columns or [], "value": value}
    if examples:
        out["examples"] = examples
    return out


def r(v, d=4):
    return None if v is None or v != v else round(float(v), d)


def infer_season(step_days: float) -> int:
    """Typical seasonal period for a series sampled every `step_days` days."""
    if step_days < 0.9:
        return 24          # hourly -> daily cycle
    if step_days < 1.5:
        return 7           # daily -> weekly cycle
    if 6 <= step_days <= 8:
        return 52          # weekly -> yearly cycle
    if 27 <= step_days <= 32:
        return 12          # monthly -> yearly cycle
    if 85 <= step_days <= 95:
        return 4           # quarterly -> yearly cycle
    return 1


def check(df, date, actual, forecast, series, train_end, season):
    f, stats = [], {}
    df = df.copy()
    df[date] = pd.to_datetime(df[date], errors="coerce")
    if df[date].isna().any():
        f.append(finding("forecast.bad_dates", "WARN", f"{int(df[date].isna().sum())} rows have unparseable dates "
                         "and are ignored.", [date], int(df[date].isna().sum())))
        df = df[df[date].notna()]
    df[actual] = pd.to_numeric(df[actual], errors="coerce")
    df[forecast] = pd.to_numeric(df[forecast], errors="coerce")
    sid = df[series].astype(str) if series else pd.Series("all", index=df.index)
    df["_sid"] = sid

    # Duplicate timestamps within a series
    dup = df.duplicated(["_sid", date], keep=False)
    if dup.any():
        ex = df.loc[dup, ["_sid", date]].drop_duplicates().head(5)
        f.append(finding(
            "forecast.duplicate_dates", "FAIL",
            f"{int(df.duplicated(['_sid', date]).sum())} rows repeat a date already present in the same series. "
            "Lags, rolling features and error metrics are all wrong on duplicated timestamps (often a double "
            "load or a join that fanned out). Deduplicate before training or scoring.",
            [date] + ([series] if series else []), int(dup.sum()),
            [f"{s} {d:%Y-%m-%d}" for s, d in ex.itertuples(index=False)]))
        df = df.drop_duplicates(["_sid", date], keep="last")

    train_end_ts = pd.Timestamp(train_end) if train_end else None
    err, ae, ae_naive, ae_snaive, ae_roll, acts = [], [], [], [], [], []
    overlap_rows = gap_series = missing_periods = short_series = no_hist = 0
    seasons, eval_lengths = [], []

    for _, g in df.sort_values(date).groupby("_sid", sort=False):
        steps = g[date].diff().dt.total_seconds().dropna() / 86400
        step = float(steps.median()) if len(steps) else 1.0
        m = season or infer_season(step)
        seasons.append(m)
        if len(steps) and step > 0:
            gaps = (steps / step).round() - 1
            if (gaps > 0).any():
                gap_series += 1
                missing_periods += int(gaps[gaps > 0].sum())

        ev_mask = g[forecast].notna() & g[actual].notna()
        if not ev_mask.any():
            continue
        first_eval = g.loc[ev_mask, date].min()
        hist = g[(g[date] < first_eval) & g[actual].notna()][actual].to_numpy(float)
        ev = g[ev_mask]
        eval_lengths.append(len(ev))
        if train_end_ts is not None:
            overlap_rows += int((ev[date] <= train_end_ts).sum())

        a = ev[actual].to_numpy(float)
        fc = ev[forecast].to_numpy(float)
        err.append(fc - a)
        ae.append(np.abs(fc - a))
        acts.append(a)

        # Baselines made at the forecast origin: last value, and the last full season repeated
        if len(hist) == 0:
            no_hist += 1
            ae_naive.append(np.full(len(a), np.nan))
            ae_snaive.append(np.full(len(a), np.nan))
        else:
            ae_naive.append(np.abs(hist[-1] - a))
            if len(hist) >= m and m > 1:
                snaive = np.array([hist[-m + (i % m)] for i in range(len(a))])
            else:
                short_series += int(m > 1)
                snaive = np.full(len(a), hist[-1])
            ae_snaive.append(np.abs(snaive - a))

        # One-step seasonal naive (uses actuals up to the day before): a floor that honest models rarely beat by much
        full = g[actual].to_numpy(float)
        pos = np.flatnonzero(ev_mask.to_numpy())
        lag = max(m, 1)
        prev = np.array([full[p - lag] if p - lag >= 0 else np.nan for p in pos])
        ae_roll.append(np.abs(prev - a))

    if not ae:
        raise SystemExit("error: no rows have both an actual and a forecast")
    err, ae, acts = np.concatenate(err), np.concatenate(ae), np.concatenate(acts)
    ae_naive, ae_snaive, ae_roll = np.concatenate(ae_naive), np.concatenate(ae_snaive), np.concatenate(ae_roll)

    mae = float(ae.mean())
    scale = float(np.abs(acts).mean()) or 1.0
    wape = float(ae.sum() / np.abs(acts).sum()) if np.abs(acts).sum() else float("nan")
    stats.update(series=int(df["_sid"].nunique()), eval_rows=int(len(ae)), season=int(np.median(seasons)),
                 mae=r(mae), wape=r(wape))

    # Train/test overlap
    if overlap_rows:
        f.append(finding(
            "forecast.train_overlap", "FAIL",
            f"{overlap_rows:,} of {len(ae):,} evaluation rows ({overlap_rows / len(ae):.0%}) are dated on or before "
            f"the training end ({train_end_ts:%Y-%m-%d}). The model is being scored on data it was trained on. "
            "Evaluate only on dates after the training cut-off.", [date], overlap_rows))

    # Naive baselines
    valid = ~np.isnan(ae_naive)
    if valid.any():
        mae_naive = float(ae_naive[valid].mean())
        mae_snaive = float(ae_snaive[valid].mean())
        mae_model = float(ae[valid].mean())
        best_name, best = min((("seasonal naive", mae_snaive), ("naive (last value)", mae_naive)), key=lambda x: x[1])
        ratio = mae_model / best if best else float("nan")
        stats.update(mae_naive=r(mae_naive), mae_seasonal_naive=r(mae_snaive), ratio_to_best_baseline=r(ratio))
        if ratio >= 1.0:
            f.append(finding(
                "forecast.no_better_than_naive", "FAIL",
                f"The forecast's MAE ({mae_model:,.4g}) is no better than a {best_name} forecast made at the same "
                f"origin ({best:,.4g}; ratio {ratio:.2f}). A model that loses to 'repeat the last "
                f"{'season' if 'seasonal' in best_name else 'value'}' adds nothing. Check seasonality, the horizon "
                "and the features before reporting it.", [forecast], r(ratio)))
        elif ratio >= 0.95:
            f.append(finding(
                "forecast.barely_beats_naive", "WARN",
                f"The forecast is only {1 - ratio:.0%} better than a {best_name} baseline ({mae_model:,.4g} vs "
                f"{best:,.4g}). Report the baseline next to the model.", [forecast], r(ratio)))
    if no_hist:
        f.append(finding(
            "forecast.no_history", "WARN",
            f"{no_hist} series have no history rows before the evaluation period, so no baseline could be built "
            "for them. Include at least one season of actuals before the first forecast date.", [actual], no_hist))
    if short_series:
        f.append(finding(
            "forecast.short_history", "WARN",
            f"{short_series} series have less than one season of history; the seasonal baseline fell back to the "
            "last value for them.", [actual], short_series))

    # Too good to be true
    roll_ok = ~np.isnan(ae_roll)
    if roll_ok.sum() >= 10:
        mae_roll = float(ae_roll[roll_ok].mean())
        stats["mae_one_step_seasonal_naive"] = r(mae_roll)
        if mae_roll and float(ae[roll_ok].mean()) < 0.2 * mae_roll:
            f.append(finding(
                "forecast.too_good", "WARN",
                f"The forecast error ({float(ae[roll_ok].mean()):,.4g}) is under a fifth of a one-step seasonal naive "
                f"forecast's ({mae_roll:,.4g}), which already sees yesterday's actuals. Forecasts this close usually "
                "mean the actual (or something computed from it) leaked into the features. Check every feature is "
                "known at the forecast origin.", [forecast], r(float(ae[roll_ok].mean()) / mae_roll)))

    # Bias
    me = float(err.mean())
    sd = float(err.std(ddof=1)) if len(err) > 1 else 0.0
    t = me / (sd / math.sqrt(len(err))) if sd else 0.0
    stats["bias_pct"] = r(me / scale)
    if abs(me / scale) > 0.05 and abs(t) > 3:
        f.append(finding(
            "forecast.bias", "WARN",
            f"The forecast is systematically {'high' if me > 0 else 'low'} by {abs(me):,.4g} on average "
            f"({abs(me / scale):.1%} of the mean actual). Accuracy metrics hide this; planners will over- or "
            "under-stock every period. Check the trend, recent level shifts and how the target was transformed.",
            [forecast], r(me / scale)))

    # Zero actuals make MAPE undefined
    zeros = int((np.abs(acts) < 1e-9).sum())
    if zeros:
        f.append(finding(
            "forecast.mape_undefined", "WARN",
            f"{zeros} evaluation actuals ({zeros / len(acts):.0%}) are zero, so MAPE divides by zero or explodes. "
            f"Report WAPE ({wape:.1%}) or MAE instead, and use an intermittent-demand method if zeros are common.",
            [actual], zeros))

    # Evaluation window shorter than a season
    if eval_lengths and stats["season"] > 1 and np.median(eval_lengths) < stats["season"]:
        f.append(finding(
            "forecast.short_evaluation", "WARN",
            f"The evaluation covers {int(np.median(eval_lengths))} periods per series, less than one season "
            f"({stats['season']}). Results may reflect which days happen to be in the window. Use a longer test "
            "period or several rolling origins.", [date], int(np.median(eval_lengths))))

    if gap_series:
        f.append(finding(
            "forecast.gaps", "WARN",
            f"{gap_series} series have missing periods ({missing_periods} in total). Lag and seasonal features "
            "computed by row position point at the wrong dates; fill the calendar first (with explicit zeros "
            "where nothing happened).", [date], missing_periods))
    return f, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--actual", required=True)
    ap.add_argument("--forecast", required=True, help="forecast column; empty on history rows")
    ap.add_argument("--series", help="series id column for many series (store, SKU)")
    ap.add_argument("--train-end", help="last date the model was trained on (YYYY-MM-DD)")
    ap.add_argument("--season", type=int, help="seasonal period in rows (default: inferred from the date step)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    df = read_table(a.data)
    for c in [a.date, a.actual, a.forecast, *([a.series] if a.series else [])]:
        if c not in df.columns:
            raise SystemExit(f"error: column '{c}' not in {a.data}")

    findings, stats = check(df, a.date, a.actual, a.forecast, a.series, a.train_end, a.season)
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "forecast-check", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"forecast-check: {status}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        for x in findings:
            print(f"  [{x['severity']}] {x['check']}: {x['message']}")
            if x.get("examples"):
                print(f"      examples: {x['examples']}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):      # input errors: print the message and exit 2 (1 is reserved for FAIL)
            print(e.code, file=sys.stderr)
            sys.exit(2)
        raise
    except pd.errors.EmptyDataError:
        print("error: an input file is empty (no header, no rows)", file=sys.stderr)
        sys.exit(2)
