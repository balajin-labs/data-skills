#!/usr/bin/env python3
"""Check whether predicted probabilities can be used as probabilities: overall level vs observed rate, expected
calibration error, over/under-confidence (calibration slope), and segments where predictions are systematically off.

Usage examples
--------------
    python check_calibration.py --pred preds.csv --y-true churned --y-score prob
    python check_calibration.py --pred preds.csv --y-true default --y-score pd --segment region,product --bins 10

Use this when scores feed expected values (expected loss, expected revenue, capacity), are shown to people as
percentages, or are compared across models or segments. Exit code: 1 if any FAIL, 0 otherwise, 2 on bad input.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


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


def to_binary(y: pd.Series):
    vals = pd.unique(y.dropna())
    if len(vals) != 2:
        return None
    strs = {str(v) for v in vals}
    if strs <= {"0", "1", "0.0", "1.0", "True", "False", "true", "false"}:
        return y.astype(str).str.lower().isin({"1", "1.0", "true"}).astype(int)
    return (y == sorted(vals, key=str)[-1]).astype(int)


def ece(p: np.ndarray, y: np.ndarray, bins: int):
    order = np.argsort(p)
    groups = np.array_split(order, bins)
    rows, total = [], 0.0
    for g in groups:
        if len(g) == 0:
            continue
        mp, rate = float(p[g].mean()), float(y[g].mean())
        total += len(g) / len(p) * abs(mp - rate)
        rows.append({"n": len(g), "mean_pred": round(mp, 4), "observed": round(rate, 4)})
    return total, rows


def check(df, ycol, scol, segments, bins):
    f, stats = [], {}
    y = to_binary(df[ycol])
    if y is None:
        raise SystemExit(f"error: '{ycol}' must have exactly two values")
    p = pd.to_numeric(df[scol], errors="coerce")
    keep = p.notna() & y.notna()
    p, y = p[keep].to_numpy(float), y[keep].to_numpy(int)
    n = len(p)
    if n < 50:
        raise SystemExit("error: need at least 50 rows with a score and a label")
    out = int(((p < 0) | (p > 1)).sum())
    if out:
        f.append(finding("calibration.not_probabilities", "FAIL",
                         f"{out:,} scores are outside [0, 1] (min {p.min():.3g}, max {p.max():.3g}). These are raw "
                         "scores or logits, not probabilities; apply the link function or calibrate first.", [scol], out))
        return f, stats

    rate, mp = float(y.mean()), float(p.mean())
    se = math.sqrt(max(rate * (1 - rate), 1e-9) / n)
    e, table = ece(p, y, bins)
    brier = float(((p - y) ** 2).mean())
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    slope, intercept = float(lr.coef_[0][0]), float(lr.intercept_[0])
    stats.update(rows=n, observed_rate=round(rate, 4), mean_predicted=round(mp, 4), ece=round(e, 4),
                 brier=round(brier, 4), calibration_slope=round(slope, 3), calibration_intercept=round(intercept, 3),
                 reliability=table)

    ratio = mp / rate if rate else float("inf")
    if abs(mp - rate) > 3 * se and (ratio > 1.25 or ratio < 0.8):
        f.append(finding(
            "calibration.mean_mismatch", "FAIL",
            f"The model predicts {mp:.1%} on average, but the observed rate is {rate:.1%} (x{ratio:.2f}). Sums of "
            "predictions (expected churners, expected losses) will be off by that factor. Typical cause: training on "
            "a rebalanced or down-sampled set, or a shift in the base rate. Recalibrate on data with the real rate.",
            [scol], round(ratio, 3)))
    if e > 0.1:
        f.append(finding("calibration.ece", "FAIL",
                         f"Expected calibration error is {e:.3f}: on average a prediction is {e:.1%} away from the "
                         "observed rate for similar scores. Do not use these scores as probabilities without "
                         "recalibration (isotonic or Platt on a held-out set).", [scol], round(e, 4)))
    elif e > 0.05:
        f.append(finding("calibration.ece", "WARN",
                         f"Expected calibration error is {e:.3f}; probabilities are approximate.", [scol], round(e, 4)))
    if slope < 0.8:
        f.append(finding(
            "calibration.overconfident", "WARN",
            f"Calibration slope is {slope:.2f} (1.0 is ideal): predictions are too extreme - high scores too high and "
            "low scores too low. Common after overfitting or boosting without calibration.", [scol], round(slope, 3)))
    elif slope > 1.25:
        f.append(finding(
            "calibration.underconfident", "WARN",
            f"Calibration slope is {slope:.2f}: predictions are squeezed toward the middle; the model is more "
            "certain than its scores say. Common with heavy regularisation or bagging.", [scol], round(slope, 3)))

    overall_gap = mp - rate
    for s in segments:
        g = df.loc[keep, s].astype(str).to_numpy()
        bad = []
        for val in pd.unique(g):
            m = g == val
            if m.sum() < 100:
                continue
            r, q = float(y[m].mean()), float(p[m].mean())
            gap = q - r
            zz = gap / math.sqrt(max(r * (1 - r), 1e-9) / m.sum())
            if abs(zz) > 3 and abs(gap) > 0.03 and abs(gap) > 2 * abs(overall_gap):
                bad.append(f"{val}: predicted {q:.1%} vs observed {r:.1%} (n={int(m.sum())})")
        if bad:
            f.append(finding(
                "calibration.segment", "WARN",
                f"Within '{s}', predictions are systematically off for {len(bad)} segment(s) even though the overall "
                "level is close. Decisions using a single threshold will treat these segments differently.",
                [s], len(bad), bad[:5]))
    return f, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="one row per scored example (ideally a held-out set)")
    ap.add_argument("--y-true", required=True)
    ap.add_argument("--y-score", required=True, help="predicted probability of the positive class")
    ap.add_argument("--segment", default="", help="comma-separated columns to check calibration within")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    df = read_table(a.pred)
    segments = [s.strip() for s in a.segment.split(",") if s.strip()]
    for c in [a.y_true, a.y_score, *segments]:
        if c not in df.columns:
            raise SystemExit(f"error: column '{c}' not in {a.pred}")
    findings, stats = check(df, a.y_true, a.y_score, segments, a.bins)
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "calibration-check", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"calibration-check: {status}")
        for k, v in stats.items():
            if k != "reliability":
                print(f"  {k}: {v}")
        if stats.get("reliability"):
            print("  reliability (mean predicted -> observed):")
            for r in stats["reliability"]:
                print(f"    {r['mean_pred']:.3f} -> {r['observed']:.3f}  (n={r['n']})")
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
