#!/usr/bin/env python3
"""Diagnose a linear regression before its coefficients are interpreted: perfect and strong multicollinearity (VIF),
heteroscedasticity (Breusch-Pagan), missing non-linearity (RESET), autocorrelated residuals (Durbin-Watson),
influential points (Cook's distance), skewed residuals and too few rows per feature.

Usage examples
--------------
    python check_regression.py --data sales.csv --target revenue --features price,ads,store_size
    python check_regression.py --data weekly.csv --target demand --order week          # time-ordered data

Fits ordinary least squares with an intercept on the numeric features (default: every numeric column except the
target). Exit code: 1 if any FAIL finding, 0 otherwise, 2 on bad input. Add --json for structured output.
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


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def chi2_sf(x, df):
    if df <= 0 or x <= 0:
        return 1.0
    if df == 1:
        return 2 * norm_sf(math.sqrt(x))
    z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return norm_sf(z)


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot else 0.0
    return beta, fitted, resid, r2


def with_const(M):
    return np.column_stack([np.ones(len(M)), M])


def check(df, target, features, order):
    f, stats = [], {}
    data = df[[target, *features] + ([order] if order else [])].copy()
    for c in [target, *features]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna()
    if order:
        data = data.sort_values(order)
    n, k = len(data), len(features)
    stats.update(rows_used=n, rows_dropped_for_missing=len(df) - n, features=k)
    if n <= k + 2:
        raise SystemExit("error: not enough complete rows to fit the model")
    X = data[features].to_numpy(float)
    y = data[target].to_numpy(float)
    Xc = with_const(X)

    if n < 10 * k:
        f.append(finding("regression.small_sample", "WARN",
                         f"{n} rows for {k} features (fewer than 10 per feature). Coefficients will be unstable and "
                         "p-values optimistic. Reduce features or get more data.", [], round(n / k, 1)))

    # Exact (or rounding-level) linear dependence: a constant, or a feature the others explain to R2 > 0.999999
    culprits = [c for c, sd in zip(features, X.std(axis=0)) if sd == 0]
    for i, c in enumerate(features):
        if c in culprits or k == 1:
            continue
        _, _, _, r2 = ols(with_const(np.delete(X, i, axis=1)), X[:, i])
        if r2 > 0.999999:
            culprits.append(c)
    if culprits or np.linalg.matrix_rank(Xc) < k + 1:
        f.append(finding(
            "regression.perfect_collinearity", "FAIL",
            f"The features are linearly dependent ({', '.join(culprits) or 'see VIFs'}): one is an exact combination "
            "of others (a total next to its parts, a dummy for every category, a constant). Coefficients for these "
            "are arbitrary. Drop the redundant column.", culprits, len(culprits)))
        return f, stats

    vifs = {}
    for i, c in enumerate(features):
        if k == 1:
            break
        _, _, _, r2 = ols(with_const(np.delete(X, i, axis=1)), X[:, i])
        vifs[c] = round(1 / (1 - r2), 2) if r2 < 1 else float("inf")
    stats["vif"] = vifs
    high = {c: v for c, v in vifs.items() if v > 10}
    if high:
        f.append(finding(
            "regression.multicollinearity", "WARN",
            f"High variance inflation factors: {high}. These features move together, so their individual "
            "coefficients (and signs) are unreliable even if predictions are fine. Do not interpret them one by one; "
            "combine or drop features.", list(high), max(high.values())))

    beta, fitted, resid, r2 = ols(Xc, y)
    stats.update(r2=round(r2, 4), coefficients={"intercept": round(float(beta[0]), 6),
                                                **{c: round(float(b), 6) for c, b in zip(features, beta[1:])}})

    _, _, _, r2_bp = ols(Xc, resid ** 2)
    lm = n * r2_bp
    p_bp = chi2_sf(lm, k)
    stats["breusch_pagan_p"] = float(f"{p_bp:.3g}")
    if p_bp < 0.001:
        f.append(finding(
            "regression.heteroscedasticity", "WARN",
            f"Residual spread changes with the features (Breusch-Pagan p = {p_bp:.2g}). Coefficients are still unbiased "
            "but standard errors and p-values are wrong. Use robust (HC3) standard errors, or model log(y).",
            [target], float(f"{p_bp:.3g}")))

    zf = (fitted - fitted.mean()) / (fitted.std() or 1)
    _, _, _, r2_reset = ols(np.column_stack([Xc, zf ** 2, zf ** 3]), resid)
    p_reset = chi2_sf(n * r2_reset, 2)
    stats["reset_p"] = float(f"{p_reset:.3g}")
    if p_reset < 0.001:
        f.append(finding(
            "regression.nonlinearity", "WARN",
            f"The residuals still depend on the fitted values (RESET p = {p_reset:.2g}): the relationship is not "
            "linear in these features. Plot residuals against each feature; add transformations, interactions or "
            "splines.", [target], float(f"{p_reset:.3g}")))

    if order:
        dw = float((np.diff(resid) ** 2).sum() / (resid ** 2).sum())
        stats["durbin_watson"] = round(dw, 3)
        if dw < 1.5 or dw > 2.5:
            f.append(finding(
                "regression.autocorrelation", "WARN",
                f"Durbin-Watson is {dw:.2f} (2 means no autocorrelation): consecutive residuals are related. Standard "
                "errors are too small and significance is overstated. Use time-series errors (Newey-West), add lags, "
                "or model the trend.", [order], round(dw, 3)))

    q, _ = np.linalg.qr(Xc)
    h = (q ** 2).sum(axis=1)
    mse = float((resid ** 2).sum() / (n - k - 1))
    cooks = (resid ** 2 / ((k + 1) * mse)) * h / np.clip((1 - h) ** 2, 1e-12, None)
    stats["max_cooks_distance"] = round(float(cooks.max()), 3)
    infl = np.flatnonzero(cooks > 1)
    if len(infl):
        f.append(finding(
            "regression.influential_points", "WARN",
            f"{len(infl)} rows have Cook's distance above 1: each one alone moves the coefficients materially. Check "
            "them for data errors and report results with and without them.", [], int(len(infl)),
            [f"row {int(data.index[i])}: Cook's D {cooks[i]:.2f}" for i in infl[:5]]))

    sk = float(pd.Series(resid).skew())
    stats["residual_skew"] = round(sk, 3)
    if abs(sk) > 2:
        f.append(finding(
            "regression.skewed_residuals", "WARN",
            f"Residuals are strongly skewed ({sk:.2f}). A log or other transform of '{target}' usually fits better, and "
            "prediction intervals from this model will be wrong.", [target], round(sk, 3)))
    return f, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--features", help="comma-separated numeric features (default: all numeric except target/order)")
    ap.add_argument("--order", help="time or sequence column, enables the autocorrelation check")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    df = read_table(a.data)
    for c in [a.target, *([a.order] if a.order else [])]:
        if c not in df.columns:
            raise SystemExit(f"error: column '{c}' not in {a.data}")
    if a.features:
        features = [c.strip() for c in a.features.split(",")]
        for c in features:
            if c not in df.columns:
                raise SystemExit(f"error: column '{c}' not in {a.data}")
    else:
        features = [c for c in df.select_dtypes("number").columns if c not in (a.target, a.order)]
    if not features:
        raise SystemExit("error: no numeric features")

    findings, stats = check(df, a.target, features, a.order)
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "regression-diagnostics", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"regression-diagnostics: {status}")
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
