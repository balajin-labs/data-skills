#!/usr/bin/env python3
"""Deterministic leakage checks for a supervised-learning dataset.

Usage examples
--------------
One file with a split column:
    python check_leakage.py --data train_test.csv --target churned --split split --entity customer_id --time snapshot_date

Separate train and test files:
    python check_leakage.py --data train.csv --test test.csv --target churned --entity customer_id

Exit code: 1 if any FAIL finding, 0 otherwise, 2 on bad input.
Add --json for machine-readable output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Column names that often describe something known only after the outcome.
POST_OUTCOME_HINTS = re.compile(
    r"(status|outcome|result|resolved|resolution|closed|cancel|refund|churn|"
    r"label|target|final|discharge|end_?date|termination|default_flag|chargeback|"
    r"won|lost|converted|after|post_|future|next_)",
    re.IGNORECASE,
)

SINGLE_FEATURE_FAIL = 0.95
SINGLE_FEATURE_WARN = 0.90
MAX_CATEGORIES = 200


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: file not found: {path}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p)


def finding(check, severity, message, columns=None, value=None):
    return {
        "check": check,
        "severity": severity,
        "message": message,
        "columns": columns or [],
        "value": value,
    }


def rank_auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC of score against a 0/1 label (ties count half)."""
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_target(y: pd.Series):
    vals = y.dropna().unique()
    if len(vals) != 2:
        return None
    positive = sorted(vals, key=lambda v: str(v))[-1]
    if set(map(str, vals)) <= {"0", "1", "0.0", "1.0", "False", "True"}:
        positive = [v for v in vals if str(v) in {"1", "1.0", "True"}][0]
    return (y == positive).astype(int)


def single_feature_strength(df: pd.DataFrame, target: str, skip: set[str], rng: np.random.Generator):
    """How well each feature alone predicts the target, measured out-of-sample.

    Numeric features: rank AUC (binary target) or |Spearman| (numeric target).
    Categorical features: category -> target mean learned on one half,
    scored on the other half, so high-cardinality IDs cannot memorise.
    """
    results = []
    y_raw = df[target]
    yb = binary_target(y_raw)
    numeric_target = yb is None and pd.api.types.is_numeric_dtype(y_raw)
    if yb is None and not numeric_target:
        return results, "target is neither binary nor numeric; single-feature check skipped"

    mask_half = rng.random(len(df)) < 0.5
    for col in df.columns:
        if col in skip or col == target:
            continue
        x = df[col]
        valid = x.notna() & y_raw.notna()
        if valid.sum() < 30:
            continue
        is_continuous = (pd.api.types.is_numeric_dtype(x)
                         and not pd.api.types.is_bool_dtype(x)
                         and x.nunique() > 10)
        if is_continuous:
            xv = x[valid].to_numpy(dtype=float)
            if yb is not None:
                auc = rank_auc(xv, yb[valid].to_numpy())
                strength = max(auc, 1 - auc)
            else:
                strength = abs(pd.Series(xv).corr(y_raw[valid].reset_index(drop=True), method="spearman"))
        else:
            if x.nunique() > MAX_CATEGORIES and x.nunique() > 0.5 * valid.sum():
                continue  # looks like an identifier; handled elsewhere
            xs = x.astype(str)
            yy = yb if yb is not None else y_raw
            fit = valid & mask_half
            score_rows = valid & ~mask_half
            if fit.sum() < 15 or score_rows.sum() < 15:
                continue
            means = yy[fit].groupby(xs[fit]).mean()
            prior = float(yy[fit].mean())
            enc = xs[score_rows].map(means).fillna(prior).to_numpy(dtype=float)
            if yb is not None:
                auc = rank_auc(enc, yb[score_rows].to_numpy())
                strength = max(auc, 1 - auc)
            else:
                strength = abs(pd.Series(enc).corr(y_raw[score_rows].reset_index(drop=True), method="spearman"))
        if strength == strength:  # not NaN
            results.append((col, float(strength)))
    return results, None


def is_datetime_like(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    # pandas < 3 stores text as object; pandas >= 3 uses a dedicated string dtype
    if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(200)
    if sample.empty:
        return False
    return bool(sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9)


def to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def audit(df: pd.DataFrame, target: str, split: str | None, entity: str | None,
          time_col: str | None, ignore: list[str], seed: int = 0):
    findings = []
    notes = []
    rng = np.random.default_rng(seed)

    if target not in df.columns:
        raise SystemExit(f"error: target column '{target}' not in data")
    for c in [split, entity, time_col, *ignore]:
        if c and c not in df.columns:
            raise SystemExit(f"error: column '{c}' not in data")

    skip = {c for c in [split, entity, time_col, *ignore] if c}

    # 1. Single-feature strength (target proxies)
    strengths, note = single_feature_strength(df, target, skip, rng)
    if note:
        notes.append(note)
    for col, s in sorted(strengths, key=lambda t: -t[1]):
        if s >= SINGLE_FEATURE_FAIL:
            findings.append(finding(
                "leakage.single_feature", "FAIL",
                f"'{col}' alone predicts the target with strength {s:.3f}. "
                "A single column this predictive is usually derived from the outcome. "
                "Confirm it is known at prediction time before using it.",
                [col], round(s, 4)))
        elif s >= SINGLE_FEATURE_WARN:
            findings.append(finding(
                "leakage.single_feature", "WARN",
                f"'{col}' alone predicts the target with strength {s:.3f}. Check how it is computed.",
                [col], round(s, 4)))

    # 2. Suspicious names
    for col in df.columns:
        if col in skip or col == target:
            continue
        if POST_OUTCOME_HINTS.search(col):
            findings.append(finding(
                "leakage.suspicious_name", "WARN",
                f"Column name '{col}' suggests information recorded after the outcome. "
                "Verify it exists at prediction time.", [col]))

    # 3. Dates recorded after the prediction time
    if time_col:
        t0 = to_datetime(df[time_col])
        for col in df.columns:
            if col in skip or col == target or not is_datetime_like(df[col]):
                continue
            tc = to_datetime(df[col])
            both = t0.notna() & tc.notna()
            if both.sum() == 0:
                continue
            share = float((tc[both] > t0[both]).mean())
            if share > 0.5:
                findings.append(finding(
                    "leakage.future_timestamp", "FAIL",
                    f"'{col}' is later than '{time_col}' in {share:.0%} of rows: "
                    "it was recorded after the prediction time.", [col, time_col], round(share, 4)))
            elif share > 0.05:
                findings.append(finding(
                    "leakage.future_timestamp", "WARN",
                    f"'{col}' is later than '{time_col}' in {share:.0%} of rows.", [col, time_col], round(share, 4)))

    # Split-aware checks
    if split:
        labels = df[split].astype(str).str.lower()
        test_mask = labels.isin({"test", "valid", "validation", "holdout", "val", "oot"})
        train_mask = labels.isin({"train", "training"})
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            notes.append(f"split column '{split}' needs 'train' and 'test'/'valid' labels; split checks skipped")
        else:
            tr, te = df[train_mask], df[test_mask]

            # 4. Entity overlap
            if entity:
                shared = set(tr[entity].dropna()) & set(te[entity].dropna())
                if shared:
                    share = len(shared) / max(te[entity].nunique(), 1)
                    findings.append(finding(
                        "leakage.entity_overlap", "FAIL",
                        f"{len(shared)} '{entity}' values ({share:.0%} of test entities) appear in both train and test. "
                        "Use a group split so each entity lives in one side only.",
                        [entity], round(share, 4)))

            # 5. Time overlap
            if time_col:
                ttr, tte = to_datetime(tr[time_col]), to_datetime(te[time_col])
                if ttr.notna().any() and tte.notna().any():
                    test_start = tte.min()
                    share = float((ttr > test_start).mean())
                    if share > 0.01:
                        findings.append(finding(
                            "leakage.time_overlap", "FAIL",
                            f"{share:.0%} of training rows are later than the start of the test period "
                            f"({test_start.date()}). Train only on data before the test window.",
                            [time_col], round(share, 4)))

            # 6. Duplicate rows across splits
            feature_cols = [c for c in df.columns if c not in skip and c != target]
            if feature_cols:
                key = lambda d: pd.util.hash_pandas_object(d[feature_cols], index=False)
                dup = key(te).isin(set(key(tr))).mean()
                if dup > 0.05:
                    findings.append(finding(
                        "leakage.duplicate_rows", "FAIL",
                        f"{dup:.0%} of test rows have an identical feature row in train.", feature_cols[:10], round(float(dup), 4)))
                elif dup > 0.01:
                    findings.append(finding(
                        "leakage.duplicate_rows", "WARN",
                        f"{dup:.0%} of test rows have an identical feature row in train.", feature_cols[:10], round(float(dup), 4)))
    else:
        if entity:
            per_entity = df.groupby(entity).size()
            if (per_entity > 1).mean() > 0.05:
                notes.append(f"'{entity}' repeats across rows; any train/test split must be grouped by it")
        if time_col:
            notes.append(f"'{time_col}' given: split by time, not at random")

    return findings, notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV/Parquet with features and target (or the train file)")
    ap.add_argument("--test", help="separate test/validation file (adds a split automatically)")
    ap.add_argument("--target", required=True)
    ap.add_argument("--split", help="column with train/test labels")
    ap.add_argument("--entity", help="entity id column (customer, patient, order)")
    ap.add_argument("--time", dest="time_col", help="prediction-time / snapshot column")
    ap.add_argument("--ignore", nargs="*", default=[], help="columns to skip (other ids)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    df = read_table(args.data)
    split = args.split
    if args.test:
        te = read_table(args.test)
        df = pd.concat([df.assign(__split="train"), te.assign(__split="test")], ignore_index=True)
        split = "__split"

    findings, notes = audit(df, args.target, split, args.entity, args.time_col, args.ignore)
    report = {"tool": "leakage-audit", "rows": int(len(df)), "findings": findings, "notes": notes,
              "status": "FAIL" if any(f["severity"] == "FAIL" for f in findings)
              else "WARN" if findings else "PASS"}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"leakage-audit: {report['status']}  ({len(df)} rows)")
        for f in findings:
            print(f"  [{f['severity']}] {f['check']}: {f['message']}")
        for n in notes:
            print(f"  note: {n}")
    return 1 if report["status"] == "FAIL" else 0


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
