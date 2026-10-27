#!/usr/bin/env python3
"""First-look profile of a new dataset, focused on problems that corrupt analysis: impossible values, placeholder
codes (-999, 'N/A', 1900-01-01), mixed types, inconsistent labels ('NY' vs 'ny '), non-unique IDs, duplicate rows,
mostly-empty and constant columns, and dates in the future.

Usage examples
--------------
    python check_profile.py --data customers.csv
    python check_profile.py --data orders.parquet --id order_id --as-of 2026-06-30

Exit code: 1 if any FAIL finding, 0 otherwise, 2 on bad input. Add --json for structured output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

NONNEG = re.compile(r"(^|_)(age|price|amount|qty|quantity|count|revenue|cost|weight|height|duration|units|sales|"
                    r"salary|income|distance|tenure|balance_due|fee|tax|days|minutes|seconds|visits|clicks)(_|$)", re.I)
AGE = re.compile(r"(^|_)age(_|$)|_age_years", re.I)
DATE_NAME = re.compile(r"(date|_at$|_on$|time|dob|birth|(^|_)day($|_)|_dt$)", re.I)
BIRTH = re.compile(r"(dob|birth)", re.I)
PLACEHOLDER_TEXT = {"n/a", "na", "null", "none", "-", "--", "?", "unknown", "undefined", "nan", "#n/a", "missing",
                    "tbd", "not available", "nil", "(blank)", "blank"}
PLACEHOLDER_NUM = {-999, -9999, -99999, -99, 9999, 99999, 999999, 9999999}
SENTINEL_DATES = {"1900-01-01", "1970-01-01", "9999-12-31", "0001-01-01", "2099-12-31", "1899-12-30", "2999-12-31"}


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: file not found: {path}")
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
        return df.astype(str).where(df.notna())
    # read raw text so placeholders like 'N/A' and leading zeros stay visible
    return pd.read_csv(p, sep="\t" if p.suffix.lower() in {".tsv", ".tab"} else ",", dtype=str,
                       keep_default_na=False, na_values=[""])


def finding(check, severity, message, columns=None, value=None, examples=None):
    out = {"check": check, "severity": severity, "message": message, "columns": columns or [], "value": value}
    if examples:
        out["examples"] = examples
    return out


def to_dates(s: pd.Series) -> pd.Series:
    """Parse to naive UTC timestamps (NaT where unparseable)."""
    try:
        d = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
    except (TypeError, ValueError):
        d = pd.to_datetime(s, errors="coerce", utc=True)
    return d.dt.tz_localize(None)


def profile(df, ids, as_of):
    f, stats = [], {"rows": len(df), "columns": len(df.columns)}
    n = len(df)
    if n == 0:
        raise SystemExit("error: the file has no rows")

    dup = int(df.duplicated().sum())
    stats["duplicate_rows"] = dup
    if dup:
        f.append(finding("profile.duplicate_rows", "WARN",
                         f"{dup:,} rows ({dup / n:.1%}) are exact duplicates of another row. Find out whether they "
                         "are double loads before counting or summing. (Without an ID column, identical rows can be "
                         "legitimate; pass --id to check the real key.)", [], dup))

    for c in ids:
        k = df[c]
        nd = int(k.duplicated(keep=False).sum())
        nn = int(k.isna().sum())
        if nd or nn:
            f.append(finding(
                "profile.id_not_unique", "FAIL",
                f"'{c}' is meant to identify rows but {nd:,} rows share an ID and {nn:,} are empty. Every join and "
                "count on this column will be off.", [c], nd + nn, k[k.duplicated()].dropna().head(5).tolist()))

    for c in df.columns:
        s = df[c]
        txt = s.dropna().astype(str)
        low = txt.str.strip().str.lower()
        is_ph = low.isin(PLACEHOLDER_TEXT)
        empty_share = float((s.isna().sum() + is_ph.sum()) / n)
        real = txt[~is_ph]

        if empty_share > 0.5:
            f.append(finding("profile.mostly_empty", "WARN",
                             f"'{c}' is empty or a placeholder in {empty_share:.0%} of rows.", [c], round(empty_share, 3)))
        if is_ph.sum() and is_ph.sum() >= max(2, 0.001 * n):
            counts = low[is_ph].value_counts().head(4)
            f.append(finding(
                "profile.placeholder_text", "WARN",
                f"'{c}' uses text placeholders for missing values ({', '.join(f'{k!r} x{v}' for k, v in counts.items())}). "
                "They are counted as real values (a category, or a parse failure). Convert them to NULL.",
                [c], int(is_ph.sum())))
        if real.empty:
            continue
        if real.nunique() == 1 and n >= 20:
            f.append(finding("profile.constant", "WARN", f"'{c}' has a single value ({real.iloc[0]!r}); it carries no "
                             "information and may signal a broken extract.", [c]))

        num = pd.to_numeric(real.str.replace(",", "", regex=False), errors="coerce")
        share_num = float(num.notna().mean())
        if 0.05 < share_num < 0.95:
            bad = real[num.isna()].value_counts().head(5).index.tolist()
            f.append(finding(
                "profile.mixed_types", "WARN",
                f"'{c}' is {share_num:.0%} numeric and {1 - share_num:.0%} text. Arithmetic will fail or silently "
                "drop the text rows. Clean or split the column.", [c], round(share_num, 3), bad))

        if share_num >= 0.95:
            v = num.dropna()
            ph = v[v.isin(PLACEHOLDER_NUM)]
            if len(ph) >= max(2, 0.001 * n):
                rest = v[~v.isin(PLACEHOLDER_NUM)]
                lo, hi = (rest.quantile(0.01), rest.quantile(0.99)) if len(rest) else (0, 0)
                outside = ph[(ph < lo) | (ph > hi)]
                if len(outside):
                    counts = outside.value_counts().head(3)
                    f.append(finding(
                        "profile.placeholder_number", "WARN",
                        f"'{c}' contains code values far outside its normal range "
                        f"({', '.join(f'{k:g} x{cnt}' for k, cnt in counts.items())}); they look like 'missing' codes. "
                        "Averages and sums will be distorted; convert them to NULL.", [c], int(len(outside))))
            clean = v[~v.isin(PLACEHOLDER_NUM)]
            if NONNEG.search(c):
                neg = clean[clean < 0]
                if len(neg):
                    f.append(finding(
                        "profile.impossible_values", "FAIL",
                        f"'{c}' should not be negative but {len(neg):,} rows are (min {neg.min():g}). These may be "
                        "returns or corrections that need their own handling, or a sign error. Decide before "
                        "aggregating.", [c], int(len(neg)), [f"{x:g}" for x in neg.head(5)]))
            if AGE.search(c):
                old = clean[clean > 120]
                if len(old):
                    f.append(finding("profile.impossible_values", "FAIL",
                                     f"'{c}' has {len(old):,} ages above 120 (max {old.max():g}).", [c], int(len(old)),
                                     [f"{x:g}" for x in old.head(5)]))
            continue

        if DATE_NAME.search(c):
            d = to_dates(real)
            if d.notna().mean() >= 0.8:
                iso = d.dropna().dt.strftime("%Y-%m-%d")
                sent = iso[iso.isin(SENTINEL_DATES)]
                if len(sent):
                    f.append(finding(
                        "profile.sentinel_dates", "WARN",
                        f"'{c}' contains placeholder dates ({', '.join(sorted(set(sent)))[:60]}; {len(sent):,} rows). "
                        "They distort ages, durations and time trends; convert them to NULL.", [c], int(len(sent))))
                fut = d[(d > as_of) & ~iso.reindex(d.index).isin(SENTINEL_DATES)]
                if len(fut):
                    sev = "FAIL" if BIRTH.search(c) else "WARN"
                    f.append(finding(
                        "profile.future_dates", sev,
                        f"'{c}' has {len(fut):,} dates after {as_of:%Y-%m-%d} (max {fut.max():%Y-%m-%d}). Check for "
                        "typos, swapped day/month, or planned dates mixed with actual ones.", [c], int(len(fut))))
                continue

        # Categorical labels that differ only by case, spacing or punctuation
        if real.nunique() <= max(200, 0.05 * n):
            norm = real.str.strip().str.lower().str.replace(r"[\s._-]+", " ", regex=True)
            groups = real.groupby(norm).agg(lambda x: sorted(set(x)))
            messy = groups[groups.map(len) > 1]
            if len(messy):
                ex = [" / ".join(repr(v) for v in variants[:4]) for variants in messy.head(4)]
                f.append(finding(
                    "profile.inconsistent_labels", "WARN",
                    f"'{c}' has {len(messy)} labels written more than one way (case, spaces or punctuation). Group-bys "
                    "split the same category into several rows. Normalise before grouping.", [c], int(len(messy)), ex))
    return f, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--id", default="", help="comma-separated columns that should uniquely identify a row")
    ap.add_argument("--as-of", help="dates after this are 'in the future' (default: today)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    df = read_table(a.data)
    ids = [c.strip() for c in a.id.split(",") if c.strip()]
    for c in ids:
        if c not in df.columns:
            raise SystemExit(f"error: column '{c}' not in {a.data}")
    as_of = pd.Timestamp(a.as_of) if a.as_of else pd.Timestamp.today().normalize()

    findings, stats = profile(df, ids, as_of)
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "data-profile", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"data-profile: {status}")
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
