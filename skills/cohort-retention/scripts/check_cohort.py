#!/usr/bin/env python3
"""Check a cohort retention table before it is charted or averaged: periods that have not finished yet (right
censoring), rates outside 0-100%, period-0 definitions, retention that rises when it cannot, and tiny cohorts.

Usage examples
--------------
    python check_cohort.py --table retention.csv --cohort cohort_month --period month_number --rate retention \
        --grain month --as-of 2026-06-15
    python check_cohort.py --table retention.csv --cohort cohort_week --period week --active active_users \
        --size cohort_size --grain week --definition unbounded

One row per cohort and period: the cohort's start date, the period index (0 = signup period), and either a rate or
active users plus cohort size. Exit code: 1 if any FAIL finding, 0 otherwise, 2 on bad input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def offset(grain: str, k):
    return {"day": pd.DateOffset(days=k), "week": pd.DateOffset(weeks=k), "month": pd.DateOffset(months=k),
            "quarter": pd.DateOffset(months=3 * k)}[grain]


def check(t, cohort, period, rate, active, size, grain, as_of, definition, min_n):
    f, stats = [], {}
    df = pd.DataFrame({"cohort": pd.to_datetime(t[cohort], errors="coerce"),
                       "period": pd.to_numeric(t[period], errors="coerce")})
    if active:
        df["active"] = pd.to_numeric(t[active], errors="coerce")
        df["size"] = pd.to_numeric(t[size], errors="coerce")
        df["rate"] = df["active"] / df["size"]
    else:
        df["rate"] = pd.to_numeric(t[rate], errors="coerce")
        if df["rate"].max() > 1.5:          # percentages
            df["rate"] = df["rate"] / 100
        if size:
            df["size"] = pd.to_numeric(t[size], errors="coerce")
    df = df.dropna(subset=["cohort", "period"])
    if df.empty:
        raise SystemExit(f"error: no rows have both a valid cohort date ('{cohort}') and period ('{period}')")
    stats.update(cohorts=int(df["cohort"].nunique()), max_period=int(df["period"].max()), cells=len(df))

    bad = df[(df["rate"] < 0) | (df["rate"] > 1.0001)]
    if len(bad):
        f.append(finding(
            "cohort.rate_out_of_range", "FAIL",
            f"{len(bad)} cells have retention below 0% or above 100% (max {df['rate'].max():.0%}). Active users are "
            "being counted from a different population than the cohort (duplicate users, events instead of users, "
            "or users without a signup in that cohort).", [], len(bad),
            [f"{r.cohort:%Y-%m-%d} period {int(r.period)}: {r.rate:.0%}" for r in bad.head(4).itertuples()]))

    ends = df.apply(lambda r: r["cohort"] + offset(grain, int(r["period"]) + 1), axis=1)
    incomplete = df[(ends > as_of) & df["rate"].notna()]
    stats["incomplete_cells"] = len(incomplete)
    if len(incomplete):
        complete = df[(ends <= as_of)]
        avg = complete.groupby("period")["rate"].mean()
        drop = [(r, avg.get(r.period)) for r in incomplete.itertuples()]
        ex = [f"{r.cohort:%Y-%m-%d} period {int(r.period)}: {r.rate:.0%}"
              + (f" (finished cohorts average {a:.0%})" if a is not None and a == a else "") for r, a in drop[:4]]
        f.append(finding(
            "cohort.incomplete_period", "FAIL",
            f"{len(incomplete)} cells cover periods that had not finished by {as_of:%Y-%m-%d}, yet they hold a "
            "retention value. Users in those cohorts have not had the full period to come back, so the numbers are "
            "too low and make recent cohorts look worse. Blank these cells (or mark them partial) and exclude them "
            "from averages.", [], len(incomplete), ex))

    p0 = df[df["period"] == 0]["rate"].dropna()
    if definition == "classic" and len(p0) and p0.mean() < 0.95:
        f.append(finding(
            "cohort.period_zero", "WARN",
            f"Period-0 retention averages {p0.mean():.0%}. With classic retention every user is active in the period "
            "they signed up, so it should be 100%. The cohort and activity definitions differ (signup vs first "
            "purchase, or a different user id).", [], round(float(p0.mean()), 3)))

    rises = []
    for c, g in df.sort_values("period").groupby("cohort"):
        r = g["rate"].dropna().to_numpy()
        jumps = [(int(g["period"].iloc[i + 1]), r[i + 1] - r[i]) for i in range(len(r) - 1)]
        limit = 0.005 if definition == "unbounded" else 0.05
        rises += [f"{c:%Y-%m-%d} period {p}: +{d:.1%}" for p, d in jumps if d > limit]
    if rises:
        sev = "FAIL" if definition == "unbounded" else "WARN"
        f.append(finding(
            "cohort.rising_retention", sev,
            f"Retention rises from one period to the next in {len(rises)} places. "
            + ("Unbounded ('still active at or after period N') retention can never rise, so the calculation is wrong."
               if definition == "unbounded" else
               "Possible with classic retention (reactivation campaigns, seasonality), but check it is real."),
            [], len(rises), rises[:4]))

    if "size" in df.columns:
        sizes = df.groupby("cohort")["size"].max()
        small = sizes[sizes < min_n]
        if len(small):
            f.append(finding(
                "cohort.small_cohort", "WARN",
                f"{len(small)} cohorts have fewer than {min_n} users; their curves are noise. Merge them or show "
                "counts next to rates.", [], len(small), [f"{c:%Y-%m-%d}: {int(n)}" for c, n in small.head(4).items()]))
    return f, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--cohort", required=True, help="cohort start date column")
    ap.add_argument("--period", required=True, help="period index column (0 = signup period)")
    ap.add_argument("--rate", help="retention rate column (0-1 or 0-100)")
    ap.add_argument("--active", help="active users column (with --size)")
    ap.add_argument("--size", help="cohort size column")
    ap.add_argument("--grain", choices=["day", "week", "month", "quarter"], default="month")
    ap.add_argument("--as-of", help="date the data was extracted (default: today)")
    ap.add_argument("--definition", choices=["classic", "unbounded"], default="classic",
                    help="classic = active in period N; unbounded = active in period N or later")
    ap.add_argument("--min-n", type=int, default=50)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    t = read_table(a.table)
    if not a.rate and not (a.active and a.size):
        raise SystemExit("error: give --rate, or --active and --size")
    for c in [a.cohort, a.period, *(x for x in (a.rate, a.active, a.size) if x)]:
        if c not in t.columns:
            raise SystemExit(f"error: column '{c}' not in {a.table}")
    as_of = pd.Timestamp(a.as_of) if a.as_of else pd.Timestamp.today().normalize()

    findings, stats = check(t, a.cohort, a.period, a.rate, a.active, a.size, a.grain, as_of, a.definition, a.min_n)
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "cohort-retention", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"cohort-retention: {status}")
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
