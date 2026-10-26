#!/usr/bin/env python3
"""Check a join before trusting its output: fan-out, unmatched keys, key-format mismatches, null keys and double counting.

Usage examples
--------------
    python check_join.py --left orders.csv --right customers.csv --on customer_id
    python check_join.py --left orders.csv --right items.csv --on order_id --how inner --sum-col order_total
    python check_join.py --left a.csv --right b.csv --left-on cust_no --right-on customer_id --expect one-to-one

--expect states the relationship you believe holds (default many-to-one: each left row matches at most one right row).
Exit code: 1 if any FAIL finding, 0 otherwise, 2 on bad input. Add --json for structured output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: file not found: {path}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    # keys are read as text so '007' stays '007'; numeric columns are converted where needed
    return pd.read_csv(p, sep="\t" if p.suffix.lower() in {".tsv", ".tab"} else ",", dtype=str, keep_default_na=True)


def finding(check, severity, message, columns=None, value=None, examples=None):
    out = {"check": check, "severity": severity, "message": message, "columns": columns or [], "value": value}
    if examples:
        out["examples"] = examples
    return out


def key_frame(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if len(cols) == 1:
        return df[cols[0]]
    return df[cols].astype(str).agg("|".join, axis=1).where(df[cols].notna().all(axis=1))


def normalise(s: pd.Series) -> pd.Series:
    def norm(v):
        if pd.isna(v):
            return v
        t = str(v).strip().lower()
        t = re.sub(r"\.0+$", "", t)          # 123.0 -> 123
        t = re.sub(r"[\s\-_]", "", t)          # 'AB-12 ' -> 'ab12'
        if t.isdigit():
            t = t.lstrip("0") or "0"           # 00123 -> 123
        return t
    return s.map(norm)


def match_rate(left_keys: pd.Series, right_keys: pd.Series) -> float:
    lk = left_keys.dropna()
    if lk.empty:
        return 0.0
    return float(lk.isin(set(right_keys.dropna())).mean())


def check(left, right, lcols, rcols, how, expect, sum_col, min_match):
    f, stats = [], {}
    lk, rk = key_frame(left, lcols), key_frame(right, rcols)
    stats.update(left_rows=len(left), right_rows=len(right))

    # Null keys
    for side, keys, cols in (("left", lk, lcols), ("right", rk, rcols)):
        share = float(keys.isna().mean())
        stats[f"{side}_null_key_share"] = round(share, 4)
        if share > 0.01:
            f.append(finding("join.null_keys", "WARN",
                             f"{share:.1%} of {side} rows have a null key {cols}; they can never match "
                             "and silently drop out of inner joins.", cols, round(share, 4)))

    # Duplicate keys
    r_counts = rk.dropna().value_counts()
    l_counts = lk.dropna().value_counts()
    r_dup_keys = r_counts[r_counts > 1]
    l_dup_keys = l_counts[l_counts > 1]
    stats.update(right_duplicate_keys=int(len(r_dup_keys)), left_duplicate_keys=int(len(l_dup_keys)))

    # Output size without materialising the join
    matches = lk.map(r_counts).fillna(0)
    out_rows = int(matches.clip(lower=1).sum()) if how == "left" else int(matches.sum())
    stats["output_rows"] = out_rows
    multiplier = out_rows / max(len(left), 1)
    stats["row_multiplier"] = round(multiplier, 4)

    if expect in ("many-to-one", "one-to-one") and len(r_dup_keys):
        affected = int((matches > 1).sum())
        f.append(finding(
            "join.fan_out", "FAIL",
            f"Right side has {len(r_dup_keys)} duplicated keys, so {affected} left rows match more than once. "
            f"The join returns {out_rows:,} rows from {len(left):,} left rows (x{multiplier:.2f}). "
            "Sums and counts over the result will be inflated. Deduplicate or aggregate the right side first, "
            "or confirm the relationship is really one-to-many.",
            rcols, round(multiplier, 4), [str(k) for k in r_dup_keys.index[:5]]))
    if expect == "one-to-one" and len(l_dup_keys):
        f.append(finding("join.left_duplicates", "FAIL",
                         f"Left side has {len(l_dup_keys)} duplicated keys but the join is expected to be one-to-one.",
                         lcols, int(len(l_dup_keys)), [str(k) for k in l_dup_keys.index[:5]]))

    # Match rates and key-format mismatch
    raw = match_rate(lk, rk)
    stats["left_match_rate"] = round(raw, 4)
    if len(lcols) == 1:
        norm = match_rate(normalise(lk), normalise(rk))
        stats["left_match_rate_after_normalising"] = round(norm, 4)
        if norm - raw > 0.05:
            f.append(finding(
                "join.key_format", "FAIL",
                f"Only {raw:.1%} of left keys match as-is, but {norm:.1%} match after normalising case, "
                "whitespace, separators and leading zeros. The keys are formatted differently on each side "
                "(e.g. '00123' vs '123'). Fix the formats before joining.",
                lcols + rcols, round(norm - raw, 4),
                [f"{a} vs {b}" for a, b in _format_examples(lk, rk)]))
    if raw < min_match:
        f.append(finding("join.unmatched", "WARN" if how == "left" else "FAIL",
                         f"{1 - raw:.1%} of left keys find no match on the right"
                         + (" and are dropped by the inner join." if how == "inner" else
                            "; their right-side columns will be null."),
                         lcols, round(1 - raw, 4)))

    # Double counting of a measure
    if sum_col:
        side = left if sum_col in left.columns else right if sum_col in right.columns else None
        if side is None:
            raise SystemExit(f"error: --sum-col '{sum_col}' not in either file")
        if side is left:
            vals = pd.to_numeric(left[sum_col], errors="coerce").fillna(0)
            weight = matches.clip(lower=1) if how == "left" else matches
            before, after = float(vals.sum()), float((vals * weight).sum())
            if how == "inner":
                before = float(vals[matches > 0].sum())
            stats.update(sum_before=round(before, 4), sum_after=round(after, 4))
            if before and abs(after - before) / abs(before) > 0.001:
                f.append(finding("join.double_count", "FAIL",
                                 f"Total '{sum_col}' changes from {before:,.2f} to {after:,.2f} after the join "
                                 f"({after / before - 1:+.1%}). Aggregate before joining or sum over distinct "
                                 "left keys.", [sum_col], round(after / before - 1, 4)))
    return f, stats


def _format_examples(lk, rk, n=3):
    rmap = {}
    for v in rk.dropna().unique():
        rmap.setdefault(normalise(pd.Series([v]))[0], v)
    out = []
    for v in lk.dropna().unique():
        nv = normalise(pd.Series([v]))[0]
        if nv in rmap and rmap[nv] != v:
            out.append((v, rmap[nv]))
            if len(out) >= n:
                break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--on", help="comma-separated key columns with the same name on both sides")
    ap.add_argument("--left-on")
    ap.add_argument("--right-on")
    ap.add_argument("--how", choices=["left", "inner"], default="left")
    ap.add_argument("--expect", choices=["many-to-one", "one-to-one", "one-to-many"], default="many-to-one")
    ap.add_argument("--sum-col", help="numeric column whose total must not change because of the join")
    ap.add_argument("--min-match", type=float, default=0.95)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.on:
        lcols = rcols = [c.strip() for c in a.on.split(",")]
    elif a.left_on and a.right_on:
        lcols = [c.strip() for c in a.left_on.split(",")]
        rcols = [c.strip() for c in a.right_on.split(",")]
    else:
        raise SystemExit("error: give --on, or both --left-on and --right-on")

    left, right = read_table(a.left), read_table(a.right)
    for c in lcols:
        if c not in left.columns:
            raise SystemExit(f"error: '{c}' not in {a.left}")
    for c in rcols:
        if c not in right.columns:
            raise SystemExit(f"error: '{c}' not in {a.right}")

    findings, stats = check(left, right, lcols, rcols, a.how, a.expect, a.sum_col, a.min_match)
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "join-guard", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"join-guard: {status}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        for x in findings:
            print(f"  [{x['severity']}] {x['check']}: {x['message']}")
            if x.get("examples"):
                print(f"      examples: {x['examples']}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
