#!/usr/bin/env python3
"""Compare a new data extract with a trusted baseline and flag silent breakage:
schema changes, type changes, null spikes, unit changes, new categories, duplicates and sentinel values.

Usage examples
--------------
Compare this month's extract with last month's:
    python check_drift.py --baseline sales_2026_09.csv --current sales_2026_10.csv --key order_id

Profile a single file (no baseline) for sentinels and mixed types:
    python check_drift.py --current sales_2026_10.csv

Exit code: 1 if any FAIL finding, 0 otherwise, 2 on bad input. Add --json for structured output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NULL_TOKENS = {"", "null", "none", "nan", "n/a", "na", "#n/a", "-", "?"}
NUMERIC_SENTINELS = {-999.0, -9999.0, -99999.0, 9999.0, 99999.0, 999999.0, -1.0e9}
DATE_SENTINELS = {"1900-01-01", "1970-01-01", "9999-12-31", "0000-00-00", "2099-12-31"}
FORMATTED_NUM = re.compile(r"^\s*[-+]?[$€£₹]?\s*[\d,]+(\.\d+)?\s*%?\s*$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def read_raw(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: file not found: {path}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p).astype("string")
    return pd.read_csv(p, sep="\t" if p.suffix.lower() in {".tsv", ".tab"} else ",",
                       dtype=str, keep_default_na=False)


def finding(check, severity, message, columns=None, value=None):
    return {"check": check, "severity": severity, "message": message, "columns": columns or [], "value": value}


def clean_nulls(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    return s.mask(s.str.strip().str.lower().isin(NULL_TOKENS))


def profile_column(raw: pd.Series) -> dict:
    s = clean_nulls(raw)
    nn = s.dropna()
    info = {"null_rate": float(s.isna().mean()), "kind": "empty", "n": int(len(s))}
    if nn.empty:
        return info
    num = pd.to_numeric(nn, errors="coerce")
    num_share = float(num.notna().mean())
    fmt_share = float(nn.str.match(FORMATTED_NUM).mean())
    date_share = float(nn.str.match(ISO_DATE).mean())
    info.update(numeric_share=num_share, formatted_share=fmt_share, date_share=date_share)
    if num_share >= 0.98:
        info["kind"] = "numeric"
        vals = num.dropna().astype(float)
        info.update(median=float(vals.median()), q1=float(vals.quantile(0.25)), q3=float(vals.quantile(0.75)),
                    sentinel_share=float(vals.isin(NUMERIC_SENTINELS).mean()),
                    # unique whole numbers on (almost) every row: an id or sequence, not a measure
                    identifier=bool(vals.nunique() > 0.95 * len(vals) and (vals % 1 == 0).all()))
    elif fmt_share >= 0.98:
        info["kind"] = "formatted_numeric"   # e.g. '1,234.50', '$12', '15%'
    elif date_share >= 0.98:
        info["kind"] = "date"
        info["sentinel_share"] = float(nn.str[:10].isin(DATE_SENTINELS).mean())
    else:
        info["kind"] = "text"
        if 0.05 < num_share < 0.98:
            info["mixed"] = True
        vc = nn.value_counts(normalize=True)
        info["n_unique"] = int(len(vc))
        if len(vc) <= 50:
            info["categories"] = vc.to_dict()
    return info


def unit_ratio(a: float, b: float):
    if a == 0 or b == 0 or np.sign(a) != np.sign(b):
        return None
    r = abs(b / a)
    for factor in (10, 100, 1000, 1_000_000):
        for target in (factor, 1 / factor):
            if abs(r / target - 1) < 0.2:
                return target
    return None


def single_file_checks(name: str, df: pd.DataFrame, profiles: dict) -> list:
    f = []
    for col, p in profiles.items():
        if p.get("sentinel_share", 0) > 0.01:
            f.append(finding("profile.sentinel", "WARN",
                             f"{name}: {p['sentinel_share']:.1%} of '{col}' are placeholder values "
                             "(e.g. -999, 9999, 1900-01-01). Treat them as missing before computing statistics.",
                             [col], round(p["sentinel_share"], 4)))
        if p.get("mixed"):
            f.append(finding("profile.mixed_types", "WARN",
                             f"{name}: '{col}' mixes numbers ({p['numeric_share']:.0%}) and text. "
                             "Numeric operations will silently drop or miscast values.",
                             [col], round(p["numeric_share"], 4)))
        if p["kind"] == "formatted_numeric":
            f.append(finding("profile.formatted_numbers", "WARN",
                             f"{name}: '{col}' holds numbers stored as formatted text (commas, currency or %). "
                             "Parse them explicitly before arithmetic.", [col]))
    return f


def compare(base: pd.DataFrame, cur: pd.DataFrame, key: str | None, ignore: list[str]):
    f, summary = [], {"baseline_rows": len(base), "current_rows": len(cur)}
    bcols = [c for c in base.columns if c not in ignore]
    ccols = [c for c in cur.columns if c not in ignore]

    removed = [c for c in bcols if c not in ccols]
    added = [c for c in ccols if c not in bcols]
    if removed:
        f.append(finding("drift.schema", "FAIL", f"Columns missing from the new extract: {removed}.", removed))
    if added:
        f.append(finding("drift.schema", "WARN", f"New columns in the new extract: {added}.", added))

    if len(base) and abs(len(cur) / len(base) - 1) > 0.5:
        f.append(finding("drift.row_count", "WARN",
                         f"Row count changed from {len(base):,} to {len(cur):,} ({len(cur) / len(base) - 1:+.0%}).",
                         value=round(len(cur) / len(base) - 1, 4)))

    bprof = {c: profile_column(base[c]) for c in bcols}
    cprof = {c: profile_column(cur[c]) for c in ccols}
    summary["columns"] = {c: {"baseline_kind": bprof[c]["kind"], "current_kind": cprof[c]["kind"],
                              "baseline_null": round(bprof[c]["null_rate"], 4),
                              "current_null": round(cprof[c]["null_rate"], 4)}
                          for c in bcols if c in cprof}

    for col in [c for c in bcols if c in cprof]:
        b, c = bprof[col], cprof[col]
        # type change
        if b["kind"] != c["kind"] and "empty" not in (b["kind"], c["kind"]):
            f.append(finding("drift.type_change", "FAIL",
                             f"'{col}' changed from {b['kind']} to {c['kind']}. Downstream casts or "
                             "aggregations will fail or silently produce nulls.", [col]))
        # null spike
        delta = c["null_rate"] - b["null_rate"]
        if delta > 0.20:
            f.append(finding("drift.null_spike", "FAIL",
                             f"'{col}' null rate jumped from {b['null_rate']:.1%} to {c['null_rate']:.1%}.",
                             [col], round(delta, 4)))
        elif delta > 0.05:
            f.append(finding("drift.null_spike", "WARN",
                             f"'{col}' null rate rose from {b['null_rate']:.1%} to {c['null_rate']:.1%}.",
                             [col], round(delta, 4)))
        # numeric scale / unit change
        if b["kind"] == c["kind"] == "numeric" and col != key and not (b["identifier"] or c["identifier"]):
            ratio = unit_ratio(b["median"], c["median"])
            iqr = max(b["q3"] - b["q1"], 1e-12)
            if ratio is not None and ratio != 1:
                label = f"x{ratio:g}" if ratio >= 1 else f"/{1 / ratio:g}"
                f.append(finding("drift.unit_change", "FAIL",
                                 f"'{col}' median moved from {b['median']:g} to {c['median']:g} ({label}). "
                                 "This looks like a unit change (cents vs dollars, grams vs kilograms, % vs fraction).",
                                 [col], ratio))
            elif abs(c["median"] - b["median"]) / iqr > 1.0:
                f.append(finding("drift.distribution_shift", "WARN",
                                 f"'{col}' median moved from {b['median']:g} to {c['median']:g}, "
                                 "more than one baseline IQR.", [col],
                                 round(abs(c["median"] - b["median"]) / iqr, 4)))
        # categories
        if b.get("categories") and c["kind"] == "text":
            cur_vals = clean_nulls(cur[col]).dropna().value_counts(normalize=True)
            new_share = float(cur_vals[~cur_vals.index.isin(list(b["categories"]))].sum())
            gone = [k for k in b["categories"] if k not in cur_vals.index]
            gone_share = float(sum(b["categories"][k] for k in gone))
            if new_share > 0.05:
                new_vals = [str(v) for v in cur_vals[~cur_vals.index.isin(list(b["categories"]))].index[:5]]
                f.append(finding("drift.new_categories", "WARN",
                                 f"{new_share:.1%} of '{col}' values are categories never seen in the baseline, "
                                 f"e.g. {new_vals}. Mappings and filters may miss them.", [col], round(new_share, 4)))
            if gone_share > 0.05:
                f.append(finding("drift.missing_categories", "WARN",
                                 f"Categories covering {gone_share:.1%} of baseline '{col}' disappeared: {gone[:5]}.",
                                 [col], round(gone_share, 4)))

    if key:
        if key not in base.columns or key not in cur.columns:
            raise SystemExit(f"error: key '{key}' must exist in both files")
        bd = float(base[key].duplicated().mean())
        cd = float(cur[key].duplicated().mean())
        summary.update(baseline_duplicate_key_share=round(bd, 4), current_duplicate_key_share=round(cd, 4))
        if cd - bd > 0.01:
            f.append(finding("drift.duplicates", "FAIL",
                             f"{cd:.1%} of rows now share a '{key}' with another row (baseline {bd:.1%}). "
                             "A load may have been run twice or a join fanned out upstream.", [key], round(cd - bd, 4)))

    f += single_file_checks("current", cur, cprof)
    return f, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", help="trusted earlier extract")
    ap.add_argument("--current", required=True, help="new extract to check")
    ap.add_argument("--key", help="column that should be unique per row")
    ap.add_argument("--ignore", nargs="*", default=[])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    cur = read_raw(a.current)
    if a.baseline:
        base = read_raw(a.baseline)
        findings, summary = compare(base, cur, a.key, a.ignore)
    else:
        prof = {c: profile_column(cur[c]) for c in cur.columns if c not in a.ignore}
        findings, summary = single_file_checks("current", cur, prof), {"current_rows": len(cur)}
        if a.key and a.key in cur.columns:
            share = float(cur[a.key].duplicated().mean())
            summary["duplicate_key_share"] = round(share, 4)
            if share > 0:
                findings.append(finding("profile.duplicate_keys", "FAIL",
                                        f"{share:.1%} of rows repeat a '{a.key}' value that should be unique.",
                                        [a.key], round(share, 4)))

    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "null-type-drift", "status": status, "summary": summary, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"null-type-drift: {status}  (rows: {summary.get('baseline_rows', '-')} -> {summary.get('current_rows')})")
        for x in findings:
            print(f"  [{x['severity']}] {x['check']}: {x['message']}")
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
