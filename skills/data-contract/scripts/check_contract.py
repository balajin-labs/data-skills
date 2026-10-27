#!/usr/bin/env python3
"""Validate a dataset against a data contract (expected schema, types, nullability, ranges, allowed values, patterns,
primary key, row count and freshness), or infer a starting contract from data you trust.

Usage examples
--------------
    python check_contract.py --data orders.csv --contract orders.contract.json
    python check_contract.py --data orders.csv --infer > orders.contract.json      # then edit by hand

Contract format (JSON):
    {
      "primary_key": ["order_id"],
      "columns": {
        "order_id":   {"type": "integer", "nullable": false},
        "amount":     {"type": "number", "min": 0, "max": 100000},
        "status":     {"type": "string", "allowed": ["paid", "refunded", "cancelled"]},
        "email":      {"type": "string", "pattern": "^[^@\\\\s]+@[^@\\\\s]+$", "nullable": true},
        "order_date": {"type": "date", "max": "today"}
      },
      "row_count": {"min": 1000},
      "freshness": {"column": "order_date", "max_age_days": 2},
      "allow_extra_columns": true
    }
Types: integer, number, string, boolean, date, datetime. Column keys: type, nullable (default true), required (default
true), min, max, allowed, pattern, unique, max_length. Exit code: 1 if any FAIL, 0 otherwise, 2 on bad input.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

BOOL = {"true", "false", "t", "f", "1", "0", "yes", "no", "y", "n"}


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: file not found: {path}")
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
        return df.astype(str).where(df.notna())
    return pd.read_csv(p, sep="\t" if p.suffix.lower() in {".tsv", ".tab"} else ",", dtype=str)


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


def parse(s: pd.Series, typ: str):
    """Return (parsed values, mask of non-null values that failed to parse)."""
    s = s.str.strip()
    nn = s.notna()
    if typ in ("integer", "number"):
        v = pd.to_numeric(s, errors="coerce")
        bad = nn & v.isna()
        if typ == "integer":
            bad |= v.notna() & (v != v.round())
        return v, bad
    if typ in ("date", "datetime"):
        v = to_dates(s)
        return v, nn & v.isna()
    if typ == "boolean":
        return s.str.lower(), nn & ~s.str.lower().isin(BOOL)
    return s, pd.Series(False, index=s.index)


def bound(value, typ, as_of):
    if typ in ("date", "datetime"):
        return as_of if str(value).lower() == "today" else pd.Timestamp(value)
    return float(value)


def ex(values, n=5):
    return [str(v) for v in pd.Series(values).drop_duplicates().head(n)]


def validate(df, contract, as_of):
    f, stats = [], {"rows": len(df), "columns_in_contract": len(contract.get("columns", {}))}
    spec = contract.get("columns", {})

    for col, rules in spec.items():
        if col not in df.columns:
            if rules.get("required", True):
                f.append(finding("contract.missing_column", "FAIL", f"Required column '{col}' is missing.", [col]))
            continue
        s = df[col]
        typ = rules.get("type", "string")
        v, bad = parse(s, typ)
        if bad.any():
            f.append(finding("contract.type", "FAIL",
                             f"'{col}' should be {typ} but {int(bad.sum()):,} values are not.", [col], int(bad.sum()),
                             ex(s[bad])))
        if not rules.get("nullable", True) and s.isna().any():
            f.append(finding("contract.nullable", "FAIL", f"'{col}' must not be null but {int(s.isna().sum()):,} rows "
                             "are.", [col], int(s.isna().sum())))
        ok = ~bad & v.notna()
        for side in ("min", "max"):
            if side in rules:
                b = bound(rules[side], typ, as_of)
                out = ok & ((v < b) if side == "min" else (v > b))
                if out.any():
                    f.append(finding(
                        "contract.range", "FAIL",
                        f"{int(out.sum()):,} values of '{col}' are {'below the minimum' if side == 'min' else 'above the maximum'} "
                        f"{rules[side]!r}.", [col], int(out.sum()), ex(s[out])))
        if "allowed" in rules:
            allowed = {str(x) for x in rules["allowed"]}
            out = s.notna() & ~s.str.strip().isin(allowed)
            if out.any():
                f.append(finding(
                    "contract.allowed_values", "FAIL",
                    f"'{col}' has {int(out.sum()):,} rows with values outside the allowed set: "
                    f"{', '.join(repr(x) for x in ex(s[out].str.strip()))}. A new upstream code must be mapped "
                    "deliberately, not passed through.", [col], int(out.sum()), ex(s[out].str.strip())))
        if "pattern" in rules:
            rx = re.compile(rules["pattern"])
            out = s.notna() & ~s.fillna("").map(lambda x: bool(rx.search(x)))
            if out.any():
                f.append(finding("contract.pattern", "FAIL",
                                 f"{int(out.sum()):,} values of '{col}' do not match {rules['pattern']!r}.", [col],
                                 int(out.sum()), ex(s[out])))
        if "max_length" in rules:
            out = s.fillna("").str.len() > int(rules["max_length"])
            if out.any():
                f.append(finding("contract.max_length", "FAIL",
                                 f"{int(out.sum()):,} values of '{col}' exceed {rules['max_length']} characters.",
                                 [col], int(out.sum())))
        if rules.get("unique") and s.dropna().duplicated().any():
            d = s.dropna()
            f.append(finding("contract.unique", "FAIL", f"'{col}' must be unique but {int(d.duplicated().sum()):,} "
                             "values repeat.", [col], int(d.duplicated().sum()), ex(d[d.duplicated()])))

    extra = [c for c in df.columns if c not in spec]
    if extra and spec and not contract.get("allow_extra_columns", True):
        f.append(finding("contract.extra_columns", "FAIL", f"Columns not in the contract: {extra}.", extra))
    elif extra and spec:
        stats["extra_columns"] = extra

    pk = contract.get("primary_key") or []
    if pk:
        missing = [c for c in pk if c not in df.columns]
        if missing:
            f.append(finding("contract.primary_key", "FAIL", f"Primary key columns missing: {missing}.", missing))
        elif len(df):
            k = df[pk].astype(str).agg("|".join, axis=1).where(df[pk].notna().all(axis=1))
            dup, null = int(k.dropna().duplicated().sum()), int(k.isna().sum())
            if dup or null:
                f.append(finding("contract.primary_key", "FAIL",
                                 f"Primary key {pk} has {dup:,} duplicated and {null:,} null values.", pk, dup + null,
                                 ex(k[k.duplicated()].dropna())))

    rc = contract.get("row_count") or {}
    if "min" in rc and len(df) < rc["min"]:
        f.append(finding("contract.row_count", "FAIL", f"{len(df):,} rows; the contract expects at least {rc['min']:,}. "
                         "A partial extract or a failed upstream job is likely.", [], len(df)))
    if "max" in rc and len(df) > rc["max"]:
        f.append(finding("contract.row_count", "FAIL", f"{len(df):,} rows; the contract expects at most {rc['max']:,}. "
                         "Look for duplicate loads or a missing filter.", [], len(df)))

    fr = contract.get("freshness")
    if fr:
        col = fr["column"]
        if col not in df.columns:
            f.append(finding("contract.freshness", "FAIL", f"Freshness column '{col}' is missing.", [col]))
        else:
            latest = to_dates(df[col]).max()
            if pd.isna(latest):
                f.append(finding("contract.freshness", "FAIL", f"Freshness column '{col}' has no valid dates, so "
                                 "freshness cannot be confirmed.", [col]))
                return f, stats
            ref = pd.Timestamp(fr["as_of"]) if fr.get("as_of") else as_of
            age = (ref - latest).total_seconds() / 86400 if pd.notna(latest) else float("inf")
            stats["latest_" + col] = None if pd.isna(latest) else str(latest)
            if age > fr.get("max_age_days", 1):
                f.append(finding(
                    "contract.freshness", "FAIL",
                    f"The newest '{col}' is {latest:%Y-%m-%d} ({age:.1f} days before {ref:%Y-%m-%d}); the contract allows "
                    f"{fr.get('max_age_days', 1)} days. The upstream feed has probably stopped.", [col], round(age, 1)))
    return f, stats


def infer(df):
    cols, pk = {}, []
    for c in df.columns:
        s = df[c].dropna().str.strip()
        rule = {}
        if s.empty:
            cols[c] = {"type": "string"}
            continue
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().all():
            rule["type"] = "integer" if (num == num.round()).all() else "number"
            rule["min"] = float(num.min()) if rule["type"] == "number" else int(num.min())
            rule["max"] = float(num.max()) if rule["type"] == "number" else int(num.max())
        elif s.str.lower().isin(BOOL).all():
            rule["type"] = "boolean"
        elif to_dates(s).notna().all() and s.str.contains(r"\d{4}", regex=True).all():
            rule["type"] = "datetime" if s.str.contains(":").any() else "date"
            rule["max"] = "today"
        else:
            rule["type"] = "string"
            if s.nunique() <= 20 and len(s) >= 50:
                rule["allowed"] = sorted(s.unique().tolist())
        rule["nullable"] = bool(df[c].isna().any())
        if not pk and not df[c].isna().any() and df[c].is_unique and re.search(r"(^id$|_id$|_key$|^key$)", c, re.I):
            pk = [c]
            rule["unique"] = True
        cols[c] = rule
    return {"primary_key": pk, "columns": cols, "row_count": {"min": max(1, int(len(df) * 0.5))},
            "allow_extra_columns": True}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--contract", help="JSON contract to validate against")
    ap.add_argument("--infer", action="store_true", help="print a starting contract inferred from the data")
    ap.add_argument("--as-of", help="reference date for 'today' and freshness (default: now)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    df = read_table(a.data)
    if a.infer:
        print(json.dumps(infer(df), indent=2))
        return 0
    if not a.contract:
        raise SystemExit("error: give --contract <file> or --infer")
    cp = Path(a.contract)
    if not cp.exists():
        raise SystemExit(f"error: file not found: {a.contract}")
    try:
        contract = json.loads(cp.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: contract is not valid JSON: {e}")
    as_of = pd.Timestamp(a.as_of) if a.as_of else pd.Timestamp.now()

    findings, stats = validate(df, contract, as_of)
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "data-contract", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"data-contract: {status}")
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
