#!/usr/bin/env python3
"""Reconcile two copies of the same data (source system vs warehouse, old pipeline vs new, report vs recomputation):
row counts, missing and extra keys, duplicate keys, column-by-column value mismatches and measure totals.

Usage examples
--------------
    python check_reconcile.py --source erp_orders.csv --target dw_orders.csv --key order_id
    python check_reconcile.py --source old.parquet --target new.parquet --key order_id,line_no \
        --measure amount,quantity --by order_date

Values are compared after trimming whitespace; numbers are compared with a relative tolerance (--rtol).
Exit code: 1 if any FAIL finding, 0 otherwise, 2 on bad input. Add --json for structured output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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


def key_series(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    k = df[keys].apply(lambda s: s.str.strip())
    return k[keys[0]] if len(keys) == 1 else k.astype(str).agg("|".join, axis=1)


def as_number(s: pd.Series):
    n = pd.to_numeric(s.str.replace(r"[,$\s]", "", regex=True), errors="coerce")
    return n if n.notna().sum() >= 0.95 * max(s.notna().sum(), 1) else None


def check(src, tgt, keys, measures, by, columns, rtol, atol):
    f, stats = [], {"source_rows": len(src), "target_rows": len(tgt)}
    src, tgt = src.copy(), tgt.copy()
    src["_k"], tgt["_k"] = key_series(src, keys), key_series(tgt, keys)

    for name, df in (("source", src), ("target", tgt)):
        dup = df["_k"].duplicated(keep=False)
        if dup.any():
            f.append(finding(
                "reconcile.duplicate_keys", "FAIL",
                f"{int(df['_k'].duplicated().sum()):,} {name} rows repeat a key {keys}. Either the key is wrong or "
                f"the {name} has duplicate loads; matching is ambiguous until this is fixed.", keys,
                int(dup.sum()), df.loc[dup, "_k"].drop_duplicates().head(5).tolist()))

    if len(src) != len(tgt):
        f.append(finding(
            "reconcile.row_count", "FAIL",
            f"Row counts differ: source {len(src):,}, target {len(tgt):,} ({len(tgt) - len(src):+,}).", [],
            len(tgt) - len(src)))

    m = src.drop_duplicates("_k").merge(tgt.drop_duplicates("_k"), on="_k", how="outer",
                                        suffixes=("_src", "_tgt"), indicator=True)
    missing, extra = m[m["_merge"] == "left_only"], m[m["_merge"] == "right_only"]
    stats.update(matched_keys=int((m["_merge"] == "both").sum()), missing_in_target=len(missing),
                 extra_in_target=len(extra))

    def where(rows, side):
        if not by:
            return rows["_k"].head(5).tolist()
        col = f"{by}_{side}" if f"{by}_{side}" in rows.columns else by
        counts = rows[col].value_counts().head(5)
        return [f"{by}={k}: {v:,} rows" for k, v in counts.items()]

    if len(missing):
        f.append(finding(
            "reconcile.missing_in_target", "FAIL",
            f"{len(missing):,} source keys ({len(missing) / max(len(src), 1):.1%}) are not in the target"
            + (f", concentrated in: {', '.join(where(missing, 'src')[:3])}" if by else "") +
            ". Look for a failed partition, a filter in the load, or late-arriving data.",
            keys, len(missing), where(missing, "src")))
    if len(extra):
        f.append(finding(
            "reconcile.extra_in_target", "FAIL",
            f"{len(extra):,} target keys are not in the source (deleted upstream, test data, or a different "
            "extraction window).", keys, len(extra), where(extra, "tgt")))

    both = m[m["_merge"] == "both"]
    common = [c for c in src.columns if c in tgt.columns and c not in keys and c != "_k"]
    cols = columns or common
    mismatched = {}
    for c in cols:
        if c not in common:
            f.append(finding("reconcile.missing_column", "FAIL", f"Column '{c}' is not present on both sides.", [c]))
            continue
        a, b = both[f"{c}_src"].str.strip(), both[f"{c}_tgt"].str.strip()
        na, nb = as_number(a), as_number(b)
        if na is not None and nb is not None:
            diff = ~(np.isclose(na, nb, rtol=rtol, atol=atol) | (na.isna() & nb.isna()))
        else:
            diff = ~((a == b) | (a.isna() & b.isna()))
        n_diff = int(diff.sum())
        if n_diff:
            mismatched[c] = n_diff
            ex = [f"{k}: {x!r} vs {y!r}" for k, x, y in zip(both.loc[diff, "_k"].head(4), a[diff].head(4),
                                                          b[diff].head(4))]
            hint = ""
            if na is not None and nb is not None:
                ratio = (nb[diff] / na[diff]).replace([np.inf, -np.inf], np.nan).dropna()
                if len(ratio) and ratio.round(6).nunique() == 1 and ratio.iloc[0] not in (0, 1):
                    hint = f" Every mismatch is exactly x{ratio.iloc[0]:g}: a unit or currency change."
                elif len(ratio) and (na[diff] - nb[diff]).abs().max() < 1:
                    hint = " All differences are below 1: rounding or truncation in one system."
            f.append(finding(
                "reconcile.value_mismatch", "FAIL",
                f"'{c}' differs for {n_diff:,} of {len(both):,} matched keys ({n_diff / max(len(both), 1):.1%}).{hint}",
                [c], n_diff, ex))
    stats["columns_compared"] = len([c for c in cols if c in common])
    stats["columns_with_mismatches"] = mismatched

    for c in measures:
        if c not in src.columns or c not in tgt.columns:
            raise SystemExit(f"error: measure '{c}' must exist in both files")
        sa = pd.to_numeric(src[c].str.replace(r"[,$\s]", "", regex=True), errors="coerce").sum()
        sb = pd.to_numeric(tgt[c].str.replace(r"[,$\s]", "", regex=True), errors="coerce").sum()
        stats[f"total_{c}"] = {"source": round(float(sa), 4), "target": round(float(sb), 4)}
        if not np.isclose(sa, sb, rtol=rtol, atol=atol):
            f.append(finding(
                "reconcile.measure_total", "FAIL",
                f"Total '{c}' differs: source {sa:,.2f}, target {sb:,.2f} ({(sb - sa) / sa if sa else float('nan'):+.2%}).",
                [c], round(float(sb - sa), 4)))
    return f, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--key", required=True, help="comma-separated key columns")
    ap.add_argument("--measure", default="", help="comma-separated numeric columns whose totals must match")
    ap.add_argument("--by", help="column to group missing keys by, e.g. a date or partition")
    ap.add_argument("--columns", help="only compare these columns (default: all shared columns)")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    src, tgt = read_table(a.source), read_table(a.target)
    keys = [k.strip() for k in a.key.split(",")]
    for k in keys:
        if k not in src.columns or k not in tgt.columns:
            raise SystemExit(f"error: key '{k}' must exist in both files")
    if a.by and (a.by not in src.columns or a.by not in tgt.columns):
        raise SystemExit(f"error: --by '{a.by}' must exist in both files")
    measures = [c.strip() for c in a.measure.split(",") if c.strip()]
    columns = [c.strip() for c in a.columns.split(",")] if a.columns else None

    findings, stats = check(src, tgt, keys, measures, a.by, columns, a.rtol, a.atol)
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "reconcile", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"reconcile: {status}")
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
