#!/usr/bin/env python3
"""Check that a pipeline step is idempotent and deterministic: run it twice on the same input and compare the outputs.
Catches reruns that append instead of overwrite, keys that change between runs, and values that depend on row order,
random sampling or the wall clock.

Usage examples
--------------
    python check_idempotency.py --first run1.csv --second run2.csv --key order_id
    python check_idempotency.py --first run1.parquet --second run2.parquet --key customer_id,day --ignore loaded_at

--first is the output after one run; --second is the output after running the same step again on the same input
(for an incremental load: after re-running the same partition). Exit code: 1 if any FAIL, 0 otherwise, 2 on bad input.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METADATA = re.compile(r"(load|loaded|ingest|insert|inserted|updated|processed|etl|batch|run|job)(_at|_ts|_time|_id|ed_at)?$|"
                      r"^_?(etl|dbt|airflow|meta)_", re.I)


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


def keyed(df, keys):
    return df[keys[0]] if len(keys) == 1 else df[keys].astype(str).agg("|".join, axis=1)


def check(a, b, keys, ignore):
    f, stats = [], {"first_rows": len(a), "second_rows": len(b)}
    ka, kb = keyed(a, keys), keyed(b, keys)
    dup_a, dup_b = int(ka.duplicated().sum()), int(kb.duplicated().sum())
    stats.update(first_duplicate_keys=dup_a, second_duplicate_keys=dup_b)

    if dup_b > dup_a:
        exact = int(b.duplicated().sum())
        appended = len(b) >= 2 * len(a) * 0.99 and exact >= 0.9 * len(a)
        f.append(finding(
            "idempotency.duplicated_on_rerun", "FAIL",
            f"The second run has {dup_b - dup_a:,} more duplicated keys than the first ({len(a):,} -> {len(b):,} rows)."
            + (" The rerun appended a full copy of the output instead of replacing it. " if appended else " ")
            + "Use overwrite, MERGE/upsert on the key, or delete-then-insert for the partition being loaded.",
            keys, dup_b - dup_a, kb[kb.duplicated()].head(5).tolist()))
    elif dup_a:
        f.append(finding("idempotency.key_not_unique", "WARN",
                         f"The output is not unique on {keys} ({dup_a:,} duplicated keys in the first run), so rows "
                         "cannot be compared one-to-one. Check the key.", keys, dup_a))

    if len(a) != len(b):
        f.append(finding("idempotency.row_count", "FAIL",
                         f"Row count changed between identical runs: {len(a):,} -> {len(b):,}.", [], len(b) - len(a)))

    sa, sb = set(ka), set(kb)
    only_a, only_b = sa - sb, sb - sa
    if only_a or only_b:
        f.append(finding(
            "idempotency.key_set", "FAIL",
            f"The set of keys changed between runs: {len(only_a):,} disappeared and {len(only_b):,} appeared. "
            "Look for sampling, LIMIT without ORDER BY, time-relative filters (now(), current_date) or "
            "non-deterministic de-duplication.", keys, len(only_a) + len(only_b),
            [f"only in first: {k}" for k in list(only_a)[:3]] + [f"only in second: {k}" for k in list(only_b)[:3]]))

    a1 = a.assign(_k=ka).drop_duplicates("_k")
    b1 = b.assign(_k=kb).drop_duplicates("_k")
    m = a1.merge(b1, on="_k", suffixes=("_1", "_2"))
    cols = [c for c in a.columns if c in b.columns and c not in keys and c not in ignore]
    changed = {}
    for c in cols:
        x, y = m[f"{c}_1"], m[f"{c}_2"]
        nx, ny = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        if nx.notna().sum() >= 0.95 * max(x.notna().sum(), 1) and ny.notna().sum() >= 0.95 * max(y.notna().sum(), 1):
            diff = ~(np.isclose(nx, ny, rtol=1e-9, atol=1e-12) | (nx.isna() & ny.isna()))
        else:
            diff = ~((x == y) | (x.isna() & y.isna()))
        n = int(diff.sum())
        if not n:
            continue
        changed[c] = n
        ex = [f"{k}: {p!r} -> {q!r}" for k, p, q in zip(m.loc[diff, "_k"].head(4), x[diff].head(4), y[diff].head(4))]
        if METADATA.search(c):
            f.append(finding(
                "idempotency.metadata_changed", "WARN",
                f"'{c}' differs for {n:,} keys; it looks like load metadata, which is expected to change. "
                f"Pass --ignore {c} if so.", [c], n, ex))
        else:
            f.append(finding(
                "idempotency.nondeterministic", "FAIL",
                f"'{c}' has different values for {n:,} of {len(m):,} keys ({n / max(len(m), 1):.1%}) across two runs "
                "on the same input. Typical causes: ROW_NUMBER()/first()/drop_duplicates without a total order, "
                "random sampling without a seed, now()/current_date in the logic, or reading a source that changed.",
                [c], n, ex))
    stats["changed_columns"] = changed
    return f, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", required=True, help="output of the first run")
    ap.add_argument("--second", required=True, help="output of re-running the same step on the same input")
    ap.add_argument("--key", required=True, help="comma-separated key columns of the output")
    ap.add_argument("--ignore", default="", help="comma-separated columns allowed to differ (load timestamps, run ids)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    first, second = read_table(a.first), read_table(a.second)
    keys = [k.strip() for k in a.key.split(",")]
    for k in keys:
        if k not in first.columns or k not in second.columns:
            raise SystemExit(f"error: key '{k}' must exist in both files")
    ignore = {c.strip() for c in a.ignore.split(",") if c.strip()}

    findings, stats = check(first, second, keys, ignore)
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "pipeline-idempotency", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"pipeline-idempotency: {status}")
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
