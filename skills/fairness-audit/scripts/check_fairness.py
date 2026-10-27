#!/usr/bin/env python3
"""Audit a model's decisions across groups: selection-rate disparity (the four-fifths rule), true-positive-rate gaps
(equal opportunity), false-positive-rate gaps, and groups too small to assess.

Usage examples
--------------
    python check_fairness.py --pred decisions.csv --group gender --y-pred approved
    python check_fairness.py --pred scored.csv --group gender,age_band --y-score prob --threshold 0.4 --y-true repaid

--y-pred is the decision (1 = favourable outcome, e.g. approved). Or give --y-score and --threshold.
With --y-true, error-rate gaps are checked too. Exit code: 1 if any FAIL, 0 otherwise, 2 on bad input.
"""
from __future__ import annotations

import argparse
import json
import math
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


def binary(s: pd.Series, favourable: str) -> pd.Series:
    t = s.astype(str).str.strip().str.lower()
    fav = favourable.lower()
    if fav in {"1", "true"}:
        return t.isin({"1", "1.0", "true", "yes"}).astype(int)
    return (t == fav).astype(int)


def two_prop_z(p1, n1, p2, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p), 1e-12) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se


def check(df, groups, pred, y, min_n):
    f, stats = [], {"rows": len(df)}
    for g in groups:
        col = df[g].astype(str).fillna("(missing)")
        rows = []
        for val in sorted(col.unique()):
            m = col == val
            r = {"group": val, "n": int(m.sum()), "selection_rate": float(pred[m].mean())}
            if y is not None:
                pos, neg = m & (y == 1), m & (y == 0)
                r["n_pos"], r["n_neg"] = int(pos.sum()), int(neg.sum())
                r["tpr"] = float(pred[pos].mean()) if pos.any() else float("nan")
                r["fpr"] = float(pred[neg].mean()) if neg.any() else float("nan")
            rows.append(r)
        stats[g] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]

        small = [r for r in rows if r["n"] < min_n]
        if small:
            f.append(finding("fairness.small_group", "WARN",
                             f"{len(small)} group(s) in '{g}' have fewer than {min_n} rows "
                             f"({', '.join(r['group'] for r in small[:5])}); their rates cannot be assessed reliably.",
                             [g], len(small)))
        big = [r for r in rows if r["n"] >= min_n]
        if len(big) < 2:
            continue
        lo = min(big, key=lambda r: r["selection_rate"])
        hi = max(big, key=lambda r: r["selection_rate"])
        if hi["selection_rate"] > 0:
            ratio = lo["selection_rate"] / hi["selection_rate"]
            z = two_prop_z(hi["selection_rate"], hi["n"], lo["selection_rate"], lo["n"])
            if ratio < 0.8 and abs(z) > 2:
                f.append(finding(
                    "fairness.disparate_impact", "FAIL",
                    f"In '{g}', group '{lo['group']}' receives the favourable outcome at {lo['selection_rate']:.1%} vs "
                    f"{hi['selection_rate']:.1%} for '{hi['group']}' (ratio {ratio:.2f}; below 0.80 fails the "
                    "four-fifths rule). Investigate which features drive the gap before using this model for "
                    "decisions about people.", [g], round(ratio, 3)))
        if y is None:
            continue
        for metric, label, nkey, check_name in (("tpr", "true positive rate", "n_pos", "fairness.equal_opportunity"),
                                                ("fpr", "false positive rate", "n_neg", "fairness.false_positive_gap")):
            ok = [r for r in big if r[nkey] >= min_n and r[metric] == r[metric]]
            if len(ok) < 2:
                continue
            a = min(ok, key=lambda r: r[metric])
            b = max(ok, key=lambda r: r[metric])
            gap = b[metric] - a[metric]
            z = two_prop_z(b[metric], b[nkey], a[metric], a[nkey])
            if gap > 0.1 and abs(z) > 2:
                who = "qualified people" if metric == "tpr" else "people who should not be selected"
                f.append(finding(
                    check_name, "WARN",
                    f"In '{g}', the {label} is {a[metric]:.1%} for '{a['group']}' vs {b[metric]:.1%} for '{b['group']}' "
                    f"(gap {gap:.1%}): the model treats {who} differently by group.", [g], round(gap, 4)))
    return f, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--group", required=True, help="comma-separated sensitive or protected attributes")
    ap.add_argument("--y-pred", help="decision column (favourable outcome = --favourable)")
    ap.add_argument("--y-score", help="score column; decisions are score >= --threshold")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--y-true", help="actual outcome, to compare error rates")
    ap.add_argument("--favourable", default="1", help="value meaning the favourable outcome (default 1)")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    df = read_table(a.pred)
    groups = [g.strip() for g in a.group.split(",")]
    for c in [*groups, *(x for x in (a.y_pred, a.y_score, a.y_true) if x)]:
        if c not in df.columns:
            raise SystemExit(f"error: column '{c}' not in {a.pred}")
    if a.y_pred:
        pred = binary(df[a.y_pred], a.favourable)
    elif a.y_score:
        pred = (pd.to_numeric(df[a.y_score], errors="coerce") >= a.threshold).astype(int)
    else:
        raise SystemExit("error: give --y-pred or --y-score")
    y = binary(df[a.y_true], a.favourable) if a.y_true else None

    findings, stats = check(df, groups, pred, y, a.min_n)
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "fairness-audit", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"fairness-audit: {status}")
        for k, v in stats.items():
            if isinstance(v, list):
                print(f"  {k}:")
                for r in v:
                    print("    " + ", ".join(f"{kk}={vv}" for kk, vv in r.items()))
            else:
                print(f"  {k}: {v}")
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
