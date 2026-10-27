#!/usr/bin/env python3
"""Compare the features a model was trained on with the features it receives in production (or in a batch scoring
job): missing features, type changes, unit and scaling differences, distribution shift, unseen categories, null-rate
changes, and - when the same entities appear in both - features computed differently for the same row.

Usage examples
--------------
    python check_skew.py --train train_features.csv --serve serving_log.csv
    python check_skew.py --train offline.parquet --serve online.parquet --key user_id,as_of_date --exclude label

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
        return pd.read_parquet(p)
    return pd.read_csv(p, sep="\t" if p.suffix.lower() in {".tsv", ".tab"} else ",")


def finding(check, severity, message, columns=None, value=None, examples=None):
    out = {"check": check, "severity": severity, "message": message, "columns": columns or [], "value": value}
    if examples:
        out["examples"] = examples
    return out


def numeric(s: pd.Series):
    n = pd.to_numeric(s, errors="coerce")
    return n if n.notna().sum() >= 0.95 * max(s.notna().sum(), 1) and s.notna().any() else None


def unit_ratio(a: float, b: float):
    if not a or not b or a != a or b != b:
        return None
    r = b / a
    for f in (10, 100, 1000, 1e6, 0.1, 0.01, 0.001, 1e-6):
        if abs(r / f - 1) < 0.15:
            return f
    return None


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, 1e-4, None)
    a = np.clip(actual, 1e-4, None)
    return float(((a - e) * np.log(a / e)).sum())


def numeric_psi(tr: pd.Series, sv: pd.Series) -> float:
    edges = np.unique(np.nanquantile(tr, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(tr, edges)[0] / len(tr)
    a = np.histogram(sv, edges)[0] / len(sv)
    return psi(e, a)


def looks_standardised(x: pd.Series) -> bool:
    return abs(float(x.mean())) < 0.1 and 0.8 < float(x.std()) < 1.2


def check(train, serve, features, key):
    f, stats = [], {"train_rows": len(train), "serve_rows": len(serve), "features": len(features)}
    missing = [c for c in features if c not in serve.columns]
    if missing:
        f.append(finding("skew.missing_feature", "FAIL",
                         f"{len(missing)} training features are absent at serving time: {missing[:8]}. The model will "
                         "see nulls or defaults for them.", missing, len(missing)))
    shifted = {}
    for c in [c for c in features if c in serve.columns]:
        tr, sv = train[c], serve[c]
        ntr, nsv = numeric(tr), numeric(sv)
        null_gap = float(sv.isna().mean() - tr.isna().mean())
        if abs(null_gap) > 0.1:
            f.append(finding("skew.null_rate", "WARN",
                             f"'{c}' is null in {tr.isna().mean():.0%} of training rows but {sv.isna().mean():.0%} of "
                             "serving rows. Check the serving join or default and how the model imputes it.", [c],
                             round(null_gap, 3)))
        if ntr is not None and nsv is None:
            f.append(finding("skew.type", "FAIL",
                             f"'{c}' is numeric in training but not at serving time (examples: "
                             f"{sv.dropna().astype(str).head(3).tolist()}).", [c]))
            continue
        if ntr is not None:
            a, b = ntr.dropna(), nsv.dropna()
            if a.empty or b.empty:
                continue
            ratio = unit_ratio(float(a.abs().median()), float(b.abs().median()))
            if ratio:
                f.append(finding(
                    "skew.scale", "FAIL",
                    f"'{c}' is about x{ratio:g} larger at serving time (median {a.median():,.4g} in training vs "
                    f"{b.median():,.4g} serving). A unit difference (cents vs dollars, ms vs s) - the model's "
                    "thresholds are off by that factor.", [c], ratio))
                continue
            if looks_standardised(a) != looks_standardised(b) and a.std() > 0 and b.std() > 0:
                which = "training" if looks_standardised(a) else "serving"
                f.append(finding(
                    "skew.scaling", "FAIL",
                    f"'{c}' is standardised (mean ~0, sd ~1) in {which} data only (training mean {a.mean():.3g}, "
                    f"sd {a.std():.3g}; serving mean {b.mean():.3g}, sd {b.std():.3g}). The scaler is applied on one "
                    "path but not the other. Ship the fitted preprocessing with the model.", [c]))
                continue
            p = numeric_psi(a.to_numpy(float), b.to_numpy(float))
        else:
            ta, sa = tr.dropna().astype(str), sv.dropna().astype(str)
            if ta.empty or sa.empty:
                continue
            unseen = ~sa.isin(set(ta))
            if unseen.mean() > 0.05:
                f.append(finding(
                    "skew.unseen_categories", "WARN",
                    f"{unseen.mean():.0%} of serving values of '{c}' never appeared in training "
                    f"({', '.join(map(repr, sa[unseen].value_counts().head(4).index))}). The encoder maps them to "
                    "'unknown' or fails. Check for renamed codes or new categories.", [c], round(float(unseen.mean()), 3)))
            cats = ta.value_counts(normalize=True)
            sc = sa.value_counts(normalize=True)
            idx = cats.index.union(sc.index)
            p = psi(cats.reindex(idx, fill_value=0).to_numpy(), sc.reindex(idx, fill_value=0).to_numpy())
        shifted[c] = round(p, 3)
        if p > 0.25:
            f.append(finding("skew.distribution", "WARN",
                             f"'{c}' has shifted strongly between training and serving (PSI {p:.2f}; above 0.25 is "
                             "a major shift). Either the population changed or the feature is computed differently.",
                             [c], round(p, 3)))
    stats["psi"] = shifted

    if key:
        if any(k not in train.columns or k not in serve.columns for k in key):
            raise SystemExit(f"error: key {key} must exist in both files")
        tk = train.assign(**{k: train[k].astype(str).str.strip() for k in key})
        sk = serve.assign(**{k: serve[k].astype(str).str.strip() for k in key})
        m = tk.drop_duplicates(key).merge(sk.drop_duplicates(key), on=key, suffixes=("_train", "_serve"))
        stats["rows_matched_on_key"] = len(m)
        for c in [c for c in features if c in serve.columns and c not in key]:
            x, y = m[f"{c}_train"], m[f"{c}_serve"]
            nx, ny = numeric(x), numeric(y)
            if nx is not None and ny is not None:
                diff = ~(np.isclose(nx, ny, rtol=1e-3, atol=1e-6) | (nx.isna() & ny.isna()))
            else:
                diff = ~((x.astype(str) == y.astype(str)) | (x.isna() & y.isna()))
            share = float(diff.mean()) if len(m) else 0.0
            if share > 0.01:
                ex = [f"{tuple(r)}: {p!r} vs {q!r}" for r, p, q in
                      zip(m.loc[diff, key].head(3).itertuples(index=False), x[diff].head(3), y[diff].head(3))]
                f.append(finding(
                    "skew.value_mismatch", "FAIL",
                    f"For the same {key}, '{c}' differs between training and serving in {share:.0%} of {len(m):,} "
                    "matched rows. The offline and online feature logic are not the same (window length, time zone, "
                    "join, or point-in-time cut-off).", [c], round(share, 4), ex))
    return f, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="features as used for training")
    ap.add_argument("--serve", required=True, help="features as received at serving/scoring time")
    ap.add_argument("--features", help="comma-separated features (default: every training column except key/exclude)")
    ap.add_argument("--key", help="comma-separated entity key to compare the same rows on both sides")
    ap.add_argument("--exclude", default="", help="comma-separated columns to skip (label, ids, timestamps)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    train, serve = read_table(a.train), read_table(a.serve)
    key = [k.strip() for k in a.key.split(",")] if a.key else []
    exclude = {c.strip() for c in a.exclude.split(",") if c.strip()} | set(key)
    features = [c.strip() for c in a.features.split(",")] if a.features else [c for c in train.columns if c not in exclude]
    for c in features:
        if c not in train.columns:
            raise SystemExit(f"error: feature '{c}' not in {a.train}")

    findings, stats = check(train, serve, features, key)
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "train-serve-skew", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"train-serve-skew: {status}")
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
