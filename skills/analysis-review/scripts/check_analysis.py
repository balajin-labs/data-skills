#!/usr/bin/env python3
"""Review a group comparison before it becomes a conclusion: Simpson's paradox, gaps explained by segment mix,
differences hidden by mix, extreme rates from tiny groups, and differences driven by a few outliers.

Usage examples
--------------
    python check_analysis.py --data patients.csv --group treatment --outcome recovered --segment severity
    python check_analysis.py --data sessions.csv --group channel --groups web,app --outcome converted \
        --segment customer_type,device
    python check_analysis.py --data counties.csv --outcome cancer_rate_flag --segment county

--group is the comparison you want to make (treatment, channel, year, region). Without --group, only the
segment-level checks run. The outcome can be 0/1 (a rate) or numeric (an average).
Exit code: 1 if any FAIL finding, 0 otherwise, 2 on bad input. Add --json for structured output.
"""
from __future__ import annotations

import argparse
import json
import math
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


def numeric(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    out = pd.to_numeric(s, errors="coerce")
    if out.notna().sum() == 0 and s.notna().any():
        out = s.astype(str).str.lower().map({"true": 1, "false": 0, "yes": 1, "no": 0})
    return out


def r(v, d=4):
    return None if v is None or v != v else round(float(v), d)


def diff_stats(ya: np.ndarray, yb: np.ndarray):
    """Difference in means b - a, its standard error and z."""
    d = float(yb.mean() - ya.mean())
    se = math.sqrt(ya.var(ddof=1) / len(ya) + yb.var(ddof=1) / len(yb)) if min(len(ya), len(yb)) > 1 else float("nan")
    return d, se, (d / se if se and se == se else 0.0)


def adjusted_diff(seg: pd.Series, grp: pd.Series, y: pd.Series, a: str, b: str):
    """Difference b - a within each segment, averaged with weights = segment size (direct standardisation).

    Returns (adjusted difference, its standard error, share of rows covered, per-segment table)."""
    rows = []
    total = len(y)
    for s, idx in seg.groupby(seg, dropna=False).groups.items():
        ya = y.loc[idx][grp.loc[idx] == a].dropna().to_numpy(float)
        yb = y.loc[idx][grp.loc[idx] == b].dropna().to_numpy(float)
        if len(ya) < 2 or len(yb) < 2:
            continue
        d, se, _ = diff_stats(ya, yb)
        rows.append({"segment": s, "n_a": len(ya), "n_b": len(yb), "mean_a": ya.mean(), "mean_b": yb.mean(),
                     "diff": d, "se": se if se == se else 0.0})
    if not rows:
        return float("nan"), float("nan"), 0.0, pd.DataFrame()
    t = pd.DataFrame(rows)
    w = (t["n_a"] + t["n_b"]) / (t["n_a"] + t["n_b"]).sum()
    adj = float((w * t["diff"]).sum())
    se = float(math.sqrt((w ** 2 * t["se"] ** 2).sum()))
    return adj, se, float((t["n_a"] + t["n_b"]).sum() / total), t


def check(df, group, groups, outcome, segments, min_n):
    f, stats = [], {}
    y = numeric(df[outcome])
    if y.notna().sum() == 0:
        raise SystemExit(f"error: outcome '{outcome}' has no numeric values")
    binary = bool(set(np.unique(y.dropna())) <= {0, 1})
    stats.update(rows=len(df), outcome_type="rate" if binary else "average")

    if group:
        g = df[group].astype(str)
        if groups:
            a, b = groups
        else:
            top = g.value_counts().index[:2].tolist()
            if len(top) < 2:
                raise SystemExit(f"error: '{group}' has fewer than two groups")
            a, b = sorted(top)
        for x in (a, b):
            if x not in set(g):
                raise SystemExit(f"error: group '{x}' not found in '{group}'")
        keep = g.isin([a, b]) & y.notna()
        sub_y, sub_g = y[keep], g[keep]
        ya, yb = sub_y[sub_g == a].to_numpy(float), sub_y[sub_g == b].to_numpy(float)
        d, se, z = diff_stats(ya, yb)
        stats.update(group_a=a, group_b=b, n_a=len(ya), n_b=len(yb), mean_a=r(ya.mean()), mean_b=r(yb.mean()),
                     difference=r(d), z=r(z, 2))
        label = "rate" if binary else "mean"

        for s in segments:
            adj, se_adj, coverage, table = adjusted_diff(df.loc[keep, s], sub_g, sub_y, a, b)
            if table.empty:
                continue
            z_adj = adj / se_adj if se_adj else 0.0
            stats[f"adjusted_difference_by_{s}"] = r(adj)
            reversed_segs = int((np.sign(table["diff"]) == -np.sign(d)).sum()) if d else 0
            mix = (df.loc[keep, s].groupby(sub_g).value_counts(normalize=True).unstack(fill_value=0))
            mix_gap = float((mix.loc[a] - mix.loc[b]).abs().sum() / 2) if a in mix.index and b in mix.index else 0.0
            examples = [f"{row.segment}: {a} {row.mean_a:.3g} vs {b} {row.mean_b:.3g} (n={row.n_a}/{row.n_b})"
                        for row in table.head(6).itertuples()]
            if abs(z) >= 2 and abs(z_adj) >= 2 and np.sign(adj) != np.sign(d):
                f.append(finding(
                    "analysis.simpsons_paradox", "FAIL",
                    f"Overall, {b} has a {'higher' if d > 0 else 'lower'} {label} than {a} ({d:+.4g}), but within "
                    f"'{s}' the difference reverses ({adj:+.4g} after weighting by segment size; {reversed_segs} of "
                    f"{len(table)} segments point the other way). The two groups have very different '{s}' mixes "
                    f"({mix_gap:.0%} of rows would need to move to match). The overall comparison is misleading; "
                    f"report the within-'{s}' comparison and explain the mix.", [group, s, outcome], r(adj), examples))
            elif abs(z) >= 2 and (np.sign(adj) != np.sign(d) or abs(adj) < 0.5 * abs(d)):
                share = "all" if np.sign(adj) != np.sign(d) else f"{1 - adj / d:.0%}"
                f.append(finding(
                    "analysis.mix_shift", "WARN",
                    f"The overall gap ({d:+.4g}) shrinks to {adj:+.4g} once '{s}' is held constant: "
                    f"{share} of it comes from {a} and {b} having different '{s}' mixes, not from "
                    f"{group} itself. Say so when presenting the gap.", [group, s, outcome], r(adj / d), examples))
            elif abs(z) < 2 and abs(z_adj) >= 3:
                f.append(finding(
                    "analysis.masked_difference", "WARN",
                    f"There is no clear overall difference ({d:+.4g}), but within '{s}' {b} differs from {a} by "
                    f"{adj:+.4g} (z = {z_adj:.1f}). A difference in '{s}' mix hides it in the total.",
                    [group, s, outcome], r(adj), examples))
            if coverage < 0.9:
                f.append(finding(
                    "analysis.segment_coverage", "WARN",
                    f"Only {coverage:.0%} of rows sit in '{s}' segments that contain both groups; the rest cannot "
                    "be compared like-for-like.", [s], r(coverage)))

        # Gap driven by a few extreme rows
        if not binary and abs(z) >= 2:
            cap = float(np.nanpercentile(sub_y, 99.5))
            capped = sub_y.clip(upper=cap)
            dc, _, zc = diff_stats(capped[sub_g == a].to_numpy(float), capped[sub_g == b].to_numpy(float))
            stats["difference_capped_99_5"] = r(dc)
            if abs(zc) < 2 or np.sign(dc) != np.sign(d) or abs(dc) < 0.5 * abs(d):
                top = sub_y.nlargest(5)
                f.append(finding(
                    "analysis.outlier_driven", "WARN",
                    f"The {d:+.4g} difference in mean '{outcome}' becomes {dc:+.4g} (z = {zc:.1f}) when values above "
                    f"the 99.5th percentile ({cap:,.4g}) are capped. A handful of extreme rows drive it. Show the "
                    "median or a capped mean next to the mean, and look at those rows.",
                    [outcome], r(dc), [f"{group}={sub_g[i]}: {v:,.4g}" for i, v in top.items()]))

    # Extreme segment values that come from tiny segments
    for s in segments:
        per = pd.DataFrame({"y": y, "s": df[s]}).dropna().groupby("s")["y"].agg(["mean", "size"])
        if len(per) < 5:
            continue
        small = per[per["size"] < min_n]
        stats[f"segments_in_{s}"] = int(len(per))
        stats[f"segments_below_min_n_in_{s}"] = int(len(small))
        extremes = [per["mean"].idxmax(), per["mean"].idxmin()]
        tiny = [e for e in extremes if per.loc[e, "size"] < min_n]
        if tiny and len(per) - len(small) >= 2:
            ex = [f"{e}: {per.loc[e, 'mean']:.3g} (n={int(per.loc[e, 'size'])})" for e in tiny]
            f.append(finding(
                "analysis.small_n_extremes", "WARN",
                f"The {'highest' if tiny[0] == extremes[0] else 'lowest'} '{outcome}' by '{s}' comes from a segment "
                f"with fewer than {min_n} rows; {len(small)} of {len(per)} segments are that small. Small groups "
                "produce extreme averages by chance alone, so a 'best' or 'worst' ranking mostly lists the smallest "
                "segments. Set a minimum size, show counts, or shrink toward the overall mean.", [s, outcome],
                len(small), ex))
    return f, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--outcome", required=True, help="0/1 or numeric column being compared")
    ap.add_argument("--group", help="column defining the comparison (treatment, channel, year, ...)")
    ap.add_argument("--groups", help="the two groups to compare, e.g. A,B (default: the two largest)")
    ap.add_argument("--segment", default="", help="comma-separated columns that could confound the comparison")
    ap.add_argument("--min-n", type=int, default=30, help="segments smaller than this are 'small'")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    df = read_table(a.data)
    segments = [s.strip() for s in a.segment.split(",") if s.strip()]
    groups = [x.strip() for x in a.groups.split(",")] if a.groups else None
    if groups and len(groups) != 2:
        raise SystemExit("error: --groups takes exactly two values")
    for c in [a.outcome, *segments, *([a.group] if a.group else [])]:
        if c not in df.columns:
            raise SystemExit(f"error: column '{c}' not in {a.data}")
    if not a.group and not segments:
        raise SystemExit("error: give --group, --segment or both")

    for s in segments:
        df[s] = df[s].astype(object).where(df[s].notna(), "(missing)").astype(str)
    findings, stats = check(df, a.group, groups, a.outcome, segments, a.min_n)
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "analysis-review", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"analysis-review: {status}")
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
