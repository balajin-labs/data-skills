#!/usr/bin/env python3
"""Check an A/B test analysis before trusting it: sample ratio mismatch, units in more than one variant,
underpowered tests, uncorrected multiple metrics, novelty effects, pre-period imbalance and outlier-driven lifts.

Usage examples
--------------
    python check_abtest.py --data users.csv --variant variant --control control --metric converted
    python check_abtest.py --data users.csv --variant variant --control A --metric converted,revenue,sessions \
        --unit user_id --time first_exposure --pre pre_revenue --expected-split 50,50

The file has one row per randomisation unit (usually a user): the variant it saw and its metric values.
The first --metric is the primary metric. Binary metrics should be 0/1 (or true/false).
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

SRM_P = 0.001          # industry-standard threshold for a sample ratio mismatch alarm
POWER = 0.8


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


# ------------------------------------------------------------------ statistics (no scipy needed)
def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def two_sided_p(z: float) -> float:
    return 2 * norm_sf(abs(z))


def norm_ppf(q: float) -> float:
    """Inverse of the standard normal CDF, by bisection (accurate to ~1e-9)."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 1 - norm_sf(mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-square distribution: exact for df=1, Wilson-Hilferty otherwise."""
    if df <= 0 or x <= 0:
        return 1.0
    if df == 1:
        return two_sided_p(math.sqrt(x))
    z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return norm_sf(z)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the original order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def compare(c: np.ndarray, t: np.ndarray) -> dict:
    """Difference in means (treatment - control) with a large-sample z-test (Welch standard error)."""
    nc, nt = len(c), len(t)
    mc, mt = float(c.mean()), float(t.mean())
    se = math.sqrt(c.var(ddof=1) / nc + t.var(ddof=1) / nt) if nc > 1 and nt > 1 else float("nan")
    diff = mt - mc
    z = diff / se if se and se == se else 0.0
    return {"n_control": nc, "n_treatment": nt, "mean_control": mc, "mean_treatment": mt,
            "diff": diff, "rel": diff / mc if mc else float("nan"), "se": se, "z": z, "p": two_sided_p(z)}


def numeric(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    out = pd.to_numeric(s, errors="coerce")
    if out.notna().sum() == 0 and s.notna().any():
        out = s.astype(str).str.lower().map({"true": 1, "false": 0, "yes": 1, "no": 0})
    return out


def r(v, d=4):
    return None if v is None or v != v else round(float(v), d)


# ------------------------------------------------------------------ checks
def check(df, variant, control, metrics, unit, time_col, pre_cols, split, alpha, target_mde):
    f, stats = [], {}
    v = df[variant].astype(str)
    variants = sorted(v.dropna().unique(), key=lambda x: (x != control, x))
    if control not in variants:
        raise SystemExit(f"error: control '{control}' not found in '{variant}' (values: {variants[:10]})")
    treatments = [x for x in variants if x != control]
    if not treatments:
        raise SystemExit("error: only one variant in the data")
    counts = v.value_counts()
    stats["rows"] = len(df)
    stats["units_per_variant"] = {k: int(counts.get(k, 0)) for k in variants}

    # Units that saw more than one variant, or appear more than once
    if unit:
        per_unit = df.groupby(unit)[variant].nunique()
        multi = per_unit[per_unit > 1]
        share = len(multi) / max(len(per_unit), 1)
        stats["units_in_multiple_variants"] = int(len(multi))
        if len(multi):
            f.append(finding(
                "ab.unit_in_multiple_variants", "FAIL" if share > 0.001 else "WARN",
                f"{len(multi):,} units ({share:.1%}) appear in more than one variant. Assignment is leaking "
                "(randomised per session or device instead of per user, or a logging bug), so the variants are "
                "contaminated and the effect is biased toward zero. Fix the assignment, or drop these units and "
                "say so.", [unit, variant], round(share, 4), [str(x) for x in multi.index[:5]]))
        dup_rows = int(df.duplicated([unit, variant]).sum())
        if dup_rows:
            f.append(finding(
                "ab.repeated_units", "WARN",
                f"{dup_rows:,} rows repeat a unit already seen in the same variant. The tests assume one row per "
                "unit; aggregate to the unit first, or standard errors will be too small.", [unit], dup_rows))

    # Sample ratio mismatch
    if split:
        weights = np.array(split, dtype=float)
        if len(weights) != len(variants):
            raise SystemExit(f"error: --expected-split has {len(weights)} values but there are "
                             f"{len(variants)} variants ({', '.join(variants)}; control first)")
    else:
        weights = np.ones(len(variants))
    expected = weights / weights.sum() * counts.sum()
    observed = np.array([counts.get(k, 0) for k in variants], dtype=float)
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    p_srm = chi2_sf(chi2, len(variants) - 1)
    stats["srm_p_value"] = float(f"{p_srm:.3g}")
    if p_srm < SRM_P:
        shares = ", ".join(f"{k} {o / observed.sum():.2%} (expected {e / observed.sum():.2%})"
                           for k, o, e in zip(variants, observed, expected))
        f.append(finding(
            "ab.srm", "FAIL",
            f"Sample ratio mismatch: {shares}; p = {p_srm:.2g}. The split is not what the assignment intended, "
            "so something removed or added users unevenly (bot filtering, a crash or redirect in one variant, "
            "logging that starts at different points). Every metric is suspect until this is explained.",
            [variant], float(f"{p_srm:.3g}")))

    # Metric comparisons
    results = []
    for m in metrics:
        y = numeric(df[m])
        for t in treatments:
            c_vals = y[v == control].dropna().to_numpy(float)
            t_vals = y[v == t].dropna().to_numpy(float)
            if len(c_vals) < 2 or len(t_vals) < 2:
                continue
            res = compare(c_vals, t_vals)
            res.update(metric=m, treatment=t, binary=bool(set(np.unique(y.dropna())) <= {0, 1}))
            results.append(res)
    if not results:
        raise SystemExit("error: no metric has data in both control and treatment")
    # Holm correction within two families: the primary metric (across treatments) and all secondary comparisons
    for family in ([x for x in results if x["metric"] == metrics[0]],
                   [x for x in results if x["metric"] != metrics[0]]):
        for x, a in zip(family, holm([x["p"] for x in family])):
            x["p_holm"] = a
    stats["results"] = [{"metric": x["metric"], "treatment": x["treatment"], "control_mean": r(x["mean_control"]),
                         "treatment_mean": r(x["mean_treatment"]), "lift": r(x["rel"]),
                         "ci95": [r(x["diff"] - 1.96 * x["se"]), r(x["diff"] + 1.96 * x["se"])],
                         "p": float(f"{x['p']:.3g}"), "p_holm": float(f"{x['p_holm']:.3g}")} for x in results]

    # Multiple metrics / variants without correction
    lost = [x for x in results if x["p"] < alpha <= x["p_holm"]]
    if lost:
        names = ", ".join(f"{x['metric']} ({x['treatment']}: p={x['p']:.3f}, adjusted {x['p_holm']:.3f})"
                          for x in lost)
        n_sec = sum(x["metric"] != metrics[0] for x in results)
        f.append(finding(
            "ab.multiple_metrics", "WARN",
            f"{len(results)} metric/variant comparisons were tested ({n_sec} on secondary metrics). These are "
            f"significant on their own but not after a Holm correction: {names}. With this many tests, one or two "
            "'wins' are expected by chance. Report them as exploratory; the decision rests on the primary metric "
            f"('{metrics[0]}').", [x["metric"] for x in lost], len(lost)))

    # Power for the primary metric
    z_need = norm_ppf(1 - alpha / 2) + norm_ppf(POWER)
    for x in (x for x in results if x["metric"] == metrics[0]):
        if x["p"] < alpha or not x["mean_control"]:
            continue
        mde = z_need * x["se"] / abs(x["mean_control"])
        if mde > target_mde:
            need = int(math.ceil(max(x["n_control"], x["n_treatment"]) * (mde / target_mde) ** 2))
            f.append(finding(
                "ab.underpowered", "WARN",
                f"'{metrics[0]}' ({x['treatment']}) is not significant, but this test could only detect a change of "
                f"about {mde:.1%} or more (80% power); the target is {target_mde:.1%}. 'Not significant' here does "
                f"not mean 'no effect'. Detecting {target_mde:.1%} needs roughly {need:,} units per variant.",
                [metrics[0]], round(mde, 4)))

    # Novelty: the effect fades between early and late exposure cohorts
    if time_col:
        when = pd.to_datetime(df[time_col], errors="coerce")
        cut = when.median()
        y = numeric(df[metrics[0]])
        for t in treatments:
            parts = []
            for mask in (when <= cut, when > cut):
                c_vals = y[mask & (v == control)].dropna().to_numpy(float)
                t_vals = y[mask & (v == t)].dropna().to_numpy(float)
                parts.append(compare(c_vals, t_vals) if len(c_vals) > 30 and len(t_vals) > 30 else None)
            early, late = parts
            if not early or not late:
                continue
            z_gap = (early["diff"] - late["diff"]) / math.sqrt(early["se"] ** 2 + late["se"] ** 2)
            if early["p"] < alpha and abs(z_gap) > 2.5 and abs(late["diff"]) < 0.5 * abs(early["diff"]):
                f.append(finding(
                    "ab.novelty", "WARN",
                    f"The effect on '{metrics[0]}' ({t}) is {early['rel']:+.1%} for units exposed up to "
                    f"{cut:%Y-%m-%d} but {late['rel']:+.1%} for units exposed after it. A fading effect usually "
                    "means novelty or a learning effect; the long-run impact is closer to the later number. "
                    "Plot the daily effect and consider running longer.", [metrics[0], time_col], r(z_gap, 2)))

    # Pre-experiment imbalance
    for col in pre_cols:
        y = numeric(df[col])
        for t in treatments:
            c_vals = y[v == control].dropna().to_numpy(float)
            t_vals = y[v == t].dropna().to_numpy(float)
            if len(c_vals) < 30 or len(t_vals) < 30:
                continue
            res = compare(c_vals, t_vals)
            sd = math.sqrt((c_vals.var(ddof=1) + t_vals.var(ddof=1)) / 2) or 1.0
            smd = res["diff"] / sd
            if res["p"] < 0.001:
                f.append(finding(
                    "ab.pre_imbalance", "FAIL" if abs(smd) >= 0.1 else "WARN",
                    f"Before the experiment started, '{col}' already differed between {control} and {t} "
                    f"(standardised difference {smd:+.2f}, p = {res['p']:.2g}). Randomisation should make "
                    "pre-period behaviour equal; this points to a broken assignment (by signup order, region or "
                    "ID range). The measured effect mixes the treatment with this pre-existing gap.",
                    [col], r(smd, 3)))

    # Lifts driven by a handful of extreme units
    for x in results:
        if x["binary"] or x["p"] >= alpha:
            continue
        y = numeric(df[x["metric"]])
        cap = float(np.nanpercentile(y[v.isin([control, x["treatment"]])], 99.5))
        capped = y.clip(upper=cap)
        res = compare(capped[v == control].dropna().to_numpy(float),
                      capped[v == x["treatment"]].dropna().to_numpy(float))
        if res["p"] >= alpha or np.sign(res["diff"]) != np.sign(x["diff"]) or abs(res["diff"]) < 0.5 * abs(x["diff"]):
            f.append(finding(
                "ab.outlier_driven", "WARN",
                f"The {x['rel']:+.1%} lift in '{x['metric']}' ({x['treatment']}) shrinks to {res['rel']:+.1%} "
                f"(p = {res['p']:.2g}) when values above the 99.5th percentile ({cap:,.2f}) are capped. A few "
                "extreme units drive the result. Report the capped result alongside it and look at those units.",
                [x["metric"]], r(res["rel"])))
    return f, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="one row per randomisation unit")
    ap.add_argument("--variant", required=True, help="column holding the variant name")
    ap.add_argument("--control", required=True, help="name of the control variant")
    ap.add_argument("--metric", required=True, help="comma-separated metric columns; the first is primary")
    ap.add_argument("--unit", help="unit id column (user_id); checks each unit saw one variant")
    ap.add_argument("--time", help="first-exposure timestamp column; checks for novelty effects")
    ap.add_argument("--pre", help="comma-separated pre-experiment columns that should be balanced")
    ap.add_argument("--expected-split", help="intended traffic split, control first, e.g. 50,50 or 80,10,10")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--mde", type=float, default=0.05, help="smallest relative effect worth detecting (0.05 = 5%%)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    df = read_table(a.data)
    metrics = [m.strip() for m in a.metric.split(",") if m.strip()]
    pre = [c.strip() for c in a.pre.split(",")] if a.pre else []
    for c in [a.variant, *metrics, *pre, *(x for x in (a.unit, a.time) if x)]:
        if c not in df.columns:
            raise SystemExit(f"error: column '{c}' not in {a.data}")
    split = [float(x) for x in a.expected_split.split(",")] if a.expected_split else None

    findings, stats = check(df, a.variant, str(a.control), metrics, a.unit, a.time, pre, split, a.alpha, a.mde)
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "ab-test-check", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"ab-test-check: {status}")
        for k, val in stats.items():
            if k != "results":
                print(f"  {k}: {val}")
        for x in stats["results"]:
            print(f"  {x['metric']} [{x['treatment']}]: control {x['control_mean']}, treatment "
                  f"{x['treatment_mean']}, lift {x['lift']}, p {x['p']} (Holm {x['p_holm']})")
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
