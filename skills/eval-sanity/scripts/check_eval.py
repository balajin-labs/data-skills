#!/usr/bin/env python3
"""Sanity checks for a model evaluation, run on a file of test-set predictions.

Usage examples
--------------
Binary classifier with scores and hard predictions:
    python check_eval.py --pred preds.csv --y-true label --y-score prob --y-pred pred_label --segment region

Compare a new model against the current one:
    python check_eval.py --pred preds.csv --y-true label --y-score prob_new --compare-score prob_old

Regression:
    python check_eval.py --pred preds.csv --y-true sales --y-pred forecast --segment store_type

Exit code: 1 if any FAIL finding, 0 otherwise, 2 on bad input. Add --json for structured output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             mean_absolute_error, precision_score,
                             recall_score, roc_auc_score)

N_BOOT = 400
SEGMENT_MIN_ROWS = 50


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: file not found: {path}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, sep="\t" if p.suffix.lower() in {".tsv", ".tab"} else ",")


def finding(check, severity, message, columns=None, value=None):
    return {"check": check, "severity": severity, "message": message,
            "columns": columns or [], "value": value}


def to_binary(y: pd.Series):
    vals = pd.unique(y.dropna())
    if len(vals) != 2:
        return None
    strs = {str(v) for v in vals}
    if strs <= {"0", "1", "0.0", "1.0", "True", "False"}:
        return y.astype(str).isin({"1", "1.0", "True"}).astype(int)
    positive = sorted(vals, key=str)[-1]
    return (y == positive).astype(int)


def boot_ci(fn, n, rng, *arrays):
    stats = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        try:
            v = fn(*(a[idx] for a in arrays))
        except ValueError:
            continue
        if v == v:
            stats.append(v)
    if len(stats) < N_BOOT // 2:
        return (float("nan"), float("nan"))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def binary_checks(df, y, score, pred, compare, segments, rng):
    f, metrics = [], {}
    n = len(y)
    prev = float(y.mean())
    metrics["rows"], metrics["prevalence"] = n, round(prev, 4)
    n_pos = int(y.sum())

    if min(n_pos, n - n_pos) < 30:
        f.append(finding("eval.small_test", "WARN",
                         f"Only {min(n_pos, n - n_pos)} examples of the rarer class in the test set; "
                         "every metric will be noisy. Report confidence intervals."))
    if prev < 0.10 or prev > 0.90:
        f.append(finding("eval.imbalanced", "WARN",
                         f"Positive rate is {prev:.1%}. Accuracy is misleading here; lead with PR-AUC, "
                         "recall at a fixed precision, or a cost-weighted metric.", value=round(prev, 4)))

    yv = y.to_numpy()
    auc = lo = hi = None
    if score is not None:
        sv = score.to_numpy(dtype=float)
        auc = roc_auc_score(yv, sv)
        lo, hi = boot_ci(roc_auc_score, n, rng, yv, sv)

    if pred is not None:
        pv = pred.to_numpy()
        acc = accuracy_score(yv, pv)
        majority = max(prev, 1 - prev)
        metrics.update(accuracy=round(acc, 4), majority_baseline=round(majority, 4),
                       predicted_positive_rate=round(float(pv.mean()), 4),
                       precision=round(precision_score(yv, pv, zero_division=0), 4),
                       recall=round(recall_score(yv, pv, zero_division=0), 4))
        if acc <= majority + 0.01:
            ranks_well = lo is not None and lo == lo and lo > 0.6
            if ranks_well:
                # The scores separate the classes; the cut-off turning them into labels is the problem.
                f.append(finding("eval.threshold", "WARN",
                                 f"Scores rank well (ROC AUC {auc:.3f}) but the chosen threshold gives accuracy "
                                 f"{acc:.3f}, below the majority-class baseline ({majority:.3f}); it flags "
                                 f"{pv.mean():.0%} of rows as positive against a {prev:.0%} positive rate. "
                                 "Choose the threshold on validation data for the cost of each error.",
                                 value=round(acc - majority, 4)))
            else:
                f.append(finding("eval.no_better_than_baseline", "FAIL",
                                 f"Accuracy {acc:.3f} is not better than always predicting the majority class "
                                 f"({majority:.3f}). Recall on the positive class is {metrics['recall']:.2f}.",
                                 value=round(acc - majority, 4)))

    if score is not None:
        ap = average_precision_score(yv, sv)
        metrics.update(roc_auc=round(auc, 4), roc_auc_ci95=[round(lo, 4), round(hi, 4)],
                       pr_auc=round(ap, 4), pr_auc_baseline=round(prev, 4))
        if lo == lo and lo <= 0.5:
            f.append(finding("eval.no_better_than_baseline", "FAIL",
                             f"ROC AUC {auc:.3f} with 95% CI [{lo:.3f}, {hi:.3f}] includes 0.5: "
                             "no evidence the model beats random ranking.", value=round(auc, 4)))
        elif ap < 1.5 * prev and prev < 0.5:
            f.append(finding("eval.weak_precision", "WARN",
                             f"PR-AUC {ap:.3f} is less than 1.5x the positive rate ({prev:.3f}); "
                             "precision on positives is barely above chance.", value=round(ap, 4)))

        if compare is not None:
            cv = compare.to_numpy(dtype=float)
            auc_b = roc_auc_score(yv, cv)
            diff = auc - auc_b
            dlo, dhi = boot_ci(lambda a, b, c: roc_auc_score(a, b) - roc_auc_score(a, c), n, rng, yv, sv, cv)
            metrics.update(compare_roc_auc=round(auc_b, 4), auc_difference=round(diff, 4),
                           auc_difference_ci95=[round(dlo, 4), round(dhi, 4)])
            if dlo == dlo and dlo <= 0 <= dhi:
                f.append(finding("eval.not_significant", "WARN",
                                 f"AUC difference {diff:+.3f} has 95% CI [{dlo:+.3f}, {dhi:+.3f}], which includes 0. "
                                 "The new model is not demonstrably better.", value=round(diff, 4)))
            elif dhi < 0:
                f.append(finding("eval.not_significant", "FAIL",
                                 f"The new model is worse: AUC difference {diff:+.3f}, CI [{dlo:+.3f}, {dhi:+.3f}].",
                                 value=round(diff, 4)))

    for seg in segments:
        seg_rows = []
        for val, idx in df.groupby(seg, dropna=False).groups.items():
            ys = y.loc[idx]
            if len(ys) < SEGMENT_MIN_ROWS or ys.nunique() < 2 or min(ys.sum(), len(ys) - ys.sum()) < 5:
                continue
            if score is not None:
                m, overall, name = roc_auc_score(ys, score.loc[idx]), metrics["roc_auc"], "ROC AUC"
                gap = overall - m
                bad = gap > 0.10
            elif pred is not None:
                m, overall, name = recall_score(ys, pred.loc[idx], zero_division=0), metrics["recall"], "recall"
                gap = overall - m
                bad = gap > 0.15
            else:
                continue
            seg_rows.append({"segment": str(val), "rows": int(len(ys)), name: round(float(m), 4)})
            if bad:
                f.append(finding("eval.weak_segment", "WARN",
                                 f"{seg}={val} ({len(ys)} rows): {name} {m:.3f} vs {overall:.3f} overall. "
                                 "The average hides a segment where the model performs much worse.",
                                 [seg], round(float(gap), 4)))
        metrics[f"segments_{seg}"] = seg_rows
    return f, metrics


def regression_checks(df, y, pred, compare, segments, rng):
    f, metrics = [], {}
    yv, pv = y.to_numpy(dtype=float), pred.to_numpy(dtype=float)
    n = len(yv)
    mae = mean_absolute_error(yv, pv)
    base = mean_absolute_error(yv, np.full(n, np.median(yv)))
    lo, hi = boot_ci(mean_absolute_error, n, rng, yv, pv)
    metrics.update(rows=n, mae=round(mae, 4), mae_ci95=[round(lo, 4), round(hi, 4)],
                   median_baseline_mae=round(base, 4))
    if mae >= 0.98 * base:
        f.append(finding("eval.no_better_than_baseline", "FAIL",
                         f"MAE {mae:.3f} is not better than predicting the median ({base:.3f}).",
                         value=round(mae / base, 4)))
    if compare is not None:
        cv = compare.to_numpy(dtype=float)
        diff = mae - mean_absolute_error(yv, cv)
        dlo, dhi = boot_ci(lambda a, b, c: mean_absolute_error(a, b) - mean_absolute_error(a, c), n, rng, yv, pv, cv)
        metrics.update(mae_difference=round(diff, 4), mae_difference_ci95=[round(dlo, 4), round(dhi, 4)])
        if dlo <= 0 <= dhi:
            f.append(finding("eval.not_significant", "WARN",
                             f"MAE difference {diff:+.3f} has 95% CI [{dlo:+.3f}, {dhi:+.3f}], which includes 0.",
                             value=round(diff, 4)))
    for seg in segments:
        for val, idx in df.groupby(seg, dropna=False).groups.items():
            if len(idx) < SEGMENT_MIN_ROWS:
                continue
            m = mean_absolute_error(y.loc[idx], pred.loc[idx])
            if m > 1.5 * mae:
                f.append(finding("eval.weak_segment", "WARN",
                                 f"{seg}={val} ({len(idx)} rows): MAE {m:.3f} vs {mae:.3f} overall.",
                                 [seg], round(m / mae, 4)))
    return f, metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="file with one row per test example")
    ap.add_argument("--y-true", required=True)
    ap.add_argument("--y-score", help="predicted probability or score (classification)")
    ap.add_argument("--y-pred", help="predicted label (classification) or value (regression)")
    ap.add_argument("--compare-score", help="score or prediction column of the model you compare against")
    ap.add_argument("--segment", action="append", default=[], help="column to slice metrics by (repeatable)")
    ap.add_argument("--task", choices=["auto", "binary", "regression"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    df = read_table(a.pred)
    for c in [a.y_true, a.y_score, a.y_pred, a.compare_score, *a.segment]:
        if c and c not in df.columns:
            raise SystemExit(f"error: column '{c}' not in {a.pred}")
    if not a.y_score and not a.y_pred:
        raise SystemExit("error: give --y-score and/or --y-pred")

    used = [c for c in [a.y_true, a.y_score, a.y_pred, a.compare_score] if c]
    df = df.dropna(subset=used).reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    yb = to_binary(df[a.y_true])
    task = a.task if a.task != "auto" else ("binary" if yb is not None else "regression")

    if task == "binary":
        if yb is None:
            raise SystemExit("error: --task binary but y-true does not have exactly two values")
        pred = to_binary(df[a.y_pred]) if a.y_pred else None
        if a.y_pred and pred is None:
            pred = (df[a.y_pred].astype(str) == str(sorted(pd.unique(df[a.y_true]), key=str)[-1])).astype(int)
        findings, metrics = binary_checks(df, yb, df[a.y_score] if a.y_score else None, pred,
                                          df[a.compare_score] if a.compare_score else None, a.segment, rng)
    else:
        if not a.y_pred:
            raise SystemExit("error: regression needs --y-pred")
        findings, metrics = regression_checks(df, df[a.y_true].astype(float), df[a.y_pred].astype(float),
                                              df[a.compare_score] if a.compare_score else None, a.segment, rng)

    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "eval-sanity", "task": task, "status": status, "metrics": metrics, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"eval-sanity ({task}): {status}")
        for k, v in metrics.items():
            if not k.startswith("segments_"):
                print(f"  {k}: {v}")
        for x in findings:
            print(f"  [{x['severity']}] {x['check']}: {x['message']}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
