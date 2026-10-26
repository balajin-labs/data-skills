#!/usr/bin/env python3
"""Generate the seeded-bug benchmark: small realistic datasets, each with one planted
data-science or data-engineering mistake, plus clean controls that must not raise a FAIL.

    python benchmark/generate.py            # writes benchmark/cases/<case-id>/
    python benchmark/run.py                 # runs every skill check against every case

Every case is reproducible from the seed, so results are comparable across versions.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
SEED = 7


def write_case(case_id: str, skill: str, description: str, files: dict, args: list,
               expect: list | None, bug: bool):
    d = CASES / case_id
    d.mkdir(parents=True, exist_ok=True)
    for name, df in files.items():
        df.to_csv(d / name, index=False)
    (d / "case.json").write_text(json.dumps({
        "id": case_id, "skill": skill, "bug": bug, "description": description,
        "args": args, "expect": expect or [],
    }, indent=2))


# ----------------------------------------------------------------- leakage
def churn_frame(rng, n, n_customers=None, start="2025-01-01", days=540):
    n_customers = n_customers or n
    cust = rng.permutation(np.arange(100000, 100000 + n_customers))
    customer_id = rng.choice(cust, n, replace=True) if n_customers < n else cust[:n]
    tenure = rng.integers(1, 72, n)
    spend = np.round(rng.gamma(4, 15, n), 2)
    calls = rng.poisson(1.2, n)
    plan = rng.choice(["basic", "plus", "pro"], n, p=[0.5, 0.35, 0.15])
    region = rng.choice(["north", "south", "east", "west"], n)
    snapshot = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n), unit="D")
    logit = -1.2 - 0.03 * tenure + 0.35 * calls + 0.004 * spend + np.where(plan == "basic", 0.4, 0)
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "customer_id": customer_id, "snapshot_date": snapshot.strftime("%Y-%m-%d"),
        "tenure_months": tenure, "monthly_spend": spend, "support_calls": calls,
        "plan": plan, "region": region, "churned": churned,
    })


def leakage_cases(rng):
    base_args = ["--data", "{dir}/data.csv", "--target", "churned"]

    # L1: target proxy column
    df = churn_frame(rng, 2000)
    flip = rng.random(len(df)) < 0.03
    df["account_status"] = np.where(df["churned"].astype(bool) ^ flip, "closed", "active")
    write_case("leak-01-target-proxy", "leakage-audit",
               "account_status is set when the customer churns: a disguised copy of the label.",
               {"data.csv": df}, base_args + ["--entity", "customer_id"],
               [{"check": "leakage.single_feature", "severity": "FAIL"}], True)

    # L2: same customer in train and test (random row split over repeated snapshots)
    df = churn_frame(rng, 3000, n_customers=1000)
    df["split"] = np.where(rng.random(len(df)) < 0.8, "train", "test")
    write_case("leak-02-entity-overlap", "leakage-audit",
               "Three snapshots per customer, split at random by row, so customers appear on both sides.",
               {"data.csv": df}, base_args + ["--split", "split", "--entity", "customer_id"],
               [{"check": "leakage.entity_overlap", "severity": "FAIL"}], True)

    # L3: random split on time-ordered data
    df = churn_frame(rng, 2000)
    df["split"] = np.where(rng.random(len(df)) < 0.8, "train", "test")
    write_case("leak-03-time-overlap", "leakage-audit",
               "Snapshots span 18 months but the split is random, so training uses the future.",
               {"data.csv": df}, base_args + ["--split", "split", "--time", "snapshot_date"],
               [{"check": "leakage.time_overlap", "severity": "FAIL"}], True)

    # L4: a date recorded after the prediction moment
    df = churn_frame(rng, 2000)
    snap = pd.to_datetime(df["snapshot_date"])
    after = rng.random(len(df)) < 0.7
    offset = np.where(after, rng.integers(1, 90, len(df)), -rng.integers(1, 30, len(df)))
    df["last_activity_date"] = (snap + pd.to_timedelta(offset, unit="D")).dt.strftime("%Y-%m-%d")
    write_case("leak-04-future-timestamp", "leakage-audit",
               "last_activity_date is filled in after the snapshot for most customers.",
               {"data.csv": df}, base_args + ["--time", "snapshot_date", "--entity", "customer_id"],
               [{"check": "leakage.future_timestamp", "severity": "FAIL"}], True)

    # L5: clean control: time-based split, unique customers, no proxies
    df = churn_frame(rng, 2500).sort_values("snapshot_date").reset_index(drop=True)
    cutoff = int(len(df) * 0.8)
    df["split"] = ["train"] * cutoff + ["test"] * (len(df) - cutoff)
    write_case("leak-05-clean", "leakage-audit",
               "Clean control: time-based split, each customer appears once.",
               {"data.csv": df}, base_args + ["--split", "split", "--time", "snapshot_date", "--entity", "customer_id"],
               None, False)


# -------------------------------------------------------------------- eval
def scored(rng, n, prevalence, strength):
    y = (rng.random(n) < prevalence).astype(int)
    score = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + strength * (y - prevalence) * 3)))
    return y, np.round(score, 5)


def eval_cases(rng):
    # E1: rare positives, model predicts the majority class
    n = 3000
    y = (rng.random(n) < 0.05).astype(int)
    df = pd.DataFrame({"label": y, "prob": np.round(rng.random(n), 5), "pred_label": 0})
    write_case("eval-01-majority-baseline", "eval-sanity",
               "5% positives; the model predicts 'no' for everyone and reports 95% accuracy.",
               {"preds.csv": df}, ["--pred", "{dir}/preds.csv", "--y-true", "label", "--y-score", "prob",
                                   "--y-pred", "pred_label"],
               [{"check": "eval.no_better_than_baseline", "severity": "FAIL"},
                {"check": "eval.imbalanced", "severity": "WARN"}], True)

    # E2: "improvement" inside the noise
    n = 600
    y, old = scored(rng, n, 0.3, 0.8)
    new = np.clip(old + rng.normal(0, 0.02, n), 0, 1).round(5)
    df = pd.DataFrame({"label": y, "prob_new": new, "prob_old": old})
    write_case("eval-02-not-significant", "eval-sanity",
               "New model differs from the old one by noise only, yet a tiny AUC gain gets reported.",
               {"preds.csv": df}, ["--pred", "{dir}/preds.csv", "--y-true", "label", "--y-score", "prob_new",
                                   "--compare-score", "prob_old"],
               [{"check": "eval.not_significant", "severity": "WARN"}], True)

    # E3: good on average, useless in one region
    n = 4000
    y, s = scored(rng, n, 0.3, 1.2)
    region = rng.choice(["north", "south", "east", "west"], n)
    west = region == "west"
    s = np.where(west, np.round(rng.random(n), 5), s)
    df = pd.DataFrame({"label": y, "prob": s, "region": region})
    write_case("eval-03-weak-segment", "eval-sanity",
               "Overall AUC looks fine, but in the west region the model is random.",
               {"preds.csv": df}, ["--pred", "{dir}/preds.csv", "--y-true", "label", "--y-score", "prob",
                                   "--segment", "region"],
               [{"check": "eval.weak_segment", "severity": "WARN"}], True)

    # E4: regression forecast worse than predicting the median
    n = 1500
    actual = np.round(rng.gamma(2, 50, n), 2)
    forecast = np.round(actual.mean() + rng.normal(0, 80, n), 2)
    df = pd.DataFrame({"sales": actual, "forecast": forecast})
    write_case("eval-04-regression-baseline", "eval-sanity",
               "Forecast is the mean plus noise: worse than a flat median forecast.",
               {"preds.csv": df}, ["--pred", "{dir}/preds.csv", "--y-true", "sales", "--y-pred", "forecast"],
               [{"check": "eval.no_better_than_baseline", "severity": "FAIL"}], True)

    # E5: clean control: real signal, clear improvement, consistent segments
    n = 4000
    y = (rng.random(n) < 0.3).astype(int)
    latent = rng.normal(0, 1, n) + 2.0 * y
    new = np.round(1 / (1 + np.exp(-latent)), 5)
    old = np.round(1 / (1 + np.exp(-(latent * 0.5 + rng.normal(0, 1.2, n)))), 5)
    region = rng.choice(["north", "south", "east", "west"], n)

    # E6: good scores, threshold left at the 0.5 default -> labels worse than majority class
    df = pd.DataFrame({"label": y, "prob": new, "pred_label": (new > 0.5).astype(int)})
    write_case("eval-06-bad-threshold", "eval-sanity",
               "Scores rank well, but the default 0.5 cut-off labels far too many rows positive.",
               {"preds.csv": df}, ["--pred", "{dir}/preds.csv", "--y-true", "label", "--y-score", "prob",
                                   "--y-pred", "pred_label"],
               [{"check": "eval.threshold", "severity": "WARN"}], True)

    df = pd.DataFrame({"label": y, "prob_new": new, "prob_old": old,
                       "pred_label": (new > 0.75).astype(int), "region": region})
    write_case("eval-05-clean", "eval-sanity",
               "Clean control: strong model, clearly better than the old one, similar across regions.",
               {"preds.csv": df}, ["--pred", "{dir}/preds.csv", "--y-true", "label", "--y-score", "prob_new",
                                   "--y-pred", "pred_label", "--compare-score", "prob_old", "--segment", "region"],
               None, False)


# -------------------------------------------------------------------- join
def orders_and_customers(rng, n_orders=3000, n_customers=800):
    customer_ids = np.arange(1001, 1001 + n_customers)
    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "segment": rng.choice(["consumer", "pro", "enterprise"], n_customers, p=[0.6, 0.3, 0.1]),
        "state": rng.choice(["NC", "SC", "VA", "GA", "TN"], n_customers),
    })
    orders = pd.DataFrame({
        "order_id": np.arange(500000, 500000 + n_orders),
        "customer_id": rng.choice(customer_ids, n_orders),
        "order_total": np.round(rng.gamma(2.5, 40, n_orders), 2),
    })
    return orders, customers


def join_cases(rng):
    # J1: dimension with history rows -> fan-out and double counted revenue
    orders, customers = orders_and_customers(rng)
    moved = customers.sample(frac=0.08, random_state=SEED).assign(state="FL")
    customers_hist = pd.concat([customers, moved], ignore_index=True)
    write_case("join-01-fan-out", "join-guard",
               "Customer table keeps history rows (8% of customers moved), so the join multiplies orders.",
               {"orders.csv": orders, "customers.csv": customers_hist},
               ["--left", "{dir}/orders.csv", "--right", "{dir}/customers.csv", "--on", "customer_id",
                "--sum-col", "order_total"],
               [{"check": "join.fan_out", "severity": "FAIL"}, {"check": "join.double_count", "severity": "FAIL"}],
               True)

    # J2: keys zero-padded on one side
    orders, customers = orders_and_customers(rng)
    orders = orders.assign(customer_id=orders["customer_id"].map(lambda v: f"{v:08d}"))
    write_case("join-02-key-format", "join-guard",
               "Orders store customer_id as zero-padded text ('00001001'); the customer table uses 1001.",
               {"orders.csv": orders, "customers.csv": customers},
               ["--left", "{dir}/orders.csv", "--right", "{dir}/customers.csv", "--on", "customer_id"],
               [{"check": "join.key_format", "severity": "FAIL"}], True)

    # J3: clean control
    orders, customers = orders_and_customers(rng)
    write_case("join-03-clean", "join-guard",
               "Clean control: unique dimension keys, matching formats.",
               {"orders.csv": orders, "customers.csv": customers},
               ["--left", "{dir}/orders.csv", "--right", "{dir}/customers.csv", "--on", "customer_id",
                "--sum-col", "order_total"],
               None, False)


# ------------------------------------------------------------------- drift
def sales_extract(rng, n, month):
    start = pd.Timestamp(f"2026-{month:02d}-01")
    return pd.DataFrame({
        "order_id": np.arange(n) + month * 1_000_000,
        "order_date": (start + pd.to_timedelta(rng.integers(0, 28, n), unit="D")).strftime("%Y-%m-%d"),
        "amount": np.round(rng.gamma(3, 25, n), 2),
        "quantity": rng.integers(1, 6, n),
        "channel": rng.choice(["store", "web", "app"], n, p=[0.55, 0.3, 0.15]),
        "email": np.where(rng.random(n) < 0.02, "", [f"c{i}@example.com" for i in rng.integers(0, 10**6, n)]),
    })


def drift_cases(rng):
    args = ["--baseline", "{dir}/baseline.csv", "--current", "{dir}/current.csv", "--key", "order_id"]

    base = sales_extract(rng, 4000, 9)
    cur = sales_extract(rng, 4200, 10)
    cur["amount"] = (cur["amount"] * 100).round(0)
    write_case("drift-01-unit-change", "null-type-drift",
               "The source switched amount from dollars to cents.",
               {"baseline.csv": base, "current.csv": cur}, args,
               [{"check": "drift.unit_change", "severity": "FAIL"}], True)

    base = sales_extract(rng, 4000, 9)
    cur = sales_extract(rng, 4100, 10)
    cur.loc[rng.random(len(cur)) < 0.4, "email"] = "N/A"
    write_case("drift-02-null-spike", "null-type-drift",
               "A new upstream system sends 'N/A' for 40% of emails.",
               {"baseline.csv": base, "current.csv": cur}, args,
               [{"check": "drift.null_spike", "severity": "FAIL"}], True)

    base = sales_extract(rng, 4000, 9)
    cur = sales_extract(rng, 4000, 10)
    cur["amount"] = cur["amount"].map(lambda v: f"${v:,.2f}")
    write_case("drift-03-type-change", "null-type-drift",
               "amount now arrives as formatted text ('$1,234.50').",
               {"baseline.csv": base, "current.csv": cur}, args,
               [{"check": "drift.type_change", "severity": "FAIL"}], True)

    base = sales_extract(rng, 4000, 9)
    cur = sales_extract(rng, 4000, 10)
    cur = pd.concat([cur, cur.sample(frac=0.08, random_state=SEED)], ignore_index=True)
    write_case("drift-04-duplicate-load", "null-type-drift",
               "Part of the October load ran twice.",
               {"baseline.csv": base, "current.csv": cur}, args,
               [{"check": "drift.duplicates", "severity": "FAIL"}], True)

    base = sales_extract(rng, 4000, 9)
    cur = sales_extract(rng, 4300, 10)
    write_case("drift-05-clean", "null-type-drift",
               "Clean control: next month's extract with normal variation.",
               {"baseline.csv": base, "current.csv": cur}, args, None, False)


def main():
    if CASES.exists():
        shutil.rmtree(CASES)
    rng = np.random.default_rng(SEED)
    leakage_cases(rng)
    eval_cases(rng)
    join_cases(rng)
    drift_cases(rng)
    n = len(list(CASES.glob("*/case.json")))
    print(f"wrote {n} cases to {CASES}")


if __name__ == "__main__":
    main()
