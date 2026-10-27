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
    for name, content in files.items():
        if isinstance(content, str):            # SQL, text
            (d / name).write_text(content)
        elif isinstance(content, (dict, list)):  # JSON contracts, notebooks
            (d / name).write_text(json.dumps(content, indent=1))
        else:
            content.to_csv(d / name, index=False)
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


# ------------------------------------------------------------------ ab test
def ab_users(rng, n, p_treat=0.5, base=0.10, lift=0.0, start="2026-03-02", days=28):
    variant = np.where(rng.random(n) < p_treat, "treatment", "control")
    exposure = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n), unit="D")
    rate = np.where(variant == "treatment", base * (1 + lift), base)
    converted = (rng.random(n) < rate).astype(int)
    return pd.DataFrame({
        "user_id": np.arange(n) + 7_000_000,
        "variant": variant,
        "first_exposure": exposure.strftime("%Y-%m-%d"),
        "pre_revenue": np.round(rng.gamma(2, 15, n), 2),
        "converted": converted,
        "revenue": np.round(converted * rng.gamma(2, 30, n), 2),
        "sessions": rng.poisson(4, n),
    })


def ab_cases(rng):
    def args(*extra):
        return ["--data", "{dir}/users.csv", "--variant", "variant", "--control", "control", *extra]

    # A1: treatment loses 6% of its users before logging (sample ratio mismatch)
    df = ab_users(rng, 20000, lift=0.05)
    df = df[~((df["variant"] == "treatment") & (rng.random(len(df)) < 0.06))]
    write_case("ab-01-srm", "ab-test-check",
               "A redirect in the treatment drops 6% of its users before they are logged.",
               {"users.csv": df}, args("--metric", "converted", "--unit", "user_id", "--expected-split", "50,50"),
               [{"check": "ab.srm", "severity": "FAIL"}], True)

    # A2: ten metrics, no real effect; one crosses p<0.05 and gets reported as the win
    n = 12000
    df = ab_users(rng, n)
    t = (df["variant"] == "treatment").to_numpy()
    for i in range(1, 11):
        df[f"m{i}"] = np.round(rng.normal(50, 10, n), 3)
    c_vals, t_vals = df.loc[~t, "m7"], df.loc[t, "m7"]
    se = np.sqrt(c_vals.var() / len(c_vals) + t_vals.var() / len(t_vals))
    df.loc[t, "m7"] += 2.3 * se - (t_vals.mean() - c_vals.mean())   # exactly z = 2.3, p ~ 0.02
    write_case("ab-02-multiple-metrics", "ab-test-check",
               "Ten metrics tested, no true effect; m7 is significant only before correction.",
               {"users.csv": df}, args("--metric", ",".join(f"m{i}" for i in range(1, 11))),
               [{"check": "ab.multiple_metrics", "severity": "WARN"}], True)

    # A3: novelty effect: strong lift for early-exposed users, none later
    df = ab_users(rng, 40000)
    early = pd.to_datetime(df["first_exposure"]) < pd.Timestamp("2026-03-16")
    boost = early & (df["variant"] == "treatment") & (rng.random(len(df)) < 0.045)
    df.loc[boost, "converted"] = 1
    write_case("ab-03-novelty", "ab-test-check",
               "The new design gets clicks while it is new; users exposed in week 3-4 behave like control.",
               {"users.csv": df}, args("--metric", "converted", "--time", "first_exposure"),
               [{"check": "ab.novelty", "severity": "WARN"}], True)

    # A4: too small to tell, reported as "no effect"
    df = ab_users(rng, 600)
    df["variant"] = rng.permutation(np.repeat(["control", "treatment"], 300))
    for name, k in (("control", 30), ("treatment", 33)):     # 10.0% vs 11.0%: a real-looking lift, p ~ 0.7
        idx = df.index[df["variant"] == name]
        df.loc[idx, "converted"] = rng.permutation(np.repeat([1, 0], [k, len(idx) - k]))
    write_case("ab-04-underpowered", "ab-test-check",
               "600 users; a 10% lift cannot be distinguished from zero, and 'no effect' gets reported.",
               {"users.csv": df}, args("--metric", "converted", "--mde", "0.05"),
               [{"check": "ab.underpowered", "severity": "WARN"}], True)

    # A5: assignment per session, so users show up in both variants
    df = ab_users(rng, 20000, lift=0.05)
    df["user_id"] = rng.integers(7_000_000, 7_015_000, len(df))
    write_case("ab-05-unit-in-both", "ab-test-check",
               "Variants were assigned per session, so returning users saw both.",
               {"users.csv": df}, args("--metric", "converted", "--unit", "user_id"),
               [{"check": "ab.unit_in_multiple_variants", "severity": "FAIL"}], True)

    # A6: no real effect on revenue; eight huge purchases in treatment create the "lift"
    df = ab_users(rng, 20000)
    whales = df.index[df["variant"] == "treatment"][:8]
    df.loc[whales, "revenue"] = 6000.0
    write_case("ab-06-outlier-driven", "ab-test-check",
               "Eight bulk purchases in treatment produce a significant revenue lift.",
               {"users.csv": df}, args("--metric", "revenue"),
               [{"check": "ab.outlier_driven", "severity": "WARN"}], True)

    # A7: assignment by a rule correlated with past spend (heavy users more likely in treatment)
    df = ab_users(rng, 20000)
    high = df["pre_revenue"] > df["pre_revenue"].median()
    df["variant"] = np.where(rng.random(len(df)) < np.where(high, 0.6, 0.4), "treatment", "control")
    write_case("ab-07-pre-imbalance", "ab-test-check",
               "Assignment leaned on past spend: heavy users were more likely to get the treatment.",
               {"users.csv": df}, args("--metric", "converted", "--pre", "pre_revenue"),
               [{"check": "ab.pre_imbalance", "severity": "FAIL"}], True)

    # A8: clean control
    df = ab_users(rng, 20000, lift=0.25)
    write_case("ab-08-clean", "ab-test-check",
               "Clean control: per-user assignment, 50/50 split, real conversion lift.",
               {"users.csv": df}, args("--metric", "converted,revenue,sessions", "--unit", "user_id",
                                       "--time", "first_exposure", "--pre", "pre_revenue",
                                       "--expected-split", "50,50"),
               None, False)


# ---------------------------------------------------------------- analysis
def two_group_frame(rng, n_per, share_b_seg, rates):
    """Rows for groups A and B; share_b_seg[g] = share of group g in segment 'hard'; rates[(g, seg)] = outcome rate."""
    rows = []
    for g in ("A", "B"):
        seg = np.where(rng.random(n_per) < share_b_seg[g], "hard", "easy")
        p = np.array([rates[(g, s)] for s in seg])
        rows.append(pd.DataFrame({"group": g, "segment": seg, "outcome": (rng.random(n_per) < p).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def analysis_cases(rng):
    # N1: Simpson's paradox: A is better for every severity, B looks better overall
    df = two_group_frame(rng, 4000, {"A": 0.75, "B": 0.25},
                         {("A", "easy"): 0.93, ("A", "hard"): 0.73, ("B", "easy"): 0.87, ("B", "hard"): 0.69})
    df = df.rename(columns={"group": "treatment", "segment": "severity", "outcome": "recovered"})
    df["treatment"] = df["treatment"].map({"A": "surgery", "B": "medication"})
    df["age_band"] = rng.choice(["18-39", "40-64", "65+"], len(df))
    write_case("analysis-01-simpsons", "analysis-review",
               "Surgery recovers better for mild and severe cases, but gets most severe cases, so looks worse overall.",
               {"data.csv": df}, ["--data", "{dir}/data.csv", "--group", "treatment", "--outcome", "recovered",
                                  "--segment", "severity,age_band"],
               [{"check": "analysis.simpsons_paradox", "severity": "FAIL"}], True)

    # N2: the app "converts better" mostly because returning customers prefer the app
    df = two_group_frame(rng, 8000, {"A": 0.3, "B": 0.7},
                         {("A", "easy"): 0.05, ("A", "hard"): 0.15, ("B", "easy"): 0.055, ("B", "hard"): 0.155})
    df = df.rename(columns={"group": "channel", "segment": "customer_type", "outcome": "converted"})
    df["channel"] = df["channel"].map({"A": "web", "B": "app"})
    df["customer_type"] = df["customer_type"].map({"easy": "new", "hard": "returning"})
    write_case("analysis-02-mix-shift", "analysis-review",
               "App converts 5 points better overall, but only half a point of that is the channel; the rest is mix.",
               {"data.csv": df}, ["--data", "{dir}/data.csv", "--group", "channel", "--outcome", "converted",
                                  "--segment", "customer_type"],
               [{"check": "analysis.mix_shift", "severity": "WARN"}], True)

    # N3: the "worst" counties are simply the smallest ones
    sizes = np.concatenate([rng.integers(5, 13, 10), rng.integers(200, 5000, 50)])
    rows = [pd.DataFrame({"county": f"county_{i:02d}", "flagged": (rng.random(s) < 0.1).astype(int)})
            for i, s in enumerate(sizes)]
    df = pd.concat(rows, ignore_index=True)
    write_case("analysis-03-small-n", "analysis-review",
               "Every county has the same 10% true rate; the top and bottom of the ranking are tiny counties.",
               {"data.csv": df}, ["--data", "{dir}/data.csv", "--outcome", "flagged", "--segment", "county"],
               [{"check": "analysis.small_n_extremes", "severity": "WARN"}], True)

    # N4: west "spends more" because of five bulk orders
    n = 6000
    df = pd.DataFrame({"region": rng.choice(["east", "west"], n),
                       "order_value": np.round(rng.gamma(3, 20, n), 2)})
    df.loc[df.index[df["region"] == "west"][:5], "order_value"] = 9000.0
    write_case("analysis-04-outlier", "analysis-review",
               "Average order value looks higher in the west because of five bulk orders.",
               {"data.csv": df}, ["--data", "{dir}/data.csv", "--group", "region", "--outcome", "order_value"],
               [{"check": "analysis.outlier_driven", "severity": "WARN"}], True)

    # N5: clean control: same mix, consistent difference in every segment
    df = two_group_frame(rng, 5000, {"A": 0.4, "B": 0.4},
                         {("A", "easy"): 0.10, ("A", "hard"): 0.20, ("B", "easy"): 0.14, ("B", "hard"): 0.25})
    df["order_value"] = np.round(rng.gamma(3, 20, len(df)), 2)
    write_case("analysis-05-clean", "analysis-review",
               "Clean control: groups have the same mix and B is better in every segment.",
               {"data.csv": df}, ["--data", "{dir}/data.csv", "--group", "group", "--outcome", "outcome",
                                  "--segment", "segment"],
               None, False)


# ---------------------------------------------------------------- forecast
WEEKLY = np.array([0.8, 0.9, 0.95, 1.0, 1.1, 1.35, 1.25])


def daily_series(rng, days=420, horizon=56, level=200.0, noise=12.0, start="2025-01-06"):
    """Daily sales with a trend and weekly cycle; returns frame with history + eval rows and the true signal."""
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    signal = (level + 0.05 * t) * WEEKLY[dates.day_of_week]
    actual = np.round(signal + rng.normal(0, noise, days), 1)
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "units": actual, "forecast": np.nan})
    return df, signal, days - horizon


def forecast_cases(rng):
    base = ["--data", "{dir}/series.csv", "--date", "date", "--actual", "units", "--forecast", "forecast"]

    # F1: flat forecast ignores the weekly cycle and loses to "same day last week"
    df, signal, origin = daily_series(rng)
    df.loc[origin:, "forecast"] = round(float(df["units"].iloc[origin - 28:origin].mean()), 1)
    write_case("forecast-01-worse-than-naive", "forecast-check",
               "A flat forecast (last 4 weeks' mean) on a strongly weekly series.",
               {"series.csv": df}, base, [{"check": "forecast.no_better_than_naive", "severity": "FAIL"}], True)

    # F2: right shape, 9% too high
    df, signal, origin = daily_series(rng, noise=30.0)
    df.loc[origin:, "forecast"] = np.round(signal[origin:] * 1.09, 1)
    write_case("forecast-02-bias", "forecast-check",
               "The forecast tracks the weekly shape but is 9% high every day.",
               {"series.csv": df}, base, [{"check": "forecast.bias", "severity": "WARN"}], True)

    # F3: model was trained on the first two weeks of the test window
    df, signal, origin = daily_series(rng)
    df.loc[origin:, "forecast"] = np.round(signal[origin:], 1)
    train_end = df["date"].iloc[origin + 13]
    write_case("forecast-03-train-overlap", "forecast-check",
               "The training cut-off is two weeks into the evaluation window.",
               {"series.csv": df}, base + ["--train-end", train_end],
               [{"check": "forecast.train_overlap", "severity": "FAIL"}], True)

    # F4: part of the history was loaded twice
    df, signal, origin = daily_series(rng)
    df.loc[origin:, "forecast"] = np.round(signal[origin:], 1)
    df = pd.concat([df, df.iloc[200:210]], ignore_index=True).sort_values("date", kind="stable")
    write_case("forecast-04-duplicate-dates", "forecast-check",
               "Ten days of history were loaded twice.",
               {"series.csv": df}, base, [{"check": "forecast.duplicate_dates", "severity": "FAIL"}], True)

    # F5: forecast built from a feature that contains the actual
    df, signal, origin = daily_series(rng)
    df.loc[origin:, "forecast"] = np.round(df["units"].iloc[origin:] + rng.normal(0, 1, len(df) - origin), 1)
    write_case("forecast-05-too-good", "forecast-check",
               "A same-day feature leaks the actual; the forecast is almost perfect.",
               {"series.csv": df}, base, [{"check": "forecast.too_good", "severity": "WARN"}], True)

    # F6: intermittent demand with many zero days, scored with MAPE
    days, horizon = 200, 56
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "units": rng.poisson(0.8, days).astype(float),
                       "forecast": np.nan})
    df.loc[days - horizon:, "forecast"] = 0.8
    write_case("forecast-06-zeros", "forecast-check",
               "Spare-part demand: most days are zero, so MAPE is undefined.",
               {"series.csv": df}, base, [{"check": "forecast.mape_undefined", "severity": "WARN"}], True)

    # F7: clean control: three stores, model captures trend and seasonality, honest cut-off
    frames = []
    for store, level in (("s1", 200.0), ("s2", 450.0), ("s3", 120.0)):
        df, signal, origin = daily_series(rng, level=level)
        df.loc[origin:, "forecast"] = np.round(signal[origin:], 1)
        frames.append(df.assign(store=store))
        train_end = df["date"].iloc[origin - 1]
    df = pd.concat(frames, ignore_index=True)
    write_case("forecast-07-clean", "forecast-check",
               "Clean control: three stores, forecasts from the true pattern, test after the cut-off.",
               {"series.csv": df}, base + ["--series", "store", "--train-end", train_end], None, False)


# ------------------------------------------------------------------- sql
def sql_cases(rng):
    args = ["--sql", "{dir}/query.sql"]
    write_case("sql-01-left-join-filtered", "sql-review",
               "Revenue by segment 'including customers with no orders', but a WHERE on the orders table drops them.",
               {"query.sql": """-- Revenue by segment, including customers with no orders
SELECT c.segment, COUNT(o.order_id) AS orders, SUM(o.total) AS revenue
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id
WHERE o.order_date >= '2026-01-01'
GROUP BY c.segment;
"""}, args, [{"check": "sql.left_join_filtered", "severity": "FAIL"}], True)

    write_case("sql-02-null-comparison", "sql-review",
               "Open tickets filtered with '= NULL' (matches nothing) and excluded via NOT IN over a nullable column.",
               {"query.sql": """SELECT t.ticket_id, t.opened_at
FROM tickets t
WHERE t.closed_at = NULL
  AND t.customer_id NOT IN (SELECT customer_id FROM churned_customers);
"""}, args, [{"check": "sql.null_comparison", "severity": "FAIL"},
             {"check": "sql.not_in_nullable", "severity": "WARN"}], True)

    write_case("sql-03-silent-shrinkers", "sql-review",
               "UNION drops repeated rows, BETWEEN cuts the last day of a timestamp range, and a rate is integer division.",
               {"query.sql": """SELECT order_id, amount FROM web_orders
WHERE created_at BETWEEN '2026-01-01' AND '2026-01-31'
UNION
SELECT order_id, amount FROM store_orders
WHERE created_at BETWEEN '2026-01-01' AND '2026-01-31';

SELECT region, SUM(converted) / COUNT(*) AS conversion_rate
FROM sessions
WHERE channel = 'paid' AND device = 'mobile' OR device = 'tablet'
GROUP BY region;
"""}, args, [{"check": "sql.union_dedup", "severity": "WARN"},
             {"check": "sql.between_timestamp", "severity": "WARN"},
             {"check": "sql.integer_division", "severity": "WARN"},
             {"check": "sql.and_or_precedence", "severity": "WARN"}], True)

    write_case("sql-04-clean", "sql-review",
               "Clean control: aggregate before joining, filter in the CTE, explicit decimal division.",
               {"query.sql": """WITH order_totals AS (
  SELECT customer_id, SUM(total) AS revenue, COUNT(*) AS orders
  FROM orders
  WHERE order_date >= DATE '2026-01-01'
  GROUP BY customer_id
)
SELECT c.segment,
       SUM(COALESCE(t.revenue, 0)) AS revenue,
       SUM(COALESCE(t.orders, 0)) * 1.0 / COUNT(*) AS orders_per_customer
FROM customers c
LEFT JOIN order_totals t ON t.customer_id = c.customer_id
WHERE c.is_test = FALSE
GROUP BY c.segment
ORDER BY revenue DESC;
"""}, args, None, False)


# ---------------------------------------------------------------- profile
def customer_frame(rng, n=3000):
    return pd.DataFrame({
        "customer_id": np.arange(n) + 10_000,
        "signup_date": (pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")).strftime("%Y-%m-%d"),
        "customer_age": rng.integers(18, 85, n),
        "state": rng.choice(["NC", "SC", "VA", "GA", "TN"], n),
        "orders": rng.poisson(3, n),
        "income": np.round(rng.gamma(4, 15000, n), 0),
    })


def profile_cases(rng):
    args = ["--data", "{dir}/data.csv", "--as-of", "2026-06-30"]
    df = customer_frame(rng)
    idx = rng.choice(len(df), 60, replace=False)
    df.loc[idx[:40], "orders"] = -rng.integers(1, 4, 40)
    df.loc[idx[40:], "customer_age"] = rng.integers(150, 260, 20)
    write_case("profile-01-impossible", "data-profile",
               "Negative order counts and ages above 120.",
               {"data.csv": df}, args + ["--id", "customer_id"],
               [{"check": "profile.impossible_values", "severity": "FAIL"}], True)

    df = customer_frame(rng)
    variants = {"NC": ["NC", "nc", " NC", "Nc"], "SC": ["SC", "sc "]}
    df["state"] = [rng.choice(variants[s]) if s in variants else s for s in df["state"]]
    df["income"] = df["income"].astype(object)
    df.loc[rng.choice(len(df), 90, replace=False), "income"] = -999
    df.loc[rng.choice(len(df), 40, replace=False), "state"] = "N/A"
    write_case("profile-02-messy", "data-profile",
               "State written several ways, 'N/A' placeholders, and -999 income codes.",
               {"data.csv": df}, args,
               [{"check": "profile.inconsistent_labels", "severity": "WARN"},
                {"check": "profile.placeholder_number", "severity": "WARN"},
                {"check": "profile.placeholder_text", "severity": "WARN"}], True)

    df = customer_frame(rng)
    df.loc[rng.choice(len(df), 50, replace=False), "customer_id"] = df["customer_id"].iloc[:50].to_numpy()
    write_case("profile-03-id-not-unique", "data-profile",
               "customer_id is supposed to be unique, but 50 rows reuse an existing ID.",
               {"data.csv": df}, args + ["--id", "customer_id"],
               [{"check": "profile.id_not_unique", "severity": "FAIL"}], True)

    write_case("profile-04-clean", "data-profile", "Clean control: tidy customer table.",
               {"data.csv": customer_frame(rng)}, args + ["--id", "customer_id"], None, False)


# --------------------------------------------------------------- contract
ORDER_CONTRACT = {
    "primary_key": ["order_id"],
    "columns": {
        "order_id": {"type": "integer", "nullable": False},
        "order_date": {"type": "date", "max": "today"},
        "amount": {"type": "number", "min": 0, "max": 100000, "nullable": False},
        "status": {"type": "string", "allowed": ["paid", "refunded", "cancelled"]},
        "email": {"type": "string", "pattern": "^[^@\\s]+@[^@\\s]+$", "nullable": True},
    },
    "row_count": {"min": 1000},
    "freshness": {"column": "order_date", "max_age_days": 2},
}


def orders_extract(rng, n=3000, end="2026-06-29"):
    return pd.DataFrame({
        "order_id": np.arange(n) + 900_000,
        "order_date": (pd.Timestamp(end) - pd.to_timedelta(rng.integers(0, 30, n), unit="D")).strftime("%Y-%m-%d"),
        "amount": np.round(rng.gamma(2.5, 40, n), 2),
        "status": rng.choice(["paid", "refunded", "cancelled"], n, p=[0.9, 0.07, 0.03]),
        "email": [f"c{i}@example.com" for i in rng.integers(0, 10**6, n)],
    })


def contract_cases(rng):
    args = ["--data", "{dir}/orders.csv", "--contract", "{dir}/contract.json", "--as-of", "2026-06-30"]
    df = orders_extract(rng)
    df.loc[rng.choice(len(df), 80, replace=False), "status"] = "chargeback"
    df.loc[rng.choice(len(df), 25, replace=False), "amount"] = -np.round(rng.gamma(2, 20, 25), 2)
    write_case("contract-01-new-code-and-negatives", "data-contract",
               "Upstream added a 'chargeback' status and sends negative amounts.",
               {"orders.csv": df, "contract.json": ORDER_CONTRACT}, args,
               [{"check": "contract.allowed_values", "severity": "FAIL"}, {"check": "contract.range", "severity": "FAIL"}],
               True)

    df = orders_extract(rng, end="2026-06-18")
    df = pd.concat([df, df.iloc[:40]], ignore_index=True)
    write_case("contract-02-stale-and-duplicated", "data-contract",
               "The feed stopped 12 days ago and part of the last load was duplicated.",
               {"orders.csv": df, "contract.json": ORDER_CONTRACT}, args,
               [{"check": "contract.freshness", "severity": "FAIL"}, {"check": "contract.primary_key", "severity": "FAIL"}],
               True)

    write_case("contract-03-clean", "data-contract", "Clean control: a delivery that meets the contract.",
               {"orders.csv": orders_extract(rng), "contract.json": ORDER_CONTRACT}, args, None, False)


# -------------------------------------------------------------- reconcile
def reconcile_cases(rng):
    def base():
        df = orders_extract(rng, n=4000)
        return df.drop(columns=["email"])

    args = ["--source", "{dir}/source.csv", "--target", "{dir}/target.csv", "--key", "order_id",
            "--measure", "amount", "--by", "order_date"]
    src = base()
    day = src["order_date"].value_counts().index[0]
    tgt = src[src["order_date"] != day]
    write_case("reconcile-01-missing-partition", "reconcile",
               "One day's partition failed to load into the warehouse.",
               {"source.csv": src, "target.csv": tgt}, args,
               [{"check": "reconcile.missing_in_target", "severity": "FAIL"},
                {"check": "reconcile.measure_total", "severity": "FAIL"}], True)

    src = base()
    tgt = src.copy()
    tgt["amount"] = tgt["amount"].round(0)
    write_case("reconcile-02-rounding", "reconcile",
               "The new pipeline casts amount to an integer type and loses the cents.",
               {"source.csv": src, "target.csv": tgt}, args,
               [{"check": "reconcile.value_mismatch", "severity": "FAIL"}], True)

    src = base()
    tgt = src.sample(frac=1, random_state=SEED).copy()
    tgt["amount"] = tgt["amount"].map(lambda v: f"{v:.4f}")
    tgt["status"] = tgt["status"] + " "
    write_case("reconcile-03-clean", "reconcile",
               "Clean control: same data in a different row order, with different number formatting and trailing spaces.",
               {"source.csv": src, "target.csv": tgt}, args, None, False)


# ------------------------------------------------------------ idempotency
def idempotency_cases(rng):
    def run():
        n = 2500
        return pd.DataFrame({"customer_id": np.arange(n) + 50_000,
                             "lifetime_value": np.round(rng.gamma(3, 120, n), 2),
                             "latest_status": rng.choice(["active", "paused", "closed"], n, p=[0.8, 0.15, 0.05]),
                             "loaded_at": "2026-06-30T02:00:00Z"})

    args = ["--first", "{dir}/run1.csv", "--second", "{dir}/run2.csv", "--key", "customer_id"]
    first = run()
    second = pd.concat([first, first], ignore_index=True).assign(loaded_at="2026-06-30T03:00:00Z")
    write_case("idempotency-01-append", "pipeline-idempotency",
               "The retry appended a second full copy instead of overwriting.",
               {"run1.csv": first, "run2.csv": second}, args + ["--ignore", "loaded_at"],
               [{"check": "idempotency.duplicated_on_rerun", "severity": "FAIL"}], True)

    first = run()
    second = first.copy()
    flip = rng.random(len(second)) < 0.15
    second.loc[flip, "latest_status"] = rng.choice(["active", "paused", "closed"], int(flip.sum()))
    second = second.assign(loaded_at="2026-06-30T03:00:00Z")
    write_case("idempotency-02-nondeterministic", "pipeline-idempotency",
               "'Latest status' is picked with ROW_NUMBER() over a non-unique timestamp, so ties resolve differently.",
               {"run1.csv": first, "run2.csv": second}, args + ["--ignore", "loaded_at"],
               [{"check": "idempotency.nondeterministic", "severity": "FAIL"}], True)

    first = run()
    second = first.sample(frac=1, random_state=SEED).assign(loaded_at="2026-06-30T03:00:00Z")
    write_case("idempotency-03-clean", "pipeline-idempotency",
               "Clean control: same output in a different order; only the load timestamp differs.",
               {"run1.csv": first, "run2.csv": second}, args, None, False)


# --------------------------------------------------------------- datetime
def datetime_cases(rng):
    n = 2000
    base = pd.Timestamp("2026-03-01") + pd.to_timedelta(rng.integers(0, 120, n), unit="D")
    us = base[: n // 2].strftime("%m/%d/%Y")
    eu = base[n // 2:].strftime("%d/%m/%Y")
    df = pd.DataFrame({"order_id": np.arange(n), "order_date": list(us) + list(eu)})
    write_case("datetime-01-mixed-day-month", "datetime-check",
               "US (mm/dd) and EU (dd/mm) store exports were concatenated.",
               {"data.csv": df}, ["--data", "{dir}/data.csv", "--as-of", "2026-12-31"],
               [{"check": "datetime.mixed_day_month", "severity": "FAIL"}], True)

    secs = int(pd.Timestamp("2026-05-01", tz="UTC").timestamp()) + rng.integers(0, 86400 * 30, n)
    ts = np.where(rng.random(n) < 0.3, secs * 1000, secs)
    df = pd.DataFrame({"event_id": np.arange(n), "event_ts": ts})
    write_case("datetime-02-epoch-units", "datetime-check",
               "The mobile app logs epoch milliseconds; the web app logs seconds.",
               {"data.csv": df}, ["--data", "{dir}/data.csv", "--as-of", "2026-12-31"],
               [{"check": "datetime.mixed_epoch_units", "severity": "FAIL"}], True)

    t = pd.Timestamp("2026-05-01") + pd.to_timedelta(rng.integers(0, 86400 * 30, n), unit="s")
    aware = t.strftime("%Y-%m-%dT%H:%M:%SZ")
    naive = t.strftime("%Y-%m-%d %H:%M:%S")
    vals = np.where(rng.random(n) < 0.4, naive, aware).astype(object)
    vals[rng.choice(n, 30, replace=False)] = "1900-01-01T00:00:00Z"
    df = pd.DataFrame({"id": np.arange(n), "created_at": vals})
    write_case("datetime-03-mixed-timezone", "datetime-check",
               "A new service writes local time without an offset; legacy rows are UTC. Some rows carry 1900-01-01.",
               {"data.csv": df}, ["--data", "{dir}/data.csv", "--as-of", "2026-12-31"],
               [{"check": "datetime.mixed_timezone", "severity": "FAIL"},
                {"check": "datetime.sentinel", "severity": "WARN"}], True)

    df = pd.DataFrame({"id": np.arange(n), "created_at": aware, "order_date": base.strftime("%Y-%m-%d")})
    write_case("datetime-04-clean", "datetime-check", "Clean control: ISO dates and UTC timestamps.",
               {"data.csv": df}, ["--data", "{dir}/data.csv", "--as-of", "2026-12-31"], None, False)


# -------------------------------------------------------------------- pii
def luhn_card(rng, prefix="4"):
    digits = [int(c) for c in prefix] + list(rng.integers(0, 10, 15 - len(prefix)))
    total = 0
    for i, d in enumerate(reversed(digits)):
        d = d * 2 if i % 2 == 0 else d
        total += d - 9 if d > 9 else d
    return "".join(map(str, digits)) + str((10 - total % 10) % 10)


def pii_cases(rng):
    n = 1500
    df = pd.DataFrame({
        "customer_key": np.arange(n),
        "tax_ref": [f"{rng.integers(100, 665)}-{rng.integers(10, 99)}-{rng.integers(1000, 9999)}" for _ in range(n)],
        "payment_ref": [luhn_card(rng) for _ in range(n)],
        "plan": rng.choice(["basic", "plus"], n),
    })
    write_case("pii-01-ssn-and-cards", "pii-scan",
               "An 'anonymised' export still carries SSNs and card numbers under innocent column names.",
               {"data.csv": df}, ["--data", "{dir}/data.csv"],
               [{"check": "pii.ssn", "severity": "FAIL"}, {"check": "pii.credit_card", "severity": "FAIL"}], True)

    notes = []
    for i in range(n):
        r = rng.random()
        if r < 0.05:
            notes.append(f"Customer asked for a callback, reach them at jane.doe{i}@mail.com after 5pm")
        elif r < 0.09:
            notes.append(f"Left voicemail on (919) 555-{rng.integers(1000, 9999)} about the delayed shipment")
        else:
            notes.append("Customer reported the delivery arrived late and the box was damaged on arrival")
    df = pd.DataFrame({"ticket_id": np.arange(n), "category": rng.choice(["shipping", "billing"], n), "notes": notes})
    write_case("pii-02-free-text", "pii-scan",
               "Support-ticket notes contain emails and phone numbers typed by agents.",
               {"data.csv": df}, ["--data", "{dir}/data.csv"],
               [{"check": "pii.free_text", "severity": "WARN"}], True)

    import hashlib
    df = pd.DataFrame({
        "customer_key": [hashlib.sha256(f"salt{i}".encode()).hexdigest() for i in range(n)],
        "segment": rng.choice(["consumer", "pro"], n),
        "state": rng.choice(["NC", "SC", "VA"], n),
        "signup_month": rng.choice(["2026-01", "2026-02", "2026-03"], n),
        "revenue": np.round(rng.gamma(2, 50, n), 2),
    })
    write_case("pii-03-clean", "pii-scan", "Clean control: hashed keys and coarse attributes only.",
               {"data.csv": df}, ["--data", "{dir}/data.csv"], None, False)


# ------------------------------------------------------------------- skew
def feature_frame(rng, n, start_id=0):
    return pd.DataFrame({
        "user_id": np.arange(n) + start_id,
        "amount_30d": np.round(rng.gamma(3, 40, n), 2),
        "orders_30d": rng.poisson(4, n),
        "tenure_days": rng.integers(1, 1500, n),
        "device": rng.choice(["ios", "android", "web"], n, p=[0.4, 0.35, 0.25]),
    })


def skew_cases(rng):
    args = ["--train", "{dir}/train.csv", "--serve", "{dir}/serve.csv", "--exclude", "user_id"]
    tr = feature_frame(rng, 4000)
    sv = feature_frame(rng, 2000, 900_000)
    sv["amount_30d"] = (sv["amount_30d"] * 100).round(0)
    write_case("skew-01-units", "train-serve-skew",
               "The online feature store returns amounts in cents; training used dollars.",
               {"train.csv": tr, "serve.csv": sv}, args,
               [{"check": "skew.scale", "severity": "FAIL"}], True)

    tr = feature_frame(rng, 3000)
    sv = tr.copy()
    sv["orders_30d"] = rng.binomial(sv["orders_30d"], 0.25)          # serving uses a 7-day window
    write_case("skew-02-window-mismatch", "train-serve-skew",
               "For the same users, serving computes orders over 7 days instead of 30.",
               {"train.csv": tr, "serve.csv": sv},
               ["--train", "{dir}/train.csv", "--serve", "{dir}/serve.csv", "--key", "user_id"],
               [{"check": "skew.value_mismatch", "severity": "FAIL"}], True)

    tr = feature_frame(rng, 4000)
    sv = feature_frame(rng, 2000, 900_000)
    sv["device"] = rng.choice(["iOS", "Android", "web"], len(sv), p=[0.4, 0.35, 0.25])
    sv.loc[rng.random(len(sv)) < 0.2, "tenure_days"] = np.nan
    write_case("skew-03-categories-and-nulls", "train-serve-skew",
               "Serving sends 'iOS'/'Android' (training had 'ios'/'android') and tenure is missing for 20% of requests.",
               {"train.csv": tr, "serve.csv": sv}, args,
               [{"check": "skew.unseen_categories", "severity": "WARN"}, {"check": "skew.null_rate", "severity": "WARN"}],
               True)

    write_case("skew-04-clean", "train-serve-skew", "Clean control: serving sample from the same population.",
               {"train.csv": feature_frame(rng, 4000), "serve.csv": feature_frame(rng, 2000, 900_000)}, args,
               None, False)


# ------------------------------------------------------------ calibration
def calibration_cases(rng):
    n = 6000
    args = ["--pred", "{dir}/preds.csv", "--y-true", "label", "--y-score", "prob"]

    z = rng.normal(-3.2, 1.0, n)
    p_true = 1 / (1 + np.exp(-z))
    y = (rng.random(n) < p_true).astype(int)
    p_model = 1 / (1 + np.exp(-(z + np.log(19))))          # trained on 50/50 down-sampled data
    write_case("calibration-01-downsampled", "calibration-check",
               "Model trained on a 50/50 down-sample: predicts ~30% where ~5% actually happen.",
               {"preds.csv": pd.DataFrame({"label": y, "prob": np.round(p_model, 5)})}, args,
               [{"check": "calibration.mean_mismatch", "severity": "FAIL"}], True)

    z = rng.normal(0, 1.3, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    write_case("calibration-02-overconfident", "calibration-check",
               "Boosted model without calibration: scores are far more extreme than reality.",
               {"preds.csv": pd.DataFrame({"label": y, "prob": np.round(1 / (1 + np.exp(-2.5 * z)), 5)})}, args,
               [{"check": "calibration.overconfident", "severity": "WARN"}], True)

    z = rng.normal(-1, 1, n)
    region = rng.choice(["north", "south", "east", "west"], n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    p = 1 / (1 + np.exp(-(z + np.where(region == "west", 1.2, 0))))
    write_case("calibration-03-segment", "calibration-check",
               "Overall level is roughly right, but the west region is over-predicted by a wide margin.",
               {"preds.csv": pd.DataFrame({"label": y, "prob": np.round(p, 5), "region": region})},
               args + ["--segment", "region"], [{"check": "calibration.segment", "severity": "WARN"}], True)

    z = rng.normal(-1, 1, n)
    p = 1 / (1 + np.exp(-z))
    y = (rng.random(n) < p).astype(int)
    write_case("calibration-04-clean", "calibration-check", "Clean control: well-calibrated probabilities.",
               {"preds.csv": pd.DataFrame({"label": y, "prob": np.round(p, 5), "region": region})},
               args + ["--segment", "region"], None, False)


# --------------------------------------------------------------- fairness
def fairness_cases(rng):
    n = 6000
    group = rng.choice(["A", "B"], n)
    approve = (rng.random(n) < np.where(group == "A", 0.50, 0.30)).astype(int)
    write_case("fairness-01-disparate-impact", "fairness-audit",
               "Group B is approved at 30% vs 50% for group A (ratio 0.6).",
               {"decisions.csv": pd.DataFrame({"applicant_group": group, "approved": approve})},
               ["--pred", "{dir}/decisions.csv", "--group", "applicant_group", "--y-pred", "approved"],
               [{"check": "fairness.disparate_impact", "severity": "FAIL"}], True)

    group = rng.choice(["A", "B"], n)
    y = (rng.random(n) < 0.4).astype(int)
    tpr = np.where(group == "A", 0.80, 0.55)
    fpr = np.where(group == "A", 0.10, 0.25)
    pred = np.where(y == 1, rng.random(n) < tpr, rng.random(n) < fpr).astype(int)
    write_case("fairness-02-error-rate-gaps", "fairness-audit",
               "Similar approval rates, but qualified B applicants are missed far more often.",
               {"decisions.csv": pd.DataFrame({"applicant_group": group, "approved": pred, "repaid": y})},
               ["--pred", "{dir}/decisions.csv", "--group", "applicant_group", "--y-pred", "approved", "--y-true", "repaid"],
               [{"check": "fairness.equal_opportunity", "severity": "WARN"},
                {"check": "fairness.false_positive_gap", "severity": "WARN"}], True)

    group = rng.choice(["A", "B", "C"], n)
    y = (rng.random(n) < 0.4).astype(int)
    pred = np.where(y == 1, rng.random(n) < 0.8, rng.random(n) < 0.1).astype(int)
    write_case("fairness-03-clean", "fairness-audit", "Clean control: same error rates in every group.",
               {"decisions.csv": pd.DataFrame({"applicant_group": group, "approved": pred, "repaid": y})},
               ["--pred", "{dir}/decisions.csv", "--group", "applicant_group", "--y-pred", "approved", "--y-true", "repaid"],
               None, False)


# ----------------------------------------------------------------- cohort
def retention_table(rng, months=12, as_of="2026-06-15", include_partial=True, size=800):
    rows = []
    start = pd.Timestamp("2025-07-01")
    asof = pd.Timestamp(as_of)
    for c in range(months):
        cohort = start + pd.DateOffset(months=c)
        n = int(size * rng.uniform(0.8, 1.2))
        for p in range(months - c):
            end = cohort + pd.DateOffset(months=p + 1)
            true_rate = 1.0 if p == 0 else 0.55 * 0.9 ** (p - 1)
            active = rng.binomial(n, true_rate)
            if end > asof:
                if not include_partial:
                    continue
                elapsed = max((asof - (cohort + pd.DateOffset(months=p))).days, 0) / 30
                active = int(active * min(elapsed, 1) * 0.6)     # not everyone has come back yet
            rows.append({"cohort_month": cohort.strftime("%Y-%m-%d"), "month_number": p,
                         "active_users": active, "cohort_size": n})
    return pd.DataFrame(rows)


def cohort_cases(rng):
    args = ["--table", "{dir}/retention.csv", "--cohort", "cohort_month", "--period", "month_number",
            "--active", "active_users", "--size", "cohort_size", "--grain", "month", "--as-of", "2026-06-15"]
    write_case("cohort-01-incomplete-periods", "cohort-retention",
               "Recent cohorts include months that have not finished, so they look like a retention collapse.",
               {"retention.csv": retention_table(rng)}, args,
               [{"check": "cohort.incomplete_period", "severity": "FAIL"}], True)

    df = retention_table(rng, include_partial=False)
    late = df["month_number"] >= 3
    df.loc[late, "active_users"] = (df.loc[late, "active_users"] * 2.2).astype(int)   # counted sessions, not users
    write_case("cohort-02-over-100", "cohort-retention",
               "Active 'users' were counted from sessions, so some cells exceed 100% and retention rises over time.",
               {"retention.csv": df}, args, [{"check": "cohort.rate_out_of_range", "severity": "FAIL"}], True)

    write_case("cohort-03-clean", "cohort-retention",
               "Clean control: only completed periods, distinct users.",
               {"retention.csv": retention_table(rng, include_partial=False)}, args, None, False)


# ------------------------------------------------------------- regression
def regression_cases(rng):
    n = 800
    args = ["--data", "{dir}/data.csv", "--target", "sales"]
    x1, x2, x3 = rng.normal(50, 10, n), rng.normal(20, 5, n), rng.normal(0, 1, n)
    sales = 10 + 2 * x1 - 1.5 * x2 + 3 * x3 + rng.normal(0, 5, n)
    df = pd.DataFrame({"tv_spend": x1, "radio_spend": x2, "total_spend": x1 + x2, "promo_index": x3,
                       "sales": sales}).round(4)
    write_case("regression-01-perfect-collinearity", "regression-diagnostics",
               "total_spend = tv_spend + radio_spend is included next to its parts.",
               {"data.csv": df}, args, [{"check": "regression.perfect_collinearity", "severity": "FAIL"}], True)

    x = rng.uniform(1, 100, n)
    sales = 5 + 0.8 * x + rng.normal(0, 0.15 * x, n)
    write_case("regression-02-heteroscedastic", "regression-diagnostics",
               "Errors grow with store size, so standard errors from plain OLS are wrong.",
               {"data.csv": pd.DataFrame({"store_size": x, "sales": sales}).round(4)}, args,
               [{"check": "regression.heteroscedasticity", "severity": "WARN"}], True)

    price = rng.uniform(1, 10, n)
    sales = 200 - 40 * price + 3.5 * price ** 2 + rng.normal(0, 4, n)
    write_case("regression-03-nonlinear", "regression-diagnostics",
               "Demand is curved in price, but a straight line is fitted.",
               {"data.csv": pd.DataFrame({"price": price, "sales": sales}).round(4)}, args,
               [{"check": "regression.nonlinearity", "severity": "WARN"}], True)

    x1, x2, x3 = rng.normal(50, 10, n), rng.normal(20, 5, n), rng.normal(0, 1, n)
    df = pd.DataFrame({"tv_spend": x1, "radio_spend": x2, "promo_index": x3,
                       "sales": 10 + 2 * x1 - 1.5 * x2 + 3 * x3 + rng.normal(0, 5, n)}).round(4)
    write_case("regression-04-clean", "regression-diagnostics", "Clean control: linear, independent features.",
               {"data.csv": df}, args, None, False)


# --------------------------------------------------------------- notebook
def nb(cells):
    out = []
    for src, count, outputs in cells:
        out.append({"cell_type": "code", "execution_count": count, "metadata": {}, "outputs": outputs or [],
                    "source": [l + "\n" for l in src.strip("\n").split("\n")]})
    return {"cells": out, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "nbformat": 4, "nbformat_minor": 5}


def notebook_cases(rng):
    args = ["--notebook", "{dir}/analysis.ipynb"]
    cells = [
        ("import pandas as pd\nfrom sklearn.model_selection import train_test_split", 1, None),
        ("X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)", 7, None),
        ("df = pd.read_csv('data/churn.csv')\nfeatures = df.drop(columns=['churned'])\ntarget = df['churned']", 3, None),
        ("model.fit(X_train, y_train)", 9, [{"output_type": "error", "ename": "NameError",
                                             "evalue": "name 'model' is not defined", "traceback": []}]),
        ("from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(max_iter=500)", 8, None),
    ]
    write_case("notebook-01-out-of-order", "notebook-reproducibility",
               "Cells were run in a different order than they appear; one has a saved NameError.",
               {"analysis.ipynb": nb(cells)}, args,
               [{"check": "notebook.out_of_order", "severity": "WARN"},
                {"check": "notebook.use_before_define", "severity": "FAIL"},
                {"check": "notebook.error_outputs", "severity": "FAIL"}], True)

    cells = [
        ("!pip install xgboost lightgbm\nimport pandas as pd\nimport numpy as np", 1, None),
        ("df = pd.read_csv('/Users/alex/Downloads/sales_final_v3.csv')", 2, None),
        ("sample = df.sample(frac=0.1)\nnoise = np.random.normal(0, 1, len(sample))", 3, None),
        ("from sklearn.ensemble import RandomForestRegressor\n"
         "rf = RandomForestRegressor(n_estimators=200)\nrf.fit(sample[['price']], sample['units'])", 4, None),
    ]
    write_case("notebook-02-unreproducible", "notebook-reproducibility",
               "No seeds, an absolute path to someone's Downloads folder, unpinned installs.",
               {"analysis.ipynb": nb(cells)}, args,
               [{"check": "notebook.no_seed", "severity": "WARN"}, {"check": "notebook.absolute_paths", "severity": "WARN"},
                {"check": "notebook.unpinned_install", "severity": "WARN"}], True)

    fake_key = "sk-" + "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), 40))
    cells = [
        ("import os\nimport requests", 1, None),
        (f"API_KEY = '{fake_key}'\nresp = requests.get('https://api.example.com/v1/data', "
         "headers={'Authorization': API_KEY})", 2, None),
    ]
    write_case("notebook-03-secret", "notebook-reproducibility",
               "An API key is pasted directly into a cell.",
               {"analysis.ipynb": nb(cells)}, args, [{"check": "notebook.secrets", "severity": "FAIL"}], True)

    cells = [
        ("from pathlib import Path\nimport numpy as np\nimport pandas as pd\n"
         "from sklearn.model_selection import train_test_split\n"
         "from sklearn.ensemble import RandomForestClassifier\nDATA = Path('data')\nSEED = 42", 1, None),
        ("df = pd.read_csv(DATA / 'churn.csv')\nfeatures = df.drop(columns=['churned'])\ntarget = df['churned']", 2, None),
        ("X_train, X_test, y_train, y_test = train_test_split(\n    features, target, test_size=0.2, random_state=SEED)",
         3, None),
        ("model = RandomForestClassifier(n_estimators=300, random_state=SEED)\nmodel.fit(X_train, y_train)\n"
         "score = model.score(X_test, y_test)\nfor name, imp in zip(features.columns, model.feature_importances_):\n"
         "    print(name, round(imp, 3))", 4, [{"output_type": "stream", "name": "stdout", "text": ["tenure 0.41\n"]}]),
    ]
    write_case("notebook-04-clean", "notebook-reproducibility", "Clean control: ordered, seeded, relative paths.",
               {"analysis.ipynb": nb(cells)}, args, None, False)


def main():
    if CASES.exists():
        shutil.rmtree(CASES)
    rng = np.random.default_rng(SEED)
    leakage_cases(rng)
    eval_cases(rng)
    join_cases(rng)
    drift_cases(rng)
    ab_cases(rng)
    analysis_cases(rng)
    forecast_cases(rng)
    sql_cases(rng)
    profile_cases(rng)
    contract_cases(rng)
    reconcile_cases(rng)
    idempotency_cases(rng)
    datetime_cases(rng)
    pii_cases(rng)
    skew_cases(rng)
    calibration_cases(rng)
    fairness_cases(rng)
    cohort_cases(rng)
    regression_cases(rng)
    notebook_cases(rng)
    n = len(list(CASES.glob("*/case.json")))
    print(f"wrote {n} cases to {CASES}")


if __name__ == "__main__":
    main()
