---
name: leakage-audit
description: Audit a supervised ML dataset or training pipeline for data leakage before trusting any model result. Use when building, training or evaluating a model, when metrics look surprisingly good (AUC above ~0.9, near-perfect accuracy), when splitting data into train and test, or before reporting model performance to anyone.
---

# Leakage audit

Treat every model metric as unverified until this audit passes. Leakage makes a model look good in evaluation and fail in production, and the code still runs without errors, so nobody notices.

## When to run it

- Before reporting any model metric.
- Right after creating a train/test split.
- Whenever a result looks too good: AUC above 0.9 on a hard problem, a big jump from a small change, or one feature dominating importance.

## Steps

1. **Pin down three facts and write them in your answer**:
   - the **target** and how it is defined,
   - the **prediction moment**: when would this model be called in production, and what is known at that moment,
   - the **entity**: the thing rows belong to (customer, patient, order, store, device).

2. **Run the deterministic checks** (requires pandas and numpy):

   ```bash
   python <skill-dir>/scripts/check_leakage.py \
     --data <train-or-combined file> --target <col> \
     [--test <test file> | --split <train/test column>] \
     [--entity <id col>] [--time <prediction-time col>] [--ignore <other id cols>]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. Add `--json` for structured output. Exit code 1 means at least one FAIL. If the data lives in a database, export the modelling table (or a large sample) to CSV/Parquet first.

3. **Read every finding and act on it**:

   | Finding | What it usually means | Fix |
   | --- | --- | --- |
   | `single_feature` FAIL | A column is a disguised copy of the target (e.g. `account_status = closed` for churn) | Drop it, or rebuild it from data available at the prediction moment |
   | `entity_overlap` | The same customer/patient is in train and test; the model memorises them | Use a group split (`GroupKFold`, `GroupShuffleSplit`) on the entity |
   | `time_overlap` | Random split on time-ordered data; the model trains on the future | Split by time: train before a cutoff, test after it |
   | `future_timestamp` | A date column is later than the prediction moment | Remove it or cap features at the prediction time |
   | `duplicate_rows` | Identical rows on both sides of the split | Deduplicate before splitting |
   | `suspicious_name` (WARN) | Name hints at post-outcome data | Confirm when the value is populated |

4. **Check what the script cannot see** by reading the pipeline code:
   - Scaling, imputation, encoding, feature selection or SMOTE fit on the **full** dataset before splitting. They must be fit on training data only, ideally inside a `Pipeline`.
   - Target encoding computed without out-of-fold logic.
   - The test set used to choose features, hyperparameters or the decision threshold.
   - Aggregates ("customer lifetime spend", "patient total visits") computed over the whole history instead of up to the prediction moment.
   - Labels and features drawn from different snapshots.

   See `references/leakage-patterns.md` for worked examples.

5. **Re-run after fixing** and compare metrics. A large drop is expected and is the honest number.

## Output

Report a short table: each check, PASS/WARN/FAIL, the columns involved and the fix applied. **Never report model metrics alongside an unresolved FAIL.** If you cannot fix a FAIL, say so plainly and label any metric as unreliable.
