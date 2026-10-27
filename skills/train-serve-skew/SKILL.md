---
name: train-serve-skew
description: Compare the features a model was trained on with the features it gets in production or batch scoring — finds missing features, type changes, unit and scaling mismatches (a scaler applied on one path only), distribution shift (PSI), unseen categories, null-rate changes, and features computed differently for the same entity offline vs online. Use whenever deploying or monitoring a model, rewriting feature code for serving, moving from notebook to pipeline, or when a model performs worse in production than in evaluation.
---

# Train/serve skew

A model that scored well offline and disappoints in production usually is not "drifting". More often it is being fed different features: a value in cents instead of dollars, a 7-day window instead of 30, a scaler that only runs in the notebook.

## Steps

1. **Collect both sides**: the training feature table, and a sample of what the model actually receives when serving (request logs, the scoring job's input, or the feature store's online values). If possible, log serving features with the entity id and timestamp so rows can be matched.

2. **Run the comparison**:

   ```bash
   python <skill-dir>/scripts/check_skew.py --train <training features> --serve <serving features> \
     [--key <entity id[,timestamp]>] [--exclude <label,ids>] [--features <cols>]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. With `--key`, rows for the same entity are compared value by value, which is the strongest test. Exit code 1 means a FAIL. Add `--json` for structured output.

3. **Act on the findings**:

   | Finding | What to do |
   | --- | --- |
   | `missing_feature` / `type` FAIL | Fix the serving pipeline before the model goes live |
   | `scale` / `scaling` FAIL | Make both paths use the same units, and ship the fitted preprocessing (scaler, encoder) inside the model artifact |
   | `value_mismatch` FAIL | Offline and online logic differ: compare window lengths, time zones, joins and point-in-time cut-offs; compute features once and share the code |
   | `unseen_categories` | Map renamed codes; decide how unknown categories are handled |
   | `null_rate` | Check serving joins and defaults; the model was not trained on that many missing values |
   | `distribution` | If the logic is identical, the population changed: re-evaluate the model on recent data |

4. **Re-run after the fix** and keep this check in the deployment pipeline, comparing a fresh serving sample against the training set.

## Output

A table of features with PSI and any findings, the matched-row mismatch rate if `--key` was used, and the verdict. Do not attribute production underperformance to "drift" before this check passes.
