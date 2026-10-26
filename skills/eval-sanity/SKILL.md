---
name: eval-sanity
description: Check that a model evaluation is meaningful before results are shared or a model is chosen. Use whenever reporting model metrics, comparing two models or versions, picking a decision threshold, or claiming that a model "improved" — for classification and regression.
---

# Evaluation sanity check

A metric is only meaningful next to a baseline, an uncertainty range and the segments it averages over. This skill makes those three things non-optional.

## Steps

1. **Run leakage-audit first** if it has not been run on this model. A clean evaluation of a leaky model is still wrong.

2. **Save test-set predictions** to a file with one row per test example: the true label, the model's score or prediction, any comparison model's score, and the segment columns the business cares about (region, product line, customer type, age band).

3. **Run the checks** (requires pandas, numpy, scikit-learn):

   ```bash
   python <skill-dir>/scripts/check_eval.py --pred <predictions file> --y-true <col> \
     [--y-score <col>] [--y-pred <col>] [--compare-score <col>] [--segment <col> ...]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. Exit code 1 means a FAIL. Add `--json` for structured output.

4. **Act on the findings**:

   | Finding | Meaning | What to do |
   | --- | --- | --- |
   | `no_better_than_baseline` FAIL | The model does not beat a trivial rule (majority class, median, or random ranking) | Do not ship or report it as a result; revisit features and framing |
   | `threshold` | Scores rank well but the cut-off turning them into labels is badly chosen | Pick the threshold on validation data for the cost of each error type |
   | `imbalanced` | Rare positive class: accuracy is meaningless | Lead with PR-AUC, recall at a fixed precision, or expected cost |
   | `not_significant` | The improvement is inside the noise | Say "no demonstrated improvement", or get more test data |
   | `weak_segment` | One segment is much worse than the average | Report it explicitly; decide whether it blocks launch |
   | `small_test` | Too few positives for a stable metric | Report the confidence interval, not a point estimate |
   | `weak_precision` | Positives barely above chance precision | Check whether the use case can tolerate the false positives |

5. **Check what the script cannot see**:
   - The decision threshold must be chosen on validation data, never on the test set.
   - The baseline that matters is often the *current* rule or model in production, not majority class. Add it with `--compare-score`.
   - If many models or settings were tried, say how many; the best of 50 tries on one test set is optimistic.
   - Time-based problems need a test period after the training period.

## Output

A short table: metric, value, 95% CI, baseline, verdict. Then one sentence naming the biggest caveat. Never state that a model "improved" when `not_significant` is present.
