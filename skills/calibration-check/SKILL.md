---
name: calibration-check
description: Check whether a model's predicted probabilities mean what they say — compares the average prediction with the observed rate, measures expected calibration error, detects over- and under-confidence (calibration slope) and segments where predictions are systematically off. Use whenever probabilities are summed into forecasts (expected churners, expected losses, demand), shown to users as percentages, used for pricing or risk, compared across models, or after training on rebalanced, down-sampled or SMOTE data.
---

# Calibration check

A model can rank customers well (high AUC) and still say "30% risk" for a group where 5% actually churn. If the scores are summed, priced or shown to people, ranking is not enough; the numbers themselves have to be right.

## Steps

1. **Score a held-out set** with the real class balance (not a rebalanced training sample), and save one row per example: label, predicted probability, and the segment columns that matter.

2. **Run the checks** (requires pandas, numpy, scikit-learn):

   ```bash
   python <skill-dir>/scripts/check_calibration.py --pred <file> --y-true <col> --y-score <col> \
     [--segment <col[,col2]>] [--bins 10]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. The report includes a reliability table (mean predicted vs observed per score bin). Exit code 1 means a FAIL. Add `--json` for structured output.

3. **Act on the findings**:

   | Finding | What to do |
   | --- | --- |
   | `not_probabilities` FAIL | Apply the sigmoid/softmax, or calibrate raw scores |
   | `mean_mismatch` FAIL | Recalibrate on data with the true base rate; if training was rebalanced, correct the prior (or use Platt scaling) |
   | `ece` | Fit isotonic or Platt calibration on a held-out calibration set (never the test set), then re-check on the test set |
   | `overconfident` / `underconfident` | Calibrate; also review overfitting or regularisation |
   | `segment` | Calibrate per segment or add the segment as a feature; report it if a single threshold is used for decisions |

4. **Re-check after calibration** on data the calibrator did not see. Calibration should barely change AUC; if it drops, something leaked.

## Output

Observed rate, mean prediction, ECE, Brier score and calibration slope before and after any fix, plus the reliability table. Do not present predicted probabilities as percentages to users while a FAIL is open.
