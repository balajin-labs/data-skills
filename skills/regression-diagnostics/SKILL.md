---
name: regression-diagnostics
description: Diagnose a linear regression before interpreting coefficients or p-values — detects perfect and strong multicollinearity (VIF), heteroscedasticity (Breusch-Pagan), missing non-linearity (RESET), autocorrelated residuals (Durbin-Watson), influential points (Cook's distance), skewed residuals and too few rows per feature. Use whenever fitting OLS or a linear model to explain drivers ("what drives sales?"), estimating elasticities or effects, reporting coefficients or significance, or building a marketing-mix or pricing model.
---

# Regression diagnostics

A regression always returns coefficients and p-values, even when they mean nothing. Before saying "price has a significant negative effect", check that the model's assumptions hold well enough for that sentence to be true.

## Steps

1. **State the question**: prediction (only accuracy matters) or explanation (coefficients will be interpreted). These checks matter most for explanation.

2. **Run the diagnostics** on the modelling table:

   ```bash
   python <skill-dir>/scripts/check_regression.py --data <file> --target <col> \
     [--features <col1,col2,...>] [--order <time col>]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. Encode categorical features as numeric dummies first (dropping one level). Exit code 1 means a FAIL. Add `--json` for structured output.

3. **Act on the findings**:

   | Finding | What to do |
   | --- | --- |
   | `perfect_collinearity` FAIL | Drop the redundant column (a total next to its parts, one dummy per category) |
   | `multicollinearity` | Do not interpret those coefficients separately; combine, drop, or use regularisation and say so |
   | `heteroscedasticity` | Report robust (HC3) standard errors, or model a log-transformed target |
   | `nonlinearity` | Add transforms, interactions or splines; plot residuals vs each feature |
   | `autocorrelation` | Use Newey-West errors, add lags or trend terms, or a time-series model |
   | `influential_points` | Check those rows for errors; show results with and without them |
   | `skewed_residuals` | Transform the target (log) and re-check |
   | `small_sample` | Fewer features, more data, or regularisation |

4. **Check what the script cannot**: omitted variables that drive both a feature and the target (confounding), reverse causality, and whether the features are measured before the outcome. Coefficients from observational data are associations unless the design supports causation.

## Output

The coefficient table (with robust standard errors if heteroscedastic), R², the diagnostics with the decision taken for each, and one sentence on what the coefficients can and cannot be read as. Never interpret coefficients while `perfect_collinearity` is open.
