# Leakage patterns seen in real projects

Each pattern lists the symptom, how to confirm it, and the fix.

## 1. Post-outcome columns

**Symptom:** one feature dominates importance; validation AUC above 0.95.
**Example:** churn model with `cancellation_reason`, `account_status`, `last_invoice_amount = 0`.
**Confirm:** for each top feature, find the code or system that populates it and the timestamp when it is written.
**Fix:** drop it, or rebuild it as of the prediction moment.

## 2. Preprocessing fit before the split

**Symptom:** small but consistent optimism; worse in production than in validation.
**Example:**
```python
X = StandardScaler().fit_transform(X)          # fit on all rows
X_train, X_test = train_test_split(X)          # split afterwards
```
**Fix:** split first, then fit inside a pipeline:
```python
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)
```
The same applies to imputers, encoders, PCA, feature selection and oversampling (SMOTE must only see training folds).

## 3. Same entity on both sides

**Symptom:** excellent offline metrics, poor results on new customers or patients.
**Example:** several visits per patient, random row split.
**Fix:** `GroupShuffleSplit` / `GroupKFold` on the entity id.

## 4. Random split on time-ordered data

**Symptom:** forecasting or risk model great offline, weak after launch.
**Fix:** time-based split (train on months 1–9, test on 10–12); for tuning, use rolling-origin validation.

## 5. Whole-history aggregates

**Symptom:** features like "total purchases" or "number of admissions" correlate strongly with the label.
**Example:** "lifetime spend" computed today for a label from last year includes spend after the label date.
**Fix:** compute aggregates with a filter `event_time < prediction_time` per row.

## 6. Test set reused for decisions

**Symptom:** the reported number is the best of many tries on the same test set.
**Fix:** tune on validation folds; touch the test set once, at the end. Report how many configurations were compared.

## 7. Target encoding without folds

**Symptom:** high-cardinality categorical feature suddenly very predictive.
**Fix:** out-of-fold target encoding (e.g. `sklearn.preprocessing.TargetEncoder`, which cross-fits by default).
