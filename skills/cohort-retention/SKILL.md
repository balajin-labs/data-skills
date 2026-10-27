---
name: cohort-retention
description: Check a cohort retention or churn table before it is charted, averaged or presented — catches periods that have not finished yet (right censoring that makes recent cohorts look worse), rates above 100%, broken period-0 definitions, retention that rises when it cannot, and tiny cohorts. Use whenever building or reviewing cohort analyses, retention curves, churn or survival charts, LTV estimates, or claims that newer cohorts retain better or worse.
---

# Cohort retention check

The classic retention-chart mistake: the latest cohorts look terrible because their users have not had time to come back yet. The chart "shows" retention getting worse while nothing has changed.

## Steps

1. **Write the definitions down**: what puts a user in a cohort (signup, first purchase), what counts as "retained" in period N (active in N, or active in N or later), the period grain, and the date the data was extracted.

2. **Build the table** with one row per cohort and period: cohort start date, period index (0 = the signup period), active users and cohort size (or the rate).

3. **Run the checks**:

   ```bash
   python <skill-dir>/scripts/check_cohort.py --table <file> --cohort <col> --period <col> \
     (--rate <col> | --active <col> --size <col>) --grain month --as-of <extract date> \
     [--definition classic|unbounded] [--min-n 50]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. Exit code 1 means a FAIL. Add `--json` for structured output.

4. **Act on the findings**:

   | Finding | What to do |
   | --- | --- |
   | `incomplete_period` FAIL | Blank or mark partial cells; compare cohorts only on periods all of them have completed |
   | `rate_out_of_range` FAIL | Count distinct users from the same cohort population in numerator and denominator |
   | `rising_retention` | Required to fix for unbounded retention; for classic, confirm with events (reactivation, seasonality) |
   | `period_zero` | Align the cohort-entry event and the activity event (and the user id) |
   | `small_cohort` | Merge cohorts (monthly instead of weekly) or show counts |

5. **When comparing cohorts**, compare the same period index (month-3 vs month-3), and only cohorts that have all completed that period. For averages across cohorts, weight by cohort size.

## Output

The definitions from step 1, the cleaned retention table or chart with partial cells marked, and the comparison claim with the periods it is based on. Never show a retention trend that includes incomplete periods without marking them.
