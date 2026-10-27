---
name: ab-test-check
description: Check an A/B test or online experiment before reporting a result or making a ship decision — detects sample ratio mismatch, users who saw more than one variant, underpowered tests, "wins" that vanish after a multiple-testing correction, novelty effects, pre-period imbalance and lifts driven by a few extreme users. Use whenever analysing an experiment, split test, feature flag rollout or randomised trial, reading an experiment dashboard, or claiming a variant "won" or "had no effect".
---

# A/B test check

An experiment readout can be computed correctly and still be wrong: the split was broken, the "win" is one of twenty metrics tested, or "no effect" really means "too small to tell". Run these checks before anyone reads the lift.

## Steps

1. **Write down the plan first**: the randomisation unit (user, account, device), the intended split, the one primary metric, and the smallest effect worth detecting. If the analysis has already started, write down what was decided before the data was seen and what was not.

2. **Build one row per randomisation unit**: unit id, variant, first exposure time, each metric aggregated to the unit over the test window, and any pre-experiment values of the same metrics (for example revenue in the 4 weeks before the test). Do this from the raw assignment log, not from a dashboard export, so users dropped by later filters are still visible.

3. **Run the checks** (requires pandas and numpy):

   ```bash
   python <skill-dir>/scripts/check_abtest.py --data <file> --variant <col> --control <name> \
     --metric <primary>[,<secondary>...] [--unit <id col>] [--time <exposure col>] \
     [--pre <pre-period cols>] [--expected-split 50,50] [--mde 0.02]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. Exit code 1 means a FAIL. Add `--json` for structured output.

4. **Act on the findings**:

   | Finding | Meaning | What to do |
   | --- | --- | --- |
   | `srm` FAIL | The traffic split differs from the plan | Stop. Find the cause (bot filters, crashes, redirects, logging differences) before reading any metric |
   | `unit_in_multiple_variants` FAIL | Some units saw both variants | Fix assignment to the stated unit; exclude those units and report how many |
   | `pre_imbalance` | Groups differed before the test started | Treat as a broken randomisation; at most, report a pre-period-adjusted (CUPED or difference-in-differences) result with that caveat |
   | `multiple_metrics` | A "win" does not survive Holm correction | Call it exploratory; the decision rests on the pre-registered primary metric |
   | `underpowered` | Not significant, but the test could not see effects of the size that matter | Say "inconclusive", not "no effect"; give the sample size needed |
   | `novelty` | The effect fades for later-exposed users | Report the later-cohort effect; run longer before deciding |
   | `outlier_driven` | A few extreme users carry the lift | Report the capped (winsorised) result too; inspect those users |
   | `repeated_units` | Several rows per unit | Aggregate to one row per unit and re-run |

5. **Check what the script cannot see**:
   - Was the test stopped the first time it looked significant? Repeated peeking inflates false positives; ask when and how often results were checked.
   - Does the metric window give every unit the same exposure time, or did late joiners have less time to convert?
   - Are ratio metrics (revenue per session, CTR) analysed at the randomisation unit? If sessions are the denominator, use the delta method or a per-user ratio.
   - Did anything else change during the test (a marketing push, an outage, a holiday)?

## Output

One table per comparison: metric, control, treatment, lift, 95% CI, p-value, Holm-adjusted p-value. Then the verdict for the primary metric only: "ship", "do not ship" or "inconclusive", with the biggest caveat in one sentence. Never report a lift while an `srm` or `unit_in_multiple_variants` FAIL is open.
