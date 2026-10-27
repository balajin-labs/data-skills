---
name: fairness-audit
description: Audit a model or decision rule for unequal treatment across groups — selection-rate disparity (four-fifths rule), true-positive-rate gaps (equal opportunity), false-positive-rate gaps, and groups too small to assess. Use whenever a model or rule decides something about people (credit, hiring, insurance, pricing, fraud flags, medical triage, content moderation, targeting), before launch, in model reviews, or when asked whether a model is biased or fair.
---

# Fairness audit

A model can be accurate overall and still approve one group at half the rate of another, or miss most qualified applicants from one group. These gaps carry legal and ethical weight, so measure them before launch.

## Steps

1. **Agree on the groups and the favourable outcome** with the owner: which attributes matter (gender, age band, ethnicity, disability, region as a proxy), and what "favourable" means (approved, not flagged). Fairness definitions conflict; say which ones you report and why.

2. **Collect decisions on a representative held-out set**, with the group attributes joined in even if the model does not use them (it can still learn proxies), and outcomes where known.

3. **Run the audit**:

   ```bash
   python <skill-dir>/scripts/check_fairness.py --pred <file> --group <attr[,attr2]> \
     (--y-pred <decision col> | --y-score <score col> --threshold <t>) [--y-true <outcome col>] [--min-n 30]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. The report includes per-group counts, selection rates and, with `--y-true`, TPR and FPR. Exit code 1 means a FAIL. Add `--json` for structured output.

4. **Act on the findings**:

   | Finding | What to do |
   | --- | --- |
   | `disparate_impact` FAIL | Find the drivers (features acting as proxies, label bias, threshold choice); involve legal/compliance before launch |
   | `equal_opportunity` | Qualified people in one group are missed more often; check label quality and representation for that group |
   | `false_positive_gap` | One group is wrongly flagged more often; review the cost of those errors |
   | `small_group` | Collect more data or report the group as not assessable; never call it "fair" by default |

5. **Report trade-offs honestly**: equalising one metric usually moves another. State which metric the decision prioritises and the accuracy cost of any mitigation.

## Output

A per-group table (n, selection rate, TPR, FPR), the findings, the mitigation chosen (or why none), and who signed off. Do not describe a model as fair based on overall accuracy or on removing the sensitive column.
