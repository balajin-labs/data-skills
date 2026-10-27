---
name: analysis-review
description: Review an analytical conclusion before it is presented — catches Simpson's paradox, gaps that come from a different segment mix rather than the factor being compared, differences hidden by mix, "best" and "worst" rankings driven by tiny groups, averages moved by a few outliers, plus survivorship and causal over-claiming. Use whenever comparing groups, periods, channels, regions, cohorts or treatments, ranking segments, explaining why a metric moved, or writing findings, insights or recommendations from data.
---

# Analysis review

Most misleading analyses contain no code errors. The arithmetic is right; the comparison is wrong. This skill checks the comparison itself.

## Steps

1. **Write the claim as one sentence**, for example: "Customers acquired through the app convert 5 points better than web customers." Name the groups, the outcome and the period. If the claim says *because*, *drives* or *causes*, mark it as a causal claim.

2. **List what else differs between the groups** (customer type, region, product mix, tenure, severity, device, season). These are the segments to test. If you can only think of one, ask what decided who ended up in each group.

3. **Run the checks** on row-level data (one row per customer, order, patient, and so on; a sample of tens of thousands is enough):

   ```bash
   python <skill-dir>/scripts/check_analysis.py --data <file> --outcome <col> \
     [--group <col>] [--groups A,B] [--segment <col>[,<col>...]] [--min-n 30]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. Exit code 1 means a FAIL. Add `--json` for structured output.

4. **Act on the findings**:

   | Finding | Meaning | What to do |
   | --- | --- | --- |
   | `simpsons_paradox` FAIL | The overall difference reverses inside the segments | Do not present the overall number as the answer. Lead with the within-segment comparison and explain the mix |
   | `mix_shift` | Most of the gap comes from the groups' different segment mix | Present the adjusted gap; say how much is mix |
   | `masked_difference` | No overall gap, but a clear one within segments | Report the within-segment difference |
   | `segment_coverage` | Many rows sit in segments with only one group | Restrict the claim to the overlapping segments |
   | `small_n_extremes` | The top or bottom of a ranking is a tiny group | Show counts, set a minimum size, or shrink toward the overall mean |
   | `outlier_driven` | A few extreme rows carry the difference | Show the median or a capped mean; inspect those rows |

5. **Check what the script cannot see**:
   - **Survivorship**: does the data contain only the customers, products or funds that are still around? Comparing survivors overstates performance.
   - **Selection**: did people choose their group (opted in, self-selected a plan)? Then the groups differ in ways no segment column captures, and the comparison is not causal.
   - **Causal language**: without randomisation or a credible natural experiment, write "is associated with", not "drives" or "causes".
   - **Denominators**: are rates per user, per session or per order on both sides? Did the definition change between periods?
   - **Regression to the mean**: a segment picked *because* it was extreme last period will look better or worse next period without any intervention.

## Output

The claim from step 1, rewritten if needed, with the overall and adjusted numbers side by side, the segment counts, and one sentence on what the data cannot show (usually causation). Never present an overall comparison while a `simpsons_paradox` FAIL is open.
