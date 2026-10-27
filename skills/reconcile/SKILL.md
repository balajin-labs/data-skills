---
name: reconcile
description: Reconcile two copies of the same data and prove they match — source system vs warehouse, old pipeline vs new pipeline, a migrated table vs the original, or a dashboard vs a recomputation. Checks row counts, missing and extra keys, duplicate keys, column-by-column value mismatches (with unit-change and rounding hints) and measure totals. Use whenever migrating, rebuilding or replacing a pipeline or table, validating a load, or when two reports disagree.
---

# Reconcile

"The numbers look about right" is not a migration sign-off. Two systems match when every key is present on both sides, every value agrees, and every total ties out, and you can show it.

## Steps

1. **Agree on the comparison**: the business key (`order_id`, or `order_id, line_no`), the time window, and which side is the reference. Extract both sides for the *same* window and filters; most "mismatches" are two different extractions.

2. **Run the checks**:

   ```bash
   python <skill-dir>/scripts/check_reconcile.py --source <file> --target <file> --key <col[,col2]> \
     [--measure <amount,quantity>] [--by <date or partition col>] [--columns <cols>] [--rtol 1e-6]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. Exports of up to a few million rows are fine; for larger tables compare per-partition aggregates first (row count and sum per day), then drill into the failing partitions. Exit code 1 means a FAIL. Add `--json` for structured output.

3. **Act on the findings**:

   | Finding | Typical cause | What to do |
   | --- | --- | --- |
   | `missing_in_target` | Failed partition, filter in the load, late data | Use `--by` to find the partition; re-run or backfill it |
   | `extra_in_target` | Hard deletes upstream, test rows, wider window | Confirm the window; decide how deletes propagate |
   | `duplicate_keys` | Double load, wrong key | Fix the key or de-duplicate the load (see `pipeline-idempotency`) |
   | `value_mismatch` | Type casts, rounding, units, time zones, changed logic | Use the hint and examples; fix the transformation, not the comparison |
   | `measure_total` | Any of the above | Must tie out before sign-off |
   | `row_count` | Summary of the above | Resolves when the key-level findings do |

4. **Re-run until PASS.** If a difference is accepted (for example a known rounding rule), document it and pass `--columns` or `--rtol` explicitly so the exception is visible.

## Output

A table with rows, matched keys, missing, extra and mismatched values per column, and each measure total on both sides, then the verdict. Never call a migration or rebuild "validated" with an open FAIL.
