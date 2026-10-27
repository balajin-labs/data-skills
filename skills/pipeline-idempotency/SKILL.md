---
name: pipeline-idempotency
description: Prove a pipeline step can be re-run safely — running it twice on the same input must give the same output. Catches reruns and backfills that append duplicates instead of overwriting, keys that appear or vanish between runs, and values that depend on row order, unseeded sampling or the current time. Use whenever building or changing an ETL/ELT job, dbt model, Airflow/Dagster task, incremental load, backfill or retry logic, or when a table's numbers change after a re-run.
---

# Pipeline idempotency

Every pipeline gets re-run: retries after a failure, backfills, someone clicking "clear task". If a re-run appends instead of replacing, or the same input gives different output, numbers drift and nobody can say why.

## Steps

1. **Write down the write strategy** of the step in one line: "overwrite partition `day=…`", "MERGE on `order_id`", "append". Append-only steps are idempotent only if they de-duplicate on a key.

2. **Run the step twice on the same input** in a scratch schema or with a dry-run target, and export the output after each run. For incremental models, re-run the *same* partition or time window. Then compare:

   ```bash
   python <skill-dir>/scripts/check_idempotency.py --first <run1 output> --second <run2 output> \
     --key <col[,col2]> [--ignore loaded_at,run_id]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. Exit code 1 means a FAIL. Add `--json` for structured output.

3. **Act on the findings**:

   | Finding | Fix |
   | --- | --- |
   | `duplicated_on_rerun` FAIL | Overwrite the partition, MERGE/upsert on the key, or delete-then-insert in one transaction |
   | `row_count` / `key_set` FAIL | Remove `now()`/`current_date` from filters (pass the run date as a parameter), seed sampling, order before `LIMIT` |
   | `nondeterministic` FAIL | Make every de-duplication and `ROW_NUMBER()` order total (add a tie-breaker column); seed randomness |
   | `metadata_changed` | Expected for load timestamps and run ids; list them in `--ignore` |
   | `key_not_unique` | The output is not at the grain you think; fix the key or the logic |

4. **Check what two runs cannot show**:
   - Late-arriving data: does re-running yesterday's partition today pick up new rows, and is that intended?
   - Partial failure: if the job dies halfway, does the next run clean up the half-written partition?
   - Concurrent runs: can two runs of the same partition overlap?

## Output

The write-strategy line, rows and keys for both runs, the columns that changed, and the verdict. A step is idempotent only when the comparison is PASS (or WARN on listed metadata columns alone).
