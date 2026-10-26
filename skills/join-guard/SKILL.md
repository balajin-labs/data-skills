---
name: join-guard
description: Verify a SQL or dataframe join before using its result — detects fan-out (duplicate keys that multiply rows), double-counted totals, unmatched keys, null keys and keys formatted differently on each side. Use whenever writing or reviewing a join, merge or lookup, when row counts or totals change unexpectedly after a join, or when building a fact table, report or feature table from several sources.
---

# Join guard

Most wrong numbers in reports come from joins that run without errors: a lookup table with duplicate keys silently doubles revenue, or `'00123'` never matches `123`. Check every join whose output feeds a number someone will read.

## Steps

1. **State the grain and the relationship before writing the join.** Write one line: "orders (one row per order) LEFT JOIN customers (one row per customer) on customer_id — many-to-one." If you cannot state it, inspect both tables first.

2. **Run the checks** on the two inputs (export samples from the warehouse to CSV/Parquet if needed; tens of thousands of rows is enough):

   ```bash
   python <skill-dir>/scripts/check_join.py --left <file> --right <file> --on <key[,key2]> \
     [--how left|inner] [--expect many-to-one|one-to-one|one-to-many] [--sum-col <measure>]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. Exit code 1 means a FAIL. Pass `--sum-col` with the main measure (revenue, quantity, cost) to prove the join does not change its total.

   In SQL, the same checks are:

   ```sql
   -- right side must be unique on the key for a many-to-one join
   SELECT key, COUNT(*) FROM right_table GROUP BY key HAVING COUNT(*) > 1 LIMIT 20;
   -- row count must not change for a LEFT many-to-one join
   SELECT (SELECT COUNT(*) FROM left_table) AS before,
          (SELECT COUNT(*) FROM left_table l LEFT JOIN right_table r ON l.key = r.key) AS after;
   ```

3. **Act on the findings**:

   | Finding | Fix |
   | --- | --- |
   | `fan_out` | Deduplicate the right side (latest record per key with `ROW_NUMBER()`), aggregate it to the key's grain, or add the missing key column (often a date or version) |
   | `double_count` | Aggregate before joining, or compute the measure at the left table's grain |
   | `key_format` | Cast and normalise both sides the same way (`TRIM`, `LOWER`, `LPAD`, consistent types) |
   | `unmatched` | Find out why: late-arriving dimension rows, deleted records, a wrong key. Never silently drop them with an inner join |
   | `null_keys` | Decide explicitly whether those rows belong in the output |
   | `left_duplicates` | The left table is not at the grain you assumed; fix the grain statement |

4. **Re-run until PASS**, or document why a remaining WARN is acceptable.

## Output

State the grain line from step 1, the row counts before and after, the measure total before and after (if any), and the verdict. Do not present aggregates computed from a join that has an open `fan_out` or `double_count` FAIL.
