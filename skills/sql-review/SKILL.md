---
name: sql-review
description: Review SQL for bugs that run without errors and return wrong numbers — LEFT JOINs silently turned into INNER JOINs by a WHERE filter, comparisons with NULL, NOT IN over nullable subqueries, UNION dropping legitimate duplicates, integer division, BETWEEN on timestamps, AND/OR precedence, LIMIT without ORDER BY, accidental cross joins and DISTINCT hiding fan-out. Use whenever writing, reviewing or debugging a SQL query, dbt model, view or report query, or when a SQL result looks too small, too large or unstable.
---

# SQL review

SQL fails quietly. A filter in the wrong clause drops every customer without orders, `= NULL` matches nothing, and an integer division turns a 37% rate into 0. None of these raise an error.

## Steps

1. **State the grain of the result** in one line before writing or reviewing the query: "one row per customer segment per month". Every join and filter should preserve or deliberately change that grain.

2. **Run the static checks** on the query file(s):

   ```bash
   python <skill-dir>/scripts/check_sql.py --sql <file.sql> [<more.sql> ...]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. No database is needed; the script reads the SQL text. Exit code 1 means a FAIL. Add `--json` for structured output.

3. **Act on the findings**:

   | Finding | Fix |
   | --- | --- |
   | `left_join_filtered` FAIL | Move the right-table condition into the `ON` clause, or pre-filter the right table in a CTE |
   | `null_comparison` FAIL | Use `IS NULL` / `IS NOT NULL` |
   | `cartesian_join` FAIL | Use explicit `JOIN ... ON` |
   | `not_in_nullable` | Use `NOT EXISTS`, or add `WHERE col IS NOT NULL` inside the subquery |
   | `union_dedup` | Use `UNION ALL` unless removing duplicates is the intent |
   | `integer_division` | Multiply by `1.0` or `CAST(... AS DECIMAL)` before dividing |
   | `between_timestamp` | Use `ts >= '2026-01-01' AND ts < '2026-02-01'` |
   | `and_or_precedence` | Add parentheses that match the intended logic |
   | `limit_without_order` | Add an `ORDER BY` that makes the order unique |
   | `distinct_with_join` | Check the join's grain with `join-guard` instead of de-duplicating the output |
   | `select_star_join` | List the columns |

4. **Check what the static review cannot see** by running small queries against the data:
   - Row count before and after each join (fan-out); use the `join-guard` skill for the full check.
   - `COUNT(*)` vs `COUNT(col)` vs `COUNT(DISTINCT col)`: is the measure counting what the grain says?
   - Time zones: are timestamps compared in the same zone as the date boundaries?
   - Filters on late-arriving or soft-deleted rows (`is_deleted`, `valid_to`).

## Output

The grain line from step 1, the list of findings with the fix applied to each, and the corrected query. Do not report numbers from a query with an open FAIL.
