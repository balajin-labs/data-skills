---
name: null-type-drift
description: Compare a new data extract, table snapshot or pipeline output with a trusted baseline and flag silent breakage — dropped columns, type changes, null spikes, unit changes (cents vs dollars), new or vanished categories, duplicate keys and placeholder values like -999 or 1900-01-01. Use before loading, publishing or analysing a refreshed dataset, when a pipeline output looks off, or when profiling an unfamiliar file.
---

# Null, type and drift check

Pipelines rarely fail loudly. An upstream team renames a column, switches amounts to cents, or starts sending `'N/A'` instead of empty values, and every downstream number shifts without a single error. Compare each new extract with the last one you trusted.

## Steps

1. **Pick a baseline**: the previous run, last month's extract, or a snapshot that produced numbers you have already validated. Same table, same filters.

2. **Run the comparison**:

   ```bash
   python <skill-dir>/scripts/check_drift.py --baseline <trusted file> --current <new file> \
     [--key <column that should be unique>] [--ignore <columns that always change, e.g. load_ts>]
   ```

   With no baseline, `--current` alone profiles one file for placeholder values, mixed types and numbers stored as text. `<skill-dir>` is the folder that contains this SKILL.md. Exit code 1 means a FAIL. For warehouse tables, export both snapshots (or the same sample of keys) to CSV/Parquet first.

3. **Act on the findings**:

   | Finding | What to do |
   | --- | --- |
   | `schema` FAIL | Stop the load; find out whether the column was renamed or removed upstream |
   | `type_change` | Fix the parse or cast; never let a numeric column fall back to text silently |
   | `null_spike` | Find the cause (new source, broken join, new null token) before analysing |
   | `unit_change` | Confirm the unit with the data owner and convert explicitly |
   | `new_categories` / `missing_categories` | Update mappings, filters and CASE statements; check for renamed codes |
   | `duplicates` | Look for a double load or an upstream fan-out join (use join-guard) |
   | `sentinel`, `mixed_types`, `formatted_numbers` | Convert placeholders to null and parse numbers before computing statistics |
   | `distribution_shift`, `row_count` (WARN) | Could be real business change; confirm before reporting it as such |

4. **Record the decision**: which findings were fixed, which were accepted as real-world change, and why.

## Output

A table of findings with severity, column and decision. Do not publish metrics computed from a current extract with an open FAIL.
