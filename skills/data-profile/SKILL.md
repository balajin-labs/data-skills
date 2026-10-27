---
name: data-profile
description: Profile a new or unfamiliar dataset before analysing it, focused on problems that silently corrupt results — impossible values (negative quantities, ages over 120), placeholder codes (-999, 'N/A', 1900-01-01), mixed numeric/text columns, labels written several ways ('NY', 'ny '), non-unique IDs, duplicate rows, mostly-empty or constant columns, and dates in the future. Use whenever loading, exploring or receiving a dataset for the first time, starting an EDA, or before building features, reports or models on data you have not checked.
---

# Data profile

Summary statistics hide the problems that matter: a `-999` drags an average down, `'N/A'` becomes its own category, and a duplicated ID quietly doubles a join. Profile first, analyse second.

## Steps

1. **Say what one row is** ("one row per order line") and which column should identify it. If you cannot, find out before going further.

2. **Run the profile**:

   ```bash
   python <skill-dir>/scripts/check_profile.py --data <file> [--id <col[,col2]>] [--as-of YYYY-MM-DD]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. The file is read as raw text, so placeholders and leading zeros stay visible. Exit code 1 means a FAIL. Add `--json` for structured output. For a database table, export a sample (100k rows is plenty) or run the equivalent SQL checks.

3. **Act on the findings**:

   | Finding | What to do |
   | --- | --- |
   | `impossible_values` FAIL | Find out what they mean (returns? data entry errors?) and handle them explicitly; never average over them silently |
   | `id_not_unique` FAIL | The grain is not what you think; find the real key or de-duplicate with a stated rule |
   | `duplicate_rows` | Decide whether they are double loads (drop) or legitimate repeats (keep) |
   | `placeholder_text` / `placeholder_number` / `sentinel_dates` | Convert to NULL at load time and note how many |
   | `mixed_types` | Clean or split the column before arithmetic |
   | `inconsistent_labels` | Normalise case, spacing and punctuation (and map synonyms) before grouping |
   | `future_dates` | Check for typos, swapped day/month or planned dates; see `datetime-check` |
   | `mostly_empty` / `constant` | Drop, or ask whether the extract is broken |

4. **Record the cleaning decisions** (what was converted, dropped or mapped, and how many rows) so the analysis can state them.

## Output

One line on the grain, the row count, the list of findings with the decision taken for each, and the rows affected. Do not start the analysis with an open FAIL unless the user accepts it explicitly.
