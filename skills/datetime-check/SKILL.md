---
name: datetime-check
description: Check date and timestamp columns before parsing, joining or aggregating by time — detects day-first and month-first dates mixed in one column, ambiguous dd/mm vs mm/dd, epoch seconds mixed with milliseconds, time-zone-aware and naive timestamps mixed, several UTC offsets, multiple formats, placeholder dates (1900-01-01) and dates in the future. Use whenever loading data with dates or timestamps, combining sources from different systems or countries, bucketing by day/week/month, or when daily counts look shifted or have spikes at midnight or on the 1st of the month.
---

# Datetime check

Dates parse without errors and still land on the wrong day. `03/04/2026` is March in one system and April in another, a naive timestamp joined to UTC shifts by hours, and a millisecond epoch read as seconds lands tens of thousands of years in the future.

## Steps

1. **Ask where each date column comes from**: the system, the country or locale, and whether timestamps are UTC or local. Sources from different regions are the usual cause of mixed formats.

2. **Run the checks on the raw values** (before any parsing in your code):

   ```bash
   python <skill-dir>/scripts/check_datetime.py --data <file> [--columns <col1,col2>] [--as-of YYYY-MM-DD]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. Without `--columns`, every column that looks like a date is checked. Exit code 1 means a FAIL. Add `--json` for structured output.

3. **Act on the findings**:

   | Finding | What to do |
   | --- | --- |
   | `mixed_day_month` FAIL | Split by source and parse each with an explicit format (`%d/%m/%Y` vs `%m/%d/%Y`) |
   | `mixed_epoch_units` FAIL | Convert each magnitude with its own unit (seconds vs milliseconds) |
   | `mixed_timezone` FAIL | Attach the correct zone to naive values (`tz_localize`), then convert everything to UTC |
   | `ambiguous_day_month` | Confirm the format with the data owner; never rely on parser defaults |
   | `multiple_offsets` | Convert to UTC before sorting, comparing or bucketing; bucket by the business time zone deliberately |
   | `mixed_formats` | Parse each format explicitly, or fix it at the source |
   | `sentinel` | Convert placeholder dates to NULL |
   | `future` | Check for swapped day/month, wrong epoch units or planned dates |
   | `unparseable` | Look at the examples and add the missing format |

4. **Always parse with an explicit format** in the final code (`format=` in pandas, `TO_TIMESTAMP(col, 'YYYY-MM-DD')` in SQL), and state the time zone used for day boundaries in the output.

## Output

For each date column: source format, time zone, range (min to max), and the findings with the fix applied. Do not aggregate by day, week or month while a FAIL is open.
