#!/usr/bin/env python3
"""Check date and time columns for the parsing mistakes that silently shift or scramble data: day-first and month-first
dates mixed in one column, ambiguous dd/mm vs mm/dd, epoch seconds mixed with milliseconds, time-zone-aware and naive
timestamps mixed, several formats in one column, placeholder dates and dates in the future.

Usage examples
--------------
    python check_datetime.py --data events.csv                      # checks every date-like column
    python check_datetime.py --data events.csv --columns created_at,event_ts --as-of 2026-06-30

Exit code: 1 if any FAIL finding, 0 otherwise, 2 on bad input. Add --json for structured output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

DATE_NAME = re.compile(r"(date|_at$|_on$|time|_ts$|^ts$|timestamp|epoch|dob|birth|(^|_)day($|_)|_dt$)", re.I)
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?\s*(Z|[+-]\d{2}:?\d{2})?$", re.I)
SLASH = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})(?:[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)?$", re.I)
TEXT = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", re.I)
COMPACT = re.compile(r"^(19|20)\d{6}$")
SENTINEL = {"1900-01-01", "1970-01-01", "9999-12-31", "0001-01-01", "2099-12-31", "1899-12-30", "2999-12-31"}


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: file not found: {path}")
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
        return df.astype(str).where(df.notna())
    return pd.read_csv(p, sep="\t" if p.suffix.lower() in {".tsv", ".tab"} else ",", dtype=str,
                       keep_default_na=False, na_values=[""])


def finding(check, severity, message, columns=None, value=None, examples=None):
    out = {"check": check, "severity": severity, "message": message, "columns": columns or [], "value": value}
    if examples:
        out["examples"] = examples
    return out


def classify(v: str) -> str:
    if ISO.match(v):
        return "iso"
    if SLASH.match(v):
        return "slash"
    if COMPACT.match(v):
        return "compact"
    if re.fullmatch(r"-?\d+(\.\d+)?", v):
        return "number"
    if TEXT.search(v):
        return "text"
    return "other"


def epoch_unit(x: float):
    a = abs(x)
    if 1e8 <= a < 1e10:
        return "seconds"
    if 1e11 <= a < 1e13:
        return "milliseconds"
    if 1e14 <= a < 1e16:
        return "microseconds"
    if 1e17 <= a < 1e19:
        return "nanoseconds"
    return None


def check_column(c, s, as_of):
    f, info = [], {}
    v = s.dropna().astype(str).str.strip()
    v = v[v != ""]
    if v.empty:
        return f, info
    kinds = v.map(classify)
    share = kinds.value_counts(normalize=True)
    info["formats"] = {k: round(float(x), 3) for k, x in share.items()}

    # Epoch numbers
    if share.get("number", 0) >= 0.9:
        units = pd.to_numeric(v[kinds == "number"]).map(epoch_unit)
        us = units.value_counts(normalize=True)
        info["epoch_units"] = {k: round(float(x), 3) for k, x in us.items()}
        real = us[us >= 0.01]
        if len(real) >= 2:
            f.append(finding(
                "datetime.mixed_epoch_units", "FAIL",
                f"'{c}' mixes epoch {' and '.join(real.index)} ({', '.join(f'{k} {x:.0%}' for k, x in real.items())}). "
                "Converting with one unit puts the other rows in 1970 or thousands of years ahead. Convert each "
                "magnitude with its own unit.", [c], len(real),
                [f"{x} ({u})" for x, u in zip(v[kinds == 'number'].head(200), units.head(200))][:4]))
            return f, info
        if len(real) == 1:
            unit = real.index[0]
            div = {"seconds": 1, "milliseconds": 1e3, "microseconds": 1e6, "nanoseconds": 1e9}[unit]
            parsed = pd.to_datetime(pd.to_numeric(v[kinds == "number"]) / div, unit="s", errors="coerce")
        else:
            return f, info
    else:
        mixed = share[share >= 0.01]
        if len(mixed) >= 2:
            ex = [v[kinds == k].iloc[0] for k in mixed.index]
            f.append(finding(
                "datetime.mixed_formats", "WARN",
                f"'{c}' contains several date formats ({', '.join(f'{k} {x:.0%}' for k, x in mixed.items())}). "
                "Parsers guess per value, so some rows can be read differently from others. Normalise at the "
                "source or parse each format explicitly.", [c], len(mixed), ex))

        dayfirst = False
        sl = v[kinds == "slash"]
        if len(sl):
            parts = sl.str.extract(SLASH).iloc[:, :2].astype(int)
            a_big, b_big = int((parts[0] > 12).sum()), int((parts[1] > 12).sum())
            if a_big and b_big:
                f.append(finding(
                    "datetime.mixed_day_month", "FAIL",
                    f"'{c}' has {a_big:,} values that must be day-first (like 31/01) and {b_big:,} that must be "
                    "month-first (like 01/31). Two sources are mixed, and every date with both parts <= 12 is "
                    "ambiguous. Parse each source with its own format.", [c], a_big + b_big,
                    [sl[parts[0] > 12].iloc[0], sl[parts[1] > 12].iloc[0]]))
            elif not a_big and not b_big and len(sl) >= 20:
                f.append(finding(
                    "datetime.ambiguous_day_month", "WARN",
                    f"All {len(sl):,} slash dates in '{c}' have both parts <= 12, so dd/mm and mm/dd cannot be told "
                    "apart from the data. Confirm the format with the source and parse it explicitly.", [c], len(sl),
                    sl.head(3).tolist()))
            dayfirst = a_big > b_big

        iso = v[kinds == "iso"]
        if len(iso):
            tz = iso.str.extract(ISO).iloc[:, 0]
            has_time = iso.str.contains(r"\d{2}:\d{2}")
            aware, naive = int(tz[has_time].notna().sum()), int(tz[has_time].isna().sum())
            if aware and naive and min(aware, naive) >= 0.01 * has_time.sum():
                f.append(finding(
                    "datetime.mixed_timezone", "FAIL",
                    f"'{c}' mixes {aware:,} timestamps with a time zone and {naive:,} without one. The naive ones are "
                    "usually local time and end up shifted by hours (and on the wrong date near midnight) when "
                    "combined with UTC. Find the source of each and attach the right zone.", [c], naive,
                    [iso[has_time & tz.notna()].iloc[0], iso[has_time & tz.isna()].iloc[0]]))
            offsets = tz.dropna().str.replace("Z", "+00:00", case=False).str.replace(r"^([+-]\d{2})(\d{2})$",
                                                                                  r"\1:\2", regex=True)
            uniq = sorted(offsets.unique())
            if len(uniq) > 2 or (len(uniq) == 2 and abs(_hours(uniq[0]) - _hours(uniq[1])) != 1):
                f.append(finding(
                    "datetime.multiple_offsets", "WARN",
                    f"'{c}' has timestamps in {len(uniq)} UTC offsets ({', '.join(uniq[:5])}). Convert to UTC before "
                    "comparing, sorting or bucketing by day.", [c], len(uniq)))
        try:
            parsed = pd.to_datetime(v, errors="coerce", format="mixed", dayfirst=dayfirst, utc=True)
        except (TypeError, ValueError):
            parsed = pd.to_datetime(v, errors="coerce", dayfirst=dayfirst, utc=True)
        parsed = parsed.dt.tz_localize(None)

    bad = int(parsed.isna().sum())
    info["unparseable"] = bad
    if bad and bad >= 0.01 * len(v):
        f.append(finding("datetime.unparseable", "WARN",
                         f"{bad:,} values in '{c}' ({bad / len(v):.1%}) cannot be parsed as dates.", [c], bad,
                         v[parsed.isna()].head(4).tolist()))
    p = parsed.dropna()
    if p.empty:
        return f, info
    day = p.dt.strftime("%Y-%m-%d")
    sent = day[day.isin(SENTINEL)]
    if len(sent):
        f.append(finding(
            "datetime.sentinel", "WARN",
            f"'{c}' contains placeholder dates ({', '.join(sorted(set(sent)))}; {len(sent):,} rows). They distort "
            "durations, ages and trends; convert them to NULL.", [c], int(len(sent))))
    fut = p[(p > as_of) & ~day.isin(SENTINEL)]
    if len(fut):
        f.append(finding(
            "datetime.future", "WARN",
            f"'{c}' has {len(fut):,} values after {as_of:%Y-%m-%d} (max {fut.max():%Y-%m-%d}). Check for day/month "
            "swaps, wrong epoch units or planned dates mixed with actual ones.", [c], int(len(fut))))
    info["min"], info["max"] = str(p.min()), str(p.max())
    return f, info


def _hours(offset: str) -> float:
    m = re.match(r"([+-])(\d{2}):?(\d{2})", offset)
    if not m:
        return 0.0
    h = int(m.group(2)) + int(m.group(3)) / 60
    return h if m.group(1) == "+" else -h


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--columns", help="comma-separated columns (default: columns that look like dates)")
    ap.add_argument("--as-of", help="dates after this are 'in the future' (default: today)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    df = read_table(a.data)
    if a.columns:
        cols = [c.strip() for c in a.columns.split(",")]
        for c in cols:
            if c not in df.columns:
                raise SystemExit(f"error: column '{c}' not in {a.data}")
    else:
        cols = [c for c in df.columns if DATE_NAME.search(c)]
        for c in df.columns:
            head = df[c].dropna().astype(str).head(200)
            if c not in cols and len(head) and sum(classify(x.strip()) in ("iso", "slash") for x in head) >= 0.8 * len(head):
                cols.append(c)
    if not cols:
        raise SystemExit("error: no date-like columns found; pass --columns")
    as_of = pd.Timestamp(a.as_of) if a.as_of else pd.Timestamp.today().normalize()

    findings, stats = [], {"rows": len(df), "columns_checked": cols}
    for c in cols:
        f, info = check_column(c, df[c], as_of)
        findings += f
        stats[c] = info
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "datetime-check", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"datetime-check: {status}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        for x in findings:
            print(f"  [{x['severity']}] {x['check']}: {x['message']}")
            if x.get("examples"):
                print(f"      examples: {x['examples']}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):      # input errors: print the message and exit 2 (1 is reserved for FAIL)
            print(e.code, file=sys.stderr)
            sys.exit(2)
        raise
    except pd.errors.EmptyDataError:
        print("error: an input file is empty (no header, no rows)", file=sys.stderr)
        sys.exit(2)
