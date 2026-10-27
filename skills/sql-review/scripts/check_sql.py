#!/usr/bin/env python3
"""Static review of SQL for bugs that run without errors: LEFT JOINs silently turned into INNER JOINs, comparisons
with NULL, NOT IN over nullable subqueries, UNION dropping rows, integer division, BETWEEN on timestamps,
AND/OR precedence, LIMIT without ORDER BY, comma joins without a condition, and DISTINCT hiding fan-out.

Usage examples
--------------
    python check_sql.py --sql query.sql
    python check_sql.py --sql models/*.sql --json

The checks are pattern-based and dialect-agnostic: they flag likely bugs for a human or agent to confirm.
Exit code: 1 if any FAIL finding, 0 otherwise, 2 on bad input.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

I = re.I | re.S
CLAUSE_END = r"(?=\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|\bUNION\b|\bQUALIFY\b|\bWINDOW\b|\bEXCEPT\b|\bINTERSECT\b|$)"
NOT_ALIAS = {"on", "where", "join", "left", "right", "inner", "outer", "full", "cross", "group", "order", "limit",
             "using", "natural", "union", "select", "as", "and", "or", "having", "lateral"}


def finding(check, severity, message, file, line=None, snippet=None):
    out = {"check": check, "severity": severity, "message": message, "columns": [], "value": None, "file": file}
    if line:
        out["line"] = line
    if snippet:
        out["examples"] = [snippet]
    return out


def strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group()), sql, flags=re.S)
    return re.sub(r"--[^\n]*", lambda m: " " * len(m.group()), sql)


def mask_strings(sql: str) -> str:
    """Blank out string literals (same length, so offsets and line numbers still match)."""
    return re.sub(r"'(?:[^']|'')*'", lambda m: "'" + "x" * (len(m.group()) - 2) + "'", sql)


def statements(sql: str):
    """Yield (start offset, text) for each ;-separated statement."""
    pos = 0
    for part in sql.split(";"):
        if part.strip():
            yield pos, part
        pos += len(part) + 1


def line_of(full: str, offset: int) -> int:
    return full.count("\n", 0, offset) + 1


def snippet(text: str, start: int, end: int) -> str:
    return " ".join(text[start:end].split())[:160]


def flatten_parens(s: str) -> str:
    """Replace every parenthesised group (innermost first) with a placeholder, leaving the top level."""
    prev = None
    while prev != s:
        prev, s = s, re.sub(r"\([^()]*\)", " _P_ ", s)
    return s


def select_blocks(stmt: str):
    """(start, end) of each SELECT block: from a SELECT keyword up to the next one. CTE chains and subqueries
    become separate blocks, so a JOIN in one CTE is not blamed on a SELECT * in another."""
    starts = [m.start() for m in re.finditer(r"\bSELECT\b", stmt, re.I)] or [0]
    return [(s, e) for s, e in zip(starts, starts[1:] + [len(stmt)])]


def review(path: str, raw: str):
    f = []
    clean = strip_comments(raw)
    masked = mask_strings(clean)
    n_stmt = n_left = 0
    for off, stmt in statements(masked):
        n_stmt += 1
        orig = clean[off:off + len(stmt)]
        is_update = re.match(r"\s*UPDATE\b", stmt, re.I) is not None

        # LEFT JOIN turned into INNER JOIN by a WHERE filter on the right table
        for b0, b1 in select_blocks(stmt):
            block = stmt[b0:b1]
            aliases = []
            for m in re.finditer(r"\bLEFT\s+(?:OUTER\s+)?JOIN\s+((?:\w+\.)*\w+)(?:\s+(?:AS\s+)?(\w+))?", block, re.I):
                n_left += 1
                alias = m.group(2) if m.group(2) and m.group(2).lower() not in NOT_ALIAS else m.group(1).split(".")[-1]
                aliases.append(alias)
            for w in re.finditer(r"\bWHERE\b(.*?)" + CLAUSE_END, block, I):
                where = w.group(1)
                for alias in aliases:
                    if re.search(rf"\b{alias}\.\w+\s+IS\s+NULL\b", where, re.I):
                        continue    # anti-join or explicit null handling
                    cond = re.search(rf"\b{alias}\.\w+\s*(=|<>|!=|>=|<=|>|<|\bNOT\s+IN\b|\bIN\b|\bLIKE\b|\bBETWEEN\b)",
                                     where, re.I)
                    if cond:
                        pos = off + b0 + w.start(1) + cond.start()
                        f.append(finding(
                            "sql.left_join_filtered", "FAIL",
                            f"The WHERE clause filters on '{alias}', the right side of a LEFT JOIN. Rows with no match "
                            f"have NULLs there, so the filter removes them and the LEFT JOIN becomes an INNER JOIN. Move "
                            f"the condition into the ON clause (or a subquery) if unmatched rows should stay.",
                            path, line_of(clean, pos), snippet(orig, b0 + w.start(1) + cond.start(),
                                                               b0 + w.start(1) + cond.end() + 40)))
                    elif re.search(rf"\b{alias}\.\w+\s+IS\s+NOT\s+NULL\b", where, re.I):
                        f.append(finding(
                            "sql.left_join_filtered", "WARN",
                            f"WHERE {alias}.<col> IS NOT NULL removes every unmatched row, so this LEFT JOIN behaves as an "
                            "INNER JOIN. Write INNER JOIN if that is intended.", path, line_of(clean, off + b0 + w.start())))

        # = NULL / <> NULL
        scope = stmt
        scope_off = 0
        if is_update:
            w = re.search(r"\bWHERE\b.*", stmt, I)
            scope, scope_off = (w.group(), w.start()) if w else ("", 0)
        for m in re.finditer(r"(=|<>|!=)\s*NULL\b", scope, re.I):
            f.append(finding(
                "sql.null_comparison", "FAIL",
                "Comparing with '= NULL' or '<> NULL' is never true in SQL (the result is NULL), so the condition "
                "silently matches nothing. Use IS NULL / IS NOT NULL.",
                path, line_of(clean, off + scope_off + m.start()), snippet(orig, scope_off + m.start() - 30,
                                                                           scope_off + m.end())))

        # NOT IN over a subquery
        for m in re.finditer(r"\bNOT\s+IN\s*\(\s*SELECT\b", stmt, re.I):
            f.append(finding(
                "sql.not_in_nullable", "WARN",
                "NOT IN (SELECT ...) returns no rows at all if the subquery returns a single NULL. Use NOT EXISTS, "
                "or filter NULLs out of the subquery.", path, line_of(clean, off + m.start()),
                snippet(orig, m.start(), m.end() + 60)))

        # UNION without ALL
        for m in re.finditer(r"\bUNION\b(?!\s+ALL\b)", stmt, re.I):
            f.append(finding(
                "sql.union_dedup", "WARN",
                "UNION removes duplicate rows across both inputs; legitimate repeated rows (two identical orders) "
                "disappear and totals drop. Use UNION ALL unless de-duplication is intended.",
                path, line_of(clean, off + m.start())))

        # Integer division of aggregates
        for m in re.finditer(r"\b(COUNT|SUM)\s*\([^()]*\)\s*/\s*(?:NULLIF\s*\(\s*)?(COUNT|SUM)\s*\(", stmt, re.I):
            ctx = stmt[max(0, m.start() - 25):m.end() + 40]
            if not re.search(r"1\.0|CAST\s*\(|::\s*(float|numeric|decimal|double|real)|\*\s*100\.0|SAFE_DIVIDE|DIV0",
                              ctx, re.I):
                f.append(finding(
                    "sql.integer_division", "WARN",
                    "Dividing COUNT/SUM of integers by another integer truncates to a whole number in PostgreSQL, "
                    "SQL Server, Redshift and others (a 37% rate becomes 0). Multiply by 1.0 or CAST to a decimal first.",
                    path, line_of(clean, off + m.start()), snippet(orig, m.start(), m.end() + 30)))

        # BETWEEN dates on a timestamp column
        for m in re.finditer(r"\b((?:\w+\.)?\w+)\s+BETWEEN\s+(?:DATE\s+)?'(\d{4}-\d{2}-\d{2})'\s+AND\s+(?:DATE\s+)?'(\d{4}-\d{2}-\d{2})'",
                             orig, re.I):
            col = m.group(1).split(".")[-1].lower()
            if re.search(r"(_at$|_ts$|time|timestamp|datetime)", col):
                f.append(finding(
                    "sql.between_timestamp", "WARN",
                    f"'{m.group(1)}' looks like a timestamp, and BETWEEN ... AND '{m.group(3)}' stops at midnight at "
                    f"the start of {m.group(3)}, dropping almost all of that day. Use >= start AND < the day after the end.",
                    path, line_of(clean, off + m.start()), snippet(orig, m.start(), m.end())))

        # AND/OR without parentheses
        for w in re.finditer(r"\b(?:WHERE|HAVING)\b(.*?)" + CLAUSE_END, stmt, I):
            cond = re.sub(r"\bBETWEEN\s+\S+\s+AND\s+\S+", "BETWEEN_X", flatten_parens(w.group(1)), flags=re.I)
            if re.search(r"\bAND\b", cond, re.I) and re.search(r"\bOR\b", cond, re.I):
                f.append(finding(
                    "sql.and_or_precedence", "WARN",
                    "This condition mixes AND and OR without parentheses. AND binds first, so 'a AND b OR c' means "
                    "'(a AND b) OR c', which usually lets unwanted rows through. Add parentheses.",
                    path, line_of(clean, off + w.start()), snippet(orig, w.start(), w.end())))

        # LIMIT / TOP without ORDER BY
        if re.search(r"\bLIMIT\s+\d+|\bSELECT\s+TOP\s+\d+", stmt, re.I) and not re.search(r"\bORDER\s+BY\b", stmt, re.I):
            m = re.search(r"\bLIMIT\b|\bTOP\b", stmt, re.I)
            f.append(finding(
                "sql.limit_without_order", "WARN",
                "LIMIT/TOP without ORDER BY returns an arbitrary set of rows that can change between runs. Add an "
                "ORDER BY that makes the result unique.", path, line_of(clean, off + m.start())))

        # Comma joins with no WHERE
        if re.search(r"\bFROM\s+(?:\w+\.)*\w+(?:\s+(?:AS\s+)?\w+)?\s*,\s*(?:\w+\.)*\w+", stmt, re.I) \
                and not re.search(r"\bWHERE\b", stmt, re.I):
            f.append(finding(
                "sql.cartesian_join", "FAIL",
                "Tables are listed with commas in FROM but there is no WHERE clause, which is a cross join: every row "
                "pairs with every other row. Use explicit JOIN ... ON.", path, line_of(clean, off)))

        # DISTINCT papering over fan-out
        blocks = [stmt[b0:b1] for b0, b1 in select_blocks(stmt)]
        dblock = next((b for b in blocks if re.match(r"SELECT\s+DISTINCT\b", b, re.I) and re.search(r"\bJOIN\b", b, re.I)), None)
        if dblock is not None:
            m = re.search(re.escape(dblock[:40]), stmt)
            f.append(finding(
                "sql.distinct_with_join", "WARN",
                "SELECT DISTINCT on top of a join often hides duplicate rows created by a fan-out join; any SUM or "
                "COUNT computed from the same join is still inflated. Check the join's grain (join-guard) instead.",
                path, line_of(clean, off + m.start())))

        # SELECT * across joins
        if any(re.match(r"SELECT\s+(?:DISTINCT\s+)?\*", b, re.I) and re.search(r"\bJOIN\b", b, re.I) for b in blocks):
            f.append(finding(
                "sql.select_star_join", "WARN",
                "SELECT * across a join returns duplicate column names and changes shape when either table changes. "
                "List the columns.", path, line_of(clean, off)))
    return f, {"statements": n_stmt, "left_joins": n_left}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sql", nargs="+", required=True, help="one or more .sql files")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    findings, stats = [], {"files": 0, "statements": 0, "left_joins": 0}
    for path in a.sql:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"error: file not found: {path}")
        f, s = review(path, p.read_text())
        findings += f
        stats["files"] += 1
        stats["statements"] += s["statements"]
        stats["left_joins"] += s["left_joins"]
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "sql-review", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"sql-review: {status}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        for x in findings:
            print(f"  [{x['severity']}] {x['check']} ({x['file']}:{x.get('line', '?')}): {x['message']}")
            if x.get("examples"):
                print(f"      > {x['examples'][0]}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):      # input errors: print the message and exit 2 (1 is reserved for FAIL)
            print(e.code, file=sys.stderr)
            sys.exit(2)
        raise
