#!/usr/bin/env python3
"""Scan a dataset for personal data before it is shared, published, logged, sent to an LLM or used in a demo:
government IDs, payment cards, bank accounts, emails, phone numbers, IP addresses, names, addresses, birth dates and
sensitive attributes, including personal data buried inside free-text columns.

Usage examples
--------------
    python check_pii.py --data export.csv
    python check_pii.py --data tickets.parquet --sample 50000

High-risk identifiers (SSNs, card numbers, bank accounts, passport or national IDs) are FAIL; other personal data is
WARN. Matches are reported by column with masked examples; values are never printed in full.
Exit code: 1 if any FAIL finding, 0 otherwise, 2 on bad input. Add --json for structured output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
PHONE_FMT = re.compile(r"(?:\+\d{1,3}[\s.-]?)?\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b|\+\d{8,15}\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")

NAME_RULES = [
    (re.compile(r"(ssn|social_?security|national_?id|nin$|aadhaar|passport|driver_?licen[cs]e|tax_?id|tin$)", re.I),
     "government_id", "FAIL"),
    (re.compile(r"(card_?number|credit_?card|cc_?num|pan$|cvv|iban|account_?number|routing_?number)", re.I),
     "financial_account", "FAIL"),
    (re.compile(r"(^|_)(first|last|full|given|family|middle|sur)_?name($|_)|^name$|customer_name|patient_name", re.I),
     "person_name", "WARN"),
    (re.compile(r"(street|address|addr_line|postcode|postal_code|zip_?code|^zip$)", re.I), "address", "WARN"),
    (re.compile(r"(dob|date_of_birth|birth_?date|birthday)", re.I), "birth_date", "WARN"),
    (re.compile(r"(email|e_mail)", re.I), "email", "WARN"),
    (re.compile(r"(phone|mobile|cell|tel$|telephone)", re.I), "phone", "WARN"),
    (re.compile(r"(ethnic|race|religio|sexual|gender_identity|health|diagnos|disab|medical|union_member|politic)", re.I),
     "sensitive_attribute", "WARN"),
    (re.compile(r"(ip_?addr|^ip$|client_ip|user_ip)", re.I), "ip_address", "WARN"),
    (re.compile(r"(lat(itude)?$|lon(gitude)?$|^lng$|geo_?point)", re.I), "precise_location", "WARN"),
]


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: file not found: {path}")
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
        return df.astype(str).where(df.notna())
    if p.suffix.lower() in {".json", ".jsonl"}:
        df = pd.read_json(p, lines=p.suffix.lower() == ".jsonl")
        return df.astype(str).where(df.notna())
    return pd.read_csv(p, sep="\t" if p.suffix.lower() in {".tsv", ".tab"} else ",", dtype=str)


def finding(check, severity, message, columns=None, value=None, examples=None):
    out = {"check": check, "severity": severity, "message": message, "columns": columns or [], "value": value}
    if examples:
        out["examples"] = examples
    return out


def luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def is_card(v: str) -> bool:
    for m in CARD.finditer(v):
        d = re.sub(r"\D", "", m.group())
        if 13 <= len(d) <= 19 and len(set(d)) > 2 and luhn(d):
            return True
    return False


def mask(v: str) -> str:
    v = str(v)
    keep = max(1, len(v) // 5)
    return v[:keep] + "*" * max(3, len(v) - 2 * keep) + v[-keep:] if len(v) > 4 else "****"


def scan(df, sample):
    f, stats = [], {"rows": len(df), "columns": len(df.columns)}
    detected = {}
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if len(s) > sample:
            s = s.sample(sample, random_state=0)
        s = s[s.str.strip() != ""]
        hits = {}
        if len(s):
            free_text = s.str.len().mean() > 30 and s.str.contains(" ").mean() > 0.5
            detectors = {
                "ssn": ("FAIL", s.map(lambda v: bool(SSN.search(v)))),
                "credit_card": ("FAIL", s.map(is_card)),
                "iban": ("FAIL", s.map(lambda v: bool(IBAN.search(v)))),
                "email": ("WARN", s.map(lambda v: bool(EMAIL.search(v)))),
                "phone": ("WARN", s.map(lambda v: bool(PHONE_FMT.search(v)))),
                "ip_address": ("WARN", s.map(lambda v: bool(IPV4.search(v)))),
            }
            for kind, (sev, mask_) in detectors.items():
                share = float(mask_.mean())
                count = int(mask_.sum())
                # structured columns need most values to match; free text needs a handful
                if (free_text and count >= 1 and share >= 0.005) or share >= 0.3 or (kind != "credit_card" and share >= 0.05):
                    if kind == "credit_card" and not free_text and share < 0.5:
                        continue          # random long numbers pass Luhn 10% of the time
                    hits[kind] = (sev, share, [mask(v) for v in s[mask_].head(3)])
            if free_text and hits:
                worst = "FAIL" if any(h[0] == "FAIL" for h in hits.values()) else "WARN"
                kinds = ", ".join(f"{k} ({h[1]:.1%} of rows)" for k, h in hits.items())
                f.append(finding(
                    "pii.free_text", worst,
                    f"Free-text column '{c}' contains personal data: {kinds}. Notes, comments and messages are the "
                    "most common way PII leaks into exports and prompts. Redact it before sharing.", [c],
                    round(max(h[1] for h in hits.values()), 4), [e for h in hits.values() for e in h[2]][:4]))
                detected[c] = list(hits)
                continue
        for kind, (sev, share, ex) in hits.items():
            f.append(finding(f"pii.{kind}", sev,
                             f"'{c}' contains {kind.replace('_', ' ')} values ({share:.0%} of sampled rows).", [c],
                             round(share, 4), ex))
        named = []
        for rx, kind, sev in NAME_RULES:
            if rx.search(c) and kind not in hits and kind not in named:
                named.append(kind)
                hashed = len(s) and s.str.fullmatch(r"[0-9a-fA-F]{32,128}").mean() > 0.9
                if hashed:
                    continue
                f.append(finding(f"pii.{kind}", sev,
                                 f"Column name '{c}' suggests {kind.replace('_', ' ')}.", [c], None,
                                 [mask(v) for v in s.head(2)] if len(s) else None))
        if hits or named:
            detected[c] = list(hits) + named
    stats["columns_with_pii"] = detected
    return f, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV, TSV, Parquet, JSON or JSONL file")
    ap.add_argument("--sample", type=int, default=20000, help="max values scanned per column")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    df = read_table(a.data)
    findings, stats = scan(df, a.sample)
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "pii-scan", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"pii-scan: {status}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        for x in findings:
            print(f"  [{x['severity']}] {x['check']}: {x['message']}")
            if x.get("examples"):
                print(f"      masked examples: {x['examples']}")
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
