#!/usr/bin/env python3
"""Run every skill check against every benchmark case and report the catch rate.

    python benchmark/generate.py
    python benchmark/run.py [--only leakage-audit] [--json]

A bug case is CAUGHT when every expected finding appears at (or above) its expected severity.
A clean case is a FALSE ALARM when any FAIL is raised.
Exit code 1 if any bug is missed or any clean case raises a FAIL.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "benchmark" / "cases"
SCRIPTS = {
    "leakage-audit": "skills/leakage-audit/scripts/check_leakage.py",
    "eval-sanity": "skills/eval-sanity/scripts/check_eval.py",
    "join-guard": "skills/join-guard/scripts/check_join.py",
    "null-type-drift": "skills/null-type-drift/scripts/check_drift.py",
}
RANK = {"WARN": 1, "FAIL": 2}


def run_case(case_dir: Path) -> dict:
    case = json.loads((case_dir / "case.json").read_text())
    script = ROOT / SCRIPTS[case["skill"]]
    args = [a.replace("{dir}", str(case_dir)) for a in case["args"]]
    proc = subprocess.run([sys.executable, str(script), *args, "--json"], capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return {**case, "outcome": "ERROR", "detail": proc.stderr.strip()[-500:]}
    report = json.loads(proc.stdout)
    found = {}
    for f in report["findings"]:
        found[f["check"]] = max(found.get(f["check"], 0), RANK[f["severity"]])

    if case["bug"]:
        missing = [e for e in case["expect"] if found.get(e["check"], 0) < RANK[e["severity"]]]
        outcome = "CAUGHT" if not missing else "MISSED"
        detail = "" if not missing else "missing: " + ", ".join(f"{m['check']}>={m['severity']}" for m in missing)
    else:
        fails = [c for c, r in found.items() if r == RANK["FAIL"]]
        outcome = "CLEAN" if not fails else "FALSE_ALARM"
        detail = "" if not fails else "raised: " + ", ".join(fails)
    return {**case, "outcome": outcome, "detail": detail, "status": report["status"],
            "found": sorted(found)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run one skill's cases")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if not CASES.exists():
        raise SystemExit("no cases found: run `python benchmark/generate.py` first")
    results = [run_case(d) for d in sorted(CASES.iterdir()) if (d / "case.json").exists()]
    if a.only:
        results = [r for r in results if r["skill"] == a.only]

    bugs = [r for r in results if r["bug"]]
    clean = [r for r in results if not r["bug"]]
    caught = sum(r["outcome"] == "CAUGHT" for r in bugs)
    false_alarms = sum(r["outcome"] == "FALSE_ALARM" for r in clean)
    errors = sum(r["outcome"] == "ERROR" for r in results)
    summary = {"bug_cases": len(bugs), "caught": caught, "clean_cases": len(clean),
               "false_alarms": false_alarms, "errors": errors}
    (ROOT / "benchmark" / "results.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    if a.json:
        print(json.dumps({"summary": summary, "results": results}, indent=2))
    else:
        width = max(len(r["id"]) for r in results)
        for r in results:
            print(f"{r['id']:<{width}}  {r['outcome']:<11}  {r['detail']}")
        print(f"\nCaught {caught}/{len(bugs)} planted bugs; "
              f"{false_alarms} false alarms on {len(clean)} clean controls; {errors} errors.")
    return 0 if caught == len(bugs) and false_alarms == 0 and errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
