#!/usr/bin/env python3
"""Check a Jupyter notebook for the problems that make its results impossible to reproduce: cells run out of order,
variables used before the cell that defines them, saved error outputs, skipped cells, unseeded randomness, absolute
paths, unpinned installs and hard-coded secrets.

Usage examples
--------------
    python check_notebook.py --notebook analysis.ipynb
    python check_notebook.py --notebook notebooks/*.ipynb --json

The notebook is read as JSON; nothing is executed. Exit code: 1 if any FAIL finding, 0 otherwise, 2 on bad input.
"""
from __future__ import annotations

import argparse
import builtins
import json
import keyword
import re
import sys
from pathlib import Path

RANDOM_CALLS = re.compile(r"\b(train_test_split|RandomForest\w*|ExtraTrees\w*|GradientBoosting\w*|HistGradientBoosting\w*|"
                          r"KMeans|MiniBatchKMeans|XGB\w*|LGBM\w*|CatBoost\w*|ShuffleSplit|StratifiedShuffleSplit|"
                          r"RandomizedSearchCV|permutation_importance|TSNE|UMAP|MLP\w*|SGD\w*|\.sample)\s*\(")
GLOBAL_RANDOM = re.compile(r"\bnp\.random\.(rand|randn|randint|choice|shuffle|permutation|normal|uniform)\b|"
                           r"\brandom\.(random|randint|choice|shuffle|sample)\b|\btorch\.(rand|randn)\b")
SEEDED = re.compile(r"np\.random\.seed\(|random\.seed\(|default_rng\(\s*\d|manual_seed\(|set_seed\(|tf\.random\.set_seed")
ABS_PATH = re.compile(r"['\"](/Users/[^'\"]+|/home/[^'\"]+|[A-Za-z]:\\\\[^'\"]+|[A-Za-z]:/[^'\"]+)['\"]")
SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|"
                    r"xox[abposr]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{30,}|"
                    r"(?i:(api[_-]?key|secret|password|passwd|token|access[_-]?key)\s*=\s*['\"][^'\"\s]{8,}['\"]))")
PIP = re.compile(r"^\s*[!%]\s*pip\s+install\s+(.+)$", re.M)
KNOWN = set(dir(builtins)) | set(keyword.kwlist) | {"self", "cls", "_", "__", "display", "get_ipython", "In", "Out"}


def finding(check, severity, message, file, cells=None, examples=None):
    out = {"check": check, "severity": severity, "message": message, "columns": [], "value": None, "file": file}
    if cells:
        out["cells"] = cells
    if examples:
        out["examples"] = examples
    return out


def source(cell) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def strip_code(code: str) -> str:
    code = re.sub(r"(?s)('''.*?'''|\"\"\".*?\"\"\")", "''", code)
    code = re.sub(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"", "''", code)
    code = re.sub(r"#[^\n]*", "", code)
    return "\n".join(l for l in code.splitlines() if not l.lstrip().startswith(("!", "%")))


def defined_names(code: str) -> set:
    """Names a cell binds: assignments, def/class, for targets, imports, 'as' aliases, lambda and def arguments."""
    names = set()
    ident = re.compile(r"[A-Za-z_]\w*")
    for line in code.splitlines():
        m = re.match(r"^\s*([A-Za-z_][\w\s,()\[\]]*?)\s*(?:[+\-*/%&|^]|//|\*\*)?=(?!=)", line)
        if m and not re.match(r"^\s*(if|elif|while|return|assert)\b", line):
            names |= set(ident.findall(m.group(1)))
        m = re.match(r"^\s*(?:async\s+)?def\s+(\w+)\s*\((.*)", line)
        if m:
            names.add(m.group(1))
            names |= set(ident.findall(re.sub(r"[:=][^,)]*", "", m.group(2))))
        m = re.match(r"^\s*class\s+(\w+)", line)
        if m:
            names.add(m.group(1))
        for target in re.findall(r"\bfor\s+([\w\s,()]+?)\s+in\b", line):
            names |= set(ident.findall(target))
        for args in re.findall(r"\blambda\s+([\w\s,]*):", line):
            names |= set(ident.findall(args))
        names |= set(re.findall(r"\bas\s+(\w+)", line))
        m = re.match(r"^\s*import\s+(.+)", line)
        if m:
            names |= {p.strip().split(" as ")[-1].split(".")[0] for p in m.group(1).split(",")}
        m = re.match(r"^\s*from\s+\S+\s+import\s+\(?(.+?)\)?\s*$", line)
        if m:
            names |= {p.strip().split(" as ")[-1] for p in m.group(1).split(",") if p.strip()}
    return names


def used_names(code: str) -> set:
    code = re.sub(r"\.\s*[A-Za-z_]\w*", "", code)                  # attributes
    code = re.sub(r"\b[A-Za-z_]\w*\s*=(?!=)", "", code)             # assignments and keyword arguments
    return set(re.findall(r"\b[A-Za-z_]\w*\b", code))


def review(path: str, nb: dict):
    f = []
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code" and source(c).strip()]
    stats = {"code_cells": len(cells)}
    counts = [c.get("execution_count") for c in cells]
    executed = [x for x in counts if x is not None]
    stats["executed_cells"] = len(executed)

    errors = [(i + 1, o.get("ename", "Error")) for i, c in enumerate(cells) for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    if errors:
        f.append(finding("notebook.error_outputs", "FAIL",
                         f"{len(errors)} cell(s) have a saved error ({', '.join(e for _, e in errors[:3])}). Results "
                         "below an error were produced by an earlier, different version of the code.", path,
                         [i for i, _ in errors]))

    if executed and any(b <= a for a, b in zip(executed, executed[1:])):
        jumps = [i + 2 for i, (a, b) in enumerate(zip(executed, executed[1:])) if b <= a]
        f.append(finding("notebook.out_of_order", "WARN",
                         "Cells were executed out of order (execution counts do not increase top to bottom). The "
                         "saved outputs may depend on a sequence nobody can replay. Restart the kernel and run all.",
                         path, jumps[:10], [f"execution counts: {executed[:15]}"]))
    skipped = [i + 1 for i, x in enumerate(counts) if x is None]
    if executed and skipped:
        f.append(finding("notebook.unexecuted_cells", "WARN",
                         f"{len(skipped)} code cell(s) were never run in the saved session, so the outputs do not "
                         "reflect them.", path, skipped[:10]))
    if executed and max(executed) > 2 * len(cells) + 10:
        f.append(finding("notebook.many_reruns", "WARN",
                         f"The highest execution count is {max(executed)} for {len(cells)} cells: cells were re-run many "
                         "times in one session, so hidden state is likely. Restart and run all before trusting results.",
                         path))

    defs = [defined_names(strip_code(source(c))) for c in cells]
    first_def = {}
    for i, d in enumerate(defs):
        for name in d:
            first_def.setdefault(name, i)
    early = []
    for i, c in enumerate(cells):
        for name in used_names(strip_code(source(c))) - KNOWN - defs[i]:
            if first_def.get(name, -1) > i:
                early.append((i + 1, name, first_def[name] + 1))
    if early:
        f.append(finding("notebook.use_before_define", "FAIL",
                         f"{len(early)} name(s) are used in a cell above the one that defines them, so 'Restart & "
                         "Run All' fails or uses different values.", path, sorted({e[0] for e in early})[:10],
                         [f"cell {u} uses '{n}', defined in cell {d}" for u, n, d in early[:5]]))

    all_code = "\n".join(strip_code(source(c)) for c in cells)
    if not SEEDED.search(all_code):
        unseeded = []
        for i, c in enumerate(cells):
            code = strip_code(source(c))
            for m in RANDOM_CALLS.finditer(code):
                depth, j = 0, m.end() - 1
                while j < len(code):
                    depth += {"(": 1, ")": -1}.get(code[j], 0)
                    if depth == 0:
                        break
                    j += 1
                if not re.search(r"random_state|seed", code[m.start():j + 1]):
                    unseeded.append(f"cell {i + 1}: {m.group(1)}(...)")
            if GLOBAL_RANDOM.search(code):
                unseeded.append(f"cell {i + 1}: {GLOBAL_RANDOM.search(code).group()}")
        if unseeded:
            f.append(finding("notebook.no_seed", "WARN",
                             f"{len(unseeded)} random operation(s) run without a seed or random_state, so splits, "
                             "samples and model scores change on every run. Set random_state (or a global seed).",
                             path, None, unseeded[:6]))

    paths = [f"cell {i + 1}: {m.group(1)[:60]}" for i, c in enumerate(cells)
             for m in ABS_PATH.finditer(source(c))]
    if paths:
        f.append(finding("notebook.absolute_paths", "WARN",
                         f"{len(paths)} absolute path(s) tie the notebook to one machine. Use paths relative to the "
                         "project, or a config variable.", path, None, paths[:5]))

    secrets = []
    for i, c in enumerate(nb.get("cells", [])):
        texts = [source(c)] + ["".join(o.get("text", "")) if isinstance(o.get("text"), (list, str)) else ""
                               for o in c.get("outputs", [])]
        for t in texts:
            for m in SECRET.finditer(t):
                s = m.group()
                secrets.append(f"cell {i + 1}: {s[:6]}...{s[-2:]}")
    if secrets:
        f.append(finding("notebook.secrets", "FAIL",
                         f"{len(secrets)} hard-coded credential(s) found. Revoke them now (they may already be in git "
                         "history), then load secrets from environment variables or a secrets manager.",
                         path, None, secrets[:5]))

    unpinned = []
    for c in cells:
        for m in PIP.finditer(source(c)):
            pkgs = [p for p in m.group(1).split() if not p.startswith("-")]
            unpinned += [p for p in pkgs if not re.search(r"[=<>~]=|@|\.txt$|\.whl$|/", p)]
    if unpinned:
        f.append(finding("notebook.unpinned_install", "WARN",
                         f"Packages installed without a version ({', '.join(unpinned[:6])}). Results can change when a "
                         "new version is released. Pin versions or use a requirements file.", path))
    return f, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", nargs="+", required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    findings, stats = [], {}
    for path in a.notebook:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"error: file not found: {path}")
        try:
            nb = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise SystemExit(f"error: {path} is not a valid notebook: {e}")
        fs, st = review(path, nb)
        findings += fs
        stats[path] = st
    status = "FAIL" if any(x["severity"] == "FAIL" for x in findings) else "WARN" if findings else "PASS"
    report = {"tool": "notebook-reproducibility", "status": status, "stats": stats, "findings": findings}
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"notebook-reproducibility: {status}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        for x in findings:
            where = f" (cells {x['cells']})" if x.get("cells") else ""
            print(f"  [{x['severity']}] {x['check']}{where}: {x['message']}")
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
