"""End-to-end test: every planted bug is caught and no clean control raises a FAIL."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_benchmark_catches_every_bug_without_false_alarms():
    subprocess.run([sys.executable, str(ROOT / "benchmark/generate.py")], check=True, capture_output=True)
    proc = subprocess.run([sys.executable, str(ROOT / "benchmark/run.py"), "--json"],
                          capture_output=True, text=True)
    summary = json.loads(proc.stdout)["summary"]
    assert summary["errors"] == 0
    assert summary["caught"] == summary["bug_cases"]
    assert summary["false_alarms"] == 0


def test_join_key_normalisation():
    import pandas as pd
    jg = load("skills/join-guard/scripts/check_join.py")
    out = jg.normalise(pd.Series(["00123", " AB-12 ", "123.0", None]))
    assert list(out[:3]) == ["123", "ab12", "123"]
    assert pd.isna(out[3])


def test_unit_ratio_detection():
    drift = load("skills/null-type-drift/scripts/check_drift.py")
    assert drift.unit_ratio(50.0, 5000.0) == 100
    assert drift.unit_ratio(5000.0, 5.0) == 1 / 1000
    assert drift.unit_ratio(50.0, 55.0) is None


def test_rank_auc():
    import numpy as np
    leak = load("skills/leakage-audit/scripts/check_leakage.py")
    y = np.array([0, 0, 1, 1])
    assert leak.rank_auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert leak.rank_auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
