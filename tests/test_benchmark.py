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


def test_ab_statistics():
    ab = load("skills/ab-test-check/scripts/check_abtest.py")
    assert abs(ab.chi2_sf(3.841, 1) - 0.05) < 0.001
    assert abs(ab.chi2_sf(5.991, 2) - 0.05) < 0.005
    assert abs(ab.norm_ppf(0.975) - 1.95996) < 1e-4
    assert ab.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_simpsons_adjusted_difference():
    import pandas as pd
    an = load("skills/analysis-review/scripts/check_analysis.py")
    # A better within each segment (0.9 vs 0.8, 0.5 vs 0.4), but A is mostly in the hard segment
    rows = ([("A", "easy", 1)] * 18 + [("A", "easy", 0)] * 2 + [("A", "hard", 1)] * 40 + [("A", "hard", 0)] * 40
            + [("B", "easy", 1)] * 64 + [("B", "easy", 0)] * 16 + [("B", "hard", 1)] * 8 + [("B", "hard", 0)] * 12)
    df = pd.DataFrame(rows, columns=["g", "s", "y"])
    adj, _, coverage, _ = an.adjusted_diff(df["s"], df["g"], df["y"], "A", "B")
    assert df[df.g == "B"].y.mean() > df[df.g == "A"].y.mean()      # B looks better overall
    assert adj < 0 and coverage == 1.0                                 # but is worse within segments


def test_forecast_season_inference():
    fc = load("skills/forecast-check/scripts/check_forecast.py")
    assert [fc.infer_season(d) for d in (1 / 24, 1, 7, 30, 91, 365)] == [24, 7, 52, 12, 4, 1]


def test_manifests_agree_and_skills_are_valid():
    """Claude Code, Codex and portable manifests must agree; every skill needs name + description frontmatter."""
    claude = json.loads((ROOT / ".claude-plugin/plugin.json").read_text())
    codex = json.loads((ROOT / ".codex-plugin/plugin.json").read_text())
    portable = json.loads((ROOT / "plugin.json").read_text())
    assert claude["name"] == codex["name"] == portable["name"] == "data-skills"
    assert claude["version"] == codex["version"] == portable["version"]
    assert (ROOT / codex["skills"]).is_dir()
    for market in (".claude-plugin/marketplace.json", ".agents/plugins/marketplace.json"):
        assert json.loads((ROOT / market).read_text())["plugins"][0]["name"] == "data-skills"
    for skill in sorted((ROOT / "skills").iterdir()):
        text = (skill / "SKILL.md").read_text()
        assert text.startswith("---\n")
        front = text.split("---\n")[1]
        assert f"name: {skill.name}\n" in front
        assert "description: " in front and len(front.split("description: ")[1]) <= 1024


def test_new_skill_helpers():
    pii = load("skills/pii-scan/scripts/check_pii.py")
    assert pii.luhn("4111111111111111") and not pii.luhn("4111111111111112")
    assert pii.mask("jane.doe@mail.com") != "jane.doe@mail.com"
    sql = load("skills/sql-review/scripts/check_sql.py")
    assert sql.flatten_parens("a AND (b OR (c AND d))") == "a AND  _P_ "
    dt = load("skills/datetime-check/scripts/check_datetime.py")
    assert dt.epoch_unit(1_780_000_000) == "seconds" and dt.epoch_unit(1_780_000_000_000) == "milliseconds"
    assert dt.classify("2026-03-01T10:00:00Z") == "iso" and dt.classify("03/01/2026") == "slash"
    nb = load("skills/notebook-reproducibility/scripts/check_notebook.py")
    assert {"model", "X", "y", "np"} <= nb.defined_names("import numpy as np\nmodel = f()\nX, y = load()")


def test_sql_review_scopes_checks_to_each_select_block(tmp_path):
    sql = load("skills/sql-review/scripts/check_sql.py")
    # dbt-style CTE chain: the JOIN lives in one CTE, SELECT * in another -> no select_star_join
    ok = """with a as (select * from raw_customers), b as (select * from raw_orders),
    final as (select a.id, b.amount from a left join b on b.cid = a.id) select * from final"""
    found, _ = sql.review("x.sql", ok)
    assert not [f for f in found if f["check"] in ("sql.select_star_join", "sql.left_join_filtered")]
    # the LEFT JOIN filter bug inside a CTE is still caught
    bad = """with final as (select c.id, o.amount from customers c left join orders o on o.cid = c.id
    where o.status = 'paid') select * from final"""
    found, _ = sql.review("x.sql", bad)
    assert any(f["check"] == "sql.left_join_filtered" and f["severity"] == "FAIL" for f in found)


def test_input_errors_exit_with_code_2(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    p = subprocess.run([sys.executable, str(ROOT / "skills/data-profile/scripts/check_profile.py"), "--data", str(empty)],
                       capture_output=True, text=True)
    assert p.returncode == 2 and "Traceback" not in p.stderr
    p = subprocess.run([sys.executable, str(ROOT / "skills/join-guard/scripts/check_join.py"), "--left", "missing.csv",
                        "--right", "missing.csv", "--on", "id"], capture_output=True, text=True)
    assert p.returncode == 2 and "not found" in p.stderr
