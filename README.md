# data-skills

**Senior-reviewer skills for AI-generated data work.** Give Claude Code, Codex, Cursor or any agent that reads `SKILL.md` the habits of a careful data scientist: catch leakage, meaningless evaluations, fan-out joins and silent data drift *before* the number reaches a slide.

AI agents write data code that runs. The dangerous mistakes are the ones that run *and* return a plausible number:

- a churn model at 0.97 AUC because `account_status` is a copy of the label,
- revenue up 8% because a customer table keeps history rows and the join doubled orders,
- a "better" model whose improvement sits inside the noise,
- a monthly extract that quietly switched from dollars to cents.

Each skill pairs instructions for the agent with a **deterministic check script**, so the verdict comes from code, not from the model's opinion of its own work.

## Skills

| Skill | Catches | Script |
| --- | --- | --- |
| [`leakage-audit`](skills/leakage-audit/SKILL.md) | Target proxies, entity overlap across splits, random splits on time-ordered data, features recorded after the prediction moment, duplicate rows across splits | `check_leakage.py` |
| [`eval-sanity`](skills/eval-sanity/SKILL.md) | Models no better than a trivial baseline, badly chosen thresholds, "improvements" inside the noise, segments where the model fails, accuracy on imbalanced data | `check_eval.py` |
| [`join-guard`](skills/join-guard/SKILL.md) | Fan-out from duplicate keys, double-counted totals, unmatched keys, zero-padding and case mismatches, null keys | `check_join.py` |
| [`null-type-drift`](skills/null-type-drift/SKILL.md) | Dropped columns, type changes, null spikes, unit changes, new or vanished categories, duplicate loads, placeholder values like `-999` | `check_drift.py` |

## Install

**Claude Code (plugin):**

```
/plugin marketplace add <github-user>/data-skills
/plugin install data-skills@data-skills
```

**Any agent that supports Agent Skills** (Codex, Cursor, Gemini CLI, Copilot, OpenCode and others):

```
npx skills add <github-user>/data-skills
```

**Manually:** copy the folders under `skills/` into your agent's skills directory (for example `~/.claude/skills/` or `.agents/skills/` in a project).

The scripts need Python 3.9+ with `pandas`, `numpy` and `scikit-learn` (`pip install -r requirements.txt`). Add `pyarrow` to read Parquet.

## How it works

Once installed, the agent loads a skill when your request matches its description, for example "train a churn model", "is this AUC right?", "join orders to customers" or "load this month's extract". You can also call one by name (`/data-skills:leakage-audit` in Claude Code).

The scripts also run on their own:

```bash
python skills/join-guard/scripts/check_join.py --left orders.csv --right customers.csv \
  --on customer_id --sum-col order_total
```

```
join-guard: FAIL
  left_rows: 3000
  right_rows: 864
  output_rows: 3233
  row_multiplier: 1.0777
  sum_before: 301913.85
  sum_after: 324256.24
  [FAIL] join.fan_out: Right side has 64 duplicated keys, so 233 left rows match more than once.
         The join returns 3,233 rows from 3,000 left rows (x1.08). ...
  [FAIL] join.double_count: Total 'order_total' changes from 301,913.85 to 324,256.24 after the join (+7.4%). ...
```

(Output trimmed; this is the `join-01-fan-out` benchmark case, where 8% of customers have a history row.)

Every script prints a readable report, supports `--json`, and exits with code 1 on any FAIL, so it also works as a CI gate.

## Benchmark

`benchmark/` generates small, realistic datasets, each with one planted mistake, plus clean controls that must pass:

```bash
python benchmark/generate.py
python benchmark/run.py
```

Current result: **15 of 15 planted bugs caught, 0 false alarms on 4 clean controls**, stable across six random seeds.

This benchmark is a regression suite for the checks themselves. The next step is the more interesting one: run coding agents on the same tasks *with and without* these skills and measure how many mistakes reach the final answer. Contributions of new seeded-bug cases are the most valuable way to help.

## Roadmap

- More skills: `sql-review` (grain and filter bugs), `experiment-honesty` (model cards, run comparisons), `pipeline-idempotency` (backfills, late data), `analysis-review` (Simpson's paradox, survivorship, causal claims from correlations).
- Agent benchmark: same tasks run by Claude Code and Codex, with and without the skills.
- Direct warehouse checks (DuckDB, Snowflake, BigQuery) instead of exported files.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New checks need a seeded-bug case that the check catches and a clean case it passes.

## License

MIT
