# data-skills

**Senior-reviewer skills for AI-generated data work.** Give Claude Code, Codex, Devin, Cursor or any agent that reads `SKILL.md` the habits of a careful data scientist: catch leakage, fan-out joins, broken A/B tests, misleading comparisons, uncalibrated or unfair models, buggy SQL, non-idempotent pipelines, leaked PII and unreproducible notebooks *before* the number reaches a slide.

AI agents write data code that runs. The dangerous mistakes are the ones that run *and* return a plausible number:

- a churn model at 0.97 AUC because `account_status` is a copy of the label,
- revenue up 8% because a customer table keeps history rows and the join doubled orders,
- a "better" model whose improvement sits inside the noise,
- a monthly extract that quietly switched from dollars to cents,
- an A/B test "win" from a split that lost 6% of treatment users,
- a channel that "converts better" only because it gets more returning customers,
- a demand forecast that loses to "same day last week",
- a SQL filter that quietly turned a LEFT JOIN into an INNER JOIN,
- a retention chart where the newest cohorts "collapse" because their months are not over yet.

Each skill pairs instructions for the agent with a **deterministic check script**, so the verdict comes from code, not from the model's opinion of its own work.

## Skills

Twenty skills, each with a check script. Every script name is `skills/<skill>/scripts/check_*.py`.

**Data quality**

| Skill | Use it when | Catches |
| --- | --- | --- |
| [`data-profile`](skills/data-profile/SKILL.md) | Exploring a dataset for the first time | Impossible values, placeholder codes (`-999`, `'N/A'`, `1900-01-01`), mixed types, labels written several ways, non-unique IDs, duplicate rows, future dates |
| [`data-contract`](skills/data-contract/SKILL.md) | Defining or enforcing what a table must look like | Missing columns, wrong types, nulls, out-of-range and unknown values, pattern breaks, duplicate primary keys, short or stale deliveries |
| [`null-type-drift`](skills/null-type-drift/SKILL.md) | Loading a new extract or batch | Dropped columns, type changes, null spikes, unit changes, new or vanished categories, duplicate loads |
| [`datetime-check`](skills/datetime-check/SKILL.md) | Parsing dates or combining sources | dd/mm mixed with mm/dd, epoch seconds mixed with milliseconds, naive mixed with UTC timestamps, several offsets, placeholder and future dates |
| [`pii-scan`](skills/pii-scan/SKILL.md) | Sharing, exporting or prompting with data | SSNs, card numbers, bank accounts, emails, phones, IPs, names, addresses, birth dates, PII inside free-text notes |

**Data engineering**

| Skill | Use it when | Catches |
| --- | --- | --- |
| [`sql-review`](skills/sql-review/SKILL.md) | Writing or reviewing SQL | LEFT JOINs turned into INNER JOINs, `= NULL`, `NOT IN` over NULLs, `UNION` dropping rows, integer division, `BETWEEN` on timestamps, AND/OR precedence, cross joins |
| [`join-guard`](skills/join-guard/SKILL.md) | Writing or reviewing a join or merge | Fan-out from duplicate keys, double-counted totals, unmatched keys, zero-padding and case mismatches, null keys |
| [`reconcile`](skills/reconcile/SKILL.md) | Migrating or rebuilding a table, or when two reports disagree | Missing and extra keys, duplicate keys, value mismatches (with unit and rounding hints), totals that do not tie out |
| [`pipeline-idempotency`](skills/pipeline-idempotency/SKILL.md) | Building ETL, backfills or retries | Reruns that append duplicates, keys that change between runs, values that depend on row order, sampling or the clock |

**Machine learning**

| Skill | Use it when | Catches |
| --- | --- | --- |
| [`leakage-audit`](skills/leakage-audit/SKILL.md) | Training or reviewing a model | Target proxies, entity overlap across splits, random splits on time-ordered data, features recorded after the prediction moment |
| [`eval-sanity`](skills/eval-sanity/SKILL.md) | Reporting metrics or comparing models | No better than a trivial baseline, bad thresholds, "improvements" inside the noise, failing segments, accuracy on imbalanced data |
| [`calibration-check`](skills/calibration-check/SKILL.md) | Using predicted probabilities as probabilities | Average prediction far from the real rate (down-sampled training), high calibration error, overconfidence, mis-calibrated segments |
| [`fairness-audit`](skills/fairness-audit/SKILL.md) | A model decides something about people | Four-fifths-rule failures, true- and false-positive-rate gaps between groups, groups too small to assess |
| [`train-serve-skew`](skills/train-serve-skew/SKILL.md) | Deploying a model, or it underperforms in production | Missing features, unit and scaling mismatches, unseen categories, null-rate changes, features computed differently online vs offline |
| [`forecast-check`](skills/forecast-check/SKILL.md) | Building or backtesting a forecast | No better than naive or seasonal-naive, test overlapping training, bias, suspiciously perfect forecasts, duplicate or missing dates, MAPE on zeros |

**Analysis and statistics**

| Skill | Use it when | Catches |
| --- | --- | --- |
| [`analysis-review`](skills/analysis-review/SKILL.md) | Comparing groups or explaining a metric | Simpson's paradox, gaps explained by segment mix, differences hidden by mix, rankings topped by tiny groups, outlier-driven averages |
| [`ab-test-check`](skills/ab-test-check/SKILL.md) | Analysing an experiment or rollout | Sample ratio mismatch, users in both variants, underpowered "no effect" claims, uncorrected multiple metrics, novelty effects, pre-period imbalance |
| [`cohort-retention`](skills/cohort-retention/SKILL.md) | Building retention or churn curves | Unfinished periods making recent cohorts look worse, rates above 100%, broken period-0 definitions, impossible rises, tiny cohorts |
| [`regression-diagnostics`](skills/regression-diagnostics/SKILL.md) | Interpreting regression coefficients | Perfect and strong multicollinearity, heteroscedasticity, non-linearity, autocorrelation, influential points, skewed residuals |

**Reproducibility**

| Skill | Use it when | Catches |
| --- | --- | --- |
| [`notebook-reproducibility`](skills/notebook-reproducibility/SKILL.md) | Sharing, committing or productionising a notebook | Cells run out of order, variables used before definition, saved errors, unseeded randomness, absolute paths, unpinned installs, hard-coded secrets |

## Quick start

```bash
git clone https://github.com/balajin-labs/data-skills && cd data-skills
pip install -r requirements.txt
./scripts/install.sh          # installs for Codex, Devin, Cursor, Gemini CLI and other .agents tools, plus Claude Code
```

Then ask your agent something like "analyse this A/B test" or "is this AUC right?". The matching skill loads on its own.

## Install

The skills use the open [Agent Skills](https://agentskills.io) format (a folder with `SKILL.md` and scripts), so the same folders work in every agent below. Pick one method per agent; installing the same skill twice makes it show up twice.

| Agent | Install |
| --- | --- |
| **Claude Code** | `/plugin marketplace add balajin-labs/data-skills` then `/plugin install data-skills@data-skills` |
| **Codex** (CLI, app, IDE) | `codex plugin marketplace add balajin-labs/data-skills`, then install *Data Skills* from `/plugins`. Or use the install script |
| **Devin / Windsurf** | The install script (Devin reads `~/.agents/skills`), or `./scripts/install.sh devin --project <repo>` to commit them to `.devin/skills/` |
| **Cursor, Gemini CLI, GitHub Copilot, OpenCode, Amp and others** | `npx skills add balajin-labs/data-skills`, or the install script |

### Install script

Works for any agent that reads a skills folder:

```bash
./scripts/install.sh                         # ~/.agents/skills + ~/.claude/skills (covers most agents)
./scripts/install.sh cursor copilot          # specific agents: agents|codex, claude, devin, cursor, copilot, all
./scripts/install.sh --project ~/my-repo     # into one repository instead of your user folders
./scripts/install.sh --link                  # symlink instead of copy, so `git pull` updates the skills
./scripts/install.sh --uninstall             # remove them again
./scripts/install.sh --help
```

| Agent name | User folder | With `--project <repo>` |
| --- | --- | --- |
| `agents` (alias `codex`) | `~/.agents/skills` | `<repo>/.agents/skills` |
| `claude` | `~/.claude/skills` | `<repo>/.claude/skills` |
| `devin` | `~/.config/devin/skills` | `<repo>/.devin/skills` |
| `cursor` | `~/.cursor/skills` | `<repo>/.cursor/skills` |
| `copilot` | `~/.copilot/skills` | `<repo>/.github/skills` |

### Cloud agents

For Devin sessions, Codex cloud tasks or the Copilot coding agent, install the skills into the repository the agent works on and commit them, so every session that checks out the repo has them:

```bash
./scripts/install.sh --project ~/my-repo
cd ~/my-repo && git add .agents .claude && git commit -m "Add data-skills"
```

Make sure the agent's environment setup runs `pip install pandas numpy scikit-learn`.

### Manually

Copy the folders under `skills/` into your agent's skills directory, for example `~/.agents/skills/`, `~/.claude/skills/` or `.agents/skills/` in a project.

## Requirements

The check scripts need Python 3.9+ with `pandas`, `numpy` and `scikit-learn` (`pip install -r requirements.txt`). Add `pyarrow` to read Parquet. The skills themselves are plain Markdown and need nothing extra.

Tested on Python 3.9 with the oldest supported libraries (pandas 2.0, numpy 1.24, scikit-learn 1.3) and on Python 3.13 with current ones, with CSV and Parquet input. On a 1-million-row file every check finishes in under 25 seconds using under 2 GB of memory, except `eval-sanity`, which bootstraps confidence intervals and takes about a minute.

`scripts/install.sh` needs a Bash shell (macOS, Linux, WSL or Git Bash). On Windows without one, use `npx skills add balajin-labs/data-skills` or copy the folders manually; the Python scripts themselves are cross-platform.

## How it works

Once installed, the agent loads a skill when your request matches its description, for example "train a churn model", "is this AUC right?", "join orders to customers", "load this month's extract", "analyse this A/B test", "why does the app convert better?", "backtest this demand forecast", "review this SQL", "is this data safe to share?" or "clean up this notebook".

You can also call a skill by name:

| Agent | Example |
| --- | --- |
| Claude Code | `/data-skills:leakage-audit` (plugin) or `/leakage-audit` (skills folder) |
| Codex | `$leakage-audit` |
| Devin | `/leakage-audit` |

The skill tells the agent to state its assumptions, run the check script, fix what it finds and report the verdict. The scripts also run on their own:

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

```bash
python skills/ab-test-check/scripts/check_abtest.py --data users.csv --variant variant \
  --control control --metric converted --expected-split 50,50
```

```
ab-test-check: FAIL
  units_per_variant: {'control': 10047, 'treatment': 9343}
  srm_p_value: 4.29e-07
  converted [treatment]: control 0.0999, treatment 0.111, lift 0.1107, p 0.0123 (Holm 0.0123)
  [FAIL] ab.srm: Sample ratio mismatch: control 51.82% (expected 50.00%), treatment 48.18% (expected 50.00%);
         p = 4.3e-07. ... Every metric is suspect until this is explained.
```

(Output trimmed; the `ab-01-srm` case, where a redirect drops 6% of treatment users. The 11% lift looks significant but cannot be trusted.)

Every script prints a readable report, supports `--json`, and exits with code 0 on PASS or WARN, 1 on any FAIL and 2 on an input error (missing file or column, empty file), so it also works as a CI gate.

## Repository layout

```
skills/<skill>/SKILL.md            instructions the agent follows
skills/<skill>/scripts/check_*.py  the deterministic check
.claude-plugin/                    Claude Code plugin and marketplace
.codex-plugin/plugin.json          Codex plugin manifest
.agents/plugins/marketplace.json   Codex marketplace
plugin.json                        portable Agent Plugins manifest
scripts/install.sh                 installer for skills folders (Codex, Devin, Cursor, Copilot, ...)
benchmark/                         seeded-bug datasets (CSV, SQL, JSON, notebooks) and runner
tests/                             pytest suite (benchmark, helpers, manifests)
```

## Benchmark

`benchmark/` generates small, realistic datasets, each with one planted mistake, plus clean controls that must pass:

```bash
python benchmark/generate.py
python benchmark/run.py
```

Current result: **65 of 65 planted bugs caught, 0 false alarms on 20 clean controls** (one per skill), stable across twelve random seeds.

This benchmark is a regression suite for the checks themselves. The next step is the more interesting one: run coding agents on the same tasks *with and without* these skills and measure how many mistakes reach the final answer. Contributions of new seeded-bug cases are the most valuable way to help.

## Roadmap

- More skills: `clustering-sanity` (unscaled features, unstable clusters), `experiment-honesty` (model cards, run comparisons), `causal-inference` (confounding, parallel trends).
- Agent benchmark: same tasks run by Claude Code, Codex and Devin, with and without the skills.
- Direct warehouse checks (DuckDB, Snowflake, BigQuery) instead of exported files.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New checks need a seeded-bug case that the check catches and a clean case it passes. When releasing, bump `version` in `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json` and `plugin.json` together; `pytest` fails if they disagree.

## License

MIT
