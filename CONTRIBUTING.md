# Contributing

The goal of this project is simple: fewer wrong numbers. The bar for a new check is that it catches a mistake people actually make, and stays quiet on correct work.

## Adding or changing a check

1. Write the check in the skill's script. Keep it deterministic (seeded randomness only) and dependent on `pandas`, `numpy` and `scikit-learn` only.
2. Return findings in the shared shape: `{"check", "severity": "WARN"|"FAIL", "message", "columns", "value"}`. Use FAIL only when the result is very likely wrong; use WARN when a human should look.
3. Write the message for the person reading it: what is wrong, the evidence (numbers), and what to do.
4. Add a seeded-bug case to `benchmark/generate.py` that the check must catch, and make sure every clean control still passes.
5. Update the skill's `SKILL.md` table so the agent knows how to act on the new finding.
6. Run `pytest -q`.

## Adding a skill

A skill is a folder under `skills/` with a `SKILL.md` (frontmatter `name` and a `description` that says *when* to use it) and, ideally, a `scripts/` folder with a check. The description decides when agents load the skill, so write it in the words people use when they ask for that kind of work.

## Reporting a false alarm or a miss

Open an issue with a minimal dataset (synthetic is fine) and the command you ran. Never attach real customer, patient or employee data.
