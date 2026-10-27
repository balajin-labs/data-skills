---
name: data-contract
description: Validate a dataset against a data contract — expected columns, types, nullability, ranges, allowed values, patterns, primary key, row count and freshness — or draft a contract from data you trust. Use whenever defining what a table or feed must look like, adding data quality checks or tests to a pipeline, onboarding a new source, or when a downstream job breaks because upstream data changed shape.
---

# Data contract

A data contract writes down what consumers rely on, so a change upstream fails loudly at the boundary instead of quietly in a dashboard three steps later.

## Steps

1. **Draft the contract from a known-good extract**:

   ```bash
   python <skill-dir>/scripts/check_contract.py --data <good extract> --infer > <name>.contract.json
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. Inference only describes this extract, so edit the result with the owners of the data. Tighten ranges to what is *valid*, not what was observed. Mark which columns may be null, list allowed status codes, set the primary key, the minimum row count and how fresh the data must be.

2. **Validate each new delivery** (in the pipeline, before anything downstream runs):

   ```bash
   python <skill-dir>/scripts/check_contract.py --data <file> --contract <name>.contract.json [--as-of YYYY-MM-DD]
   ```

   Exit code 1 means a FAIL, which should stop the pipeline. Add `--json` for structured output.

3. **Act on the findings**:

   | Finding | What to do |
   | --- | --- |
   | `missing_column` / `extra_columns` / `type` | Schema change upstream: agree on it, then version the contract |
   | `allowed_values` | A new code appeared: map it deliberately, never pass it through unmapped |
   | `range` / `pattern` / `max_length` | Bad records or a unit or format change: quarantine the rows and tell the producer |
   | `nullable` | Required field missing: find which upstream path stopped populating it |
   | `primary_key` / `unique` | Duplicate load or wrong grain; see `pipeline-idempotency` |
   | `row_count` | Partial or duplicated extract |
   | `freshness` | The feed stopped; alert the owner before anyone reads stale numbers |

4. **Keep the contract in version control** next to the pipeline, and change it only through review, the same as code.

## Output

The contract file (or the diff to it), the validation verdict, and for each finding the rows affected and the decision taken (block, quarantine or accept with a contract change).
