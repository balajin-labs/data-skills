---
name: notebook-reproducibility
description: Check a Jupyter notebook before its results are shared, committed or turned into a pipeline — detects cells run out of order, variables used above the cell that defines them, saved error outputs, skipped cells, unseeded randomness (splits, sampling, models), absolute paths, unpinned pip installs and hard-coded secrets. Use whenever finishing, reviewing, committing or handing over a notebook, converting a notebook to a script or pipeline, or when someone cannot reproduce a notebook's numbers.
---

# Notebook reproducibility

A notebook's saved outputs show what happened in one session, in whatever order the cells were clicked. If "Restart & Run All" gives different numbers, or fails, the results are not reproducible and should not be reported.

## Steps

1. **Run the static check** (nothing is executed):

   ```bash
   python <skill-dir>/scripts/check_notebook.py --notebook <file.ipynb> [<more.ipynb> ...]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. Exit code 1 means a FAIL. Add `--json` for structured output.

2. **Act on the findings**:

   | Finding | What to do |
   | --- | --- |
   | `secrets` FAIL | Revoke the credential first (it may be in git history already), then read it from an environment variable |
   | `use_before_define` FAIL | Move the defining cell up, or the using cell down |
   | `error_outputs` FAIL | Fix the error and re-run everything below it |
   | `out_of_order` / `many_reruns` / `unexecuted_cells` | Restart the kernel and run all cells top to bottom, then save |
   | `no_seed` | Pass `random_state=` to splits, samplers and models, or set a global seed at the top |
   | `absolute_paths` | Use project-relative paths (`Path(__file__).parent`, a `DATA_DIR` setting) |
   | `unpinned_install` | Pin versions (`pip install pandas==2.2.2`) or install from a requirements file |

3. **Prove it**: restart the kernel, run all cells, and check the key numbers match what was reported. If the environment allows it, do this headlessly:

   ```bash
   jupyter nbconvert --to notebook --execute <file.ipynb> --output /tmp/rerun.ipynb
   ```

4. **Before committing**, clear large outputs and any outputs containing data rows (see `pii-scan`), and record the environment (Python and package versions).

## Output

The findings with the fix applied to each, confirmation that a clean top-to-bottom run reproduces the reported numbers, and the environment used. Do not share results from a notebook with an open FAIL.
