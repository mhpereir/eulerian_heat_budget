---
description: Run pytest on the eulerian_heat_budget test suite using the mamba dev_env environment
allowed-tools: Bash, Read, Grep, Glob
---

Run the project's pytest test suite inside the `dev_env` mamba environment.

## Arguments

- If `$ARGUMENTS` is empty, run the full test suite.
- If `$ARGUMENTS` names a specific test file (e.g. `test_grid`), run only that file.
- If `$ARGUMENTS` contains a test function pattern (e.g. `test_grid::test_crop`), run only that test.
- If `$ARGUMENTS` is `--last-failed` or `-lf`, rerun only previously failed tests.

## Instructions

1. Run pytest from the project root (`/home/mhpereir/eulerian_heat_budget`) using:
   ```
   mamba run -n dev_env python -m pytest tests/$TARGET -v --tb=short 2>&1
   ```
   where `$TARGET` is derived from `$ARGUMENTS` (empty = all tests).

2. Read the output carefully. For each failure:
   - Identify the failing test name and file location.
   - Read the relevant test code and the source code it exercises.
   - Diagnose the root cause — consider whether the failure is:
     - A numerical tolerance issue (common with `np.testing.assert_allclose`)
     - A dimension ordering or coordinate mismatch
     - A missing or renamed function/variable
     - An actual physics or logic bug
   - Report a concise diagnosis for each failure.

3. Summarize results:
   - Total passed / failed / skipped / errors
   - For failures: file, test name, one-line diagnosis
   - Suggest fixes if the cause is clear

Do NOT attempt to run the full pipeline (`run_budget.py`) or submit PBS jobs. Only run pytest.
