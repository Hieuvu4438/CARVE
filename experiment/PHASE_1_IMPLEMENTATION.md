# Phase 1 implementation prepared — 2026-09-24 09:13 +07

Phase 0 is complete. Section 0.7 of `EXPERIMENT_PLAN.md` requires the user's
"tiếp" before training or evaluation for the next phase. This report records
code prepared in advance; **Phase 1 acceptance and STOP-1 are not complete**.

| Item | Code under `experiment/` | Acceptance still required after "tiếp" |
|---|---|---|
| C1 | `scripts/freeze_and_test.py` reads each run's proposer seed and candidate file, maximizes mean dev Arg-C, supports `--fix`, `--nms_modes`, cached test probabilities and tagged predictions. Registered test variants are checked before test. | Freeze seed 42 and compare the full released rule except timestamp; run byte-identical test regression. |
| C2 | `scripts/propose.py insample --seed S` writes only to `experiment/data/cands`. | Generate 1,278 train rows per seed and measure positive rate. |
| C3 | `clave/verifier.py` supports `use_feats=False`; `scripts/train_verifier.py` supports `--no_feats` and `--train_cands`. Its default feature-enabled state initialization was compared to the released class with a controlled RNG test. | Two one-epoch smoke training runs and the released seed-42 regression. |
| C4 | `scripts/proposer_baseline.py eval` accepts split, τ, min_len, tag and seeds. Its dev invocation writes a timestamped fixed rule before the registered argmax test invocation is allowed. | Run dev command; test command is reserved for Phase 7. |
| C5 | `scripts/collect.py`, `collect_archive.py`, `tables.py`, `analysis.py`, and `bootstrap.py` aggregate official metrics, bootstrap output, candidate costs, oracles and errors into `results/*.json` and T1–T8 Markdown. | Exercise against each phase's actual predictions; full tables remain pending. |

Static verification: `python3 -m compileall -q experiment` passed;
`python3 -m unittest discover -s experiment/tests -p 'test_*.py' -v`
passed all 4 tests (read-only path resolution, registered rule constraints,
feature-enabled verifier state/RNG compatibility plus text-only head shape,
and the test-look ledger using mock data).
`git status --short` reports only `?? experiment/`. No training or new test
evaluation was run in this continuation.

The test scripts now add a `RUNNING` row to `TEST_LOG.md` **before** accessing
test data. They pin the frozen rule by SHA-256, reject a second look under the
same ID, and permit `--resume` only for an interrupted look with the same rule
and prediction paths. Completed rows record Arg-C mean ± sample standard
deviation. T-P1 has the same guard. This was added in the static review on
2026-09-24 09:18 +07; no real test command was invoked.

The archive has a source discrepancy relevant to T6: its single-factor Table 12
states **48.34 ± 1.60** for the single-head-only dev result, while its later
same-grid 2×2 table states **48.31 ± 1.62**. The plan quotes 48.34. Both are
preserved in `results/published_baselines.json` and the T6 table generator;
the 2×2 table will use its own 48.31 figure with a footnote.

All new output directories and code remain inside `experiment/`. Released
checkpoints, candidates, rules, predictions and source files are read only.

On 2026-09-24, `scripts/reproduce.sh` was added as a one-phase-at-a-time
runner for Phases 1–7. It checks disk and available GPU memory before every
training run, records each run in `RUN_LOG.md` through `scripts/log_run.py`,
uses only experiment-owned output paths, and never advances automatically
across a STOP. `scripts/check_test_log.py` audits all ten registered test IDs
and verifies the frozen rule predates each prediction before Phase-7 test
analysis. The runner has passed `bash -n`; it has **not been executed**.

Static table rendering on 2026-09-24 loaded the published baseline rows and
marked later experimental cells pending, without scoring dev or test. The
original CARVE rule was located at
`/home/haipd/SciEvent/method/SciEvent-Next/artifacts/decoding_rules.json` and
added to the Phase-4 collection command. That archive JSON has no `frozen_at`
field and its current file mtime is later than the archived prediction mtime,
so its freeze time cannot be verified from those files. The result will be
labelled as archive without an invented timestamp. The released
`assets/proposer_rule.json` also lacks `frozen_at` and uses `dev_mean`/`dev_std`
field names; the collector records the missing timestamp explicitly and T5
reads its development score from those released fields.
