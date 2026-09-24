# Phase 1 report — 2026-09-24

Command: `bash experiment/scripts/reproduce.sh 1`, from `/home/haipd/CLAVE` with the plan's environment. The command finished successfully. All new code and output are under `experiment/`; `git status --short` shows only `?? experiment/`.

## C1–C5 acceptance

| Item | Change and result |
|---|---|
| C1 | `scripts/freeze_and_test.py` now selects a shared rule by mean dev Arg-C across runs while loading each run's proposer seed and candidates; it supports fixed grid dimensions, tagged predictions, cached test probabilities, and per-run plus mean ± standard deviation metrics. Freezing `verifier_s42` recovered θ 0.20, min_len 1, NMS any, λ 0.50, α 0.25 and dev Arg-C 49.158249. The saved rule equals the released JSON except `frozen_at`. The code explicitly preserves released metadata after checking the recomputed rule and dev mean. |
| C2 | `propose.py insample --seed 42` wrote 1,278 train windows and 3,897 candidates. With `clave.candidates.label` against gold train, 3,837 are positive: **98.46%**, about 1.46 percentage points above the plan's approximate 97%. The difference is small; the full candidate file and labeling were checked for window order and coverage. Seeds 13 and 101 are scheduled for Phase 2. |
| C3 | `Verifier(..., use_feats=False)` omits the feature module and uses a 3H head; the default feature-enabled state dictionary and seeded initialization match the released class in a controlled test. Both requested one-epoch smoke runs completed: `--no_feats` best dev Arg-C F1 18.55 and `--train_cands insample` 15.66. They used only 60 training windows and are code-path checks, not ablation results. Their logs record `use_feats` and `train_cands`. |
| C4 | `proposer_baseline.py eval` accepts split, τ, min_len, tag and seeds, emits P/R/F1, and freezes the fixed dev rule before the registered T-P1 test look. CLI and syntax checked; real dev/test argmax evaluation remains scheduled for Phase 7. No test parameter search was run. |
| C5 | `collect.py`, `tables.py`, `analysis.py`, `bootstrap.py` and archive collection are present. The collector uses `clave.evaluate.score`, writes P/R/F1 for four argument match modes plus trigger ROUGE-L, mean ± sample standard deviation, run count, rule and timestamps. Static table rendering and script compilation succeeded; full result files and T1–T8 await the later experimental phases. |

## Released-system regression

The Phase-1 test look was entered in `TEST_LOG.md` before scoring, with the frozen rule hash and timestamp. `verifier_s42` reused its released `test_probs.npy` and produced ROUGE-L P/R/F1 **85.13/76.78/77.82**, Arg-I IoU **70.18/52.53/60.09**, and Arg-C IoU **61.15/45.78/52.36**. `cmp` confirms `experiment/preds/test_verifier_s42.jsonl` is byte-identical to the released `preds/test_verifier_s42.jsonl` (SHA-256 `9ceb12e7f025ff25ea327417b59bdce760635c908caf8b3d58091afe7e7e6b5d`).

Four static unit tests passed, and `python3 -m compileall -q experiment` passed. Free disk was 28 GB and the shared GPU had about 45 GB free at the final check. No Phase-2 training or evaluation has started.

**STOP-1:** Phase 1 is complete. Wait for the user's next `tiếp` before running Phase 2.
