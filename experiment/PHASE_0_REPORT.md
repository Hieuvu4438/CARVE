# Phase 0 report — 2026-09-24

All commands used `SCIEVENT_ROOT=/home/haipd/SciEvent/third_party/SciEvent`
and `PYTHONPATH=.` from `/home/haipd/CLAVE`.

| Check | Result |
|---|---|
| `python3 tests/test_contract.py` | `ALL CONTRACT TESTS PASSED`; dev and test oracle Arg-C, Arg-I, ROUGE-L each **100.00** |
| Released verifier regression, seed 42, rule `assets/decoding_rule.json` | ROUGE-L P/R/F1 **85.13/76.78/77.82**; Arg-I **70.18/52.53/60.09**; Arg-C **61.15/45.78/52.36** |
| Byte comparison | `cmp` passed between released `preds/test_verifier_s42.jsonl` and the new isolated output. Both SHA-256: `9ceb12e7f025ff25ea327417b59bdce760635c908caf8b3d58091afe7e7e6b5d`. |
| GPU at preflight | 2365 / 49140 MiB used; three other Python processes visible. |
| Disk at preflight | 28 GB free on `/`, above the Phase 0 threshold of 20 GB. |

The regression ran the unchanged released module through
`experiment/scripts/regression_phase0.py`. Its output path was redirected to
`experiment/phase0_sandbox/`, because the user's instruction restricts writes
to `experiment/`. The released prediction, rule and checkpoint were only read.
`experiment/phase0_regression.log` contains the command output.

**STOP-0:** Phase 0 passed. Training and evaluation for later phases await the
user's instruction to continue, as required by §0.7 of `EXPERIMENT_PLAN.md`.
