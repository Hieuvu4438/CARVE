#!/usr/bin/env bash
# Reproduce one phase of the three-seed experiment, or the original release.
# Run from the repository root after `bash scripts/setup_data.sh`.
#
# The released mode's stages 1-4 use train + dev only. Stage 5 touches test
# once with the checkpoints and rules already frozen. The three-seed mode
# follows experiment/EXPERIMENT_PLAN.md and stops after each phase.
#
# Cost on one 48 GB GPU: proposers ~10 min each (3 + 5 runs), verifier ~10-20 min.
set -euo pipefail
case "${1:-}" in
  3-seed)
    if [[ $# -ne 2 || ! "$2" =~ ^[1-7]$ ]]; then
      printf 'Usage: bash scripts/reproduce.sh 3-seed PHASE (1–7)\n' >&2
      exit 2
    fi
    exec bash experiment/scripts/reproduce.sh "$2"
    ;;
  p2-train|p2-test)
    if [[ $# -ne 1 ]]; then
      printf 'Usage: bash scripts/reproduce.sh p2-train|p2-test\n' >&2
      exit 2
    fi
    exec bash experiment/scripts/reproduce_p2.sh "${1#p2-}"
    ;;
  released)
    if [[ $# -ne 1 ]]; then
      printf 'Usage: bash scripts/reproduce.sh released\n' >&2
      exit 2
    fi
    ;;
  *)
    printf 'Usage: bash scripts/reproduce.sh {3-seed PHASE|p2-train|p2-test|released}\n' >&2
    exit 2
    ;;
esac
export PYTHONPATH=.
SEED=42                   # proposer and verifier seed of the released system
BASELINE_SEEDS=(42 13 101)

echo "=== Stage 0: correctness tests (data contract + official-scorer oracle) ==="
python3 tests/test_contract.py

echo "=== Stage 1: CARVE-simple proposers on the full train split ==="
for S in "${BASELINE_SEEDS[@]}"; do
  python3 -m clave.train -c configs/proposer.json --set run_name=proposer_s${S} seed=${S}
done

echo "=== Stage 2: out-of-fold candidates (5 folds) and dev/test candidates ==="
for K in 0 1 2 3 4; do
  python3 scripts/propose.py oof --fold ${K} --seed ${SEED}
done
python3 scripts/propose.py eval --seed ${SEED}

echo "=== Stage 3: verifier (epoch selected on dev) ==="
python3 scripts/train_verifier.py --prop_seed ${SEED} --seed ${SEED}

echo "=== Stage 4: freeze both decoding rules on DEV ONLY ==="
python3 scripts/proposer_baseline.py freeze --seeds "${BASELINE_SEEDS[@]}"
python3 scripts/freeze_and_test.py freeze --runs verifier_s${SEED}

echo "=== Stage 5: single frozen evaluation on TEST ==="
python3 scripts/proposer_baseline.py test --seeds "${BASELINE_SEEDS[@]}"
python3 scripts/freeze_and_test.py test --runs verifier_s${SEED}
python3 scripts/bootstrap_ci.py --sys "preds/test_verifier_s${SEED}.jsonl" --ref "preds/test_proposer_s${SEED}.jsonl"
