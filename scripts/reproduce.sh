#!/usr/bin/env bash
# End-to-end reproduction of CARVE-simple + HONE. Run from the repository root
# after `bash scripts/setup_data.sh`.
#
# Protocol: stages 1-4 use train + dev only. Stage 5 touches the test split once,
# with every checkpoint and decoding rule already frozen.
#
# Cost on one 48 GB GPU: proposers ~10 min each (3 + 5 runs), verifier ~10-20 min.
set -euo pipefail
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
