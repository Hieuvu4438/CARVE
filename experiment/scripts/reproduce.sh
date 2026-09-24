#!/usr/bin/env bash
# Run exactly one authorized phase. Invoke again only after its STOP report and
# the user's instruction to continue. All generated files stay in experiment/.
set -euo pipefail

phase=${1:?usage: bash experiment/scripts/reproduce.sh PHASE_NUMBER}
cd "$(dirname "$0")/../.."
export SCIEVENT_ROOT=${SCIEVENT_ROOT:-/home/haipd/SciEvent/third_party/SciEvent}
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

require_disk() {
  local free_gb
  free_gb=$(df -BG / | awk 'NR == 2 {gsub(/G/, "", $4); print $4}')
  if (( free_gb < 5 )); then
    printf 'Less than 5 GB free; training stopped.\n' >&2
    exit 1
  fi
}

require_gpu_memory() {
  local needed_gb=$1 free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if (( free_mib < needed_gb * 1024 )); then
    printf 'Less than %s GB GPU memory free; training stopped.\n' "$needed_gb" >&2
    exit 1
  fi
}

run_logged() {
  local phase_id=$1 run_name=$2 seed=$3 gpu_gb=$4
  shift 4
  require_disk
  require_gpu_memory "$gpu_gb"
  local started finished stdout_log
  started=$(date -Iseconds)
  stdout_log="experiment/logs/${run_name}.log"
  python3 experiment/scripts/log_run.py start --phase "$phase_id" --run "$run_name" \
    --seed "$seed" --command "$*" --started "$started" --stdout-log "$stdout_log"
  if "$@" > "$stdout_log" 2>&1; then
    finished=$(date -Iseconds)
    python3 experiment/scripts/log_run.py finish --phase "$phase_id" --run "$run_name" \
      --seed "$seed" --command "$*" --started "$started" --finished "$finished" \
      --stdout-log "$stdout_log"
  else
    printf 'Run %s failed. See %s and RUN_LOG.md.\n' "$run_name" "$stdout_log" >&2
    exit 1
  fi
}

cleanup_ablation() {
  local variant=$1 seed run
  for seed in 42 13 101; do
    run="verifier_${variant}_v${seed}_p${seed}"
    for artifact in log.json dev_probs.npy test_probs.npy; do
      if [[ ! -f "experiment/runs/${run}/${artifact}" ]]; then
        printf 'Cannot clean %s: missing %s.\n' "$run" "$artifact" >&2
        exit 1
      fi
    done
  done
  local result=a1
  if [[ "$variant" == nofeat ]]; then result=a2; fi
  if [[ ! -f "experiment/results/${result}.json" ]]; then
    printf 'Cannot clean %s: result JSON is missing.\n' "$variant" >&2
    exit 1
  fi
  for seed in 42 13 101; do
    run="verifier_${variant}_v${seed}_p${seed}"
    rm -f "experiment/runs/${run}/best.pt"
  done
}

main_runs=(verifier_v42_p42 verifier_v13_p13 verifier_v101_p101)

case "$phase" in
  1)
    python3 experiment/scripts/freeze_and_test.py freeze --runs verifier_s42 \
      > experiment/logs/phase1_freeze_s42.log
    python3 - <<'PY'
import json
from pathlib import Path
old = json.loads(Path('assets/decoding_rule.json').read_text())
new = json.loads(Path('experiment/assets/decoding_rule.json').read_text())
old.pop('frozen_at')
new.pop('frozen_at')
assert new == old, (new, old)
print('seed-42 frozen rule matches the released rule except frozen_at')
PY
    python3 experiment/scripts/freeze_and_test.py test --runs verifier_s42 \
      > experiment/logs/phase1_test_s42.log
    cmp preds/test_verifier_s42.jsonl experiment/preds/test_verifier_s42.jsonl
    python3 experiment/scripts/propose.py insample --seed 42 \
      > experiment/logs/phase1_insample_s42.log
    run_logged Phase1 smoke_nofeat_s42 42 10 python3 experiment/scripts/train_verifier.py \
      --smoke --epochs 1 --prop_seed 42 --seed 42 --name smoke_nofeat_s42 \
      --no_feats --save_model 0
    run_logged Phase1 smoke_insample_s42 42 10 python3 experiment/scripts/train_verifier.py \
      --smoke --epochs 1 --prop_seed 42 --seed 42 --name smoke_insample_s42 \
      --train_cands insample --save_model 0
    python3 -m unittest discover -s experiment/tests -p 'test_*.py' -v \
      > experiment/logs/phase1_static_tests.log 2>&1
    ;;
  2)
    for seed in 13 101; do
      python3 experiment/scripts/propose.py eval --seed "$seed" \
        > "experiment/logs/propose_eval_s${seed}.log" 2>&1
    done
    for seed in 13 101; do
      for fold in 0 1 2 3 4; do
        run="oof_k${fold}_s${seed}"
        if [[ -f "experiment/data/cands/${run}.jsonl" ]]; then
          printf '%s candidate output already exists; inspect its log before continuing.\n' "$run"
          continue
        fi
        run_logged Phase2 "$run" "$seed" 22 python3 experiment/scripts/propose.py oof \
          --fold "$fold" --seed "$seed"
      done
    done
    for seed in 42 13 101; do
      python3 experiment/scripts/propose.py insample --seed "$seed" \
        > "experiment/logs/propose_insample_s${seed}.log" 2>&1
    done
    python3 experiment/scripts/analysis.py candidate-stats --out candidate_stats_phase2.json
    ;;
  3)
    for seed in 13 101; do
      run="verifier_v${seed}_p${seed}"
      run_logged Phase3 "$run" "$seed" 10 python3 experiment/scripts/train_verifier.py \
        --prop_seed "$seed" --seed "$seed" --name "$run"
    done
    ;;
  4)
    rule=decoding_rule_3seed.json
    python3 experiment/scripts/freeze_and_test.py freeze --runs "${main_runs[@]}" \
      --rule_name "$rule" > experiment/logs/freeze_main.log
    python3 experiment/scripts/freeze_and_test.py test --runs "${main_runs[@]}" \
      --rule_name "$rule" > experiment/logs/test_main.log
    python3 experiment/scripts/collect.py --id main --split test \
      --pred 'experiment/preds/test_verifier_v*_p*__decoding_rule_3seed.jsonl' \
      --rule "experiment/assets/$rule"
    python3 experiment/scripts/collect.py --id carve_simple --split test \
      --pred 'preds/test_proposer_s*.jsonl' --rule assets/proposer_rule.json --archive
    python3 experiment/scripts/collect.py --id carve_orig --split test \
      --pred '../SciEvent/method/SciEvent-Next/artifacts/preds/test_preds_s*.jsonl' \
      --rule '../SciEvent/method/SciEvent-Next/artifacts/decoding_rules.json' --archive
    python3 experiment/scripts/collect.py --id clave_s42 --split test \
      --pred preds/test_verifier_s42.jsonl --rule assets/decoding_rule.json --archive
    python3 experiment/scripts/bootstrap.py --mode main
    python3 experiment/scripts/tables.py
    ;;
  5)
    for seed in 42 13 101; do
      for variant in insample nofeat; do
        run="verifier_${variant}_v${seed}_p${seed}"
        args=()
        if [[ "$variant" == insample ]]; then
          args=(--train_cands insample)
        else
          args=(--no_feats)
        fi
        run_logged Phase5 "$run" "$seed" 10 python3 experiment/scripts/train_verifier.py \
          --prop_seed "$seed" --seed "$seed" --name "$run" "${args[@]}"
      done
    done
    for variant in insample nofeat; do
      runs=()
      for seed in 42 13 101; do runs+=("verifier_${variant}_v${seed}_p${seed}"); done
      rule="rule_abl_${variant}.json"
      result=a1
      if [[ "$variant" == nofeat ]]; then result=a2; fi
      python3 experiment/scripts/freeze_and_test.py freeze --runs "${runs[@]}" \
        --rule_name "$rule" > "experiment/logs/freeze_${result}.log"
      python3 experiment/scripts/freeze_and_test.py test --runs "${runs[@]}" \
        --rule_name "$rule" > "experiment/logs/test_${result}.log"
      python3 experiment/scripts/collect.py --id "$result" --split test \
        --pred "experiment/preds/test_verifier_${variant}_v*_p*__${rule%.json}.jsonl" \
        --rule "experiment/assets/$rule"
      python3 experiment/scripts/bootstrap.py --mode ablation --id "$result" \
        --sys "experiment/preds/test_verifier_${variant}_v*_p*__${rule%.json}.jsonl"
      cleanup_ablation "$variant"
    done
    python3 experiment/scripts/tables.py
    ;;
  6)
    # Names and fixed grid dimensions are pre-registered in freeze_and_test.py.
    variants=(
      'a3 rule_abl_role_verifier.json role_alpha=1.0'
      'a4 rule_abl_role_proposer.json role_alpha=0.0'
      'a5 rule_abl_no_type.json type_lambda=0'
      'a6 rule_abl_pure_verifier.json role_alpha=1.0,type_lambda=0'
      'a7_iou rule_abl_nms_iou.json nms=iou'
      'a7_wis rule_abl_nms_wis.json nms=wis'
    )
    for spec in "${variants[@]}"; do
      read -r result rule fixes <<< "$spec"
      fix_args=()
      IFS=, read -ra fixed <<< "$fixes"
      for item in "${fixed[@]}"; do fix_args+=(--fix "$item"); done
      python3 experiment/scripts/freeze_and_test.py freeze --runs "${main_runs[@]}" \
        --rule_name "$rule" "${fix_args[@]}" > "experiment/logs/freeze_${result}.log"
      python3 experiment/scripts/freeze_and_test.py test --runs "${main_runs[@]}" \
        --rule_name "$rule" > "experiment/logs/test_${result}.log"
      python3 experiment/scripts/collect.py --id "$result" --split test \
        --pred "experiment/preds/test_verifier_v*_p*__${rule%.json}.jsonl" \
        --rule "experiment/assets/$rule"
      python3 experiment/scripts/bootstrap.py --mode ablation --id "$result" \
        --sys "experiment/preds/test_verifier_v*_p*__${rule%.json}.jsonl"
    done
    python3 experiment/scripts/tables.py
    ;;
  7)
    python3 experiment/scripts/proposer_baseline.py eval --split dev --tau 0 \
      --min_len 1 --tag argmax > experiment/logs/proposer_argmax_dev.log
    python3 experiment/scripts/proposer_baseline.py eval --split test --tau 0 \
      --min_len 1 --tag argmax > experiment/logs/proposer_argmax_test.log
    python3 experiment/scripts/collect.py --id proposer_argmax --split test \
      --pred 'experiment/preds/test_proposer_argmax_s*.jsonl' \
      --rule experiment/assets/proposer_argmax_rule.json
    if [[ "${CLAVE_RUN_P2:-0}" == 1 ]]; then
      bash experiment/scripts/reproduce_p2.sh train
      bash experiment/scripts/reproduce_p2.sh test
    fi
    python3 experiment/scripts/check_test_log.py
    python3 experiment/scripts/analysis.py oracle-dev
    python3 experiment/scripts/analysis.py candidate-stats
    python3 experiment/scripts/analysis.py test \
      --pred 'experiment/preds/test_verifier_v*_p*__decoding_rule_3seed.jsonl' \
      --baseline 'preds/test_proposer_s*.jsonl'
    python3 experiment/scripts/tables.py --require-complete
    ;;
  *)
    printf 'Unknown or optional phase %s. Run one of 1–7 only.\n' "$phase" >&2
    exit 2
    ;;
esac
