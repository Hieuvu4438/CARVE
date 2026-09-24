#!/usr/bin/env bash
# Registered optional T-P2. The archive code is copied into experiment/ so
# source archive checkpoints, rules and predictions remain untouched.
set -euo pipefail

mode=${1:?usage: bash experiment/scripts/reproduce_p2.sh train|test}
repo=$(cd "$(dirname "$0")/../.." && pwd)
archive="$repo/experiment/archive_carve_full"
# The isolated copy of the archived original-CARVE code is not tracked in this
# repository (it lives in github.com/Hieuvu4438/SciEvent, method/CARVE-full).
# Rebuild it on demand; only the freeze/test split of freeze_and_test_simple.py
# differs from the archive (tracked here as p2_freeze_and_test_simple.py).
src=${CARVE_FULL_ARCHIVE:-/home/haipd/SciEvent/method/CARVE-full}
if [[ ! -f "$archive/carve/train.py" ]]; then
  mkdir -p "$archive"
  cp -r "$src/carve" "$src/configs" "$src/scripts" "$archive/"
  mkdir -p "$archive/assets" && cp -n "$src"/assets/*.json "$archive/assets/"
  cp "$repo/experiment/scripts/p2_freeze_and_test_simple.py" "$archive/scripts/freeze_and_test_simple.py"
fi
test -f "$archive/carve/train.py"
export SCIEVENT_ROOT=${SCIEVENT_ROOT:-/home/haipd/SciEvent/third_party/SciEvent}
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$archive"

case "$mode" in
  train)
    for variant in abl_no_type_cond abl_single_head; do
      for seed in 42 13 101; do
        run="${variant}_s${seed}"
        if [[ -e "runs/$run" ]]; then
          printf 'Run already exists: %s\n' "$run" >&2
          exit 1
        fi
        free_gb=$(df -BG / | awk 'NR == 2 {gsub(/G/, "", $4); print $4}')
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if (( free_gb < 5 || free_mib < 24000 )); then
          printf 'Insufficient disk or GPU memory for %s\n' "$run" >&2
          exit 1
        fi
        python3 -m carve.train -c "configs/$variant.json" \
          --set "run_name=$run" "seed=$seed" \
          > "$repo/experiment/logs/p2_${run}.log" 2>&1
      done
    done
    python3 "$repo/experiment/scripts/log_p2_runs.py"
    ;;
  test)
    for variant in abl_no_type_cond abl_single_head; do
      for seed in 42 13 101; do
        test -f "runs/${variant}_s${seed}/best.pt"
        test -f "runs/${variant}_s${seed}/log.json"
      done
    done
    for target in assets/rules_abl_no_type_cond.json \
                  assets/decoding_rules.abl_single_head.json \
                  preds/test_abl_no_type_cond_s{42,13,101}.jsonl \
                  preds/test_abl_single_head_s{42,13,101}.jsonl; do
      if [[ -e "$target" ]]; then
        printf 'T-P2 test artifact already exists: %s\n' "$target" >&2
        exit 1
      fi
    done
    if grep -qE '\| T-P2[ab] \|' "$repo/experiment/TEST_LOG.md"; then
      printf 'T-P2 test look was already registered. Refusing to overwrite rules.\n' >&2
      exit 1
    fi
    sha256sum -c "$repo/experiment/results/archive_frozen_rules.sha256"
    python3 - <<'PY'
import torch
for seed in (42, 13, 101):
    state = torch.load(f"runs/abl_no_type_cond_s{seed}/best.pt",
                       map_location="cpu", mmap=True, weights_only=True)
    if torch.count_nonzero(state["type_emb.weight"]).item():
        raise ValueError(f"no-type checkpoint seed {seed} has a nonzero type embedding")
    del state
print("no-type checkpoints have zero type embeddings; archive decoder is equivalent")
PY

    variant=abl_no_type_cond
    # the archive decoder writes metrics.test.json next to --out without creating the directory
    mkdir -p preds
    python3 scripts/freeze_rules.py "$variant" "" rules_abl_no_type_cond.json \
      > "$repo/experiment/logs/p2_freeze_no_type.log" 2>&1
    python3 "$repo/experiment/scripts/p2_protocol.py" freeze "$variant"
    python3 "$repo/experiment/scripts/p2_protocol.py" register "$variant"
    for seed in 42 13 101; do
      python3 -m carve.decode --ckpt "runs/${variant}_s${seed}/best.pt" --split test \
        --rules assets/rules_abl_no_type_cond.json \
        --out "preds/test_${variant}_s${seed}.jsonl" \
        > "$repo/experiment/logs/p2_test_${variant}_s${seed}.log" 2>&1
      mv preds/metrics.test.json "preds/metrics.test_${variant}_s${seed}.json"
    done
    cd "$repo"
    python3 experiment/scripts/collect.py --id "$variant" --split test \
      --pred "experiment/archive_carve_full/preds/test_${variant}_s*.jsonl" \
      --rule experiment/archive_carve_full/assets/rules_abl_no_type_cond.json
    python3 experiment/scripts/p2_protocol.py finish "$variant"

    cd "$archive"
    variant=abl_single_head
    python3 scripts/freeze_and_test_simple.py "$variant" freeze \
      > "$repo/experiment/logs/p2_freeze_single_head.log" 2>&1
    python3 "$repo/experiment/scripts/p2_protocol.py" freeze "$variant"
    python3 "$repo/experiment/scripts/p2_protocol.py" register "$variant"
    python3 scripts/freeze_and_test_simple.py "$variant" test \
      > "$repo/experiment/logs/p2_test_single_head.log" 2>&1
    cd "$repo"
    python3 experiment/scripts/collect.py --id "$variant" --split test \
      --pred "experiment/archive_carve_full/preds/test_${variant}_s*.jsonl" \
      --rule experiment/archive_carve_full/assets/decoding_rules.abl_single_head.json
    python3 experiment/scripts/p2_protocol.py finish "$variant"
    python3 experiment/scripts/check_test_log.py
    python3 experiment/scripts/check_test_log.py --ids T-P2a T-P2b
    python3 experiment/scripts/tables.py
    sha256sum -c experiment/results/archive_frozen_rules.sha256
    ;;
  *)
    printf 'usage: bash experiment/scripts/reproduce_p2.sh train|test\n' >&2
    exit 2
    ;;
esac
