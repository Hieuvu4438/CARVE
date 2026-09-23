"""Stage 1 -> 2: candidate spans from the CARVE-simple proposer.

  eval   Candidates for dev and test from the proposer trained on the FULL train
         split (runs/proposer_s{seed}/best.pt). The verifier is validated on the
         dev candidates and tested once on the test candidates.

  oof    Out-of-fold candidates for the TRAIN split. The proposer memorises its
         own training windows, so candidates it produces on them contain almost
         no realistic errors (97.5 % correct, against ~38 % out of sample) and a
         verifier trained on them learns to keep everything. The train split is
         therefore cut into K = 5 folds; for each fold a proposer is trained on
         the other folds (epoch selected on the real dev split) and decodes the
         held-out fold. Every train window receives candidates from a model that
         never saw it -- the same relation dev/test windows have to the
         full-train proposer. Folds are window-level and stratified by event
         type. Fold checkpoints are deleted once their candidates are written.

Usage:
  python3 scripts/propose.py eval --seed 42
  python3 scripts/propose.py oof  --fold 0 --seed 42        # for fold in 0..4
"""

import argparse
import collections
import json
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from clave import candidates as C
from clave.data import AAO_LABEL2ID, AAO_TYPES, MERGED_LABEL2ID, load_split
from clave.decode import posteriors
from clave.paths import default_data_dir, repo_root

MODEL = "microsoft/deberta-v3-large"
K_FOLDS = 5
FOLD_SEED = 0
ROOT = repo_root()
ID2MERGED = {i: l for l, i in MERGED_LABEL2ID.items()}


def aao_onehot(p):
    """AAO / trigger spans come from the same merged tagger: keep its argmax
    decision mapped into the AAO inventory (non-AAO -> O), stored one-hot so the
    downstream argmax decode reproduces the proposer's own decode exactly."""
    out = np.zeros((p.shape[0], len(AAO_LABEL2ID)), dtype=np.float32)
    for i, lid in enumerate(p.argmax(-1)):
        lab = ID2MERGED[int(lid)]
        keep = lab != "O" and lab.split("-", 1)[1] in AAO_TYPES
        out[i, AAO_LABEL2ID[lab] if keep else AAO_LABEL2ID["O"]] = 1.0
    return out


def rows_for(windows, span_p, type_p, seed):
    return [{"sent_id": w.sent_id,
             "cands": C.extract(span_p[i], ID2MERGED, seed),
             "type_post": [round(float(x), 6) for x in type_p[i]],
             "aao_post": aao_onehot(span_p[i]).tolist()}
            for i, w in enumerate(windows)]


def write_candidates(ckpt, windows, out, seed):
    tok = AutoTokenizer.from_pretrained(MODEL)
    span_p, type_p = posteriors(str(ckpt), MODEL, windows, tok)
    C.save(str(out), rows_for(windows, span_p, type_p, seed))
    n = sum(len(r["cands"]) for r in C.load(str(out)))
    print(f"{len(windows)} windows, {n} candidates -> {out}")


def mode_eval(seed):
    ck = ROOT / "runs" / f"proposer_s{seed}" / "best.pt"
    assert ck.exists(), f"missing proposer checkpoint {ck} (train it with clave.train first)"
    for split in ["dev", "test"]:
        print(f"[eval] {split}: ", end="")
        write_candidates(ck, load_split(split), ROOT / "data" / "cands" / f"{split}_s{seed}.jsonl", seed)


def build_folds():
    """Write K fold directories (idempotent). Window-level, stratified by type."""
    src = default_data_dir()
    raw = [json.loads(l) for l in open(os.path.join(src, "train.oneie.json")) if l.strip()]
    dev_lines = open(os.path.join(src, "dev.oneie.json")).read()
    by_type = collections.defaultdict(list)
    for r in raw:
        by_type[r["event_mentions"][0]["event_type"]].append(r)
    rng = random.Random(FOLD_SEED)
    fold_of = {}
    for t in sorted(by_type):
        rows = by_type[t][:]
        rng.shuffle(rows)
        for j, r in enumerate(rows):
            fold_of[r["sent_id"]] = j % K_FOLDS
    for k in range(K_FOLDS):
        d = ROOT / "data" / "folds" / str(k)
        if (d / "heldout.oneie.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "train.oneie.json", "w") as f:
            for r in raw:
                if fold_of[r["sent_id"]] != k:
                    f.write(json.dumps(r) + "\n")
        with open(d / "heldout.oneie.json", "w") as f:
            for r in raw:
                if fold_of[r["sent_id"]] == k:
                    f.write(json.dumps(r) + "\n")
        (d / "dev.oneie.json").write_text(dev_lines)
    return fold_of


def mode_oof(fold, seed, epochs=None):
    build_folds()
    fold_dir = ROOT / "data" / "folds" / str(fold)
    out = ROOT / "data" / "cands" / f"oof_k{fold}_s{seed}.jsonl"
    if out.exists():
        print(f"[oof] {out} already exists, skipping")
        return
    from clave.train import load_config, run
    run_name = f"oof_k{fold}_s{seed}"
    cfg = load_config(ROOT / "configs" / "proposer.json")
    cfg.update({"run_name": run_name, "seed": seed, "data_dir": str(fold_dir), "eval_split": "dev",
                "out_dir": str(ROOT / "runs"), "save_model": True})
    if epochs is not None:             # smoke tests only
        cfg["epochs"] = epochs
    run(cfg)
    torch.cuda.empty_cache()
    ck = ROOT / "runs" / run_name / "best.pt"
    print(f"[oof] fold {fold} seed {seed}: ", end="")
    write_candidates(ck, load_split("heldout", str(fold_dir)), out, seed)
    ck.unlink()                       # keep disk usage flat; log.json is kept
    print(f"[oof] deleted {ck}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "oof"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "eval":
        mode_eval(a.seed)
    else:
        assert a.fold is not None, "--fold is required for oof"
        mode_oof(a.fold, a.seed, a.epochs)
