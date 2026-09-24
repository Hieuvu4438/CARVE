"""Stage 2: train the HONE verifier on out-of-fold candidates; select on dev.

Train candidates: the out-of-fold proposals for every train window
(data/cands/oof_k*_s{prop_seed}.jsonl). Dev candidates: proposals of the
full-train proposer with the same seed, so both sides see the same candidate
distribution. After each epoch the dev candidates are decoded under a dev-tuned
verifier-only rule (select.tune) and scored with the official scorer; the best
epoch is kept (runs/<name>/best.pt, dev_probs.npy, log.json).

The defaults are the released recipe: 5 epochs, batch 8 x 2 accumulation steps,
AdamW (encoder 1e-5, head 1e-4, weight decay 0.01, 10 % warmup), dropout 0.1,
gradient checkpointing, bf16 autocast.

Usage:
  python3 scripts/train_verifier.py --prop_seed 42 --seed 42          # -> runs/verifier_s42
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from clave import candidates as C
from clave.data import load_split
from clave.evaluate import score
from clave.paths import split_path
from clave.select import records, tune
from clave.verifier import MARK_CLOSE, MARK_OPEN, Verifier, VerifierDataset, build_examples, collate
from common import EXPERIMENT_ROOT, candidate_path, run_dir

ROOT = EXPERIMENT_ROOT
K_FOLDS = 5


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_train_rows(prop_seed, train_cands="oof"):
    if train_cands == "insample":
        path = candidate_path(f"insample_s{prop_seed}.jsonl")
        return C.pool_rows([C.load(str(path))])
    rows = []
    for k in range(K_FOLDS):
        rows += C.load(str(candidate_path(f"oof_k{k}_s{prop_seed}.jsonl")))
    return C.pool_rows([rows])


def load_eval_rows(split, prop_seed):
    return C.pool_rows([C.load(str(candidate_path(f"{split}_s{prop_seed}.jsonl")))])


@torch.no_grad()
def predict(model, dl, n, device, amp):
    model.eval()
    out = np.zeros((n, len(C.LABELS)), dtype=np.float32)
    for b in dl:
        with torch.autocast("cuda", dtype=amp):
            lg = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                       b["open"].to(device), b["close"].to(device), b["feats"].to(device))
        out[b["idx"].numpy()] = torch.softmax(lg.float(), -1).cpu().numpy()
    return out


def regroup(examples, probs, rows):
    """Flat per-candidate probabilities -> per-window arrays aligned with rows."""
    pos = {(x["sent_id"], x["cand_idx"]): i for i, x in enumerate(examples)}
    per_w = []
    for r in rows:
        idx = [pos[(r["sent_id"], j)] for j in range(len(r["cands"]))]
        per_w.append(probs[idx] if idx else np.zeros((0, len(C.LABELS)), dtype=np.float32))
    return per_w


def main(a):
    set_seed(a.seed)
    device = "cuda"
    amp = torch.bfloat16
    output_dir = run_dir(a.name, write=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {output_dir}")

    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MARK_OPEN, MARK_CLOSE]})

    train_w = {w.sent_id: w for w in load_split("train")}
    dev_w = {w.sent_id: w for w in load_split("dev")}

    if a.smoke:   # code-path test only: train on a few windows, numbers meaningless
        if a.train_cands == "insample":
            tr_rows = load_train_rows(a.prop_seed, "insample")[:60]
        else:
            tr_rows = load_eval_rows("dev", a.prop_seed)[:60]
            train_w = dev_w
    else:
        tr_rows = load_train_rows(a.prop_seed, a.train_cands)
    for r in tr_rows:
        C.label(train_w[r["sent_id"]], r["cands"])
    dv_rows = load_eval_rows("dev", a.prop_seed)
    for r in dv_rows:
        C.label(dev_w[r["sent_id"]], r["cands"])
    dv_windows = [dev_w[r["sent_id"]] for r in dv_rows]

    tr_ex = build_examples(train_w, tr_rows)
    dv_ex = build_examples(dev_w, dv_rows)
    pos = sum(x["label"] != 0 for x in tr_ex) / len(tr_ex)
    print(f"train candidates {len(tr_ex)} (positive {pos:.1%}) | dev candidates {len(dv_ex)}", flush=True)

    tr_dl = DataLoader(VerifierDataset(tr_ex, tok), batch_size=a.batch_size, shuffle=True,
                       collate_fn=lambda b: collate(b, tok.pad_token_id), num_workers=2)
    dv_dl = DataLoader(VerifierDataset(dv_ex, tok), batch_size=a.eval_batch_size, shuffle=False,
                       collate_fn=lambda b: collate(b, tok.pad_token_id), num_workers=2)

    model = Verifier(a.model_name, len(tok), dropout=a.dropout,
                     use_feats=a.use_feats).to(device)
    if a.grad_ckpt:   # trades compute for activation memory
        model.encoder.gradient_checkpointing_enable()
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr_encoder}, {"params": head, "lr": a.lr_head}],
                            weight_decay=0.01)
    steps = (len(tr_dl) + a.accum - 1) // a.accum * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    ce = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    gold = split_path("dev")
    best, log, t0 = None, [], time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for step, b in enumerate(tr_dl):
            with torch.autocast("cuda", dtype=amp):
                lg = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                           b["open"].to(device), b["close"].to(device), b["feats"].to(device))
                loss = ce(lg.float(), b["label"].to(device))
            (loss / a.accum).backward()
            if (step + 1) % a.accum == 0 or step + 1 == len(tr_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item()

        probs = predict(model, dv_dl, len(dv_ex), device, amp)
        per_w = regroup(dv_ex, probs, dv_rows)
        cand_acc = float(np.mean(probs.argmax(-1) == np.array([x["label"] for x in dv_ex])))
        rule = tune(dv_windows, dv_rows, per_w, gold)
        full = score(records(dv_windows, dv_rows, per_w, rule["theta"], rule["min_len"], rule["nms"]), gold)
        row = {"epoch": ep, "loss": tot / len(tr_dl), "cand_acc": cand_acc * 100,
               "arg_c_iou_f1": full["arg_c_iou"]["f1"], "arg_c_p": full["arg_c_iou"]["p"],
               "arg_c_r": full["arg_c_iou"]["r"], "arg_i_iou_f1": full["arg_i_iou"]["f1"],
               "rougeL": full["trigger_rougeL"]["f1"], "rule": rule, "secs": round(time.time() - t0)}
        log.append(row)
        print(f"ep{ep:02d} loss {row['loss']:.3f} | cand-acc {row['cand_acc']:.1f} | "
              f"ArgC-IoU {row['arg_c_iou_f1']:.2f} (P{row['arg_c_p']:.1f}/R{row['arg_c_r']:.1f}) | "
              f"ArgI {row['arg_i_iou_f1']:.2f} | rule th={rule['theta']} ml={rule['min_len']} "
              f"nms={rule['nms']} | {row['secs']}s", flush=True)
        if best is None or row["arg_c_iou_f1"] > best["arg_c_iou_f1"]:
            best = dict(row)
            np.save(output_dir / "dev_probs.npy", probs)
            if a.save_model:
                torch.save(model.state_dict(), output_dir / "best.pt")

    json.dump({"args": vars(a), "log": log, "best": best}, open(output_dir / "log.json", "w"), indent=2)
    print("BEST", json.dumps(best, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="run directory under runs/ (default verifier_s<seed>)")
    ap.add_argument("--prop_seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_name", default="microsoft/deberta-v3-large")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--lr_encoder", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--save_model", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no_feats", action="store_true")
    ap.add_argument("--train_cands", choices=["oof", "insample"], default="oof")
    a = ap.parse_args()
    a.name = a.name or f"verifier_exp_v{a.seed}_p{a.prop_seed}"
    a.use_feats = not a.no_feats
    main(a)
