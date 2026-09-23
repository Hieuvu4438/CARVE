"""Train the CARVE-simple proposer (stage 1).

    python3 -m carve.train -c configs/proposer.json --set run_name=proposer_s42 seed=42

Checkpoint selection is on dev, under the same calibrated decoding the proposer
is evaluated with (a confidence threshold swept per epoch), not under argmax.
`data_dir` / `eval_split` let `scripts/propose.py oof` train fold proposers.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from carve.data import AAO_TYPES, EVENT_TYPE2ID, EVENT_TYPES, MERGED_LABEL2ID, ROLE_TYPES, bio_to_spans, load_split, spans_to_bio
from carve.evaluate import score, to_oneie_record, write_jsonl
from carve.model import CarveTagger
from carve.paths import repo_root, split_path

ID2MERGED = {i: l for l, i in MERGED_LABEL2ID.items()}


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class WindowDataset(Dataset):
    def __init__(self, windows, tokenizer, max_len=640):
        self.w = windows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        w = self.w[i]
        enc = self.tok(w.tokens, is_split_into_words=True, truncation=True, max_length=self.max_len,
                       return_tensors=None)
        first_sub = {}
        for pos, wid in enumerate(enc.word_ids()):
            if wid is not None and wid not in first_sub:
                first_sub[wid] = pos
        n_words = len(w.tokens)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "word_index": [first_sub.get(j, 0) for j in range(n_words)],
            "word_valid": [1 if j in first_sub else 0 for j in range(n_words)],
            "merged_y": spans_to_bio(list(w.role_spans) + list(w.aao_spans), n_words, MERGED_LABEL2ID),
            "type_y": EVENT_TYPE2ID[w.event_type],
            "idx": i,
        }


def collate(batch, pad_id):
    B = len(batch)
    L = max(len(b["input_ids"]) for b in batch)
    W = max(len(b["word_index"]) for b in batch)
    out = {
        "input_ids": torch.full((B, L), pad_id, dtype=torch.long),
        "attention_mask": torch.zeros((B, L), dtype=torch.long),
        "word_index": torch.zeros((B, W), dtype=torch.long),
        "word_mask": torch.zeros((B, W), dtype=torch.float),
        "merged_y": torch.full((B, W), -100, dtype=torch.long),
        "type_y": torch.zeros(B, dtype=torch.long),
        "idx": torch.zeros(B, dtype=torch.long),
    }
    for i, b in enumerate(batch):
        l, w = len(b["input_ids"]), len(b["word_index"])
        out["input_ids"][i, :l] = torch.tensor(b["input_ids"])
        out["attention_mask"][i, :l] = torch.tensor(b["attention_mask"])
        out["word_index"][i, :w] = torch.tensor(b["word_index"])
        out["word_mask"][i, :w] = torch.tensor(b["word_valid"], dtype=torch.float)
        y = torch.tensor(b["merged_y"])
        y[~torch.tensor(b["word_valid"], dtype=torch.bool)] = -100
        out["merged_y"][i, :w] = y
        out["type_y"][i] = b["type_y"]
        out["idx"][i] = b["idx"]
    return out


def param_groups(model, cfg):
    """Encoder at lr_encoder, task heads at lr_head (AdamW is per-parameter, so
    the grouping itself does not affect the update)."""
    enc = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    return [{"params": enc, "lr": cfg["lr_encoder"]}, {"params": head, "lr": cfg["lr_head"]}]


def split_spans(span_ids):
    """Merged-inventory BIO ids -> (scored role spans, AAO spans)."""
    merged = bio_to_spans(span_ids, ID2MERGED)
    return [x for x in merged if x[2] in ROLE_TYPES], [x for x in merged if x[2] in AAO_TYPES]


@torch.no_grad()
def predict(model, loader, windows, device, amp_dtype):
    """Dev decoding. Returns (argmax records, per-window candidates with span
    confidences for threshold sweeps, event-type accuracy)."""
    model.eval()
    recs_argmax, cands = [], []
    type_correct = 0
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=amp_dtype):
            tl, sl = model(b["input_ids"], b["attention_mask"], b["word_index"], b["word_mask"])
        tp = tl.float().argmax(-1).cpu()
        prob = torch.softmax(sl.float(), -1).cpu().numpy()
        for j, idx in enumerate(batch["idx"].tolist()):
            w = windows[idx]
            n = len(w.tokens)
            nv = int(batch["word_mask"][j, :n].sum())
            ids = prob[j, :nv].argmax(-1).tolist()
            role_spans, aao_spans = split_spans(ids)
            etype = EVENT_TYPES[int(tp[j])]
            type_correct += int(etype == w.event_type)
            trig = next(((s, e) for s, e, t in aao_spans if t == "Action"), (0, min(1, n)))
            aao_out = [(s, e, t) for s, e, t in aao_spans if t != "Action"]
            recs_argmax.append(to_oneie_record(w.sent_id, w.tokens, etype, role_spans, aao_out, trig))
            conf = [(s, e, t, float(np.mean([prob[j, i, ids[i]] for i in range(s, e)]))) for s, e, t in role_spans]
            cands.append({"conf": conf, "etype": etype, "aao": aao_out, "trig": trig,
                          "sent_id": w.sent_id, "tokens": w.tokens, "n": n})
    return recs_argmax, cands, type_correct / max(1, len(windows))


def records_at(cands, tau, min_len):
    from carve.decode import apply_decoding_rules
    return [to_oneie_record(c["sent_id"], c["tokens"], c["etype"],
                            apply_decoding_rules(c["conf"], tau, min_len), c["aao"], c["trig"])
            for c in cands]


def run(cfg):
    set_seed(cfg["seed"])
    device = "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    train_w = load_split("train", cfg.get("data_dir"))
    dev_w = load_split(cfg.get("eval_split", "dev"), cfg.get("data_dir"))
    gold_path = split_path(cfg.get("eval_split", "dev"), cfg.get("data_dir"))

    coll = lambda b: collate(b, tok.pad_token_id)  # noqa: E731
    tr_dl = DataLoader(WindowDataset(train_w, tok, cfg["max_len"]), batch_size=cfg["batch_size"], shuffle=True,
                       collate_fn=coll, num_workers=2, drop_last=False)
    dv_dl = DataLoader(WindowDataset(dev_w, tok, cfg["max_len"]), batch_size=cfg["eval_batch_size"], shuffle=False,
                       collate_fn=coll, num_workers=2)

    model = CarveTagger(cfg["model_name"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg), weight_decay=cfg["weight_decay"])
    steps = len(tr_dl) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps), steps)
    ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=cfg.get("label_smoothing", 0.0))
    ce_type = nn.CrossEntropyLoss()

    run_dir = os.path.join(cfg["out_dir"], cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    best, log, t0 = {"arg_c_iou_f1": -1.0}, [], time.time()

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tot = 0.0
        for batch in tr_dl:
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype):
                tl, sl = model(b["input_ids"], b["attention_mask"], b["word_index"], b["word_mask"], training=True)
                loss = (cfg["w_role"] * ce(sl.reshape(-1, sl.size(-1)).float(), b["merged_y"].reshape(-1))
                        + cfg["w_type"] * ce_type(tl.float(), b["type_y"]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            tot += loss.item()

        recs_a, cands, type_acc = predict(model, dv_dl, dev_w, device, amp_dtype)
        m_argmax = score(recs_a, gold_path)
        m, recs, sel_tau = None, None, None
        for t in cfg["dev_tau_grid"]:
            r = records_at(cands, t, cfg["dev_min_len"])
            mm = score(r, gold_path)
            if m is None or mm["arg_c_iou"]["f1"] > m["arg_c_iou"]["f1"]:
                m, recs, sel_tau = mm, r, t
        row = {
            "epoch": ep, "loss": tot / len(tr_dl), "type_acc": type_acc * 100,
            "arg_c_iou_f1": m["arg_c_iou"]["f1"], "argmax_arg_c_iou_f1": m_argmax["arg_c_iou"]["f1"],
            "sel_tau": sel_tau, "arg_i_iou_f1": m["arg_i_iou"]["f1"],
            "arg_c_iou_p": m["arg_c_iou"]["p"], "arg_c_iou_r": m["arg_c_iou"]["r"],
            "arg_c_em_f1": m["arg_c_exact"]["f1"], "arg_i_em_f1": m["arg_i_exact"]["f1"],
            "trigger_rougeL_f1": m["trigger_rougeL"]["f1"], "secs": round(time.time() - t0),
        }
        log.append(row)
        print(f"ep{ep:02d} loss {row['loss']:.3f} | type {row['type_acc']:.1f} | "
              f"ArgC-IoU {row['arg_c_iou_f1']:.2f} (P{row['arg_c_iou_p']:.1f}/R{row['arg_c_iou_r']:.1f}) | "
              f"ArgI-IoU {row['arg_i_iou_f1']:.2f} | raw {row['argmax_arg_c_iou_f1']:.2f} | t{sel_tau} | "
              f"RgL {row['trigger_rougeL_f1']:.2f} | {row['secs']}s", flush=True)
        if row["arg_c_iou_f1"] > best["arg_c_iou_f1"]:
            best = dict(row)
            best["metrics"] = m
            write_jsonl(recs, os.path.join(run_dir, "dev_preds.jsonl"))
            if cfg.get("save_model", True):
                torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))

    # log.json is written last: its presence marks best.pt as complete
    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump({"config": cfg, "log": log, "best": best}, f, indent=2)
    print("\nBEST:", json.dumps({k: v for k, v in best.items() if k != "metrics"}, indent=2))
    return best


DEFAULTS = {
    "model_name": "microsoft/deberta-v3-large",
    "run_name": "proposer",
    "out_dir": str(repo_root() / "runs"),
    "seed": 42,
    "epochs": 30,
    "batch_size": 8,
    "eval_batch_size": 16,
    "max_len": 640,
    "lr_encoder": 1e-5,
    "lr_head": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "dropout": 0.1,
    "max_grad_norm": 1.0,
    "w_role": 1.0,
    "w_type": 0.5,
    "label_smoothing": 0.0,
    "dev_tau_grid": [0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95],
    "dev_min_len": 3,
    "eval_split": "dev",
    "save_model": True,
    "data_dir": None,
}


def load_config(path=None, overrides=()):
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(json.load(open(path)))
    for kv in overrides:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except ValueError:
            pass
        cfg[k] = v
    cfg["out_dir"] = os.path.abspath(cfg["out_dir"])
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    run(load_config(a.config, a.set))
