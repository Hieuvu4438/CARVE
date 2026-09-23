"""Stage 3: freeze the decoding rule on dev, then (separately) evaluate test once.

  freeze  Grid over (theta, min_len, collision rule, role alpha, type lambda) on
          the saved dev probabilities of the given verifier run(s). With several
          runs the criterion is the MEAN dev Arg-C IoU, so the rule is not fitted
          to one seed. Writes assets/decoding_rule.json with a timestamp.

  test    Loads the frozen rule, runs each verifier on the test candidates,
          decodes with that one rule, scores with the official scorer and writes
          preds/test_<run>.jsonl. Run only after `freeze`.

Usage:
  python3 scripts/freeze_and_test.py freeze --runs verifier_s42
  python3 scripts/freeze_and_test.py test   --runs verifier_s42
"""

import argparse
import json
import statistics as st
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from carve import candidates as C
from carve.data import load_split
from carve.evaluate import score, write_jsonl
from carve.paths import repo_root, split_path
from carve.select import ALPHAS, MIN_LENS, NMS_MODES, THETAS, mix_roles, records, role_given_type
from carve.verifier import MARK_CLOSE, MARK_OPEN, Verifier, VerifierDataset, build_examples, collate

ROOT = repo_root()
MODEL = "microsoft/deberta-v3-large"
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


def prop_seed_of(runs):
    def seed(args):   # the reported run's log predates the rename prop_seeds -> prop_seed
        return args["prop_seed"] if "prop_seed" in args else args["prop_seeds"][0]
    seeds = {seed(json.load(open(ROOT / "runs" / r / "log.json"))["args"]) for r in runs}
    assert len(seeds) == 1, f"runs use different proposer seeds: {seeds}"
    return seeds.pop()


def rows_for(split, prop_seed, windows_by_id):
    rows = C.pool_rows([C.load(str(ROOT / "data" / "cands" / f"{split}_s{prop_seed}.jsonl"))])
    return rows, [windows_by_id[r["sent_id"]] for r in rows]


def regroup(examples, probs, rows):
    pos = {(x["sent_id"], x["cand_idx"]): i for i, x in enumerate(examples)}
    return [probs[[pos[(r["sent_id"], j)] for j in range(len(r["cands"]))]] if r["cands"]
            else np.zeros((0, len(C.LABELS)), dtype=np.float32) for r in rows]


def cmd_freeze(a):
    dev_by = {w.sent_id: w for w in load_split("dev")}
    ps = prop_seed_of(a.runs)
    rows, wins = rows_for("dev", ps, dev_by)
    ex = build_examples(dev_by, rows, with_labels=False)
    per_run = [regroup(ex, np.load(ROOT / "runs" / r / "dev_probs.npy"), rows) for r in a.runs]
    lrt = role_given_type(load_split("train"))
    gold = split_path("dev")
    best = None
    for alpha in ALPHAS:
        mixed = [mix_roles(rows, pw, alpha) for pw in per_run]
        for lam in LAMBDAS:
            for nms in NMS_MODES:
                for ml in MIN_LENS:
                    for th in THETAS:
                        fs = [score(records(wins, rows, pw, th, ml, nms, lam, lrt), gold,
                                    with_rouge=False)["arg_c_iou"]["f1"] for pw in mixed]
                        mu = st.mean(fs)
                        if best is None or mu > best["mean"]:
                            best = {"mean": mu, "std": st.stdev(fs) if len(fs) > 1 else 0.0,
                                    "per_run": fs, "theta": th, "min_len": ml, "nms": nms,
                                    "type_lambda": lam, "role_alpha": alpha}
    rule = {"rule": {k: best[k] for k in ["theta", "min_len", "nms", "type_lambda", "role_alpha"]},
            "selection": "max MEAN dev Arg-C IoU over verifier runs",
            "runs": a.runs, "prop_seeds": [ps],
            "dev_arg_c_iou_mean": best["mean"], "dev_arg_c_iou_std": best["std"],
            "dev_per_run": best["per_run"], "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = ROOT / "assets" / a.rule_name
    out.parent.mkdir(exist_ok=True)
    json.dump(rule, open(out, "w"), indent=2)
    print(json.dumps(rule, indent=2))


@torch.no_grad()
def cmd_test(a):
    rule = json.load(open(ROOT / "assets" / a.rule_name))
    R = rule["rule"]
    print(f"frozen rule (frozen at {rule['frozen_at']}): {R}")
    test_by = {w.sent_id: w for w in load_split("test")}
    rows, wins = rows_for("test", prop_seed_of(a.runs), test_by)
    ex = build_examples(test_by, rows, with_labels=False)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.add_special_tokens({"additional_special_tokens": [MARK_OPEN, MARK_CLOSE]})
    dl = DataLoader(VerifierDataset(ex, tok), batch_size=32, shuffle=False,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    lrt = role_given_type(load_split("train"))
    gold = split_path("test")
    (ROOT / "preds").mkdir(exist_ok=True)
    results = []
    for run in a.runs:
        model = Verifier(MODEL, len(tok)).cuda()
        model.load_state_dict(torch.load(ROOT / "runs" / run / "best.pt", map_location="cuda"))
        model.eval()
        probs = np.zeros((len(ex), len(C.LABELS)), dtype=np.float32)
        for b in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(b["input_ids"].cuda(), b["attention_mask"].cuda(),
                           b["open"].cuda(), b["close"].cuda(), b["feats"].cuda())
            probs[b["idx"].numpy()] = torch.softmax(lg.float(), -1).cpu().numpy()
        del model
        torch.cuda.empty_cache()
        pw = mix_roles(rows, regroup(ex, probs, rows), R["role_alpha"])
        recs = records(wins, rows, pw, R["theta"], R["min_len"], R["nms"], R["type_lambda"], lrt)
        write_jsonl(recs, str(ROOT / "preds" / f"test_{run}.jsonl"))
        m = score(recs, gold)
        results.append(m)
        for k, lab in [("trigger_rougeL", "ROUGE-L"), ("arg_i_iou", "Arg-I IoU"), ("arg_c_iou", "Arg-C IoU")]:
            print(f"{run}: {lab:9s} P {m[k]['p']:.2f}  R {m[k]['r']:.2f}  F1 {m[k]['f1']:.2f}")
    if len(results) > 1:
        for k, lab in [("trigger_rougeL", "ROUGE-L"), ("arg_i_iou", "Arg-I IoU"), ("arg_c_iou", "Arg-C IoU")]:
            v = [m[k]["f1"] for m in results]
            print(f"{lab:10s} {st.mean(v):.2f} ± {st.stdev(v):.2f}   per run {[round(x, 2) for x in v]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "test"])
    ap.add_argument("--runs", nargs="+", default=["verifier_s42"])
    ap.add_argument("--rule_name", default="decoding_rule.json")
    a = ap.parse_args()
    cmd_freeze(a) if a.cmd == "freeze" else cmd_test(a)
