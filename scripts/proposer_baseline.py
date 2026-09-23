"""Baseline without the verifier: CARVE-simple decoded with its calibrated threshold.

  freeze  One (tau, min_len) maximising the MEAN dev Arg-C IoU over the given
          proposer seeds (merge_gap 0) -> assets/proposer_rule.json.
  test    That one rule applied to every seed on test, scored once ->
          preds/test_proposer_s{seed}.jsonl.

Usage:
  python3 scripts/proposer_baseline.py freeze --seeds 42 13 101
  python3 scripts/proposer_baseline.py test   --seeds 42 13 101
"""

import argparse
import json
import statistics as st
import time

from transformers import AutoTokenizer

from carve.data import load_split
from carve.decode import posteriors, threshold_records
from carve.evaluate import score, write_jsonl
from carve.paths import repo_root, split_path

MODEL = "microsoft/deberta-v3-large"
TAUS = [0.5, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
MIN_LENS = [2, 3, 4]
ROOT = repo_root()


def post(split, seeds):
    wins, tok = load_split(split), AutoTokenizer.from_pretrained(MODEL)
    return wins, {s: posteriors(str(ROOT / "runs" / f"proposer_s{s}" / "best.pt"), MODEL, wins, tok) for s in seeds}


def cmd_freeze(a):
    wins, P = post("dev", a.seeds)
    best = None
    for ml in MIN_LENS:
        for t in TAUS:
            fs = [score(threshold_records(wins, *P[s], t, ml), split_path("dev"), with_rouge=False)["arg_c_iou"]["f1"]
                  for s in a.seeds]
            if best is None or st.mean(fs) > best["mean"]:
                best = {"mean": st.mean(fs), "std": st.stdev(fs) if len(fs) > 1 else 0.0,
                        "tau": t, "min_len": ml, "per_seed": fs}
    rule = {"rules": {"tau": best["tau"], "min_len": best["min_len"], "merge_gap": 0},
            "selection": "max mean dev Arg-C IoU F1 over proposer seeds", "seeds": a.seeds,
            "dev_arg_c_iou_mean": best["mean"], "dev_arg_c_iou_std": best["std"],
            "dev_per_seed": best["per_seed"], "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(rule, open(ROOT / "assets" / "proposer_rule.json", "w"), indent=2)
    print(json.dumps(rule, indent=2))


def cmd_test(a):
    R = json.load(open(ROOT / "assets" / "proposer_rule.json"))["rules"]
    tau = R["tau"]["_"] if isinstance(R["tau"], dict) else R["tau"]
    print(f"frozen rule: tau={tau} min_len={R['min_len']}")
    wins, P = post("test", a.seeds)
    (ROOT / "preds").mkdir(exist_ok=True)
    res = []
    for s in a.seeds:
        recs = threshold_records(wins, *P[s], tau, R["min_len"])
        write_jsonl(recs, str(ROOT / "preds" / f"test_proposer_s{s}.jsonl"))
        m = score(recs, split_path("test"))
        res.append(m)
        print(f"seed {s}: " + "  ".join(f"{k} P {m[k]['p']:.2f} R {m[k]['r']:.2f} F1 {m[k]['f1']:.2f}"
                                       for k in ["trigger_rougeL", "arg_i_iou", "arg_c_iou"]))
    if len(res) > 1:
        for k in ["trigger_rougeL", "arg_i_iou", "arg_c_iou"]:
            v = [m[k]["f1"] for m in res]
            print(f"{k:15s} {st.mean(v):.2f} ± {st.stdev(v):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "test"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 13, 101])
    a = ap.parse_args()
    cmd_freeze(a) if a.cmd == "freeze" else cmd_test(a)
