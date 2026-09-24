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
import datetime as dt
import hashlib
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from clave.data import load_split
from clave.decode import posteriors, threshold_records
from clave.evaluate import score, write_jsonl
from clave.paths import split_path
from common import EXPERIMENT_ROOT, output_path, rule_path, run_dir

MODEL = "microsoft/deberta-v3-large"
TAUS = [0.5, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
MIN_LENS = [2, 3, 4]
ROOT = EXPERIMENT_ROOT


def post(split, seeds):
    wins, tok = load_split(split), AutoTokenizer.from_pretrained(MODEL)
    return wins, {s: posteriors(str(run_dir(f"proposer_s{s}") / "best.pt"), MODEL, wins, tok) for s in seeds}


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
    out = rule_path("proposer_rule.json", write=True)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rule, open(out, "w"), indent=2)
    print(json.dumps(rule, indent=2))


def cmd_test(a):
    raise ValueError("the released baseline test already exists; use eval for registered T-P1")


def cmd_eval(a):
    if a.tau is None or a.min_len is None or a.tag is None:
        raise ValueError("eval requires --tau, --min_len and --tag")
    if a.split == "test" and (a.tau != 0 or a.min_len != 1 or a.tag != "argmax"
                               or a.seeds != [42, 13, 101]):
        raise ValueError("only pre-registered T-P1 argmax test is allowed")
    if not (0 <= a.tau <= 1 and a.min_len >= 1):
        raise ValueError("invalid tau or min_len")
    if not a.tag or not all(c.isalnum() or c in "_-" for c in a.tag):
        raise ValueError("invalid tag")
    frozen_path = rule_path(f"proposer_{a.tag}_rule.json", write=True)
    if a.split == "test":
        if not frozen_path.exists():
            raise FileNotFoundError("run registered argmax eval on dev before test")
        frozen = json.loads(frozen_path.read_text())
        if (frozen.get("rules") != {"tau": a.tau, "min_len": a.min_len, "merge_gap": 0}
                or frozen.get("seeds") != a.seeds or not frozen.get("frozen_at")):
            raise ValueError("test arguments differ from frozen dev rule")
    elif frozen_path.exists():
        raise FileExistsError(frozen_path)
    paths = {s: output_path("preds", f"{a.split}_proposer_{a.tag}_s{s}.jsonl")
             for s in a.seeds}
    log_path = EXPERIMENT_ROOT / "TEST_LOG.md"
    log_row = None
    if a.split == "test":
        if not log_path.exists():
            raise FileNotFoundError(log_path)
        lines = log_path.read_text().splitlines()
        previous = [line for line in lines if line.startswith("|") and
                    len(line.split("|")) > 2 and line.split("|")[2].strip() == "T-P1"]
        digest = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
        if previous:
            if not a.resume or len(previous) != 1 or "RUNNING" not in previous[0]:
                raise RuntimeError("T-P1 was already evaluated; only an incomplete look can resume")
            log_row = previous[0]
            if (f"sha256={digest}" not in log_row or
                    any(str(path) not in log_row for path in paths.values())):
                raise RuntimeError("T-P1 resume differs from the registered rule or prediction paths")
        else:
            if a.resume:
                raise RuntimeError("no incomplete T-P1 look to resume")
            if any(path.exists() for path in paths.values()):
                raise FileExistsError("T-P1 prediction exists before registration")
            when = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            log_row = (f"| {when} | T-P1 | proposer_s42, proposer_s13, proposer_s101 | "
                       f"{frozen_path} | {frozen['frozen_at']} | "
                       f"{', '.join(str(path) for path in paths.values())} | RUNNING | "
                       f"pre-registered; sha256={digest} |")
            with log_path.open("a") as f:
                f.write(log_row + "\n")
    missing = [s for s in a.seeds if not (a.split == "test" and paths[s].exists())]
    wins, P = post(a.split, missing) if missing else ([], {})
    results = []
    for s in a.seeds:
        out = paths[s]
        if out.exists():
            if a.split != "test" or not a.resume:
                raise FileExistsError(out)
            recs = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        else:
            recs = threshold_records(wins, *P[s], a.tau, a.min_len)
            temp = out.with_suffix(out.suffix + ".tmp") if a.split == "test" else out
            if temp.exists():
                temp.unlink()  # only experiment-owned partial output of this look
            write_jsonl(recs, str(temp))
            if a.split == "test":
                temp.replace(out)
        m = score(recs, split_path(a.split))
        results.append(m)
        print(f"seed {s}: " + "  ".join(f"{k} P {m[k]['p']:.2f} R {m[k]['r']:.2f} F1 {m[k]['f1']:.2f}"
                                       for k in ("trigger_rougeL", "arg_i_iou", "arg_c_iou")))
    if len(results) > 1:
        for k in ("trigger_rougeL", "arg_i_iou", "arg_c_iou"):
            print(f"{k:15s} " + "  ".join(
                f"{q.upper()} {st.mean([m[k][q] for m in results]):.2f} ± "
                f"{st.stdev([m[k][q] for m in results]):.2f}" for q in ("p", "r", "f1")))
    if a.split == "dev":
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        frozen_path.write_text(json.dumps({
            "rules": {"tau": a.tau, "min_len": a.min_len, "merge_gap": 0},
            "selection": "fixed before test; no parameter search",
            "seeds": a.seeds, "dev_arg_c_iou_per_seed": [m["arg_c_iou"]["f1"] for m in results],
            "dev_arg_c_iou_mean": st.mean(m["arg_c_iou"]["f1"] for m in results),
            "dev_arg_c_iou_std": st.stdev(m["arg_c_iou"]["f1"] for m in results)
            if len(results) > 1 else 0.0,
            "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))
    else:
        values = [m["arg_c_iou"]["f1"] for m in results]
        summary = f"{st.mean(values):.2f} ± {st.stdev(values):.2f}"
        content = log_path.read_text()
        if content.count(log_row) != 1:
            raise RuntimeError("T-P1 log row changed during evaluation")
        log_path.write_text(content.replace(log_row, log_row.replace("| RUNNING |", f"| {summary} |"), 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "test", "eval"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 13, 101])
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--min_len", type=int)
    ap.add_argument("--tag")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.cmd == "freeze":
        cmd_freeze(a)
    elif a.cmd == "test":
        cmd_test(a)
    else:
        cmd_eval(a)
