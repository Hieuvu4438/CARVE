"""Freeze a shared dev rule, then evaluate registered test variants.

All new rules, probabilities and predictions are written under experiment/.
The published seed-42 run and assets are read only.
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

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from clave import candidates as C
from clave.data import load_split
from clave.evaluate import score, write_jsonl
from clave.paths import split_path
from clave.select import ALPHAS, MIN_LENS, NMS_MODES, THETAS, mix_roles, records, role_given_type
from clave.verifier import MARK_CLOSE, MARK_OPEN, Verifier, VerifierDataset, build_examples, collate
from common import EXPERIMENT_ROOT, SOURCE_ROOT, candidate_path, output_path, rule_path, run_dir

MODEL = "microsoft/deberta-v3-large"
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
MAIN_RUNS = [f"verifier_v{s}_p{s}" for s in (42, 13, 101)]
REGISTERED = {
    "decoding_rule_3seed.json": MAIN_RUNS,
    "rule_abl_insample.json": [f"verifier_insample_v{s}_p{s}" for s in (42, 13, 101)],
    "rule_abl_nofeat.json": [f"verifier_nofeat_v{s}_p{s}" for s in (42, 13, 101)],
    "rule_abl_role_verifier.json": MAIN_RUNS,
    "rule_abl_role_proposer.json": MAIN_RUNS,
    "rule_abl_no_type.json": MAIN_RUNS,
    "rule_abl_pure_verifier.json": MAIN_RUNS,
    "rule_abl_nms_iou.json": MAIN_RUNS,
    "rule_abl_nms_wis.json": MAIN_RUNS,
    "decoding_rule.json": ["verifier_s42"],  # Phase-0/1 regression only
}
REGISTERED_FIXES = {
    "rule_abl_role_verifier.json": {"role_alpha": 1.0},
    "rule_abl_role_proposer.json": {"role_alpha": 0.0},
    "rule_abl_no_type.json": {"type_lambda": 0.0},
    "rule_abl_pure_verifier.json": {"role_alpha": 1.0, "type_lambda": 0.0},
    "rule_abl_nms_iou.json": {"nms": "iou"},
    "rule_abl_nms_wis.json": {"nms": "wis"},
}
RULE_TEST_ID = {
    "decoding_rule_3seed.json": "T-main",
    "rule_abl_insample.json": "T-A1",
    "rule_abl_nofeat.json": "T-A2",
    "rule_abl_role_verifier.json": "T-A3",
    "rule_abl_role_proposer.json": "T-A4",
    "rule_abl_no_type.json": "T-A5",
    "rule_abl_pure_verifier.json": "T-A6",
    "rule_abl_nms_iou.json": "T-A7a",
    "rule_abl_nms_wis.json": "T-A7b",
    "decoding_rule.json": "P1 regression",
}


def proposer_seed(run):
    args = json.loads((run_dir(run) / "log.json").read_text())["args"]
    return int(args["prop_seed"] if "prop_seed" in args else args["prop_seeds"][0])


def rows_for(split, prop_seed, windows_by_id):
    rows = C.pool_rows([C.load(str(candidate_path(f"{split}_s{prop_seed}.jsonl")))])
    return rows, [windows_by_id[r["sent_id"]] for r in rows]


def regroup(examples, probs, rows):
    pos = {(x["sent_id"], x["cand_idx"]): i for i, x in enumerate(examples)}
    return [probs[[pos[(r["sent_id"], j)] for j in range(len(r["cands"]))]] if r["cands"]
            else np.zeros((0, len(C.LABELS)), dtype=np.float32) for r in rows]


def dimensions(a):
    grids = {"role_alpha": ALPHAS, "type_lambda": LAMBDAS,
             "nms": a.nms_modes or NMS_MODES, "min_len": MIN_LENS, "theta": THETAS}
    fixed = {}
    for item in a.fix:
        if "=" not in item:
            raise ValueError(f"--fix requires KEY=VAL: {item}")
        key, raw = item.split("=", 1)
        if key not in grids or key in fixed:
            raise ValueError(f"invalid or repeated fixed key: {key}")
        val = raw if key == "nms" else json.loads(raw)
        if val not in grids[key]:
            raise ValueError(f"{key}={val} is not in the registered grid {grids[key]}")
        grids[key] = [val]
        fixed[key] = val
    expected = REGISTERED_FIXES.get(a.rule_name)
    if expected is not None and fixed != expected:
        raise ValueError(f"{a.rule_name} requires --fix values {expected}")
    if a.rule_name in REGISTERED and REGISTERED[a.rule_name] != a.runs:
        raise ValueError(f"{a.rule_name} requires runs {REGISTERED[a.rule_name]}")
    if a.rule_name == "decoding_rule.json" and (fixed or a.nms_modes):
        raise ValueError("released-rule regression requires the original full grid")
    return grids, fixed


def cmd_freeze(a):
    grids, fixed = dimensions(a)
    dev_by = {w.sent_id: w for w in load_split("dev")}
    per_run = []
    for run in a.runs:
        seed = proposer_seed(run)
        rows, wins = rows_for("dev", seed, dev_by)
        ex = build_examples(dev_by, rows, with_labels=False)
        probs = np.load(run_dir(run) / "dev_probs.npy")
        if probs.shape != (len(ex), len(C.LABELS)):
            raise ValueError(f"dev probability shape mismatch for {run}: {probs.shape}")
        per_run.append({"run": run, "prop_seed": seed, "rows": rows, "wins": wins,
                        "probs": regroup(ex, probs, rows)})
    lrt = role_given_type(load_split("train"))
    gold = split_path("dev")
    best = None
    for alpha in grids["role_alpha"]:
        mixed = [mix_roles(x["rows"], x["probs"], alpha) for x in per_run]
        for lam in grids["type_lambda"]:
            for nms in grids["nms"]:
                for ml in grids["min_len"]:
                    for th in grids["theta"]:
                        fs = [score(records(x["wins"], x["rows"], pw, th, ml, nms, lam, lrt),
                                    gold, with_rouge=False)["arg_c_iou"]["f1"]
                              for x, pw in zip(per_run, mixed)]
                        mu = st.mean(fs)
                        if best is None or mu > best["mean"]:
                            best = {"mean": mu, "std": st.stdev(fs) if len(fs) > 1 else 0.0,
                                    "per_run": fs, "theta": th, "min_len": ml,
                                    "nms": nms, "type_lambda": lam, "role_alpha": alpha}
    rule = {"rule": {k: best[k] for k in ("theta", "min_len", "nms", "type_lambda", "role_alpha")},
            "selection": "max MEAN dev Arg-C IoU over verifier runs",
            "runs": a.runs, "prop_seeds": [x["prop_seed"] for x in per_run],
            "dev_arg_c_iou_mean": best["mean"], "dev_arg_c_iou_std": best["std"],
            "dev_per_run": best["per_run"],
            "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if fixed:
        rule["fixed"] = fixed
    if a.rule_name == "decoding_rule.json" and a.runs == ["verifier_s42"]:
        released = json.loads((SOURCE_ROOT / "assets" / "decoding_rule.json").read_text())
        if (rule["rule"] != released["rule"] or
                abs(rule["dev_arg_c_iou_mean"] - released["dev_arg_c_iou_mean"]) > 1e-9):
            raise AssertionError("seed-42 dev regression differs from the published rule")
        rule = {**released, "frozen_at": rule["frozen_at"]}
    out = rule_path(a.rule_name, write=True)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rule, indent=2))
    print(json.dumps(rule, indent=2))


@torch.no_grad()
def predict_test(run, examples):
    path = run_dir(run) / "test_probs.npy"
    if path.exists():
        probs = np.load(path)
        if probs.shape != (len(examples), len(C.LABELS)):
            raise ValueError(f"test probability shape mismatch for {run}: {probs.shape}")
        print(f"{run}: reused {path}")
        return probs
    target = run_dir(run, write=True) / "test_probs.npy"
    args = json.loads((run_dir(run) / "log.json").read_text())["args"]
    model_name = args.get("model_name", MODEL)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.add_special_tokens({"additional_special_tokens": [MARK_OPEN, MARK_CLOSE]})
    dl = DataLoader(VerifierDataset(examples, tok), batch_size=32, shuffle=False,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    model = Verifier(model_name, len(tok), dropout=args.get("dropout", 0.1),
                     use_feats=args.get("use_feats", not args.get("no_feats", False))).cuda()
    model.load_state_dict(torch.load(run_dir(run) / "best.pt", map_location="cuda"))
    model.eval()
    probs = np.zeros((len(examples), len(C.LABELS)), dtype=np.float32)
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(b["input_ids"].cuda(), b["attention_mask"].cuda(),
                       b["open"].cuda(), b["close"].cuda(), b["feats"].cuda())
        probs[b["idx"].numpy()] = torch.softmax(lg.float(), -1).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    np.save(target, probs)
    print(f"{run}: saved {target}")
    return probs


def prediction_name(run, rule_name, pred_tag):
    tag = pred_tag if pred_tag is not None else Path(rule_name).stem
    if not tag or not all(c.isalnum() or c in "_-" for c in tag):
        raise ValueError(f"invalid prediction tag: {tag}")
    if rule_name == "decoding_rule.json" and run == "verifier_s42" and pred_tag is None:
        return "test_verifier_s42.jsonl"
    return f"test_{run}__{tag}.jsonl"


def prepare_test_log(a, rule):
    """Register the exact test look before reading test candidates or labels."""
    test_id = RULE_TEST_ID[a.rule_name]
    log_path = EXPERIMENT_ROOT / "TEST_LOG.md"
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    lines = log_path.read_text().splitlines()
    old = [line for line in lines if line.startswith("|") and
           len(line.split("|")) > 2 and line.split("|")[2].strip() == test_id]
    paths = [output_path("preds", prediction_name(run, a.rule_name, a.pred_tag))
             for run in a.runs]
    rule_file = rule_path(a.rule_name)
    digest = hashlib.sha256(rule_file.read_bytes()).hexdigest()
    if old:
        if not a.resume or len(old) != 1 or "RUNNING" not in old[0]:
            raise RuntimeError(f"test ID {test_id} was already used; only an incomplete look can resume")
        if any(str(p) not in old[0] for p in paths) or f"sha256={digest}" not in old[0]:
            raise RuntimeError("resume rule or prediction paths differ from the registered look")
        return log_path, old[0], paths
    if a.resume:
        raise RuntimeError(f"no incomplete test look to resume for {test_id}")
    if any(p.exists() for p in paths):
        raise FileExistsError("test prediction already exists before registration")
    when = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    row = (f"| {when} | {test_id} | {', '.join(a.runs)} | {rule_file} | "
           f"{rule['frozen_at']} | {', '.join(str(p) for p in paths)} | RUNNING | "
           f"pre-registered; sha256={digest} |")
    with log_path.open("a") as f:
        f.write(row + "\n")
    return log_path, row, paths


def finish_test_log(log_path, row, results):
    values = [m["arg_c_iou"]["f1"] for m in results]
    summary = (f"{st.mean(values):.2f} ± {st.stdev(values):.2f}" if len(values) > 1
               else f"{values[0]:.2f} (single run)")
    updated = row.replace("| RUNNING |", f"| {summary} |")
    content = log_path.read_text()
    if content.count(row) != 1:
        raise RuntimeError("test log row changed while evaluation was running")
    log_path.write_text(content.replace(row, updated, 1))


def print_metrics(run, metrics):
    for key, label in (("trigger_rougeL", "ROUGE-L"), ("arg_i_iou", "Arg-I IoU"),
                       ("arg_c_iou", "Arg-C IoU")):
        print(f"{run}: {label:9s} P {metrics[key]['p']:.2f}  "
              f"R {metrics[key]['r']:.2f}  F1 {metrics[key]['f1']:.2f}")


def cmd_test(a):
    if a.rule_name not in REGISTERED or REGISTERED[a.rule_name] != a.runs:
        raise ValueError("test evaluation is not registered in §2 or Phase 0")
    rule = json.loads(rule_path(a.rule_name).read_text())
    if not rule.get("frozen_at"):
        raise ValueError("test requires a timestamped frozen rule")
    if a.rule_name != "decoding_rule.json":
        if rule.get("runs") != a.runs:
            raise ValueError("frozen rule run list differs from test run list")
        expected_seeds = [proposer_seed(run) for run in a.runs]
        if rule.get("prop_seeds") != expected_seeds:
            raise ValueError("frozen rule proposer seeds differ from run logs")
        if rule.get("fixed", {}) != REGISTERED_FIXES.get(a.rule_name, {}):
            raise ValueError("frozen ablation constraints do not match registration")
    R = rule["rule"]
    print(f"frozen rule (frozen at {rule['frozen_at']}): {R}")
    log_path, log_row, paths = prepare_test_log(a, rule)
    test_by = {w.sent_id: w for w in load_split("test")}
    lrt = role_given_type(load_split("train"))
    gold = split_path("test")
    results = []
    for run, out in zip(a.runs, paths):
        if out.exists():
            if not a.resume:
                raise FileExistsError(out)
            recs = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
            m = score(recs, gold)
            results.append(m)
            print(f"{run}: reused existing prediction {out}")
            print_metrics(run, m)
            continue
        seed = proposer_seed(run)
        rows, wins = rows_for("test", seed, test_by)
        ex = build_examples(test_by, rows, with_labels=False)
        probs = predict_test(run, ex)
        pw = mix_roles(rows, regroup(ex, probs, rows), R["role_alpha"])
        recs = records(wins, rows, pw, R["theta"], R["min_len"], R["nms"], R["type_lambda"], lrt)
        temp = out.with_suffix(out.suffix + ".tmp")
        if temp.exists():
            temp.unlink()  # only experiment-owned partial output from this same look
        write_jsonl(recs, str(temp))
        temp.replace(out)
        m = score(recs, gold)
        results.append(m)
        print_metrics(run, m)
    if len(results) > 1:
        for k, lab in (("trigger_rougeL", "ROUGE-L"), ("arg_i_iou", "Arg-I IoU"),
                       ("arg_c_iou", "Arg-C IoU")):
            print(f"{lab:10s} " + "  ".join(
                f"{metric.upper()} {st.mean([m[k][metric] for m in results]):.2f} ± "
                f"{st.stdev([m[k][metric] for m in results]):.2f}"
                for metric in ("p", "r", "f1")))
    finish_test_log(log_path, log_row, results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "test"])
    ap.add_argument("--runs", nargs="+", default=["verifier_s42"])
    ap.add_argument("--rule_name", default="decoding_rule.json")
    ap.add_argument("--fix", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--nms_modes", nargs="+", choices=NMS_MODES)
    ap.add_argument("--pred_tag")
    ap.add_argument("--resume", action="store_true", help="continue the same logged test look")
    args = ap.parse_args()
    cmd_freeze(args) if args.cmd == "freeze" else cmd_test(args)
