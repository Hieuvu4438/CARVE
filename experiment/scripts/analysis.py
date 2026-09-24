"""Phase-7 candidate, oracle and descriptive error analysis.

Metrics remain those of clave.evaluate; the local match assignment below is
only for descriptive error buckets and gold-span length groups.
"""

import argparse
import collections
import glob
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from clave import candidates as C
from clave.data import EXCLUDED_FROM_SCORING, EVENT_TYPES, load_split
from clave.evaluate import official_scorer, score
from clave.paths import split_path
from clave.select import mix_roles, records, role_given_type
from common import EXPERIMENT_ROOT, SOURCE_ROOT, candidate_path, output_path, rule_path, run_dir

BINS = [(1, 5, "1–4"), (5, 10, "5–9"), (10, 20, "10–19"),
        (20, 40, "20–39"), (40, 10**9, "≥40")]


def files(pattern):
    path = Path(pattern)
    if not path.is_absolute():
        path = SOURCE_ROOT / path
    found = sorted(Path(p) for p in glob.glob(str(path)))
    if not found:
        raise FileNotFoundError(pattern)
    return found


def safe_write(name, value):
    path = output_path("results", name)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    print(path)


def candidate_stats(seeds, output_name="candidate_stats.json"):
    train = {w.sent_id: w for w in load_split("train")}
    dev = {w.sent_id: w for w in load_split("dev")}
    out = {}
    costs = {}
    parameters = {}
    for seed in seeds:
        groups = {"oof": [candidate_path(f"oof_k{k}_s{seed}.jsonl") for k in range(5)],
                  "dev": [candidate_path(f"dev_s{seed}.jsonl")],
                  "insample": [candidate_path(f"insample_s{seed}.jsonl")]}
        out[str(seed)] = {}
        costs[str(seed)] = {}
        for stage, name in [("full_proposer", f"proposer_s{seed}")]:
            log_path = run_dir(name) / "log.json"
            if log_path.exists():
                log = json.loads(log_path.read_text()).get("log", [])
                costs[str(seed)][stage] = log[-1].get("secs") if log else None
        oof_seconds = []
        for k in range(5):
            log_path = run_dir(f"oof_k{k}_s{seed}") / "log.json"
            if log_path.exists():
                log = json.loads(log_path.read_text()).get("log", [])
                if log and log[-1].get("secs") is not None:
                    oof_seconds.append(log[-1]["secs"])
        costs[str(seed)]["oof_total_seconds"] = sum(oof_seconds) if oof_seconds else None
        for stage, name in [("main_verifier", f"verifier_v{seed}_p{seed}"),
                            ("insample_verifier", f"verifier_insample_v{seed}_p{seed}"),
                            ("nofeat_verifier", f"verifier_nofeat_v{seed}_p{seed}")]:
            log_path = run_dir(name) / "log.json" if (seed == 42 and stage == "main_verifier") or \
                (EXPERIMENT_ROOT / "runs" / name).exists() else None
            if log_path and log_path.exists():
                log = json.loads(log_path.read_text()).get("log", [])
                costs[str(seed)][stage] = log[-1].get("secs") if log else None
        for variant in ("abl_no_type_cond", "abl_single_head"):
            archive_log = (EXPERIMENT_ROOT / "archive_carve_full" / "runs" /
                           f"{variant}_s{seed}" / "log.json")
            if archive_log.exists():
                log = json.loads(archive_log.read_text()).get("log", [])
                costs[str(seed)][variant] = log[-1].get("secs") if log else None
        for group, paths in groups.items():
            rows = sum((C.load(str(p)) for p in paths), [])
            windows = train if group != "dev" else dev
            positive = total = 0
            for row in rows:
                C.label(windows[row["sent_id"]], row["cands"])
                total += len(row["cands"])
                positive += sum(c["label"] != C.REJECT for c in row["cands"])
            out[str(seed)][group] = {
                "windows": len(rows), "candidates": total,
                "per_window": round(total / len(rows), 3),
                "positive_pct": round(positive / total * 100, 3) if total else 0,
                "files": [str(p) for p in paths],
            }
    for stage, name in (("proposer", "proposer_s42"),
                        ("verifier", "verifier_s42")):
        weights = torch.load(run_dir(name) / "best.pt", map_location="cpu",
                             mmap=True, weights_only=True)
        parameters[stage] = sum(t.numel() for t in weights.values())
        del weights
    parameters["combined"] = parameters["proposer"] + parameters["verifier"]
    for variant in ("abl_no_type_cond", "abl_single_head"):
        checkpoint = (EXPERIMENT_ROOT / "archive_carve_full" / "runs" /
                      f"{variant}_s42" / "best.pt")
        if checkpoint.exists():
            weights = torch.load(checkpoint, map_location="cpu", mmap=True,
                                 weights_only=True)
            parameters[variant] = sum(t.numel() for t in weights.values())
            del weights
    safe_write(output_name, {"id": "candidate_stats",
               "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "seeds": out, "gpu_seconds_from_logs": costs,
               "parameters": parameters})


def oracle_dev(seeds, rule_name):
    rule = json.loads(rule_path(rule_name).read_text())["rule"]
    dev = load_split("dev")
    by_id = {w.sent_id: w for w in dev}
    lrt = role_given_type(load_split("train"))
    out = {}
    for seed in seeds:
        rows = C.pool_rows([C.load(str(candidate_path(f"dev_s{seed}.jsonl")))])
        flat = np.load(run_dir(f"verifier_v{seed}_p{seed}") / "dev_probs.npy")
        if flat.shape != (sum(len(r["cands"]) for r in rows), len(C.LABELS)):
            raise ValueError(f"dev probability shape mismatch for seed {seed}: {flat.shape}")
        offsets = np.cumsum([0] + [len(r["cands"]) for r in rows])
        actual = [flat[offsets[i]:offsets[i + 1]] for i in range(len(rows))]
        actual = mix_roles(rows, actual, rule["role_alpha"])
        pw = []
        for row in rows:
            C.label(by_id[row["sent_id"]], row["cands"])
            p = np.zeros((len(row["cands"]), len(C.LABELS)), dtype=np.float32)
            for i, cand in enumerate(row["cands"]):
                p[i, C.LABEL2ID[cand["label"]]] = 1.0
            pw.append(p)
        wins = [by_id[r["sent_id"]] for r in rows]
        args = (rule["theta"], rule["min_len"], rule["nms"], rule["type_lambda"], lrt)
        keep_role = score(records(wins, rows, pw, *args), split_path("dev"))
        baseline = score(records(wins, rows, actual, *args), split_path("dev"))
        original = [r["type_post"] for r in rows]
        for row, win in zip(rows, wins):
            onehot = np.zeros(len(EVENT_TYPES), dtype=np.float32)
            onehot[EVENT_TYPES.index(win.event_type)] = 1.0
            row["type_post"] = onehot.tolist()
        # λ=0 guarantees that the one-hot gold type is used as the oracle type.
        type_args = (rule["theta"], rule["min_len"], rule["nms"], 0.0, lrt)
        type_oracle = score(records(wins, rows, actual, *type_args), split_path("dev"))
        joint_oracle = score(records(wins, rows, pw, *type_args), split_path("dev"))
        for row, old in zip(rows, original):
            row["type_post"] = old
        out[str(seed)] = {
            "baseline_arg_c_iou": baseline["arg_c_iou"],
            "keep_role_arg_c_iou": keep_role["arg_c_iou"],
            "type_oracle_arg_c_iou": type_oracle["arg_c_iou"],
            "joint_oracle_arg_c_iou": joint_oracle["arg_c_iou"],
            "candidate_count": sum(len(r["cands"]) for r in rows),
        }
    safe_write("oracle_dev.json", {"id": "oracle_dev", "rule": rule_name,
               "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": out})


def named_prf(counts):
    if "matched" not in counts:
        return {k: named_prf(v) for k, v in counts.items()}
    matched = counts["matched"]
    pred = counts.get("pred_total", counts.get("pred", 0))
    gold = counts.get("gold_total", counts.get("gold", 0))
    p = 100 * matched / pred if pred else 0
    r = 100 * matched / gold if gold else 0
    f = 2 * p * r / (p + r) if p + r else 0
    return {"p": p, "r": r, "f1": f, "matched": matched, "pred": pred, "gold": gold}


def descriptive_errors(pred, gold, scorer):
    _, pred_roles = scorer.extract_triggers_and_roles(pred)
    _, gold_roles = scorer.extract_triggers_and_roles(gold)
    bucket = {label: [0, 0] for _, _, label in BINS}
    errors = collections.Counter()
    for sid, g_all in gold_roles.items():
        p = [x for x in pred_roles.get(sid, []) if x[1][2] not in EXCLUDED_FROM_SCORING]
        g = [x for x in g_all if x[1][2] not in EXCLUDED_FROM_SCORING]
        used_g = set()
        for x in p:
            # One-to-one diagnostic assignment, prioritising official true matches.
            candidates = [(j, y, C.iou(x[1][:2], y[1][:2])) for j, y in enumerate(g)
                          if j not in used_g and x[0][2] == y[0][2]]
            exact = [(j, y, iou) for j, y, iou in candidates
                     if x[1][2] == y[1][2] and iou > 0.5]
            wrong_role = [(j, y, iou) for j, y, iou in candidates
                          if x[1][2] != y[1][2] and iou > 0.5]
            boundary = [(j, y, iou) for j, y, iou in candidates
                        if 0 < iou <= 0.5]
            category, matches = next(
                ((label, vals) for label, vals in
                 (("correct", exact), ("wrong_role", wrong_role), ("boundary", boundary))
                 if vals), ("extra", []))
            errors[category] += 1
            if matches:
                j, y, _ = max(matches, key=lambda item: item[2])
                used_g.add(j)
                if category == "correct":
                    length = y[1][1] - y[1][0]
                    for lo, hi, label in BINS:
                        if lo <= length < hi:
                            bucket[label][0] += 1
                            break
        errors["missing"] += len(g) - len(used_g)
        for j, y in enumerate(g):
            length = y[1][1] - y[1][0]
            for lo, hi, label in BINS:
                if lo <= length < hi:
                    bucket[label][1] += 1
                    break
    return {"errors": dict(errors),
            "length_recall": {label: {"matched": m, "gold": n,
                                     "recall": 100 * m / n if n else 0}
                              for label, (m, n) in bucket.items()}}


def analyze_test(pred_pattern, baseline_pattern):
    M = official_scorer()
    gold = M.load_jsonl(split_path("test"))
    paths = files(pred_pattern)
    base_paths = files(baseline_pattern) if baseline_pattern else []
    all_results = []
    for path in paths:
        pred = M.load_jsonl(str(path))
        metrics = score(pred, split_path("test"))
        err = descriptive_errors(pred, gold, M)
        all_results.append({
            "file": str(path),
            "rolewise": named_prf(metrics["rolewise_iou"]),
            "eventtype": named_prf(metrics["eventtype_iou"]),
            "domain": named_prf(metrics["domain_iou"]),
            "rougeL_domain": metrics["rougeL_domain"],
            "rougeL_eventtype": metrics["rougeL_eventtype"],
            **err, "arg_c_iou": metrics["arg_c_iou"], "arg_i_iou": metrics["arg_i_iou"],
        })
    shift = None
    if base_paths:
        base = [score(M.load_jsonl(str(p)), split_path("test")) for p in base_paths]
        system_metrics = [score(M.load_jsonl(str(p)), split_path("test")) for p in paths]
        shift = {key: {field: st.mean(x[key][field] for x in
                                    system_metrics)
                              - st.mean(x[key][field] for x in base)
                       for field in ("p", "r", "f1")}
                 for key in ("arg_c_iou", "arg_i_iou")}
    safe_write("analysis.json", {"id": "analysis",
               "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "source": [str(p) for p in paths], "per_run": all_results,
               "precision_recall_shift_vs_baseline": shift})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    stats = sub.add_parser("candidate-stats")
    stats.add_argument("--seeds", type=int, nargs="+", default=[42, 13, 101])
    stats.add_argument("--out", default="candidate_stats.json")
    oracle = sub.add_parser("oracle-dev")
    oracle.add_argument("--seeds", type=int, nargs="+", default=[42, 13, 101])
    oracle.add_argument("--rule_name", default="decoding_rule_3seed.json")
    test = sub.add_parser("test")
    test.add_argument("--pred", required=True)
    test.add_argument("--baseline")
    a = ap.parse_args()
    if a.cmd == "candidate-stats":
        candidate_stats(a.seeds, a.out)
    elif a.cmd == "oracle-dev":
        oracle_dev(a.seeds, a.rule_name)
    else:
        analyze_test(a.pred, a.baseline)
