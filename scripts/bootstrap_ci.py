"""Paired window bootstrap on test between two systems (default: CARVE-simple +
HONE vs the CARVE-simple threshold baseline with the same proposer seed).

Resamples the 163 test windows with replacement (the unit of annotation). In
every replicate the *same* windows are scored for both systems; each side's F1
is the mean over its prediction files (seeds). Matching uses the benchmark's own
functions (IoU > 0.5, greedy one-to-one, event type must agree, Agent /
PrimaryObject / SecondaryObject excluded), so the point estimates equal the
headline numbers. Reads prediction files only.

Usage:
  python3 scripts/bootstrap_ci.py --sys 'preds/test_verifier_s42.jsonl' \
                                  --ref 'preds/test_proposer_s42.jsonl' [--n 5000]
"""

import argparse
import glob

import numpy as np

from carve.data import EXCLUDED_FROM_SCORING
from carve.evaluate import official_scorer
from carve.paths import repo_root, split_path

BASELINE = {"arg_c_iou": 41.61, "arg_i_iou": 53.57}   # OneIE, SciEvent paper Table 4

ap = argparse.ArgumentParser()
ap.add_argument("--sys", default="preds/test_verifier_s42.jsonl")
ap.add_argument("--ref", default="preds/test_proposer_s42.jsonl")
ap.add_argument("--n", type=int, default=5000)
a = ap.parse_args()

ROOT = repo_root()
M = official_scorer()
gold = M.load_jsonl(split_path("test"))
sent_ids = [e["sent_id"] for e in gold]
_, gold_roles = M.extract_triggers_and_roles(gold)


def counts(pred_file, use_role):
    _, pr = M.extract_triggers_and_roles(M.load_jsonl(pred_file))
    out = np.zeros((len(sent_ids), 3))
    for k, sid in enumerate(sent_ids):
        g = [x for x in gold_roles[sid] if x[1][2] not in EXCLUDED_FROM_SCORING]
        p = [x for x in pr.get(sid, []) if x[1][2] not in EXCLUDED_FROM_SCORING]
        m, used = 0, set()
        for x in p:
            for i, y in enumerate(g):
                if i in used or x[0][2] != y[0][2] or (use_role and x[1][2] != y[1][2]):
                    continue
                if M.iou_overlap(x[1][:2], y[1][:2]):
                    m += 1
                    used.add(i)
                    break
        out[k] = (m, len(p), len(g))
    return out


def f1(c):
    p = np.where(c[..., 1] > 0, c[..., 0] / np.maximum(c[..., 1], 1), 0)
    r = np.where(c[..., 2] > 0, c[..., 0] / np.maximum(c[..., 2], 1), 0)
    return np.where(p + r > 0, 200 * p * r / np.maximum(p + r, 1e-12), 0)


sys_files = sorted(glob.glob(str(ROOT / a.sys)))
ref_files = sorted(glob.glob(str(ROOT / a.ref)))
assert sys_files and ref_files, (sys_files, ref_files)
rng = np.random.default_rng(0)
W = np.stack([np.bincount(rng.integers(0, len(sent_ids), len(sent_ids)), minlength=len(sent_ids))
              for _ in range(a.n)])
print(f"{a.n} paired resamples of {len(sent_ids)} test windows; sys {len(sys_files)} file(s), ref {len(ref_files)}\n")
print(f"{'metric':10} {'sys':>7} {'ref':>7} {'Δ':>6} {'Δ 95% CI':>17} {'P(Δ>0)':>7} {'sys 95% CI':>17} {'OneIE':>6}")
for key, use_role in [("arg_c_iou", True), ("arg_i_iou", False)]:
    def side(files):
        cs = [counts(f, use_role) for f in files]
        return np.mean([f1(c.sum(0)) for c in cs]), np.mean([f1(W @ c) for c in cs], axis=0)
    sp, sb = side(sys_files)
    rp, rb = side(ref_files)
    d = sb - rb
    lo, hi = np.percentile(d, [2.5, 97.5])
    slo, shi = np.percentile(sb, [2.5, 97.5])
    print(f"{key:10} {sp:7.2f} {rp:7.2f} {sp - rp:+6.2f} [{lo:+6.2f}, {hi:+6.2f}] {np.mean(d > 0):7.3f} "
          f"[{slo:6.2f}, {shi:6.2f}] {BASELINE[key]:6.2f}")
