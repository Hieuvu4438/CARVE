"""Run the released paired-window bootstrap and save its output as JSON."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SOURCE_ROOT, output_path

MAIN = "experiment/preds/test_verifier_v*_p*__decoding_rule_3seed.jsonl"
SIMPLE = "preds/test_proposer_s*.jsonl"
ORIGINAL = "../SciEvent/method/SciEvent-Next/artifacts/preds/test_preds_s*.jsonl"
LINE = re.compile(
    r"^(arg_[ci]_iou)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)\s+"
    r"\[\s*([+-][\d.]+),\s*([+-][\d.]+)\]\s+([\d.]+)\s+"
    r"\[\s*([\d.]+),\s*([\d.]+)\]\s+([\d.]+)$"
)


def run_bootstrap(system, reference, n):
    env = os.environ.copy()
    env.setdefault("SCIEVENT_ROOT", "/home/haipd/SciEvent/third_party/SciEvent")
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    proc = subprocess.run(
        [sys.executable, str(SOURCE_ROOT / "scripts" / "bootstrap_ci.py"),
         "--sys", system, "--ref", reference, "--n", str(n)],
        cwd=SOURCE_ROOT, env=env, text=True, capture_output=True, check=True)
    out = {}
    for line in proc.stdout.splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        key, sy, ref, delta, low, high, prob, sys_low, sys_high, oneie = m.groups()
        out[key] = {"sys": float(sy), "ref": float(ref), "delta": float(delta),
                    "ci_low": float(low), "ci_high": float(high),
                    "p_gt_zero": float(prob), "sys_ci_low": float(sys_low),
                    "sys_ci_high": float(sys_high), "oneie": float(oneie)}
    if set(out) != {"arg_c_iou", "arg_i_iou"}:
        raise RuntimeError(f"cannot parse bootstrap output:\n{proc.stdout}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["main", "ablation"], required=True)
    ap.add_argument("--id", help="ablation result suffix, e.g. a1")
    ap.add_argument("--sys", help="quoted prediction glob for ablation")
    ap.add_argument("--n", type=int, default=5000)
    a = ap.parse_args()
    if a.mode == "main":
        name = "bootstrap_main"
        comparisons = {
            "CLAVE vs CARVE-simple": run_bootstrap(MAIN, SIMPLE, a.n),
            "CLAVE vs original CARVE": run_bootstrap(MAIN, ORIGINAL, a.n),
        }
        result = {"id": name, "n_resamples": a.n, "comparisons": comparisons}
    else:
        if not a.id or not a.sys or not re.fullmatch(r"[a-z0-9_]+", a.id):
            raise ValueError("ablation requires --id and --sys")
        name = f"bootstrap_{a.id}"
        result = {"id": name, "n_resamples": a.n,
                  **run_bootstrap(a.sys, MAIN, a.n)}
    result["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dest = output_path("results", f"{name}.json")
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(result, indent=2))
    print(dest)


if __name__ == "__main__":
    main()
