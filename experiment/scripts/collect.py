"""Collect official metrics from prediction files into experiment/results/<id>.json."""

import argparse
import glob
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clave.evaluate import score
from clave.data import load_split
from clave.paths import split_path
from common import EXPERIMENT_ROOT, SOURCE_ROOT, output_path

METRICS = ["trigger_rougeL"] + [
    f"arg_{kind}_{mode}" for mode in ("exact", "overlap", "scirex", "iou")
    for kind in ("i", "c")
]


def locate(pattern):
    path = Path(pattern)
    if not path.is_absolute():
        path = SOURCE_ROOT / path
    return sorted(Path(p) for p in glob.glob(str(path)))


def seed_order(path):
    name = path.name
    for seed in (42, 13, 101):
        if f"_v{seed}_p{seed}" in name or f"_s{seed}." in name:
            return (42, 13, 101).index(seed)
    return 999


def summarize(per_run):
    out = {}
    for key in METRICS:
        out[key] = {}
        for field in ("p", "r", "f1"):
            vals = [r["metrics"][key][field] for r in per_run]
            out[key][field] = {"mean": st.mean(vals),
                               "std": st.stdev(vals) if len(vals) > 1 else 0.0}
    return out


def collect(args):
    if not args.id or not all(c.isalnum() or c in "_-" for c in args.id):
        raise ValueError("id must use letters, digits, _ or -")
    files = []
    for pattern in args.pred:
        files.extend(locate(pattern))
    files = sorted(set(files), key=lambda p: (seed_order(p), p.name))
    if not files:
        raise FileNotFoundError(args.pred)
    if args.runs and len(args.runs) != len(files):
        raise ValueError("number of --runs does not match prediction files")
    rule = None
    if args.rule:
        rpath = Path(args.rule)
        if not rpath.is_absolute():
            rpath = SOURCE_ROOT / rpath
        content = json.loads(rpath.read_text())
        rule = {"path": str(rpath), "content": content,
                "frozen_at": content.get("frozen_at"),
                "mtime": rpath.stat().st_mtime,
                "timestamp_status": ("recorded in rule" if content.get("frozen_at") else
                                     "unavailable in archived rule")}
        if args.split == "test" and not args.archive:
            for path in files:
                if path.stat().st_mtime <= rule["mtime"]:
                    raise ValueError(f"prediction predates frozen rule: {path}")
    per_run = []
    expected_ids = {w.sent_id for w in load_split(args.split)}
    for i, path in enumerate(files):
        recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        predicted_ids = [r["sent_id"] for r in recs]
        if len(predicted_ids) != len(expected_ids) or set(predicted_ids) != expected_ids:
            raise ValueError(f"prediction coverage or duplicate IDs: {path}")
        metrics = score(recs, split_path(args.split))
        per_run.append({"run": args.runs[i] if args.runs else path.stem,
                        "seed": (42, 13, 101)[seed_order(path)] if seed_order(path) < 3 else None,
                        "prediction_file": str(path), "n_windows": len(recs), "metrics": metrics})
    out = {"id": args.id, "split": args.split, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n_runs": len(per_run), "archive": args.archive,
           "rule": rule, "per_run": per_run, "summary": summarize(per_run)}
    dest = output_path("results", f"{args.id}.json")
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(out, indent=2))
    print(f"{args.id}: {len(per_run)} run(s) -> {dest}")
    for key in ("trigger_rougeL", "arg_i_iou", "arg_c_iou"):
        parts = [f"{field.upper()} {out['summary'][key][field]['mean']:.2f} ± "
                 f"{out['summary'][key][field]['std']:.2f}" for field in ("p", "r", "f1")]
        print(key, "  ".join(parts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--pred", nargs="+", required=True, help="quoted glob(s), relative to CLAVE root")
    ap.add_argument("--rule", help="frozen rule JSON, relative to CLAVE root")
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--archive", action="store_true")
    collect(ap.parse_args())
