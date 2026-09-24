"""Append the six optional CARVE archive retrains to RUN_LOG after completion."""

import datetime as dt
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive_carve_full"
RUN_LOG = ROOT / "RUN_LOG.md"


def formatted(ts):
    return dt.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def birth(path):
    value = int(subprocess.check_output(["stat", "-c", "%W", str(path)], text=True))
    if value <= 0:
        raise RuntimeError(f"filesystem birth timestamp unavailable for {path}")
    return value


content = RUN_LOG.read_text()
rows = []
for variant in ("abl_no_type_cond", "abl_single_head"):
    for seed in (42, 13, 101):
        run = f"{variant}_s{seed}"
        key = f"run={run};"
        if key in content:
            raise ValueError(f"RUN_LOG already has {run}")
        log = ROOT / "logs" / f"p2_{run}.log"
        run_dir = ARCHIVE / "runs" / run
        data_path = run_dir / "log.json"
        checkpoint = run_dir / "best.pt"
        if not (log.is_file() and data_path.is_file() and checkpoint.is_file()):
            raise FileNotFoundError(run)
        data = json.loads(data_path.read_text())
        if len(data["log"]) != 30 or data["config"]["seed"] != seed:
            raise ValueError(f"incomplete or mismatched training log for {run}")
        best = data["best"]
        command = (f"python3 -m carve.train -c configs/{variant}.json "
                   f"--set run_name={run} seed={seed}")
        rows.append(
            f"| Phase7-T-P2 | `{command}` | {formatted(birth(log))} | "
            f"{formatted(data_path.stat().st_mtime)} | best epoch {best['epoch']}; "
            f"dev Arg-C F1 {best['arg_c_iou_f1']:.2f}; elapsed {data['log'][-1]['secs']}s | "
            f"run={run}; seed={seed}; archive code copied into experiment; "
            f"stdout=experiment/logs/p2_{run}.log; checkpoint retained |")
with RUN_LOG.open("a") as f:
    f.write("\n".join(rows) + "\n")
print(f"recorded {len(rows)} T-P2 retrains")
