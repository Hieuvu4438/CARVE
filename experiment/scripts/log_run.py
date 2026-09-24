"""Record each experiment-owned training run in RUN_LOG.md."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EXPERIMENT_ROOT, run_dir


def clean(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "finish"])
    ap.add_argument("--phase", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--command", required=True)
    ap.add_argument("--started", required=True)
    ap.add_argument("--finished")
    ap.add_argument("--stdout-log", required=True)
    a = ap.parse_args()
    path = EXPERIMENT_ROOT / "RUN_LOG.md"
    if not path.exists():
        raise FileNotFoundError(path)
    marker = f"run={a.run}; seed={a.seed}"
    text = path.read_text()
    matching = [line for line in text.splitlines() if line.startswith("|") and marker in line]
    if a.action == "start":
        if matching:
            raise RuntimeError(f"training run already logged: {marker}")
        row = (f"| {clean(a.phase)} | `{clean(a.command)}` | {a.started} | "
               f"RUNNING | pending | {marker}; stdout={clean(a.stdout_log)} |")
        with path.open("a") as f:
            f.write(row + "\n")
        return
    if len(matching) != 1 or "| RUNNING |" not in matching[0] or not a.finished:
        raise RuntimeError(f"cannot finish unregistered run: {marker}")
    run_log = run_dir(a.run) / "log.json"
    info = json.loads(run_log.read_text())
    history = info.get("log", [])
    best = info.get("best", {})
    best_f1 = best.get("arg_c_iou_f1")
    epoch = best.get("epoch")
    if epoch is None and best_f1 is not None:
        epoch = next((i for i, row in enumerate(history, 1)
                      if row.get("arg_c_iou_f1") == best_f1), None)
    duration = history[-1].get("secs") if history else None
    result = (f"best epoch {epoch}; dev Arg-C F1 {best_f1:.2f}; "
              f"elapsed {duration}s" if best_f1 is not None else "best metric unavailable")
    old = matching[0]
    new = old.replace("| RUNNING | pending |", f"| {a.finished} | {result} |")
    path.write_text(text.replace(old, new, 1))


if __name__ == "__main__":
    main()
