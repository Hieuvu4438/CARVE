"""Timestamp and audit the two registered T-P2 archive-model test looks."""

import argparse
import datetime as dt
import hashlib
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive_carve_full"
LOG = ROOT / "TEST_LOG.md"
VARIANTS = {
    "abl_no_type_cond": "T-P2a",
    "abl_single_head": "T-P2b",
}


def paths(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    rule = ARCHIVE / "assets" / (
        "rules_abl_no_type_cond.json" if variant == "abl_no_type_cond"
        else "decoding_rules.abl_single_head.json")
    preds = [ARCHIVE / "preds" / f"test_{variant}_s{seed}.jsonl"
             for seed in (42, 13, 101)]
    return rule, preds


def row_for(test_id):
    rows = [line for line in LOG.read_text().splitlines()
            if line.startswith("|") and len(line.split("|")) == 10
            and line.split("|")[2].strip() == test_id]
    if len(rows) > 1:
        raise ValueError(f"duplicate test log ID: {test_id}")
    return rows[0] if rows else None


def freeze(variant):
    rule, _ = paths(variant)
    data = json.loads(rule.read_text())
    if data.get("frozen_at"):
        raise ValueError("rule already timestamped")
    data["frozen_at"] = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    data["seeds"] = [42, 13, 101]
    rule.write_text(json.dumps(data, indent=2))
    print(rule, data["frozen_at"])


def register(variant):
    test_id = VARIANTS[variant]
    if row_for(test_id):
        raise ValueError(f"{test_id} already registered")
    rule, preds = paths(variant)
    data = json.loads(rule.read_text())
    if not data.get("frozen_at") or any(p.exists() for p in preds):
        raise ValueError("rule must be frozen and predictions must not exist")
    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    digest = hashlib.sha256(rule.read_bytes()).hexdigest()
    row = (f"| {stamp} | {test_id} | {', '.join(f'{variant}_s{s}' for s in (42,13,101))} | "
           f"{rule} | {data['frozen_at']} | {', '.join(str(p) for p in preds)} | "
           f"RUNNING | pre-registered T-P2; isolated archive copy; sha256={digest} |")
    with LOG.open("a") as f:
        f.write(row + "\n")
    print(row)


def finish(variant):
    test_id = VARIANTS[variant]
    row = row_for(test_id)
    if row is None or "| RUNNING |" not in row:
        raise ValueError(f"{test_id} has no running registration")
    rule, preds = paths(variant)
    digest = hashlib.sha256(rule.read_bytes()).hexdigest()
    if f"sha256={digest}" not in row:
        raise ValueError("frozen rule changed after test registration")
    if not all(p.is_file() and p.stat().st_mtime > rule.stat().st_mtime for p in preds):
        raise ValueError("missing predictions or prediction predates rule")
    result = json.loads((ROOT / "results" / f"{variant}.json").read_text())
    if result["n_runs"] != 3 or {r["prediction_file"] for r in result["per_run"]} != \
            {str(p) for p in preds}:
        raise ValueError("collected predictions do not match registered paths")
    scores = [r["metrics"]["arg_c_iou"]["f1"] for r in result["per_run"]]
    summary = f"{st.mean(scores):.2f} ± {st.stdev(scores):.2f}"
    content = LOG.read_text()
    if content.count(row) != 1:
        raise ValueError("test log row changed")
    LOG.write_text(content.replace(row, row.replace("| RUNNING |", f"| {summary} |"), 1))
    print(test_id, summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["freeze", "register", "finish"])
    ap.add_argument("variant", choices=list(VARIANTS))
    a = ap.parse_args()
    {"freeze": freeze, "register": register, "finish": finish}[a.action](a.variant)
