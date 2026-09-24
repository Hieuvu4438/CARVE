"""Verify every required pre-registered test look is finished and auditable."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EXPERIMENT_ROOT

REQUIRED = ["T-main", "T-A1", "T-A2", "T-A3", "T-A4", "T-A5",
            "T-A6", "T-A7a", "T-A7b", "T-P1"]


def audit(ids):
    log = EXPERIMENT_ROOT / "TEST_LOG.md"
    rows = {}
    for line in log.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 8 or cells[1] not in ids:
            continue
        if cells[1] in rows:
            raise ValueError(f"duplicate test log ID {cells[1]}")
        rows[cells[1]] = cells
    for test_id in ids:
        if test_id not in rows:
            raise ValueError(f"missing TEST_LOG row for {test_id}")
        cells = rows[test_id]
        if cells[6] == "RUNNING" or not cells[4] or not cells[7].startswith("pre-registered"):
            raise ValueError(f"unfinished or invalid TEST_LOG row for {test_id}")
        rule_file = Path(cells[3])
        if not rule_file.is_file():
            raise FileNotFoundError(rule_file)
        rule = json.loads(rule_file.read_text())
        if rule.get("frozen_at") != cells[4]:
            raise ValueError(f"frozen_at mismatch for {test_id}")
        for pred in cells[5].split(", "):
            pred_file = Path(pred)
            if not pred_file.is_file():
                raise FileNotFoundError(pred_file)
            if rule_file.stat().st_mtime >= pred_file.stat().st_mtime:
                raise ValueError(f"prediction not later than frozen rule: {pred_file}")
        print(f"{test_id}: {cells[6]}; {len(cells[5].split(', '))} prediction file(s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", default=REQUIRED)
    audit(ap.parse_args().ids)
