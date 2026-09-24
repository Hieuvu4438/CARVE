"""Extract published baseline tables from read-only paper notes into results JSON."""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SOURCE_ROOT, output_path


def section(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def rows(text, numeric_only=True):
    out = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in ("Method", "configuration", "cấu hình", "---"):
            continue
        if not numeric_only or all(re.match(r"^[\d.]+(?: ± [\d.]+)?$", x) for x in cells[1:4]):
            out[cells[0]] = cells[1:]
    return out


def main():
    source = SOURCE_ROOT / "docs" / "PAPER_NOTES.md"
    doc = source.read_text()
    trigger = rows(section(doc, "### 4.1 Trigger ROUGE-L", "### 4.2 Argument extraction"))
    argument = rows(section(doc, "### 4.2 Argument extraction", "### 4.3 All four matching"))
    archive = SOURCE_ROOT.parent / "SciEvent" / "method" / "CARVE-full" / "docs" / "PAPER_NOTES.md"
    archived_doc = archive.read_text()
    design = rows(section(archived_doc, "### 5.10 Combined ablation", "(\\* the"),
                  numeric_only=False)
    vi_archive = archive.with_name("PAPER_VI.md")
    vi_doc = vi_archive.read_text()
    ablations = rows(section(vi_doc, "Bảng 12 — Ablation đơn-yếu-tố", "### 7.6"),
                     numeric_only=False)
    out = {"id": "published_baselines", "archive": True,
           "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "source": [str(source), str(archive), str(vi_archive)],
           "trigger_rougeL": trigger, "argument_iou": argument,
           "proposer_design_same_grid": design,
           "proposer_single_factor_dev": ablations,
           "notes": ["The plan lists 48.34 for the single-head-only dev mean. "
                     "The archive's same-grid 2x2 table lists 48.31 ± 1.62; "
                     "48.34 ± 1.60 appears in an earlier grid."]}
    dest = output_path("results", "published_baselines.json")
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(out, indent=2))
    print(f"{len(trigger)} trigger and {len(argument)} argument rows -> {dest}")


if __name__ == "__main__":
    main()
