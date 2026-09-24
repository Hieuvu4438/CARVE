"""Run the released test command with all outputs confined to experiment/.

The original freeze_and_test module is loaded unchanged. Its ROOT is redirected
to a small workspace with read-only links to the released artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "experiment" / "phase0_sandbox"


def link(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        assert dest.resolve() == source.resolve(), dest
    elif dest.exists():
        raise FileExistsError(dest)
    else:
        dest.symlink_to(source)


def main() -> None:
    link(REPO / "data" / "cands", WORK / "data" / "cands")
    link(REPO / "runs" / "verifier_s42", WORK / "runs" / "verifier_s42")
    link(REPO / "assets" / "decoding_rule.json", WORK / "assets" / "decoding_rule.json")

    spec = importlib.util.spec_from_file_location(
        "released_freeze_and_test", REPO / "scripts" / "freeze_and_test.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = WORK
    module.cmd_test(
        argparse.Namespace(runs=["verifier_s42"], rule_name="decoding_rule.json")
    )


if __name__ == "__main__":
    main()
