"""Resolve released inputs and experiment-owned outputs safely."""

from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = EXPERIMENT_ROOT.parent
RELEASED_RUNS = {"verifier_s42", "proposer_s42", "proposer_s13", "proposer_s101"}


def run_dir(name: str, *, write: bool = False) -> Path:
    if name in RELEASED_RUNS:
        if write:
            raise ValueError(f"refusing to write to released run {name}")
        return SOURCE_ROOT / "runs" / name
    if name == "verifier_v42_p42" and not write:
        return SOURCE_ROOT / "runs" / "verifier_s42"
    path = EXPERIMENT_ROOT / "runs" / name
    if write and (path.is_symlink() or
                  not path.resolve().is_relative_to(EXPERIMENT_ROOT.resolve())):
        raise ValueError(f"run output escapes experiment/: {path}")
    if write or path.exists():
        return path
    released = SOURCE_ROOT / "runs" / name
    if released.exists():
        return released
    raise FileNotFoundError(path)


def candidate_path(name: str, *, write: bool = False) -> Path:
    path = EXPERIMENT_ROOT / "data" / "cands" / name
    if write and (path.is_symlink() or
                  not path.resolve().is_relative_to(EXPERIMENT_ROOT.resolve())):
        raise ValueError(f"candidate output escapes experiment/: {path}")
    if write or path.exists():
        return path
    source = SOURCE_ROOT / "data" / "cands" / name
    if source.exists():
        return source
    raise FileNotFoundError(path)


def rule_path(name: str, *, write: bool = False) -> Path:
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError(f"rule name must be a JSON basename: {name}")
    path = EXPERIMENT_ROOT / "assets" / name
    if write and (path.is_symlink() or
                  not path.resolve().is_relative_to(EXPERIMENT_ROOT.resolve())):
        raise ValueError(f"rule output escapes experiment/: {path}")
    if write or path.exists():
        return path
    source = SOURCE_ROOT / "assets" / name
    if source.exists():
        return source
    raise FileNotFoundError(path)


def output_path(*parts: str) -> Path:
    path = EXPERIMENT_ROOT.joinpath(*parts)
    if not path.resolve().is_relative_to(EXPERIMENT_ROOT.resolve()):
        raise ValueError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
