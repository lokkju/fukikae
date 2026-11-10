"""Which turns and clusters identity reads: the names stage's refined pair when its
`names.json` says names were used, else the diarize stage's (Task 4's helper for the characters
stage, `registry resolve`, `reembed`, `adapt` and `eval identity`)."""

from pathlib import Path

from fukikae.artifacts import read_json
from fukikae.names.models import NamesResult

NAMES_DIR = "04c-names"
DIARIZE_DIR = "04-diarize"


def names_path(work: Path) -> Path:
    return Path(work) / NAMES_DIR / "names.json"


def identity_inputs(work: Path) -> tuple[Path, Path, NamesResult | None]:
    """(turns.json, clusters.npz, names result) for an episode's work directory: the refined
    pair under `04c-names/` when names were used, else `04-diarize/`'s, with the result (None
    when the names stage never ran)."""
    path = names_path(work)
    result = read_json(path, NamesResult) if path.exists() else None
    d = Path(work) / (NAMES_DIR if result is not None and result.used else DIARIZE_DIR)
    return d / "turns.json", d / "clusters.npz", result
