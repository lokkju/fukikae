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


# What a held-out episode's names.json may say: it read no names (`holdout`), or names were
# off for the whole run (a baseline arm). A missing file is a workspace from before the stage.
HELD_OUT_VERDICTS = frozenset({"holdout", "off"})


class HeldOutNamesError(ValueError):
    """A held-out episode whose names stage read its subtitle names."""


def check_held_out(work: Path, episode: str) -> None:
    """Refuse to grade `episode` as held out when its names stage read names (verdict other than
    `holdout` or `off`): the run would be graded on its own labels (Review Focus 3). Every eval
    path that scores a held-out episode calls this first."""
    path = names_path(work)
    if not path.exists():
        return
    result = read_json(path, NamesResult)
    if result.verdict not in HELD_OUT_VERDICTS:
        raise HeldOutNamesError(
            f"held-out episode {episode} read its subtitle names (names verdict "
            f"{result.verdict!r}); add it to [names] holdout_episodes and rerun it from names, "
            "or score it as a tuning episode"
        )
